"""Tests for registry-backed spatial position embeddings."""

import pytest
import torch

from glinext.config import (
    AudioSegmentationHeadConfig,
    ObjectDetectionHeadConfig,
    StructuringHeadConfig,
)
from glinext.layers.position import (
    AnchorRefinementPositionEmbeddings,
    PositionEmbedding,
    normalized_grid_1d,
    normalized_grid_2d,
)


def test_normalized_grid_supports_rectangular_feature_maps():
    coordinates = normalized_grid_2d(2, 3, device=torch.device("cpu"))

    assert coordinates.shape == (6, 2)
    assert torch.allclose(coordinates[0], torch.tensor([1 / 6, 1 / 4]))
    assert torch.allclose(coordinates[-1], torch.tensor([5 / 6, 3 / 4]))


def test_normalized_temporal_grid_uses_cell_centres():
    coordinates = normalized_grid_1d(4, device=torch.device("cpu"))

    assert coordinates.shape == (4, 1)
    assert torch.allclose(
        coordinates[:, 0],
        torch.tensor([0.125, 0.375, 0.625, 0.875]),
    )


@pytest.mark.parametrize("position_type", ["sine2d", "linear2d", "mlp2d"])
def test_coordinate_position_strategies_preserve_batch_shape_and_dtype(position_type):
    coordinates = torch.rand(2, 6, 2)
    layer = PositionEmbedding.from_config(position_type, hidden_size=8)

    positions = layer(coordinates, dtype=torch.bfloat16, device=coordinates.device)

    assert positions.shape == (2, 6, 8)
    assert positions.dtype == torch.bfloat16


def test_bbox_sine_positions_preserve_shape_dtype_and_registry_metadata():
    coordinates = torch.rand(2, 6, 4)
    layer = PositionEmbedding.from_config("sine_bbox2d", hidden_size=16)

    positions = layer(coordinates, dtype=torch.bfloat16)

    assert layer.coordinate_dimensions == 4
    assert positions.shape == (2, 6, 16)
    assert positions.dtype == torch.bfloat16


def test_bbox_sine_positions_encode_width_and_height_independently():
    layer = PositionEmbedding.from_config("sine_bbox2d", hidden_size=16)
    baseline = torch.tensor([[[0.25, 0.50, 0.0, 0.0]]])
    wider = baseline.clone()
    wider[..., 2] = 0.5
    taller = baseline.clone()
    taller[..., 3] = 0.75

    baseline_positions = layer(baseline)
    wider_positions = layer(wider)
    taller_positions = layer(taller)

    assert torch.equal(baseline_positions[..., :8], wider_positions[..., :8])
    assert not torch.equal(baseline_positions[..., 8:12], wider_positions[..., 8:12])
    assert torch.equal(baseline_positions[..., 12:], wider_positions[..., 12:])
    assert torch.equal(baseline_positions[..., :12], taller_positions[..., :12])
    assert not torch.equal(baseline_positions[..., 12:], taller_positions[..., 12:])


def test_bbox_sine_positions_validate_shape_and_hidden_size():
    with pytest.raises(ValueError, match="divisible by 8"):
        PositionEmbedding.from_config("sine_bbox2d", hidden_size=12)

    layer = PositionEmbedding.from_config("sine_bbox2d", hidden_size=16)
    with pytest.raises(ValueError, match=r"\.\.\., 4"):
        layer(torch.rand(2, 6, 2))


@pytest.mark.parametrize(
    "position_kwargs",
    [
        {"temperature": 0.0},
        {"temperature": float("nan")},
        {"scale": float("inf")},
    ],
)
def test_bbox_sine_positions_validate_numeric_configuration(position_kwargs):
    with pytest.raises(ValueError, match="temperature|scale"):
        PositionEmbedding.from_config(
            "sine_bbox2d",
            hidden_size=16,
            **position_kwargs,
        )


@pytest.mark.parametrize("position_type", ["sine1d", "linear1d", "mlp1d"])
def test_temporal_position_strategies_preserve_shape_and_dtype(position_type):
    coordinates = torch.rand(2, 6, 1)
    layer = PositionEmbedding.from_config(position_type, hidden_size=8)

    positions = layer(coordinates, dtype=torch.bfloat16)

    assert positions.shape == (2, 6, 8)
    assert positions.dtype == torch.bfloat16


@pytest.mark.parametrize(
    "position_type",
    ["fixed_sinusoidal", "fourier"],
)
def test_fixed_text_positions_are_parameter_and_checkpoint_free(position_type):
    layer = PositionEmbedding.from_config(position_type, hidden_size=8)
    coordinates = torch.tensor([[0.1], [0.4], [0.9]])

    positions = layer(coordinates, dtype=torch.bfloat16)

    assert positions.shape == (3, 8)
    assert positions.dtype == torch.bfloat16
    assert list(layer.parameters()) == []
    assert layer.state_dict() == {}
    assert not torch.equal(positions[0], positions[1])


def test_fourier_positions_validate_frequency_configuration():
    with pytest.raises(ValueError, match="must be positive"):
        PositionEmbedding.from_config(
            "fourier",
            hidden_size=8,
            min_frequency=0.0,
        )
    with pytest.raises(ValueError, match="at least min_frequency"):
        PositionEmbedding.from_config(
            "fourier",
            hidden_size=8,
            min_frequency=4.0,
            max_frequency=2.0,
        )
    with pytest.raises(ValueError, match="frequency_spacing"):
        PositionEmbedding.from_config(
            "fourier",
            hidden_size=8,
            frequency_spacing="random",
        )


def test_anchor_refinement_positions_ignore_padding_in_memory_coordinates():
    positions = AnchorRefinementPositionEmbeddings(
        hidden_size=8,
        memory_type="fourier",
        query_type="fixed_sinusoidal",
        num_query_embeddings=3,
    )
    memory = torch.randn(2, 5, 8)
    memory_mask = torch.tensor(
        [
            [True, False, True, False, True],
            [True, True, True, True, True],
        ]
    )
    anchors = torch.randn(2, 3, 8)

    query_positions, memory_positions = positions(
        anchors,
        memory,
        memory_mask=memory_mask,
    )

    expected_valid = positions.memory_embedding(
        normalized_grid_1d(3, device=memory.device),
    )
    torch.testing.assert_close(
        memory_positions[0, memory_mask[0]],
        expected_valid,
    )
    assert not memory_positions[0, ~memory_mask[0]].any()
    assert query_positions.shape == (1, 3, 8)


def test_position_bucket_attention_bias_is_local_and_padding_invariant():
    positions = AnchorRefinementPositionEmbeddings(hidden_size=8)
    anchors = torch.zeros(1, 2, 8)
    padded_memory = torch.zeros(1, 6, 8)
    memory_mask = torch.tensor([[True, True, False, True, True, False]])

    padded_bias = positions.position_bucket_attention_bias(
        anchors,
        padded_memory,
        memory_mask=memory_mask,
        sigma=0.5,
    )
    compact_bias = positions.position_bucket_attention_bias(
        anchors,
        torch.zeros(1, 4, 8),
        sigma=0.5,
    )

    torch.testing.assert_close(padded_bias[..., memory_mask[0]], compact_bias)
    assert padded_bias[0, 0, 0] > padded_bias[0, 0, 3]
    assert padded_bias[0, 1, 3] > padded_bias[0, 1, 0]


def test_structuring_position_config_is_independent_for_memory_and_queries():
    config = StructuringHeadConfig(
        anchor_mode="position_buckets",
        num_fixed_slots=5,
        anchor_refine_layers=2,
        memory_position_embedding_type="fourier",
        query_position_embedding_type="fixed_sinusoidal",
        memory_position_embedding_kwargs={"max_frequency": 32.0},
        query_position_embedding_kwargs={"max_position": 512.0},
    )

    assert config.memory_position_embedding_kwargs == {"max_frequency": 32.0}
    assert config.query_position_embedding_kwargs == {"max_position": 512.0}
    with pytest.raises(ValueError, match="one-dimensional"):
        StructuringHeadConfig(
            memory_position_embedding_type="sine2d",
        )


def test_structuring_fixed_sinusoidal_extent_equals_fixed_anchor_count():
    config = StructuringHeadConfig(
        anchor_mode="position_buckets",
        num_fixed_slots=7,
        anchor_refine_layers=2,
        memory_position_embedding_type="fixed_sinusoidal",
        query_position_embedding_type="fixed_sinusoidal",
    )

    assert config.memory_position_embedding_kwargs["max_position"] == 7.0
    assert config.query_position_embedding_kwargs["max_position"] == 7.0


def test_structuring_position_bucket_stabilization_config_is_validated():
    config = StructuringHeadConfig(
        anchor_mode="position_buckets",
        anchor_refine_layers=1,
        position_bucket_normalization="center-rms",
        position_bucket_attention_bias_type="gaussian",
        position_bucket_attention_sigma=0.5,
    )

    assert config.position_bucket_normalization == "center_rms"
    with pytest.raises(ValueError, match="require anchor_mode"):
        StructuringHeadConfig(
            anchor_mode="fixed",
            position_bucket_normalization="center_rms",
        )
    with pytest.raises(ValueError, match="anchor_refine_layers"):
        StructuringHeadConfig(
            anchor_mode="position_buckets",
            position_bucket_attention_bias_type="gaussian",
        )
    with pytest.raises(ValueError, match="sigma"):
        StructuringHeadConfig(
            anchor_mode="position_buckets",
            position_bucket_attention_sigma=0.0,
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        StructuringHeadConfig(negatives=1.1)


def test_learned_query_positions_use_configured_slot_capacity():
    layer = PositionEmbedding.from_config(
        "learned",
        hidden_size=8,
        num_embeddings=4,
    )

    assert layer(count=3).shape == (3, 8)
    with pytest.raises(ValueError, match="only 4"):
        layer(count=5)


def test_learned_grid_positions_interpolate_to_rectangular_resolution():
    layer = PositionEmbedding.from_config(
        "learned_grid2d",
        hidden_size=8,
        grid_size=(2, 2),
    )

    positions = layer(spatial_shape=(3, 5), count=15, dtype=torch.float16)

    assert positions.shape == (1, 15, 8)
    assert positions.dtype == torch.float16


def test_position_registry_is_extensible_from_detection_config():
    class ConstantPositionEmbedding(PositionEmbedding, position_type="test_constant"):
        def __init__(self, hidden_size, **kwargs):
            super().__init__()
            self.hidden_size = hidden_size

        def forward(self, coordinates=None, *, count=None, dtype=None, device=None, **kwargs):
            shape = (*coordinates.shape[:-1], self.hidden_size)
            return torch.ones(shape, dtype=dtype, device=device)

    config = ObjectDetectionHeadConfig(
        memory_position_embedding_type="test_constant",
    )
    layer = PositionEmbedding.from_config(
        config.memory_position_embedding_type,
        hidden_size=8,
    )

    assert isinstance(layer, ConstantPositionEmbedding)


def test_detection_config_rejects_coordinate_queries_without_references():
    with pytest.raises(ValueError, match="require reference_box_mode"):
        ObjectDetectionHeadConfig(
            query_position_embedding_type="sine2d",
            reference_box_mode="none",
        )


def test_detection_config_requires_sigmoid_probabilities_for_multilabel():
    with pytest.raises(ValueError, match="requires class_probability='sigmoid'"):
        ObjectDetectionHeadConfig(
            multi_label=True,
            class_probability="softmax",
        )


def test_detection_config_rejects_fixed_capacity_positions_for_dynamic_anchors():
    with pytest.raises(ValueError, match="require a fixed anchor mode"):
        ObjectDetectionHeadConfig(
            anchor_mode="features",
            query_position_embedding_type="learned",
            reference_box_mode="none",
        )


def test_position_config_rejects_misspelled_strategy_options():
    with pytest.raises(ValueError, match="temprature"):
        ObjectDetectionHeadConfig(
            memory_position_embedding_kwargs={"temprature": 10_000.0},
        )


@pytest.mark.parametrize(
    "position_kwargs",
    [
        {"temperature": 0.0},
        {"temperature": float("nan")},
        {"scale": float("inf")},
    ],
)
def test_position_config_rejects_invalid_sine_numbers(position_kwargs):
    with pytest.raises(ValueError, match="temperature|scale"):
        ObjectDetectionHeadConfig(
            memory_position_embedding_kwargs=position_kwargs,
        )


def test_position_config_rejects_invalid_learned_initialization():
    with pytest.raises(ValueError, match="init_std"):
        ObjectDetectionHeadConfig(
            query_position_embedding_type="learned",
            query_position_embedding_kwargs={"init_std": -1.0},
        )


def test_learned_grid_query_gets_a_configured_base_grid():
    config = ObjectDetectionHeadConfig(
        num_fixed_slots=5,
        query_position_embedding_type="learned_grid2d",
    )

    assert config.query_position_embedding_kwargs["grid_size"] == (3, 2)


def test_detection_config_accepts_full_box_query_positions():
    config = ObjectDetectionHeadConfig(
        query_position_embedding_type="sine_bbox2d",
    )

    assert (
        PositionEmbedding.strategy_class(config.query_position_embedding_type)
        .coordinate_dimensions
        == 4
    )


def test_detection_config_rejects_inert_iterative_options():
    with pytest.raises(ValueError, match="anchor_refine_layers > 0"):
        ObjectDetectionHeadConfig(
            anchor_refine_layers=0,
            iterative_box_refinement=True,
        )
    with pytest.raises(ValueError, match="at least two decoder layers"):
        ObjectDetectionHeadConfig(
            iterative_box_refinement=True,
            anchor_refine_layers=1,
            auxiliary_detection_loss_coef=1.0,
        )
    with pytest.raises(ValueError, match="reference_box_initialization"):
        ObjectDetectionHeadConfig(reference_box_initialization="diagonal")


def test_audio_segmentation_positions_are_temporal_and_configurable():
    config = AudioSegmentationHeadConfig(
        num_fixed_slots=5,
        query_position_embedding_type="learned",
    )

    assert config.anchor_refine_layers == 2
    assert config.query_position_embedding_kwargs["num_embeddings"] == 5
    with pytest.raises(ValueError, match="one-dimensional"):
        AudioSegmentationHeadConfig(
            memory_position_embedding_type="sine2d",
        )


def test_audio_segmentation_rejects_fixed_capacity_positions_for_dynamic_anchors():
    with pytest.raises(ValueError, match="require a fixed anchor mode"):
        AudioSegmentationHeadConfig(
            anchor_mode="features",
            query_position_embedding_type="learned",
        )
