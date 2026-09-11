from dataclasses import asdict

import torch

from gliformer.config import (
    AudioSegmentationHeadConfig,
    ClassificationHeadConfig,
    ObjectDetectionHeadConfig,
)
from gliformer.layers import (
    GaussianDistanceAttentionBias,
    PreNormAnchorRefinementBlock,
    RMSNormAnchorNormalization,
)
from gliformer.tasks import TaskFlatInputs
from gliformer.tasks.audio.model import AudioSegmentationHead
from gliformer.tasks.classification.model import ClassificationHead
from gliformer.tasks.vision.model import ObjectDetectionHead
from tests.heads.conftest import B, C, D, W, make_config


def _flat_inputs(*, vision=False):
    inputs = TaskFlatInputs(
        words_embedding=torch.randn(B, W, D),
        mask=torch.ones(B, W, dtype=torch.long),
        parent_embedding=torch.randn(B, D),
        child_embedding=torch.randn(B, C, D),
        child_mask=torch.ones(B, C, dtype=torch.long),
        batch_origin=torch.arange(B),
    )
    if vision:
        inputs.feature_spatial_shape = torch.tensor([[3, 3]]).expand(B, -1)
        inputs.feature_prefix_tokens = torch.ones(B, dtype=torch.long)
    return inputs


def _component_options(*, dimensions=1):
    query_position = "sine1d" if dimensions == 1 else "sine_bbox2d"
    memory_position = "sine1d" if dimensions == 1 else "sine2d"
    bias_params = {"sigma": 0.3}
    if dimensions == 2:
        bias_params.update(
            {"query_dimensions": [0, 1], "key_dimensions": [0, 1]}
        )
    return dict(
        anchor_normalization={"type": "rms_norm", "eps": 1e-5},
        anchor_refinement={
            "type": "cross_attention",
            "params": {
                "layers": 2,
                "heads": 4,
                "norm_style": "pre_norm",
                "norm_type": "rms_norm",
                "ffn_multiplier": 2,
            },
        },
        anchor_memory_position={"type": memory_position},
        anchor_query_position={"type": query_position},
        anchor_cross_attention_bias={
            "type": "gaussian_distance",
            "params": bias_params,
        },
    )


def test_text_head_directly_composes_modular_anchor_components(shared):
    options = _component_options()
    options.update(
        anchor_layer={"type": "fixed", "params": {"num_slots": 3}},
        anchor_modeling={"type": "mlp", "params": {"dropout": 0.0}},
    )
    config = make_config(
        classification_config=asdict(ClassificationHeadConfig(**options))
    )
    head = ClassificationHead.from_config(config).eval()

    output = head(shared, {}, flat_inputs=_flat_inputs())

    assert output.logits.shape == (B, C)
    assert isinstance(head.anchor_normalizer, RMSNormAnchorNormalization)
    assert isinstance(head.anchor_refine.layers[0], PreNormAnchorRefinementBlock)
    assert isinstance(
        head.anchor_cross_attention_bias,
        GaussianDistanceAttentionBias,
    )
    assert not hasattr(head, "anchor_pipeline")


def test_explicit_refinement_does_not_inherit_legacy_disabled_layer_count():
    config = make_config(
        classification_config=asdict(
            ClassificationHeadConfig(
                anchor_refine_layers=0,
                anchor_refinement={"type": "cross_attention"},
            )
        )
    )

    head = ClassificationHead.from_config(config)

    assert head.anchor_refine.num_layers == 1


def test_vision_head_uses_raw_2d_geometry_with_same_components(shared):
    options = _component_options(dimensions=2)
    options.update(
        anchor_layer={"type": "fixed", "params": {"num_slots": 4}},
        iterative_box_refinement=True,
    )
    config = make_config(
        object_detection_config=asdict(ObjectDetectionHeadConfig(**options))
    )
    head = ObjectDetectionHead.from_config(config).eval()

    output = head(shared, {}, flat_inputs=_flat_inputs(vision=True))

    assert output.logits.shape == (B, 4, C)
    assert output.extra["bbox_preds"].shape == (B, 4, 4)
    assert head.anchor_refine_positions.query_strategy.coordinate_dimensions == 4


def test_audio_head_uses_automatic_temporal_coordinates(shared):
    options = _component_options()
    options.update(
        anchor_layer={"type": "fixed", "params": {"num_slots": 3}},
        anchor_memory_position_usage="keys_and_values",
    )
    config = make_config(
        audio_segmentation_config=asdict(
            AudioSegmentationHeadConfig(**options)
        )
    )
    head = AudioSegmentationHead.from_config(config).eval()

    output = head(shared, {}, flat_inputs=_flat_inputs())

    assert output.logits.shape == (B, 3, C)
    assert output.extra["segment_preds"].shape == (B, 3, 2)
    assert head.anchor_refine_positions.memory_position_in_values


def test_audio_head_loads_pre_component_position_keys_strictly():
    config = make_config(
        audio_segmentation_config=asdict(
            AudioSegmentationHeadConfig(
                num_fixed_slots=3,
                memory_position_embedding_type="linear1d",
                query_position_embedding_type="learned",
            )
        )
    )
    source = AudioSegmentationHead.from_config(config)
    legacy_state = source.state_dict()
    for key in tuple(legacy_state):
        if key.startswith("anchor_refine_positions.memory_embedding."):
            suffix = key.removeprefix(
                "anchor_refine_positions.memory_embedding."
            )
            legacy_state[f"memory_position_embedding.{suffix}"] = (
                legacy_state.pop(key)
            )
        elif key.startswith("anchor_refine_positions.query_embedding."):
            suffix = key.removeprefix(
                "anchor_refine_positions.query_embedding."
            )
            legacy_state[f"query_position_embedding.{suffix}"] = (
                legacy_state.pop(key)
            )

    destination = AudioSegmentationHead.from_config(config)
    result = destination.load_state_dict(legacy_state, strict=True)

    assert not result.missing_keys
    assert not result.unexpected_keys
