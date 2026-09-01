import pytest
import torch

from glinext.layers import (
    AnchorCrossAttentionLayer,
    AttentionBias,
    PostNormAnchorRefinementBlock,
    PreNormAnchorRefinementBlock,
    RefinementPositionEncoding,
)


def test_refinement_position_object_generates_padding_independent_1d_positions():
    positions = RefinementPositionEncoding.from_config(
        {
            "memory": {"type": "sine1d"},
            "query": {"type": "sine1d"},
            "memory_usage": "keys_and_values",
        },
        hidden_size=16,
        num_query_embeddings=3,
    )
    short = torch.zeros(1, 3, 16)
    padded = torch.zeros(1, 6, 16)

    short_positions = positions.memory_positions(
        short,
        torch.ones(1, 3, dtype=torch.bool),
    )
    padded_positions = positions.memory_positions(
        padded,
        torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.bool),
    )

    torch.testing.assert_close(short_positions, padded_positions[:, :3])
    assert positions.memory_position_in_values


def test_refinement_position_object_accepts_explicit_2d_and_box_coordinates():
    positions = RefinementPositionEncoding.from_config(
        {
            "memory": "sine2d",
            "query": "sine_bbox2d",
        },
        hidden_size=16,
        num_query_embeddings=2,
    )
    anchors = torch.zeros(1, 2, 16)
    memory = torch.zeros(1, 4, 16)
    query_coordinates = torch.rand(1, 2, 4)
    memory_coordinates = torch.rand(4, 2)

    query_positions, memory_positions = positions(
        anchors,
        memory,
        query_coordinates=query_coordinates,
        memory_coordinates=memory_coordinates,
    )

    assert query_positions.shape == anchors.shape
    assert memory_positions.shape == memory.shape


def test_memory_usage_none_disables_memory_positions_only():
    positions = RefinementPositionEncoding.from_config(
        {
            "memory": "sine1d",
            "query": "sine1d",
            "memory_usage": "none",
        },
        hidden_size=16,
        num_query_embeddings=2,
    )

    assert positions.memory_positions(torch.zeros(1, 4, 16)) is None
    assert positions.query_positions(torch.zeros(1, 2, 16)) is not None


@pytest.mark.parametrize(
    ("norm_style", "block_type"),
    [
        ("pre_norm", PreNormAnchorRefinementBlock),
        ("post_norm", PostNormAnchorRefinementBlock),
    ],
)
def test_refinement_factory_selects_concrete_norm_variant(norm_style, block_type):
    refinement = AnchorCrossAttentionLayer.from_config(
        {
            "type": "cross_attention",
            "params": {
                "layers": 2,
                "heads": 4,
                "norm_style": norm_style,
                "norm_type": "rms_norm",
                "ffn_multiplier": 2,
                "activation": "silu",
            },
        },
        hidden_size=16,
        dropout=0.0,
    )

    assert refinement.num_layers == 2
    assert all(isinstance(layer, block_type) for layer in refinement.layers)


def test_explicit_refinement_component_defaults_to_one_layer():
    refinement = AnchorCrossAttentionLayer.from_config(
        {"type": "cross_attention"},
        hidden_size=16,
        dropout=0.0,
    )

    assert refinement.num_layers == 1


def test_refinement_composes_positions_and_biases_without_a_pipeline_wrapper():
    positions = RefinementPositionEncoding.from_config(
        {"memory": "sine1d", "query": "sine1d"},
        hidden_size=16,
        num_query_embeddings=3,
    )
    refinement = AnchorCrossAttentionLayer.from_config(
        {"type": "cross_attention", "layers": 1, "heads": 4},
        hidden_size=16,
        dropout=0.0,
    )
    cross_bias = AttentionBias.from_config(
        {"type": "gaussian_distance", "sigma": 0.5},
        num_heads=4,
    )
    anchors = torch.randn(2, 3, 16, requires_grad=True)
    memory = torch.randn(2, 5, 16, requires_grad=True)

    output = refinement(
        anchors,
        memory,
        position_encoding=positions,
        cross_attention_bias_module=cross_bias,
    )
    output.sum().backward()

    assert output.shape == anchors.shape
    assert anchors.grad is not None
    assert memory.grad is not None


def test_refinement_backpropagates_into_per_head_gaussian_bias_parameters():
    positions = RefinementPositionEncoding.from_config(
        {"memory": "sine1d", "query": "sine1d"},
        hidden_size=16,
        num_query_embeddings=3,
    )
    refinement = AnchorCrossAttentionLayer.from_config(
        {"type": "cross_attention", "layers": 1, "heads": 4},
        hidden_size=16,
        dropout=0.0,
    )
    cross_bias = AttentionBias.from_config(
        {
            "type": "gaussian_distance",
            "params": {
                "sigma": 0.5,
                "weight": 1.0,
                "learnable_sigma": True,
                "learnable_weight": True,
                "per_head": True,
            },
        },
        num_heads=4,
    )
    anchors = torch.randn(2, 3, 16, requires_grad=True)
    memory = torch.randn(2, 5, 16, requires_grad=True)

    output = refinement(
        anchors,
        memory,
        position_encoding=positions,
        cross_attention_bias_module=cross_bias,
    )
    output.square().mean().backward()

    assert output.shape == anchors.shape
    assert cross_bias.raw_sigma.grad is not None
    assert cross_bias.raw_weight.grad is not None
