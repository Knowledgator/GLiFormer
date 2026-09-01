"""Focused regression tests for masked anchor memory and query selection."""

import pytest
import torch

from glinext.layers.anchor_layer import (
    FixedTransformerAnchorLayer,
    QueryRNNAnchorLayer,
    QueryTransformerAnchorLayer,
)
from glinext.layers.groups import AnchorCrossAttentionLayer


HIDDEN_SIZE = 16


def _perturb_masked(features: torch.Tensor, feature_mask: torch.Tensor) -> torch.Tensor:
    perturbed = features.clone()
    perturbed[~feature_mask] = torch.randn_like(perturbed[~feature_mask]) * 10_000
    return perturbed


def test_fixed_transformer_ignores_masked_memory_features():
    torch.manual_seed(1)
    layer = FixedTransformerAnchorLayer(
        HIDDEN_SIZE,
        num_slots=3,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    ).eval()
    context = torch.randn(2, HIDDEN_SIZE)
    features = torch.randn(2, 5, HIDDEN_SIZE)
    feature_mask = torch.tensor(
        [[True, True, False, False, False], [True, False, True, False, False]],
    )

    first, first_mask = layer(context, features, feature_mask=feature_mask)
    second, second_mask = layer(
        context,
        _perturb_masked(features, feature_mask),
        feature_mask=feature_mask,
    )

    assert torch.equal(first_mask, second_mask)
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "layer",
    [
        QueryRNNAnchorLayer(HIDDEN_SIZE, max_count=4),
        QueryTransformerAnchorLayer(
            HIDDEN_SIZE,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            max_count=4,
        ),
    ],
    ids=["rnn", "transformer"],
)
def test_query_selection_excludes_masked_features(layer):
    torch.manual_seed(2)
    layer.eval()
    context = torch.randn(1, HIDDEN_SIZE)
    features = torch.randn(1, 5, HIDDEN_SIZE)
    feature_mask = torch.tensor([[True, False, True, False, False]])
    count = torch.tensor([2])

    first, first_mask = layer(
        context,
        features,
        count=count,
        feature_mask=feature_mask,
    )
    second, second_mask = layer(
        context,
        _perturb_masked(features, feature_mask),
        count=count,
        feature_mask=feature_mask,
    )

    assert first_mask.all()
    assert torch.equal(first_mask, second_mask)
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "layer",
    [
        QueryRNNAnchorLayer(HIDDEN_SIZE, max_count=4),
        QueryTransformerAnchorLayer(
            HIDDEN_SIZE,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            max_count=4,
        ),
    ],
    ids=["rnn", "transformer"],
)
def test_threshold_query_selection_excludes_masked_features(layer):
    torch.manual_seed(6)
    layer.eval()
    context = torch.randn(1, HIDDEN_SIZE)
    features = torch.randn(1, 5, HIDDEN_SIZE)
    feature_mask = torch.tensor([[True, False, True, False, False]])

    first, first_mask = layer(
        context,
        features,
        threshold=-1_000_000.0,
        feature_mask=feature_mask,
    )
    second, second_mask = layer(
        context,
        _perturb_masked(features, feature_mask),
        threshold=-1_000_000.0,
        feature_mask=feature_mask,
    )

    assert first.shape[1] == int(feature_mask.sum().item())
    assert first_mask.all()
    assert torch.equal(first_mask, second_mask)
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "layer",
    [
        QueryRNNAnchorLayer(HIDDEN_SIZE, max_count=4),
        QueryTransformerAnchorLayer(
            HIDDEN_SIZE,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            max_count=4,
        ),
    ],
    ids=["rnn", "transformer"],
)
def test_query_all_valid_mask_preserves_unmasked_behavior(layer):
    torch.manual_seed(5)
    layer.eval()
    context = torch.randn(2, HIDDEN_SIZE)
    features = torch.randn(2, 5, HIDDEN_SIZE)
    count = torch.tensor([2, 3])

    unmasked, unmasked_anchor_mask = layer(context, features, count=count)
    explicitly_valid, valid_anchor_mask = layer(
        context,
        features,
        count=count,
        feature_mask=torch.ones(2, 5, dtype=torch.bool),
    )

    assert torch.equal(unmasked_anchor_mask, valid_anchor_mask)
    assert torch.allclose(unmasked, explicitly_valid, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "layer",
    [
        QueryRNNAnchorLayer(HIDDEN_SIZE, max_count=4),
        QueryTransformerAnchorLayer(
            HIDDEN_SIZE,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            max_count=4,
        ),
    ],
    ids=["rnn", "transformer"],
)
def test_all_masked_query_features_use_only_invalid_finite_sentinel(layer):
    layer.eval()
    context = torch.randn(2, HIDDEN_SIZE)
    features = torch.randn(2, 4, HIDDEN_SIZE)
    feature_mask = torch.zeros(2, 4, dtype=torch.bool)

    anchors, anchor_mask = layer(
        context,
        features,
        count=torch.tensor([2, 1]),
        feature_mask=feature_mask,
    )

    assert torch.isfinite(anchors).all()
    assert not anchor_mask.any()


def test_anchor_refinement_is_finite_and_invariant_for_all_masked_memory():
    torch.manual_seed(3)
    layer = AnchorCrossAttentionLayer(
        HIDDEN_SIZE,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    ).eval()
    anchors = torch.randn(2, 3, HIDDEN_SIZE)
    features = torch.randn(2, 5, HIDDEN_SIZE)
    feature_mask = torch.zeros(2, 5, dtype=torch.bool)
    memory_positions = torch.randn(1, 5, HIDDEN_SIZE)

    first = layer(
        anchors,
        features,
        token_mask=feature_mask,
        memory_pos_emb=memory_positions,
    )
    second = layer(
        anchors,
        _perturb_masked(features, feature_mask),
        token_mask=feature_mask,
        memory_pos_emb=memory_positions,
    )

    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)


def test_anchor_refinement_ignores_partially_masked_memory():
    torch.manual_seed(4)
    layer = AnchorCrossAttentionLayer(
        HIDDEN_SIZE,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    ).eval()
    anchors = torch.randn(1, 3, HIDDEN_SIZE)
    features = torch.randn(1, 5, HIDDEN_SIZE)
    feature_mask = torch.tensor([[True, True, False, False, False]])

    first = layer(anchors, features, token_mask=feature_mask)
    second = layer(
        anchors,
        _perturb_masked(features, feature_mask),
        token_mask=feature_mask,
    )

    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)


def test_anchor_refinement_invalid_queries_cannot_influence_valid_queries():
    torch.manual_seed(7)
    layer = AnchorCrossAttentionLayer(
        HIDDEN_SIZE,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    ).eval()
    anchors = torch.randn(1, 4, HIDDEN_SIZE)
    features = torch.randn(1, 5, HIDDEN_SIZE)
    query_mask = torch.tensor([[True, True, False, False]])
    perturbed = anchors.clone()
    perturbed[:, 2:] = torch.randn_like(perturbed[:, 2:]) * 10_000

    first = layer(anchors, features, query_mask=query_mask)
    second = layer(perturbed, features, query_mask=query_mask)

    assert torch.allclose(first[:, :2], second[:, :2], atol=1e-6, rtol=1e-6)
    assert not first[:, 2:].any()
    assert not second[:, 2:].any()


def test_anchor_refinement_all_invalid_queries_are_finite_zeros():
    layer = AnchorCrossAttentionLayer(
        HIDDEN_SIZE,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    ).eval()
    result = layer(
        torch.randn(2, 3, HIDDEN_SIZE),
        torch.randn(2, 4, HIDDEN_SIZE),
        query_mask=torch.zeros(2, 3, dtype=torch.bool),
    )

    assert torch.isfinite(result).all()
    assert not result.any()
