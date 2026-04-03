"""Tests for configurable anchor acquisition layers."""

import pytest
import torch

from glinext.layers.anchor_layer import (
    AnchorLayer, ParentAnchorLayer, FixedAnchorLayer,
    RotaryAnchorLayer, QueryLSTMAnchorLayer, QueryTransformerAnchorLayer,
)

D = 16


class TestAnchorLayerFactory:
    def test_parent(self):
        layer = AnchorLayer.from_config("parent", D)
        assert isinstance(layer, ParentAnchorLayer)

    def test_fixed(self):
        layer = AnchorLayer.from_config("fixed", D, num_slots=5)
        assert isinstance(layer, FixedAnchorLayer)
        assert layer.num_slots == 5

    def test_rotary(self):
        layer = AnchorLayer.from_config("rotary", D, max_count=10)
        assert isinstance(layer, RotaryAnchorLayer)

    def test_lstm_alias(self):
        layer = AnchorLayer.from_config("lstm", D)
        assert isinstance(layer, RotaryAnchorLayer)

    def test_query_lstm(self):
        layer = AnchorLayer.from_config("query_lstm", D)
        assert isinstance(layer, QueryLSTMAnchorLayer)

    def test_query_transformer(self):
        layer = AnchorLayer.from_config("query_transformer", D, num_heads=2, num_layers=1)
        assert isinstance(layer, QueryTransformerAnchorLayer)

    def test_unknown_mode(self):
        with pytest.raises(ValueError, match="Unknown anchor_mode"):
            AnchorLayer.from_config("nonexistent", D)


class TestParentAnchorLayer:
    def test_output_shape(self):
        layer = ParentAnchorLayer(D)
        ctx = torch.randn(2, 4, D)
        anchors, mask = layer(ctx)
        assert anchors.shape == (2, 1, D)
        assert mask.shape == (2, 1)
        assert mask.all()

    def test_mean_pooled(self):
        layer = ParentAnchorLayer(D)
        ctx = torch.randn(1, 3, D)
        anchors, _ = layer(ctx)
        expected = ctx.mean(dim=1, keepdim=True)
        assert torch.allclose(anchors, expected)


class TestFixedAnchorLayer:
    def test_output_shape(self):
        layer = FixedAnchorLayer(D, num_slots=5)
        ctx = torch.randn(2, 3, D)
        anchors, mask = layer(ctx)
        assert anchors.shape == (2, 5, D)
        assert mask.shape == (2, 5)
        assert mask.all()

    def test_with_count(self):
        layer = FixedAnchorLayer(D, num_slots=5)
        ctx = torch.randn(2, 3, D)
        count = torch.tensor([3, 2])
        anchors, mask = layer(ctx, count=count)
        assert anchors.shape == (2, 5, D)
        assert mask[0, :3].all() and not mask[0, 3:].any()
        assert mask[1, :2].all() and not mask[1, 2:].any()

    def test_learnable(self):
        layer = FixedAnchorLayer(D, num_slots=3)
        assert layer.anchor_table.weight.requires_grad

    def test_shared_across_batch(self):
        layer = FixedAnchorLayer(D, num_slots=3)
        ctx = torch.randn(4, 2, D)
        anchors, _ = layer(ctx)
        # All batch items should share the same anchor embeddings
        assert torch.equal(anchors[0], anchors[1])
        assert torch.equal(anchors[0], anchors[3])

    def test_scalar_count(self):
        layer = FixedAnchorLayer(D, num_slots=5)
        ctx = torch.randn(2, 3, D)
        count = torch.tensor(3)
        anchors, mask = layer(ctx, count=count)
        assert mask[0, :3].all() and not mask[0, 3:].any()
        assert mask[1, :3].all() and not mask[1, 3:].any()


class TestRotaryAnchorLayer:
    def test_output_types(self):
        layer = RotaryAnchorLayer(D, max_count=5)
        ctx = torch.randn(2, 3, D)
        count = torch.tensor([2, 3])
        anchors, mask = layer(ctx, count=count)
        assert anchors.dim() == 3
        assert mask.dim() == 2
        assert mask.dtype == torch.bool

    def test_variable_counts(self):
        layer = RotaryAnchorLayer(D, max_count=10)
        ctx = torch.randn(2, 4, D)
        count = torch.tensor([1, 3])
        anchors, mask = layer(ctx, count=count)
        # Max anchors should be max(1, 3) = 3
        assert anchors.shape[1] == 3
        assert mask[0, 0] and not mask[0, 1]
        assert mask[1, :3].all()


class TestQueryLSTMAnchorLayer:
    def test_no_word_embeddings(self):
        layer = QueryLSTMAnchorLayer(D)
        ctx = torch.randn(2, 3, D)
        anchors, mask = layer(ctx, word_embeddings=None)
        assert anchors.shape == (2, 0, D)
        assert mask.shape == (2, 0)

    def test_with_word_embeddings(self):
        layer = QueryLSTMAnchorLayer(D, max_count=5)
        ctx = torch.randn(2, 3, D)
        words = torch.randn(2, 10, D)
        count = torch.tensor([2, 3])
        anchors, mask = layer(ctx, word_embeddings=words, count=count)
        assert anchors.dim() == 3
        assert mask.dim() == 2


class TestQueryTransformerAnchorLayer:
    def test_no_word_embeddings(self):
        layer = QueryTransformerAnchorLayer(D, num_heads=2, num_layers=1)
        ctx = torch.randn(2, 3, D)
        anchors, mask = layer(ctx, word_embeddings=None)
        assert anchors.shape == (2, 0, D)
        assert mask.shape == (2, 0)

    def test_with_word_embeddings(self):
        layer = QueryTransformerAnchorLayer(D, num_heads=2, num_layers=1)
        ctx = torch.randn(2, 3, D)
        words = torch.randn(2, 10, D)
        count = torch.tensor([2, 3])
        anchors, mask = layer(ctx, word_embeddings=words, count=count)
        assert anchors.dim() == 3
        assert mask.dim() == 2
