"""Tests for configurable anchor acquisition layers."""

import pytest
import torch

from glinext.layers.anchor_layer import (
    AnchorLayer, ParentAnchorLayer, FixedAnchorLayer,
    FixedLSTMAnchorLayer, FixedTransformerAnchorLayer,
    RotaryAnchorLayer, QueryLSTMAnchorLayer, QueryTransformerAnchorLayer,
)

D = 16


class TestAnchorLayerRegistry:
    def test_registry_has_all_modes(self):
        for mode in ["parent", "fixed", "fixed_lstm", "fixed_transformer",
                      "rotary", "lstm", "query_lstm", "query_transformer"]:
            assert mode in AnchorLayer._registry

    def test_registry_types(self):
        assert AnchorLayer._registry["parent"] is ParentAnchorLayer
        assert AnchorLayer._registry["fixed"] is FixedAnchorLayer
        assert AnchorLayer._registry["fixed_lstm"] is FixedLSTMAnchorLayer
        assert AnchorLayer._registry["fixed_transformer"] is FixedTransformerAnchorLayer
        assert AnchorLayer._registry["rotary"] is RotaryAnchorLayer
        assert AnchorLayer._registry["lstm"] is RotaryAnchorLayer
        assert AnchorLayer._registry["query_lstm"] is QueryLSTMAnchorLayer
        assert AnchorLayer._registry["query_transformer"] is QueryTransformerAnchorLayer


class TestAnchorLayerFactory:
    def test_parent(self):
        layer = AnchorLayer.from_config("parent", D)
        assert isinstance(layer, ParentAnchorLayer)

    def test_fixed(self):
        layer = AnchorLayer.from_config("fixed", D, num_slots=5)
        assert isinstance(layer, FixedAnchorLayer)
        assert layer.num_slots == 5

    def test_fixed_lstm(self):
        layer = AnchorLayer.from_config("fixed_lstm", D, num_slots=5)
        assert isinstance(layer, FixedLSTMAnchorLayer)
        assert layer.num_slots == 5

    def test_fixed_transformer(self):
        layer = AnchorLayer.from_config("fixed_transformer", D, num_slots=5, num_heads=2, num_layers=1)
        assert isinstance(layer, FixedTransformerAnchorLayer)
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
        ctx = torch.randn(2, D)  # (B, D)
        anchors, mask = layer(ctx)
        assert anchors.shape == (2, 1, D)
        assert mask.shape == (2, 1)
        assert mask.all()

    def test_passthrough(self):
        layer = ParentAnchorLayer(D)
        ctx = torch.randn(1, D)
        anchors, _ = layer(ctx)
        expected = ctx.unsqueeze(1)
        assert torch.allclose(anchors, expected)


class TestFixedAnchorLayer:
    def test_output_shape(self):
        layer = FixedAnchorLayer(D, num_slots=5)
        ctx = torch.randn(2, D)
        anchors, mask = layer(ctx)
        assert anchors.shape == (2, 5, D)
        assert mask.shape == (2, 5)
        assert mask.all()

    def test_with_count(self):
        layer = FixedAnchorLayer(D, num_slots=5)
        ctx = torch.randn(2, D)
        count = torch.tensor([3, 2])
        anchors, mask = layer(ctx, count=count)
        assert anchors.shape == (2, 5, D)
        assert mask[0, :3].all() and not mask[0, 3:].any()
        assert mask[1, :2].all() and not mask[1, 2:].any()

    def test_learnable(self):
        layer = FixedAnchorLayer(D, num_slots=3)
        assert layer.anchor_table.weight.requires_grad

    def test_context_conditioning(self):
        layer = FixedAnchorLayer(D, num_slots=3)
        ctx1 = torch.randn(1, D)
        ctx2 = torch.randn(1, D)
        anchors1, _ = layer(ctx1)
        anchors2, _ = layer(ctx2)
        # Different contexts should produce different anchors
        assert not torch.equal(anchors1, anchors2)

    def test_scalar_count(self):
        layer = FixedAnchorLayer(D, num_slots=5)
        ctx = torch.randn(2, D)
        count = torch.tensor(3)
        anchors, mask = layer(ctx, count=count)
        assert mask[0, :3].all() and not mask[0, 3:].any()
        assert mask[1, :3].all() and not mask[1, 3:].any()


class TestFixedLSTMAnchorLayer:
    def test_output_shape(self):
        layer = FixedLSTMAnchorLayer(D, num_slots=5)
        ctx = torch.randn(2, D)
        anchors, mask = layer(ctx)
        assert anchors.shape == (2, 5, D)
        assert mask.shape == (2, 5)
        assert mask.all()

    def test_with_count(self):
        layer = FixedLSTMAnchorLayer(D, num_slots=5)
        ctx = torch.randn(2, D)
        count = torch.tensor([3, 2])
        anchors, mask = layer(ctx, count=count)
        assert anchors.shape == (2, 5, D)
        assert mask[0, :3].all() and not mask[0, 3:].any()
        assert mask[1, :2].all() and not mask[1, 2:].any()

    def test_context_conditioning(self):
        layer = FixedLSTMAnchorLayer(D, num_slots=3)
        ctx1 = torch.randn(1, D)
        ctx2 = torch.randn(1, D)
        anchors1, _ = layer(ctx1)
        anchors2, _ = layer(ctx2)
        assert not torch.equal(anchors1, anchors2)

    def test_gradient_flows(self):
        layer = FixedLSTMAnchorLayer(D, num_slots=3)
        ctx = torch.randn(1, D, requires_grad=True)
        anchors, _ = layer(ctx)
        anchors.sum().backward()
        assert ctx.grad is not None


class TestFixedTransformerAnchorLayer:
    def test_output_shape(self):
        layer = FixedTransformerAnchorLayer(D, num_slots=5, num_heads=4, num_layers=1)
        ctx = torch.randn(2, D)
        anchors, mask = layer(ctx)
        assert anchors.shape == (2, 5, D)
        assert mask.shape == (2, 5)
        assert mask.all()

    def test_with_word_embeddings(self):
        layer = FixedTransformerAnchorLayer(D, num_slots=5, num_heads=4, num_layers=1)
        ctx = torch.randn(2, D)
        words = torch.randn(2, 10, D)
        anchors, mask = layer(ctx, word_embeddings=words)
        assert anchors.shape == (2, 5, D)

    def test_with_count(self):
        layer = FixedTransformerAnchorLayer(D, num_slots=5, num_heads=4, num_layers=1)
        ctx = torch.randn(2, D)
        count = torch.tensor([3, 2])
        anchors, mask = layer(ctx, count=count)
        assert mask[0, :3].all() and not mask[0, 3:].any()
        assert mask[1, :2].all() and not mask[1, 2:].any()

    def test_context_conditioning(self):
        layer = FixedTransformerAnchorLayer(D, num_slots=3, num_heads=4, num_layers=1)
        ctx1 = torch.randn(1, D)
        ctx2 = torch.randn(1, D)
        anchors1, _ = layer(ctx1)
        anchors2, _ = layer(ctx2)
        assert not torch.equal(anchors1, anchors2)

    def test_gradient_flows(self):
        layer = FixedTransformerAnchorLayer(D, num_slots=3, num_heads=4, num_layers=1)
        ctx = torch.randn(1, D, requires_grad=True)
        words = torch.randn(1, 5, D, requires_grad=True)
        anchors, _ = layer(ctx, word_embeddings=words)
        anchors.sum().backward()
        assert ctx.grad is not None
        assert words.grad is not None


class TestRotaryAnchorLayer:
    def test_output_types(self):
        layer = RotaryAnchorLayer(D, max_count=5)
        ctx = torch.randn(2, D)
        count = torch.tensor([2, 3])
        anchors, mask = layer(ctx, count=count)
        assert anchors.dim() == 3
        assert mask.dim() == 2
        assert mask.dtype == torch.bool

    def test_variable_counts(self):
        layer = RotaryAnchorLayer(D, max_count=10)
        ctx = torch.randn(2, D)
        count = torch.tensor([1, 3])
        anchors, mask = layer(ctx, count=count)
        # Max anchors should be max(1, 3) = 3
        assert anchors.shape[1] == 3
        assert mask[0, 0] and not mask[0, 1]
        assert mask[1, :3].all()


class TestQueryLSTMAnchorLayer:
    def test_no_word_embeddings(self):
        layer = QueryLSTMAnchorLayer(D)
        ctx = torch.randn(2, D)
        anchors, mask = layer(ctx, word_embeddings=None)
        assert anchors.shape == (2, 0, D)
        assert mask.shape == (2, 0)

    def test_with_word_embeddings(self):
        layer = QueryLSTMAnchorLayer(D, max_count=5)
        ctx = torch.randn(2, D)
        words = torch.randn(2, 10, D)
        count = torch.tensor([2, 3])
        anchors, mask = layer(ctx, word_embeddings=words, count=count)
        assert anchors.dim() == 3
        assert mask.dim() == 2


class TestQueryTransformerAnchorLayer:
    def test_no_word_embeddings(self):
        layer = QueryTransformerAnchorLayer(D, num_heads=2, num_layers=1)
        ctx = torch.randn(2, D)
        anchors, mask = layer(ctx, word_embeddings=None)
        assert anchors.shape == (2, 0, D)
        assert mask.shape == (2, 0)

    def test_with_word_embeddings(self):
        layer = QueryTransformerAnchorLayer(D, num_heads=2, num_layers=1)
        ctx = torch.randn(2, D)
        words = torch.randn(2, 10, D)
        count = torch.tensor([2, 3])
        anchors, mask = layer(ctx, word_embeddings=words, count=count)
        assert anchors.dim() == 3
        assert mask.dim() == 2
