"""Tests for group/anchor generation layers."""

import pytest
import torch

from glinext.layers.groups import (
    AnchorCrossAttentionLayer, RotaryGroupLSTM, QueryGroupLSTM, QueryGroupTransformer,
)

D = 16


class TestAnchorCrossAttentionLayer:
    def test_output_shape(self):
        layer = AnchorCrossAttentionLayer(D, num_heads=4, num_layers=2)
        anchors = torch.randn(2, 3, D)
        tokens = torch.randn(2, 10, D)
        out = layer(anchors, tokens)
        assert out.shape == (2, 3, D)

    def test_single_layer(self):
        layer = AnchorCrossAttentionLayer(D, num_heads=4, num_layers=1)
        anchors = torch.randn(1, 2, D)
        tokens = torch.randn(1, 5, D)
        out = layer(anchors, tokens)
        assert out.shape == (1, 2, D)

    def test_with_mask(self):
        layer = AnchorCrossAttentionLayer(D, num_heads=4, num_layers=1)
        anchors = torch.randn(1, 2, D)
        tokens = torch.randn(1, 5, D)
        mask = torch.tensor([[True, True, True, False, False]])
        out = layer(anchors, tokens, token_mask=mask)
        assert out.shape == (1, 2, D)

    def test_gradient_flows(self):
        layer = AnchorCrossAttentionLayer(D, num_heads=4, num_layers=1)
        anchors = torch.randn(1, 2, D, requires_grad=True)
        tokens = torch.randn(1, 5, D, requires_grad=True)
        out = layer(anchors, tokens)
        out.sum().backward()
        assert anchors.grad is not None
        assert tokens.grad is not None


class TestRotaryGroupLSTM:
    def test_output_shape(self):
        layer = RotaryGroupLSTM(D, max_count=10)
        pc_emb = torch.randn(3, D)  # 3 fields
        out = layer(pc_emb, gold_count_val=4)
        assert out.shape == (4, 3, D)  # (count, M, D)

    def test_single_count(self):
        layer = RotaryGroupLSTM(D, max_count=10)
        pc_emb = torch.randn(2, D)
        out = layer(pc_emb, gold_count_val=1)
        assert out.shape == (1, 2, D)

    def test_exceeds_max_count(self):
        layer = RotaryGroupLSTM(D, max_count=5)
        pc_emb = torch.randn(2, D)
        out = layer(pc_emb, gold_count_val=8)
        # Should handle counts exceeding max by repeating
        assert out.shape == (8, 2, D)

    def test_single_field(self):
        layer = RotaryGroupLSTM(D, max_count=5)
        pc_emb = torch.randn(1, D)
        out = layer(pc_emb, gold_count_val=3)
        assert out.shape == (3, 1, D)


class TestQueryGroupLSTM:
    def test_with_gold_count(self):
        layer = QueryGroupLSTM(D, max_count=10)
        pc_emb = torch.randn(3, D)
        token_emb = torch.randn(2, 10, D)
        count = torch.tensor([2, 3])
        out, mask = layer(pc_emb, token_emb, gold_count_val=count)
        assert out.shape[0] == 2  # batch
        assert out.shape[2] == D
        assert mask.shape[0] == 2

    def test_without_gold_count(self):
        layer = QueryGroupLSTM(D, max_count=10)
        pc_emb = torch.randn(3, D)
        token_emb = torch.randn(2, 8, D)
        out, mask = layer(pc_emb, token_emb, gold_count_val=None, threshold=0.5)
        assert out.shape == (2, 8, D)  # all tokens
        assert mask.shape == (2, 8)

    def test_output_types(self):
        layer = QueryGroupLSTM(D, max_count=10)
        pc_emb = torch.randn(2, D)
        token_emb = torch.randn(1, 5, D)
        count = torch.tensor([3])
        out, mask = layer(pc_emb, token_emb, gold_count_val=count)
        assert out.dtype == torch.float32
        assert mask.dtype == torch.bool


class TestQueryGroupTransformer:
    def test_with_gold_count(self):
        layer = QueryGroupTransformer(D, num_heads=4, num_layers=1)
        pc_emb = torch.randn(3, D)
        token_emb = torch.randn(2, 10, D)
        count = torch.tensor([2, 3])
        out, mask = layer(pc_emb, token_emb, gold_count_val=count)
        assert out.shape[0] == 2
        assert out.shape[2] == D
        assert mask.shape[0] == 2

    def test_without_gold_count(self):
        layer = QueryGroupTransformer(D, num_heads=4, num_layers=1)
        pc_emb = torch.randn(3, D)
        token_emb = torch.randn(2, 8, D)
        out, mask = layer(pc_emb, token_emb, gold_count_val=None, threshold=0.5)
        assert out.shape == (2, 8, D)
        assert mask.shape == (2, 8)

    def test_gradient_flows(self):
        layer = QueryGroupTransformer(D, num_heads=4, num_layers=1)
        pc_emb = torch.randn(2, D, requires_grad=True)
        token_emb = torch.randn(1, 5, D, requires_grad=True)
        count = torch.tensor([3])
        out, mask = layer(pc_emb, token_emb, gold_count_val=count)
        out.sum().backward()
        assert pc_emb.grad is not None
        assert token_emb.grad is not None
