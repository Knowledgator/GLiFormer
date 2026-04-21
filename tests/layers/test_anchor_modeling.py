"""Tests for anchor modeling layers."""

import pytest
import torch

from glinext.layers.anchor_modeling import (
    AnchorModeling, LinearAnchorModeling, LSTMAnchorModeling, MLPAnchorModeling,
    TransformerAnchorModeling,
)

D = 16


class TestAnchorModelingFactory:
    def test_linear(self):
        layer = AnchorModeling.from_config("linear", D)
        assert isinstance(layer, LinearAnchorModeling)

    def test_lstm(self):
        layer = AnchorModeling.from_config("lstm", D)
        assert isinstance(layer, LSTMAnchorModeling)

    def test_mlp(self):
        layer = AnchorModeling.from_config("mlp", D)
        assert isinstance(layer, MLPAnchorModeling)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown modeling_type"):
            AnchorModeling.from_config("anything_else", D)

    def test_transformer(self):
        layer = AnchorModeling.from_config("transformer", D)
        assert isinstance(layer, TransformerAnchorModeling)

    def test_registry_has_all_types(self):
        for t in ["linear", "lstm", "mlp", "transformer"]:
            assert t in AnchorModeling._registry


class TestLinearAnchorModeling:
    def test_output_shape(self):
        layer = LinearAnchorModeling(D)
        anchors = torch.randn(2, 3, D)   # 3 anchors
        children = torch.randn(2, 5, D)  # 5 children
        out = layer(anchors, children)
        assert out.shape == (2, 3, 5, D)

    def test_single_anchor_child(self):
        layer = LinearAnchorModeling(D)
        anchors = torch.randn(1, 1, D)
        children = torch.randn(1, 1, D)
        out = layer(anchors, children)
        assert out.shape == (1, 1, 1, D)

    def test_gradient_flows(self):
        layer = LinearAnchorModeling(D)
        a = torch.randn(1, 2, D, requires_grad=True)
        c = torch.randn(1, 3, D, requires_grad=True)
        out = layer(a, c)
        out.sum().backward()
        assert a.grad is not None
        assert c.grad is not None


class TestLSTMAnchorModeling:
    def test_output_shape(self):
        layer = LSTMAnchorModeling(D)
        anchors = torch.randn(2, 3, D)
        children = torch.randn(2, 5, D)
        out = layer(anchors, children)
        assert out.shape == (2, 3, 5, D)

    def test_single_anchor(self):
        layer = LSTMAnchorModeling(D)
        anchors = torch.randn(1, 1, D)
        children = torch.randn(1, 4, D)
        out = layer(anchors, children)
        assert out.shape == (1, 1, 4, D)

    def test_single_child(self):
        layer = LSTMAnchorModeling(D)
        anchors = torch.randn(1, 3, D)
        children = torch.randn(1, 1, D)
        out = layer(anchors, children)
        assert out.shape == (1, 3, 1, D)


class TestMLPAnchorModeling:
    def test_output_shape(self):
        layer = MLPAnchorModeling(D)
        anchors = torch.randn(2, 3, D)
        children = torch.randn(2, 5, D)
        out = layer(anchors, children)
        assert out.shape == (2, 3, 5, D)

    def test_with_dropout(self):
        layer = MLPAnchorModeling(D, dropout=0.5)
        anchors = torch.randn(1, 2, D)
        children = torch.randn(1, 3, D)
        # Eval mode for deterministic output
        layer.eval()
        out = layer(anchors, children)
        assert out.shape == (1, 2, 3, D)


class TestTransformerAnchorModeling:
    def test_output_shape(self):
        layer = TransformerAnchorModeling(D, num_heads=4)
        anchors = torch.randn(2, 3, D)
        children = torch.randn(2, 5, D)
        out = layer(anchors, children)
        assert out.shape == (2, 3, 5, D)

    def test_single_anchor(self):
        layer = TransformerAnchorModeling(D, num_heads=4)
        anchors = torch.randn(1, 1, D)
        children = torch.randn(1, 4, D)
        out = layer(anchors, children)
        assert out.shape == (1, 1, 4, D)

    def test_single_child(self):
        layer = TransformerAnchorModeling(D, num_heads=4)
        anchors = torch.randn(1, 3, D)
        children = torch.randn(1, 1, D)
        out = layer(anchors, children)
        assert out.shape == (1, 3, 1, D)

    def test_gradient_flows(self):
        layer = TransformerAnchorModeling(D, num_heads=4)
        a = torch.randn(1, 2, D, requires_grad=True)
        c = torch.randn(1, 3, D, requires_grad=True)
        out = layer(a, c)
        out.sum().backward()
        assert a.grad is not None
        assert c.grad is not None
