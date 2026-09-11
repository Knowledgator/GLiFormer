"""Tests for MLP and projection utilities."""

import pytest
import torch
from torch import nn
from types import SimpleNamespace

from gliformer.layers.mlp import create_mlp, FeaturesProjector


class TestCreateMlp:
    def test_output_shape(self):
        mlp = create_mlp(32, [64], 16)
        x = torch.randn(2, 32)
        assert mlp(x).shape == (2, 16)

    def test_multiple_intermediate(self):
        mlp = create_mlp(32, [64, 128], 16)
        x = torch.randn(2, 32)
        assert mlp(x).shape == (2, 16)

    def test_no_intermediate(self):
        mlp = create_mlp(32, [], 16)
        x = torch.randn(2, 32)
        assert mlp(x).shape == (2, 16)
        # Should be just a single linear layer
        assert len(mlp) == 1

    def test_with_layer_norm(self):
        mlp = create_mlp(32, [64], 16, add_layer_norm=True)
        assert any(isinstance(m, nn.LayerNorm) for m in mlp.modules())

    def test_without_layer_norm(self):
        mlp = create_mlp(32, [64], 16, add_layer_norm=False)
        assert not any(isinstance(m, nn.LayerNorm) for m in mlp.modules())

    def test_no_dropout(self):
        mlp = create_mlp(32, [64], 16, dropout=0.0)
        assert not any(isinstance(m, nn.Dropout) for m in mlp.modules())

    def test_activations(self):
        for act in ("relu", "tanh", "sigmoid", "leaky_relu", "gelu"):
            mlp = create_mlp(16, [32], 8, activation=act)
            x = torch.randn(1, 16)
            out = mlp(x)
            assert out.shape == (1, 8)

    def test_batch_dims(self):
        mlp = create_mlp(16, [32], 8)
        x = torch.randn(4, 10, 16)
        assert mlp(x).shape == (4, 10, 8)

    def test_returns_sequential(self):
        mlp = create_mlp(16, [32], 8)
        assert isinstance(mlp, nn.Sequential)


class TestFeaturesProjector:
    @pytest.fixture
    def config(self):
        enc = SimpleNamespace(hidden_size=64)
        return SimpleNamespace(
            encoder_config=enc,
            hidden_size=128,
            dropout=0.1,
            projector_hidden_act="gelu",
        )

    def test_output_shape(self, config):
        proj = FeaturesProjector(config)
        x = torch.randn(2, 10, 64)
        out = proj(x)
        assert out.shape == (2, 10, 64)  # projects back to encoder hidden size

    def test_bottleneck(self, config):
        proj = FeaturesProjector(config)
        # linear_1: 64 -> 128, linear_2: 128 -> 64
        assert proj.linear_1.in_features == 64
        assert proj.linear_1.out_features == 128
        assert proj.linear_2.in_features == 128
        assert proj.linear_2.out_features == 64