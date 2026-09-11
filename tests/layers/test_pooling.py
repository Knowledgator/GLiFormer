"""Tests for configurable pooling layers."""

import pytest
import torch

from gliformer.layers.pooling import (
    Pooling, MeanPooling, CLSPooling, MaxPooling, WeightedPooling,
)

D = 16  # hidden size for tests


@pytest.fixture
def embeddings():
    """(B=2, L=5, D=16) token embeddings."""
    return torch.randn(2, 5, D)


@pytest.fixture
def mask():
    """(B=2, L=5) attention mask — second sample has 3 valid tokens."""
    return torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0],
    ])


class TestPoolingRegistry:
    def test_registered_types(self):
        for name in ("mean", "cls", "max", "weighted"):
            assert name in Pooling._registry

    def test_from_config_mean(self):
        p = Pooling.from_config("mean")
        assert isinstance(p, MeanPooling)

    def test_from_config_cls(self):
        assert isinstance(Pooling.from_config("cls"), CLSPooling)

    def test_from_config_max(self):
        assert isinstance(Pooling.from_config("max"), MaxPooling)

    def test_from_config_weighted(self):
        p = Pooling.from_config("weighted", hidden_size=D)
        assert isinstance(p, WeightedPooling)

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown pooling type"):
            Pooling.from_config("nonexistent")


class TestMeanPooling:
    def test_output_shape(self, embeddings, mask):
        p = MeanPooling()
        out = p(embeddings, mask)
        assert out.shape == (2, D)

    def test_masking(self):
        emb = torch.ones(1, 4, D)
        mask = torch.tensor([[1, 1, 0, 0]])
        p = MeanPooling()
        out = p(emb, mask)
        # Mean of first 2 tokens (all ones) = 1.0
        assert torch.allclose(out, torch.ones(1, D))

    def test_all_masked(self):
        emb = torch.randn(1, 3, D)
        mask = torch.zeros(1, 3)
        p = MeanPooling()
        out = p(emb, mask)
        # Clamped denominator prevents NaN
        assert not torch.isnan(out).any()


class TestCLSPooling:
    def test_output_shape(self, embeddings, mask):
        p = CLSPooling()
        out = p(embeddings, mask)
        assert out.shape == (2, D)

    def test_returns_first_token(self, embeddings, mask):
        p = CLSPooling()
        out = p(embeddings, mask)
        assert torch.equal(out, embeddings[:, 0])

    def test_empty_sequence_returns_shape_safe_zeros(self):
        embeddings = torch.empty(2, 0, D)
        mask = torch.empty(2, 0)

        out = CLSPooling()(embeddings, mask)

        assert out.shape == (2, D)
        assert torch.equal(out, torch.zeros(2, D))


class TestMaxPooling:
    def test_output_shape(self, embeddings, mask):
        p = MaxPooling()
        out = p(embeddings, mask)
        assert out.shape == (2, D)

    def test_masking(self):
        emb = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        mask = torch.tensor([[1, 1, 0]])
        p = MaxPooling()
        out = p(emb, mask)
        # Max of first 2 tokens: [3.0, 4.0]
        assert torch.allclose(out, torch.tensor([[3.0, 4.0]]))


class TestWeightedPooling:
    def test_output_shape(self, embeddings, mask):
        p = WeightedPooling(hidden_size=D)
        out = p(embeddings, mask)
        assert out.shape == (2, D)

    def test_has_learnable_weights(self):
        p = WeightedPooling(hidden_size=D)
        assert hasattr(p, "pool_weights")
        assert p.pool_weights.in_features == D

    def test_masking(self):
        p = WeightedPooling(hidden_size=4)
        emb = torch.randn(1, 3, 4)
        mask = torch.tensor([[1, 0, 0]])
        out = p(emb, mask)
        assert out.shape == (1, 4)
        # With only 1 valid token, output should equal that token's embedding
        # (softmax over single valid = weight 1.0)
        expected = emb[:, 0]
        assert torch.allclose(out, expected, atol=1e-5)
