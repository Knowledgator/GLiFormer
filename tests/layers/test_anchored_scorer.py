"""Tests for anchored span scorer."""

import pytest
import torch

from glinext.layers.anchored_scorer import AnchoredSpanScorer

D = 16


class TestAnchoredSpanScorer:
    def test_output_shape(self):
        scorer = AnchoredSpanScorer(hidden_size=D)
        fused = torch.randn(2, 3, D)   # 3 classes
        words = torch.randn(2, 10, D)  # 10 tokens
        out = scorer(fused, words)
        assert out.shape == (2, 3, 10, 3)  # (B, N, L, 3)

    def test_single_class(self):
        scorer = AnchoredSpanScorer(hidden_size=D)
        fused = torch.randn(1, 1, D)
        words = torch.randn(1, 5, D)
        out = scorer(fused, words)
        assert out.shape == (1, 1, 5, 3)

    def test_word_mask(self):
        scorer = AnchoredSpanScorer(hidden_size=D)
        fused = torch.randn(1, 2, D)
        words = torch.randn(1, 6, D)
        mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
        out = scorer(fused, words, word_mask=mask)
        # Masked positions should be zero
        assert torch.allclose(out[0, :, 3:, :], torch.zeros(2, 3, 3))

    def test_no_mask(self):
        scorer = AnchoredSpanScorer(hidden_size=D)
        fused = torch.randn(2, 4, D)
        words = torch.randn(2, 8, D)
        out = scorer(fused, words, word_mask=None)
        assert out.shape == (2, 4, 8, 3)

    def test_gradient_flows(self):
        scorer = AnchoredSpanScorer(hidden_size=D)
        fused = torch.randn(1, 2, D, requires_grad=True)
        words = torch.randn(1, 5, D, requires_grad=True)
        out = scorer(fused, words)
        loss = out.sum()
        loss.backward()
        assert fused.grad is not None
        assert words.grad is not None
