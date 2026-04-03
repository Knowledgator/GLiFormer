"""Tests for rotary position embeddings."""

import pytest
import torch

from glinext.layers.rotary import RotaryEmbedding, rotate_half, apply_rotary_pos_emb


class TestRotaryEmbedding:
    def test_init(self):
        rope = RotaryEmbedding(dim=16)
        assert rope.dim == 16
        assert rope.inv_freq.shape == (8,)

    def test_odd_dim_raises(self):
        with pytest.raises(ValueError, match="even"):
            RotaryEmbedding(dim=15)

    def test_forward_shapes(self):
        rope = RotaryEmbedding(dim=16)
        x_like = torch.randn(2, 5, 16)
        position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
        cos, sin = rope(x_like, position_ids)
        assert cos.shape == (2, 5, 16)
        assert sin.shape == (2, 5, 16)

    def test_cos_sin_range(self):
        rope = RotaryEmbedding(dim=8)
        x_like = torch.randn(1, 10, 8)
        pos = torch.arange(10).unsqueeze(0)
        cos, sin = rope(x_like, pos)
        assert cos.abs().max() <= 1.0 + 1e-5
        assert sin.abs().max() <= 1.0 + 1e-5

    def test_different_positions_give_different_embeddings(self):
        rope = RotaryEmbedding(dim=16)
        x_like = torch.randn(1, 4, 16)
        pos = torch.arange(4).unsqueeze(0)
        cos, sin = rope(x_like, pos)
        # Position 0 and position 3 should differ
        assert not torch.allclose(cos[0, 0], cos[0, 3])

    def test_attention_scaling(self):
        rope1 = RotaryEmbedding(dim=16, attention_scaling=1.0)
        rope2 = RotaryEmbedding(dim=16, attention_scaling=2.0)
        x_like = torch.randn(1, 5, 16)
        pos = torch.arange(5).unsqueeze(0)
        cos1, _ = rope1(x_like, pos)
        cos2, _ = rope2(x_like, pos)
        # Scaling 2.0 should produce 2x the values
        assert torch.allclose(cos2, cos1 * 2.0, atol=1e-5)


class TestRotateHalf:
    def test_shape_preserved(self):
        x = torch.randn(2, 3, 8)
        assert rotate_half(x).shape == (2, 3, 8)

    def test_double_rotation_negates(self):
        x = torch.randn(2, 3, 8)
        # Applying rotate_half twice: (-x2, x1) -> (-x1, -x2) = -x
        assert torch.allclose(rotate_half(rotate_half(x)), -x)


class TestApplyRotaryPosEmb:
    def test_shape_preserved(self):
        q = torch.randn(2, 5, 16)
        rope = RotaryEmbedding(dim=16)
        pos = torch.arange(5).unsqueeze(0).expand(2, -1)
        cos, sin = rope(q, pos)
        out = apply_rotary_pos_emb(q, cos, sin)
        assert out.shape == q.shape

    def test_zero_positions(self):
        """At position 0, cos≈1 and sin≈0, so output ≈ input."""
        q = torch.randn(1, 1, 16)
        rope = RotaryEmbedding(dim=16)
        pos = torch.zeros(1, 1, dtype=torch.long)
        cos, sin = rope(q, pos)
        out = apply_rotary_pos_emb(q, cos, sin)
        assert torch.allclose(out, q, atol=1e-5)
