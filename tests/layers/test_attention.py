"""Tests for attention blocks and fusers."""

import pytest
import torch

from glinext.layers.attention import (
    SelfAttentionBlock, CrossAttentionBlock, Fuser, LayerwiseAttention,
)

D = 16
HEADS = 4


class TestSelfAttentionBlock:
    def test_output_shape(self):
        block = SelfAttentionBlock(d_model=D, num_heads=HEADS)
        x = torch.randn(2, 5, D)
        out = block(x)
        assert out.shape == (2, 5, D)

    def test_residual_connection(self):
        block = SelfAttentionBlock(d_model=D, num_heads=HEADS)
        x = torch.randn(1, 3, D)
        out = block(x)
        # Output should differ from input (attention + norm changes values)
        assert not torch.equal(out, x)

    def test_single_token(self):
        block = SelfAttentionBlock(d_model=D, num_heads=HEADS)
        x = torch.randn(1, 1, D)
        out = block(x)
        assert out.shape == (1, 1, D)


class TestCrossAttentionBlock:
    def test_output_shape(self):
        # nn.MultiheadAttention expects (L, B, D) format
        block = CrossAttentionBlock(d_model=D, num_heads=HEADS)
        q = torch.randn(3, 2, D)   # (L_q, B, D)
        kv = torch.randn(7, 2, D)  # (L_kv, B, D)
        out = block(q, kv, kv)
        assert out.shape == (3, 2, D)

    def test_different_kv_lengths(self):
        block = CrossAttentionBlock(d_model=D, num_heads=HEADS)
        q = torch.randn(2, 1, D)
        kv = torch.randn(10, 1, D)
        out = block(q, kv, kv)
        assert out.shape == (2, 1, D)


class TestFuser:
    def test_output_shape(self):
        # Fuser uses nn.MultiheadAttention (seq-first by default)
        fuser = Fuser(d_model=D, num_heads=HEADS, num_layers=2)
        q = torch.randn(5, 2, D)  # (L_q, B, D)
        k = torch.randn(7, 2, D)  # (L_k, B, D)
        out = fuser(q, k)
        assert out.shape == (5, 2, D)

    def test_single_layer(self):
        fuser = Fuser(d_model=D, num_heads=HEADS, num_layers=1)
        q = torch.randn(3, 1, D)
        k = torch.randn(4, 1, D)
        out = fuser(q, k)
        assert out.shape == (3, 1, D)

    def test_num_layers(self):
        fuser = Fuser(d_model=D, num_heads=HEADS, num_layers=3)
        assert len(fuser.layers) == 3
        for layer_pair in fuser.layers:
            assert isinstance(layer_pair[0], SelfAttentionBlock)
            assert isinstance(layer_pair[1], CrossAttentionBlock)


class TestLayerwiseAttention:
    def test_output_shape(self):
        la = LayerwiseAttention(num_layers=4, hidden_size=D)
        layers = [torch.randn(2, 10, D) for _ in range(4)]
        out = la(layers)
        assert out.shape == (2, 10, D)

    def test_custom_output_size(self):
        la = LayerwiseAttention(num_layers=4, hidden_size=D, output_size=32)
        layers = [torch.randn(2, 10, D) for _ in range(4)]
        out = la(layers)
        assert out.shape == (2, 10, 32)

    def test_two_layers(self):
        la = LayerwiseAttention(num_layers=2, hidden_size=D)
        layers = [torch.randn(1, 5, D) for _ in range(2)]
        out = la(layers)
        assert out.shape == (1, 5, D)

    def test_single_sample(self):
        la = LayerwiseAttention(num_layers=3, hidden_size=D)
        layers = [torch.randn(1, 8, D) for _ in range(3)]
        out = la(layers)
        assert out.shape == (1, 8, D)