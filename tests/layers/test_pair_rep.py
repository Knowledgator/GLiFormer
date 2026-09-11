"""Tests for pair representation and prompt relation extraction layers."""

import pytest
import torch

from gliformer.layers.pair_rep import PairRepLayer, PromptRelationExtractor

D = 16


class TestPairRepLayer:
    @pytest.mark.parametrize("pair_type", ["concat_proj", "bilinear", "additive", "mlp"])
    def test_output_shape(self, pair_type):
        layer = PairRepLayer(hidden_size=D, pair_rep_type=pair_type)
        head = torch.randn(2, 5, D)
        tail = torch.randn(2, 5, D)
        out = layer(head, tail)
        assert out.shape == (2, 5, D)

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown pair_rep_type"):
            PairRepLayer(hidden_size=D, pair_rep_type="nonexistent")

    def test_batch_dim(self):
        layer = PairRepLayer(hidden_size=D, pair_rep_type="concat_proj")
        head = torch.randn(4, 3, D)
        tail = torch.randn(4, 3, D)
        out = layer(head, tail)
        assert out.shape == (4, 3, D)

    def test_single_entity(self):
        layer = PairRepLayer(hidden_size=D, pair_rep_type="bilinear")
        head = torch.randn(1, 1, D)
        tail = torch.randn(1, 1, D)
        out = layer(head, tail)
        assert out.shape == (1, 1, D)


class TestPromptRelationExtractor:
    def test_output_shape(self):
        ext = PromptRelationExtractor(hidden_size=D)
        entities = torch.randn(2, 5, D)
        prompts = torch.randn(2, 3, D)
        out = ext(entities, prompts)
        # (B, E_src, E_tgt, C)
        assert out.shape == (2, 5, 5, 3)

    def test_self_loops_masked(self):
        ext = PromptRelationExtractor(hidden_size=D)
        entities = torch.randn(1, 4, D)
        prompts = torch.randn(1, 2, D)
        out = ext(entities, prompts)
        # Diagonal should be zero (no self-loops)
        for i in range(4):
            assert torch.allclose(out[0, i, i], torch.zeros(2))

    def test_entity_mask(self):
        ext = PromptRelationExtractor(hidden_size=D)
        entities = torch.randn(1, 4, D)
        prompts = torch.randn(1, 2, D)
        mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.float)
        out = ext(entities, prompts, entity_mask=mask)
        # Masked entities (2, 3) should produce zero scores
        assert torch.allclose(out[0, 2, :, :], torch.zeros(4, 2))
        assert torch.allclose(out[0, :, 2, :], torch.zeros(4, 2))

    def test_single_entity(self):
        ext = PromptRelationExtractor(hidden_size=D)
        entities = torch.randn(1, 1, D)
        prompts = torch.randn(1, 3, D)
        out = ext(entities, prompts)
        # Only self-loop which is masked -> all zeros
        assert out.shape == (1, 1, 1, 3)
        assert torch.allclose(out, torch.zeros_like(out))
