"""Tests for EmbeddingHead and EmbeddingLoss."""

import pytest
import torch
from types import SimpleNamespace

from glinext.tasks.embedding.model import EmbeddingHead, EmbeddingLoss, MSELoss, ContrastiveLoss, TripletLoss
from tests.heads.conftest import make_config, D, B, W, C
from dataclasses import asdict
from glinext.config import EmbeddingHeadConfig


# ── EmbeddingLoss registry ───────────────────────────────────────────────

class TestEmbeddingLossRegistry:
    def test_mse(self):
        loss = EmbeddingLoss.from_config("mse")
        assert isinstance(loss, MSELoss)

    def test_contrastive(self):
        loss = EmbeddingLoss.from_config("contrastive")
        assert isinstance(loss, ContrastiveLoss)

    def test_triplet(self):
        loss = EmbeddingLoss.from_config("triplet")
        assert isinstance(loss, TripletLoss)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown loss function"):
            EmbeddingLoss.from_config("nonexistent")


class TestEmbeddingLossForward:
    def test_mse_perfect(self):
        loss = MSELoss()
        sims = torch.tensor([1.0, 0.0])
        labels = torch.tensor([1.0, 0.0])
        assert loss(sims, labels).item() == pytest.approx(0.0, abs=1e-6)

    def test_mse_nonzero(self):
        loss = MSELoss()
        sims = torch.tensor([0.5])
        labels = torch.tensor([1.0])
        assert loss(sims, labels).item() > 0

    def test_contrastive(self):
        loss = ContrastiveLoss(margin=1.0)
        sims = torch.tensor([0.9, 0.1])
        labels = torch.tensor([1.0, 0.0])
        val = loss(sims, labels)
        assert val.item() >= 0


# ── EmbeddingHead construction ───────────────────────────────────────────

class TestEmbeddingHeadConstruction:
    def test_from_config_disabled(self):
        config = make_config(embedding_config=None)
        config.embedding_config = None
        assert EmbeddingHead.from_config(config) is None

    def test_from_config_default(self):
        config = make_config(embedding_config=asdict(EmbeddingHeadConfig()))
        head = EmbeddingHead.from_config(config)
        assert head is not None
        assert head.name == "embedding"
        assert head.similarity_fn == "cosine"

    def test_from_config_dot_similarity(self):
        config = make_config(embedding_config=asdict(EmbeddingHeadConfig(similarity_fn="dot")))
        head = EmbeddingHead.from_config(config)
        assert head.similarity_fn == "dot"

    def test_from_config_l2_similarity(self):
        config = make_config(embedding_config=asdict(EmbeddingHeadConfig(similarity_fn="l2")))
        head = EmbeddingHead.from_config(config)
        assert head.similarity_fn == "l2"

    def test_from_config_projection_dim(self):
        config = make_config(embedding_config=asdict(EmbeddingHeadConfig(projection_dim=32)))
        head = EmbeddingHead.from_config(config)
        assert head.projection is not None
        pair_idx = torch.tensor([[0, 1]])
        shared = SimpleNamespace(
            words_embedding=torch.randn(B, W, D),
            mask=torch.ones(B, W, dtype=torch.long),
        )
        out = head(shared, {}, embedding_pair_idx=pair_idx)
        assert out.logits.shape == (1,)


# ── EmbeddingHead forward ───────────────────────────────────────────────

class TestEmbeddingHeadForward:
    def _make_head(self, **emb_kwargs):
        config = make_config(embedding_config=asdict(EmbeddingHeadConfig(**emb_kwargs)))
        return EmbeddingHead.from_config(config)

    def test_no_pair_idx_returns_empty(self, shared):
        head = self._make_head()
        out = head(shared, {})
        assert out.loss is None
        assert out.logits is None

    def test_inference_no_labels(self, shared):
        head = self._make_head()
        pair_idx = torch.tensor([[0, 1]])  # compare batch items 0 and 1
        out = head(shared, {}, embedding_pair_idx=pair_idx)
        assert out.loss is None
        assert out.logits is not None
        assert out.logits.shape == (1,)

    def test_training_with_labels(self, shared):
        head = self._make_head()
        pair_idx = torch.tensor([[0, 1]])
        labels = torch.tensor([0.9])
        out = head(shared, {}, embedding_pair_idx=pair_idx, embedding_labels=labels)
        assert out.loss is not None
        assert out.loss.item() >= 0
        assert out.logits.shape == (1,)

    def test_cosine_similarity_bounded(self, shared):
        head = self._make_head(similarity_fn="cosine")
        pair_idx = torch.tensor([[0, 1]])
        out = head(shared, {}, embedding_pair_idx=pair_idx)
        assert out.logits.item() >= -1.0 - 1e-5
        assert out.logits.item() <= 1.0 + 1e-5

    def test_multiple_pairs(self, shared):
        head = self._make_head()
        pair_idx = torch.tensor([[0, 1], [0, 0], [1, 1]])
        out = head(shared, {}, embedding_pair_idx=pair_idx)
        assert out.logits.shape == (3,)
        # Self-similarity should be high for cosine
        assert out.logits[1].item() > out.logits[0].item() - 0.5  # relaxed check

    def test_dot_similarity(self, shared):
        head = self._make_head(similarity_fn="dot")
        pair_idx = torch.tensor([[0, 1]])
        out = head(shared, {}, embedding_pair_idx=pair_idx)
        assert out.logits.shape == (1,)
        # Dot product is unbounded
        assert out.logits.isfinite().all()

    def test_l2_similarity(self, shared):
        head = self._make_head(similarity_fn="l2")
        pair_idx = torch.tensor([[0, 1]])
        out = head(shared, {}, embedding_pair_idx=pair_idx)
        assert out.logits.shape == (1,)
        assert out.logits.item() <= 0  # negative L2 distance

    def test_gradient_flows(self, shared):
        head = self._make_head()
        pair_idx = torch.tensor([[0, 1]])
        labels = torch.tensor([0.5])
        # Enable grads on input
        shared.words_embedding.requires_grad_(True)
        out = head(shared, {}, embedding_pair_idx=pair_idx, embedding_labels=labels)
        out.loss.backward()
        assert shared.words_embedding.grad is not None
        shared.words_embedding.requires_grad_(False)

    def test_contrastive_loss(self, shared):
        head = self._make_head(loss_fn="contrastive")
        pair_idx = torch.tensor([[0, 1]])
        labels = torch.tensor([1.0])
        out = head(shared, {}, embedding_pair_idx=pair_idx, embedding_labels=labels)
        assert out.loss is not None
        assert out.loss.item() >= 0
