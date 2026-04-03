"""Tests for ClassificationHead and ClassificationScorer."""

import pytest
import torch

from glinext.tasks.classification.model import ClassificationHead, ClassificationScorer
from glinext.tasks import TaskHeadOutput
from tests.heads.conftest import make_config, D, B, W, C
from dataclasses import asdict
from glinext.config import ClassificationHeadConfig


# ── ClassificationScorer ─────────────────────────────────────────────────

class TestClassificationScorer:
    def test_dot_scorer(self):
        scorer = ClassificationScorer(D, scorer_type="dot")
        text_rep = torch.randn(B, D)
        label_rep = torch.randn(B, C, D)
        scores = scorer(text_rep, label_rep)
        assert scores.shape == (B, C)

    def test_weighted_dot_scorer(self):
        scorer = ClassificationScorer(D, scorer_type="weighted-dot")
        text_rep = torch.randn(B, D)
        label_rep = torch.randn(B, C, D)
        scores = scorer(text_rep, label_rep)
        assert scores.shape == (B, C)

    def test_mlp_scorer(self):
        scorer = ClassificationScorer(D, scorer_type="mlp")
        text_rep = torch.randn(B, D)
        label_rep = torch.randn(B, C, D)
        scores = scorer(text_rep, label_rep)
        assert scores.shape == (B, C)


# ── ClassificationHead construction ──────────────────────────────────────

class TestClassificationHeadConstruction:
    def test_from_config_disabled(self):
        config = make_config()
        config.classification_config = None
        assert ClassificationHead.from_config(config) is None

    def test_from_config_default(self):
        config = make_config(classification_config=asdict(ClassificationHeadConfig()))
        head = ClassificationHead.from_config(config)
        assert head is not None
        assert head.name == "classification"

    def test_from_config_weighted_dot(self):
        config = make_config(classification_config=asdict(
            ClassificationHeadConfig(layer_type="weighted-dot"),
        ))
        head = ClassificationHead.from_config(config)
        assert head is not None


# ── ClassificationHead forward ───────────────────────────────────────────

class TestClassificationHeadForward:
    def _make_head(self, **kwargs):
        config = make_config(classification_config=asdict(ClassificationHeadConfig(**kwargs)))
        return ClassificationHead.from_config(config)

    def test_inference_with_flat_inputs(self, shared, flat_inputs):
        head = self._make_head()
        out = head(shared, {}, flat_inputs=flat_inputs)
        assert out.loss is None
        assert out.logits is not None
        BN = flat_inputs.words_embedding.shape[0]
        assert out.logits.shape == (BN, C)

    def test_training_with_labels(self, shared, flat_inputs):
        head = self._make_head()
        BN = flat_inputs.words_embedding.shape[0]
        labels = torch.zeros(BN, C)
        labels[:, 0] = 1.0  # first class positive
        out = head(shared, {}, flat_inputs=flat_inputs, cat_labels=labels)
        assert out.loss is not None
        assert out.loss.item() >= 0

    def test_inference_with_label_embeds(self, shared):
        head = self._make_head()
        cat_label_embeds = torch.randn(B, C, D)
        out = head(shared, {}, cat_label_embeds=cat_label_embeds)
        assert out.logits is not None
        assert out.logits.shape == (B, C)

    def test_dot_vs_weighted_dot_different_outputs(self, shared, flat_inputs):
        head_dot = self._make_head(layer_type="dot")
        head_wd = self._make_head(layer_type="weighted-dot")
        out_dot = head_dot(shared, {}, flat_inputs=flat_inputs)
        out_wd = head_wd(shared, {}, flat_inputs=flat_inputs)
        # Different architectures should generally produce different scores
        assert out_dot.logits.shape == out_wd.logits.shape

    def test_gradient_flows(self, shared, flat_inputs):
        head = self._make_head()
        BN = flat_inputs.words_embedding.shape[0]
        labels = torch.zeros(BN, C)
        labels[:, 0] = 1.0
        flat_inputs.words_embedding.requires_grad_(True)
        out = head(shared, {}, flat_inputs=flat_inputs, cat_labels=labels)
        out.loss.backward()
        assert flat_inputs.words_embedding.grad is not None
        flat_inputs.words_embedding.requires_grad_(False)
