"""Tests for ClassificationHead."""

import pytest
import torch

from glinext.tasks.classification.model import ClassificationHead
from glinext.tasks.classification.scorer import ClassificationScorer
from glinext.tasks import TaskHeadOutput, TaskFlatInputs
from tests.heads.conftest import make_config, D, B, W, C
from dataclasses import asdict
from glinext.config import ClassificationHeadConfig


def _make_head(**kwargs):
    config = make_config(classification_config=asdict(ClassificationHeadConfig(**kwargs)))
    return ClassificationHead.from_config(config)


# ── ClassificationScorer factory ─────────────────────────────────────────

class TestClassificationScorerFactory:
    def test_registry_has_all_types(self):
        expected = {"dot", "weighted-dot", "mlp", "hopfield"}
        assert expected.issubset(set(ClassificationScorer._registry))

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown scorer type"):
            ClassificationScorer.from_config("nonexistent", hidden_size=D)

    @pytest.mark.parametrize("scorer_type", ["dot", "weighted-dot", "mlp", "hopfield"])
    def test_scorer_forward_shape(self, scorer_type):
        scorer = ClassificationScorer.from_config(scorer_type, hidden_size=D)
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
        head = _make_head()
        assert head is not None
        assert head.name == "classification"
        assert hasattr(head, "anchor_layer")
        assert hasattr(head, "anchor_modeling")
        assert hasattr(head, "pooling")
        assert hasattr(head, "scorer")

    def test_from_config_with_refine(self):
        head = _make_head(anchor_refine_layers=1)
        assert hasattr(head, "anchor_refine")

    @pytest.mark.parametrize("scorer_type", ["dot", "weighted-dot", "mlp", "hopfield"])
    def test_from_config_scorer_types(self, scorer_type):
        head = _make_head(scorer_type=scorer_type)
        assert head is not None
        assert isinstance(head.scorer, ClassificationScorer)


# ── ClassificationHead forward ───────────────────────────────────────────

class TestClassificationHeadForward:
    def test_inference_with_flat_inputs(self, shared, flat_inputs):
        head = _make_head()
        out = head(shared, {}, flat_inputs=flat_inputs)
        assert out.loss is None
        assert out.logits is not None
        BN = flat_inputs.words_embedding.shape[0]
        assert out.logits.shape == (BN, C)

    def test_training_with_labels(self, shared, flat_inputs):
        head = _make_head()
        BN = flat_inputs.words_embedding.shape[0]
        labels = torch.zeros(BN, C)
        labels[:, 0] = 1.0  # first class positive
        out = head(shared, {}, flat_inputs=flat_inputs, cat_labels=labels)
        assert out.loss is not None
        assert out.loss.item() >= 0

    def test_inference_with_different_bn(self, shared, flat_inputs):
        """Test with BN > B (multiple groups per batch item)."""
        head = _make_head()
        BN = 4
        fi = TaskFlatInputs(
            words_embedding=torch.randn(BN, W, D),
            mask=torch.ones(BN, W, dtype=torch.long),
            parent_embedding=torch.randn(BN, D),
            child_embedding=torch.randn(BN, C, D),
            child_mask=torch.ones(BN, C, dtype=torch.long),
            batch_origin=torch.tensor([0, 0, 1, 1]),
        )
        out = head(shared, {}, flat_inputs=fi)
        assert out.logits is not None
        assert out.logits.shape == (BN, C)

    def test_gradient_flows(self, shared, flat_inputs):
        head = _make_head()
        BN = flat_inputs.words_embedding.shape[0]
        labels = torch.zeros(BN, C)
        labels[:, 0] = 1.0
        flat_inputs.words_embedding.requires_grad_(True)
        out = head(shared, {}, flat_inputs=flat_inputs, cat_labels=labels)
        out.loss.backward()
        assert flat_inputs.words_embedding.grad is not None
        flat_inputs.words_embedding.requires_grad_(False)

    @pytest.mark.parametrize("scorer_type", ["dot", "weighted-dot", "mlp", "hopfield"])
    def test_forward_all_scorers(self, shared, flat_inputs, scorer_type):
        head = _make_head(scorer_type=scorer_type)
        out = head(shared, {}, flat_inputs=flat_inputs)
        BN = flat_inputs.words_embedding.shape[0]
        assert out.logits.shape == (BN, C)
