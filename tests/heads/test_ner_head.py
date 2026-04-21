"""Tests for NERHead."""

import pytest
import torch

from glinext.tasks.ner.model import NERHead
from glinext.tasks import TaskHeadOutput
from tests.heads.conftest import make_config, D, B, W, C
from dataclasses import asdict
from glinext.config import NERHeadConfig


def _make_head(**kwargs):
    config = make_config(ner_config=asdict(NERHeadConfig(**kwargs)))
    return NERHead.from_config(config)


# ── Construction ─────────────────────────────────────────────────────────

class TestNERHeadConstruction:
    def test_from_config_disabled(self):
        config = make_config()
        config.ner_config = None
        assert NERHead.from_config(config) is None

    def test_from_config_default(self):
        head = _make_head()
        assert head is not None
        assert head.name == "ner"
        assert hasattr(head, "anchor_layer")
        assert hasattr(head, "anchor_modeling")

    def test_from_config_with_span_rep(self):
        head = _make_head(represent_spans=True)
        assert head.represent_spans is True
        assert hasattr(head, "span_rep_layer")

    def test_from_config_with_refine(self):
        head = _make_head(anchor_refine_layers=2)
        assert hasattr(head, "anchor_refine")


# ── Forward ──────────────────────────────────────────────────────────────

class TestNERHeadForward:
    def test_inference(self, shared, flat_inputs):
        head = _make_head()
        out = head(shared, {}, flat_inputs=flat_inputs)
        assert out.logits is not None
        assert out.logits.shape[0] == B
        assert out.logits.shape[-1] == 3  # start/end/inside
        assert out.loss is None

    def test_output_shape_squeezed(self, shared, flat_inputs):
        """Parent anchor mode (A=1) squeezes to (B, W, C, 3)."""
        head = _make_head()
        out = head(shared, {}, flat_inputs=flat_inputs)
        assert out.logits.dim() == 4  # (B, W, C, 3), not (B, A, W, C, 3)

    def test_inference_flat_inputs(self, shared, flat_inputs):
        head = _make_head()
        out = head(shared, {}, flat_inputs=flat_inputs)
        assert out.logits is not None
        BN = flat_inputs.words_embedding.shape[0]
        assert out.logits.shape[0] == BN

    def test_training_with_labels(self, shared, flat_inputs):
        head = _make_head()
        ner_labels = torch.zeros(B, W, C, 3)
        ner_labels[0, 0, 0, 0] = 1.0  # start signal

        from gliner.modeling.loss_functions import focal_loss_with_logits
        out = head(shared, {}, flat_inputs=flat_inputs, ner_labels=ner_labels, base_loss_fn=focal_loss_with_logits)
        assert out.loss is not None
        assert out.loss.item() >= 0

    def test_output_extra_contains_embeddings(self, shared, flat_inputs):
        head = _make_head()
        out = head(shared, {}, flat_inputs=flat_inputs)
        assert "words_embedding" in out.extra
        assert "mask" in out.extra


# ── _fit_length ──────────────────────────────────────────────────────────

class TestFitLength:
    def test_pad(self, shared):
        head = _make_head()
        tensor = torch.randn(B, 5, D)
        mask = torch.ones(B, 5)
        t, m = head._fit_length(tensor, mask, 8)
        assert t.shape == (B, 8, D)
        assert m.shape == (B, 8)
        assert (m[:, 5:] == 0).all()

    def test_trim(self, shared):
        head = _make_head()
        tensor = torch.randn(B, 10, D)
        mask = torch.ones(B, 10)
        t, m = head._fit_length(tensor, mask, 6)
        assert t.shape == (B, 6, D)
        assert m.shape == (B, 6)

    def test_exact_size(self, shared):
        head = _make_head()
        tensor = torch.randn(B, 7, D)
        mask = torch.ones(B, 7)
        t, m = head._fit_length(tensor, mask, 7)
        assert t.shape == (B, 7, D)
        assert torch.equal(t, tensor)


# ── Gradient flow ────────────────────────────────────────────────────────

class TestNERGradient:
    def test_gradient_flows(self, shared, flat_inputs):
        head = _make_head()
        flat_inputs.words_embedding.requires_grad_(True)
        ner_labels = torch.zeros(B, W, C, 3)
        from gliner.modeling.loss_functions import focal_loss_with_logits
        out = head(shared, {}, flat_inputs=flat_inputs, ner_labels=ner_labels, base_loss_fn=focal_loss_with_logits)
        out.loss.backward()
        assert flat_inputs.words_embedding.grad is not None
        flat_inputs.words_embedding.requires_grad_(False)
