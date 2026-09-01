"""Tests for NERHead."""

import pytest
import torch

from glinext.tasks.ner.model import NERHead
from glinext.tasks import TaskHeadOutput
from glinext.tasks.losses import binary_focal_or_bce
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

    def test_loss_is_normalized_by_active_bn_token_class_cells(self):
        head = _make_head()
        scores = torch.zeros(2, 1, 3, 2, 3)
        labels = torch.zeros_like(scores)
        anchor_mask = torch.ones(2, 1)
        word_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        child_mask = torch.tensor([[1, 1], [1, 0]])

        loss = head._bio_loss(
            scores,
            labels,
            anchor_mask,
            word_mask,
            child_mask,
            base_loss_fn=lambda logits, targets: torch.ones_like(logits),
        )

        # Active L*C cells: 2*2 + 1*1 = 5. Each cell has three BIO terms.
        assert loss.item() == pytest.approx(3.0)

    def test_direct_head_uses_configured_focal_parameters(
        self,
        shared,
        flat_inputs,
    ):
        head = _make_head(
            focal_loss_alpha=0.4,
            focal_loss_gamma=1.5,
            focal_loss_prob_margin=0.1,
        )
        labels = torch.zeros(B, W, C, 3)

        output = head(
            shared,
            {},
            flat_inputs=flat_inputs,
            ner_labels=labels,
        )

        expected = binary_focal_or_bce(
            output.logits,
            labels,
            focal_loss_alpha=0.4,
            focal_loss_gamma=1.5,
            focal_loss_prob_margin=0.1,
        ).sum() / (B * W * C)
        torch.testing.assert_close(output.loss, expected)

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
