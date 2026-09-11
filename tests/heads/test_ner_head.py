"""Tests for NERHead."""

import pytest
import torch

from gliformer.tasks.ner.model import NERHead
from gliformer.tasks import TaskHeadOutput
from gliformer.tasks.losses import binary_focal_or_bce
from tests.heads.conftest import make_config, D, B, W, C
from dataclasses import asdict
from gliformer.config import NERHeadConfig


def _ones_loss(logits, targets):
    """Elementwise loss of 1.0 per cell, so reductions are readable."""
    return torch.ones_like(logits)


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

    def test_span_loss_default_is_a_mean_over_active_span_class_cells(
        self,
        shared,
        flat_inputs,
    ):
        """The auxiliary span term is reduced like the token term.

        Left as a masked sum it outweighs the mean-reduced token loss by
        roughly the active (BN x S x C) cell count.
        """
        head = _make_head(represent_spans=True)
        assert head.span_loss_reduction == "mean"
        span_count = 4
        ner_labels = torch.zeros(B, W, C, 3)
        span_idx = torch.zeros(B, span_count, 2, dtype=torch.long)
        span_idx[:, :, 1] = 1
        span_mask = torch.ones(B, span_count, dtype=torch.bool)
        span_labels = torch.zeros(B, span_count, C)
        span_labels[0, 0, 0] = 1.0
        call = dict(
            flat_inputs=flat_inputs,
            ner_labels=ner_labels,
            span_idx=span_idx,
            span_mask=span_mask,
            span_labels=span_labels,
            base_loss_fn=_ones_loss,
        )

        token_only = head(shared, {}, flat_inputs=flat_inputs,
                          ner_labels=ner_labels, base_loss_fn=_ones_loss)
        with_spans = head(shared, {}, **call)

        # Every element costs 1.0. The token term is 3.0 (three BIO channels
        # per active cell, which are excluded from its denominator) and the
        # span term is the mean over B * span_count * C cells, so 1.0.
        assert token_only.loss.item() == pytest.approx(3.0)
        assert with_spans.loss.item() == pytest.approx(4.0)

    def test_span_loss_sum_keeps_the_historical_objective(
        self,
        shared,
        flat_inputs,
    ):
        head = _make_head(represent_spans=True, span_loss_reduction="sum")
        span_count = 4
        ner_labels = torch.zeros(B, W, C, 3)
        span_idx = torch.zeros(B, span_count, 2, dtype=torch.long)
        span_idx[:, :, 1] = 1
        span_mask = torch.ones(B, span_count, dtype=torch.bool)
        span_labels = torch.zeros(B, span_count, C)

        output = head(
            shared,
            {},
            flat_inputs=flat_inputs,
            ner_labels=ner_labels,
            span_idx=span_idx,
            span_mask=span_mask,
            span_labels=span_labels,
            base_loss_fn=_ones_loss,
        )

        # Token mean (3.0) plus the unnormalized span sum over B * S * C,
        # which is the term that used to swamp it.
        assert output.loss.item() == pytest.approx(3.0 + B * span_count * C)

    def test_span_loss_reduction_rejects_unknown_values(self):
        with pytest.raises(ValueError, match="span_loss_reduction"):
            NERHeadConfig(span_loss_reduction="avg")

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
