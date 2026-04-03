"""Tests for OpenRelexHead."""

import pytest
import torch

from glinext.tasks.open_relex.model import OpenRelexHead
from glinext.tasks import TaskHeadOutput
from tests.heads.conftest import make_config, D, B, W, C
from dataclasses import asdict
from glinext.config import OpenRelexHeadConfig


def _make_head(**kwargs):
    defaults = dict(
        anchor_mode="fixed",
        num_fixed_slots=3,
        max_count=5,
    )
    defaults.update(kwargs)
    config = make_config(open_relex_config=asdict(OpenRelexHeadConfig(**defaults)))
    return OpenRelexHead.from_config(config)


# ── Construction ─────────────────────────────────────────────────────────

class TestOpenRelexHeadConstruction:
    def test_from_config_disabled(self):
        config = make_config()
        config.open_relex_config = None
        assert OpenRelexHead.from_config(config) is None

    def test_from_config_fixed_anchors(self):
        head = _make_head(anchor_mode="fixed")
        assert head is not None
        assert head.name == "open_relex"

    def test_from_config_with_span_rep(self):
        head = _make_head(represent_spans=True)
        assert head.represent_spans is True
        assert hasattr(head, "span_rep_layer")
        assert hasattr(head, "head_span_proj")
        assert hasattr(head, "tail_span_proj")

    def test_from_config_with_refine(self):
        head = _make_head(anchor_refine_layers=1)
        assert hasattr(head, "anchor_refine")

    def test_has_dual_scorers(self):
        head = _make_head()
        assert hasattr(head, "head_scorer")
        assert hasattr(head, "tail_scorer")


# ── Forward ──────────────────────────────────────────────────────────────

class TestOpenRelexHeadForward:
    def test_inference_with_flat_inputs(self, shared, flat_inputs):
        head = _make_head()
        out = head(shared, {}, flat_inputs=flat_inputs)
        assert out.logits is not None
        # Shape: (BN, X, C, L, 2, 3)
        BN = flat_inputs.words_embedding.shape[0]
        assert out.logits.shape[0] == BN
        assert out.logits.shape[-2] == 2  # head/tail
        assert out.logits.shape[-1] == 3  # start/inside/end

    def test_empty_rel_embedding(self, shared, flat_inputs):
        head = _make_head()
        # Force empty child embeddings
        flat_inputs.child_embedding = torch.randn(B, 0, D)
        flat_inputs.child_mask = torch.ones(B, 0)
        out = head(shared, {}, flat_inputs=flat_inputs)
        assert out.logits is None  # empty rel embedding → empty output

    def test_extra_contains_anchors(self, shared, flat_inputs):
        head = _make_head()
        out = head(shared, {}, flat_inputs=flat_inputs)
        assert "anchors" in out.extra
        assert "anchor_mask" in out.extra

    def test_training_with_labels(self, shared, flat_inputs):
        head = _make_head(num_fixed_slots=2)
        out_infer = head(shared, {}, flat_inputs=flat_inputs)
        # Get output shape to build matching labels
        X = out_infer.logits.shape[1]  # anchors
        C_out = out_infer.logits.shape[2]  # rel classes
        L = out_infer.logits.shape[3]  # seq len
        BN = flat_inputs.words_embedding.shape[0]

        labels = torch.zeros(BN, X, C_out, L, 2, 3)
        from gliner.modeling.loss_functions import focal_loss_with_logits
        out = head(
            shared, {},
            flat_inputs=flat_inputs,
            open_rel_labels=labels,
            base_loss_fn=focal_loss_with_logits,
        )
        assert out.loss is not None
        assert out.loss.item() >= 0

    def test_span_representation_forward(self, shared, flat_inputs):
        head = _make_head(represent_spans=True)
        S = 4  # number of span candidates
        BN = flat_inputs.words_embedding.shape[0]
        span_idx = torch.randint(0, W, (BN, S, 2))
        # Ensure start <= end
        span_idx[:, :, 1] = span_idx.max(dim=-1).values
        span_idx[:, :, 0] = span_idx.min(dim=-1).values
        span_mask = torch.ones(BN, S, dtype=torch.bool)

        out = head(
            shared, {},
            flat_inputs=flat_inputs,
            open_rel_span_idx=span_idx,
            open_rel_span_mask=span_mask,
        )
        assert out.extra["span_logits"] is not None
        # span_logits: (B, S, X, C, 2)
        assert out.extra["span_logits"].shape[1] == S
        assert out.extra["span_logits"].shape[-1] == 2  # head/tail


class TestOpenRelexGradient:
    def test_gradient_flows(self, shared, flat_inputs):
        head = _make_head(num_fixed_slots=2)
        flat_inputs.words_embedding.requires_grad_(True)

        out_infer = head(shared, {}, flat_inputs=flat_inputs)
        X = out_infer.logits.shape[1]
        C_out = out_infer.logits.shape[2]
        L = out_infer.logits.shape[3]
        BN = flat_inputs.words_embedding.shape[0]

        labels = torch.zeros(BN, X, C_out, L, 2, 3)
        from gliner.modeling.loss_functions import focal_loss_with_logits

        # Need fresh forward with grad enabled
        flat_inputs.words_embedding = flat_inputs.words_embedding.detach().requires_grad_(True)
        out = head(
            shared, {},
            flat_inputs=flat_inputs,
            open_rel_labels=labels,
            base_loss_fn=focal_loss_with_logits,
        )
        out.loss.backward()
        assert flat_inputs.words_embedding.grad is not None