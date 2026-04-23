"""Tests for StructuringHead."""

import pytest
import torch

from glinext.tasks.structuring.model import StructuringHead
import glinext.tasks.structuring.model as structuring_model
from glinext.tasks import TaskHeadOutput
from tests.heads.conftest import make_config, D, B, W, C
from dataclasses import asdict
from glinext.config import StructuringHeadConfig


def _make_head(**kwargs):
    defaults = dict(
        anchor_mode="fixed",
        num_fixed_slots=3,
        max_count=5,
    )
    defaults.update(kwargs)
    config = make_config(structuring_config=asdict(StructuringHeadConfig(**defaults)))
    return StructuringHead.from_config(config)


# ── Construction ─────────────────────────────────────────────────────────

class TestStructuringHeadConstruction:
    def test_from_config_disabled(self):
        config = make_config()
        config.structuring_config = None
        assert StructuringHead.from_config(config) is None

    def test_from_config_fixed(self):
        head = _make_head(anchor_mode="fixed")
        assert head is not None
        assert head.name == "structuring"

    def test_from_config_rotary(self):
        head = _make_head(anchor_mode="rotary")
        assert head is not None

    def test_from_config_query_lstm(self):
        head = _make_head(anchor_mode="query_lstm")
        assert head is not None

    def test_from_config_with_span_rep(self):
        head = _make_head(represent_spans=True)
        assert head.represent_spans is True
        assert hasattr(head, "span_rep_layer")

    def test_from_config_with_refine(self):
        head = _make_head(anchor_refine_layers=1)
        assert hasattr(head, "anchor_refine")

    def test_anchor_modeling_types(self):
        for am_type in ["linear", "lstm", "mlp"]:
            head = _make_head(anchor_modeling=am_type)
            assert head is not None


# ── Forward ──────────────────────────────────────────────────────────────

class TestStructuringHeadForward:
    def test_inference_with_flat_inputs(self, shared, flat_inputs):
        head = _make_head()
        out = head(shared, {}, flat_inputs=flat_inputs)
        assert out.logits is not None
        # Shape: (BN, X, C, L, 3) — but C is from child_embedding dim
        BN = flat_inputs.words_embedding.shape[0]
        assert out.logits.shape[0] == BN
        assert out.logits.shape[-1] == 3  # start/inside/end

    def test_empty_child_embedding(self, shared, flat_inputs):
        head = _make_head()
        flat_inputs.child_embedding = torch.randn(B, 0, D)
        flat_inputs.child_mask = torch.ones(B, 0)
        out = head(shared, {}, flat_inputs=flat_inputs)
        assert out.logits is None

    def test_extra_contains_group_info(self, shared, flat_inputs):
        head = _make_head()
        out = head(shared, {}, flat_inputs=flat_inputs)
        assert "groups_output" in out.extra
        assert "anchor_mask" in out.extra

    def test_training_with_labels(self, shared, flat_inputs):
        head = _make_head(num_fixed_slots=2)
        out_infer = head(shared, {}, flat_inputs=flat_inputs)
        # Logits: (BN, X, C, L, 3) but labels expected as (BN, X, L, C, 3)
        BN, X, C_out, L, _ = out_infer.logits.shape
        labels = torch.zeros(BN, X, L, C_out, 3)

        from gliner.modeling.loss_functions import focal_loss_with_logits
        out = head(
            shared, {},
            flat_inputs=flat_inputs,
            structuring_labels=labels,
            base_loss_fn=focal_loss_with_logits,
        )
        assert out.loss is not None
        assert out.loss.item() >= 0

    def test_with_gold_count(self, shared, flat_inputs):
        head = _make_head()
        gold_count = torch.tensor([2, 3])
        out = head(shared, {}, flat_inputs=flat_inputs, gold_count_val=gold_count)
        assert out.logits is not None

    def test_span_representation_forward(self, shared, flat_inputs):
        head = _make_head(represent_spans=True)
        S = 4
        BN = flat_inputs.words_embedding.shape[0]
        span_idx = torch.randint(0, W, (BN, S, 2))
        span_idx[:, :, 1] = span_idx.max(dim=-1).values
        span_idx[:, :, 0] = span_idx.min(dim=-1).values
        span_mask = torch.ones(BN, S, dtype=torch.bool)

        # Labels required — span-level is a training-only signal.
        out_infer = head(shared, {}, flat_inputs=flat_inputs)
        BN_out, X, C_out, L, _ = out_infer.logits.shape
        labels = torch.zeros(BN_out, X, L, C_out, 3)
        span_labels = torch.zeros(BN_out, X, S, C_out)

        out = head(
            shared, {},
            flat_inputs=flat_inputs,
            structuring_span_idx=span_idx,
            structuring_span_mask=span_mask,
            structuring_span_labels=span_labels,
            structuring_labels=labels,
        )
        assert out.extra["span_logits"] is not None
        # span_logits: (B, X, S, C)
        assert out.extra["span_logits"].shape[2] == S

    def test_span_representation_skipped_at_inference(self, shared, flat_inputs):
        """At inference (no labels) the span-level head is skipped so the BIO
        path drives decoding with consistent threshold semantics."""
        head = _make_head(represent_spans=True)
        out = head(shared, {}, flat_inputs=flat_inputs)
        assert out.extra["span_logits"] is None


class TestStructuringGradient:
    def test_gradient_flows(self, shared, flat_inputs):
        head = _make_head(num_fixed_slots=2)
        flat_inputs.words_embedding = flat_inputs.words_embedding.detach().requires_grad_(True)

        out_infer = head(shared, {}, flat_inputs=flat_inputs)
        # Logits: (BN, X, C, L, 3) but labels expected as (BN, X, L, C, 3)
        BN, X, C_out, L, _ = out_infer.logits.shape
        labels = torch.zeros(BN, X, L, C_out, 3)

        from gliner.modeling.loss_functions import focal_loss_with_logits

        flat_inputs.words_embedding = flat_inputs.words_embedding.detach().requires_grad_(True)
        out = head(
            shared, {},
            flat_inputs=flat_inputs,
            structuring_labels=labels,
            base_loss_fn=focal_loss_with_logits,
        )
        out.loss.backward()
        assert flat_inputs.words_embedding.grad is not None
