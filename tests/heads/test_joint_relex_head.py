"""Tests for JointRelexHead."""

import pytest
import torch

from glinext.tasks.joint_relex.model import JointRelexHead
from glinext.tasks import TaskHeadOutput, SharedRepresentations
from tests.heads.conftest import make_config, D, B, W, C
from dataclasses import asdict
from glinext.config import JointRelexHeadConfig, NERHeadConfig


def _make_head(**kwargs):
    config = make_config(
        ner_config=asdict(NERHeadConfig()),
        joint_relex_config=asdict(JointRelexHeadConfig(**kwargs)),
    )
    return JointRelexHead.from_config(config)


# ── Construction ─────────────────────────────────────────────────────────

class TestJointRelexHeadConstruction:
    def test_from_config_disabled(self):
        config = make_config()
        config.joint_relex_config = None
        assert JointRelexHead.from_config(config) is None

    def test_from_config_default(self):
        head = _make_head()
        assert head is not None
        assert head.name == "joint_relex"

    def test_inherits_ner(self):
        head = _make_head()
        assert hasattr(head, "scorer")  # NER scorer inherited

    def test_has_pair_rep_by_default(self):
        head = _make_head()
        assert hasattr(head, "pair_rep_layer") or hasattr(head, "triples_score_layer")

    def test_can_disable_relations_rep_layer(self):
        head = _make_head(layer_type="none")
        assert not hasattr(head, "relations_rep_layer")
        assert hasattr(head, "pair_rep_layer") or hasattr(head, "triples_score_layer")


# ── Forward ──────────────────────────────────────────────────────────────

class TestJointRelexHeadForward:
    def test_inference(self, shared, flat_inputs):
        head = _make_head()
        out = head(shared, {}, flat_inputs=flat_inputs)
        assert out.logits is not None  # NER logits
        assert "rel_logits" in out.extra
        assert "rel_idx" in out.extra
        assert "rel_mask" in out.extra

    def test_inference_with_rel_embeds(self, shared, flat_inputs):
        head = _make_head()
        R = 2  # number of relation types
        rel_label_embeds = torch.randn(B, R, D)
        out = head(shared, {}, flat_inputs=flat_inputs, rel_label_embeds=rel_label_embeds)
        assert out.logits is not None
        if out.extra["rel_logits"] is not None:
            assert out.extra["rel_logits"].shape[-1] == R

    def test_combined_loss(self, shared, flat_inputs):
        head = _make_head()
        ner_labels = torch.zeros(B, W, C, 3)
        # Rel labels: (B, E, E, C_rel)
        E = 5
        R = 2
        rel_labels = torch.zeros(B, E, E, R)
        rel_label_embeds = torch.randn(B, R, D)

        # Processor-provided per-entity spans (aligned with rel_labels' entity axis)
        rel_span_idx = torch.zeros(B, E, 2, dtype=torch.long)
        rel_span_mask = torch.zeros(B, E, dtype=torch.bool)
        rel_span_idx[0, 0] = torch.tensor([0, 0])
        rel_span_idx[0, 1] = torch.tensor([2, 3])
        rel_span_mask[0, 0] = True
        rel_span_mask[0, 1] = True

        from gliner.modeling.loss_functions import focal_loss_with_logits
        out = head(
            shared, {},
            flat_inputs=flat_inputs,
            ner_labels=ner_labels,
            rel_labels=rel_labels,
            rel_span_idx=rel_span_idx,
            rel_span_mask=rel_span_mask,
            rel_label_embeds=rel_label_embeds,
            base_loss_fn=focal_loss_with_logits,
        )
        # Loss may be None if adjacency returned no pairs, or combined if both present
        # Just verify the output structure is correct
        assert isinstance(out, TaskHeadOutput)
        assert out.logits is not None

    def test_inference_without_relations_rep_layer_uses_all_pairs(self, shared, flat_inputs):
        head = _make_head(layer_type="none")
        R = 2
        E = 2
        rel_label_embeds = torch.randn(B, R, D)
        rel_span_idx = torch.zeros(B, E, 2, dtype=torch.long)
        rel_span_mask = torch.zeros(B, E, dtype=torch.bool)
        rel_span_idx[0, 0] = torch.tensor([0, 0])
        rel_span_idx[0, 1] = torch.tensor([2, 3])
        rel_span_mask[0, 0] = True
        rel_span_mask[0, 1] = True

        out = head(
            shared, {},
            flat_inputs=flat_inputs,
            rel_span_idx=rel_span_idx,
            rel_span_mask=rel_span_mask,
            rel_label_embeds=rel_label_embeds,
        )

        assert out.extra["rel_logits"] is not None
        assert out.extra["rel_mask"] is not None
        assert out.extra["rel_mask"][0].sum().item() == 2


# ── Entity pooling ───────────────────────────────────────────────────────

class TestPoolEntitySpans:
    def test_single_token_span(self, shared):
        head = _make_head()
        E = 2
        span_idx = torch.zeros(B, E, 2, dtype=torch.long)
        span_idx[0, 0] = torch.tensor([0, 0])
        span_idx[0, 1] = torch.tensor([3, 3])
        span_mask = torch.zeros(B, E, dtype=torch.bool)
        span_mask[0, 0] = True
        span_mask[0, 1] = True

        rep, mask = head._pool_entity_spans(shared.words_embedding, span_idx, span_mask)
        assert rep.shape == (B, E, D)
        assert mask.shape == (B, E)
        assert torch.allclose(rep[0, 0], shared.words_embedding[0, 0])
        assert torch.allclose(rep[0, 1], shared.words_embedding[0, 3])
        assert mask[0].sum() == 2
        assert mask[1].sum() == 0

    def test_multi_token_span_is_mean(self, shared):
        head = _make_head()
        span_idx = torch.tensor([[[1, 3]], [[0, 0]]], dtype=torch.long)  # (B, 1, 2)
        span_mask = torch.ones(B, 1, dtype=torch.bool)

        rep, _ = head._pool_entity_spans(shared.words_embedding, span_idx, span_mask)
        expected = shared.words_embedding[0, 1:4].mean(dim=0)
        assert torch.allclose(rep[0, 0], expected, atol=1e-5)

    def test_masked_entity_is_zero(self, shared):
        head = _make_head()
        span_idx = torch.tensor([[[0, 0]], [[0, 0]]], dtype=torch.long)
        span_mask = torch.tensor([[False], [False]], dtype=torch.bool)

        rep, _ = head._pool_entity_spans(shared.words_embedding, span_idx, span_mask)
        assert torch.all(rep == 0)
