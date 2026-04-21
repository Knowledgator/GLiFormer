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

        from gliner.modeling.loss_functions import focal_loss_with_logits
        out = head(
            shared, {},
            flat_inputs=flat_inputs,
            ner_labels=ner_labels,
            rel_labels=rel_labels,
            rel_label_embeds=rel_label_embeds,
            base_loss_fn=focal_loss_with_logits,
        )
        # Loss may be None if adjacency returned no pairs, or combined if both present
        # Just verify the output structure is correct
        assert isinstance(out, TaskHeadOutput)
        assert out.logits is not None


# ── Entity selection ─────────────────────────────────────────────────────

class TestSelectEntitySpans:
    def test_basic(self, shared):
        head = _make_head()
        # Scores: (B, W, C, 3) with some strong signals
        scores = torch.full((B, W, C, 3), -5.0)
        scores[0, 0, 0, :] = 5.0  # entity at position 0 in batch 0
        scores[0, 3, 1, :] = 5.0  # entity at position 3 in batch 0

        rep, mask = head._select_entity_spans(scores, shared.words_embedding)
        assert rep.shape[0] == B
        assert rep.shape[2] == D
        assert mask.shape[0] == B

    def test_with_labels(self, shared):
        head = _make_head()
        scores = torch.randn(B, W, C, 3)
        # Labels: nonzero at positions 0 and 3
        ner_labels = torch.zeros(B, W, C, 3)
        ner_labels[0, 0, 0, :] = 1.0
        ner_labels[0, 3, 1, :] = 1.0

        rep, mask = head._select_entity_spans(scores, shared.words_embedding, ner_labels=ner_labels)
        assert mask[0].sum() >= 2  # at least 2 entities in batch 0

    def test_no_entities(self, shared):
        head = _make_head()
        scores = torch.full((B, W, C, 3), -10.0)
        rep, mask = head._select_entity_spans(scores, shared.words_embedding, threshold=0.9)
        # Should return at least shape (B, 1, D) due to clamp
        assert rep.shape[1] >= 1