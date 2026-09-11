"""Conv record anchors and centre-based matching in the set structuring head."""

from dataclasses import asdict

import pytest
import torch
from gliner.modeling.loss_functions import focal_loss_with_logits

from gliformer.config import SetStructuringHeadConfig
from gliformer.layers.anchor_layer import ConvAnchorLayer
from gliformer.tasks import SharedRepresentations, TaskFlatInputs
from gliformer.tasks.set_structuring.model import SetStructuringHead
from tests.heads.conftest import make_config, D, C

B = 2
W = 24     # long enough for several conv anchors at stride 4
STRIDE = 4


def _make_head(**overrides):
    defaults = dict(
        reuse_ner_head=False,
        anchor_layer={
            "type": "conv",
            "params": {
                "kernel_size": 3,
                "stride": STRIDE,
                "num_layers": 2,
                "dilation": 2,
                "max_slots": 32,
                "dropout": 0.0,
            },
        },
        anchor_matching_strategy="center",
        anchor_normalization="none",
        anchor_refinement="none",
        anchor_refine_layers=0,
        multi_level=False,
        anchor_objectness=True,
    )
    defaults.update(overrides)
    config = make_config(
        set_structuring_config=asdict(SetStructuringHeadConfig(**defaults)),
        structuring_config=None,
    )
    head = SetStructuringHead.from_config(config)
    head.eval()
    return head


@pytest.fixture
def shared():
    return SharedRepresentations(
        token_embeds=torch.randn(B, W, D),
        input_ids=torch.ones(B, W, dtype=torch.long),
        attention_mask=torch.ones(B, W, dtype=torch.long),
        words_embedding=torch.randn(B, W, D),
        mask=torch.ones(B, W, dtype=torch.long),
        prompts_embedding=torch.randn(B, C, D),
        prompts_embedding_mask=torch.ones(B, C, dtype=torch.long),
    )


@pytest.fixture
def flat_inputs():
    return TaskFlatInputs(
        words_embedding=torch.randn(B, W, D),
        mask=torch.ones(B, W, dtype=torch.long),
        parent_embedding=torch.randn(B, D),
        child_embedding=torch.randn(B, C, D),
        child_mask=torch.ones(B, C, dtype=torch.long),
        batch_origin=torch.arange(B),
    )


class TestConstruction:
    def test_builds_with_conv_anchors(self):
        head = _make_head()
        assert isinstance(head.record_anchor_layer, ConvAnchorLayer)
        assert head.anchor_matching_strategy == "center"

    def test_conv_anchors_are_not_truncated_by_gold_count(self):
        # `static_slots` keeps the gold record count out of the anchor layer:
        # a conv slot index is a document position, not a record ordinal.
        assert _make_head()._record_uses_fixed_slots is True

    def test_center_matching_rejects_a_non_positional_layer(self):
        with pytest.raises(ValueError, match="anchor_word_positions"):
            _make_head(
                anchor_layer={"type": "fixed", "params": {"num_slots": 8}},
            )

    def test_hungarian_still_accepts_a_non_positional_layer(self):
        head = _make_head(
            anchor_layer={"type": "fixed", "params": {"num_slots": 8}},
            anchor_matching_strategy="hungarian",
        )
        assert head.anchor_matching_strategy == "hungarian"


class TestForward:
    def _labels(self, head, shared, flat_inputs, span_idx, records):
        """Membership labels ``(B, E, G, C)`` from record -> entity indices."""
        entity_count = span_idx.shape[1]
        labels = torch.zeros(B, entity_count, len(records), C)
        for record_idx, entities in enumerate(records):
            for entity_idx in entities:
                labels[:, entity_idx, record_idx, 0] = 1.0
        return labels

    def test_anchor_axis_follows_the_document(self, shared, flat_inputs):
        head = _make_head()
        out = head(shared, {}, flat_inputs=flat_inputs)
        membership = out.extra["membership_logits"]
        assert membership.shape[1] == -(-W // STRIDE)
        assert bool(out.extra["anchor_mask"].all())

    def test_records_match_the_anchor_covering_them(
        self, shared, flat_inputs
    ):
        head = _make_head()
        span_idx = torch.tensor([[[2, 3], [5, 6], [17, 18], [20, 21]]])
        span_idx = span_idx.expand(B, -1, -1).contiguous()
        span_mask = torch.ones(B, span_idx.shape[1], dtype=torch.bool)
        # Record 0 owns the early spans, record 1 the late ones.
        labels = self._labels(
            head, shared, flat_inputs, span_idx, [[0, 1], [2, 3]]
        )

        out = head(
            shared,
            {},
            flat_inputs=flat_inputs,
            set_structuring_span_idx=span_idx,
            set_structuring_span_mask=span_mask,
            set_structuring_span_labels=labels,
            base_loss_fn=focal_loss_with_logits,
        )
        matches = out.extra["anchor_matches"]
        assert matches is not None
        by_gold = {gold: anchor for anchor, gold in matches[0]}
        # Anchor k is centred on word k*stride; record 0 starts at word 2 and
        # record 1 at word 17, so they land on the nearest covering anchors.
        assert by_gold[0] == round(2 / STRIDE)
        assert by_gold[1] == round(17 / STRIDE)

    def test_matching_is_independent_of_the_membership_scores(
        self, shared, flat_inputs
    ):
        # The property centre matching exists for: the pairing must not move
        # when the scores move, or a slot is trained positive on one step and
        # negative on the next.
        span_idx = torch.tensor([[[1, 2], [15, 16]]]).expand(B, -1, -1)
        span_idx = span_idx.contiguous()
        span_mask = torch.ones(B, 2, dtype=torch.bool)
        labels = torch.zeros(B, 2, 2, C)
        labels[:, 0, 0, 0] = 1.0
        labels[:, 1, 1, 0] = 1.0

        seen = []
        for seed in range(4):
            torch.manual_seed(seed)
            head = _make_head()      # fresh random membership scorer
            out = head(
                shared,
                {},
                flat_inputs=flat_inputs,
                set_structuring_span_idx=span_idx,
                set_structuring_span_mask=span_mask,
                set_structuring_span_labels=labels,
                base_loss_fn=focal_loss_with_logits,
            )
            seen.append(sorted(out.extra["anchor_matches"][0]))
        assert all(match == seen[0] for match in seen)

    def test_loss_is_finite(self, shared, flat_inputs):
        head = _make_head()
        span_idx = torch.tensor([[[1, 2], [15, 16]]]).expand(B, -1, -1)
        span_idx = span_idx.contiguous()
        labels = torch.zeros(B, 2, 2, C)
        labels[:, 0, 0, 0] = 1.0
        labels[:, 1, 1, 0] = 1.0
        out = head(
            shared,
            {},
            flat_inputs=flat_inputs,
            set_structuring_span_idx=span_idx,
            set_structuring_span_mask=torch.ones(B, 2, dtype=torch.bool),
            set_structuring_span_labels=labels,
            base_loss_fn=focal_loss_with_logits,
        )
        assert out.loss is not None
        assert torch.isfinite(out.loss)
