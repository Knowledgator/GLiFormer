"""Tests for entity-first open relation extraction."""

import warnings

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from glinext.config import GLiNextConfig, OpenRelexHeadConfig
from glinext.layers import AnchorLayer
from glinext.model import BaseGLiNextModel, GLiNExTTextModel
from glinext.tasks.ner.model import NERHead
from glinext.tasks.open_relex.model import OpenRelexHead
from tests.heads.conftest import D, make_config

FIXED_WIDTH_ANCHOR_LAYERS = [
    "position_buckets",
    "topk_norm",
    "topk_distinct",
    "topk_parent",
    "topk_density_distinct",
    "fixed",
    "fixed_rnn",
    "fixed_transformer",
]

OPEN_RELEX_ANCHOR_COMPONENTS = [
    ("parent", {}, 1),
    ("features", {}, 3),
    ("feature", {}, 3),
    ("position_buckets", {"num_slots": 4}, 4),
    ("topk_norm", {"num_slots": 4}, 4),
    ("topk_distinct", {"num_slots": 4}, 4),
    ("topk_parent", {"num_slots": 4}, 4),
    ("topk_density_distinct", {"num_slots": 4}, 4),
    ("fixed", {"num_slots": 4}, 4),
    ("fixed_rnn", {"num_slots": 4}, 4),
    (
        "fixed_transformer",
        {"num_slots": 4, "num_heads": 4, "num_layers": 1},
        4,
    ),
    ("rotary", {"max_count": 4}, 4),
    ("rnn", {"max_count": 4}, 4),
    ("query_rnn", {"max_count": 4}, 3),
    (
        "query_transformer",
        {"max_count": 4, "num_heads": 4, "num_layers": 1},
        3,
    ),
]


def _make_head(**kwargs):
    defaults = {
        "anchor_mode": "fixed_transformer",
        "num_fixed_slots": 2,
        "max_count": 4,
        "anchor_num_heads": 4,
        "anchor_num_layers": 1,
    }
    defaults.update(kwargs)
    config = make_config(
        default_ner_config=False,
        open_relex_config=defaults,
    )
    return OpenRelexHead.from_config(config)


class _KnownSpanRepresentations(nn.Module):
    def __init__(self, representations):
        super().__init__()
        self.register_buffer("representations", representations)

    def forward(self, features, span_idx):
        return self.representations.expand(features.shape[0], -1, -1)


class _EntityMemoryAnchors(nn.Module):
    num_slots = 2

    def forward(
        self,
        context_embedding,
        feature_embeddings=None,
        feature_mask=None,
        **kwargs,
    ):
        del context_embedding, kwargs
        # Relation memory is [parent, entity_0, entity_1, ...].
        return feature_embeddings[:, 1:3], feature_mask[:, 1:3]


class TestOpenRelexConfiguration:
    def test_anchor_coverage_matrix_includes_every_registered_mode(self):
        covered_modes = {
            anchor_type
            for anchor_type, _, _ in OPEN_RELEX_ANCHOR_COMPONENTS
        }

        assert covered_modes == set(AnchorLayer._registry)

    def test_has_canonical_config_and_head(self):
        config = make_config(
            default_ner_config=False,
            open_relex_config={"num_fixed_slots": 2},
        )

        assert isinstance(config.open_relex_config, OpenRelexHeadConfig)
        head = OpenRelexHead.from_config(config)
        assert isinstance(head, OpenRelexHead)
        assert isinstance(head, NERHead)

    @pytest.mark.parametrize(
        "anchor_type,params,expected_mode",
        [
            (anchor_type, params, "rotary" if anchor_type == "rnn" else anchor_type)
            for anchor_type, params, _ in OPEN_RELEX_ANCHOR_COMPONENTS
        ],
    )
    def test_accepts_every_registered_anchor_component(
        self,
        anchor_type,
        params,
        expected_mode,
    ):
        config = OpenRelexHeadConfig(
            anchor_layer={"type": anchor_type, "params": params},
        )

        assert config.effective_anchor_mode() == expected_mode

    @pytest.mark.parametrize("anchor_type", FIXED_WIDTH_ANCHOR_LAYERS)
    def test_accepts_every_fixed_width_anchor_component(self, anchor_type):
        config = OpenRelexHeadConfig(
            anchor_layer={
                "type": anchor_type,
                "params": {"num_slots": 4},
            },
        )

        assert config.effective_anchor_mode() == anchor_type
        assert config.effective_anchor_num_slots() == 4

    def test_position_bucket_component_supports_large_slot_capacity(self):
        config = OpenRelexHeadConfig(
            anchor_layer={
                "type": "position_buckets",
                "params": {"num_slots": 100},
            },
        )
        model_config = make_config(
            default_ner_config=False,
            open_relex_config=config,
        )
        head = OpenRelexHead.from_config(model_config)

        assert head.relation_anchor_layer.num_slots == 100
        assert head._relation_uses_fixed_slots

    def test_config_round_trip(self):
        config = make_config(
            default_ner_config=False,
            open_relex_config={"num_fixed_slots": 2},
        )
        reloaded = GLiNextConfig(**config.to_dict())

        assert reloaded.open_relex_config.head_type == "open_relex"
        assert reloaded.open_relex_config.num_fixed_slots == 2

    def test_model_registers_canonical_head(self, monkeypatch):
        config = GLiNextConfig(
            model_name="unused",
            hidden_size=D,
            vocab_size=32,
            encoder_config={"hidden_size": D, "model_type": "deberta-v2"},
            default_ner_config=False,
            open_relex_config={
                "num_fixed_slots": 2,
                "anchor_num_heads": 4,
                "anchor_num_layers": 1,
            },
        )
        monkeypatch.setattr(
            BaseGLiNextModel,
            "_init_token_rep_layer",
            lambda *args, **kwargs: nn.Identity(),
        )

        model = GLiNExTTextModel(config)

        assert "open_relex" in model.heads
        assert isinstance(model.heads["open_relex"], OpenRelexHead)


class TestOpenRelexEntityFirstFlow:
    @pytest.mark.parametrize(
        "anchor_type,params,expected_anchor_count",
        OPEN_RELEX_ANCHOR_COMPONENTS,
    )
    def test_all_anchor_layers_run_training_forward(
        self,
        anchor_type,
        params,
        expected_anchor_count,
        shared,
        flat_inputs,
    ):
        head = _make_head(
            anchor_layer={
                "type": anchor_type,
                "params": params,
            },
            anchor_objectness=True,
        )
        head.eval()
        batch_size = flat_inputs.words_embedding.shape[0]
        relation_count = flat_inputs.child_embedding.shape[1]
        spans = torch.tensor(
            [[[0, 0], [2, 2]], [[1, 1], [3, 3]]],
            dtype=torch.long,
        )
        span_mask = torch.ones(batch_size, 2, dtype=torch.bool)
        relation_labels = torch.zeros(batch_size, 1, relation_count)
        relation_labels[:, 0, 0] = 1.0
        assignment_labels = torch.zeros(
            batch_size,
            1,
            2,
            2,
        )
        assignment_labels[:, 0, 0, 0] = 1.0
        assignment_labels[:, 0, 1, 1] = 1.0

        output = head(
            shared,
            {},
            flat_inputs=flat_inputs,
            base_loss_fn=lambda logits, targets: (
                F.binary_cross_entropy_with_logits(
                    logits,
                    targets,
                    reduction="none",
                )
            ),
            open_rel_span_idx=spans,
            open_rel_span_mask=span_mask,
            open_rel_labels=relation_labels,
            open_rel_assignment_labels=assignment_labels,
            open_rel_count=torch.ones(batch_size, dtype=torch.long),
        )

        assert output.logits.shape == (
            batch_size,
            expected_anchor_count,
            relation_count,
        )
        assert output.extra["assignment_logits"].shape == (
            batch_size,
            expected_anchor_count,
            2,
            2,
        )
        assert output.extra["anchor_mask"].any(dim=1).all()
        assert all(len(matches) == 1 for matches in output.extra["anchor_matches"])
        assert output.loss.isfinite()

    def test_public_and_endpoint_tensor_contracts(self, shared, flat_inputs):
        head = _make_head()
        span_idx = torch.tensor(
            [[[0, 0], [2, 3]], [[1, 1], [4, 4]]],
            dtype=torch.long,
        )
        span_mask = torch.ones(span_idx.shape[:2], dtype=torch.bool)

        output = head(
            shared,
            {},
            flat_inputs=flat_inputs,
            open_rel_span_idx=span_idx,
            open_rel_span_mask=span_mask,
        )

        batch_size = flat_inputs.words_embedding.shape[0]
        relation_count = flat_inputs.child_embedding.shape[1]
        assert output.logits.shape == (batch_size, 2, relation_count)
        assert output.extra["assignment_logits"].shape == (
            batch_size,
            2,
            2,
            2,
        )
        assert output.extra["entity_logits"].shape == (
            batch_size,
            flat_inputs.words_embedding.shape[1],
            relation_count,
            3,
        )
        assert torch.equal(output.extra["span_idx"], span_idx)

    def test_inference_extracts_spans_from_entity_logits(
        self,
        shared,
        flat_inputs,
        monkeypatch,
    ):
        head = _make_head()
        head.eval()
        extracted_spans = torch.tensor(
            [[[0, 0], [2, 2]], [[1, 1], [3, 3]]],
            dtype=torch.long,
        )
        extracted_mask = torch.ones(2, 2, dtype=torch.bool)
        captured = {}

        def fake_extract(scores, labels=None, threshold=0.5):
            captured["scores"] = scores
            captured["labels"] = labels
            captured["threshold"] = threshold
            return extracted_spans, extracted_mask

        monkeypatch.setattr(
            "glinext.tasks.open_relex.model.extract_spans_from_tokens",
            fake_extract,
        )

        output = head(
            shared,
            {},
            flat_inputs=flat_inputs,
            threshold=0.4,
        )

        assert captured["scores"] is output.extra["entity_logits"]
        assert captured["labels"] is None
        assert captured["threshold"] == 0.4
        assert torch.equal(output.extra["span_idx"], extracted_spans)
        assert torch.equal(output.extra["span_mask"], extracted_mask)

    def test_relation_queries_use_pooled_entities(self, shared, flat_inputs):
        head = _make_head()
        head.eval()
        head.relation_anchor_layer = _EntityMemoryAnchors()
        entities = torch.zeros(1, 2, D)
        entities[0, 0, 0] = 5.0
        entities[0, 1, 1] = 7.0
        head.entity_span_rep_layer = _KnownSpanRepresentations(entities)
        spans = torch.tensor(
            [[[0, 0], [2, 2]], [[0, 0], [2, 2]]],
            dtype=torch.long,
        )
        mask = torch.ones(spans.shape[:2], dtype=torch.bool)

        first = head(
            shared,
            {},
            flat_inputs=flat_inputs,
            open_rel_span_idx=spans,
            open_rel_span_mask=mask,
        )
        entity_logits = first.extra["entity_logits"].clone()
        changed_entities = entities.clone()
        changed_entities[0, 0, 0] = -11.0
        changed_entities[0, 1, 1] = 13.0
        head.entity_span_rep_layer = _KnownSpanRepresentations(
            changed_entities
        )
        second = head(
            shared,
            {},
            flat_inputs=flat_inputs,
            open_rel_span_idx=spans,
            open_rel_span_mask=mask,
        )

        assert torch.equal(second.extra["entity_logits"], entity_logits)
        assert torch.equal(first.extra["anchors"], entities.expand(2, -1, -1))
        assert torch.equal(
            second.extra["anchors"],
            changed_entities.expand(2, -1, -1),
        )

    def test_duplicate_inference_spans_are_coalesced(self):
        spans = torch.tensor([[[1, 1], [1, 1], [3, 4]]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        unique, unique_mask, labels = OpenRelexHead._coalesce_spans(
            spans,
            mask,
        )

        assert labels is None
        assert unique.tolist() == [[[1, 1], [3, 4]]]
        assert unique_mask.tolist() == [[True, True]]


class TestOpenRelexMatching:
    @staticmethod
    def _loss_fn(predictions, labels):
        return F.binary_cross_entropy_with_logits(
            predictions,
            labels,
            reduction="none",
        )

    def test_hungarian_loss_is_gold_order_invariant(self):
        head = _make_head()
        relation_logits = torch.tensor([[[8.0], [-8.0]]])
        assignment_logits = torch.tensor(
            [[[[8.0, -8.0], [-8.0, 8.0]],
              [[-8.0, 8.0], [8.0, -8.0]]]]
        )
        labels = torch.ones(1, 2, 1)
        assignments = torch.tensor(
            [[[[1.0, 0.0], [0.0, 1.0]],
              [[0.0, 1.0], [1.0, 0.0]]]]
        )
        anchor_mask = torch.ones(1, 2, dtype=torch.bool)
        relation_mask = torch.ones(1, 1, dtype=torch.bool)
        entity_mask = torch.ones(1, 2, dtype=torch.bool)

        loss, matches = head._assignment_loss(
            relation_logits,
            assignment_logits,
            labels,
            assignments,
            anchor_mask,
            relation_mask,
            entity_mask,
            torch.tensor([2]),
            self._loss_fn,
        )
        reversed_loss, reversed_matches = head._assignment_loss(
            relation_logits,
            assignment_logits,
            labels.flip(1),
            assignments.flip(1),
            anchor_mask,
            relation_mask,
            entity_mask,
            torch.tensor([2]),
            self._loss_fn,
        )

        assert torch.allclose(loss, reversed_loss)
        assert {predicted for predicted, _ in matches[0]} == {0, 1}
        assert {predicted for predicted, _ in reversed_matches[0]} == {0, 1}

    def test_gold_overflow_warns_once_and_matches_available_slots(self):
        head = _make_head()
        relation_logits = torch.zeros(1, 2, 1)
        assignment_logits = torch.zeros(1, 2, 3, 2)
        labels = torch.ones(1, 3, 1)
        assignments = torch.zeros(1, 3, 3, 2)
        for gold_idx in range(3):
            assignments[0, gold_idx, gold_idx, :] = 1.0
        args = (
            relation_logits,
            assignment_logits,
            labels,
            assignments,
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones(1, 1, dtype=torch.bool),
            torch.ones(1, 3, dtype=torch.bool),
            torch.tensor([3]),
            self._loss_fn,
        )

        with pytest.warns(
            UserWarning,
            match=r"3 gold pairs, 2 slots.*unmatched gold pairs are ignored",
        ):
            loss, matches = head._assignment_loss(*args)

        assert loss.isfinite()
        assert len(matches[0]) == 2
        assert {predicted for predicted, _ in matches[0]} == {0, 1}

        # Repeated oversized batches should not flood long training runs.
        with warnings.catch_warnings(record=True) as repeated_warnings:
            warnings.simplefilter("always")
            repeated_loss, repeated_matches = head._assignment_loss(*args)
        assert not repeated_warnings
        assert repeated_loss.isfinite()
        assert len(repeated_matches[0]) == 2

    def test_padded_gold_capacity_does_not_warn_without_active_overflow(self):
        head = _make_head()
        relation_logits = torch.zeros(1, 2, 1)
        assignment_logits = torch.zeros(1, 2, 3, 2)
        labels = torch.ones(1, 3, 1)
        assignments = torch.zeros(1, 3, 3, 2)
        assignments[0, 0, 0, :] = 1.0
        assignments[0, 1, 1, :] = 1.0

        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            loss, matches = head._assignment_loss(
                relation_logits,
                assignment_logits,
                labels,
                assignments,
                torch.ones(1, 2, dtype=torch.bool),
                torch.ones(1, 1, dtype=torch.bool),
                torch.ones(1, 3, dtype=torch.bool),
                torch.tensor([2]),
                self._loss_fn,
            )

        assert not recorded_warnings
        assert loss.isfinite()
        assert len(matches[0]) == 2

    def test_inactive_fixed_width_slots_are_not_counted_as_capacity(self):
        head = _make_head()
        relation_logits = torch.zeros(1, 4, 1)
        assignment_logits = torch.zeros(1, 4, 3, 2)
        labels = torch.ones(1, 3, 1)
        assignments = torch.zeros(1, 3, 3, 2)
        for gold_idx in range(3):
            assignments[0, gold_idx, gold_idx, :] = 1.0

        with pytest.warns(
            UserWarning,
            match=r"3 gold pairs, 2 slots active out of 4",
        ):
            loss, matches = head._assignment_loss(
                relation_logits,
                assignment_logits,
                labels,
                assignments,
                torch.tensor([[True, True, False, False]]),
                torch.ones(1, 1, dtype=torch.bool),
                torch.ones(1, 3, dtype=torch.bool),
                torch.tensor([3]),
                self._loss_fn,
            )

        assert loss.isfinite()
        assert len(matches[0]) == 2

    def test_objectness_targets_follow_matches(self):
        logits = torch.zeros(1, 3)
        loss = OpenRelexHead._objectness_loss(
            logits,
            [[(2, 0)]],
            torch.ones(1, 3, dtype=torch.bool),
            base_loss_fn=self._loss_fn,
        )
        expected_targets = torch.tensor([[0.0, 0.0, 1.0]])
        expected = self._loss_fn(logits, expected_targets).mean()

        assert torch.allclose(loss, expected)

    def test_mean_reduction_normalizes_entity_bio_elements(self):
        head = _make_head(bio_loss_reduction="mean")
        reduced = head._reduce_entity_loss(
            torch.tensor(15.0),
            torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
            torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
        )

        # (2 words * 2 relations + 1 word * 1 relation) * 3 BIO roles.
        assert torch.equal(reduced, torch.tensor(1.0))

    def test_zero_gold_groups_supervise_unused_slots(
        self,
        shared,
        flat_inputs,
    ):
        head = _make_head(anchor_objectness=True)
        batch_size = flat_inputs.words_embedding.shape[0]
        sequence_length = flat_inputs.words_embedding.shape[1]
        relation_count = flat_inputs.child_embedding.shape[1]
        output = head(
            shared,
            {},
            flat_inputs=flat_inputs,
            base_loss_fn=self._loss_fn,
            open_rel_entity_labels=torch.zeros(
                batch_size,
                sequence_length,
                relation_count,
                3,
            ),
            open_rel_span_idx=torch.zeros(
                batch_size,
                1,
                2,
                dtype=torch.long,
            ),
            open_rel_span_mask=torch.zeros(
                batch_size,
                1,
                dtype=torch.bool,
            ),
            open_rel_labels=torch.zeros(
                batch_size,
                1,
                relation_count,
            ),
            open_rel_assignment_labels=torch.zeros(
                batch_size,
                1,
                1,
                2,
            ),
            open_rel_count=torch.zeros(batch_size, dtype=torch.long),
        )

        assert output.extra["anchor_matches"] == [
            [] for _ in range(batch_size)
        ]
        assert output.extra["assignment_loss"].isfinite()
        assert output.extra["assignment_loss"] > 0
        assert output.extra["objectness_loss"].isfinite()
        assert output.extra["objectness_loss"] > 0
        assert output.loss.isfinite()

    def test_combined_loss_backpropagates(self, shared, flat_inputs):
        from gliner.modeling.loss_functions import focal_loss_with_logits

        head = _make_head(anchor_objectness=True)
        flat_inputs.words_embedding = (
            flat_inputs.words_embedding.detach().requires_grad_(True)
        )
        batch_size = flat_inputs.words_embedding.shape[0]
        sequence_length = flat_inputs.words_embedding.shape[1]
        relation_count = flat_inputs.child_embedding.shape[1]
        spans = torch.tensor(
            [[[0, 0], [2, 2]], [[0, 0], [2, 2]]],
            dtype=torch.long,
        )
        span_mask = torch.ones(batch_size, 2, dtype=torch.bool)
        entity_labels = torch.zeros(
            batch_size, sequence_length, relation_count, 3
        )
        entity_labels[:, 0, 0] = 1.0
        relation_labels = torch.zeros(batch_size, 1, relation_count)
        relation_labels[:, 0, 0] = 1.0
        assignment_labels = torch.zeros(
            batch_size, 1, 2, 2
        )
        assignment_labels[:, 0, 0, 0] = 1.0
        assignment_labels[:, 0, 1, 1] = 1.0

        output = head(
            shared,
            {},
            flat_inputs=flat_inputs,
            base_loss_fn=focal_loss_with_logits,
            open_rel_entity_labels=entity_labels,
            open_rel_span_idx=spans,
            open_rel_span_mask=span_mask,
            open_rel_labels=relation_labels,
            open_rel_assignment_labels=assignment_labels,
            open_rel_count=torch.ones(batch_size, dtype=torch.long),
        )
        output.loss.backward()

        assert output.loss.isfinite()
        assert flat_inputs.words_embedding.grad is not None
        assert output.extra["objectness_loss"].isfinite()
