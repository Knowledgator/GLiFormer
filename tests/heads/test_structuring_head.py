"""Tests for the canonical entity-first structuring task."""

from dataclasses import asdict
from unittest import mock

import pytest
import torch
from torch import nn

from gliformer.config import (
    GLiFormerConfig,
    NERHeadConfig,
    StructuringHeadConfig,
)
from gliformer.model import BaseGLiFormerModel, GLiFormerTextModel
from gliformer.tasks import TaskHeadOutput
from gliformer.tasks.ner.model import NERHead
from gliformer.tasks.structuring.model import StructuringHead
from tests.heads.conftest import D, make_config


def _binary_loss_fn(predictions, labels):
    return torch.nn.functional.binary_cross_entropy_with_logits(
        predictions,
        labels.to(predictions.dtype),
        reduction="none",
    )


def _make_head(**kwargs):
    defaults = {
        "anchor_mode": "fixed_transformer",
        "num_fixed_slots": 2,
        "max_count": 4,
        "anchor_num_heads": 4,
        "anchor_num_layers": 1,
        "anchor_objectness": False,
    }
    defaults.update(kwargs)
    config = make_config(
        structuring_config=asdict(
            StructuringHeadConfig(**defaults)
        )
    )
    return StructuringHead.from_config(config)


def _make_reused_head(**kwargs):
    defaults = {
        "reuse_ner_head": True,
        "anchor_mode": "fixed_transformer",
        "num_fixed_slots": 2,
        "max_count": 4,
        "anchor_num_heads": 4,
        "anchor_num_layers": 1,
        "anchor_objectness": False,
    }
    defaults.update(kwargs)
    config = make_config(
        ner_config=asdict(NERHeadConfig()),
        structuring_config=asdict(
            StructuringHeadConfig(**defaults)
        ),
    )
    ner_head = NERHead.from_config(config)
    return (
        StructuringHead.from_config(config, ner_head=ner_head),
        ner_head,
    )


def _make_text_model(monkeypatch, **task_configs):
    config = GLiFormerConfig(
        model_name="unused",
        hidden_size=D,
        vocab_size=32,
        encoder_config={"hidden_size": D, "model_type": "deberta-v2"},
        default_ner_config=False,
        **task_configs,
    )
    monkeypatch.setattr(
        BaseGLiFormerModel,
        "_init_token_rep_layer",
        lambda *args, **kwargs: nn.Identity(),
    )
    return GLiFormerTextModel(config)


class _KnownAnchors(nn.Module):
    num_slots = 2

    def __init__(self, anchors):
        super().__init__()
        self.register_buffer("anchors", anchors)

    def forward(self, context_embedding, feature_embeddings=None, **kwargs):
        anchors = self.anchors.expand(context_embedding.shape[0], -1, -1)
        mask = torch.ones(
            anchors.shape[:2],
            dtype=torch.bool,
            device=anchors.device,
        )
        return anchors, mask


class _KnownNERScorer(nn.Module):
    def forward(self, field_representations, word_embs, word_mask=None):
        batch_size, class_count = field_representations.shape[:2]
        sequence_length = word_embs.shape[1]
        values = torch.arange(
            batch_size * class_count * sequence_length * 3,
            device=word_embs.device,
            dtype=word_embs.dtype,
        )
        logits = values.reshape(
            batch_size,
            class_count,
            sequence_length,
            3,
        )
        if word_mask is not None:
            logits = logits * word_mask[:, None, :, None]
        return logits


class _KnownSpanRepresentations(nn.Module):
    def __init__(self, representations):
        super().__init__()
        self.register_buffer("representations", representations)

    def forward(self, features, span_idx):
        return self.representations.expand(features.shape[0], -1, -1)


class _KnownAnchorRefinement(nn.Module):
    def __init__(self, anchors):
        super().__init__()
        self.register_buffer("anchors", anchors)
        self.input_anchors = None

    def forward(self, anchors, feature_embeddings, **kwargs):
        self.input_anchors = anchors.detach().clone()
        return self.anchors.expand(anchors.shape[0], -1, -1)


class TestStructuringConfiguration:
    def test_has_entity_first_config_and_head(self):
        config = make_config(structuring_config={"num_fixed_slots": 2})

        assert isinstance(
            config.structuring_config,
            StructuringHeadConfig,
        )
        head = StructuringHead.from_config(config)
        assert isinstance(head, StructuringHead)
        assert isinstance(head, NERHead)

    def test_legacy_discriminator_migrates_to_canonical_config(self):
        config = make_config(
            structuring_config={
                "head_type": "set_structuring",
                "num_fixed_slots": 2,
            }
        )

        assert isinstance(
            config.structuring_config,
            StructuringHeadConfig,
        )
        assert config.structuring_config.head_type == "structuring"
        assert isinstance(StructuringHead.from_config(config), StructuringHead)

    def test_from_config_disabled_without_structuring(self):
        config = make_config()
        assert StructuringHead.from_config(config) is None

    def test_config_round_trip_keeps_the_canonical_field(self):
        config = make_config(structuring_config={"num_fixed_slots": 2})
        reloaded = GLiFormerConfig(**config.to_dict())

        assert isinstance(
            reloaded.structuring_config,
            StructuringHeadConfig,
        )
        assert reloaded.structuring_config.head_type == "structuring"

    def test_matcher_defaults_preserve_membership_only_assignment(self):
        config = StructuringHeadConfig()

        assert config.bio_loss_reduction == "mean"
        assert config.matcher_membership_cost == 1.0
        assert config.matcher_dice_cost == 0.0
        assert config.matcher_objectness_cost == 0.0
        assert config.matcher_membership_temperature == 1.0
        assert config.matcher_objectness_temperature == 1.0
        assert config.ner_focal_loss_alpha is None
        assert config.matching_focal_loss_alpha is None
        assert config.objectness_focal_loss_alpha is None

    @pytest.mark.parametrize("component", ["ner", "matching", "objectness"])
    @pytest.mark.parametrize(
        ("suffix", "value"),
        [
            ("alpha", 1.1),
            ("alpha", float("inf")),
            ("gamma", float("nan")),
            ("prob_margin", float("inf")),
        ],
    )
    def test_component_focal_controls_are_validated(
        self,
        component,
        suffix,
        value,
    ):
        name = f"{component}_focal_loss_{suffix}"

        with pytest.raises(ValueError, match=name):
            StructuringHeadConfig(**{name: value})

    def test_model_registers_the_structuring_head_when_requested(
        self,
        monkeypatch,
    ):
        config = GLiFormerConfig(
            model_name="unused",
            hidden_size=D,
            vocab_size=32,
            encoder_config={"hidden_size": D, "model_type": "deberta-v2"},
            default_ner_config=False,
            structuring_config={
                "num_fixed_slots": 2,
                "anchor_num_heads": 4,
                "anchor_num_layers": 1,
            },
        )
        monkeypatch.setattr(
            BaseGLiFormerModel,
            "_init_token_rep_layer",
            lambda *args, **kwargs: nn.Identity(),
        )

        model = GLiFormerTextModel(config)

        assert list(model.heads) == ["structuring"]

    def test_flat_head_adds_no_hierarchy_state_keys(self):
        head = _make_head(multi_level=False)

        assert not hasattr(head, "anchor_relations_rep_layer")
        assert not any(
            key.startswith("anchor_relations_rep_layer.")
            for key in head.state_dict()
        )

    def test_reuse_ner_head_round_trips(self):
        config = make_config(
            ner_config=asdict(NERHeadConfig()),
            structuring_config={
                "reuse_ner_head": True,
                "num_fixed_slots": 2,
            },
        )

        reloaded = GLiFormerConfig(**config.to_dict())

        assert config.structuring_config.reuse_ner_head is True
        assert reloaded.structuring_config.reuse_ner_head is True

    @pytest.mark.parametrize("value", [1, "true", None])
    def test_reuse_ner_head_requires_a_boolean(self, value):
        with pytest.raises(TypeError, match="reuse_ner_head"):
            StructuringHeadConfig(reuse_ner_head=value)

    @pytest.mark.parametrize(
        "name",
        [
            "matcher_membership_cost",
            "matcher_dice_cost",
            "matcher_objectness_cost",
        ],
    )
    @pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
    def test_matcher_cost_weights_must_be_finite_and_non_negative(
        self,
        name,
        value,
    ):
        with pytest.raises(ValueError, match=name):
            StructuringHeadConfig(**{name: value})

    @pytest.mark.parametrize(
        "name",
        [
            "matcher_membership_temperature",
            "matcher_objectness_temperature",
        ],
    )
    @pytest.mark.parametrize(
        "value",
        [0.0, -1.0, float("inf"), float("nan")],
    )
    def test_matcher_temperatures_must_be_finite_and_positive(
        self,
        name,
        value,
    ):
        with pytest.raises(ValueError, match=name):
            StructuringHeadConfig(**{name: value})

    def test_reuse_ner_head_requires_enabled_compatible_ner(self):
        with pytest.raises(ValueError, match="requires an enabled ner_config"):
            GLiFormerConfig(
                default_ner_config=False,
                structuring_config={"reuse_ner_head": True},
            )

        with pytest.raises(ValueError, match="anchor_mode='parent'"):
            GLiFormerConfig(
                ner_config={
                    "anchor_layer": {
                        "type": "fixed",
                        "params": {"num_slots": 1},
                    },
                },
                structuring_config={"reuse_ner_head": True},
            )

    def test_model_reuses_ner_without_duplicate_registration(
        self,
        monkeypatch,
    ):
        model = _make_text_model(
            monkeypatch,
            ner_config=asdict(NERHeadConfig()),
            structuring_config={
                "reuse_ner_head": True,
                "anchor_layer": {
                    "type": "fixed",
                    "params": {"num_slots": 2},
                },
                "anchor_refinement": "none",
            },
        )
        ner_head = model.heads["ner"]
        structuring_head = model.heads["structuring"]
        state_keys = tuple(model.state_dict())

        assert structuring_head._owns_ner_head is False
        assert structuring_head.__dict__["_reused_ner_head"] is ner_head
        assert "_reused_ner_head" not in structuring_head._modules
        assert not hasattr(structuring_head, "scorer")
        assert any(key.startswith("heads.ner.scorer.") for key in state_keys)
        assert not any(
            key.startswith("heads.structuring.scorer.")
            for key in state_keys
        )
        ner_parameter_ids = {id(parameter) for parameter in ner_head.parameters()}
        assert ner_parameter_ids.isdisjoint(
            id(parameter) for parameter in structuring_head.parameters()
        )

    def test_private_mode_keeps_its_own_ner_when_standalone_ner_exists(
        self,
        monkeypatch,
    ):
        model = _make_text_model(
            monkeypatch,
            ner_config=asdict(NERHeadConfig()),
            structuring_config={
                "reuse_ner_head": False,
                "anchor_layer": {
                    "type": "fixed",
                    "params": {"num_slots": 2},
                },
                "anchor_refinement": "none",
            },
        )
        structuring_head = model.heads["structuring"]

        assert structuring_head._owns_ner_head is True
        assert hasattr(structuring_head, "scorer")
        assert any(
            key.startswith("heads.structuring.scorer.")
            for key in model.state_dict()
        )


class TestStructuringCheckpointCompatibility:
    @staticmethod
    def _structuring_config():
        return {
            "anchor_layer": {
                "type": "fixed",
                "params": {"num_slots": 2},
            },
            "anchor_refinement": "none",
            "anchor_objectness": False,
        }

    def test_legacy_head_prefix_loads_strictly(self, monkeypatch):
        model = _make_text_model(
            monkeypatch,
            structuring_config=self._structuring_config(),
        )
        current = model.state_dict()
        target_prefix = "heads.structuring."
        legacy_prefix = "heads.set_structuring."
        legacy = {}
        migrated_keys = []
        for key, value in current.items():
            if key.startswith(target_prefix):
                suffix = key[len(target_prefix):]
                legacy[f"{legacy_prefix}{suffix}"] = value.clone()
                migrated_keys.append(key)
            else:
                legacy[key] = value.clone()

        incompatible = model.load_state_dict(legacy, strict=True)

        assert migrated_keys
        assert incompatible.missing_keys == []
        assert incompatible.unexpected_keys == []
        loaded = model.state_dict()
        for key in migrated_keys:
            torch.testing.assert_close(loaded[key], current[key])

    def test_legacy_prefix_keeps_incompatible_suffix_unexpected(
        self,
        monkeypatch,
    ):
        model = _make_text_model(
            monkeypatch,
            structuring_config=self._structuring_config(),
        )
        state = {key: value.clone() for key, value in model.state_dict().items()}
        suffix = "scorer.out_mlp.3.bias"
        legacy_key = f"heads.set_structuring.{suffix}"
        state[legacy_key] = torch.zeros(4)

        incompatible = model.load_state_dict(state, strict=False)

        assert legacy_key in incompatible.unexpected_keys

    def test_reuse_model_strictly_loads_private_entity_stage(
        self,
        monkeypatch,
    ):
        private_model = _make_text_model(
            monkeypatch,
            ner_config=asdict(NERHeadConfig()),
            structuring_config={
                **self._structuring_config(),
                "reuse_ner_head": False,
            },
        )
        shared_model = _make_text_model(
            monkeypatch,
            ner_config=asdict(NERHeadConfig()),
            structuring_config={
                **self._structuring_config(),
                "reuse_ner_head": True,
            },
        )
        private_state = private_model.state_dict()

        assert any(
            key.startswith("heads.structuring.scorer.")
            for key in private_state
        )
        assert not any(
            key.startswith("heads.structuring.scorer.")
            for key in shared_model.state_dict()
        )

        incompatible = shared_model.load_state_dict(
            private_state,
            strict=True,
        )

        assert incompatible.missing_keys == []
        assert incompatible.unexpected_keys == []


class TestStructuringEntityFirstFlow:
    def test_reused_ner_runs_on_structuring_field_inputs(
        self,
        shared,
        flat_inputs,
        monkeypatch,
    ):
        head, ner_head = _make_reused_head(
            anchor_layer={"type": "fixed", "params": {"num_slots": 2}},
            anchor_refinement="none",
        )
        entity_logits = torch.randn(
            flat_inputs.words_embedding.shape[0],
            flat_inputs.words_embedding.shape[1],
            flat_inputs.child_embedding.shape[1],
            3,
        )
        calls = []

        def record_ner_call(*args, **kwargs):
            calls.append((args, kwargs))
            return TaskHeadOutput(
                logits=entity_logits,
                extra={
                    "words_embedding": kwargs["flat_inputs"].words_embedding,
                    "mask": kwargs["flat_inputs"].mask,
                },
            )

        monkeypatch.setattr(ner_head, "forward", record_ner_call)
        spans = torch.zeros(
            flat_inputs.words_embedding.shape[0],
            1,
            2,
            dtype=torch.long,
        )
        span_mask = torch.ones(spans.shape[:2], dtype=torch.bool)
        incompatible_cached_output = TaskHeadOutput(
            logits=torch.randn(1, 1, 1, 3),
        )

        output = head(
            shared,
            {"ner": incompatible_cached_output},
            flat_inputs=flat_inputs,
            structuring_span_idx=spans,
            structuring_span_mask=span_mask,
        )

        assert len(calls) == 1
        assert calls[0][1]["flat_inputs"] is flat_inputs
        assert calls[0][1]["ner_labels"] is None
        assert output.logits is entity_logits
        assert output.logits is not incompatible_cached_output.logits

    def test_reused_ner_entity_loss_is_included_once(
        self,
        shared,
        flat_inputs,
        monkeypatch,
    ):
        head, ner_head = _make_reused_head(
            entity_loss_coef=2.5,
            anchor_layer={"type": "fixed", "params": {"num_slots": 2}},
            anchor_refinement="none",
        )
        entity_loss = torch.tensor(7.0)
        entity_logits = torch.zeros(
            flat_inputs.words_embedding.shape[0],
            flat_inputs.words_embedding.shape[1],
            flat_inputs.child_embedding.shape[1],
            3,
        )

        def entity_forward(*args, **kwargs):
            return TaskHeadOutput(
                loss=entity_loss,
                logits=entity_logits,
                extra={
                    "words_embedding": kwargs["flat_inputs"].words_embedding,
                    "mask": kwargs["flat_inputs"].mask,
                },
            )

        monkeypatch.setattr(ner_head, "forward", entity_forward)
        labels = torch.zeros(
            flat_inputs.words_embedding.shape[0],
            1,
            flat_inputs.words_embedding.shape[1],
            flat_inputs.child_embedding.shape[1],
            3,
        )

        output = head(
            shared,
            {},
            flat_inputs=flat_inputs,
            structuring_labels=labels,
        )

        assert output.loss.item() == pytest.approx(17.5)
        assert output.extra["entity_loss"].item() == pytest.approx(7.0)

    def test_structuring_entity_gradients_reach_reused_ner(
        self,
        shared,
        flat_inputs,
    ):
        from gliner.modeling.loss_functions import focal_loss_with_logits

        head, ner_head = _make_reused_head(
            anchor_layer={"type": "fixed", "params": {"num_slots": 2}},
            anchor_refinement="none",
        )
        batch_size = flat_inputs.words_embedding.shape[0]
        token_count = flat_inputs.words_embedding.shape[1]
        field_count = flat_inputs.child_embedding.shape[1]
        labels = torch.zeros(batch_size, 1, token_count, field_count, 3)
        labels[:, 0, 0, 0] = 1.0
        spans = torch.zeros(batch_size, 1, 2, dtype=torch.long)
        span_mask = torch.ones(batch_size, 1, dtype=torch.bool)
        span_labels = torch.zeros(batch_size, 1, 1, field_count)
        span_labels[:, 0, 0, 0] = 1.0

        output = head(
            shared,
            {},
            flat_inputs=flat_inputs,
            structuring_labels=labels,
            structuring_count=torch.ones(batch_size, dtype=torch.long),
            structuring_span_idx=spans,
            structuring_span_mask=span_mask,
            structuring_span_labels=span_labels,
            base_loss_fn=focal_loss_with_logits,
        )
        output.loss.backward()

        assert output.loss.isfinite()
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in ner_head.scorer.parameters()
        )
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in head.record_anchor_layer.parameters()
        )

    def test_teacher_forced_duplicate_boundaries_merge_membership_targets(self):
        spans = torch.tensor([[[1, 1], [1, 1], [3, 3]]])
        span_mask = torch.ones(1, 3, dtype=torch.bool)
        labels = torch.zeros(1, 3, 2, 2)
        labels[0, 0, 0, 0] = 1.0
        labels[0, 1, 1, 1] = 1.0
        labels[0, 2, 0, 1] = 1.0

        unique_spans, unique_mask, unique_labels = (
            StructuringHead._coalesce_spans(
                spans,
                span_mask,
                labels,
            )
        )

        assert unique_spans.tolist() == [[[1, 1], [3, 3]]]
        assert unique_mask.tolist() == [[True, True]]
        assert unique_labels[0, 0, 0, 0] == 1.0
        assert unique_labels[0, 0, 1, 1] == 1.0
        assert unique_labels[0, 1, 0, 1] == 1.0

    def test_extracts_entities_before_record_scoring(
        self,
        shared,
        flat_inputs,
        monkeypatch,
    ):
        head = _make_head()
        head.scorer = _KnownNERScorer()
        known_anchors = torch.zeros(1, 2, D)
        known_anchors[0, 0, 0] = 2.0
        known_anchors[0, 1, 1] = 3.0
        head.record_anchor_layer = _KnownAnchors(known_anchors)
        known_entities = torch.zeros(1, 2, D)
        known_entities[0, 0, 0] = 5.0
        known_entities[0, 1, 1] = 7.0
        head.entity_span_rep_layer = _KnownSpanRepresentations(known_entities)
        spans = torch.tensor(
            [[[0, 1], [3, 4]], [[0, 1], [3, 4]]],
            dtype=torch.long,
        )
        span_mask = torch.ones(spans.shape[:2], dtype=torch.bool)
        extraction_calls = []

        def extract(scores, labels=None, threshold=0.5):
            extraction_calls.append(scores.clone())
            return spans, span_mask

        monkeypatch.setattr(
            "gliformer.tasks.structuring.model.extract_spans_from_tokens",
            extract,
        )

        output = head(shared, {}, flat_inputs=flat_inputs)

        assert len(extraction_calls) == 1
        assert torch.equal(extraction_calls[0], output.logits)
        assert torch.equal(output.extra["entity_spans"], spans)
        assert output.extra["entity_representations"].shape == (2, 2, D)
        assert output.extra["entity_anchor_logits"].shape == (2, 2, 2)
        assert output.extra["entity_field_logits"].shape == (2, 2, 3)
        assert output.extra["structuring_logits"].shape == (2, 2, 2)
        assert output.extra["membership_logits"] is output.extra[
            "structuring_logits"
        ]

        first_entity_logits = output.logits.clone()
        first_field_logits = output.extra["entity_field_logits"].clone()
        head.record_anchor_layer = _KnownAnchors(known_anchors.flip(1))
        changed = head(shared, {}, flat_inputs=flat_inputs)
        assert torch.equal(changed.logits, first_entity_logits)
        assert torch.equal(
            changed.extra["entity_field_logits"],
            first_field_logits,
        )
        assert not torch.equal(
            changed.extra["entity_anchor_logits"],
            output.extra["entity_anchor_logits"],
        )

    def test_second_stage_scores_only_anchor_membership(self):
        entities = torch.zeros(1, 2, D)
        entities[0, 0, 0] = 5.0
        entities[0, 1, 1] = 7.0
        anchors = torch.zeros(1, 2, D)
        anchors[0, 0, 0] = 2.0
        anchors[0, 1, 1] = 3.0
        entity_mask = torch.ones(1, 2, dtype=torch.bool)
        anchor_mask = torch.ones(1, 2, dtype=torch.bool)

        logits = StructuringHead._score_anchor_membership(
            entities,
            entity_mask,
            anchors,
            anchor_mask,
        )

        expected = torch.einsum("BED,BAD->BAE", entities, anchors)
        assert torch.equal(logits, expected)

    def test_membership_uses_post_refinement_anchor_representations(
        self,
        shared,
        flat_inputs,
    ):
        head = _make_head()
        raw_anchors = torch.zeros(1, 2, D)
        raw_anchors[0, 0, 2] = 11.0
        raw_anchors[0, 1, 3] = 13.0
        refined_anchors = torch.zeros(1, 2, D)
        refined_anchors[0, 0, 0] = 2.0
        refined_anchors[0, 1, 1] = 3.0
        entities = torch.zeros(1, 2, D)
        entities[0, 0, 0] = 5.0
        entities[0, 1, 1] = 7.0

        head.record_anchor_layer = _KnownAnchors(raw_anchors)
        refinement = _KnownAnchorRefinement(refined_anchors)
        head.record_anchor_refine = refinement
        head.entity_span_rep_layer = _KnownSpanRepresentations(entities)
        spans = torch.tensor(
            [[[0, 0], [1, 1]], [[0, 0], [1, 1]]],
            dtype=torch.long,
        )
        span_mask = torch.ones(spans.shape[:2], dtype=torch.bool)

        output = head(
            shared,
            {},
            flat_inputs=flat_inputs,
            structuring_span_idx=spans,
            structuring_span_mask=span_mask,
        )

        expected_anchors = refined_anchors.expand(2, -1, -1)
        expected_entities = entities.expand(2, -1, -1)
        expected_membership = torch.einsum(
            "BED,BAD->BAE",
            expected_entities,
            expected_anchors,
        )
        assert torch.equal(
            refinement.input_anchors,
            raw_anchors.expand(2, -1, -1),
        )
        assert torch.equal(output.extra["groups_output"], expected_anchors)
        assert torch.equal(
            output.extra["membership_logits"],
            expected_membership,
        )

    def test_selected_span_field_logits_come_from_classical_ner(self):
        ner_logits = torch.full((1, 4, 2, 3), -10.0)
        ner_logits[0, 1, 0, 0] = 4.0
        ner_logits[0, 2, 0, 1] = 3.0
        ner_logits[0, 1:3, 0, 2] = 2.0
        ner_logits.requires_grad_(True)
        spans = torch.tensor([[[1, 2]]])
        span_mask = torch.ones(1, 1, dtype=torch.bool)

        field_logits = StructuringHead._span_field_logits(
            ner_logits,
            spans,
            span_mask,
        )

        assert field_logits.shape == (1, 1, 2)
        # Decoder-only field scores must not retain one CopySlices autograd
        # node per entity; a sufficiently long chain overflows the C stack
        # when a training output is released.
        assert field_logits.requires_grad is False
        assert field_logits[0, 0, 0].item() == pytest.approx(2.0)
        assert field_logits[0, 0, 1].item() == pytest.approx(-10.0)

    def test_supervised_empty_spans_do_not_use_prediction_fallback(
        self,
        shared,
        flat_inputs,
        monkeypatch,
    ):
        head = _make_head()

        def unexpected_extraction(*args, **kwargs):
            raise AssertionError("prediction span extraction was called")

        monkeypatch.setattr(
            "gliformer.tasks.structuring.model.extract_spans_from_tokens",
            unexpected_extraction,
        )
        batch_size, word_count = flat_inputs.words_embedding.shape[:2]
        field_count = flat_inputs.child_embedding.shape[1]
        labels = torch.zeros(
            batch_size,
            1,
            word_count,
            field_count,
            3,
        )

        output = head(
            shared,
            {},
            flat_inputs=flat_inputs,
            structuring_labels=labels,
        )

        assert output.extra["entity_spans"].shape == (batch_size, 1, 2)
        assert not output.extra["entity_mask"].any()

    def test_invalid_and_empty_source_spans_are_safe(self):
        spans = torch.tensor([[[0, 0], [4, 5], [-1, 0]]])
        span_mask = torch.ones(1, 3, dtype=torch.bool)

        safe_spans, safe_mask = StructuringHead._sanitize_spans(
            spans,
            span_mask,
            sequence_length=0,
        )

        assert safe_spans.tolist() == [[[0, 0], [0, 0], [0, 0]]]
        assert not safe_mask.any()

        head = _make_head()
        pooled = head._pool_entity_spans(
            torch.empty(1, 0, D),
            safe_spans,
            safe_mask,
        )
        assert pooled.shape == (1, 3, D)
        assert not pooled.any()

    def test_span_at_last_valid_word_is_not_suppressed_by_padding(self):
        ner_logits = torch.zeros(1, 3, 1, 3)
        ner_logits[0, 1, 0] = 4.0
        spans = torch.tensor([[[1, 1]]])
        span_mask = torch.ones(1, 1, dtype=torch.bool)
        word_mask = torch.tensor([[True, True, False]])

        field_logits = StructuringHead._span_field_logits(
            ner_logits,
            spans,
            span_mask,
            word_mask=word_mask,
        )

        assert field_logits[0, 0, 0].item() == pytest.approx(4.0)

    def test_public_output_keeps_both_stages_separate(
        self,
        shared,
        flat_inputs,
    ):
        head = _make_head()
        spans = torch.tensor(
            [[[0, 1], [3, 4]], [[0, 1], [3, 4]]],
            dtype=torch.long,
        )
        span_mask = torch.ones(spans.shape[:2], dtype=torch.bool)
        output = head(
            shared,
            {},
            flat_inputs=flat_inputs,
            structuring_span_idx=spans,
            structuring_span_mask=span_mask,
        )

        assert output.logits.shape == (
            flat_inputs.words_embedding.shape[0],
            flat_inputs.words_embedding.shape[1],
            flat_inputs.child_embedding.shape[1],
            3,
        )
        assert output.extra["structuring_logits"].shape == (
            flat_inputs.words_embedding.shape[0],
            2,
            spans.shape[1],
        )
        assert output.extra["entity_field_logits"].shape == (
            flat_inputs.words_embedding.shape[0],
            spans.shape[1],
            flat_inputs.child_embedding.shape[1],
        )
        assert output.extra["groups_output"].shape == (2, 2, D)
        assert output.extra["anchor_mask"].shape == (2, 2)

    def test_entity_and_assignment_losses_train_together(
        self,
        shared,
        flat_inputs,
    ):
        from gliner.modeling.loss_functions import focal_loss_with_logits

        head = _make_head()
        flat_inputs.words_embedding = (
            flat_inputs.words_embedding.detach().requires_grad_(True)
        )
        batch_size = flat_inputs.words_embedding.shape[0]
        token_count = flat_inputs.words_embedding.shape[1]
        field_count = flat_inputs.child_embedding.shape[1]
        spans = torch.tensor([[[0, 0]], [[0, 0]]], dtype=torch.long)
        span_mask = torch.ones(batch_size, 1, dtype=torch.bool)
        labels = torch.zeros(batch_size, 2, token_count, field_count, 3)
        labels[:, 0, 0, 0] = 1.0
        span_labels = torch.zeros(batch_size, 1, 2, field_count)
        span_labels[:, 0, 0, 0] = 1.0

        output = head(
            shared,
            {},
            flat_inputs=flat_inputs,
            structuring_labels=labels,
            structuring_count=torch.ones(batch_size, dtype=torch.long),
            structuring_span_idx=spans,
            structuring_span_mask=span_mask,
            structuring_span_labels=span_labels,
            base_loss_fn=focal_loss_with_logits,
        )
        output.loss.backward()

        assert output.loss.isfinite()
        assert output.extra["entity_loss"].isfinite()
        assert output.extra["assignment_loss"].isfinite()
        assert flat_inputs.words_embedding.grad is not None

    def test_assignment_warns_and_truncates_gold_beyond_query_capacity(self):
        head = _make_head(num_fixed_slots=2)
        logits = torch.zeros(1, 2, 1)
        labels = torch.zeros(1, 1, 3, 1)
        labels[0, 0, 2, 0] = 1.0

        with pytest.warns(RuntimeWarning, match="3 visible records"):
            loss, matches, prediction_mask = head._assignment_loss(
                logits=logits,
                labels=labels,
                anchor_mask=torch.ones(1, 2, dtype=torch.bool),
                entity_mask=torch.ones(1, 1, dtype=torch.bool),
                label_count=torch.tensor([3]),
                base_loss_fn=lambda pred, target: (pred - target) ** 2,
            )

        assert loss.isfinite()
        assert len(matches[0]) == 2
        assert prediction_mask.tolist() == [[True, True]]

    def test_objectness_uses_the_configured_focal_loss(self):
        head = _make_head(anchor_objectness=True)
        observed = {}

        def focal_loss(logits, targets):
            observed["targets"] = targets.clone()
            return torch.where(
                targets.bool(),
                torch.full_like(logits, 2.0),
                torch.full_like(logits, 4.0),
            )

        loss = head._objectness_loss(
            logits=torch.zeros(1, 3),
            matches=[[(1, 0)]],
            anchor_mask=torch.tensor([[True, True, False]]),
            base_loss_fn=focal_loss,
        )

        assert observed["targets"].tolist() == [[0.0, 1.0, 0.0]]
        assert loss.item() == pytest.approx(3.0)

    def test_entity_loss_is_normalized_exactly_once(self, shared, flat_inputs):
        """The reported entity loss is the NER stage's own normalized loss.

        ``NERHead._bio_loss`` already divides by the active (word x field)
        cells; a head-level reduction on top of that used to shrink the entity
        term by a further factor of sum_b(entities_b * fields_b).
        """
        head = _make_head(bio_loss_reduction="mean")
        observed = []
        original_bio_loss = type(head)._bio_loss

        def spy(self, *args, **kwargs):
            loss = original_bio_loss(self, *args, **kwargs)
            observed.append(loss)
            return loss

        batch_size = flat_inputs.words_embedding.shape[0]
        token_count = flat_inputs.words_embedding.shape[1]
        field_count = flat_inputs.child_embedding.shape[1]
        spans = torch.tensor([[[0, 0], [1, 2]], [[0, 1], [2, 3]]])
        span_mask = torch.ones(batch_size, 2, dtype=torch.bool)
        labels = torch.zeros(batch_size, 2, token_count, field_count, 3)
        labels[:, 0, 0, 0] = 1.0
        span_labels = torch.zeros(batch_size, 2, spans.shape[1], field_count)

        with mock.patch.object(type(head), "_bio_loss", spy):
            output = head(
                shared,
                {},
                flat_inputs=flat_inputs,
                structuring_labels=labels,
                structuring_count=torch.ones(batch_size, dtype=torch.long),
                structuring_span_idx=spans,
                structuring_span_mask=span_mask,
                structuring_span_labels=span_labels,
                base_loss_fn=_binary_loss_fn,
            )

        assert len(observed) == 1
        assert output.extra["entity_loss"].item() == pytest.approx(
            observed[0].item()
        )

    def test_mean_matching_loss_masks_anchors_without_gold(self):
        head = _make_head(num_fixed_slots=3, bio_loss_reduction="mean")
        loss, matches, prediction_mask = head._assignment_loss(
            logits=torch.zeros(2, 3, 3),
            labels=torch.zeros(2, 3, 1, 1),
            anchor_mask=torch.tensor(
                [[True, True, False], [True, False, False]]
            ),
            entity_mask=torch.tensor(
                [[True, True, True], [True, False, False]]
            ),
            label_count=torch.zeros(2, dtype=torch.long),
            base_loss_fn=lambda predictions, targets: torch.ones_like(
                predictions
            ),
        )

        assert loss.item() == pytest.approx(0.0)
        assert matches == [[], []]
        assert prediction_mask.tolist() == [
            [True, True, False],
            [True, False, False],
        ]

    def test_matching_loss_masks_unmatched_negative_anchors(self):
        head = _make_head(num_fixed_slots=2, bio_loss_reduction="mean")
        logits = torch.tensor(
            [[[0.2, -0.2], [4.0, 4.0]]],
            requires_grad=True,
        )
        labels = torch.zeros(1, 2, 1, 1)
        labels[0, 0, 0, 0] = 1.0

        loss, matches, prediction_mask = head._assignment_loss(
            logits=logits,
            labels=labels,
            anchor_mask=torch.ones(1, 2, dtype=torch.bool),
            entity_mask=torch.ones(1, 2, dtype=torch.bool),
            label_count=torch.ones(1, dtype=torch.long),
            base_loss_fn=lambda predictions, targets: (
                predictions - targets
            ) ** 2,
        )
        loss.backward()

        assert matches == [[(0, 0)]]
        assert prediction_mask.tolist() == [[True, True]]
        assert loss.item() == pytest.approx(0.34)
        assert logits.grad[0, 0].abs().sum() > 0
        assert logits.grad[0, 1].abs().sum() == 0

    def test_each_loss_stage_has_independent_focal_controls(self):
        head = _make_head(
            ner_focal_loss_alpha=0.1,
            ner_focal_loss_gamma=1.0,
            ner_focal_loss_prob_margin=0.01,
            matching_focal_loss_alpha=0.2,
            matching_focal_loss_gamma=2.0,
            matching_focal_loss_prob_margin=0.02,
            objectness_focal_loss_alpha=0.3,
            objectness_focal_loss_gamma=3.0,
            objectness_focal_loss_prob_margin=0.03,
        )
        observed = {}

        def loss_fn(logits, targets, **kwargs):
            observed[len(observed)] = kwargs
            return torch.ones_like(logits)

        logits = torch.zeros(1)
        targets = torch.zeros(1)
        for component in ("ner", "matching", "objectness"):
            head._component_loss_fn(loss_fn, component)(logits, targets)

        assert observed == {
            0: {
                "focal_loss_alpha": 0.1,
                "focal_loss_gamma": 1.0,
                "focal_loss_prob_margin": 0.01,
            },
            1: {
                "focal_loss_alpha": 0.2,
                "focal_loss_gamma": 2.0,
                "focal_loss_prob_margin": 0.02,
            },
            2: {
                "focal_loss_alpha": 0.3,
                "focal_loss_gamma": 3.0,
                "focal_loss_prob_margin": 0.03,
            },
        }


class TestStructuringHungarianCosts:
    @staticmethod
    def _match(
        head,
        predictions,
        labels,
        *,
        prediction_mask=None,
        gold_mask=None,
        entity_mask=None,
        objectness_logits=None,
    ):
        batch_size, prediction_count, entity_count = predictions.shape
        gold_count = labels.shape[1]
        if prediction_mask is None:
            prediction_mask = torch.ones(
                batch_size,
                prediction_count,
                dtype=torch.bool,
            )
        if gold_mask is None:
            gold_mask = torch.ones(
                batch_size,
                gold_count,
                dtype=torch.bool,
            )
        if entity_mask is None:
            entity_mask = torch.ones(
                batch_size,
                entity_count,
                dtype=torch.bool,
            )
        return head._match_record_anchors(
            predictions,
            labels,
            prediction_mask,
            gold_mask,
            entity_mask,
            nn.functional.binary_cross_entropy_with_logits,
            objectness_logits=objectness_logits,
        )

    def test_perfect_dice_overlap_has_negative_one_cost(self):
        cost = StructuringHead._dice_matching_cost(
            probabilities=torch.tensor([[1.0, 0.0, 1.0]]),
            labels=torch.tensor([[1.0, 0.0, 1.0]]),
            entity_mask=torch.ones(3, dtype=torch.bool),
        )

        assert cost.item() == pytest.approx(-1.0)

    def test_zero_dice_overlap_has_positive_one_cost(self):
        cost = StructuringHead._dice_matching_cost(
            probabilities=torch.tensor([[0.0, 1.0]]),
            labels=torch.tensor([[1.0, 0.0]]),
            entity_mask=torch.ones(2, dtype=torch.bool),
        )

        assert cost.item() == pytest.approx(1.0, abs=1e-5)

    def test_false_positives_and_false_negatives_worsen_dice(self):
        labels = torch.tensor([[1.0, 1.0, 0.0]])
        mask = torch.ones(3, dtype=torch.bool)
        perfect = StructuringHead._dice_matching_cost(
            torch.tensor([[1.0, 1.0, 0.0]]),
            labels,
            mask,
        )
        false_positive = StructuringHead._dice_matching_cost(
            torch.tensor([[1.0, 1.0, 1.0]]),
            labels,
            mask,
        )
        false_negative = StructuringHead._dice_matching_cost(
            torch.tensor([[1.0, 0.0, 0.0]]),
            labels,
            mask,
        )

        assert false_positive.item() > perfect.item()
        assert false_negative.item() > perfect.item()

    def test_membership_and_objectness_costs_are_bounded(self):
        adjusted, probabilities = (
            StructuringHead._threshold_aware_probability(
                torch.tensor([-100.0, 0.0, 100.0]),
                threshold=0.5,
                temperature=1.0,
            )
        )
        membership = StructuringHead._membership_matching_cost(
            adjusted[:, None],
            labels=torch.tensor([[0.0], [1.0]]),
            entity_mask=torch.ones(1, dtype=torch.bool),
        )
        objectness = StructuringHead._objectness_matching_cost(
            probabilities,
            gold_count=2,
        )

        assert torch.all((-1.0 <= membership) & (membership <= 1.0))
        assert torch.all((-1.0 <= objectness) & (objectness <= 1.0))

    def test_inference_thresholds_map_to_zero_signed_cost(self):
        objectness_threshold = 0.7
        raw_logit = torch.logit(torch.tensor([objectness_threshold]))
        _, probabilities = (
            StructuringHead._threshold_aware_probability(
                raw_logit,
                threshold=objectness_threshold,
                temperature=0.25,
            )
        )
        # Both costs are centered on the same cutoff used at inference.
        membership = StructuringHead._membership_matching_cost(
            raw_logit[:, None],
            labels=torch.ones(1, 1),
            entity_mask=torch.ones(1, dtype=torch.bool),
            threshold=objectness_threshold,
        )
        objectness = StructuringHead._objectness_matching_cost(
            probabilities,
            gold_count=1,
        )

        assert membership.item() == pytest.approx(0.0, abs=1e-6)
        assert objectness.item() == pytest.approx(0.0, abs=1e-6)

    def test_higher_objectness_wins_tied_membership_and_dice(self):
        head = _make_head(
            anchor_objectness=True,
            matcher_objectness_cost=1.0,
        )
        matches = self._match(
            head,
            predictions=torch.zeros(1, 2, 2),
            labels=torch.tensor([[[1.0, 0.0]]]),
            objectness_logits=torch.tensor([[-4.0, 4.0]]),
        )

        assert matches == [[(1, 0)]]

    def test_objectness_does_not_select_gold_for_a_fixed_prediction(self):
        head = _make_head(
            anchor_objectness=True,
            matcher_objectness_cost=1.0,
        )
        predictions = torch.tensor([[[6.0, -6.0]]])
        labels = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

        low_objectness = self._match(
            head,
            predictions,
            labels,
            objectness_logits=torch.tensor([[-8.0]]),
        )
        high_objectness = self._match(
            head,
            predictions,
            labels,
            objectness_logits=torch.tensor([[8.0]]),
        )

        assert low_objectness == high_objectness == [[(0, 0)]]

    def test_invalid_anchors_are_excluded_and_padded_indices_restored(self):
        head = _make_head(
            anchor_objectness=True,
            matcher_objectness_cost=1.0,
        )
        matches = self._match(
            head,
            predictions=torch.tensor([[[8.0, -8.0], [-8.0, 8.0]]]),
            labels=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            prediction_mask=torch.tensor([[False, True]]),
            gold_mask=torch.tensor([[False, True]]),
            objectness_logits=torch.tensor([[8.0, -8.0]]),
        )

        assert matches == [[(1, 1)]]

    def test_unavailable_objectness_is_omitted_from_cost_and_denominator(self):
        head = _make_head(
            anchor_objectness=True,
            matcher_membership_cost=0.0,
            matcher_dice_cost=1.0,
            matcher_objectness_cost=7.0,
        )
        logits = torch.tensor([[2.0, -2.0]])
        labels = torch.tensor([[1.0, 0.0]])
        entity_mask = torch.ones(2, dtype=torch.bool)
        cost = head._record_anchor_pair_cost(
            logits,
            labels,
            entity_mask,
            objectness_logits=None,
        )
        expected = head._dice_matching_cost(
            logits.sigmoid(),
            labels,
            entity_mask,
        )

        assert torch.allclose(cost, expected)

    def test_empty_targets_and_entities_remain_finite(self):
        head = _make_head(anchor_objectness=True)
        cost = head._record_anchor_pair_cost(
            predicted_logits=torch.empty(2, 0),
            labels=torch.empty(1, 0),
            entity_mask=torch.empty(0, dtype=torch.bool),
            objectness_logits=torch.zeros(2),
        )
        matches = self._match(
            head,
            predictions=torch.empty(1, 2, 0),
            labels=torch.empty(1, 1, 0),
            gold_mask=torch.zeros(1, 1, dtype=torch.bool),
            entity_mask=torch.empty(1, 0, dtype=torch.bool),
            objectness_logits=torch.zeros(1, 2),
        )

        assert torch.isfinite(cost).all()
        assert matches == [[]]

    def test_zero_new_costs_preserve_membership_only_matching(self):
        head = _make_head(
            anchor_objectness=True,
            matcher_dice_cost=0.0,
            matcher_objectness_cost=0.0,
        )
        matches = self._match(
            head,
            predictions=torch.tensor(
                [[[8.0, -8.0], [-8.0, 8.0]]]
            ),
            labels=torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]),
            objectness_logits=torch.tensor([[8.0, -8.0]]),
        )

        assert matches == [[(0, 1), (1, 0)]]
