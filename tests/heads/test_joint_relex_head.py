"""Tests for JointRelexHead."""

from dataclasses import asdict

import pytest
import torch
from torch import nn

from glinext.config import GLiNextConfig, JointRelexHeadConfig, NERHeadConfig
from glinext.model import BaseGLiNextModel, GLiNExTTextModel
from glinext.tasks import TaskHeadOutput
from glinext.tasks.joint_relex.model import JointRelexHead
from glinext.tasks.ner.model import NERHead
from tests.heads.conftest import B, C, D, W, make_config


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

    def test_reuses_initialized_ner_without_registering_duplicate_modules(self):
        config = make_config(
            ner_config=asdict(NERHeadConfig()),
            joint_relex_config=asdict(JointRelexHeadConfig()),
        )
        ner_head = NERHead.from_config(config)

        head = JointRelexHead.from_config(config, ner_head=ner_head)

        assert head._owns_ner_head is False
        assert head.__dict__["_reused_ner_head"] is ner_head
        assert "_reused_ner_head" not in head._modules
        assert not hasattr(head, "scorer")
        ner_parameter_ids = {id(parameter) for parameter in ner_head.parameters()}
        assert ner_parameter_ids.isdisjoint(
            id(parameter) for parameter in head.parameters()
        )

    @pytest.mark.parametrize("reuse_ner", [True, False])
    def test_model_reuses_ner_when_available_and_owns_it_otherwise(
        self,
        monkeypatch,
        reuse_ner,
    ):
        config = GLiNextConfig(
            model_name="unused",
            hidden_size=D,
            vocab_size=32,
            encoder_config={"hidden_size": D, "model_type": "deberta-v2"},
            default_ner_config=reuse_ner,
            joint_relex_config=asdict(JointRelexHeadConfig()),
        )
        monkeypatch.setattr(
            BaseGLiNextModel,
            "_init_token_rep_layer",
            lambda *args, **kwargs: nn.Identity(),
        )

        model = GLiNExTTextModel(config)
        joint_head = model.heads["joint_relex"]
        state_keys = tuple(model.state_dict())

        assert joint_head._owns_ner_head is not reuse_ner
        if reuse_ner:
            assert joint_head.__dict__["_reused_ner_head"] is model.heads["ner"]
            assert any(key.startswith("heads.ner.scorer.") for key in state_keys)
            assert not any(
                key.startswith("heads.joint_relex.scorer.")
                for key in state_keys
            )
        else:
            assert "ner" not in model.heads
            assert hasattr(joint_head, "scorer")
            assert any(
                key.startswith("heads.joint_relex.scorer.")
                for key in state_keys
            )

    def test_reuse_model_strictly_loads_legacy_duplicate_ner_state(
        self,
        monkeypatch,
    ):
        config = GLiNextConfig(
            model_name="unused",
            hidden_size=D,
            vocab_size=32,
            encoder_config={"hidden_size": D, "model_type": "deberta-v2"},
            default_ner_config=True,
            joint_relex_config=asdict(JointRelexHeadConfig()),
        )
        monkeypatch.setattr(
            BaseGLiNextModel,
            "_init_token_rep_layer",
            lambda *args, **kwargs: nn.Identity(),
        )
        model = GLiNExTTextModel(config)
        legacy_state = model.state_dict()
        for key, value in tuple(legacy_state.items()):
            if not key.startswith("heads.ner."):
                continue
            suffix = key.removeprefix("heads.ner.")
            legacy_state[f"heads.joint_relex.{suffix}"] = value.clone()

        incompatible = model.load_state_dict(legacy_state, strict=True)

        assert incompatible.missing_keys == []
        assert incompatible.unexpected_keys == []

    def test_has_pair_rep_by_default(self):
        head = _make_head()
        assert isinstance(head.pair_rep_layer, torch.nn.Sequential)
        assert head.pair_rep_layer[0].in_features == D * 2
        assert head.pair_rep_layer[-1].out_features == D

    def test_can_disable_relations_rep_layer(self):
        head = _make_head(relations_layer=None)
        assert not hasattr(head, "relations_rep_layer")
        assert hasattr(head, "pair_rep_layer")

    def test_relation_safeguards_are_disabled_by_default(self):
        config = JointRelexHeadConfig()
        assert config.relation_loss_reduction == "mean"
        assert config.max_relation_span_width is None
        assert config.relation_span_nms is False
        assert config.max_relation_entities is None
        assert config.relation_top_k_neighbors is None

    @pytest.mark.parametrize(
        "field",
        [
            "max_relation_span_width",
            "max_relation_entities",
            "relation_top_k_neighbors",
        ],
    )
    def test_relation_limits_must_be_positive(self, field):
        with pytest.raises(ValueError, match=field):
            JointRelexHeadConfig(**{field: 0})

    def test_sparse_neighbors_require_dot_adjacency(self):
        with pytest.raises(ValueError, match="relations_layer='dot'"):
            JointRelexHeadConfig(
                relations_layer="mlp", relation_top_k_neighbors=2,
            )

    def test_relation_loss_reduction_must_be_supported(self):
        with pytest.raises(ValueError, match="relation_loss_reduction"):
            JointRelexHeadConfig(relation_loss_reduction="median")

    def test_anchor_modeling_builds_bounded_pair_selector(self):
        head = _make_head(
            relations_layer="anchor-modeling",
            anchor_layer={"type": "fixed", "params": {"num_slots": 3}},
            anchor_refinement="none",
        )

        assert hasattr(head, "anchor_relations_layer")
        assert not hasattr(head, "relations_rep_layer")
        assert head.anchor_relations_layer.max_anchors == 3
        assert head.anchor_relations_layer.endpoint_roles.shape == (2, D)

    def test_anchor_capacity_must_be_positive(self):
        with pytest.raises(ValueError, match="num_fixed_slots"):
            JointRelexHeadConfig(num_fixed_slots=0)


class TestAnchorPairSelection:
    def test_selector_multiplies_anchors_by_entities(self):
        head = _make_head(
            relations_layer="anchor_modeling",
            anchor_layer={"type": "fixed", "params": {"num_slots": 3}},
            anchor_refinement="none",
        )
        entity_representations = torch.randn(B, 4, D)
        entity_mask = torch.tensor([
            [True, True, True, False],
            [True, True, False, False],
        ])

        output = head.anchor_relations_layer(
            torch.randn(B, D),
            entity_representations,
            entity_mask,
        )

        assert output.assignment_logits.shape == (B, 3, 4, 2)
        assert output.pair_idx.shape == (B, 3, 2)
        assert output.anchor_mask.shape == (B, 3)
        for batch_idx, entity_count in enumerate((3, 2)):
            selected = output.pair_idx[batch_idx][output.pair_mask[batch_idx]]
            assert (selected >= 0).all()
            assert (selected < entity_count).all()
            assert (selected[:, 0] != selected[:, 1]).all()

    def test_hungarian_matching_uses_only_active_anchors(self):
        assignment_logits = torch.zeros(1, 2, 2, 2)
        # The inactive anchor is deliberately the best endpoint prediction.
        assignment_logits[0, 1, 0, 0] = 10.0
        assignment_logits[0, 1, 1, 1] = 10.0
        anchor_mask = torch.tensor([[True, False]])
        target_pair_idx = torch.tensor([[[0, 1]]])
        target_pair_mask = torch.tensor([[True]])

        matches = JointRelexHead._match_anchor_pairs(
            assignment_logits,
            anchor_mask,
            target_pair_idx,
            target_pair_mask,
        )
        pair_idx, pair_mask = JointRelexHead._matched_anchor_pair_indices(
            target_pair_idx,
            anchor_mask,
            matches,
        )

        assert matches == [[(0, 0)]]
        assert pair_mask.tolist() == [[True, False]]
        assert pair_idx.tolist() == [[[0, 1], [-1, -1]]]
        assignment_logits.requires_grad_()
        loss = JointRelexHead._anchor_assignment_loss(
            assignment_logits,
            target_pair_idx,
            anchor_mask,
            matches,
        )
        loss.backward()
        assert assignment_logits.grad[0, 0].abs().sum() > 0
        assert assignment_logits.grad[0, 1].abs().sum() == 0

    def test_pair_targets_exclude_padding_and_self_pairs(self):
        candidates = torch.ones(1, 3, 3)
        entity_mask = torch.tensor([[True, True, False]])

        pair_idx, pair_mask = JointRelexHead._pack_anchor_pair_targets(
            candidates,
            entity_mask,
        )

        assert pair_idx[0, pair_mask[0]].tolist() == [[0, 1], [1, 0]]

    def test_training_keeps_matched_positives_and_unmatched_background(self):
        predicted_pair_idx = torch.tensor([
            [[2, 1], [1, 2], [0, 1], [2, 0]],
        ])
        predicted_pair_mask = torch.ones(1, 4, dtype=torch.bool)
        anchor_mask = torch.tensor([[True, True, True, False]])
        target_pair_idx = torch.tensor([[[0, 1]]])
        rel_labels = torch.zeros(1, 3, 3, 1)
        rel_labels[0, 0, 1, 0] = 1.0

        pair_idx, pair_mask, matched_mask, background_mask = (
            JointRelexHead._training_anchor_pairs(
                predicted_pair_idx,
                predicted_pair_mask,
                target_pair_idx,
                anchor_mask,
                matches=[[(0, 0)]],
                rel_labels=rel_labels,
            )
        )

        assert pair_idx.tolist() == [[[0, 1], [1, 2], [-1, -1], [-1, -1]]]
        assert pair_mask.tolist() == [[True, True, False, False]]
        assert matched_mask.tolist() == [[True, False, False, False]]
        assert background_mask.tolist() == [[False, True, False, False]]


# ── Forward ──────────────────────────────────────────────────────────────

class TestJointRelexHeadForward:
    def test_relation_span_decoding_matches_ner_greedy_selection(
        self, flat_inputs,
    ):
        head = _make_head(relations_layer=None)
        logits = torch.full((B, W, C, 3), -10.0)
        # A weaker outer class-0 span overlaps a stronger inner class-1 span.
        logits[0, 0, 0, 0] = 5.0
        logits[0, 2, 0, 1] = 5.0
        logits[0, 0:3, 0, 2] = 5.0
        logits[0, 1, 1, :] = 7.0

        span_idx, span_mask, span_class_idx = (
            head._decode_relation_entity_spans(
                logits,
                flat_inputs,
                threshold=0.5,
                flat_ner=True,
                multi_label=False,
            )
        )

        assert span_mask[0].tolist() == [True]
        assert span_idx[0, 0].tolist() == [1, 1]
        assert span_class_idx[0, 0].item() == 1
        assert not span_mask[1].any()

    def test_alignment_validation_rejects_wrong_flattened_batch_order(
        self, flat_inputs,
    ):
        entity_count = 2
        rel_labels = torch.zeros(B, entity_count, entity_count, 1)
        rel_labels[0, 0, 1, 0] = 1.0
        rel_pair_mask = torch.zeros(B, entity_count, entity_count)
        rel_pair_mask[0, 0, 1] = 1.0
        rel_span_idx = torch.tensor([
            [[0, 0], [1, 1]],
            [[0, 0], [0, 0]],
        ])
        rel_span_mask = torch.tensor([
            [True, True],
            [False, False],
        ])

        with pytest.raises(ValueError, match="rel_batch_idx"):
            JointRelexHead._validate_relation_alignment(
                flat_inputs,
                torch.ones(B, 1, dtype=torch.bool),
                rel_labels,
                rel_pair_mask,
                torch.ones(B, dtype=torch.bool),
                torch.tensor([1, 0]),
                rel_span_idx,
                rel_span_mask,
                torch.tensor([[0, 1], [-1, -1]]),
            )

    def test_alignment_validation_rejects_positive_outside_pair_mask(
        self, flat_inputs,
    ):
        rel_labels = torch.zeros(B, 2, 2, 1)
        rel_labels[0, 0, 1, 0] = 1.0

        with pytest.raises(ValueError, match="include every positive"):
            JointRelexHead._validate_relation_alignment(
                flat_inputs,
                torch.ones(B, 1, dtype=torch.bool),
                rel_labels,
                torch.zeros(B, 2, 2),
                torch.ones(B, dtype=torch.bool),
                torch.arange(B),
                torch.tensor([
                    [[0, 0], [1, 1]],
                    [[0, 0], [0, 0]],
                ]),
                torch.tensor([
                    [True, True],
                    [False, False],
                ]),
                torch.tensor([[0, 1], [-1, -1]]),
            )

    def test_relation_class_axis_mismatch_is_not_silently_resized(
        self, shared,
    ):
        head = _make_head(relations_layer=None)

        with pytest.raises(ValueError, match="same class axis"):
            head._forward_relations(
                shared,
                torch.randn(B, 2, D),
                torch.ones(B, 2, dtype=torch.bool),
                torch.zeros(B, 2, 2, 2),
                0.5,
                None,
                flat_rel_prompts=torch.randn(B, 1, D),
                flat_rel_prompts_mask=torch.ones(B, 1, dtype=torch.bool),
            )

    @pytest.mark.parametrize(
        ("reduction", "expected_loss"),
        [("mean", 1.0), ("sum", 14.0)],
    )
    def test_relation_loss_reduction_uses_only_valid_pair_label_cells(
        self, shared, reduction, expected_loss,
    ):
        head = _make_head(
            relations_layer=None,
            relation_loss_reduction=reduction,
        )
        entity_count = 3
        relation_count = 2
        entity_mask = torch.tensor([
            [True, True, True],
            [True, True, False],
        ])
        prompt_mask = torch.tensor([
            [True, True],
            [True, False],
        ])

        def unit_loss(logits, labels, **kwargs):
            return torch.ones_like(logits)

        output = head._forward_relations(
            shared,
            torch.randn(B, entity_count, D),
            entity_mask,
            torch.zeros(B, entity_count, entity_count, relation_count),
            0.5,
            None,
            flat_rel_prompts=torch.randn(B, relation_count, D),
            flat_rel_prompts_mask=prompt_mask,
            base_loss_fn=unit_loss,
        )

        # Six/two directed pairs and two/one valid relation labels yield 14
        # valid cells. Padded cells are excluded from both reductions.
        assert output.extra["rel_mask"].sum(dim=1).tolist() == [6, 2]
        assert output.loss.item() == pytest.approx(expected_loss)

    def test_mean_relation_loss_uses_active_entity_pair_normalizer(self, shared):
        head = _make_head(
            relations_layer="dot",
            relation_loss_reduction="mean",
            adjacency_loss_coef=0.0,
        )
        entity_mask = torch.tensor([
            [True, True, True],
            [True, True, False],
        ])
        prompt_mask = torch.tensor([
            [True, True],
            [True, False],
        ])
        pair_mask = torch.zeros(B, 3, 3)
        pair_mask[0, 0, 1] = 1.0
        pair_mask[1, 0, 1] = 1.0

        output = head._forward_relations(
            shared,
            torch.randn(B, 3, D),
            entity_mask,
            torch.zeros(B, 3, 3, 2),
            0.5,
            None,
            flat_rel_prompts=torch.randn(B, 2, D),
            flat_rel_prompts_mask=prompt_mask,
            rel_pair_mask=pair_mask,
            base_loss_fn=lambda logits, labels, **kwargs: torch.ones_like(logits),
        )

        # The sampled numerator has 2 + 1 cells. The normalizer is
        # 3*(3-1)*2 + 2*(2-1)*1 = 14 active pair/class cells.
        assert output.loss.item() == pytest.approx(3.0 / 14.0)

    def test_relation_specific_focal_settings_override_generic_values(self, shared):
        head = _make_head(relations_layer=None)
        captured = {}

        def capture_loss(logits, labels, **kwargs):
            captured.update(kwargs)
            return torch.ones_like(logits)

        loss_kwargs = head._relation_loss_kwargs({
            "focal_loss_alpha": 0.9,
            "focal_loss_gamma": 3.0,
            "rel_focal_loss_alpha": 0.25,
            "rel_focal_loss_gamma": 2.0,
        })
        head._forward_relations(
            shared,
            torch.randn(B, 2, D),
            torch.ones(B, 2, dtype=torch.bool),
            torch.zeros(B, 2, 2, 1),
            0.5,
            None,
            flat_rel_prompts=torch.randn(B, 1, D),
            flat_rel_prompts_mask=torch.ones(B, 1, dtype=torch.bool),
            base_loss_fn=capture_loss,
            loss_kwargs=loss_kwargs,
        )

        assert captured["focal_loss_alpha"] == pytest.approx(0.25)
        assert captured["focal_loss_gamma"] == pytest.approx(2.0)

    def test_configured_relation_focal_settings_override_runtime_values(self):
        head = _make_head(
            relation_focal_loss_alpha=0.4,
            relation_focal_loss_gamma=1.5,
            relation_focal_loss_prob_margin=0.1,
        )

        kwargs = head._relation_loss_kwargs({
            "rel_focal_loss_alpha": 0.2,
            "rel_focal_loss_gamma": 3.0,
            "rel_focal_loss_prob_margin": 0.0,
        })

        assert kwargs["focal_loss_alpha"] == pytest.approx(0.4)
        assert kwargs["focal_loss_gamma"] == pytest.approx(1.5)
        assert kwargs["focal_loss_prob_margin"] == pytest.approx(0.1)

    def test_mean_relation_loss_is_zero_without_valid_pairs(self, shared):
        head = _make_head(
            relations_layer=None,
            relation_loss_reduction="mean",
        )

        output = head._forward_relations(
            shared,
            torch.randn(B, 2, D),
            torch.zeros(B, 2, dtype=torch.bool),
            torch.zeros(B, 2, 2, 1),
            0.5,
            None,
            flat_rel_prompts=torch.randn(B, 1, D),
            flat_rel_prompts_mask=torch.ones(B, 1, dtype=torch.bool),
            base_loss_fn=lambda logits, labels, **kwargs: torch.ones_like(logits),
        )

        assert torch.isfinite(output.loss)
        assert output.loss.item() == pytest.approx(0.0)

    def test_ner_only_inference_skips_relation_pair_construction(
        self, shared, flat_inputs, monkeypatch,
    ):
        head = _make_head()

        def fail_extract(*args, **kwargs):
            raise AssertionError("NER-only inference must not construct relation pairs")

        monkeypatch.setattr(
            "glinext.tasks.joint_relex.model.extract_spans_from_tokens",
            fail_extract,
        )
        out = head(shared, {}, flat_inputs=flat_inputs)

        assert out.logits is not None
        assert out.extra["rel_logits"] is None
        assert out.extra["rel_idx"] is None

    def test_reused_ner_consumes_cached_dependency_output(
        self, shared, flat_inputs, monkeypatch,
    ):
        config = make_config(
            ner_config=asdict(NERHeadConfig()),
            joint_relex_config=asdict(JointRelexHeadConfig()),
        )
        ner_head = NERHead.from_config(config)
        head = JointRelexHead.from_config(config, ner_head=ner_head)
        ner_output = ner_head(shared, {}, flat_inputs=flat_inputs)
        ner_output.loss = torch.tensor(7.0)

        def fail_recompute(*args, **kwargs):
            raise AssertionError("Joint Relex recomputed an available NER output")

        monkeypatch.setattr(ner_head, "forward", fail_recompute)
        output = head(
            shared,
            {"ner": ner_output},
            flat_inputs=flat_inputs,
        )

        assert output.logits is ner_output.logits
        # The standalone NER head contributes this loss through its own task;
        # Joint Relex must not add it to the model loss a second time.
        assert output.loss is None

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
        head = _make_head(relations_layer=None)
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

    def test_adjacency_training_uses_candidate_pair_mask(self, shared, flat_inputs):
        head = _make_head(relations_layer="dot")
        R = 1
        E = 3
        ner_labels = torch.zeros(B, W, C, 3)
        rel_labels = torch.zeros(B, E, E, R)
        rel_labels[0, 0, 1, 0] = 1.0
        rel_pair_mask = torch.zeros(B, E, E)
        rel_pair_mask[0, 0, 1] = 1.0  # positive relation
        rel_pair_mask[0, 1, 0] = 1.0  # sampled no-relation candidate

        rel_label_embeds = torch.randn(B, R, D)
        rel_span_idx = torch.tensor([
            [[0, 0], [1, 1], [2, 2]],
            [[0, 0], [0, 0], [0, 0]],
        ], dtype=torch.long)
        rel_span_mask = torch.tensor([
            [True, True, True],
            [False, False, False],
        ])

        from gliner.modeling.loss_functions import focal_loss_with_logits
        out = head(
            shared, {},
            flat_inputs=flat_inputs,
            ner_labels=ner_labels,
            rel_labels=rel_labels,
            rel_pair_mask=rel_pair_mask,
            rel_span_idx=rel_span_idx,
            rel_span_mask=rel_span_mask,
            rel_label_embeds=rel_label_embeds,
            base_loss_fn=focal_loss_with_logits,
        )

        assert out.extra["rel_mask"][0].sum().item() == 2
        assert out.loss is not None

    def test_anchor_training_matches_and_gathers_candidate_pairs(
        self, shared, flat_inputs,
    ):
        head = _make_head(
            relations_layer="anchor_modeling",
            anchor_layer={"type": "fixed", "params": {"num_slots": 3}},
            anchor_refinement="none",
        )
        relation_count = 2
        entity_count = 3
        rel_labels = torch.zeros(
            B, entity_count, entity_count, relation_count,
        )
        rel_labels[0, 0, 1, 0] = 1.0
        rel_labels[0, 1, 0, 1] = 1.0
        rel_pair_mask = torch.zeros(B, entity_count, entity_count)
        rel_pair_mask[0, 0, 1] = 1.0
        rel_pair_mask[0, 1, 0] = 1.0
        rel_pair_mask[1, 0, 1] = 1.0
        rel_span_idx = torch.tensor([
            [[0, 0], [1, 1], [2, 2]],
            [[0, 0], [1, 1], [2, 2]],
        ])
        rel_span_mask = torch.tensor([
            [True, True, True],
            [True, True, False],
        ])

        from gliner.modeling.loss_functions import focal_loss_with_logits
        output = head(
            shared,
            {},
            flat_inputs=flat_inputs,
            ner_labels=torch.zeros(B, W, C, 3),
            rel_labels=rel_labels,
            rel_pair_mask=rel_pair_mask,
            rel_span_idx=rel_span_idx,
            rel_span_mask=rel_span_mask,
            rel_label_embeds=torch.randn(B, relation_count, D),
            base_loss_fn=focal_loss_with_logits,
        )

        assert output.loss is not None and torch.isfinite(output.loss)
        assert output.extra["rel_assignment_logits"].shape == (
            B, 3, entity_count, 2,
        )
        assert output.extra["rel_anchor_mask"].shape == (B, 3)
        # Only relation-bearing pairs participate in Hungarian matching. The
        # sampled no-relation pair in item 1 is relation background instead.
        assert [len(matches) for matches in output.extra["rel_anchor_matches"]] == [2, 0]
        assert output.extra["rel_mask"][0].sum().item() >= 2
        selected_first = {
            tuple(pair)
            for pair in output.extra["rel_idx"][0][
                output.extra["rel_mask"][0]
            ].tolist()
        }
        assert {(0, 1), (1, 0)} <= selected_first
        assert torch.isfinite(output.extra["rel_assignment_loss"])

        output.loss.backward()
        assert head.anchor_relations_layer.endpoint_roles.grad is not None

    def test_anchor_inference_supports_triples_scorer(
        self, shared, flat_inputs,
    ):
        head = _make_head(
            relations_layer="anchor_modeling",
            triples_layer="DistMult",
            anchor_layer={"type": "fixed", "params": {"num_slots": 2}},
            anchor_refinement="none",
        )
        rel_span_idx = torch.tensor([
            [[0, 0], [1, 1], [2, 2]],
            [[0, 0], [1, 1], [2, 2]],
        ])
        rel_span_mask = torch.ones(B, 3, dtype=torch.bool)

        output = head(
            shared,
            {},
            flat_inputs=flat_inputs,
            rel_span_idx=rel_span_idx,
            rel_span_mask=rel_span_mask,
            rel_label_embeds=torch.randn(B, 2, D),
        )

        assert output.extra["rel_logits"].shape == (B, 2, 2)
        assert output.extra["rel_idx"].shape == (B, 2, 2)

    def test_sparse_neighbors_do_not_materialize_dense_adjacency(
        self, shared, flat_inputs, monkeypatch,
    ):
        head = _make_head(
            relations_layer="dot",
            relation_top_k_neighbors=1,
            relation_neighbor_chunk_size=1,
        )

        def fail_dense(*args, **kwargs):
            raise AssertionError("dense adjacency must not be materialized")

        monkeypatch.setattr(head.relations_rep_layer, "forward", fail_dense)
        entity_count = 4
        rel_span_idx = torch.tensor([
            [[0, 0], [1, 1], [2, 2], [3, 3]],
            [[0, 0], [1, 1], [2, 2], [3, 3]],
        ])
        rel_span_mask = torch.ones(B, entity_count, dtype=torch.bool)
        output = head(
            shared, {}, flat_inputs=flat_inputs,
            rel_span_idx=rel_span_idx,
            rel_span_mask=rel_span_mask,
            rel_label_embeds=torch.randn(B, 2, D),
            adjacency_threshold=0.0,
        )

        assert output.extra["rel_idx"].shape[1] == entity_count
        assert output.extra["rel_mask"].sum(dim=1).tolist() == [entity_count] * B

    def test_sparse_training_uses_chunked_adjacency_loss(
        self, shared, flat_inputs, monkeypatch,
    ):
        head = _make_head(
            relations_layer="dot",
            relation_top_k_neighbors=1,
            relation_neighbor_chunk_size=1,
        )

        def fail_dense(*args, **kwargs):
            raise AssertionError("dense adjacency must not be materialized")

        monkeypatch.setattr(head.relations_rep_layer, "forward", fail_dense)
        rel_span_idx = torch.tensor([
            [[0, 0], [1, 1], [2, 2]],
            [[0, 0], [0, 0], [0, 0]],
        ])
        rel_span_mask = torch.tensor([
            [True, True, True],
            [False, False, False],
        ])
        rel_labels = torch.zeros(B, 3, 3, 1)
        rel_labels[0, 0, 1, 0] = 1.0
        rel_pair_mask = torch.zeros(B, 3, 3)
        rel_pair_mask[0, 0, 1] = 1.0
        rel_pair_mask[0, 1, 0] = 1.0

        from gliner.modeling.loss_functions import focal_loss_with_logits
        output = head(
            shared, {}, flat_inputs=flat_inputs,
            ner_labels=torch.zeros(B, W, C, 3),
            rel_labels=rel_labels,
            rel_pair_mask=rel_pair_mask,
            rel_span_idx=rel_span_idx,
            rel_span_mask=rel_span_mask,
            rel_label_embeds=torch.randn(B, 1, D),
            base_loss_fn=focal_loss_with_logits,
        )

        assert output.loss is not None
        assert output.extra["rel_mask"][0].sum().item() == 2


class TestRelationSpanSafeguards:
    def test_width_nms_and_cap_compact_spans(self, monkeypatch):
        head = _make_head(
            max_relation_span_width=2,
            relation_span_nms=True,
            max_relation_entities=2,
        )
        span_idx = torch.tensor([
            [[0, 3], [0, 1], [1, 2], [4, 4], [6, 6]],
            [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        ])
        span_mask = torch.tensor([
            [True, True, True, True, True],
            [False, False, False, False, False],
        ])

        def fixed_scores(_scores, _spans, source_ids, _batch_idx):
            by_source = {1: 0.8, 2: 0.9, 3: 0.7, 4: 0.6}
            return torch.tensor(
                [by_source[source_id] for source_id in source_ids.tolist()],
                device=source_ids.device,
            )

        monkeypatch.setattr(head, "_score_relation_spans", fixed_scores)
        selected, selected_mask, source_indices = head._select_relation_spans(
            torch.zeros(B, W, C, 3), span_idx, span_mask,
        )

        # [0,3] is too wide. [1,2] outranks and suppresses overlapping [0,1].
        assert selected[0, :2].tolist() == [[1, 2], [4, 4]]
        assert selected_mask[0].tolist() == [True, True]
        assert source_indices[0].tolist() == [2, 3]
        assert not selected_mask[1].any()

    def test_filtered_entities_remap_relation_targets(self):
        rel_labels = torch.zeros(B, 4, 4, 1)
        rel_labels[0, 2, 3, 0] = 1.0
        rel_pair_mask = torch.zeros(B, 4, 4)
        rel_pair_mask[0, 2, 3] = 1.0
        rel_pair_mask[0, 3, 2] = 1.0
        source_indices = torch.tensor([[2, 3], [-1, -1]])
        span_mask = torch.tensor([[True, True], [False, False]])

        selected_labels, selected_pairs = JointRelexHead._remap_relation_targets(
            rel_labels, rel_pair_mask, source_indices, span_mask,
        )

        assert selected_labels.shape == (B, 2, 2, 1)
        assert selected_labels[0, 0, 1, 0] == 1.0
        assert selected_pairs[0, 0, 1] == 1.0
        assert selected_pairs[0, 1, 0] == 1.0
        assert selected_labels[1].sum() == 0


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
        assert mask[0].sum() == 2
        assert mask[1].sum() == 0

    def test_multi_token_span_uses_span_rep_layer(self, shared):
        head = _make_head()
        span_idx = torch.tensor([[[1, 3]], [[0, 0]]], dtype=torch.long)  # (B, 1, 2)
        span_mask = torch.ones(B, 1, dtype=torch.bool)

        rep, mask = head._pool_entity_spans(shared.words_embedding, span_idx, span_mask)
        assert rep.shape == (B, 1, D)
        assert mask.sum().item() == B

    def test_masked_entity_is_zero(self, shared):
        head = _make_head()
        span_idx = torch.tensor([[[0, 0]], [[0, 0]]], dtype=torch.long)
        span_mask = torch.tensor([[False], [False]], dtype=torch.bool)

        rep, _ = head._pool_entity_spans(shared.words_embedding, span_idx, span_mask)
        assert torch.all(rep == 0)
