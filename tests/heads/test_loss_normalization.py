"""Loss-normalization controls for the NER and Joint Relex heads.

Both heads historically returned an unnormalized masked sum, so their scale
grew with batch size, sequence length and the prompted label count. These
tests pin the opt-in reductions and the config-over-trainer precedence of the
focal-loss controls.
"""

from dataclasses import asdict

import pytest
import torch

from gliformer.config import (
    ClassificationHeadConfig,
    JointRelexHeadConfig,
    NERHeadConfig,
)
from gliformer.tasks import TaskFlatInputs
from gliformer.tasks.classification.model import ClassificationHead
from gliformer.tasks.joint_relex.model import JointRelexHead
from gliformer.tasks.losses import binary_focal_or_bce
from gliformer.tasks.ner.model import NERHead
from tests.heads.conftest import B, C, D, W, make_config


def _loss_fn(logits, labels, **kwargs):
    kwargs.setdefault("focal_loss_alpha", 0.8)
    kwargs.setdefault("focal_loss_gamma", 2.0)
    kwargs["reduction"] = "none"
    return binary_focal_or_bce(logits, labels, **kwargs)


def _ner_head(**kwargs):
    config = make_config(ner_config=asdict(NERHeadConfig(**kwargs)))
    return NERHead.from_config(config)


def _partial_flat_inputs(active_w, active_c):
    """TaskFlatInputs whose word and child masks are only partly active."""
    word_mask = torch.zeros(B, W, dtype=torch.long)
    word_mask[:, :active_w] = 1
    child_mask = torch.zeros(B, C, dtype=torch.long)
    child_mask[:, :active_c] = 1
    return TaskFlatInputs(
        words_embedding=torch.randn(B, W, D),
        mask=word_mask,
        parent_embedding=torch.randn(B, D),
        child_embedding=torch.randn(B, C, D),
        child_mask=child_mask,
        batch_origin=torch.arange(B),
    )


def _ner_labels():
    labels = torch.zeros(B, W, C, 3)
    labels[0, 1, 0, 0] = 1.0
    labels[1, 2, 1, 2] = 1.0
    return labels


# ── NER ──────────────────────────────────────────────────────────────────

class TestNERBioLossReduction:
    def test_default_is_the_historical_sum(self):
        assert NERHeadConfig().bio_loss_reduction == "sum"
        assert _ner_head().bio_loss_reduction == "sum"

    def test_rejects_unknown_reduction(self):
        with pytest.raises(ValueError, match="bio_loss_reduction"):
            NERHeadConfig(bio_loss_reduction="avg")

    @pytest.mark.parametrize("active_w,active_c", [(W, C), (6, 2), (3, 1)])
    def test_mean_divides_by_active_bn_w_c(self, shared, active_w, active_c):
        """mean == sum / (BN x active W x active C)."""
        labels = _ner_labels()
        losses = {}
        for reduction in ("sum", "mean"):
            torch.manual_seed(0)
            head = _ner_head(bio_loss_reduction=reduction)
            head.eval()
            torch.manual_seed(1)
            flat_inputs = _partial_flat_inputs(active_w, active_c)
            out = head(
                shared, {}, flat_inputs=flat_inputs,
                base_loss_fn=_loss_fn, ner_labels=labels,
            )
            losses[reduction] = out.loss.item()

        supervised_cells = B * active_w * active_c
        assert losses["mean"] == pytest.approx(
            losses["sum"] / supervised_cells, rel=1e-5,
        )

    def test_mean_is_stable_across_label_count(self, shared):
        """The whole point: scale must not track the prompted class count."""
        labels = _ner_labels()
        scales = []
        for active_c in (1, 2, 3):
            torch.manual_seed(0)
            head = _ner_head(bio_loss_reduction="mean")
            head.eval()
            torch.manual_seed(1)
            flat_inputs = _partial_flat_inputs(W, active_c)
            out = head(
                shared, {}, flat_inputs=flat_inputs,
                base_loss_fn=_loss_fn, ner_labels=labels,
            )
            scales.append(out.loss.item())

        # Sums over these three would differ by ~3x; means stay the same order.
        assert max(scales) / min(scales) < 3.0

    def test_forward_override_beats_config(self, shared):
        """Composite heads pin "sum" so their own reduction cannot compound."""
        labels = _ner_labels()
        torch.manual_seed(0)
        head = _ner_head(bio_loss_reduction="mean")
        head.eval()
        torch.manual_seed(1)
        flat_inputs = _partial_flat_inputs(W, C)
        overridden = head(
            shared, {}, flat_inputs=flat_inputs, base_loss_fn=_loss_fn,
            ner_labels=labels, bio_loss_reduction="sum",
        ).loss.item()

        torch.manual_seed(0)
        summed_head = _ner_head(bio_loss_reduction="sum")
        summed_head.eval()
        torch.manual_seed(1)
        flat_inputs = _partial_flat_inputs(W, C)
        expected = summed_head(
            shared, {}, flat_inputs=flat_inputs,
            base_loss_fn=_loss_fn, ner_labels=labels,
        ).loss.item()

        assert overridden == pytest.approx(expected, rel=1e-6)

    def test_custom_focal_params_reach_the_loss(self, shared):
        """ner_config focal controls must change the produced loss."""
        labels = _ner_labels()
        results = []
        for alpha in (0.2, 0.9):
            torch.manual_seed(0)
            head = _ner_head(bio_loss_reduction="mean")
            head.eval()
            torch.manual_seed(1)
            flat_inputs = _partial_flat_inputs(W, C)
            out = head(
                shared, {}, flat_inputs=flat_inputs, ner_labels=labels,
                base_loss_fn=lambda lg, lb: binary_focal_or_bce(
                    lg, lb, focal_loss_alpha=alpha,
                    focal_loss_gamma=2.0, reduction="none",
                ),
            )
            results.append(out.loss.item())
        assert results[0] != pytest.approx(results[1])


# ── Joint Relex ──────────────────────────────────────────────────────────

class TestRelationLossReduction:
    def test_rejects_unknown_reduction(self):
        with pytest.raises(ValueError, match="relation_loss_reduction"):
            JointRelexHeadConfig(relation_loss_reduction="avg")

    @pytest.mark.parametrize("entities", [2, 3, 4])
    @pytest.mark.parametrize("n_rel", [1, 4])
    def test_entity_pairs_divides_by_e_times_e_minus_one_and_c_rel(
        self, entities, n_rel,
    ):
        """entity_pairs == sum / sum_b e_b*(e_b-1)*C_rel_b."""
        n_ent_slots = 5
        losses = {}
        for reduction in ("sum", "entity_pairs", "mean"):
            torch.manual_seed(0)
            config = make_config(
                joint_relex_config=asdict(
                    JointRelexHeadConfig(relation_loss_reduction=reduction),
                ),
            )
            head = JointRelexHead.from_config(config)
            head.eval()

            torch.manual_seed(1)
            span_rep = torch.randn(B, n_ent_slots, D)
            span_mask = torch.zeros(B, n_ent_slots, dtype=torch.bool)
            span_mask[:, :entities] = True
            rel_labels = torch.zeros(B, n_ent_slots, n_ent_slots, n_rel)
            rel_labels[0, 0, 1, 0] = 1.0
            prompts = torch.randn(B, n_rel, D)
            prompts_mask = torch.ones(B, n_rel, dtype=torch.bool)


            out = head._forward_relations(
                shared=None,
                target_span_rep=span_rep,
                target_span_mask=span_mask,
                rel_labels=rel_labels,
                adjacency_threshold=0.5,
                rel_label_embeds=None,
                flat_rel_prompts=prompts,
                flat_rel_prompts_mask=prompts_mask,
                base_loss_fn=_loss_fn,
            )
            losses[reduction] = out.loss.item()

        expected_divisor = B * entities * (entities - 1) * n_rel
        assert losses["entity_pairs"] == pytest.approx(
            losses["sum"] / expected_divisor, rel=1e-5,
        )
        # build_all_entity_pairs emits exactly the e*(e-1) off-diagonal pairs,
        # so with no candidate pruning this coincides with plain "mean".
        assert losses["entity_pairs"] == pytest.approx(losses["mean"], rel=1e-5)

    def test_entity_pairs_survives_fewer_than_two_entities(self):
        """e < 2 yields no pairs; the clamped divisor must not blow up."""
        n_rel, n_ent_slots = 3, 4
        config = make_config(
            joint_relex_config=asdict(
                JointRelexHeadConfig(relation_loss_reduction="entity_pairs"),
            ),
        )
        torch.manual_seed(0)
        head = JointRelexHead.from_config(config)
        head.eval()

        span_mask = torch.zeros(B, n_ent_slots, dtype=torch.bool)
        span_mask[:, :1] = True  # a single entity in every item
        out = head._forward_relations(
            shared=None,
            target_span_rep=torch.randn(B, n_ent_slots, D),
            target_span_mask=span_mask,
            rel_labels=torch.zeros(B, n_ent_slots, n_ent_slots, n_rel),
            adjacency_threshold=0.5,
            rel_label_embeds=None,
            flat_rel_prompts=torch.randn(B, n_rel, D),
            flat_rel_prompts_mask=torch.ones(B, n_rel, dtype=torch.bool),
            base_loss_fn=_loss_fn,
        )
        assert torch.isfinite(out.loss)


class TestRelationFocalPrecedence:
    def _kwargs(self, batch, **cfg_overrides):
        config = make_config(
            joint_relex_config=asdict(JointRelexHeadConfig(**cfg_overrides)),
        )
        head = JointRelexHead.from_config(config)
        return head._relation_loss_kwargs(batch)

    def test_head_config_beats_trainer_rel_overrides(self):
        batch = {
            "focal_loss_alpha": 0.9,
            "rel_focal_loss_alpha": 0.7,
            "focal_loss_gamma": 2.0,
        }
        kwargs = self._kwargs(batch, focal_loss_alpha=0.35)
        assert kwargs["focal_loss_alpha"] == 0.35
        # An unset config field still falls back to the trainer value.
        assert kwargs["focal_loss_gamma"] == 2.0

    def test_trainer_rel_override_used_when_config_is_unset(self):
        batch = {"focal_loss_alpha": 0.9, "rel_focal_loss_alpha": 0.7}
        assert self._kwargs(batch)["focal_loss_alpha"] == 0.7

    def test_global_focal_used_when_no_rel_override(self):
        batch = {"focal_loss_alpha": 0.9}
        assert self._kwargs(batch)["focal_loss_alpha"] == 0.9

    def test_prob_margin_zero_from_config_is_honoured(self):
        """0.0 is a real setting, not "unset"."""
        batch = {"focal_loss_prob_margin": 0.4}
        kwargs = self._kwargs(batch, focal_loss_prob_margin=0.0)
        assert kwargs["focal_loss_prob_margin"] == 0.0

    def test_entity_pairs_uses_per_item_class_counts(self):
        """C_rel is counted per item, not taken from the padded tensor width."""
        n_rel, n_ent_slots, entities = 4, 5, 3
        active_per_item = [4, 2]
        assert len(active_per_item) == B

        losses = {}
        for reduction in ("sum", "entity_pairs"):
            torch.manual_seed(0)
            config = make_config(
                joint_relex_config=asdict(
                    JointRelexHeadConfig(relation_loss_reduction=reduction),
                ),
            )
            head = JointRelexHead.from_config(config)
            head.eval()

            torch.manual_seed(1)
            span_mask = torch.zeros(B, n_ent_slots, dtype=torch.bool)
            span_mask[:, :entities] = True
            prompts_mask = torch.zeros(B, n_rel, dtype=torch.bool)
            for item, active in enumerate(active_per_item):
                prompts_mask[item, :active] = True

            out = head._forward_relations(
                shared=None,
                target_span_rep=torch.randn(B, n_ent_slots, D),
                target_span_mask=span_mask,
                rel_labels=torch.zeros(B, n_ent_slots, n_ent_slots, n_rel),
                adjacency_threshold=0.5,
                rel_label_embeds=None,
                flat_rel_prompts=torch.randn(B, n_rel, D),
                flat_rel_prompts_mask=prompts_mask,
                base_loss_fn=_loss_fn,
            )
            losses[reduction] = out.loss.item()

        pairs = entities * (entities - 1)
        expected_divisor = sum(pairs * active for active in active_per_item)
        assert losses["entity_pairs"] == pytest.approx(
            losses["sum"] / expected_divisor, rel=1e-5,
        )

    def test_entity_pairs_diverges_from_mean_once_pairs_are_pruned(self):
        """The reason the mode exists: it ignores the selector's recall.

        With an adjacency layer the candidate set is a pruned subset, so
        "mean" tracks the surviving pair count while "entity_pairs" stays on
        the full e*(e-1) grid.
        """
        n_rel, n_ent_slots, entities = 3, 5, 4
        losses = {}
        for reduction in ("mean", "entity_pairs"):
            torch.manual_seed(0)
            config = make_config(
                joint_relex_config=asdict(
                    JointRelexHeadConfig(
                        relations_layer="dot",
                        relation_loss_reduction=reduction,
                    ),
                ),
            )
            head = JointRelexHead.from_config(config)
            head.eval()

            torch.manual_seed(1)
            span_mask = torch.zeros(B, n_ent_slots, dtype=torch.bool)
            span_mask[:, :entities] = True
            # Supervise a single pair, so pair selection keeps far fewer than
            # the e*(e-1) candidates.
            rel_labels = torch.zeros(B, n_ent_slots, n_ent_slots, n_rel)
            rel_labels[:, 0, 1, 0] = 1.0
            rel_pair_mask = torch.zeros(B, n_ent_slots, n_ent_slots)
            rel_pair_mask[:, 0, 1] = 1.0

            out = head._forward_relations(
                shared=None,
                target_span_rep=torch.randn(B, n_ent_slots, D),
                target_span_mask=span_mask,
                rel_labels=rel_labels,
                adjacency_threshold=0.5,
                rel_label_embeds=None,
                flat_rel_prompts=torch.randn(B, n_rel, D),
                flat_rel_prompts_mask=torch.ones(B, n_rel, dtype=torch.bool),
                rel_pair_mask=rel_pair_mask,
                base_loss_fn=_loss_fn,
            )
            losses[reduction] = out.loss.item()

        assert losses["mean"] != pytest.approx(losses["entity_pairs"], rel=1e-3)


# ── Classification ───────────────────────────────────────────────────────

class TestClassificationLossReduction:
    def test_default_is_the_historical_sum(self):
        assert ClassificationHeadConfig().loss_reduction == "sum"

    def test_rejects_unknown_reduction(self):
        with pytest.raises(ValueError, match="loss_reduction"):
            ClassificationHeadConfig(loss_reduction="avg")

    @pytest.mark.parametrize("active_c", [1, 2, C])
    def test_mean_divides_by_active_bn_c(self, shared, active_c):
        """The class mask comes from flat_inputs.child_mask, not a kwarg."""
        cat_labels = torch.zeros(B, C)
        cat_labels[0, 0] = 1.0

        losses = {}
        for reduction in ("sum", "mean"):
            torch.manual_seed(0)
            config = make_config(
                classification_config=asdict(
                    ClassificationHeadConfig(loss_reduction=reduction),
                ),
            )
            head = ClassificationHead.from_config(config)
            head.eval()
            torch.manual_seed(1)
            flat_inputs = _partial_flat_inputs(W, active_c)
            out = head(
                shared, {}, flat_inputs=flat_inputs,
                base_loss_fn=_loss_fn, cat_labels=cat_labels,
            )
            losses[reduction] = out.loss.item()

        assert losses["mean"] == pytest.approx(
            losses["sum"] / (B * active_c), rel=1e-5,
        )


# ── Head loss policy ─────────────────────────────────────────────────────

class TestHeadLossPolicy:
    """Heads expose their configured loss policies and normalize relation losses."""

    @staticmethod
    def _built_model_without_reduction_fields():
        """A model as loaded from a checkpoint predating the reduction fields."""
        from gliformer.model import GLiFormerModel

        config = make_config(
            ner_config=asdict(NERHeadConfig()),
            classification_config=asdict(ClassificationHeadConfig()),
            joint_relex_config=asdict(JointRelexHeadConfig()),
            encoder_config={
                "hidden_size": D,
                "model_type": "deberta-v2",
                "num_attention_heads": 4,
                "num_hidden_layers": 1,
                "intermediate_size": D * 2,
                "vocab_size": 128,
            },
        )
        return GLiFormerModel(config), config

    def test_head_attribute_names_match_their_config_fields(self):
        """Live heads expose loss policies under the corresponding config names."""
        model, _ = self._built_model_without_reduction_fields()
        for task, field in (
            ("ner", "bio_loss_reduction"),
            ("classification", "loss_reduction"),
            ("joint_relex", "relation_loss_reduction"),
        ):
            assert hasattr(model.heads[task], field), (task, field)

    @pytest.mark.parametrize("rows", [1, 2, 5])
    def test_entity_pairs_divisor_includes_the_bn_row_count(self, rows):
        """The divisor sums over BN rows, so it scales with the flat batch.

        ``entity_pairs`` names only one of three factors; the denominator is
        sum over BN rows of e*(e-1) * C_rel, and this pins the BN factor that
        the mode's name does not mention.
        """
        n_rel, n_ent_slots, entities = 3, 6, 4

        losses = {}
        for reduction in ("sum", "entity_pairs"):
            torch.manual_seed(0)
            config = make_config(
                joint_relex_config=asdict(
                    JointRelexHeadConfig(relation_loss_reduction=reduction),
                ),
            )
            head = JointRelexHead.from_config(config)
            head.eval()

            torch.manual_seed(1)
            span_mask = torch.zeros(rows, n_ent_slots, dtype=torch.bool)
            span_mask[:, :entities] = True
            out = head._forward_relations(
                shared=None,
                target_span_rep=torch.randn(rows, n_ent_slots, D),
                target_span_mask=span_mask,
                rel_labels=torch.zeros(rows, n_ent_slots, n_ent_slots, n_rel),
                adjacency_threshold=0.5,
                rel_label_embeds=None,
                flat_rel_prompts=torch.randn(rows, n_rel, D),
                flat_rel_prompts_mask=torch.ones(rows, n_rel, dtype=torch.bool),
                base_loss_fn=_loss_fn,
            )
            losses[reduction] = out.loss.item()

        expected_divisor = rows * entities * (entities - 1) * n_rel
        assert losses["entity_pairs"] == pytest.approx(
            losses["sum"] / expected_divisor, rel=1e-5,
        )

    def test_entity_pairs_divisor_sums_heterogeneous_rows(self):
        """Rows differ in both e and C_rel; each contributes its own product."""
        n_ent_slots, n_rel = 7, 5
        per_row_entities = [5, 2]
        per_row_classes = [5, 3]
        assert len(per_row_entities) == B

        losses = {}
        for reduction in ("sum", "entity_pairs"):
            torch.manual_seed(0)
            config = make_config(
                joint_relex_config=asdict(
                    JointRelexHeadConfig(relation_loss_reduction=reduction),
                ),
            )
            head = JointRelexHead.from_config(config)
            head.eval()

            torch.manual_seed(1)
            span_mask = torch.zeros(B, n_ent_slots, dtype=torch.bool)
            prompts_mask = torch.zeros(B, n_rel, dtype=torch.bool)
            for row, (ents, cls) in enumerate(
                zip(per_row_entities, per_row_classes)
            ):
                span_mask[row, :ents] = True
                prompts_mask[row, :cls] = True

            out = head._forward_relations(
                shared=None,
                target_span_rep=torch.randn(B, n_ent_slots, D),
                target_span_mask=span_mask,
                rel_labels=torch.zeros(B, n_ent_slots, n_ent_slots, n_rel),
                adjacency_threshold=0.5,
                rel_label_embeds=None,
                flat_rel_prompts=torch.randn(B, n_rel, D),
                flat_rel_prompts_mask=prompts_mask,
                base_loss_fn=_loss_fn,
            )
            losses[reduction] = out.loss.item()

        expected_divisor = sum(
            ents * (ents - 1) * cls
            for ents, cls in zip(per_row_entities, per_row_classes)
        )
        assert losses["entity_pairs"] == pytest.approx(
            losses["sum"] / expected_divisor, rel=1e-5,
        )
