"""Normalization and scope of the set-structuring membership loss."""

import pytest
import torch
from types import SimpleNamespace

from gliformer.config import StructuringHeadConfig
from gliformer.tasks.matcher import matched_prediction_mask
from gliformer.tasks.set_structuring.model import SetStructuringHead


def _reduce(losses, prediction_mask, entity_mask, reduction="mean"):
    return SetStructuringHead._reduce_membership_loss(
        torch.as_tensor(losses),
        torch.as_tensor(prediction_mask),
        torch.as_tensor(entity_mask),
        reduction,
    )


class TestMembershipLossReduction:
    def test_mean_divides_by_supervised_entities(self):
        # Row 0 keeps 3 of 4 slots and 2 of 3 entities; row 1 has no slot.
        losses = torch.ones(2, 4, 3)
        prediction_mask = torch.tensor(
            [[True, True, True, False], [False] * 4]
        )
        entity_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        # 6 supervised cells over 2 supervised entities.
        assert _reduce(losses, prediction_mask, entity_mask) == 3.0

    def test_rows_without_an_active_slot_are_ignored(self):
        losses = torch.ones(2, 4, 3)
        entity_mask = torch.ones(2, 3)
        both = torch.tensor([[True] * 4, [False] * 4])
        only_first = torch.tensor([[True] * 4])
        assert _reduce(losses, both, entity_mask) == _reduce(
            losses[:1], only_first, entity_mask[:1]
        )

    def test_padded_entities_are_excluded(self):
        losses = torch.ones(1, 2, 4)
        prediction_mask = torch.ones(1, 2, dtype=torch.bool)
        assert _reduce(
            losses, prediction_mask, torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        ) == 2.0

    def test_positive_weight_is_independent_of_the_slot_count(self):
        # One positive cell, nothing else. The cell mean shrank this with the
        # slot count; normalizing per entity must not.
        def one_positive(num_slots):
            losses = torch.zeros(1, num_slots, 2)
            losses[0, 0, 0] = 1.0
            return _reduce(
                losses,
                torch.ones(1, num_slots, dtype=torch.bool),
                torch.ones(1, 2),
            )

        assert one_positive(32) == one_positive(100)

    def test_negative_mass_still_grows_with_the_slot_count(self):
        # More slots is more to suppress, so the negative side must scale.
        def all_negative(num_slots):
            return _reduce(
                torch.ones(1, num_slots, 2),
                torch.ones(1, num_slots, dtype=torch.bool),
                torch.ones(1, 2),
            )

        assert all_negative(100) == 100.0
        assert all_negative(32) == 32.0

    def test_sum_reduction_is_unnormalized(self):
        losses = torch.ones(1, 4, 2)
        assert _reduce(
            losses,
            torch.ones(1, 4, dtype=torch.bool),
            torch.ones(1, 2),
            reduction="sum",
        ) == 8.0

    def test_no_supervised_entity_does_not_divide_by_zero(self):
        loss = _reduce(
            torch.ones(1, 4, 2),
            torch.zeros(1, 4, dtype=torch.bool),
            torch.zeros(1, 2),
        )
        assert torch.isfinite(loss) and loss == 0.0


class TestMatchedNegativeScope:
    """`assignment_negative_scope: matched` restricts the anchor axis."""

    ACTIVE = torch.ones(1, 5, dtype=torch.bool)
    MATCHES = [[(1, 0), (3, 1)]]  # slots 1 and 3 own a gold record

    def test_only_matched_slots_are_supervised(self):
        mask = matched_prediction_mask(self.MATCHES, self.ACTIVE)
        assert mask.tolist() == [[False, True, False, True, False]]

    def test_inactive_slots_are_never_revived(self):
        active = torch.tensor([[True, True, True, False, True]])
        mask = matched_prediction_mask(self.MATCHES, active)
        assert mask.tolist() == [[False, True, False, False, False]]

    def test_no_matches_supervises_nothing(self):
        assert not matched_prediction_mask([[]], self.ACTIVE).any()

    def test_positive_gradient_is_unchanged_by_the_scope(self):
        # The whole point: narrowing the scope must remove negatives only.
        # One positive on a matched slot, nothing else.
        losses = torch.zeros(1, 5, 2)
        losses[0, 1, 0] = 1.0
        entity_mask = torch.ones(1, 2)
        matched = matched_prediction_mask(self.MATCHES, self.ACTIVE)
        assert _reduce(losses, self.ACTIVE, entity_mask) == _reduce(
            losses, matched, entity_mask
        )

    def test_negative_mass_drops_to_the_matched_slots(self):
        losses = torch.ones(1, 5, 2)
        entity_mask = torch.ones(1, 2)
        matched = matched_prediction_mask(self.MATCHES, self.ACTIVE)
        assert _reduce(losses, self.ACTIVE, entity_mask) == 5.0
        assert _reduce(losses, matched, entity_mask) == 2.0

    def test_denominator_still_counts_supervised_entities(self):
        # Rows keep their entities as long as some slot owns a record, so the
        # per-positive scale set by `assignment_loss_coef` is preserved.
        losses = torch.ones(1, 5, 3)
        entity_mask = torch.tensor([[1.0, 1.0, 0.0]])
        matched = matched_prediction_mask(self.MATCHES, self.ACTIVE)
        # 2 matched slots x 2 valid entities, over 2 supervised entities.
        assert _reduce(losses, matched, entity_mask) == 2.0


class TestObjectnessFocalOverride:
    """`anchor_objectness_focal_loss_*` is scoped to the objectness head."""

    def test_override_is_collected_under_its_own_prefix(self):
        from gliformer.tasks.losses import focal_override_kwargs

        config = SimpleNamespace(
            anchor_objectness_focal_loss_alpha=0.9,
            anchor_objectness_focal_loss_gamma=None,
            anchor_objectness_focal_loss_prob_margin=None,
            focal_loss_alpha=0.75,
        )
        assert focal_override_kwargs(config, "anchor_objectness") == {
            "focal_loss_alpha": 0.9
        }

    def test_unset_override_leaves_the_task_level_loss_untouched(self):
        from gliformer.tasks.losses import focal_override_kwargs, with_focal_overrides

        config = SimpleNamespace(
            anchor_objectness_focal_loss_alpha=None,
            anchor_objectness_focal_loss_gamma=None,
            anchor_objectness_focal_loss_prob_margin=None,
        )
        overrides = focal_override_kwargs(config, "anchor_objectness")
        assert overrides == {}

        def base(logits, targets, **kwargs):
            return kwargs

        assert with_focal_overrides(base, overrides) is base

    def test_override_reaches_the_loss_function(self):
        from gliformer.tasks.losses import with_focal_overrides

        def base(logits, targets, **kwargs):
            return kwargs

        wrapped = with_focal_overrides(base, {"focal_loss_alpha": 0.9})
        assert wrapped(None, None)["focal_loss_alpha"] == 0.9

    def test_alpha_above_one_is_rejected_by_config(self):
        with pytest.raises(ValueError, match="anchor_objectness_focal_loss_alpha"):
            StructuringHeadConfig(anchor_objectness_focal_loss_alpha=1.5)
