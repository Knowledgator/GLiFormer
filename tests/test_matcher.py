"""Tests for shared set-prediction matching utilities."""

import math

import pytest
import torch

from gliformer.tasks.matcher import (
    apply_objectness_cost,
    apply_slot_position_bias,
    center_distance_cost,
    minimum_cost_assignment,
    record_word_positions,
)


def _selected_rows(cost):
    return sorted(row for row, _ in minimum_cost_assignment(cost))


class TestSlotPositionBias:
    def test_zero_weight_is_identity(self):
        cost = torch.randn(5, 2)
        assert torch.equal(apply_slot_position_bias(cost, 0.0), cost)

    def test_identity_when_predictions_do_not_outnumber_gold(self):
        # Every prediction row is used, so a row-constant term cancels; the
        # helper skips it rather than perturbing the cost pointlessly.
        for shape in [(2, 2), (2, 4)]:
            cost = torch.randn(*shape)
            assert torch.equal(apply_slot_position_bias(cost, 5.0), cost)

    def test_penalty_is_row_constant_and_monotone(self):
        cost = torch.randn(4, 2)
        delta = apply_slot_position_bias(cost, 0.5, mode="relative") - cost
        assert torch.allclose(delta[0], torch.zeros(2), atol=1e-6)
        assert torch.allclose(
            delta[-1],
            0.5 * cost.std(unbiased=False) * torch.ones(2),
            atol=1e-6,
        )
        for row in delta:
            assert torch.allclose(row, row[0].expand(2), atol=1e-6)
        assert bool((delta[1:, 0] > delta[:-1, 0]).all())

    def test_absolute_mode_spans_exactly_the_weight(self):
        cost = torch.zeros(3, 1)
        biased = apply_slot_position_bias(cost, 2.0, mode="absolute")
        assert torch.allclose(
            biased.flatten(), torch.tensor([0.0, 1.0, 2.0])
        )

    def test_flat_cost_is_unchanged(self):
        cost = torch.full((4, 2), 3.0)
        assert torch.equal(apply_slot_position_bias(cost, 1.0), cost)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="relative"):
            apply_slot_position_bias(torch.randn(3, 1), 1.0, mode="nope")


class TestSlotPositionBiasAssignment:
    def test_square_assignment_is_invariant_to_the_bias(self):
        # The guarantee that keeps gold-order invariance: when every slot is
        # matched, the per-slot offsets sum to the same total for any
        # permutation, so the argmin cannot move.
        torch.manual_seed(0)
        for _ in range(50):
            cost = torch.randn(4, 4)
            biased = cost + (
                3.0 * cost.std(unbiased=False)
            ) * (torch.arange(4.0) / 3)[:, None]
            assert (
                minimum_cost_assignment(cost)
                == minimum_cost_assignment(biased)
            )

    def test_near_tie_resolves_toward_the_earlier_slot(self):
        cost = torch.tensor([[1.00], [0.99], [5.0], [6.0]])
        assert _selected_rows(cost) == [1]
        biased = apply_slot_position_bias(cost, 0.5, mode="relative")
        assert _selected_rows(biased) == [0]

    def test_clear_winner_survives_the_bias(self):
        cost = torch.tensor([[5.0], [0.1], [5.0], [6.0]])
        biased = apply_slot_position_bias(cost, 0.5, mode="relative")
        assert _selected_rows(biased) == [1]


class TestObjectnessCost:
    def test_zero_weight_is_identity(self):
        cost = torch.randn(5, 2)
        objectness = torch.randn(5)
        assert torch.equal(apply_objectness_cost(cost, objectness, 0.0), cost)

    def test_identity_when_predictions_do_not_outnumber_gold(self):
        # Row-constant, so it cancels once every slot is matched.
        for shape in [(3, 3), (3, 5)]:
            cost = torch.randn(*shape)
            objectness = torch.randn(shape[0])
            assert torch.equal(
                apply_objectness_cost(cost, objectness, 5.0), cost
            )

    def test_penalty_is_row_constant_and_follows_objectness(self):
        cost = torch.randn(4, 2)
        # Ascending objectness must produce a descending penalty.
        objectness = torch.tensor([-4.0, -1.0, 1.0, 4.0])
        delta = apply_objectness_cost(cost, objectness, 0.5) - cost
        for row in delta:
            assert torch.allclose(row, row[0].expand(2), atol=1e-6)
        assert bool((delta[1:, 0] < delta[:-1, 0]).all())
        assert bool((delta >= 0).all())

    def test_penalty_scales_with_the_cost_spread(self):
        cost = torch.randn(3, 1)
        objectness = torch.zeros(3)
        delta = apply_objectness_cost(cost, objectness, 2.0) - cost
        # sigmoid(0) = 0.5, so every slot pays half the full weight.
        assert torch.allclose(
            delta,
            torch.full_like(delta, 2.0 * 0.5 * cost.std(unbiased=False)),
            atol=1e-6,
        )

    def test_flat_cost_is_unchanged(self):
        cost = torch.full((4, 2), 3.0)
        assert torch.equal(
            apply_objectness_cost(cost, torch.randn(4), 1.0), cost
        )

    def test_mismatched_objectness_shape_raises(self):
        with pytest.raises(ValueError, match="one value per prediction row"):
            apply_objectness_cost(torch.randn(4, 1), torch.randn(3), 1.0)


class TestObjectnessCostAssignment:
    def test_square_assignment_is_invariant_to_the_cost(self):
        # Same guarantee as the position bias: a row-constant term cannot
        # reorder gold records across matched slots.
        torch.manual_seed(0)
        for _ in range(50):
            cost = torch.randn(4, 4)
            objectness = torch.randn(4)
            biased = cost + (
                3.0 * cost.std(unbiased=False)
            ) * (1.0 - torch.sigmoid(objectness))[:, None]
            assert (
                minimum_cost_assignment(cost)
                == minimum_cost_assignment(biased)
            )

    def test_near_tie_resolves_toward_the_confident_slot(self):
        # The duplicate-anchor case: two slots fit the record equally well and
        # only one of them predicts it holds a record.
        cost = torch.tensor([[1.00], [0.99], [5.0], [6.0]])
        assert _selected_rows(cost) == [1]
        objectness = torch.tensor([4.0, -4.0, 0.0, 0.0])
        assert _selected_rows(apply_objectness_cost(cost, objectness, 0.5)) == [0]

    def test_clear_membership_winner_survives_the_cost(self):
        cost = torch.tensor([[5.0], [0.1], [5.0], [6.0]])
        objectness = torch.tensor([4.0, -4.0, 0.0, 0.0])
        assert _selected_rows(apply_objectness_cost(cost, objectness, 0.5)) == [1]


def _membership(records):
    """Build ``(1, G, E)`` gold membership from record -> entity indices."""

    entity_count = 1 + max(e for entities in records for e in entities)
    gold = torch.zeros(1, len(records), entity_count)
    for record_idx, entities in enumerate(records):
        for entity_idx in entities:
            gold[0, record_idx, entity_idx] = 1.0
    return gold


def _spans(starts):
    return torch.tensor([[[start, start] for start in starts]])


class TestRecordWordPositions:
    def test_record_sits_at_its_earliest_field(self):
        gold = _membership([[1, 0], [3, 2]])
        positions = record_word_positions(gold, _spans([8, 11, 25, 28]))
        assert positions.tolist() == [[8.0, 25.0]]

    def test_field_order_within_a_record_does_not_matter(self):
        spans = _spans([8, 11, 25, 28])
        first = record_word_positions(_membership([[0, 1]]), spans)
        second = record_word_positions(_membership([[1, 0]]), spans)
        assert torch.equal(first, second)

    def test_invisible_record_is_nan(self):
        gold = torch.zeros(1, 2, 3)
        gold[0, 0, 0] = 1.0
        positions = record_word_positions(gold, _spans([4, 9, 15]))
        assert positions[0, 0].item() == 4.0
        assert math.isnan(positions[0, 1].item())

    def test_masked_entities_are_ignored(self):
        gold = _membership([[0, 1]])
        spans = _spans([8, 11])
        masked = record_word_positions(
            gold, spans, entity_mask=torch.tensor([[0, 1]])
        )
        assert masked.tolist() == [[11.0]]

    def test_rejects_wrong_shapes(self):
        with pytest.raises(ValueError, match=r"\(B, G, E\)"):
            record_word_positions(torch.zeros(2, 3), _spans([0]))
        with pytest.raises(ValueError, match=r"\(B, E, 2\)"):
            record_word_positions(torch.zeros(1, 1, 1), torch.zeros(1, 1, 3))


class TestCenterDistanceCost:
    def test_records_land_on_the_nearest_anchor(self):
        anchors = torch.arange(8.0) * 4  # stride 4
        records = torch.tensor([8.0, 25.0])
        matches = minimum_cost_assignment(center_distance_cost(anchors, records))
        assert sorted(matches, key=lambda pair: pair[1]) == [(2, 0), (6, 1)]

    def test_cost_ignores_membership_scores(self):
        # The whole point: the pairing cannot re-roll as training moves the
        # scores, because no score enters the cost.
        anchors = torch.arange(6.0) * 4
        records = torch.tensor([4.0, 16.0])
        first = center_distance_cost(anchors, records)
        second = center_distance_cost(anchors, records)
        assert torch.equal(first, second)

    def test_unlocated_record_takes_a_leftover_anchor(self):
        anchors = torch.arange(4.0) * 4
        records = torch.tensor([4.0, float("nan")])
        cost = center_distance_cost(anchors, records)
        assert torch.isfinite(cost).all()
        assert torch.equal(cost[:, 1], torch.zeros(4))
        matches = dict(
            (gold, prediction)
            for prediction, gold in minimum_cost_assignment(cost)
        )
        # The locatable record keeps its own anchor.
        assert matches[0] == 1

    def test_two_records_cannot_share_one_anchor(self):
        anchors = torch.arange(4.0) * 4
        records = torch.tensor([5.0, 6.0])
        matches = minimum_cost_assignment(center_distance_cost(anchors, records))
        assert len({prediction for prediction, _ in matches}) == 2

    def test_rejects_non_vector_positions(self):
        with pytest.raises(ValueError, match="1-D"):
            center_distance_cost(torch.zeros(2, 2), torch.zeros(2))
