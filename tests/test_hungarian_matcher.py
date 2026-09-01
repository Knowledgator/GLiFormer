import pytest
import torch

from glinext.tasks.matcher import HungarianMatcher, minimum_cost_assignment


@pytest.mark.parametrize(
    ("cost", "expected"),
    [
        (
            torch.tensor([[5.0, 1.0, 4.0], [2.0, 3.0, 0.0]]),
            [(0, 1), (1, 2)],
        ),
        (
            torch.tensor([[5.0, 1.0], [2.0, 3.0], [0.0, 4.0]]),
            [(0, 1), (2, 0)],
        ),
    ],
)
def test_minimum_cost_assignment_supports_rectangular_costs(cost, expected):
    assert minimum_cost_assignment(cost) == expected


def test_minimum_cost_assignment_rejects_non_finite_costs():
    with pytest.raises(ValueError, match="finite"):
        minimum_cost_assignment(torch.tensor([[float("nan")]]))


def test_hungarian_matcher_returns_original_padded_target_indices():
    matcher = HungarianMatcher(cost_class=0.0, cost_geometry=1.0)
    class_logits = torch.zeros(3, 2)
    geometry_preds = torch.tensor(
        [
            [0.80, 0.90],
            [0.10, 0.20],
            [0.40, 0.40],
        ]
    )
    target_classes = torch.tensor([0, 0, 1])
    target_geometry = torch.tensor(
        [
            [0.00, 0.00],
            [0.10, 0.19],
            [0.79, 0.91],
        ]
    )
    target_mask = torch.tensor([0, 1, 1])

    matches = matcher(
        class_logits,
        geometry_preds,
        target_classes,
        target_geometry,
        target_mask,
    )

    assert matches == [(0, 2), (1, 1)]


def test_hungarian_matcher_handles_empty_targets():
    matcher = HungarianMatcher()

    matches = matcher(
        torch.zeros(3, 2),
        torch.zeros(3, 4),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, 4),
        torch.zeros(2),
    )

    assert matches == []


def test_hungarian_matcher_respects_prediction_mask():
    matcher = HungarianMatcher(cost_class=0.0, cost_geometry=1.0)

    matches = matcher(
        torch.zeros(3, 2),
        torch.tensor(
            [
                [0.10, 0.10],
                [0.90, 0.90],
                [0.11, 0.11],
            ]
        ),
        torch.tensor([0]),
        torch.tensor([[0.10, 0.10]]),
        torch.ones(1),
        prediction_mask=torch.tensor([0, 1, 1]),
    )

    assert matches == [(2, 0)]


def test_hungarian_matcher_uses_sigmoid_class_probabilities_by_default():
    matcher = HungarianMatcher(cost_class=1.0, cost_geometry=0.0)

    matches = matcher(
        # Query 0 has the higher independent class-0 probability even though
        # class 1 competes closely. Multi-label matching must still select it.
        torch.tensor([[10.0, 9.0], [2.0, -2.0]]),
        torch.zeros(2, 4),
        torch.tensor([0]),
        torch.zeros(1, 4),
        torch.ones(1),
    )

    assert matches == [(0, 0)]


def test_hungarian_matcher_can_use_softmax_class_probabilities():
    matcher = HungarianMatcher(
        cost_class=1.0,
        cost_geometry=0.0,
        class_probability="softmax",
    )

    matches = matcher(
        # Query 0 has the highest absolute class-0 logit, but class 1 competes
        # closely. Query 1 has the higher softmax probability for class 0.
        torch.tensor([[10.0, 9.0], [2.0, -2.0]]),
        torch.zeros(2, 4),
        torch.tensor([0]),
        torch.zeros(1, 4),
        torch.ones(1),
    )

    assert matches == [(1, 0)]


def test_hungarian_matcher_accepts_class_probability_callback():
    matcher = HungarianMatcher(
        cost_class=1.0,
        cost_geometry=0.0,
        class_probability=lambda logits: -logits,
    )

    matches = matcher(
        torch.tensor([[0.0], [1.0]]),
        torch.zeros(2, 4),
        torch.tensor([0]),
        torch.zeros(1, 4),
        torch.ones(1),
    )

    assert matches == [(0, 0)]


@pytest.mark.parametrize("class_probability", ["unknown", 1])
def test_hungarian_matcher_rejects_invalid_class_probability(class_probability):
    with pytest.raises((TypeError, ValueError), match="class_probability"):
        HungarianMatcher(class_probability=class_probability)


def test_hungarian_matcher_rejects_obsolete_class_conditioned_geometry():
    matcher = HungarianMatcher(cost_class=0.0, cost_geometry=1.0)
    class_logits = torch.zeros(2, 2)
    geometry_preds = torch.tensor(
        [
            [[0.90, 0.90], [0.10, 0.10]],
            [[0.10, 0.10], [0.90, 0.90]],
        ]
    )

    with pytest.raises(ValueError, match="geometry must have shape"):
        matcher(
            class_logits,
            geometry_preds,
            torch.tensor([1]),
            torch.tensor([[0.10, 0.10]]),
            torch.ones(1),
        )


def test_hungarian_matcher_uses_shared_pairwise_giou_cost():
    matcher = HungarianMatcher(cost_class=0.0, cost_geometry=0.0, cost_giou=1.0)

    matches = matcher(
        torch.zeros(2, 1),
        # Both predictions have L1 distance 1 from the target, but the first
        # has a higher GIoU because it encloses the complete target box.
        torch.tensor([[-1.0, 0.0, 1.0, 1.0], [0.5, 0.5, 1.0, 1.0]]),
        torch.tensor([0]),
        torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        torch.ones(1),
    )

    assert matches == [(0, 0)]


def test_hungarian_matcher_can_use_separate_l1_and_giou_box_formats():
    matcher = HungarianMatcher(cost_class=0.0, cost_geometry=0.0, cost_giou=1.0)

    matches = matcher(
        torch.zeros(2, 1),
        # L1 geometry may be cxcywh while GIoU always receives xyxy corners.
        torch.zeros(2, 4),
        torch.tensor([0]),
        torch.zeros(1, 4),
        torch.ones(1),
        giou_geometry_preds=torch.tensor(
            [[0.0, 0.0, 1.0, 1.0], [0.8, 0.8, 1.0, 1.0]]
        ),
        target_giou_geometry=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
    )

    assert matches == [(0, 0)]
