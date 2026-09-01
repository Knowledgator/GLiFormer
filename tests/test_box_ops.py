import pytest
import torch

from glinext.tasks.box_ops import (
    aligned_box_iou,
    aligned_generalized_box_iou,
    box_area,
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    pairwise_box_iou,
    pairwise_generalized_box_iou,
)


def test_box_coordinate_conversions_round_trip():
    cxcywh = torch.tensor(
        [
            [0.50, 0.50, 0.40, 0.20],
            [0.25, 0.75, 0.10, 0.30],
        ]
    )

    xyxy = box_cxcywh_to_xyxy(cxcywh)

    torch.testing.assert_close(
        xyxy,
        torch.tensor([[0.30, 0.40, 0.70, 0.60], [0.20, 0.60, 0.30, 0.90]]),
    )
    torch.testing.assert_close(box_xyxy_to_cxcywh(xyxy), cxcywh)


def test_box_area_clamps_inverted_sides():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 2.0, 3.0],
            [2.0, 0.0, 1.0, 3.0],
        ]
    )

    torch.testing.assert_close(box_area(boxes), torch.tensor([6.0, 0.0]))


def test_pairwise_box_iou_and_giou():
    boxes1 = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.5, 0.5]])
    boxes2 = torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]])

    torch.testing.assert_close(
        pairwise_box_iou(boxes1, boxes2),
        torch.tensor([[1.0, 0.0], [0.25, 0.0]]),
    )
    torch.testing.assert_close(
        pairwise_generalized_box_iou(boxes1, boxes2),
        torch.tensor([[1.0, -0.5], [0.25, -0.6875]]),
    )


def test_aligned_box_iou_and_giou_preserve_leading_dimensions():
    boxes1 = torch.tensor(
        [
            [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5]],
        ]
    )
    boxes2 = torch.tensor(
        [
            [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]],
            [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]],
        ]
    )

    torch.testing.assert_close(
        aligned_box_iou(boxes1, boxes2),
        torch.tensor([[1.0, 0.0], [0.25, 0.0]]),
    )
    torch.testing.assert_close(
        aligned_generalized_box_iou(boxes1, boxes2),
        torch.tensor([[1.0, -0.5], [0.25, -0.6875]]),
    )


def test_box_metrics_handle_empty_and_degenerate_boxes_without_nan():
    empty = torch.empty(0, 4)
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    degenerate = torch.zeros(2, 4)

    assert pairwise_box_iou(empty, boxes).shape == (0, 1)
    assert pairwise_generalized_box_iou(boxes, empty).shape == (1, 0)
    assert torch.isfinite(aligned_box_iou(degenerate, degenerate)).all()
    assert torch.isfinite(aligned_generalized_box_iou(degenerate, degenerate)).all()


def test_generalized_box_iou_is_differentiable():
    boxes1 = torch.tensor([[0.1, 0.1, 0.6, 0.6]], requires_grad=True)
    boxes2 = torch.tensor([[0.2, 0.2, 0.8, 0.8]])

    aligned_generalized_box_iou(boxes1, boxes2).sum().backward()

    assert boxes1.grad is not None
    assert torch.isfinite(boxes1.grad).all()
    assert boxes1.grad.abs().sum() > 0


def test_box_ops_validate_box_shapes():
    with pytest.raises(ValueError, match=r"\(\.\.\., 4\)"):
        box_area(torch.zeros(3, 2))
    with pytest.raises(ValueError, match="identical shapes"):
        aligned_box_iou(torch.zeros(2, 4), torch.zeros(3, 4))
    with pytest.raises(ValueError, match=r"\(N, 4\)"):
        pairwise_box_iou(torch.zeros(1, 2, 4), torch.zeros(3, 4))
