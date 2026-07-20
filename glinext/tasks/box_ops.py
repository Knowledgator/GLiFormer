"""Bounding-box conversions and overlap metrics shared by task heads."""

import torch


def _validate_last_dimension(boxes: torch.Tensor, name: str) -> None:
    if boxes.ndim == 0 or boxes.shape[-1] != 4:
        raise ValueError(f"{name} must have shape (..., 4), got {tuple(boxes.shape)}")


def _validate_pairwise_boxes(boxes: torch.Tensor, name: str) -> None:
    _validate_last_dimension(boxes, name)
    if boxes.ndim != 2:
        raise ValueError(f"{name} must have shape (N, 4), got {tuple(boxes.shape)}")


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert center-size boxes ``(..., cx, cy, w, h)`` to ``(..., x1, y1, x2, y2)``."""

    _validate_last_dimension(boxes, "boxes")
    centers = boxes[..., :2]
    half_sizes = boxes[..., 2:] / 2
    return torch.cat((centers - half_sizes, centers + half_sizes), dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert corner boxes ``(..., x1, y1, x2, y2)`` to ``(..., cx, cy, w, h)``."""

    _validate_last_dimension(boxes, "boxes")
    top_left = boxes[..., :2]
    bottom_right = boxes[..., 2:]
    sizes = bottom_right - top_left
    return torch.cat((top_left + sizes / 2, sizes), dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Return the non-negative area of ``xyxy`` boxes with arbitrary leading dimensions."""

    _validate_last_dimension(boxes, "boxes")
    return (boxes[..., 2:] - boxes[..., :2]).clamp(min=0).prod(dim=-1)


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    if not denominator.is_floating_point():
        denominator = denominator.float()
        numerator = numerator.float()
    tiny = torch.finfo(denominator.dtype).tiny
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(tiny),
        torch.zeros_like(numerator),
    )


def _pairwise_intersection_and_union(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    intersection_top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    intersection_bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (intersection_bottom_right - intersection_top_left).clamp(min=0).prod(dim=-1)
    union = box_area(boxes1)[:, None] + box_area(boxes2)[None, :] - intersection
    return intersection, union


def pairwise_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Return pairwise IoU for two sets of ``xyxy`` boxes as an ``(N, M)`` tensor."""

    _validate_pairwise_boxes(boxes1, "boxes1")
    _validate_pairwise_boxes(boxes2, "boxes2")
    intersection, union = _pairwise_intersection_and_union(boxes1, boxes2)
    return _safe_ratio(intersection, union)


def pairwise_generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Return pairwise generalized IoU for two sets of ``xyxy`` boxes."""

    _validate_pairwise_boxes(boxes1, "boxes1")
    _validate_pairwise_boxes(boxes2, "boxes2")
    intersection, union = _pairwise_intersection_and_union(boxes1, boxes2)
    iou = _safe_ratio(intersection, union)

    enclosing_top_left = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    enclosing_bottom_right = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclosing_area = (enclosing_bottom_right - enclosing_top_left).clamp(min=0).prod(dim=-1)
    penalty = _safe_ratio((enclosing_area - union).clamp(min=0), enclosing_area)
    return iou - penalty


def _validate_aligned_boxes(boxes1: torch.Tensor, boxes2: torch.Tensor) -> None:
    _validate_last_dimension(boxes1, "boxes1")
    _validate_last_dimension(boxes2, "boxes2")
    if boxes1.shape != boxes2.shape:
        raise ValueError(
            "aligned boxes must have identical shapes, "
            f"got {tuple(boxes1.shape)} and {tuple(boxes2.shape)}"
        )


def _aligned_intersection_and_union(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    intersection_top_left = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    intersection_bottom_right = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    intersection = (intersection_bottom_right - intersection_top_left).clamp(min=0).prod(dim=-1)
    union = box_area(boxes1) + box_area(boxes2) - intersection
    return intersection, union


def aligned_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Return elementwise IoU for equally shaped ``xyxy`` box tensors."""

    _validate_aligned_boxes(boxes1, boxes2)
    intersection, union = _aligned_intersection_and_union(boxes1, boxes2)
    return _safe_ratio(intersection, union)


def aligned_generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Return elementwise generalized IoU for equally shaped ``xyxy`` box tensors."""

    _validate_aligned_boxes(boxes1, boxes2)
    intersection, union = _aligned_intersection_and_union(boxes1, boxes2)
    iou = _safe_ratio(intersection, union)

    enclosing_top_left = torch.minimum(boxes1[..., :2], boxes2[..., :2])
    enclosing_bottom_right = torch.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclosing_area = (enclosing_bottom_right - enclosing_top_left).clamp(min=0).prod(dim=-1)
    penalty = _safe_ratio((enclosing_area - union).clamp(min=0), enclosing_area)
    return iou - penalty


__all__ = [
    "aligned_box_iou",
    "aligned_generalized_box_iou",
    "box_area",
    "box_cxcywh_to_xyxy",
    "box_xyxy_to_cxcywh",
    "pairwise_box_iou",
    "pairwise_generalized_box_iou",
]
