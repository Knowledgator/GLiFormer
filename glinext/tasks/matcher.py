"""Shared matching utilities for set-prediction task heads."""

from collections.abc import Callable

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from .box_ops import pairwise_generalized_box_iou

ClassProbability = str | Callable[[torch.Tensor], torch.Tensor]
AnchorMatches = list[list[tuple[int, int]]]
PairCostBuilder = Callable[
    [int, torch.Tensor, torch.Tensor],
    torch.Tensor,
]


@torch.no_grad()
def minimum_cost_assignment(cost: torch.Tensor) -> list[tuple[int, int]]:
    """Return a minimum-cost one-to-one assignment for a rectangular matrix.

    The result contains ``min(rows, columns)`` ``(row, column)`` pairs. Cost
    construction stays with each task, while dtype conversion and the CPU
    linear-assignment boundary are shared here.
    """

    if cost.dim() != 2:
        raise ValueError("Hungarian cost must be a two-dimensional matrix")
    if cost.shape[0] == 0 or cost.shape[1] == 0:
        return []

    cost = cost.detach().float()
    if not torch.isfinite(cost).all():
        raise ValueError("Hungarian cost matrix must contain only finite values")
    rows, cols = linear_sum_assignment(cost.cpu().numpy())
    return [
        (int(row), int(col))
        for row, col in zip(rows, cols, strict=True)
    ]


def gold_anchor_mask(
    labels: torch.Tensor,
    label_count: torch.Tensor | list[int] | int | None,
    *,
    anchor_dim: int = 1,
) -> torch.Tensor:
    """Return the valid-gold mask shared by set-prediction heads."""

    labels = labels.movedim(anchor_dim, 1)
    batch_size, anchor_count = labels.shape[:2]
    if label_count is None:
        return (
            labels.detach().abs().reshape(batch_size, anchor_count, -1)
            .sum(dim=-1)
            > 0
        )
    if not torch.is_tensor(label_count):
        label_count = torch.as_tensor(label_count, device=labels.device)
    if label_count.dim() == 0:
        label_count = label_count.unsqueeze(0).expand(batch_size)
    if label_count.numel() != batch_size:
        raise ValueError(
            "label_count must contain one value per batch item"
        )
    label_count = label_count.to(device=labels.device).long().clamp(
        min=0,
        max=anchor_count,
    )
    return (
        torch.arange(anchor_count, device=labels.device).unsqueeze(0)
        < label_count.unsqueeze(1)
    )


@torch.no_grad()
def batched_masked_assignment(
    prediction_mask: torch.Tensor,
    gold_mask: torch.Tensor,
    cost_builder: PairCostBuilder,
) -> AnchorMatches:
    """Run rectangular matching for each batch using task-specific costs."""

    if prediction_mask.dim() != 2 or gold_mask.dim() != 2:
        raise ValueError("prediction and gold masks must have shape (B, A)")
    if prediction_mask.shape[0] != gold_mask.shape[0]:
        raise ValueError("prediction and gold masks must share a batch axis")

    matches: AnchorMatches = [[] for _ in range(prediction_mask.shape[0])]
    for batch_idx in range(prediction_mask.shape[0]):
        prediction_ids = torch.where(prediction_mask[batch_idx].bool())[0]
        gold_ids = torch.where(gold_mask[batch_idx].bool())[0]
        if prediction_ids.numel() == 0 or gold_ids.numel() == 0:
            continue
        pair_cost = cost_builder(batch_idx, prediction_ids, gold_ids)
        expected = (prediction_ids.numel(), gold_ids.numel())
        if tuple(pair_cost.shape) != expected:
            raise ValueError(
                f"pair-cost builder returned {tuple(pair_cost.shape)}, "
                f"expected {expected}"
            )
        matches[batch_idx] = [
            (
                int(prediction_ids[predicted_idx].item()),
                int(gold_ids[gold_idx].item()),
            )
            for predicted_idx, gold_idx in minimum_cost_assignment(pair_cost)
        ]
    return matches


def matched_anchor_targets(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    matches: AnchorMatches,
) -> torch.Tensor:
    """Scatter gold records onto matched prediction slots."""

    targets = torch.zeros_like(predictions)
    for batch_idx, batch_matches in enumerate(matches):
        for predicted_anchor, gold_anchor in batch_matches:
            targets[batch_idx, predicted_anchor] = labels[
                batch_idx,
                gold_anchor,
            ]
    return targets


def matched_objectness_loss(
    logits: torch.Tensor,
    matches: AnchorMatches | None,
    anchor_mask: torch.Tensor,
    *,
    gold_mask: torch.Tensor | None = None,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Normalized objectness loss for matched or positional anchors."""

    if loss_fn is None:
        from .losses import binary_focal_or_bce

        loss_fn = binary_focal_or_bce
    targets = torch.zeros_like(logits)
    if matches is not None:
        for batch_idx, batch_matches in enumerate(matches):
            for predicted_anchor, _ in batch_matches:
                if predicted_anchor < targets.shape[1]:
                    targets[batch_idx, predicted_anchor] = 1.0
    elif gold_mask is not None:
        anchor_count = min(targets.shape[1], gold_mask.shape[1])
        targets[:, :anchor_count] = gold_mask[:, :anchor_count].to(
            targets.dtype
        )
    losses = loss_fn(logits.float(), targets.float())
    mask = anchor_mask.to(losses.dtype)
    return (losses * mask).sum() / mask.sum().clamp(min=1.0)


class HungarianMatcher(nn.Module):
    """Match anchor-slot predictions to labeled targets with Hungarian assignment.

    The matcher is used by image and audio set-prediction heads, where each
    anchor predicts a class score vector and a geometry vector such as a box or
    temporal segment. It returns prediction indices paired with the original
    target indices from the padded target tensor.
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_geometry: float = 1.0,
        cost_giou: float = 0.0,
        class_probability: ClassProbability = "sigmoid",
    ):
        super().__init__()
        self.cost_class = float(cost_class)
        self.cost_geometry = float(cost_geometry)
        self.cost_giou = float(cost_giou)
        if isinstance(class_probability, str):
            class_probability = class_probability.lower()
            if class_probability not in {"sigmoid", "softmax"}:
                raise ValueError(
                    "class_probability must be 'sigmoid', 'softmax', or a callable, "
                    f"got {class_probability!r}"
                )
        elif not callable(class_probability):
            raise TypeError(
                "class_probability must be 'sigmoid', 'softmax', or a callable, "
                f"got {type(class_probability).__name__}"
            )
        self.class_probability = class_probability
        if self.cost_class == 0.0 and self.cost_geometry == 0.0 and self.cost_giou == 0.0:
            raise ValueError("at least one Hungarian matching cost must be non-zero")

    def _class_probabilities(self, class_logits: torch.Tensor) -> torch.Tensor:
        if self.class_probability == "sigmoid":
            probabilities = class_logits.sigmoid()
        elif self.class_probability == "softmax":
            probabilities = class_logits.softmax(dim=-1)
        else:
            probabilities = self.class_probability(class_logits)
        if not isinstance(probabilities, torch.Tensor):
            raise TypeError("class probability callback must return a torch.Tensor")
        if probabilities.shape != class_logits.shape:
            raise ValueError(
                "class probability callback must preserve the logits shape, "
                f"got {tuple(probabilities.shape)} for logits {tuple(class_logits.shape)}"
            )
        return probabilities

    @torch.no_grad()
    def forward(
        self,
        class_logits: torch.Tensor,
        geometry_preds: torch.Tensor,
        target_classes: torch.Tensor,
        target_geometry: torch.Tensor,
        target_mask: torch.Tensor,
        prediction_mask: torch.Tensor | None = None,
        giou_geometry_preds: torch.Tensor | None = None,
        target_giou_geometry: torch.Tensor | None = None,
    ) -> list[tuple[int, int]]:
        """Return ``(prediction_index, target_index)`` matches for one sample."""

        valid_idx = torch.nonzero(target_mask > 0, as_tuple=False).squeeze(-1)
        if prediction_mask is None:
            pred_idx = torch.arange(class_logits.shape[0], device=class_logits.device)
        else:
            pred_idx = torch.nonzero(prediction_mask > 0, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0 or pred_idx.numel() == 0:
            return []

        # Matching is non-differentiable and ends in a NumPy linear-assignment,
        # so resolve to float32: bf16 has no cdist kernel and no NumPy dtype.
        class_logits = class_logits[pred_idx].float()
        geometry_preds = geometry_preds[pred_idx].float()
        target_classes = target_classes[valid_idx].long()
        target_geometry = target_geometry[valid_idx].float()
        if giou_geometry_preds is not None:
            giou_geometry_preds = giou_geometry_preds[pred_idx].float()
        if target_giou_geometry is not None:
            target_giou_geometry = target_giou_geometry[valid_idx].float()

        # Independent sigmoid probabilities are the default for multi-label and
        # open-vocabulary tasks. Mutually exclusive heads can explicitly request
        # softmax, while specialized heads may supply a probability callback.
        probs = self._class_probabilities(class_logits)
        class_cost = probs.new_zeros((probs.shape[0], target_classes.numel()))
        valid_classes = (target_classes >= 0) & (target_classes < probs.shape[1])
        if valid_classes.any():
            class_cost[:, valid_classes] = -probs[:, target_classes[valid_classes]]

        if geometry_preds.dim() != 2 or target_geometry.dim() != 2:
            raise ValueError(
                "set-prediction geometry must have shape (items, dimensions)"
            )
        if geometry_preds.shape[-1] != target_geometry.shape[-1]:
            raise ValueError(
                "prediction and target geometry dimensions must match"
            )
        geometry_cost = torch.cdist(geometry_preds, target_geometry, p=1)
        if self.cost_giou != 0.0 and geometry_preds.shape[-1] == 4:
            # Negative GIoU as a cost: better-overlapping pairs are cheaper.
            # Pairs the boxes that L1 alone leaves ambiguous for small objects.
            giou_predictions = (
                giou_geometry_preds
                if giou_geometry_preds is not None
                else geometry_preds
            )
            giou_targets = (
                target_giou_geometry
                if target_giou_geometry is not None
                else target_geometry
            )
            giou_cost = -pairwise_generalized_box_iou(
                giou_predictions,
                giou_targets,
            )
        else:
            giou_cost = geometry_cost.new_zeros(geometry_cost.shape)
        cost = (
            self.cost_class * class_cost
            + self.cost_geometry * geometry_cost
            + self.cost_giou * giou_cost
        )
        assignment = minimum_cost_assignment(cost)
        return [
            (int(pred_idx[row].item()), int(valid_idx[col].item()))
            for row, col in assignment
        ]
