"""Shared matching utilities for set-prediction task heads."""

from typing import List, Optional, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


def _pairwise_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise GIoU between ``(N, 4)`` and ``(M, 4)`` xyxy boxes -> ``(N, M)``."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=-1)
    union = area1[:, None] + area2[None, :] - inter + 1e-6
    iou = inter / union
    elt = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    erb = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclosing = (erb - elt).clamp(min=0).prod(dim=-1) + 1e-6
    return iou - (enclosing - union) / enclosing


class HungarianMatcher(nn.Module):
    """Match anchor-slot predictions to labeled targets with Hungarian assignment.

    The matcher is used by image and audio set-prediction heads, where each
    anchor predicts a class score vector and a geometry vector such as a box or
    temporal segment. It returns prediction indices paired with the original
    target indices from the padded target tensor.
    """

    def __init__(self, cost_class: float = 1.0, cost_geometry: float = 1.0, cost_giou: float = 0.0):
        super().__init__()
        self.cost_class = float(cost_class)
        self.cost_geometry = float(cost_geometry)
        self.cost_giou = float(cost_giou)
        if self.cost_class == 0.0 and self.cost_geometry == 0.0 and self.cost_giou == 0.0:
            raise ValueError("at least one Hungarian matching cost must be non-zero")

    @torch.no_grad()
    def forward(
        self,
        class_logits: torch.Tensor,
        geometry_preds: torch.Tensor,
        target_classes: torch.Tensor,
        target_geometry: torch.Tensor,
        target_mask: torch.Tensor,
        prediction_mask: Optional[torch.Tensor] = None,
    ) -> List[Tuple[int, int]]:
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

        # Detection heads train matched queries with softmax cross-entropy, and
        # inference uses the same softmax distribution. Matching must use that
        # distribution as well; sigmoid ignores competition between labels and
        # can assign a query whose target logit is high but not its winning class.
        probs = class_logits.softmax(dim=-1)
        class_cost = probs.new_zeros((probs.shape[0], target_classes.numel()))
        valid_classes = (target_classes >= 0) & (target_classes < probs.shape[1])
        if valid_classes.any():
            class_cost[:, valid_classes] = -probs[:, target_classes[valid_classes]]

        if geometry_preds.dim() == 3:
            num_classes = geometry_preds.shape[1]
            geometry_cost = geometry_preds.new_zeros((geometry_preds.shape[0], target_classes.numel()))
            valid_geometry_classes = (target_classes >= 0) & (target_classes < num_classes)
            if valid_geometry_classes.any():
                target_cls = target_classes[valid_geometry_classes]
                pred_geometry = geometry_preds[:, target_cls, :]
                geometry_cost[:, valid_geometry_classes] = (
                    pred_geometry - target_geometry[valid_geometry_classes].unsqueeze(0)
                ).abs().sum(dim=-1)
            giou_cost = geometry_cost.new_zeros(geometry_cost.shape)
        else:
            geometry_cost = torch.cdist(geometry_preds, target_geometry, p=1)
            if self.cost_giou != 0.0 and geometry_preds.shape[-1] == 4:
                # Negative GIoU as a cost: better-overlapping pairs are cheaper.
                # Pairs the boxes that L1 alone leaves ambiguous for small objects.
                giou_cost = -_pairwise_giou(geometry_preds, target_geometry)
            else:
                giou_cost = geometry_cost.new_zeros(geometry_cost.shape)
        cost = (
            self.cost_class * class_cost
            + self.cost_geometry * geometry_cost
            + self.cost_giou * giou_cost
        )
        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
        return [
            (int(pred_idx[int(row)].item()), int(valid_idx[int(col)].item()))
            for row, col in zip(rows, cols)
        ]
