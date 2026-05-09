"""Shared matching utilities for set-prediction task heads."""

from typing import List, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """Match anchor-slot predictions to labeled targets with Hungarian assignment.

    The matcher is used by image and audio set-prediction heads, where each
    anchor predicts a class score vector and a geometry vector such as a box or
    temporal segment. It returns prediction indices paired with the original
    target indices from the padded target tensor.
    """

    def __init__(self, cost_class: float = 1.0, cost_geometry: float = 1.0):
        super().__init__()
        self.cost_class = float(cost_class)
        self.cost_geometry = float(cost_geometry)
        if self.cost_class == 0.0 and self.cost_geometry == 0.0:
            raise ValueError("at least one Hungarian matching cost must be non-zero")

    @torch.no_grad()
    def forward(
        self,
        class_logits: torch.Tensor,
        geometry_preds: torch.Tensor,
        target_classes: torch.Tensor,
        target_geometry: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> List[Tuple[int, int]]:
        """Return ``(prediction_index, target_index)`` matches for one sample."""

        valid_idx = torch.nonzero(target_mask > 0, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0 or class_logits.shape[0] == 0:
            return []

        target_classes = target_classes[valid_idx].long()
        target_geometry = target_geometry[valid_idx]

        probs = class_logits.sigmoid()
        class_cost = probs.new_zeros((probs.shape[0], target_classes.numel()))
        valid_classes = (target_classes >= 0) & (target_classes < probs.shape[1])
        if valid_classes.any():
            class_cost[:, valid_classes] = -probs[:, target_classes[valid_classes]]

        geometry_cost = torch.cdist(geometry_preds, target_geometry, p=1)
        cost = self.cost_class * class_cost + self.cost_geometry * geometry_cost
        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
        return [(int(row), int(valid_idx[int(col)].item())) for row, col in zip(rows, cols)]
