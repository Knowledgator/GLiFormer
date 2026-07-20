"""Shared matching utilities for set-prediction task heads."""

from collections.abc import Callable

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from .box_ops import pairwise_generalized_box_iou

ClassProbability = str | Callable[[torch.Tensor], torch.Tensor]


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
        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
        return [
            (int(pred_idx[int(row)].item()), int(valid_idx[int(col)].item()))
            for row, col in zip(rows, cols, strict=True)
        ]
