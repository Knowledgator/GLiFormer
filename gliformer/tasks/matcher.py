"""Shared matching utilities for set-prediction task heads."""

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from .box_ops import aligned_generalized_box_iou

ClassProbability = str | Callable[[torch.Tensor], torch.Tensor]
AnchorMatches = list[list[tuple[int, int]]]
PairCostBuilder = Callable[
    [int, torch.Tensor, torch.Tensor],
    torch.Tensor,
]

_NON_FINITE_COST = "Hungarian cost matrix must contain only finite values"


def host_to_device(array: np.ndarray, device: torch.device | str) -> torch.Tensor:
    """Copy a host index array to ``device`` without a device synchronization.

    A plain ``torch.tensor(..., device="cuda")`` copy blocks the host until the
    GPU stream has drained. Pinned memory plus ``non_blocking`` only enqueues
    the copy, so matching results can flow back to the GPU while it is busy.
    """

    tensor = torch.from_numpy(np.ascontiguousarray(array))
    device = torch.device(device)
    if device.type == "cpu":
        return tensor
    if device.type == "cuda":
        tensor = tensor.pin_memory()
    return tensor.to(device, non_blocking=True)


def match_index_array(matches: Sequence[Sequence[tuple[int, int]]]) -> np.ndarray:
    """Flatten per-group ``(prediction, target)`` pairs into a ``(3, M)`` array.

    Rows are group index, prediction index and target index, in the original
    group-major, pair order.
    """

    groups = []
    pairs = []
    for group, group_matches in enumerate(matches):
        for prediction, target in group_matches:
            groups.append(group)
            pairs.append((int(prediction), int(target)))
    if not pairs:
        return np.zeros((3, 0), dtype=np.int64)
    flat = np.empty((3, len(pairs)), dtype=np.int64)
    flat[0] = groups
    flat[1:] = np.asarray(pairs, dtype=np.int64).T
    return flat


def match_index_tensors(
    matches: Sequence[Sequence[tuple[int, int]]] | np.ndarray,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(group, prediction, target)`` index tensors on ``device``.

    One asynchronous host-to-device copy replaces per-pair Python indexing.
    """

    flat = matches if isinstance(matches, np.ndarray) else match_index_array(matches)
    flat = host_to_device(flat, device)
    return flat[0], flat[1], flat[2]


def _assign_numpy(cost: np.ndarray) -> list[tuple[int, int]]:
    if cost.shape[0] == 0 or cost.shape[1] == 0:
        return []
    if not np.isfinite(cost).all():
        raise ValueError(_NON_FINITE_COST)
    rows, cols = linear_sum_assignment(cost)
    return [
        (int(row), int(col))
        for row, col in zip(rows, cols, strict=True)
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
    return _assign_numpy(cost.detach().float().cpu().numpy())


@torch.no_grad()
def batched_minimum_cost_assignment(
    cost: torch.Tensor,
    prediction_mask: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
) -> AnchorMatches:
    """Match every ``(N, P, T)`` cost slice restricted to its masked rows/columns.

    The whole padded cost and both masks cross the device boundary in a single
    copy; the per-slice linear assignments then run purely in NumPy. Returned
    pairs use the padded (unmasked) prediction and target indices, exactly as
    matching each compacted sub-matrix and mapping the indices back would.
    Only the selected sub-matrices must be finite.
    """

    if cost.dim() != 3:
        raise ValueError("batched Hungarian cost must have shape (N, P, T)")
    count, predictions, targets = cost.shape
    if prediction_mask is None:
        prediction_mask = torch.ones(
            count, predictions, dtype=torch.bool, device=cost.device,
        )
    if target_mask is None:
        target_mask = torch.ones(
            count, targets, dtype=torch.bool, device=cost.device,
        )
    if tuple(prediction_mask.shape) != (count, predictions):
        raise ValueError("prediction_mask must have shape (N, P)")
    if tuple(target_mask.shape) != (count, targets):
        raise ValueError("target_mask must have shape (N, T)")
    if count == 0:
        return []

    packed = torch.cat(
        [
            cost.detach().float().reshape(count, predictions * targets),
            (prediction_mask.detach() > 0).to(
                device=cost.device, dtype=torch.float32,
            ),
            (target_mask.detach() > 0).to(
                device=cost.device, dtype=torch.float32,
            ),
        ],
        dim=1,
    ).cpu().numpy()
    split = predictions * targets
    host_cost = packed[:, :split].reshape(count, predictions, targets)
    host_predictions = packed[:, split:split + predictions] > 0
    host_targets = packed[:, split + predictions:] > 0

    matches: AnchorMatches = []
    for index in range(count):
        rows = np.flatnonzero(host_predictions[index])
        cols = np.flatnonzero(host_targets[index])
        if rows.size == 0 or cols.size == 0:
            matches.append([])
            continue
        assignment = _assign_numpy(host_cost[index][np.ix_(rows, cols)])
        matches.append([
            (int(rows[row]), int(cols[col]))
            for row, col in assignment
        ])
    return matches


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
    """Run rectangular matching for each batch using task-specific costs.

    The masks are read once and every group's cost is gathered into a single
    device-to-host copy, so the number of synchronizations does not grow with
    the batch. Heads that can build all group costs at once should prefer
    :func:`batched_minimum_cost_assignment`.
    """

    if prediction_mask.dim() != 2 or gold_mask.dim() != 2:
        raise ValueError("prediction and gold masks must have shape (B, A)")
    if prediction_mask.shape[0] != gold_mask.shape[0]:
        raise ValueError("prediction and gold masks must share a batch axis")

    batch_size = prediction_mask.shape[0]
    matches: AnchorMatches = [[] for _ in range(batch_size)]
    if batch_size == 0:
        return matches
    prediction_width = prediction_mask.shape[1]
    host_masks = torch.cat(
        [
            prediction_mask.detach().bool().to(gold_mask.device),
            gold_mask.detach().bool(),
        ],
        dim=1,
    ).cpu().numpy()

    pending = []
    costs = []
    for batch_idx in range(batch_size):
        prediction_rows = np.flatnonzero(host_masks[batch_idx, :prediction_width])
        gold_rows = np.flatnonzero(host_masks[batch_idx, prediction_width:])
        if prediction_rows.size == 0 or gold_rows.size == 0:
            continue
        prediction_ids = host_to_device(prediction_rows, prediction_mask.device)
        gold_ids = host_to_device(gold_rows, gold_mask.device)
        pair_cost = cost_builder(batch_idx, prediction_ids, gold_ids)
        expected = (prediction_rows.size, gold_rows.size)
        if tuple(pair_cost.shape) != expected:
            raise ValueError(
                f"pair-cost builder returned {tuple(pair_cost.shape)}, "
                f"expected {expected}"
            )
        pending.append((batch_idx, prediction_rows, gold_rows))
        costs.append(pair_cost.detach().float().reshape(-1))
    if not costs:
        return matches

    host_costs = torch.cat(
        [cost.to(costs[0].device) for cost in costs]
    ).cpu().numpy()
    offset = 0
    for batch_idx, prediction_rows, gold_rows in pending:
        size = prediction_rows.size * gold_rows.size
        cost = host_costs[offset:offset + size].reshape(
            prediction_rows.size, gold_rows.size,
        )
        offset += size
        matches[batch_idx] = [
            (int(prediction_rows[row]), int(gold_rows[col]))
            for row, col in _assign_numpy(cost)
        ]
    return matches


def matched_anchor_targets(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    matches: AnchorMatches,
) -> torch.Tensor:
    """Scatter gold records onto matched prediction slots."""

    targets = torch.zeros_like(predictions)
    batch_idx, predicted_anchor, gold_anchor = match_index_tensors(
        matches, predictions.device,
    )
    if batch_idx.numel() > 0:
        # A one-to-one assignment never writes a slot twice.
        targets[batch_idx, predicted_anchor] = labels[
            batch_idx.to(labels.device),
            gold_anchor.to(labels.device),
        ].to(device=targets.device, dtype=targets.dtype)
    return targets


def matched_prediction_mask(
    reference: torch.Tensor,
    matches: AnchorMatches | None,
) -> torch.Tensor:
    """Return a ``reference``-shaped boolean ``(B, A)`` mask of matched slots.

    Pairs whose prediction index lies outside ``A`` are ignored.
    """

    mask = torch.zeros(reference.shape[:2], dtype=torch.bool, device=reference.device)
    if matches is None:
        return mask
    flat = match_index_array(matches)
    flat = flat[:, (flat[1] >= 0) & (flat[1] < mask.shape[1])]
    if flat.shape[1] > 0:
        batch_idx, predicted_anchor, _ = match_index_tensors(flat, mask.device)
        mask[batch_idx, predicted_anchor] = True
    return mask


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
    if matches is not None:
        targets = matched_prediction_mask(logits, matches).to(logits.dtype)
    else:
        targets = torch.zeros_like(logits)
        if gold_mask is not None:
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
    def pairwise_cost(
        self,
        class_logits: torch.Tensor,
        geometry_preds: torch.Tensor,
        target_classes: torch.Tensor,
        target_geometry: torch.Tensor,
        giou_geometry_preds: torch.Tensor | None = None,
        target_giou_geometry: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the padded ``(N, P, T)`` matching cost for a whole batch.

        Every entry equals the cost the single-sample matcher computes for the
        same prediction/target pair; masking happens in the assignment step.
        """

        if geometry_preds.dim() != 3 or target_geometry.dim() != 3:
            raise ValueError(
                "set-prediction geometry must have shape (items, dimensions)"
            )
        if geometry_preds.shape[-1] != target_geometry.shape[-1]:
            raise ValueError(
                "prediction and target geometry dimensions must match"
            )
        # Matching is non-differentiable and ends in a NumPy linear-assignment,
        # so resolve to float32: bf16 has no cdist kernel and no NumPy dtype.
        class_logits = class_logits.float()
        geometry_preds = geometry_preds.float()
        target_classes = target_classes.long()
        target_geometry = target_geometry.float()
        count, predictions, class_count = class_logits.shape
        targets = target_classes.shape[1]

        # Independent sigmoid probabilities are the default for multi-label and
        # open-vocabulary tasks. Mutually exclusive heads can explicitly request
        # softmax, while specialized heads may supply a probability callback.
        probs = self._class_probabilities(
            class_logits.reshape(count * predictions, class_count)
        ).reshape(count, predictions, class_count)
        valid_classes = (target_classes >= 0) & (target_classes < class_count)
        if class_count > 0:
            gathered = probs.gather(
                2,
                target_classes.clamp(0, class_count - 1)
                .unsqueeze(1)
                .expand(-1, predictions, -1),
            )
            class_cost = torch.where(
                valid_classes.unsqueeze(1),
                -gathered,
                torch.zeros_like(gathered),
            )
        else:
            class_cost = probs.new_zeros(count, predictions, targets)

        geometry_cost = torch.cdist(geometry_preds, target_geometry, p=1)
        if self.cost_giou != 0.0 and geometry_preds.shape[-1] == 4:
            # Negative GIoU as a cost: better-overlapping pairs are cheaper.
            # Pairs the boxes that L1 alone leaves ambiguous for small objects.
            giou_predictions = (
                giou_geometry_preds
                if giou_geometry_preds is not None
                else geometry_preds
            ).float()
            giou_targets = (
                target_giou_geometry
                if target_giou_geometry is not None
                else target_geometry
            ).float()
            shape = (count, predictions, targets, 4)
            giou_cost = -aligned_generalized_box_iou(
                giou_predictions.unsqueeze(2).expand(shape),
                giou_targets.unsqueeze(1).expand(shape),
            )
        else:
            giou_cost = geometry_cost.new_zeros(geometry_cost.shape)
        return (
            self.cost_class * class_cost
            + self.cost_geometry * geometry_cost
            + self.cost_giou * giou_cost
        )

    @torch.no_grad()
    def match_batch(
        self,
        class_logits: torch.Tensor,
        geometry_preds: torch.Tensor,
        target_classes: torch.Tensor,
        target_geometry: torch.Tensor,
        target_mask: torch.Tensor,
        prediction_mask: torch.Tensor | None = None,
        giou_geometry_preds: torch.Tensor | None = None,
        target_giou_geometry: torch.Tensor | None = None,
    ) -> AnchorMatches:
        """Batched :meth:`forward`: one ``(prediction, target)`` list per sample."""

        cost = self.pairwise_cost(
            class_logits,
            geometry_preds,
            target_classes,
            target_geometry,
            giou_geometry_preds=giou_geometry_preds,
            target_giou_geometry=target_giou_geometry,
        )
        return batched_minimum_cost_assignment(
            cost,
            None if prediction_mask is None else prediction_mask > 0,
            target_mask > 0,
        )

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

        if geometry_preds.dim() != 2 or target_geometry.dim() != 2:
            raise ValueError(
                "set-prediction geometry must have shape (items, dimensions)"
            )

        def batched(value):
            return None if value is None else value.unsqueeze(0)

        return self.match_batch(
            class_logits.unsqueeze(0),
            geometry_preds.unsqueeze(0),
            target_classes.unsqueeze(0),
            target_geometry.unsqueeze(0),
            target_mask.unsqueeze(0),
            batched(prediction_mask),
            giou_geometry_preds=batched(giou_geometry_preds),
            target_giou_geometry=batched(target_giou_geometry),
        )[0]
