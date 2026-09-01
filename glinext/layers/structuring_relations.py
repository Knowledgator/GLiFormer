"""Shared anchor-relation layer utilities for structuring heads."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import torch
from gliner.modeling.multitask.relations_layers import RelationsRepLayer
from torch import nn

from ..tasks.losses import (
    binary_focal_or_bce,
    binary_loss_with_focal_overrides,
)

AnchorMatches = Sequence[Sequence[tuple[int, int]]]


def validate_structuring_anchor_capacity(
    labels,
    label_count,
    anchor_count,
    *,
    anchor_dim,
    task_name="Structuring",
):
    """Warn when supervision cannot fit in the predicted record slots."""

    max_gold = None
    if label_count is not None and label_count.numel() > 0:
        observed = int(label_count.detach().max().item())
        if observed > anchor_count:
            max_gold = observed

    if (
        max_gold is None
        and labels is not None
        and labels.shape[anchor_dim] > anchor_count
    ):
        tail = labels.narrow(
            anchor_dim,
            anchor_count,
            labels.shape[anchor_dim] - anchor_count,
        )
        if torch.count_nonzero(tail).item() > 0:
            max_gold = labels.shape[anchor_dim]

    if max_gold is not None:
        excess = max_gold - anchor_count
        warnings.warn(
            f"{task_name} supervision contains {max_gold} visible records, "
            f"but the head produced only {anchor_count} record anchors. "
            f"The {excess} excess record(s) will not contribute to this "
            "batch's loss. Increase anchor_layer.params.num_slots/max_count "
            "to train on all records.",
            RuntimeWarning,
            stacklevel=2,
        )


def initialize_anchor_relations(
    owner: nn.Module,
    task_config,
    hidden_size: int,
) -> None:
    """Attach the optional hierarchy scorer without introducing a wrapper module.

    Checkpoints already store the learned scorer directly below
    ``anchor_relations_rep_layer`` on each structuring head.  Keeping this
    helper stateless consolidates construction while preserving that exact
    registered-module path.
    """

    owner.multi_level = bool(getattr(task_config, "multi_level", False))
    owner.anchor_relations_loss_coef = getattr(
        task_config,
        "anchor_relations_loss_coef",
        1.0,
    )
    for suffix in ("alpha", "gamma", "prob_margin"):
        name = f"anchor_relations_focal_loss_{suffix}"
        setattr(owner, name, getattr(task_config, name, None))
    if owner.multi_level:
        owner.anchor_relations_rep_layer = RelationsRepLayer(
            in_dim=hidden_size,
            relation_mode=task_config.anchor_relations_layer,
            hidden_dim=hidden_size,
        )


def score_anchor_relations(
    relation_layer: nn.Module | None,
    anchors: torch.Tensor,
    anchor_mask: torch.Tensor,
    *,
    compact: bool = False,
) -> torch.Tensor | None:
    """Score directed anchor pairs, optionally compacting sparse slot masks.

    Entity-first structuring compacts active record slots because selected
    token anchors may be sparse, then scatters both relation axes back to their
    stable public slot indices.
    """

    if relation_layer is None:
        return None
    if not compact:
        return relation_layer(anchors, anchor_mask)

    batch_size, anchor_count, hidden_size = anchors.shape
    active_rows = torch.where(anchor_mask.bool().any(dim=1))[0]
    if active_rows.numel() == 0:
        return anchors.new_zeros(batch_size, anchor_count, anchor_count)

    active_mask = anchor_mask[active_rows].bool()
    compact_count = int(active_mask.sum(dim=1).max().item())
    compact_indices = torch.zeros(
        active_rows.numel(),
        compact_count,
        dtype=torch.long,
        device=anchors.device,
    )
    compact_mask = torch.zeros_like(compact_indices, dtype=torch.bool)
    for compact_batch_idx, source_batch_idx in enumerate(active_rows.tolist()):
        indices = torch.where(anchor_mask[source_batch_idx].bool())[0]
        compact_indices[compact_batch_idx, :indices.numel()] = indices
        compact_mask[compact_batch_idx, :indices.numel()] = True

    compact_anchors = anchors[active_rows].gather(
        1,
        compact_indices.unsqueeze(-1).expand(-1, -1, hidden_size),
    )
    compact_scores = relation_layer(compact_anchors, compact_mask)
    compact_scores = compact_scores * (
        compact_mask.unsqueeze(1) & compact_mask.unsqueeze(2)
    ).to(compact_scores.dtype)

    column_scattered = compact_scores.new_zeros(
        active_rows.numel(),
        compact_count,
        anchor_count,
    ).scatter_add(
        2,
        compact_indices.unsqueeze(1).expand(-1, compact_count, -1),
        compact_scores,
    )
    active_scores = compact_scores.new_zeros(
        active_rows.numel(),
        anchor_count,
        anchor_count,
    ).scatter_add(
        1,
        compact_indices.unsqueeze(-1).expand(-1, -1, anchor_count),
        column_scattered,
    )
    return compact_scores.new_zeros(
        batch_size,
        anchor_count,
        anchor_count,
    ).index_copy(0, active_rows, active_scores)


def maybe_anchor_relation_loss(
    relation_scores: torch.Tensor | None,
    relation_labels: torch.Tensor | None,
    anchor_mask: torch.Tensor,
    *,
    base_loss_fn=None,
    relation_group_mask: torch.Tensor | None = None,
    anchor_matches: AnchorMatches | None = None,
    label_count: torch.Tensor | None = None,
    loss_coef: float = 1.0,
    focal_loss_alpha: float | None = None,
    focal_loss_gamma: float | None = None,
    focal_loss_prob_margin: float | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return the raw and weighted optional hierarchy losses."""

    if (
        relation_scores is None
        or relation_labels is None
        or base_loss_fn is None
    ):
        return None, None
    relation_loss = anchor_relation_loss(
        relation_scores,
        relation_labels,
        anchor_mask,
        base_loss_fn=base_loss_fn,
        relation_group_mask=relation_group_mask,
        anchor_matches=anchor_matches,
        label_count=label_count,
        focal_loss_alpha=focal_loss_alpha,
        focal_loss_gamma=focal_loss_gamma,
        focal_loss_prob_margin=focal_loss_prob_margin,
    )
    return relation_loss, float(loss_coef) * relation_loss


def remap_anchor_relation_targets(
    relation_scores: torch.Tensor,
    relation_labels: torch.Tensor,
    anchor_mask: torch.Tensor,
    *,
    relation_group_mask: torch.Tensor | None = None,
    anchor_matches: AnchorMatches | None = None,
    label_count: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map gold adjacency onto predicted anchor slots.

    Hungarian matching is permutation-invariant only when the same assignment
    is applied to both axes of the gold adjacency matrix. When an assignment is
    available, relations are conditional on matched record existence, so
    unmatched prediction slots are excluded as endpoints. Without an explicit
    assignment, all active slots remain eligible. Padding, disabled groups, and
    self-pairs are always excluded from the objective.
    """

    if relation_scores.dim() != 3 or relation_scores.shape[1] != relation_scores.shape[2]:
        raise ValueError("relation_scores must have shape (batch, anchors, anchors)")
    if relation_labels.dim() != 3 or relation_labels.shape[1] != relation_labels.shape[2]:
        raise ValueError("relation_labels must have shape (batch, anchors, anchors)")

    batch_size, prediction_count, _ = relation_scores.shape
    if relation_labels.shape[0] != batch_size:
        raise ValueError(
            "relation_labels and relation_scores must have the same batch size"
        )
    if anchor_mask.dim() != 2 or anchor_mask.shape != (
        batch_size,
        prediction_count,
    ):
        raise ValueError(
            "anchor_mask must have shape (batch, predicted anchors)"
        )
    if anchor_matches is not None and len(anchor_matches) != batch_size:
        raise ValueError("anchor_matches must contain one assignment per group")

    relation_labels = relation_labels.to(device=relation_scores.device)
    gold_count = relation_labels.shape[1]
    target = relation_scores.new_zeros(relation_scores.shape)
    eligible_endpoint_mask = anchor_mask.to(
        device=relation_scores.device,
        dtype=torch.bool,
    )
    endpoint_mask = (
        eligible_endpoint_mask.clone()
        if anchor_matches is None
        else torch.zeros_like(eligible_endpoint_mask)
    )

    if relation_group_mask is not None:
        group_mask = relation_group_mask.to(
            device=relation_scores.device,
            dtype=torch.bool,
        ).reshape(-1)
        if group_mask.numel() != batch_size:
            raise ValueError("relation_group_mask must contain one value per group")
    else:
        group_mask = torch.ones(
            batch_size,
            dtype=torch.bool,
            device=relation_scores.device,
        )

    normalized_count = None
    if label_count is not None:
        normalized_count = torch.as_tensor(
            label_count,
            device=relation_scores.device,
        ).long().reshape(-1)
        if normalized_count.numel() == 1 and batch_size != 1:
            normalized_count = normalized_count.expand(batch_size)
        if normalized_count.numel() != batch_size:
            raise ValueError("label_count must contain one value per group")

    for batch_idx in range(batch_size):
        if not group_mask[batch_idx]:
            continue

        if anchor_matches is not None:
            matched_pairs = anchor_matches[batch_idx]
        else:
            positional_count = min(prediction_count, gold_count)
            if normalized_count is not None:
                positional_count = min(
                    positional_count,
                    max(0, int(normalized_count[batch_idx].item())),
                )
            matched_pairs = [(idx, idx) for idx in range(positional_count)]

        gold_to_prediction = {}
        for predicted_anchor, gold_anchor in matched_pairs:
            predicted_anchor = int(predicted_anchor)
            gold_anchor = int(gold_anchor)
            if (
                0 <= predicted_anchor < prediction_count
                and 0 <= gold_anchor < gold_count
            ):
                gold_to_prediction[gold_anchor] = predicted_anchor

        if not gold_to_prediction:
            continue
        predicted_indices = tuple(gold_to_prediction.values())
        gold_indices = torch.tensor(
            tuple(gold_to_prediction),
            device=relation_scores.device,
            dtype=torch.long,
        )
        predicted_indices = torch.tensor(
            predicted_indices,
            device=relation_scores.device,
            dtype=torch.long,
        )
        valid_endpoints = eligible_endpoint_mask[batch_idx].index_select(
            0, predicted_indices,
        )
        gold_indices = gold_indices[valid_endpoints]
        predicted_indices = predicted_indices[valid_endpoints]
        if predicted_indices.numel() == 0:
            continue
        if predicted_indices.unique().numel() != predicted_indices.numel():
            raise ValueError(
                "anchor_matches must assign every predicted anchor at most once"
            )
        if anchor_matches is not None:
            endpoint_mask[batch_idx, predicted_indices] = True
        gold_subgraph = relation_labels[batch_idx].index_select(
            0, gold_indices,
        ).index_select(1, gold_indices)
        target[batch_idx][
            predicted_indices[:, None], predicted_indices[None, :]
        ] = gold_subgraph.to(target.dtype)

    pair_mask = endpoint_mask.unsqueeze(2) & endpoint_mask.unsqueeze(1)
    pair_mask &= group_mask[:, None, None]
    diagonal = torch.eye(
        prediction_count,
        dtype=torch.bool,
        device=relation_scores.device,
    ).unsqueeze(0)
    pair_mask &= ~diagonal
    return target, pair_mask


def anchor_relation_loss(
    relation_scores: torch.Tensor,
    relation_labels: torch.Tensor,
    anchor_mask: torch.Tensor,
    *,
    base_loss_fn=None,
    relation_group_mask: torch.Tensor | None = None,
    anchor_matches: AnchorMatches | None = None,
    label_count: torch.Tensor | None = None,
    focal_loss_alpha: float | None = None,
    focal_loss_gamma: float | None = None,
    focal_loss_prob_margin: float | None = None,
) -> torch.Tensor:
    """Return mean loss over valid directed, non-self anchor pairs.

    For ``n`` eligible anchors the denominator is ``n * (n - 1)``. It is not
    merely ``n`` because every ordered source/target decision is independently
    supervised, and it is not ``n**2`` because self-relations are masked out.
    """

    targets, pair_mask = remap_anchor_relation_targets(
        relation_scores,
        relation_labels,
        anchor_mask,
        relation_group_mask=relation_group_mask,
        anchor_matches=anchor_matches,
        label_count=label_count,
    )
    loss_fn = base_loss_fn or binary_focal_or_bce
    # GLiNER's relation layer already applies sigmoid.  Require the loss
    # callable to acknowledge that contract explicitly instead of silently
    # retrying after any TypeError (which could hide an error inside the loss).
    losses = binary_loss_with_focal_overrides(
        loss_fn,
        relation_scores.float(),
        targets.float(),
        focal_loss_alpha=focal_loss_alpha,
        focal_loss_gamma=focal_loss_gamma,
        focal_loss_prob_margin=focal_loss_prob_margin,
        normalize_prob=False,
    )
    if losses.shape != relation_scores.shape:
        raise ValueError(
            "anchor relation loss must return one value per relation score; "
            "configure it with reduction='none'"
        )
    mask = pair_mask.to(losses.dtype)
    return (losses * mask).sum() / mask.sum().clamp(min=1.0)


__all__ = [
    "anchor_relation_loss",
    "initialize_anchor_relations",
    "maybe_anchor_relation_loss",
    "remap_anchor_relation_targets",
    "score_anchor_relations",
    "validate_structuring_anchor_capacity",
]
