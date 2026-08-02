"""Directed parent-child relation utilities for structuring anchors."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..losses import binary_focal_or_bce

AnchorMatches = Sequence[Sequence[tuple[int, int]]]


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
) -> torch.Tensor:
    """Return mean elementwise loss for directed anchor adjacency probabilities."""

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
    losses = loss_fn(
        relation_scores.float(),
        targets.float(),
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
    "remap_anchor_relation_targets",
]
