"""Typed structuring fields: class options and quantity distributions.

For every (record slot, field) pair the module builds one vector: a query
made from the record anchor and the field marker embedding reads the text
through one cross-attention layer.  Two small heads use that vector:

* class fields score each ``[OPTION]`` embedding of their field (plus a
  learned "none" score for single-choice fields);
* quantity fields predict a histogram over ``num_bins`` bins of the unit
  coordinate, an offset inside every bin, and a presence logit.  The anchor
  span (when the field tagged one) and the origin mode join the query, and
  forecast steps add a step embedding.

Losses are averaged inside each (record, field) unit first and then over the
supervised units, so fields with many options or long horizons do not
dominate, and batches with few typed fields do not shrink the loss.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ...layers.mlp import create_mlp
from ...processing.typed_fields import (
    ORIGIN_ABSOLUTE,
    ORIGIN_ANCHOR,
    ORIGIN_REFERENCE,
)


def target_cdf(
    levels: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    num_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Target CDF at the right edge of each bin, vectorized.

    ``levels``/``values``/``mask`` are ``(..., P)`` quantile levels, unit
    values, and validity.  One point (or points with one value) is a point
    mass; more points are joined linearly with tails extended at the edge
    slopes and clipped to ``[0, 1]``.  Returns ``(cdf (..., K), valid (...))``.
    """

    mask = mask.bool()
    count = mask.sum(-1)
    valid = count > 0
    sentinel = 2.0
    sorted_values, _ = values.masked_fill(~mask, sentinel).sort(-1)
    sorted_levels, _ = levels.masked_fill(~mask, sentinel).sort(-1)
    edges = torch.arange(
        1, num_bins + 1, device=values.device, dtype=values.dtype
    ) / num_bins
    edges = edges.expand(*values.shape[:-1], num_bins).contiguous()

    last_index = (count - 1).clamp(min=0).unsqueeze(-1)
    first = sorted_values[..., 0]
    last = sorted_values.gather(-1, last_index).squeeze(-1)
    middle = sorted_values.gather(-1, (count // 2).clamp(min=0).unsqueeze(-1).clamp(
        max=values.shape[-1] - 1
    )).squeeze(-1)
    point_case = (count <= 1) | (last - first < 1e-6)
    step = (edges >= (middle.unsqueeze(-1) - 1e-12)).to(values.dtype)

    points = values.shape[-1]
    if points < 2:
        return step, valid
    index = torch.searchsorted(sorted_values.contiguous(), edges)
    left = (index - 1).clamp(min=0)
    left = torch.minimum(left, (count - 2).clamp(min=0).unsqueeze(-1))
    right = (left + 1).clamp(max=points - 1)
    u_left = sorted_values.gather(-1, left)
    u_right = sorted_values.gather(-1, right)
    q_left = sorted_levels.gather(-1, left)
    q_right = sorted_levels.gather(-1, right)
    slope = (q_right - q_left) / (u_right - u_left).clamp(min=1e-6)
    interpolated = (q_left + slope * (edges - u_left)).clamp(0.0, 1.0)
    cdf = torch.where(point_case.unsqueeze(-1), step, interpolated)
    return cdf, valid


def _masked_unit_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the last axis inside each unit, then over valid units."""

    mask = mask.to(values.dtype)
    per_unit = (values * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
    has_unit = (mask.sum(-1) > 0).to(values.dtype)
    return (per_unit * has_unit).sum() / has_unit.sum().clamp(min=1.0)


class TypedFieldModule(nn.Module):
    """Class and quantity heads over (record, field) representations."""

    def __init__(
        self,
        hidden_size: int,
        dropout: float,
        *,
        num_bins: int = 256,
        num_heads: int = 8,
        max_horizon: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_bins = int(num_bins)
        self.max_horizon = int(max_horizon)
        heads = int(num_heads)
        if heads < 1 or hidden_size % heads != 0:
            heads = 1
        self.query = create_mlp(
            input_dim=3 * hidden_size,
            intermediate_dims=[hidden_size],
            output_dim=hidden_size,
            dropout=dropout,
            activation="gelu",
        )
        self.readout = nn.MultiheadAttention(
            hidden_size,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.readout_norm = nn.LayerNorm(hidden_size)
        self.readout_ffn = create_mlp(
            input_dim=hidden_size,
            intermediate_dims=[2 * hidden_size],
            output_dim=hidden_size,
            dropout=dropout,
            activation="gelu",
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

        self.class_query = nn.Linear(hidden_size, hidden_size)
        self.class_key = nn.Linear(hidden_size, hidden_size)
        self.class_none = nn.Linear(hidden_size, 1)

        self.anchor_proj = nn.Linear(hidden_size, hidden_size)
        self.null_anchor = nn.Parameter(torch.zeros(hidden_size))
        self.origin_embedding = nn.Embedding(3, hidden_size)
        self.step_embedding = nn.Embedding(self.max_horizon, hidden_size)
        self.quantity_head = create_mlp(
            input_dim=hidden_size,
            intermediate_dims=[hidden_size],
            output_dim=2 * self.num_bins,
            dropout=dropout,
            activation="gelu",
        )
        self.presence_head = nn.Linear(hidden_size, 1)

    # -- representations -----------------------------------------------------

    def record_field_vectors(
        self,
        records: torch.Tensor,
        fields: torch.Tensor,
        words: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> torch.Tensor:
        """``(N, A, D)`` records x ``(N, C, D)`` fields -> ``(N, A, C, D)``."""

        batch, num_records, dim = records.shape
        num_fields = fields.shape[1]
        if num_records == 0 or num_fields == 0:
            return records.new_zeros(batch, num_records, num_fields, dim)
        record = records.unsqueeze(2).expand(-1, -1, num_fields, -1)
        field = fields.unsqueeze(1).expand(-1, num_records, -1, -1)
        query = self.query(torch.cat([record, field, record * field], dim=-1))
        flat = query.reshape(batch, num_records * num_fields, dim)

        key_padding = ~word_mask.bool()
        if words.shape[1] == 0:
            return query
        # Rows without visible words would make attention undefined.
        empty = key_padding.all(dim=-1)
        if empty.any():
            key_padding = key_padding.clone()
            key_padding[empty, 0] = False
        attended, _ = self.readout(
            flat,
            words,
            words,
            key_padding_mask=key_padding,
            need_weights=False,
        )
        hidden = self.readout_norm(flat + attended)
        hidden = self.ffn_norm(hidden + self.readout_ffn(hidden))
        return hidden.reshape(batch, num_records, num_fields, dim)

    # -- class fields ------------------------------------------------------

    def class_logits(
        self,
        vectors: torch.Tensor,
        option_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(N, A, Cc, D)`` x ``(N, Cc, Kc, D)`` -> logits and none logits."""

        query = self.class_query(vectors)
        key = self.class_key(option_embeddings)
        logits = torch.einsum("nacd,nckd->nack", query, key) / math.sqrt(
            self.hidden_size
        )
        none = self.class_none(vectors).squeeze(-1)
        return logits, none

    # -- quantity fields -----------------------------------------------------

    def quantity_outputs(
        self,
        vectors: torch.Tensor,
        anchor_repr: torch.Tensor,
        anchor_present: torch.Tensor,
        origin_mode: torch.Tensor,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Histogram logits, in-bin offsets (0-1), and presence logits.

        ``vectors``/``anchor_repr`` are ``(N, A, Cq, D)``; outputs are
        ``(N, A, Cq, H, K)`` twice and ``(N, A, Cq)``.
        """

        anchor = torch.where(
            anchor_present.unsqueeze(-1),
            self.anchor_proj(anchor_repr),
            self.null_anchor.to(vectors.dtype).expand_as(vectors),
        )
        base = vectors + anchor + self.origin_embedding(origin_mode.clamp(0, 2))
        presence = self.presence_head(base).squeeze(-1)
        horizon = max(1, min(int(horizon), self.max_horizon))
        steps = self.step_embedding.weight[:horizon].to(base.dtype)
        stepped = base.unsqueeze(-2) + steps
        raw = self.quantity_head(stepped)
        logits, offsets = raw.split(self.num_bins, dim=-1)
        return logits, torch.sigmoid(offsets), presence

    # -- losses ------------------------------------------------------------

    @staticmethod
    def class_loss(
        logits: torch.Tensor,
        none_logits: torch.Tensor,
        targets: torch.Tensor,
        option_mask: torch.Tensor,
        unit_mask: torch.Tensor,
        multi: torch.Tensor,
        binary_loss_fn,
    ) -> torch.Tensor:
        """Per-unit mean over options, then mean over supervised units.

        ``logits``/``targets`` ``(G, R, Cc, Kc)``, ``option_mask``
        ``(G, Cc, Kc)``, ``unit_mask`` ``(G, R, Cc)``, ``multi`` ``(G, Cc)``.
        """

        unit = unit_mask.bool()
        if not unit.any():
            return logits.sum() * 0.0
        options = option_mask.bool().unsqueeze(1).expand_as(logits)
        element = unit.unsqueeze(-1) & options
        elementwise = binary_loss_fn(logits, targets)
        element_f = element.to(elementwise.dtype)
        multi_unit = (elementwise * element_f).sum(-1) / element_f.sum(-1).clamp(min=1.0)

        # Single choice: softmax over this field's options plus "none".
        none_target = (targets * options.to(targets.dtype)).sum(-1).eq(0).to(targets.dtype)
        full_logits = torch.cat(
            [logits.masked_fill(~options, -1e4), none_logits.unsqueeze(-1)],
            dim=-1,
        )
        full_targets = torch.cat(
            [targets * options.to(targets.dtype), none_target.unsqueeze(-1)],
            dim=-1,
        )
        log_probs = F.log_softmax(full_logits.float(), dim=-1).to(full_logits.dtype)
        single_unit = -(full_targets * log_probs).sum(-1) / full_targets.sum(-1).clamp(min=1.0)

        is_multi = multi.bool().unsqueeze(1).expand_as(unit)
        unit_loss = torch.where(is_multi, multi_unit, single_unit)
        unit_f = unit.to(unit_loss.dtype)
        return (unit_loss * unit_f).sum() / unit_f.sum().clamp(min=1.0)

    def quantity_loss(
        self,
        logits: torch.Tensor,
        offsets: torch.Tensor,
        presence_logits: torch.Tensor,
        *,
        levels: torch.Tensor,
        values: torch.Tensor,
        point_mask: torch.Tensor,
        point: torch.Tensor,
        point_value_mask: torch.Tensor,
        presence: torch.Tensor,
        unit_mask: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Distribution (CDF), point, and presence losses.

        Shapes: ``logits``/``offsets`` ``(G, R, Cq, H, K)``; ``levels``/
        ``values``/``point_mask`` ``(G, R, Cq, H, P)``; ``point``/
        ``point_value_mask``/``step_mask`` ``(G, R, Cq, H)``; ``presence``/
        ``unit_mask`` ``(G, R, Cq)``.
        """

        unit = unit_mask.bool()
        zero = logits.sum() * 0.0
        if not unit.any():
            return zero, zero, zero
        num_bins = logits.shape[-1]
        present = unit & presence.gt(0.5)

        cdf_target, has_points = target_cdf(
            levels.to(logits.dtype),
            values.to(logits.dtype),
            point_mask,
            num_bins,
        )
        probs = F.softmax(logits.float(), dim=-1)
        cdf = probs.cumsum(-1).to(logits.dtype)
        crps = (cdf - cdf_target).pow(2).mean(-1)
        dist_mask = present.unsqueeze(-1) & has_points & step_mask.bool()
        dist_loss = _masked_unit_mean(crps, dist_mask)

        point_mask_steps = present.unsqueeze(-1) & point_value_mask.bool() & step_mask.bool()
        scaled = (point.to(logits.dtype) * num_bins).clamp(0.0, num_bins - 1e-4)
        target_bin = scaled.floor().long().clamp(0, num_bins - 1)
        fraction = scaled - target_bin.to(scaled.dtype)
        point_ce = F.cross_entropy(
            logits.float().reshape(-1, num_bins),
            target_bin.reshape(-1),
            reduction="none",
        ).reshape(target_bin.shape).to(logits.dtype)
        chosen_offset = offsets.gather(-1, target_bin.unsqueeze(-1)).squeeze(-1)
        offset_error = (chosen_offset - fraction).abs()
        point_loss = _masked_unit_mean(point_ce + offset_error, point_mask_steps)

        presence_losses = F.binary_cross_entropy_with_logits(
            presence_logits.float(),
            presence.to(torch.float32),
            reduction="none",
        ).to(logits.dtype)
        unit_f = unit.to(presence_losses.dtype)
        presence_loss = (presence_losses * unit_f).sum() / unit_f.sum().clamp(min=1.0)
        return dist_loss, point_loss, presence_loss


def gold_record_anchors(
    record_anchors: torch.Tensor,
    anchor_mask: torch.Tensor,
    matches,
    num_gold: int,
    gold_count: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the record slot matched to every gold record.

    Returns ``(G, R, D)`` anchors and a ``(G, R)`` mask.  Without Hungarian
    matches (for example, a batch with no span targets) gold record ``r`` uses
    slot ``r``.
    """

    groups, slots, dim = record_anchors.shape
    index = torch.zeros(groups, num_gold, dtype=torch.long, device=record_anchors.device)
    mask = torch.zeros(groups, num_gold, dtype=torch.bool, device=record_anchors.device)
    if matches is not None:
        for group, group_matches in enumerate(matches):
            for prediction, gold in group_matches:
                if gold < num_gold and prediction < slots:
                    index[group, gold] = int(prediction)
                    mask[group, gold] = True
    else:
        for group in range(groups):
            count = num_gold if gold_count is None else int(gold_count[group].item())
            for gold in range(min(count, num_gold, slots)):
                if bool(anchor_mask[group, gold]):
                    index[group, gold] = gold
                    mask[group, gold] = True
    gathered = record_anchors.gather(1, index.unsqueeze(-1).expand(-1, -1, dim))
    return gathered, mask


def select_inference_anchors(
    field_logits: torch.Tensor,
    membership_logits: torch.Tensor,
    entity_mask: torch.Tensor,
    quantity_field_idx: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Best tagged span per (slot, quantity field), or ``-1``.

    ``field_logits`` ``(G, E, C)``, ``membership_logits`` ``(G, A, E)``.
    Returns entity indices ``(G, A, Cq)`` and their presence mask.
    """

    groups, slots, entities = membership_logits.shape
    num_fields = quantity_field_idx.shape[1]
    if entities == 0 or num_fields == 0 or field_logits.shape[-1] == 0:
        empty = torch.full(
            (groups, slots, num_fields), -1, dtype=torch.long, device=membership_logits.device
        )
        return empty, empty.ge(0)
    field_index = quantity_field_idx.clamp(0, field_logits.shape[-1] - 1)
    field_probs = torch.sigmoid(field_logits).gather(
        2, field_index.unsqueeze(1).expand(-1, entities, -1)
    )  # (G, E, Cq)
    member_probs = torch.sigmoid(membership_logits)  # (G, A, E)
    scores = member_probs.unsqueeze(-1) * field_probs.unsqueeze(1)  # (G, A, E, Cq)
    member_ok = member_probs.gt(threshold).unsqueeze(-1)
    field_ok = field_probs.gt(threshold).unsqueeze(1)
    valid = member_ok & field_ok & entity_mask.bool()[:, None, :, None]
    scores = scores.masked_fill(~valid, -1.0)
    best_score, best = scores.max(dim=2)  # (G, A, Cq)
    present = best_score.gt(0)
    return torch.where(present, best, torch.full_like(best, -1)), present


def gold_anchor_entities(
    span_labels: torch.Tensor | None,
    quantity_field_idx: torch.Tensor,
    num_gold: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Entity index of the gold anchor span per (gold record, quantity field).

    ``span_labels`` is ``(G, E, R, C)``.  Returns ``(G, R, Cq)`` indices and
    mask.
    """

    groups, num_fields = quantity_field_idx.shape
    device = quantity_field_idx.device
    if span_labels is None or span_labels.shape[1] == 0:
        empty = torch.zeros(groups, num_gold, num_fields, dtype=torch.long, device=device)
        return empty, empty.bool()
    labels = span_labels[:, :, :num_gold]
    if labels.shape[2] < num_gold:
        labels = F.pad(labels, (0, 0, 0, num_gold - labels.shape[2]))
    field_index = quantity_field_idx.clamp(0, labels.shape[-1] - 1)
    gathered = labels.gather(
        3,
        field_index[:, None, None, :].expand(-1, labels.shape[1], labels.shape[2], -1),
    )  # (G, E, R, Cq)
    present = gathered.amax(dim=1).gt(0.5)
    best = gathered.argmax(dim=1)
    return best, present


__all__ = [
    "ORIGIN_ABSOLUTE",
    "ORIGIN_ANCHOR",
    "ORIGIN_REFERENCE",
    "TypedFieldModule",
    "gold_anchor_entities",
    "gold_record_anchors",
    "select_inference_anchors",
    "target_cdf",
]
