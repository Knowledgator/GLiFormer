"""Regression heads for typed number and date/time structuring fields.

The (record, field) vectors of :class:`TypedFieldModule` feed:

* a **number head**: sign, exponent class (with a zero class), fractional
  log10 digits, a direct log10 regression (decade edges and quantile centre),
  a bounded-range median, and non-crossing quantile offsets;
* **date/time heads**: the year by classification (era CE/BCE plus four
  10-way digit classifiers on the year as people write it), month/day/hour/
  minute/second classifiers, and quantile offsets in days (dates) or minutes
  (times, datetimes);
* a presence logit.

A **numeric memory** gives the readout explicit magnitudes: every number or
year written in the input becomes one extra memory vector built from its
pooled words and features of its parsed value, so the heads can attend to
"4.3 billion" as a quantity rather than as sub-word pieces.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ...layers.mlp import create_mlp
from ...processing.typed_regression import (
    YEAR_DIGITS,
    FAMILY_DATE,
    FAMILY_DATETIME,
    FAMILY_NUMBER,
    FAMILY_TIME,
    MEDIAN_INDEX,
    NUM_EXP_CLASSES,
    NUM_MENTION_FEATURES,
    NUM_QUANTILE_OFFSETS,
    QUANTILE_LEVELS,
)

SPACE_LOG = 0
SPACE_BOUNDED = 1
SPACE_SIGNED = 2

# Number head output layout.
_N_SIGN = 1
_N_EXP = NUM_EXP_CLASSES
_N_DIGITS = 1
_N_Z = 1
_N_BOUNDED = 1
NUMBER_OUTPUTS = _N_SIGN + _N_EXP + _N_DIGITS + _N_Z + _N_BOUNDED + NUM_QUANTILE_OFFSETS

# Temporal head output layout.
TEMPORAL_CLASSES = {"month": 12, "day": 31, "hour": 24, "minute": 60, "second": 60}
TEMPORAL_OUTPUTS = sum(TEMPORAL_CLASSES.values()) + 1 + NUM_QUANTILE_OFFSETS  # + abs year

_FOURIER_FREQUENCIES = (0.25, 0.5, 1.0, 2.0, 4.0)
_LEVELS = torch.tensor(QUANTILE_LEVELS)


def split_number_outputs(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    parts = raw.split([_N_SIGN, _N_EXP, _N_DIGITS, _N_Z, _N_BOUNDED, NUM_QUANTILE_OFFSETS], dim=-1)
    return {
        "sign_logit": parts[0].squeeze(-1),
        "exp_logits": parts[1],
        "digits": torch.sigmoid(parts[2].squeeze(-1)),
        "z": parts[3].squeeze(-1),
        "bounded_median": torch.sigmoid(parts[4].squeeze(-1)),
        "offset_raw": parts[5],
    }


def split_temporal_outputs(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    sizes = list(TEMPORAL_CLASSES.values()) + [1, NUM_QUANTILE_OFFSETS]
    parts = raw.split(sizes, dim=-1)
    result = {f"{name}_logits": part for name, part in zip(TEMPORAL_CLASSES, parts)}
    # Unused slot of the former year regression (kept so checkpoints load).
    result["absolute_year"] = parts[len(TEMPORAL_CLASSES)].squeeze(-1)
    result["offset_raw"] = parts[-1]
    return result


def quantile_offsets(offset_raw: torch.Tensor) -> torch.Tensor:
    """Signed offsets from the median at every grid level (non-crossing).

    ``offset_raw`` has ``NUM_QUANTILE_OFFSETS`` values: the lower half
    accumulates downwards from the median, the upper half upwards.
    """

    half = MEDIAN_INDEX
    steps = F.softplus(offset_raw)
    lower = -torch.cumsum(steps[..., :half], dim=-1).flip(-1)
    upper = torch.cumsum(steps[..., half:], dim=-1)
    zero = torch.zeros_like(steps[..., :1])
    return torch.cat([lower, zero, upper], dim=-1)


def pinball(residual: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Pinball loss of ``residual = target - prediction`` at ``levels``."""

    return torch.maximum(levels * residual, (levels - 1.0) * residual)


def days_from_civil(year: torch.Tensor, month: torch.Tensor, day: torch.Tensor) -> torch.Tensor:
    """Days since 1970-01-01 for proleptic Gregorian dates (vectorized)."""

    year = year - (month <= 2).to(year.dtype)
    era = torch.div(year, 400, rounding_mode="floor")
    yoe = year - era * 400
    mp = (month + 9) % 12
    doy = torch.div(153 * mp + 2, 5, rounding_mode="floor") + day - 1
    doe = yoe * 365 + torch.div(yoe, 4, rounding_mode="floor") - torch.div(yoe, 100, rounding_mode="floor") + doy
    return era * 146097 + doe - 719468


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def _unit_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the last axis per unit, then over units that have terms."""

    mask = mask.to(values.dtype)
    per_unit = (values * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
    has = (mask.sum(-1) > 0).to(values.dtype)
    return (per_unit * has).sum() / has.sum().clamp(min=1.0)


class TypedRegressionModule(nn.Module):
    """Numeric memory plus number and date/time regression heads."""

    def __init__(
        self,
        hidden_size: int,
        dropout: float,
        *,
        max_horizon: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_horizon = int(max_horizon)
        feature_dim = NUM_MENTION_FEATURES + 2 * len(_FOURIER_FREQUENCIES)
        self.mention_features = create_mlp(
            input_dim=feature_dim,
            intermediate_dims=[hidden_size],
            output_dim=hidden_size,
            dropout=dropout,
            activation="gelu",
        )
        self.mention_merge = nn.Linear(2 * hidden_size, hidden_size)
        self.mention_norm = nn.LayerNorm(hidden_size)

        self.step_embedding = nn.Embedding(self.max_horizon, hidden_size)
        self.number_head = create_mlp(
            input_dim=hidden_size,
            intermediate_dims=[hidden_size],
            output_dim=NUMBER_OUTPUTS,
            dropout=dropout,
            activation="gelu",
        )
        self.temporal_head = create_mlp(
            input_dim=hidden_size,
            intermediate_dims=[hidden_size],
            output_dim=TEMPORAL_OUTPUTS,
            dropout=dropout,
            activation="gelu",
        )
        self.presence_head = nn.Linear(hidden_size, 1)
        # Year: era (CE/BCE) and one 10-way classifier per digit of the year
        # as written (44 BCE -> BCE, 0 0 4 4). The temporal head's
        # regression slot for the year is no longer used.
        self.year_head = create_mlp(
            input_dim=hidden_size,
            intermediate_dims=[hidden_size],
            output_dim=2 + 10 * YEAR_DIGITS,
            dropout=dropout,
            activation="gelu",
        )

    # -- numeric memory --------------------------------------------------------

    @staticmethod
    def _value_features(values: torch.Tensor) -> torch.Tensor:
        log_magnitude = values[..., 1:2]
        angles = log_magnitude * torch.tensor(
            _FOURIER_FREQUENCIES, device=values.device, dtype=values.dtype
        ) * math.pi
        scaled = values.clone()
        scaled[..., 1] = values[..., 1] / 10.0
        return torch.cat([scaled, torch.sin(angles), torch.cos(angles)], dim=-1)

    def mention_memory(
        self,
        words: torch.Tensor,
        word_mask: torch.Tensor,
        spans: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(G, M, D)`` mention vectors and their validity.

        ``spans``/``values``/``mask`` are already gathered per group.
        Mentions past the visible words (truncation) are masked.
        """

        groups, length, dim = words.shape
        if spans.shape[1] == 0 or length == 0:
            return words.new_zeros(groups, spans.shape[1], dim), mask.new_zeros(mask.shape)
        start = spans[..., 0].clamp(0, length - 1)
        end = spans[..., 1].clamp(0, length - 1)
        visible = mask.bool() & (spans[..., 1] < length) & word_mask.bool().gather(1, end)
        positions = torch.arange(length, device=words.device)
        inside = (positions[None, None, :] >= start[..., None]) & (
            positions[None, None, :] <= end[..., None]
        )
        inside = inside & word_mask.bool()[:, None, :]
        weights = inside.to(words.dtype)
        pooled = torch.einsum("gml,gld->gmd", weights, words) / weights.sum(-1, keepdim=True).clamp(min=1.0)
        features = self.mention_features(self._value_features(values.to(words.dtype)))
        memory = self.mention_norm(self.mention_merge(torch.cat([pooled, features], dim=-1)))
        return memory * visible[..., None].to(memory.dtype), visible

    # -- heads ----------------------------------------------------------------

    def forward_heads(self, vectors: torch.Tensor, horizon: int) -> dict[str, torch.Tensor]:
        """Number (per step), temporal, and presence outputs.

        ``vectors`` is ``(G, A, Cq, D)``; number outputs gain an ``H`` axis.
        """

        horizon = max(1, min(int(horizon), self.max_horizon))
        steps = self.step_embedding.weight[:horizon].to(vectors.dtype)
        number = split_number_outputs(self.number_head(vectors.unsqueeze(-2) + steps))
        temporal = split_temporal_outputs(self.temporal_head(vectors))
        year = self.year_head(vectors)
        temporal["era_logits"] = year[..., :2]
        temporal["year_digit_logits"] = year[..., 2:].unflatten(-1, (YEAR_DIGITS, 10))
        presence = self.presence_head(vectors).squeeze(-1)
        return {"number": number, "temporal": temporal, "presence_logit": presence}

    # -- losses ----------------------------------------------------------------

    def number_losses(self, outputs: dict, targets: dict, unit: torch.Tensor, step: torch.Tensor):
        """Median and quantile losses of the number head.

        ``unit`` ``(G, R, Cq)`` selects present number values; ``step``
        ``(G, R, Cq, H)`` the valid horizon steps with a point label.
        """

        space = targets["space"][:, None, :, None].expand_as(step)
        mask = unit[..., None] & step
        log_space = mask & space.eq(SPACE_LOG)
        nonzero = log_space & ~targets["zero"].bool()
        bounded = mask & space.eq(SPACE_BOUNDED)
        direct = mask & ~space.eq(SPACE_BOUNDED)

        exp_log_probs = F.log_softmax(outputs["exp_logits"].float(), dim=-1)
        exp_loss = -(targets["exp_soft"] * exp_log_probs).sum(-1)
        digits_loss = (outputs["digits"] - targets["digits"]).abs()
        sign_loss = F.binary_cross_entropy_with_logits(
            outputs["sign_logit"].float(), targets["negative"].float(), reduction="none",
        )
        z = targets["z"]
        direct_loss = F.smooth_l1_loss(outputs["z"], z, reduction="none", beta=0.05)
        bounded_loss = (outputs["bounded_median"] - z).abs()

        median_terms = torch.stack(
            [
                exp_loss * log_space,
                digits_loss * nonzero,
                sign_loss * nonzero,
                direct_loss * direct,
                bounded_loss * bounded,
            ],
            dim=-1,
        )
        median_mask = torch.stack([log_space, nonzero, nonzero, direct, bounded], dim=-1)
        median_loss = _unit_mean(
            median_terms.flatten(-2), median_mask.flatten(-2)
        )

        center = torch.where(bounded, outputs["bounded_median"], outputs["z"])
        levels_q = center[..., None] + quantile_offsets(outputs["offset_raw"])
        levels = _LEVELS.to(levels_q.device, levels_q.dtype)
        not_median = torch.ones_like(levels, dtype=torch.bool)
        not_median[MEDIAN_INDEX] = False
        point_pinball = pinball(z[..., None] - levels_q, levels)
        point_mask = mask[..., None] & not_median
        labelled = targets["quantiles_mask"].bool() & mask[..., None] & not_median
        labelled_loss = (levels_q - targets["quantiles"]).abs()
        quantile_terms = torch.cat([point_pinball, labelled_loss], dim=-1)
        quantile_mask = torch.cat([point_mask, labelled], dim=-1)
        quantile_loss = _unit_mean(quantile_terms.flatten(-3), quantile_mask.flatten(-3))
        return median_loss, quantile_loss

    def temporal_losses(self, outputs: dict, targets: dict, unit: torch.Tensor):
        """Year, calendar-component, and quantile losses of date/time fields.

        ``unit`` ``(G, R, Cq)`` selects present date/time values.
        """

        terms = []
        masks = []
        for name, classes in TEMPORAL_CLASSES.items():
            target = targets[name]
            valid = unit & target.ge(0)
            loss = F.cross_entropy(
                outputs[f"{name}_logits"].float().reshape(-1, classes),
                target.clamp(min=0).reshape(-1),
                reduction="none",
            ).reshape(target.shape)
            terms.append(loss)
            masks.append(valid)

        has_year = unit & targets["year_mask"].bool()
        era_loss = F.cross_entropy(
            outputs["era_logits"].float().reshape(-1, 2),
            targets["era"].clamp(min=0).reshape(-1),
            reduction="none",
        ).reshape(has_year.shape)
        terms.append(era_loss)
        masks.append(has_year)
        digits = targets["year_digits"]
        digit_logits = outputs["year_digit_logits"].float()
        for position in range(YEAR_DIGITS):
            loss = F.cross_entropy(
                digit_logits[..., position, :].reshape(-1, 10),
                digits[..., position].clamp(min=0).reshape(-1),
                reduction="none",
            ).reshape(has_year.shape)
            terms.append(loss)
            masks.append(has_year)
        return self._finish_temporal_losses(outputs, targets, unit, terms, masks)

    def _finish_temporal_losses(self, outputs, targets, unit, terms, masks):
        component_loss = _unit_mean(torch.stack(terms, -1), torch.stack(masks, -1))

        offsets = quantile_offsets(outputs["offset_raw"])
        labelled = targets["quantiles_mask"].bool() & unit[..., None]
        labelled[..., MEDIAN_INDEX] = False
        scale = targets["quantile_scale"][:, None, :, None]
        quantile_loss = _unit_mean(
            (offsets - targets["quantiles"] / scale).abs(), labelled,
        )
        return component_loss, quantile_loss


__all__ = [
    "NUMBER_OUTPUTS",
    "SPACE_BOUNDED",
    "SPACE_LOG",
    "SPACE_SIGNED",
    "TEMPORAL_CLASSES",
    "TEMPORAL_OUTPUTS",
    "TypedRegressionModule",
    "days_from_civil",
    "pinball",
    "quantile_offsets",
    "split_number_outputs",
    "split_temporal_outputs",
]
