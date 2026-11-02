"""Constrained decoding of the normalization head's calendar classifiers.

Each component has value classes plus a last "null" class (not written).
Years are classes ``[first .. last]``, then "out of range" (the year comes
from the era + digit head when the head has one), then null.  An empty year
range leaves only "out of range" and null, so the digit head reads every
year.

The decoder picks the best joint combination over the allowed patterns
(date: ``Y``, ``YM``, ``YMD``, ``MD``; time: ``H``, ``HM``, ``HMS``; a time may
follow ``YMD``/``MD`` in a datetime) and over valid calendar dates, so it
never returns February 30. With ``partial=False`` only full values are
allowed (``YMD``; ``HM``/``HMS`` for times; ``YMD`` + ``HMS`` for datetimes).
"""

from __future__ import annotations

import functools
import math

import torch

from .values import FAMILY_DATE, FAMILY_DATETIME, FAMILY_TIME

_MONTH_DAYS = torch.tensor([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
_NEG = -1e30


def _month_lengths(years: torch.Tensor) -> torch.Tensor:
    """``(Y, 12)`` days per month for proleptic Gregorian ``years``."""

    leap = (years % 4 == 0) & ((years % 100 != 0) | (years % 400 == 0))
    lengths = _MONTH_DAYS.unsqueeze(0).repeat(years.numel(), 1)
    lengths[:, 1] += leap.long()
    return lengths


def _ymd_mask(years: torch.Tensor) -> torch.Tensor:
    """``(Y, 12, 31)`` valid (month, day) per year."""

    lengths = _month_lengths(years)
    days = torch.arange(1, 32)
    return days[None, None, :] <= lengths[:, :, None]


@functools.lru_cache(maxsize=8)
def date_pattern_mask(first: int, last: int, partial: bool) -> torch.Tensor:
    """``(Y + 2, 13, 32)`` validity for in-range years, out of range, null.

    The out-of-range row is filled per prediction (it depends on the year
    the digit head reads), so here it is all False.
    """

    years = torch.arange(first, last + 1)
    span = years.numel()
    mask = torch.zeros(span + 2, 13, 32, dtype=torch.bool)
    if span:
        mask[:span, :12, :31] = _ymd_mask(years)  # YMD
    if partial and span:
        mask[:span, :12, 31] = True  # YM
        mask[:span, 12, 31] = True  # Y
    if partial:
        # MD without a year: February 29 is allowed.
        mask[span + 1, :12, :31] = _ymd_mask(torch.tensor([2000]))[0]
    return mask


def _row_for_year(year: int, partial: bool) -> torch.Tensor:
    row = torch.zeros(13, 32, dtype=torch.bool)
    row[:12, :31] = _ymd_mask(torch.tensor([year]))[0]
    if partial:
        row[:12, 31] = True
        row[12, 31] = True
    return row


def _log(probs: torch.Tensor) -> torch.Tensor:
    return torch.log(probs.detach().float().cpu().clamp(min=1e-30))


def _best_date(year_probs, month_probs, day_probs, *, first, last, fallback_year, partial):
    """Best ``(score, year, month, day)`` for time-allowed and time-free dates.

    Returns ``{"with_time": (...), "no_time": (...)}``; ``None`` components
    are not written. Time-allowed patterns are ``YMD``/``MD``.
    """

    span = last - first + 1
    mask = date_pattern_mask(first, last, partial).clone()
    if fallback_year is not None:
        mask[span] = _row_for_year(int(fallback_year), partial)
    scores = (
        _log(year_probs)[:, None, None]
        + _log(month_probs)[None, :, None]
        + _log(day_probs)[None, None, :]
    )
    day_written = torch.zeros(13, 32, dtype=torch.bool)
    day_written[:12, :31] = True  # YMD / MD
    results = {}
    for name, keep in (("with_time", day_written), ("no_time", ~day_written)):
        allowed = mask & keep[None]
        if not allowed.any():
            results[name] = None
            continue
        masked = scores.masked_fill(~allowed, _NEG)
        flat = int(masked.argmax())
        y_index, rest = divmod(flat, 13 * 32)
        m_index, d_index = divmod(rest, 32)
        if y_index < span:
            year = first + y_index
        elif y_index == span:
            year = int(fallback_year) if fallback_year is not None else None
        else:
            year = None
        month = None if m_index == 12 else m_index + 1
        day = None if d_index == 31 else d_index + 1
        results[name] = (float(masked.view(-1)[flat]), year, month, day)
    return results


def _best_time(hour_probs, minute_probs, second_probs, allowed):
    """Best ``(score, hour, minute, second)`` over ``allowed`` patterns.

    Pattern ``""`` is the null time (every component null).
    """

    lh, lm, ls = _log(hour_probs), _log(minute_probs), _log(second_probs)
    h_score, h = (float(value) for value in lh[:24].max(0))
    m_score, m = (float(value) for value in lm[:60].max(0))
    s_score, s = (float(value) for value in ls[:60].max(0))
    h, m, s = int(h), int(m), int(s)
    null_h, null_m, null_s = float(lh[24]), float(lm[60]), float(ls[60])
    candidates = {
        "": (null_h + null_m + null_s, None, None, None),
        "H": (h_score + null_m + null_s, h, None, None),
        "HM": (h_score + m_score + null_s, h, m, None),
        "HMS": (h_score + m_score + s_score, h, m, s),
    }
    return max((candidates[name] for name in allowed), key=lambda item: item[0])


def decode_calendar(
    family: int,
    *,
    year_probs: torch.Tensor | None = None,
    month_probs: torch.Tensor | None = None,
    day_probs: torch.Tensor | None = None,
    hour_probs: torch.Tensor | None = None,
    minute_probs: torch.Tensor | None = None,
    second_probs: torch.Tensor | None = None,
    year_range: tuple[int, int] = (1800, 2199),
    fallback_year: int | None = None,
    partial: bool = True,
) -> tuple[dict[str, int | None], float]:
    """Components (``None`` = not written) and the joint probability."""

    first, last = (int(value) for value in year_range)
    components = dict(year=None, month=None, day=None, hour=None, minute=None, second=None)
    if family == FAMILY_TIME:
        allowed = ("H", "HM", "HMS") if partial else ("HM", "HMS")
        score, hour, minute, second = _best_time(hour_probs, minute_probs, second_probs, allowed)
        components.update(hour=hour, minute=minute, second=second)
        return components, math.exp(score)

    dates = _best_date(
        year_probs, month_probs, day_probs,
        first=first, last=last, fallback_year=fallback_year, partial=partial,
    )
    if family == FAMILY_DATE:
        best = max((item for item in dates.values() if item is not None), key=lambda item: item[0])
        score, year, month, day = best
        components.update(year=year, month=month, day=day)
        return components, math.exp(score)

    assert family == FAMILY_DATETIME
    options = []
    if dates["with_time"] is not None:
        time_allowed = ("", "H", "HM", "HMS") if partial else ("HMS",)
        time = _best_time(hour_probs, minute_probs, second_probs, time_allowed)
        options.append((dates["with_time"], time))
    if partial and dates["no_time"] is not None:
        time = _best_time(hour_probs, minute_probs, second_probs, ("",))
        options.append((dates["no_time"], time))
    date_part, time_part = max(options, key=lambda pair: pair[0][0] + pair[1][0])
    components.update(
        year=date_part[1], month=date_part[2], day=date_part[3],
        hour=time_part[1], minute=time_part[2], second=time_part[3],
    )
    return components, math.exp(date_part[0] + time_part[0])


__all__ = ["date_pattern_mask", "decode_calendar"]
