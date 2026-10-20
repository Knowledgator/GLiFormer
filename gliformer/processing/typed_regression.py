"""Regression targets and decoding for typed number and date/time fields.

Typed fields are either **extracted** (``$type: "text"`` with an optional
``$format`` that parses the copied span) or **regressed** (``number``,
``duration``, ``date``, ``time``, ``datetime``, ``estimate``, ``forecast``).
This module covers the regressed side; it is torch-free.

Numbers (unbounded) are decomposed as ``x = sign * 10 ** (k + f)``:

* ``sign``: negative or not;
* ``k``: an exponent class in ``[EXP_MIN, EXP_MAX]``, or a dedicated zero
  class (index 0);
* ``f``: the fractional log10 part (the digits) in ``[0, 1)``;
* ``z = log10|x|``: a direct regression target used to settle decade edges
  and to centre the quantiles.

Bounded numbers (``$min``/``$max``) regress their position ``u`` in the
range; signed fields regress ``asinh(x)``. Quantiles are non-crossing
offsets around the median on a fixed grid of levels, in the same space.

Dates and times are decomposed into calendar components: the year is
regressed, and month, day, hour, minute, and second are classes. Date/time
quantiles are offsets (days or minutes) around the point.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Sequence

from .typed_fields import (
    DATE_RE,
    NUMBER_RE,
    TypedFieldSpec,
    _coerce_datetime,
    _parse_number_match,
    parse_text_value,
    stat_level,
)

# -- constants --------------------------------------------------------------

EXP_MIN = -6
EXP_MAX = 15
# Index 0 is the zero class; index 1 + (k - EXP_MIN) is exponent k.
NUM_EXP_CLASSES = 1 + (EXP_MAX - EXP_MIN + 1)
ZERO_EPSILON = 1e-9

# Quantile levels the head predicts; other requested levels interpolate.
QUANTILE_LEVELS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
MEDIAN_INDEX = QUANTILE_LEVELS.index(0.5)
NUM_QUANTILE_OFFSETS = len(QUANTILE_LEVELS) - 1

FAMILY_NUMBER = 0
FAMILY_DATE = 1
FAMILY_TIME = 2
FAMILY_DATETIME = 3
FAMILY_NAMES = {
    FAMILY_NUMBER: "number",
    FAMILY_DATE: "date",
    FAMILY_TIME: "time",
    FAMILY_DATETIME: "datetime",
}

ABSOLUTE_YEAR_CENTER = 2000.0
ABSOLUTE_YEAR_SCALE = 50.0

_EPOCH = date(1970, 1, 1)


def quantity_family(spec: TypedFieldSpec) -> int:
    value_type = spec.value_type
    if value_type == "date":
        return FAMILY_DATE
    if value_type == "time":
        return FAMILY_TIME
    if value_type == "datetime":
        return FAMILY_DATETIME
    return FAMILY_NUMBER


def quantile_unit_name(family: int) -> str:
    """Unit of date/time quantile offsets: days for dates, minutes otherwise."""

    return "days" if family == FAMILY_DATE else "minutes"


# -- number decomposition ----------------------------------------------------


def exponent_index(k: int) -> int:
    return 1 + min(max(int(k), EXP_MIN), EXP_MAX) - EXP_MIN


def index_exponent(index: int) -> int:
    return EXP_MIN + int(index) - 1


@dataclass
class NumberTarget:
    """Regression targets of one number in the field's space."""

    z: float  # log10|x| (unbounded), u (bounded), or asinh(x) (signed)
    negative: bool = False
    zero: bool = False
    exponent_soft: list[float] = field(default_factory=list)  # NUM_EXP_CLASSES
    digits: float = 0.0


def number_space(spec: TypedFieldSpec) -> str:
    """``bounded``, ``signed`` or ``log`` regression space of a number field."""

    if spec.bounded:
        return "bounded"
    if getattr(spec, "signed", False):
        return "signed"
    return "log"


def number_to_space(x: float, spec: TypedFieldSpec) -> float:
    space = number_space(spec)
    if space == "bounded":
        return spec.to_unit(x, None)
    if space == "signed":
        return math.asinh(x / _scale(spec))
    magnitude = abs(x) / _scale(spec)
    return math.log10(max(magnitude, ZERO_EPSILON))


def number_from_space(z: float, spec: TypedFieldSpec, *, negative: bool = False) -> float:
    space = number_space(spec)
    if space == "bounded":
        return spec.from_unit(min(max(z, 0.0), 1.0), None)
    if space == "signed":
        return math.sinh(z) * _scale(spec)
    value = 10.0 ** min(max(z, EXP_MIN - 1), EXP_MAX + 1) * _scale(spec)
    return -value if negative else value


def _scale(spec: TypedFieldSpec) -> float:
    scale = getattr(spec, "number_scale", None)
    return float(scale) if scale else 1.0


def number_target(x: float, spec: TypedFieldSpec, *, edge_smoothing: float = 0.2) -> NumberTarget:
    """Decompose ``x``; soft exponent labels near decade edges."""

    space = number_space(spec)
    z = number_to_space(x, spec)
    if space != "log":
        return NumberTarget(z=z, negative=x < 0)
    magnitude = abs(x) / _scale(spec)
    soft = [0.0] * NUM_EXP_CLASSES
    if magnitude < ZERO_EPSILON:
        soft[0] = 1.0
        return NumberTarget(z=z, negative=False, zero=True, exponent_soft=soft, digits=0.0)
    k = math.floor(z + 1e-12)
    digits = min(max(z - k, 0.0), 1.0 - 1e-9)
    index = exponent_index(k)
    soft[index] = 1.0
    if edge_smoothing > 0:
        if digits < 0.05 and index > 1:
            soft[index] -= edge_smoothing
            soft[index - 1] += edge_smoothing
        elif digits > 0.95 and index < NUM_EXP_CLASSES - 1:
            soft[index] -= edge_smoothing
            soft[index + 1] += edge_smoothing
    return NumberTarget(z=z, negative=x < 0, zero=False, exponent_soft=soft, digits=digits)


def decode_number_median(
    exponent_probs: Sequence[float],
    digits: float,
    z_direct: float,
    negative_prob: float,
    spec: TypedFieldSpec,
    bounded_median: float | None = None,
) -> tuple[float, float, bool]:
    """Return ``(x, z_median, negative)`` for one prediction."""

    space = number_space(spec)
    if space == "bounded":
        z = bounded_median if bounded_median is not None else z_direct
        return number_from_space(z, spec), z, False
    if space == "signed":
        return number_from_space(z_direct, spec), z_direct, z_direct < 0
    negative = negative_prob > 0.5
    best = max(range(len(exponent_probs)), key=lambda index: exponent_probs[index])
    if best == 0:
        return 0.0, math.log10(ZERO_EPSILON), False
    # Settle decade edges: the exponent class next to the argmax whose
    # k + f lies closest to the direct log regression.
    candidates = [best + delta for delta in (-1, 0, 1) if 1 <= best + delta < NUM_EXP_CLASSES]
    chosen = min(candidates, key=lambda index: abs(index_exponent(index) + digits - z_direct))
    z = index_exponent(chosen) + digits
    return number_from_space(z, spec, negative=negative), z, negative


def interpolate_level(level: float, values: Sequence[float]) -> float:
    """Value at a quantile level from values on ``QUANTILE_LEVELS``."""

    levels = QUANTILE_LEVELS
    if level <= levels[0]:
        return values[0]
    if level >= levels[-1]:
        return values[-1]
    for index in range(len(levels) - 1):
        left, right = levels[index], levels[index + 1]
        if left <= level <= right:
            weight = (level - left) / (right - left)
            return values[index] + weight * (values[index + 1] - values[index])
    return values[MEDIAN_INDEX]


def quantile_targets(
    gold: Any,
    spec: TypedFieldSpec,
    horizon: int,
) -> tuple[list[float | None], list[list[float | None]]] | None:
    """Point and labelled-quantile values per horizon step, as floats.

    Returns ``(points, quantiles)`` where ``quantiles[step][level_index]`` is
    the labelled value at that grid level or ``None``. ``None`` overall when
    the gold value cannot be interpreted.
    """

    if isinstance(gold, dict):
        stats = {str(key).lower(): value for key, value in gold.items()}
    else:
        stats = {"value": gold}
    points: list[float | None] = [None] * horizon
    quantiles: list[list[float | None]] = [[None] * len(QUANTILE_LEVELS) for _ in range(horizon)]
    any_value = False
    for name, raw in stats.items():
        try:
            level = stat_level(name)
        except ValueError:
            continue
        if horizon == 1:
            series = raw[:1] if isinstance(raw, list) and len(raw) == 1 else [raw]
        elif isinstance(raw, list) and len(raw) == horizon:
            series = raw
        else:
            return None
        for step, item in enumerate(series):
            if item is None:
                continue
            x = spec.encode(item)
            if x is None:
                return None
            any_value = True
            if level is None:
                points[step] = x
            else:
                for index, grid_level in enumerate(QUANTILE_LEVELS):
                    if abs(grid_level - level) < 1e-6:
                        quantiles[step][index] = x
    if not any_value:
        return None
    for step in range(horizon):
        if points[step] is None:
            points[step] = quantiles[step][MEDIAN_INDEX]
        if points[step] is None:
            return None
    return points, quantiles


# -- dates and times ---------------------------------------------------------


@dataclass
class TemporalTarget:
    year: int | None
    month: int | None  # 1-12
    day: int | None  # 1-31
    hour: int | None
    minute: int | None
    second: int | None


# Years are astronomical internally (1 BCE = 0, 44 BCE = -43, proleptic
# Gregorian) and written as expanded ISO years outside 1..9999 ("-0043").
_EXPANDED_ISO = re.compile(
    r"^\s*([+-]\d{4,}|\d{4})-(\d{2})-(\d{2})"
    r"(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?\s*$"
)
YEAR_DIGITS = 4
ERA_CE = 0
ERA_BCE = 1


def is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def days_in_month(year: int, month: int) -> int:
    if month == 2:
        return 29 if is_leap_year(year) else 28
    return 30 if month in (4, 6, 9, 11) else 31


def days_from_civil(year: int, month: int, day: int) -> int:
    """Days since 1970-01-01 for any proleptic Gregorian date."""

    year -= month <= 2
    era = year // 400
    yoe = year - era * 400
    doy = (153 * ((month + 9) % 12) + 2) // 5 + day - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def civil_from_days(days: int) -> tuple[int, int, int]:
    days += 719468
    era = days // 146097
    doe = days - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    day = doy - (153 * mp + 2) // 5 + 1
    month = mp + 3 if mp < 10 else mp - 9
    return yoe + era * 400 + (month <= 2), month, day


def format_year(year: int) -> str:
    return f"{year:04d}" if 1 <= year <= 9999 else (f"-{-year:04d}" if year <= 0 else f"+{year}")


def year_to_era_digits(year: int) -> tuple[int, list[int]]:
    """Era (CE/BCE) and the 4 digits of the year as people write it."""

    if year >= 1:
        era, magnitude = ERA_CE, year
    else:
        era, magnitude = ERA_BCE, 1 - year  # 1 BCE = astronomical 0
    magnitude = min(magnitude, 10 ** YEAR_DIGITS - 1)
    return era, [magnitude // 10 ** power % 10 for power in range(YEAR_DIGITS - 1, -1, -1)]


def era_digits_to_year(era: int, digits: Sequence[int]) -> int:
    magnitude = 0
    for digit in digits:
        magnitude = magnitude * 10 + int(digit)
    if era == ERA_BCE:
        return 1 - max(magnitude, 1)
    return max(magnitude, 1)


def _expanded_components(value: str):
    match = _EXPANDED_ISO.match(value)
    if not match:
        return None
    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
    if not 1 <= month <= 12 or not 1 <= day <= days_in_month(year, month):
        return None
    hour, minute, second = (int(match.group(index) or 0) for index in (4, 5, 6))
    return year, month, day, hour, minute, second


def temporal_target(value: Any, family: int) -> TemporalTarget | None:
    """Calendar components of an ISO date/time/datetime (wall clock)."""

    if value is None:
        return None
    if family == FAMILY_TIME:
        seconds = parse_text_value(str(value), "time") if not isinstance(value, (int, float)) else None
        if seconds is None:
            return None
        seconds = int(round(seconds)) % 86400
        return TemporalTarget(None, None, None, seconds // 3600, seconds % 3600 // 60, seconds % 60)
    expanded = _expanded_components(value) if isinstance(value, str) else None
    if expanded is not None:
        year, month, day, hour, minute, second = expanded
    else:
        moment = _coerce_datetime(value)
        if moment is None:
            return None
        year, month, day = moment.year, moment.month, moment.day
        hour, minute, second = moment.hour, moment.minute, moment.second
    if family == FAMILY_DATE:
        return TemporalTarget(year, month, day, None, None, None)
    return TemporalTarget(year, month, day, hour, minute, second)


def temporal_units(value: Any, family: int) -> float | None:
    """Point in quantile units: days since 1970 (date) or minutes."""

    target = temporal_target(value, family)
    if target is None:
        return None
    if family == FAMILY_TIME:
        return target.hour * 60.0 + target.minute + target.second / 60.0
    days = days_from_civil(target.year, target.month, target.day)
    if family == FAMILY_DATE:
        return float(days)
    return days * 1440.0 + target.hour * 60.0 + target.minute + target.second / 60.0


def compose_temporal(
    family: int,
    *,
    year: int | None = None,
    month: int | None = None,
    day: int | None = None,
    hour: int | None = None,
    minute: int | None = None,
    second: int | None = None,
) -> str | None:
    if family == FAMILY_TIME:
        hour, minute, second = int(hour or 0), int(minute or 0), int(second or 0)
        if second:
            return f"{hour:02d}:{minute:02d}:{second:02d}"
        return f"{hour:02d}:{minute:02d}"
    if year is None or month is None or day is None:
        return None
    year, month = int(year), int(month)
    day = min(max(int(day), 1), days_in_month(year, month))
    text = f"{format_year(year)}-{month:02d}-{day:02d}"
    if family == FAMILY_DATE:
        return text
    return f"{text}T{int(hour or 0):02d}:{int(minute or 0):02d}:{int(second or 0):02d}"


def shift_temporal(iso: str, family: int, offset_units: float) -> str | None:
    """Move an ISO value by a quantile offset (days or minutes)."""

    if family == FAMILY_TIME:
        hours, minutes, *rest = [int(part) for part in iso.split(":")]
        total = (hours * 60 + minutes + int(offset_units)) % 1440
        return f"{total // 60:02d}:{total % 60:02d}"
    target = temporal_target(iso, family)
    if target is None:
        return None
    days = days_from_civil(target.year, target.month, target.day)
    if family == FAMILY_DATE:
        year, month, day = civil_from_days(days + int(offset_units))
        return compose_temporal(family, year=year, month=month, day=day)
    minutes = days * 1440 + target.hour * 60 + target.minute + int(offset_units)
    year, month, day = civil_from_days(minutes // 1440)
    rest = minutes % 1440
    return compose_temporal(
        family, year=year, month=month, day=day, hour=rest // 60, minute=rest % 60,
        second=target.second,
    )


# -- number and year mentions in the input ------------------------------------

_JOIN_INSIDE_NUMBER = re.compile(r"[\d][,.:/\-]$|^[,.:/\-]$")


@dataclass
class Mention:
    """A number or year written in the input, located on the word grid."""

    start: int  # first word index
    end: int  # last word index (inclusive)
    value: float
    is_year: bool = False
    year: int = 0


def _compact_text(words: Sequence[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join words, gluing pieces of numbers ("1", ",", "250") together."""

    pieces: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    previous = ""
    for index, word in enumerate(words):
        following = words[index + 1] if index + 1 < len(words) else ""
        glue = index > 0 and (
            (
                re.fullmatch(r"[.:/\-]", word) and re.search(r"\d$", previous)
            )
            or (
                # A comma joins digit groups only before exactly three digits
                # ("1 , 250"), never in dates ("October 14 , 2024").
                word == "," and re.search(r"\d$", previous)
                and re.fullmatch(r"\d{3}(?:[.,]\d+)?", following)
            )
            or (
                re.fullmatch(r"[.:/\-]", previous) and re.match(r"\d", word)
                and index >= 2 and re.search(r"\d$", words[index - 2])
            )
            or (
                previous == "," and re.fullmatch(r"\d{3}(?:[.,]\d+)?", word)
                and index >= 2 and re.search(r"\d$", words[index - 2])
            )
        )
        # Punctuation sits against the previous word, as in written text.
        if index > 0 and not glue and word not in {",", ";", ")"}:
            pieces.append(" ")
            cursor += 1
        spans.append((cursor, cursor + len(word)))
        pieces.append(word)
        cursor += len(word)
        previous = word
    return "".join(pieces), spans


def _word_range(spans: list[tuple[int, int]], start: int, end: int) -> tuple[int, int] | None:
    first = last = None
    for index, (word_start, word_end) in enumerate(spans):
        if word_end <= start or word_start >= end:
            continue
        if first is None:
            first = index
        last = index
    return (first, last) if first is not None else None


_YEAR_WORD_RE = re.compile(r"(?<![\d.,])(19\d{2}|20\d{2}|21\d{2})(?![\d.,]\d)")
_YEAR_IN_WORD_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2}|21\d{2})(?!\d)")


def find_mentions(words: Sequence[str], max_mentions: int = 64) -> list[Mention]:
    """Numbers and years in a word sequence, in reading order.

    Years come from parsed dates ("March 3, 2024", "2024-03-03") and from
    standalone four-digit years in 1900-2199.
    """

    if not words:
        return []
    text, spans = _compact_text(words)
    mentions: dict[tuple[int, int], Mention] = {}
    for match in DATE_RE.finditer(text):
        days = parse_text_value(match.group(0), "date")
        word_range = _word_range(spans, match.start(), match.end())
        if days is None or word_range is None:
            continue
        year = (_EPOCH + timedelta(days=int(days))).year
        mentions[word_range] = Mention(*word_range, value=float(year), is_year=True, year=year)
    date_words = {
        index
        for mention in mentions.values()
        for index in range(mention.start, mention.end + 1)
    }
    for match in NUMBER_RE.finditer(text):
        word_range = _word_range(spans, match.start(), match.end())
        if word_range is None or word_range in mentions:
            continue
        if any(index in date_words for index in range(word_range[0], word_range[1] + 1)):
            continue  # part of a date; the date mention carries its year
        if re.match(r":\d", text[match.end():match.end() + 2]) or re.search(
            r"\d:$", text[max(0, match.start() - 2):match.start()]
        ):
            continue  # part of a clock time
        value = _parse_number_match(match)
        if value is None or not math.isfinite(value):
            continue
        year_match = _YEAR_WORD_RE.fullmatch(match.group(0).strip())
        is_year = bool(year_match) and float(value).is_integer()
        mentions[word_range] = Mention(
            *word_range, value=value, is_year=is_year, year=int(value) if is_year else 0,
        )
    covered = {
        index
        for mention in mentions.values()
        for index in range(mention.start, mention.end + 1)
    }
    for index, word in enumerate(words):
        if index in covered:
            continue
        year_match = _YEAR_IN_WORD_RE.search(word)
        if year_match and re.search(r"[A-Za-z]", word):  # "FY2023", "Q4-2024"
            year = int(year_match.group(1))
            mentions[(index, index)] = Mention(index, index, float(year), True, year)
    ordered = sorted(mentions.values(), key=lambda mention: (mention.start, mention.end))
    return ordered[:max_mentions]


def mention_features(mention: Mention) -> list[float]:
    """Raw per-mention features: sign, log10|v|, is_zero, is_year, year."""

    magnitude = abs(mention.value)
    return [
        1.0 if mention.value < 0 else 0.0,
        math.log10(magnitude) if magnitude > ZERO_EPSILON else 0.0,
        1.0 if magnitude <= ZERO_EPSILON else 0.0,
        1.0 if mention.is_year else 0.0,
        (mention.year - ABSOLUTE_YEAR_CENTER) / ABSOLUTE_YEAR_SCALE if mention.is_year else 0.0,
    ]


NUM_MENTION_FEATURES = 5


__all__ = [
    "ABSOLUTE_YEAR_CENTER",
    "ABSOLUTE_YEAR_SCALE",
    "EXP_MAX",
    "EXP_MIN",
    "FAMILY_DATE",
    "FAMILY_DATETIME",
    "FAMILY_NAMES",
    "FAMILY_NUMBER",
    "FAMILY_TIME",
    "MEDIAN_INDEX",
    "Mention",
    "NUMBER_RE",
    "NUM_EXP_CLASSES",
    "NUM_MENTION_FEATURES",
    "NUM_QUANTILE_OFFSETS",
    "NumberTarget",
    "QUANTILE_LEVELS",
    "TemporalTarget",
    "ERA_BCE",
    "ERA_CE",
    "YEAR_DIGITS",
    "civil_from_days",
    "compose_temporal",
    "days_from_civil",
    "era_digits_to_year",
    "format_year",
    "is_leap_year",
    "year_to_era_digits",
    "days_in_month",
    "decode_number_median",
    "exponent_index",
    "find_mentions",
    "index_exponent",
    "interpolate_level",
    "mention_features",
    "number_from_space",
    "number_space",
    "number_target",
    "number_to_space",
    "quantile_targets",
    "quantile_unit_name",
    "quantity_family",
    "shift_temporal",
    "temporal_target",
    "temporal_units",
]
