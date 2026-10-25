"""Normalized values of spans: value families, calendar parts, target codes.

The normalization head reads one normalized value from a span.  A span of a
number field holds a number; a span of a date, time or datetime field holds
calendar components, any of which may be missing from the text ("May 2024"
has no day, "14:30" has no seconds).  This module is torch-free: it turns
gold values into per-component codes and composes decoded components back
into ISO text.

Component codes used in label tensors:

* ``>= 0``: the written value (years 1 CE and later, months 1-12, days
  1-31, hours 0-23, minutes and seconds 0-59);
* ``NULL`` (-1): the component is not written in the span;
* ``IGNORE`` (-100): no supervision (the component does not apply to the
  span's family, or the span has no temporal target).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ...processing.typed_fields import (
    _coerce_datetime,
    parse_text_value,
)

FAMILY_NONE = -1
FAMILY_NUMBER = 0
FAMILY_DATE = 1
FAMILY_TIME = 2
FAMILY_DATETIME = 3
TEMPORAL_FAMILIES = (FAMILY_DATE, FAMILY_TIME, FAMILY_DATETIME)
FAMILY_NAMES = {
    FAMILY_NUMBER: "number",
    FAMILY_DATE: "date",
    FAMILY_TIME: "time",
    FAMILY_DATETIME: "datetime",
}

# Durations are read as plain numbers.
VALUE_TYPE_FAMILY = {
    "number": FAMILY_NUMBER,
    "duration": FAMILY_NUMBER,
    "date": FAMILY_DATE,
    "time": FAMILY_TIME,
    "datetime": FAMILY_DATETIME,
}

NULL = -1
IGNORE = -100
COMPONENTS = ("year", "month", "day", "hour", "minute", "second")
DATE_COMPONENTS = ("year", "month", "day")
TIME_COMPONENTS = ("hour", "minute", "second")

YEAR_DIGITS = 4


@dataclass
class TemporalParts:
    """Calendar components; ``None`` marks a component that is not written."""

    year: int | None = None
    month: int | None = None  # 1-12
    day: int | None = None  # 1-31
    hour: int | None = None
    minute: int | None = None
    second: int | None = None


# -- calendar arithmetic -----------------------------------------------------


def is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def days_in_month(year: int, month: int) -> int:
    if month == 2:
        return 29 if is_leap_year(year) else 28
    return 30 if month in (4, 6, 9, 11) else 31


def format_year(year: int) -> str:
    """Four-digit year, or an expanded ISO year outside 1..9999."""

    if 1 <= year <= 9999:
        return f"{year:04d}"
    return f"-{-year:04d}" if year <= 0 else f"+{year}"


# -- written patterns --------------------------------------------------------

DATE_PATTERNS = ("Y", "YM", "YMD", "MD")
TIME_PATTERNS = ("H", "HM", "HMS")
# Date patterns a time of day may follow in a datetime.
DATE_PATTERNS_WITH_TIME = ("YMD", "MD")


def _code(*pairs) -> str:
    return "".join(letter for letter, value in pairs if value is not None)


def date_pattern(parts: TemporalParts) -> str:
    return _code(("Y", parts.year), ("M", parts.month), ("D", parts.day))


def time_pattern(parts: TemporalParts) -> str:
    return _code(("H", parts.hour), ("M", parts.minute), ("S", parts.second))


def valid_parts(parts: TemporalParts, family: int, *, partial: bool = True) -> bool:
    """Components in range and forming an allowed written pattern."""

    if parts.month is not None and not 1 <= parts.month <= 12:
        return False
    if parts.day is not None:
        if parts.month is None:
            return False
        limit = (
            days_in_month(parts.year, parts.month)
            if parts.year is not None
            else (29 if parts.month == 2 else days_in_month(2001, parts.month))
        )
        if not 1 <= parts.day <= limit:
            return False
    if parts.hour is not None and not 0 <= parts.hour <= 23:
        return False
    if parts.minute is not None and not 0 <= parts.minute <= 59:
        return False
    if parts.second is not None and not 0 <= parts.second <= 59:
        return False
    time_code = time_pattern(parts)
    if family == FAMILY_TIME:
        return time_code in (TIME_PATTERNS if partial else ("HM", "HMS"))
    date_code = date_pattern(parts)
    if family == FAMILY_DATE:
        return time_code == "" and date_code in (DATE_PATTERNS if partial else ("YMD",))
    if not partial:
        return date_code == "YMD" and time_code in ("HM", "HMS")
    if date_code not in DATE_PATTERNS:
        return False
    if time_code == "":
        return True
    return date_code in DATE_PATTERNS_WITH_TIME and time_code in TIME_PATTERNS


# ISO 8601 reduced precision: "2024", "2024-05", "2024-05-18", "--05-18",
# optionally followed by "T14", "T14:30", "T14:30:05" (datetimes).
_PARTIAL_DATE = re.compile(
    r"^\s*(?:(?P<year>[+-]?\d{4,})(?:-(?P<month>\d{2})(?:-(?P<day>\d{2}))?)?"
    r"|--(?P<md_month>\d{2})-(?P<md_day>\d{2}))"
    r"(?:[T ](?P<hour>\d{1,2})(?::(?P<minute>\d{2})(?::(?P<second>\d{2})(?:\.\d+)?)?)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?)?\s*$"
)
_PARTIAL_TIME = re.compile(
    r"^\s*T?(?P<hour>\d{1,2})(?::(?P<minute>\d{2})(?::(?P<second>\d{2})(?:\.\d+)?)?)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?\s*$"
)


def _int_or_none(value) -> int | None:
    return None if value is None else int(value)


def is_iso_value(value: str, family: int) -> bool:
    """Whether ``value`` is ISO text (full or reduced precision) of ``family``."""

    pattern = _PARTIAL_TIME if family == FAMILY_TIME else _PARTIAL_DATE
    return bool(pattern.match(value))


def _full_parts(value: Any, family: int) -> TemporalParts | None:
    """Components of a full date/time/datetime value (wall clock)."""

    if family == FAMILY_TIME:
        if isinstance(value, (int, float)):
            return None
        seconds = parse_text_value(str(value), "time")
        if seconds is None:
            return None
        seconds = int(round(seconds)) % 86400
        return TemporalParts(None, None, None, seconds // 3600, seconds % 3600 // 60, seconds % 60)
    moment = _coerce_datetime(value)
    if moment is None:
        return None
    if family == FAMILY_DATE:
        return TemporalParts(moment.year, moment.month, moment.day)
    return TemporalParts(
        moment.year, moment.month, moment.day, moment.hour, moment.minute, moment.second,
    )


def temporal_parts(value: Any, family: int, *, partial: bool = True) -> TemporalParts | None:
    """Calendar components of a gold value, ``None`` for components not written.

    Accepts ISO reduced-precision strings ("2024-05", "--05-18", "14") and
    falls back to full values for anything the date parser understands.
    """

    if value is None or family not in TEMPORAL_FAMILIES:
        return None
    if isinstance(value, str):
        if family == FAMILY_TIME:
            match = _PARTIAL_TIME.match(value)
            if match:
                parts = TemporalParts(
                    None, None, None,
                    int(match.group("hour")),
                    _int_or_none(match.group("minute")),
                    _int_or_none(match.group("second")),
                )
                return parts if valid_parts(parts, family, partial=partial) else None
        else:
            match = _PARTIAL_DATE.match(value)
            if match:
                if match.group("md_month") is not None:
                    year, month, day = None, int(match.group("md_month")), int(match.group("md_day"))
                else:
                    year = int(match.group("year"))
                    month = _int_or_none(match.group("month"))
                    day = _int_or_none(match.group("day"))
                hour = minute = second = None
                if family == FAMILY_DATETIME:
                    hour = _int_or_none(match.group("hour"))
                    minute = _int_or_none(match.group("minute"))
                    second = _int_or_none(match.group("second"))
                parts = TemporalParts(year, month, day, hour, minute, second)
                return parts if valid_parts(parts, family, partial=partial) else None
    parts = _full_parts(value, family)
    if parts is None or not valid_parts(parts, family, partial=True):
        return None
    return parts


def component_codes(parts: TemporalParts | None, family: int) -> dict[str, int]:
    """Label codes per component (see the module docstring)."""

    codes = dict.fromkeys(COMPONENTS, IGNORE)
    if parts is None or family not in TEMPORAL_FAMILIES:
        return codes
    names = (
        TIME_COMPONENTS if family == FAMILY_TIME
        else DATE_COMPONENTS if family == FAMILY_DATE
        else COMPONENTS
    )
    for name in names:
        value = getattr(parts, name)
        codes[name] = NULL if value is None else int(value)
    if codes["year"] != NULL and codes["year"] != IGNORE and codes["year"] < 1:
        # Years before 1 CE are not supported: no year target.
        codes["year"] = IGNORE
    return codes


def compose_partial(family: int, parts: TemporalParts) -> str | None:
    """ISO reduced-precision text of the written components."""

    time_text = None
    if parts.hour is not None:
        time_text = f"{int(parts.hour):02d}"
        if parts.minute is not None:
            time_text += f":{int(parts.minute):02d}"
            if parts.second is not None:
                time_text += f":{int(parts.second):02d}"
    if family == FAMILY_TIME:
        return time_text
    if parts.year is not None:
        date_text = format_year(int(parts.year))
        if parts.month is not None:
            date_text += f"-{int(parts.month):02d}"
            if parts.day is not None:
                date_text += f"-{int(parts.day):02d}"
    elif parts.month is not None and parts.day is not None:
        date_text = f"--{int(parts.month):02d}-{int(parts.day):02d}"
    else:
        return None
    if family == FAMILY_DATETIME and time_text is not None:
        return f"{date_text}T{time_text}"
    return date_text


def year_class(year: int | None, first: int, last: int) -> int:
    """Categorical year class: offset in range, then out of range, then null."""

    span = last - first + 1
    if year is None:
        return span + 1
    if first <= year <= last:
        return year - first
    return span


__all__ = [
    "COMPONENTS",
    "FAMILY_DATE",
    "FAMILY_DATETIME",
    "FAMILY_NAMES",
    "FAMILY_NONE",
    "FAMILY_NUMBER",
    "FAMILY_TIME",
    "IGNORE",
    "NULL",
    "TEMPORAL_FAMILIES",
    "TemporalParts",
    "VALUE_TYPE_FAMILY",
    "component_codes",
    "compose_partial",
    "days_in_month",
    "is_iso_value",
    "temporal_parts",
    "valid_parts",
    "year_class",
]
