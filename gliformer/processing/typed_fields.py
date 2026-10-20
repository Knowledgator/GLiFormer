"""Typed structuring fields: specifications, value codecs, and targets.

Structuring fields come in three kinds:

* ``span`` fields (``text`` and lists) are copied from the input by the
  entity-first span tagger, exactly as before.
* ``class`` fields choose labels from a closed option list.  Options are
  rendered in the prompt after their field as ``[OPTION] label`` markers and
  scored against a (record, field) representation.
* ``quantity`` fields (numbers, durations, dates, times, datetimes, and their
  estimate/forecast variants) are predicted by one distribution head.

Every quantity is mapped to a real number ``x`` by a type codec, then to a
unit coordinate ``u`` in ``[0, 1]``:

* bounded fields (``$min``/``$max``) use a linear or log map of ``x``;
* unbounded fields predict an offset ``x - origin`` through a symmetric
  ``asinh`` map.  The origin is a span in the text tagged for the same field
  (``anchor``), the request's reference date (``reference``), or a fixed
  default (``absolute``).  A value copied from the text is simply an anchor
  with a zero offset.

The head predicts a histogram over ``num_bins`` bins of ``u`` plus an offset
inside every bin, so any statistic (point value, median, any quantile) can be
read from the same distribution.  This module is torch-free; it only builds
targets and decodes predictions.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from datetime import time as clock_time
from typing import Any

SPAN_KIND = "span"
CLASS_KIND = "class"
QUANTITY_KIND = "quantity"

# Integer codes shared by processor tensors and the head.
FIELD_KIND_IDS = {SPAN_KIND: 0, CLASS_KIND: 1, QUANTITY_KIND: 2}

ORIGIN_ABSOLUTE = 0
ORIGIN_REFERENCE = 1
ORIGIN_ANCHOR = 2
ORIGIN_NAMES = {
    ORIGIN_ABSOLUTE: "absolute",
    ORIGIN_REFERENCE: "reference",
    ORIGIN_ANCHOR: "anchor",
}

# Canonical type names after alias resolution.
SPAN_TYPES = frozenset({"text", "list", "bool"})
CLASS_TYPES = frozenset({"class"})
QUANTITY_BASE_TYPES = frozenset({"number", "duration", "date", "time", "datetime"})
QUANTITY_TYPES = QUANTITY_BASE_TYPES | {"estimate", "forecast"}
TEMPORAL_TYPES = frozenset({"date", "time", "datetime"})
TYPED_NAMES = SPAN_TYPES | CLASS_TYPES | QUANTITY_TYPES

TYPE_ALIASES = {
    "str": "text",
    "string": "text",
    "text": "text",
    "span": "text",
    "int": "number",
    "integer": "number",
    "float": "number",
    "number": "number",
    "numeric": "number",
    "bool": "bool",
    "boolean": "bool",
    "class": "class",
    "enum": "class",
    "choice": "class",
    "choices": "class",
    "category": "class",
    "label": "class",
    "labels": "class",
    "date": "date",
    "time": "time",
    "datetime": "datetime",
    "date-time": "datetime",
    "date_time": "datetime",
    "timestamp": "datetime",
    "duration": "duration",
    "estimate": "estimate",
    "forecast": "forecast",
    "list": "list",
    "array": "list",
}

# Options accepted on a field descriptor, both with and without ``$``.
SPEC_OPTION_KEYS = frozenset(
    {
        "type",
        "description",
        "options",
        "labels",
        "enum",
        "multi",
        "min",
        "max",
        "minimum",
        "maximum",
        "step",
        "scale",
        "unit",
        "decimals",
        "integer",
        "horizon",
        "interval",
        "stats",
        "quantiles",
        "items",
        "base",
    }
)
DESCRIPTOR_OPTION_KEYS = frozenset(f"${key}" for key in SPEC_OPTION_KEYS)

DEFAULT_POINT_STATS = ("value",)
DEFAULT_INTERVAL_STATS = ("value", "median", "p10", "p90")

# Offsets are measured in these units before the asinh map.
_UNBOUNDED_SCALE = {
    # base type: (scale s, max |offset| / s)
    "number": (1.0, 1e12),
    "duration": (1.0, 1e9),
    "date": (1.0, 200 * 365.25),  # days
    "datetime": (60.0, 200 * 365.25 * 1440),  # seconds, scale one minute
    "time": (60.0, 1440.0),  # seconds, scale one minute
}

_EPOCH_DATE = date(1970, 1, 1)
_EPOCH_DATETIME = datetime(1970, 1, 1)
# Used as the absolute origin of temporal fields when no reference exists.
_DEFAULT_REFERENCE_DATE = date(2000, 1, 1)

_SECONDS = {
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "m": 60.0,
    "min": 60.0,
    "mins": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hrs": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "d": 86400.0,
    "day": 86400.0,
    "days": 86400.0,
    "w": 604800.0,
    "wk": 604800.0,
    "wks": 604800.0,
    "week": 604800.0,
    "weeks": 604800.0,
    "month": 2629800.0,
    "months": 2629800.0,
    "mo": 2629800.0,
    "mos": 2629800.0,
    "quarter": 7889400.0,
    "quarters": 7889400.0,
    "y": 31557600.0,
    "yr": 31557600.0,
    "yrs": 31557600.0,
    "year": 31557600.0,
    "years": 31557600.0,
}

_NUMBER_WORD_MULTIPLIERS = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "mn": 1e6,
    "million": 1e6,
    "millions": 1e6,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
    "billions": 1e9,
    "t": 1e12,
    "trillion": 1e12,
}

_NUMBER_PATTERN = (
    r"(?<![\w.])[-+−]?(?:\d{1,3}(?:[,   ]\d{3})+|\d+)(?:\.\d+)?"
    r"|(?<![\w.])[-+−]?\.\d+"
)
NUMBER_RE = re.compile(
    r"(?P<num>" + _NUMBER_PATTERN + r")"
    r"(?:\s?(?P<mult>k|thousand|mn|million|millions|bn|billion|billions|"
    r"trillion|[mbt])\b)?",
    re.IGNORECASE,
)
_UNIT_WORDS = "|".join(sorted(_SECONDS, key=len, reverse=True))
DURATION_RE = re.compile(
    r"(?P<num>" + _NUMBER_PATTERN + r")\s?-?\s?(?P<unit>" + _UNIT_WORDS + r")\b",
    re.IGNORECASE,
)
_MONTHS = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\.?"
)
DATE_RE = re.compile(
    r"\b(?:"
    r"\d{4}-\d{1,2}-\d{1,2}"
    r"|\d{4}/\d{1,2}/\d{1,2}"
    r"|\d{1,2}[/.-]\d{1,2}[/.-]\d{4}"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MONTHS + r",?\s+\d{4}"
    r"|" + _MONTHS + r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
    r"|" + _MONTHS + r"\s+\d{4}"
    r")\b",
    re.IGNORECASE,
)
TIME_RE = re.compile(
    r"\b\d{1,2}(?::\d{2}){1,2}(?:\s?[AaPp]\.?[Mm]\b)?(?!\d)"
    r"|\b\d{1,2}\s?[AaPp]\.?[Mm]\b",
)
DATETIME_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?"
)

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%B %d, %Y",
    "%B %d %Y",
    "%b %d, %Y",
    "%b %d %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%d %B, %Y",
    "%d %b, %Y",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%B %Y",
    "%b %Y",
)


def normalize_typed_name(type_name: Any) -> str:
    """Resolve aliases such as ``int`` or ``date-time`` to canonical names."""

    name = str(type_name or "text").strip().lower()
    return TYPE_ALIASES.get(name, name)


def stat_level(name: str) -> float | None:
    """Return the quantile level of a statistic name, ``None`` for points."""

    key = str(name).strip().lower()
    if key in {"value", "point", "mean", "mode"}:
        return None
    if key == "median":
        return 0.5
    match = re.fullmatch(r"[pq](\d+(?:\.\d+)?)", key)
    if match:
        level = float(match.group(1)) / 100.0
        if 0.0 < level < 1.0:
            return level
    raise ValueError(f"Unknown quantity statistic {name!r}")


def quantile_stat_name(level: float) -> str:
    """Name a quantile level the way targets and outputs spell it."""

    if abs(level - 0.5) < 1e-9:
        return "median"
    percent = level * 100.0
    if abs(percent - round(percent)) < 1e-9:
        return f"p{int(round(percent))}"
    return f"p{percent:g}"


def _optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    return float(value)


@dataclass
class TypedFieldSpec:
    """Typed description of one structuring field.

    ``type`` is the canonical public type.  ``estimate`` and ``forecast`` are
    quantities over ``base`` (``number`` by default) with interval statistics
    and, for forecasts, a horizon.
    """

    type: str = "text"
    description: str | None = None
    options: tuple[str, ...] = ()
    multi: bool = True
    minimum: Any = None
    maximum: Any = None
    step: float | None = None
    scale: str | None = None
    unit: str | None = None
    decimals: int | None = None
    integer: bool = False
    horizon: int = 1
    interval: str | None = None
    stats: tuple[str, ...] = ()
    items: str | None = None
    base: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.type = normalize_typed_name(self.type)
        if self.type not in TYPED_NAMES:
            raise ValueError(f"Unknown structuring field type {self.type!r}")
        self.options = tuple(str(option) for option in self.options or ())
        if self.type in CLASS_TYPES:
            if not self.options:
                raise ValueError("A class field needs at least one option")
            if len(set(self.options)) != len(self.options):
                raise ValueError("Class field options must be distinct")
        self.multi = bool(self.multi)
        self.integer = bool(self.integer)
        if self.decimals is not None:
            self.decimals = int(self.decimals)
            if not 0 <= self.decimals <= 12:
                raise ValueError("decimals must be in [0, 12]")
        self.step = _optional_float(self.step, "step")
        if self.step is not None and self.step <= 0:
            raise ValueError("step must be positive")
        if self.scale is not None:
            self.scale = str(self.scale).lower()
            if self.scale not in {"linear", "log"}:
                raise ValueError("scale must be 'linear' or 'log'")
        self.horizon = int(self.horizon or 1)
        if self.type == "forecast" and not 1 <= self.horizon <= 64:
            raise ValueError("forecast horizon must be in [1, 64]")
        if self.type != "forecast":
            self.horizon = 1
        if self.base is not None:
            self.base = normalize_typed_name(self.base)
            if self.base not in QUANTITY_BASE_TYPES:
                raise ValueError(f"Unsupported quantity base {self.base!r}")
        if self.items is not None:
            self.items = normalize_typed_name(self.items)
        if self.type in QUANTITY_TYPES:
            if not self.stats:
                self.stats = (
                    DEFAULT_INTERVAL_STATS
                    if self.type in {"estimate", "forecast"}
                    else DEFAULT_POINT_STATS
                )
            self.stats = tuple(dict.fromkeys(str(name).lower() for name in self.stats))
            for name in self.stats:
                stat_level(name)
            if self.bounded:
                low = self.encode(self.minimum)
                high = self.encode(self.maximum)
                if low is None or high is None or not low < high:
                    raise ValueError("min must be below max")
                if self.scale == "log" and low <= 0:
                    raise ValueError("log scale needs a positive min")
        else:
            self.stats = ()

    # -- classification ----------------------------------------------------

    @property
    def kind(self) -> str:
        if self.type in CLASS_TYPES:
            return CLASS_KIND
        if self.type in QUANTITY_TYPES:
            return QUANTITY_KIND
        return SPAN_KIND

    @property
    def value_type(self) -> str:
        """Base type of a quantity: number, duration, date, time, datetime."""

        if self.type in {"estimate", "forecast"}:
            return self.base or "number"
        return self.type

    @property
    def bounded(self) -> bool:
        return (
            self.kind == QUANTITY_KIND
            and self.minimum is not None
            and self.maximum is not None
        )

    @property
    def temporal(self) -> bool:
        return self.value_type in TEMPORAL_TYPES

    @property
    def point_stats(self) -> bool:
        return any(stat_level(name) is None for name in self.stats)

    @property
    def quantile_levels(self) -> tuple[float, ...]:
        levels = {stat_level(name) for name in self.stats}
        return tuple(sorted(level for level in levels if level is not None))

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the public ``$``-prefixed descriptor form."""

        result: dict[str, Any] = {"$type": self.type}
        if self.description:
            result["$description"] = self.description
        if self.options:
            result["$options"] = list(self.options)
            if not self.multi:
                result["$multi"] = False
        for key, value in (
            ("$min", self.minimum),
            ("$max", self.maximum),
            ("$step", self.step),
            ("$scale", self.scale),
            ("$unit", self.unit),
            ("$decimals", self.decimals),
            ("$interval", self.interval),
            ("$items", self.items),
            ("$base", self.base),
        ):
            if value is not None:
                result[key] = value
        if self.integer:
            result["$integer"] = True
        if self.type == "forecast":
            result["$horizon"] = self.horizon
        if self.kind == QUANTITY_KIND:
            default = (
                DEFAULT_INTERVAL_STATS
                if self.type in {"estimate", "forecast"}
                else DEFAULT_POINT_STATS
            )
            if tuple(self.stats) != default:
                result["$stats"] = list(self.stats)
        return result

    @classmethod
    def from_value(cls, spec: Any) -> TypedFieldSpec:
        """Build from a spec, a descriptor mapping, or a bare type name."""

        if isinstance(spec, TypedFieldSpec):
            return spec
        if isinstance(spec, str):
            return cls(type=spec)
        if not isinstance(spec, Mapping):
            raise TypeError(f"Cannot build a typed field from {spec!r}")
        values: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for raw_key, value in spec.items():
            key = str(raw_key)
            bare = key[1:] if key.startswith("$") else key
            if bare in SPEC_OPTION_KEYS:
                values[bare] = value
            else:
                extra[key] = value
        options = values.get("options", values.get("labels", values.get("enum")))
        type_name = values.get("type")
        if type_name is None:
            type_name = "class" if options is not None else "text"
        type_name = normalize_typed_name(type_name)
        quantiles = values.get("quantiles")
        stats = values.get("stats")
        if quantiles is not None:
            stats = list(stats or ["value"]) + [
                quantile_stat_name(float(level)) for level in quantiles
            ]
        integer = values.get("integer")
        if integer is None and str(spec.get("$type", spec.get("type", ""))).lower() in {
            "int",
            "integer",
        }:
            integer = True
        return cls(
            type=type_name,
            description=values.get("description"),
            options=tuple(options or ()) if type_name in CLASS_TYPES else (),
            multi=values.get("multi", True),
            minimum=values.get("min", values.get("minimum")),
            maximum=values.get("max", values.get("maximum")),
            step=values.get("step"),
            scale=values.get("scale"),
            unit=values.get("unit"),
            decimals=values.get("decimals"),
            integer=bool(integer),
            horizon=values.get("horizon", 1),
            interval=values.get("interval"),
            stats=tuple(stats or ()),
            items=values.get("items"),
            base=values.get("base"),
            extra=extra,
        )

    # -- prompt ------------------------------------------------------------

    def prompt_suffix(self) -> str:
        """Plain-word type description rendered after the field name."""

        if self.type == "text":
            return ""
        parts = [self.type if self.type != "list" else f"list of {self.items or 'text'}"]
        if self.kind == CLASS_KIND:
            parts.append("multiple" if self.multi else "single")
        if self.kind == QUANTITY_KIND:
            if self.type in {"estimate", "forecast"} and self.base:
                parts.append(self.base)
            if self.unit:
                parts.append(str(self.unit))
            if self.bounded:
                parts.append(f"{_plain(self.minimum)} - {_plain(self.maximum)}")
            if self.step is not None:
                parts.append(f"step {_plain(self.step)}")
            if self.scale == "log":
                parts.append("log")
            if self.integer:
                parts.append("integer")
            elif self.decimals is not None:
                parts.append(f"{self.decimals} decimals")
            if self.type == "forecast":
                interval = f" {self.interval}" if self.interval else ""
                parts.append(f"{self.horizon}{interval} steps")
        return "( " + " , ".join(parts) + " )"

    # -- codecs ------------------------------------------------------------

    def encode(self, value: Any) -> float | None:
        """Map a JSON value to the real line (``None`` when impossible)."""

        return encode_value(value, self.value_type, unit=self.unit)

    def decode(self, x: float) -> Any:
        """Map a real number back to a typed JSON value with rounding."""

        value_type = self.value_type
        if value_type in {"number", "duration"}:
            if self.bounded:
                low, high = self.encode(self.minimum), self.encode(self.maximum)
                x = min(max(x, low), high)
                if self.step is not None:
                    x = low + round((x - low) / self.step) * self.step
                    x = min(max(x, low), high)
            elif self.step is not None:
                x = round(x / self.step) * self.step
            if self.integer:
                return int(round(x))
            if self.decimals is not None:
                return round(x, self.decimals)
            if self.step is not None:
                decimals = max(0, -int(math.floor(math.log10(self.step))) + 6)
                x = round(x, decimals)
            return _clean_float(x)
        if value_type == "date":
            if self.bounded:
                x = min(max(x, self.encode(self.minimum)), self.encode(self.maximum))
            return decode_value(x, "date")
        if value_type == "datetime":
            return decode_value(x, "datetime")
        return decode_value(x, "time")

    def parse_text(self, text: str) -> float | None:
        """Parse a span copied from the text into the real line."""

        return parse_text_value(text, self.value_type, unit=self.unit)

    # -- normalization -------------------------------------------------------

    def to_unit(self, x: float, origin: float | None) -> float:
        """Map ``x`` (relative to ``origin`` when unbounded) into ``[0, 1]``."""

        if self.bounded:
            low, high = self.encode(self.minimum), self.encode(self.maximum)
            if self.scale == "log":
                x = max(x, low)
                u = (math.log(x) - math.log(low)) / (math.log(high) - math.log(low))
            else:
                u = (x - low) / (high - low)
            return min(max(u, 0.0), 1.0)
        scale, limit = _UNBOUNDED_SCALE[self.value_type]
        offset = x - (origin if origin is not None else 0.0)
        radius = math.asinh(limit)
        u = 0.5 + 0.5 * math.asinh(offset / scale) / radius
        return min(max(u, 0.0), 1.0)

    def from_unit(self, u: float, origin: float | None) -> float:
        u = min(max(float(u), 0.0), 1.0)
        if self.bounded:
            low, high = self.encode(self.minimum), self.encode(self.maximum)
            if self.scale == "log":
                return math.exp(math.log(low) + u * (math.log(high) - math.log(low)))
            return low + u * (high - low)
        scale, limit = _UNBOUNDED_SCALE[self.value_type]
        radius = math.asinh(limit)
        offset = scale * math.sinh((2.0 * u - 1.0) * radius)
        return (origin if origin is not None else 0.0) + offset

    def default_origin(self) -> float:
        """Absolute origin used when there is no anchor and no reference."""

        value_type = self.value_type
        if value_type == "date":
            return float((_DEFAULT_REFERENCE_DATE - _EPOCH_DATE).days)
        if value_type == "datetime":
            return float(
                (datetime.combine(_DEFAULT_REFERENCE_DATE, clock_time()) - _EPOCH_DATETIME)
                .total_seconds()
            )
        return 0.0

    def reference_origin(self, reference: Any) -> float | None:
        """Encode a request reference date/datetime for this field."""

        if reference is None or not self.temporal:
            return None
        value_type = self.value_type
        if value_type == "time":
            # Times of day are already absolute within a day.
            return None
        parsed = _coerce_datetime(reference)
        if parsed is None:
            return None
        if value_type == "date":
            return float((parsed.date() - _EPOCH_DATE).days)
        return float((parsed - _EPOCH_DATETIME).total_seconds())

    def resolve_origin(
        self,
        anchor_x: float | None,
        reference: Any = None,
    ) -> tuple[int, float | None]:
        """Pick the origin mode and value for one prediction."""

        if self.bounded:
            return ORIGIN_ABSOLUTE, None
        if anchor_x is not None:
            return ORIGIN_ANCHOR, anchor_x
        reference_x = self.reference_origin(reference)
        if reference_x is not None:
            return ORIGIN_REFERENCE, reference_x
        return ORIGIN_ABSOLUTE, self.default_origin()


def _plain(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _clean_float(value: float) -> float:
    """Drop floating-point noise beyond 10 significant digits."""

    value = float(value)
    if not math.isfinite(value) or value == 0.0:
        return value
    digits = 10 - int(math.floor(math.log10(abs(value)))) - 1
    return float(round(value, max(digits, 0)))


# -- value codecs ---------------------------------------------------------


def _coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, date):
        result = datetime.combine(value, clock_time())
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            result = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            parsed = _parse_date_text(text)
            if parsed is None:
                return None
            result = datetime.combine(parsed, clock_time())
    else:
        return None
    if result.tzinfo is not None:
        result = result.astimezone(timezone.utc).replace(tzinfo=None)
    return result


def _time_seconds(value: Any) -> float | None:
    if isinstance(value, clock_time):
        return value.hour * 3600.0 + value.minute * 60.0 + value.second
    if isinstance(value, datetime):
        return _time_seconds(value.time())
    if not isinstance(value, str):
        return None
    text = value.strip()
    try:
        parsed = clock_time.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return _parse_time_text(text)
    return parsed.hour * 3600.0 + parsed.minute * 60.0 + parsed.second + parsed.microsecond / 1e6


def encode_value(value: Any, value_type: str, *, unit: str | None = None) -> float | None:
    """Encode a JSON value of ``value_type`` to a float."""

    if value is None:
        return None
    if value_type in {"number", "duration"}:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value) if math.isfinite(value) else None
        if isinstance(value, str):
            return parse_text_value(value, value_type, unit=unit)
        return None
    if value_type == "date":
        if isinstance(value, datetime):
            value = value.date()
        if isinstance(value, date):
            return float((value - _EPOCH_DATE).days)
        if isinstance(value, str):
            try:
                return float((date.fromisoformat(value.strip()[:10]) - _EPOCH_DATE).days)
            except ValueError:
                parsed = _parse_date_text(value)
                return float((parsed - _EPOCH_DATE).days) if parsed else None
        return None
    if value_type == "datetime":
        parsed = _coerce_datetime(value)
        return float((parsed - _EPOCH_DATETIME).total_seconds()) if parsed else None
    if value_type == "time":
        return _time_seconds(value)
    raise ValueError(f"Unsupported quantity type {value_type!r}")


def decode_value(x: float, value_type: str) -> Any:
    """Decode a float to an ISO date, datetime, or time string."""

    if value_type == "date":
        return (_EPOCH_DATE + timedelta(days=int(round(x)))).isoformat()
    if value_type == "datetime":
        seconds = int(round(x))
        return (_EPOCH_DATETIME + timedelta(seconds=seconds)).isoformat()
    if value_type == "time":
        seconds = int(round(x)) % 86400
        hours, rest = divmod(seconds, 3600)
        minutes, secs = divmod(rest, 60)
        if secs:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{hours:02d}:{minutes:02d}"
    return _clean_float(float(x))


def _parse_number_match(match: re.Match) -> float | None:
    raw = match.group("num").replace("−", "-")
    raw = re.sub(r"[,   ]", "", raw)
    try:
        number = float(raw)
    except ValueError:
        return None
    multiplier = match.groupdict().get("mult")
    if multiplier:
        number *= _NUMBER_WORD_MULTIPLIERS.get(multiplier.lower(), 1.0)
    return number


def _unit_seconds(unit: str | None) -> float | None:
    if not unit:
        return None
    return _SECONDS.get(str(unit).strip().lower())


def _parse_date_text(text: str) -> date | None:
    cleaned = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text.strip())
    cleaned = re.sub(r"\bof\s+", "", cleaned)
    cleaned = cleaned.replace("Sept", "Sep").replace(".", " ") if not re.fullmatch(
        r"\d{1,2}\.\d{1,2}\.\d{4}", cleaned
    ) else cleaned
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    match = re.fullmatch(r"Q([1-4])\s+(\d{4})", cleaned, re.IGNORECASE)
    if match:
        return date(int(match.group(2)), 3 * int(match.group(1)) - 2, 1)
    match = re.fullmatch(r"(\d{4})", cleaned)
    if match:
        return date(int(match.group(1)), 1, 1)
    return None


def _parse_time_text(text: str) -> float | None:
    match = re.fullmatch(
        r"\s*(\d{1,2})(?::(\d{2}))?(?::(\d{2}))?\s*([AaPp])?\.?(?:[Mm]\.?)?\s*",
        text,
    )
    if not match or (match.group(2) is None and match.group(4) is None):
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    meridiem = (match.group(4) or "").lower()
    if meridiem:
        if not 1 <= hours <= 12:
            return None
        hours = hours % 12 + (12 if meridiem == "p" else 0)
    if hours > 23 or minutes > 59 or seconds > 59:
        return None
    return hours * 3600.0 + minutes * 60.0 + seconds


def parse_text_value(text: str, value_type: str, *, unit: str | None = None) -> float | None:
    """Parse free text (a tagged span) into a float of ``value_type``."""

    if not isinstance(text, str) or not text.strip():
        return None
    # Span text rebuilt from words can carry spaces around punctuation
    # ("March 3 , 2024", "$ 1 , 250"); undo that before parsing.
    text = re.sub(r"\s+([,.;:)%])", r"\1", text.strip())
    text = re.sub(r"([($])\s+", r"\1", text)
    text = re.sub(r"(\d),\s+(\d{3})\b", r"\1,\2", text)
    text = re.sub(r"(\d)\.\s+(\d)", r"\1.\2", text)
    text = re.sub(r"\s+", " ", text)
    if value_type == "number":
        match = NUMBER_RE.search(text)
        return _parse_number_match(match) if match else None
    if value_type == "duration":
        target = _unit_seconds(unit)
        match = DURATION_RE.search(text)
        if match and target:
            number = _parse_number_match(match)
            source = _SECONDS.get(match.group("unit").lower())
            if number is not None and source:
                return number * source / target
        match = NUMBER_RE.search(text)
        return _parse_number_match(match) if match else None
    if value_type == "date":
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            try:
                return float((date.fromisoformat(text) - _EPOCH_DATE).days)
            except ValueError:
                return None
        parsed = _parse_date_text(text)
        return float((parsed - _EPOCH_DATE).days) if parsed else None
    if value_type == "datetime":
        parsed = _coerce_datetime(text)
        if parsed is not None:
            return float((parsed - _EPOCH_DATETIME).total_seconds())
        date_match = DATE_RE.search(text)
        time_match = TIME_RE.search(text)
        if date_match and time_match:
            day = _parse_date_text(date_match.group(0))
            seconds = _parse_time_text(time_match.group(0))
            if day is not None and seconds is not None:
                base = datetime.combine(day, clock_time()) - _EPOCH_DATETIME
                return base.total_seconds() + seconds
        return None
    if value_type == "time":
        seconds = _parse_time_text(text)
        if seconds is not None:
            return seconds
        match = TIME_RE.search(text)
        return _parse_time_text(match.group(0)) if match else None
    raise ValueError(f"Unsupported quantity type {value_type!r}")


def candidate_spans(text: str, value_type: str) -> list[tuple[int, int]]:
    """Character spans that may hold a value of ``value_type``."""

    patterns = {
        "number": (NUMBER_RE,),
        "duration": (DURATION_RE, NUMBER_RE),
        "date": (DATE_RE,),
        "datetime": (DATETIME_RE, DATE_RE),
        "time": (TIME_RE,),
    }[value_type]
    spans: list[tuple[int, int]] = []
    seen = set()
    if value_type == "datetime":
        # Dates and clock times are often written apart: "March 3, 2024 at
        # 5:30 pm" or "5:30 pm on March 3, 2024". Pair nearby mentions.
        times = list(TIME_RE.finditer(text))
        for date_match in DATE_RE.finditer(text):
            for time_match in times:
                if 0 <= time_match.start() - date_match.end() <= 30 or (
                    0 <= date_match.start() - time_match.end() <= 30
                ):
                    start = min(date_match.start(), time_match.start())
                    end = max(date_match.end(), time_match.end())
                    between = text[min(date_match.end(), time_match.end()):max(
                        date_match.start(), time_match.start()
                    )]
                    if re.search(r"[.;\n]", between):
                        continue
                    if (start, end) not in seen:
                        seen.add((start, end))
                        spans.append((start, end))
    for pattern in patterns:
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if span not in seen and match.group(0).strip():
                seen.add(span)
                spans.append(span)
    return spans


# Largest offset accepted from a non-matching temporal anchor.
_MAX_ANCHOR_OFFSET = {
    "date": 3 * 366.0,  # days
    "datetime": 3 * 366.0 * 86400.0,  # seconds
    "time": 86400.0,  # seconds
}


def find_anchor(
    text: str,
    spec: TypedFieldSpec,
    target_x: float,
    *,
    snippets: Iterable[str] = (),
    allow_offset: bool = True,
) -> dict[str, Any] | None:
    """Find the span a quantity is read or derived from.

    Evidence snippets are searched first, then the whole text.  A span whose
    parsed value equals ``target_x`` is preferred (zero offset).  For dates
    and times without an exact match, the closest parseable span becomes the
    anchor, so "due two weeks after March 3" is learned as anchor + 14 days.
    Returns ``{"text", "start", "end", "x"}`` with character offsets and the
    anchor's own parsed value ``x``.
    """

    value_type = spec.value_type
    tolerance = {
        "number": 1e-6,
        "duration": 1e-6,
        "date": 0.5,
        "time": 0.5,
        "datetime": 0.5,
    }[value_type]

    def matches(x: float | None) -> bool:
        if x is None:
            return False
        if value_type in {"number", "duration"}:
            return abs(x - target_x) <= tolerance * max(1.0, abs(target_x))
        return abs(x - target_x) <= tolerance

    evidence_regions: list[tuple[int, int]] = []
    for snippet in snippets or ():
        if not isinstance(snippet, str) or not snippet:
            continue
        start = text.find(snippet)
        while start >= 0:
            evidence_regions.append((start, start + len(snippet)))
            start = text.find(snippet, start + 1)

    def candidates(regions):
        for region_start, region_end in regions:
            region = text[region_start:region_end]
            for start, end in candidate_spans(region, value_type):
                span_text = region[start:end]
                parsed = spec.parse_text(span_text)
                if parsed is not None:
                    yield span_text, region_start + start, region_start + end, parsed

    whole = [(0, len(text))]
    for regions in (evidence_regions, whole):
        for span_text, start, end, parsed in candidates(regions):
            if matches(parsed):
                return {"text": span_text, "start": start, "end": end, "x": parsed}

    limit = _MAX_ANCHOR_OFFSET.get(value_type)
    if not allow_offset or limit is None:
        return None
    for regions in (evidence_regions, whole):
        best = None
        for span_text, start, end, parsed in candidates(regions):
            offset = abs(parsed - target_x)
            if offset <= limit and (best is None or offset < best[0]):
                best = (offset, span_text, start, end, parsed)
        if best is not None:
            _, span_text, start, end, parsed = best
            return {"text": span_text, "start": start, "end": end, "x": parsed}
    return None


# -- targets -------------------------------------------------------------


def target_cdf(
    points: Sequence[tuple[float, float]],
    num_bins: int,
) -> list[float]:
    """Target CDF at the right edge of every bin.

    ``points`` are ``(level, u)`` pairs.  One point (or one distinct ``u``)
    is a point mass; two or more are joined linearly with tails extended at
    the edge slopes and clipped to ``[0, 1]``.
    """

    edges = [(index + 1) / num_bins for index in range(num_bins)]
    cleaned = sorted(
        {(float(level), min(max(float(u), 0.0), 1.0)) for level, u in points}
    )
    if not cleaned:
        raise ValueError("target_cdf needs at least one point")
    levels = [level for level, _ in cleaned]
    # Enforce a non-decreasing quantile function even for noisy labels.
    values = sorted(u for _, u in cleaned)
    if len(cleaned) == 1 or values[-1] - values[0] < 1e-9:
        point = values[len(values) // 2]
        return [1.0 if edge >= point - 1e-12 else 0.0 for edge in edges]

    def interpolate(u: float) -> float:
        if u <= values[0]:
            slope = _segment_slope(values[0], values[1], levels[0], levels[1])
            return levels[0] + slope * (u - values[0])
        if u >= values[-1]:
            slope = _segment_slope(values[-2], values[-1], levels[-2], levels[-1])
            return levels[-1] + slope * (u - values[-1])
        for index in range(len(values) - 1):
            left, right = values[index], values[index + 1]
            if left <= u <= right:
                if right - left < 1e-12:
                    return levels[index + 1]
                weight = (u - left) / (right - left)
                return levels[index] + weight * (levels[index + 1] - levels[index])
        return levels[-1]

    return [min(max(interpolate(edge), 0.0), 1.0) for edge in edges]


def _segment_slope(u_left: float, u_right: float, q_left: float, q_right: float) -> float:
    width = max(u_right - u_left, 1e-6)
    return (q_right - q_left) / width


@dataclass
class QuantityTarget:
    """Unit-space supervision for one (record, field) quantity."""

    present: bool
    origin_mode: int = ORIGIN_ABSOLUTE
    origin: float | None = None
    # Per horizon step: (level, u) points and an optional point value.
    points: list[list[tuple[float, float]]] = field(default_factory=list)
    point_u: list[float | None] = field(default_factory=list)


def _stat_series(value: Any, horizon: int) -> list[Any] | None:
    if horizon == 1:
        if isinstance(value, list):
            return value[:1] if len(value) == 1 else None
        return [value]
    if not isinstance(value, list) or len(value) != horizon:
        return None
    return value


def build_quantity_target(
    gold: Any,
    spec: TypedFieldSpec,
    *,
    origin_mode: int,
    origin: float | None,
) -> QuantityTarget | None:
    """Convert a gold JSON value into unit-space supervision.

    Accepted gold shapes: a scalar (number or ISO string), a forecast series
    list, or a mapping of statistic name to scalar/series.  Returns
    ``QuantityTarget(present=False)`` for ``None`` and ``None`` when the
    value cannot be interpreted (so the caller masks the unit).
    """

    if gold is None:
        return QuantityTarget(present=False)
    horizon = spec.horizon
    if isinstance(gold, Mapping):
        stats = {str(key).lower(): value for key, value in gold.items()}
    elif isinstance(gold, list) and horizon > 1:
        stats = {"value": gold}
    else:
        stats = {"value": gold}

    points: list[list[tuple[float, float]]] = [[] for _ in range(horizon)]
    point_u: list[float | None] = [None] * horizon
    median_u: list[float | None] = [None] * horizon
    any_value = False
    for name, raw in stats.items():
        try:
            level = stat_level(name)
        except ValueError:
            continue
        series = _stat_series(raw, horizon)
        if series is None:
            return None
        for step, item in enumerate(series):
            x = spec.encode(item)
            if x is None:
                if item is None:
                    continue
                return None
            u = spec.to_unit(x, origin)
            any_value = True
            if level is None:
                point_u[step] = u
            else:
                points[step].append((level, u))
                if abs(level - 0.5) < 1e-9:
                    median_u[step] = u
    if not any_value:
        return None
    for step in range(horizon):
        if point_u[step] is None and median_u[step] is not None:
            point_u[step] = median_u[step]
        if not points[step] and point_u[step] is not None:
            # A point label alone supervises the CDF as a point mass.
            points[step].append((0.5, point_u[step]))
    return QuantityTarget(
        present=True,
        origin_mode=origin_mode,
        origin=origin,
        points=points,
        point_u=point_u,
    )


# -- decoding ------------------------------------------------------------


def quantile_from_histogram(probs: Sequence[float], level: float) -> float:
    """Unit-space quantile, linear inside the crossing bin."""

    num_bins = len(probs)
    total = 0.0
    for index, mass in enumerate(probs):
        mass = max(float(mass), 0.0)
        if total + mass >= level and mass > 0:
            inside = (level - total) / mass
            return (index + min(max(inside, 0.0), 1.0)) / num_bins
        total += mass
    return 1.0


def point_from_histogram(probs: Sequence[float], offsets: Sequence[float]) -> float:
    """Most likely bin refined by its in-bin offset."""

    num_bins = len(probs)
    best = max(range(num_bins), key=lambda index: probs[index])
    offset = min(max(float(offsets[best]), 0.0), 1.0)
    return (best + offset) / num_bins


def decode_quantity(
    spec: TypedFieldSpec,
    probs_by_step: Sequence[Sequence[float]],
    offsets_by_step: Sequence[Sequence[float]],
    *,
    origin: float | None,
    stats: Sequence[str] | None = None,
) -> Any:
    """Read the requested statistics and decode them to typed values."""

    names = tuple(stats or spec.stats)
    horizon = spec.horizon
    results: dict[str, list[Any]] = {name: [] for name in names}
    for step in range(horizon):
        probs = probs_by_step[step]
        offsets = offsets_by_step[step]
        for name in names:
            level = stat_level(name)
            u = (
                point_from_histogram(probs, offsets)
                if level is None
                else quantile_from_histogram(probs, level)
            )
            results[name].append(spec.decode(spec.from_unit(u, origin)))
    if horizon == 1:
        flat = {name: values[0] for name, values in results.items()}
    else:
        flat = results
    if names == ("value",):
        return flat["value"]
    return flat


# -- schema adapters -------------------------------------------------------


def is_typed_descriptor(value: Any) -> bool:
    """Whether a mapping is a typed field descriptor like ``{"$type": ...}``."""

    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(key, str) and key.startswith("$") for key in value)
        and ("$type" in value or "$options" in value)
        and set(value) <= (DESCRIPTOR_OPTION_KEYS | {"$required", "$default", "$nullable"})
    )


def spec_from_json_schema(prop: Mapping[str, Any]) -> TypedFieldSpec:
    """Convert one JSON Schema property (with ``x-*`` hints) to a spec."""

    field_type = prop.get("x-field-type")
    description = prop.get("description")
    json_type = prop.get("type")
    json_types = set(json_type) if isinstance(json_type, list) else {json_type}
    json_types.discard("null")
    if field_type is None:
        if "array" in json_types and isinstance(prop.get("items"), Mapping) and prop["items"].get("enum"):
            field_type = "class"
        elif prop.get("format") in {"date", "time", "date-time"}:
            field_type = {"date-time": "datetime"}.get(prop["format"], prop["format"])
        elif json_types & {"number", "integer"}:
            field_type = "number"
        elif "array" in json_types:
            field_type = "list"
        else:
            field_type = "text"
    field_type = normalize_typed_name(field_type)
    kwargs: dict[str, Any] = {"type": field_type, "description": description}
    if field_type == "class":
        items = prop.get("items") if isinstance(prop.get("items"), Mapping) else {}
        kwargs["options"] = tuple(items.get("enum") or prop.get("enum") or ())
        kwargs["multi"] = prop.get("type") == "array" or "array" in json_types
    elif field_type in {"number", "duration"}:
        kwargs["integer"] = "integer" in json_types
        if "multipleOf" in prop and not kwargs["integer"]:
            step = float(prop["multipleOf"])
            decimals = max(0, -int(math.floor(math.log10(step) + 1e-9)))
            kwargs["decimals"] = decimals
        kwargs["unit"] = prop.get("x-unit")
    elif field_type in {"estimate", "forecast"}:
        inner = prop.get("properties", {}).get("value", {})
        if field_type == "forecast":
            inner = inner.get("items", {})
        kwargs.update(
            minimum=inner.get("minimum"),
            maximum=inner.get("maximum"),
            step=inner.get("x-step"),
            scale=prop.get("x-scale"),
            unit=prop.get("x-unit"),
        )
        if field_type == "forecast":
            kwargs["horizon"] = prop.get("x-horizon", 1)
            kwargs["interval"] = prop.get("x-interval")
    elif field_type == "list":
        kwargs["items"] = prop.get("x-items") or "text"
    return TypedFieldSpec(**kwargs)


def specs_from_json_schema(schema: Mapping[str, Any]) -> dict[str, TypedFieldSpec]:
    """Typed specs for every (dotted) leaf property of an object schema."""

    result: dict[str, TypedFieldSpec] = {}

    def visit(node: Mapping[str, Any], prefix: tuple[str, ...]) -> None:
        properties = node.get("properties")
        if not isinstance(properties, Mapping):
            return
        for name, prop in properties.items():
            if not isinstance(prop, Mapping):
                continue
            path = prefix + (str(name),)
            json_type = prop.get("type")
            types = set(json_type) if isinstance(json_type, list) else {json_type}
            if "x-field-type" in prop:
                result[".".join(path)] = spec_from_json_schema(prop)
            elif "object" in types and isinstance(prop.get("properties"), Mapping):
                visit(prop, path)
            elif "array" in types and isinstance(prop.get("items"), Mapping) and (
                prop["items"].get("type") == "object"
            ):
                visit(prop["items"], path)
            else:
                result[".".join(path)] = spec_from_json_schema(prop)

    root = schema
    if root.get("type") == "array" and isinstance(root.get("items"), Mapping):
        root = root["items"]
    visit(root, ())
    return result


def specs_to_dicts(specs: Mapping[str, TypedFieldSpec]) -> dict[str, dict[str, Any]]:
    return {label: spec.to_dict() for label, spec in specs.items()}


def specs_from_dicts(raw: Mapping[str, Any] | None) -> dict[str, TypedFieldSpec]:
    if not raw:
        return {}
    return {str(label): TypedFieldSpec.from_value(spec) for label, spec in raw.items()}


def lookup_group_types(
    raw_types: Any,
    *,
    names: Sequence[str],
) -> dict[str, Any]:
    """Return the label -> spec mapping that applies to one schema group.

    ``raw_types`` is either keyed by schema name (``{"$root": {...}}``) or a
    single global ``label -> spec`` mapping.
    """

    if not isinstance(raw_types, Mapping) or not raw_types:
        return {}
    for name in names:
        group = raw_types.get(name)
        if isinstance(group, Mapping) and not is_typed_descriptor(group):
            return dict(group)
    values = list(raw_types.values())
    if all(
        isinstance(value, str) or is_typed_descriptor(value) or isinstance(value, TypedFieldSpec)
        or (isinstance(value, Mapping) and ("type" in value or "$type" in value))
        for value in values
    ):
        return dict(raw_types)
    return {}


__all__ = [
    "CLASS_KIND",
    "FIELD_KIND_IDS",
    "ORIGIN_ABSOLUTE",
    "ORIGIN_ANCHOR",
    "ORIGIN_NAMES",
    "ORIGIN_REFERENCE",
    "QUANTITY_KIND",
    "QuantityTarget",
    "SPAN_KIND",
    "TypedFieldSpec",
    "build_quantity_target",
    "candidate_spans",
    "decode_quantity",
    "decode_value",
    "encode_value",
    "find_anchor",
    "is_typed_descriptor",
    "lookup_group_types",
    "normalize_typed_name",
    "parse_text_value",
    "point_from_histogram",
    "quantile_from_histogram",
    "quantile_stat_name",
    "spec_from_json_schema",
    "specs_from_dicts",
    "specs_from_json_schema",
    "specs_to_dicts",
    "stat_level",
    "target_cdf",
]
