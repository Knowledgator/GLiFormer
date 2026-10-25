"""Typed structuring fields: specifications and value parsing.

Structuring fields come in two kinds:

* ``span`` fields are copied from the input by the entity-first span
  tagger. Besides ``text``, ``list`` and ``bool`` this includes the value
  types ``number``, ``duration``, ``date``, ``time`` and ``datetime``: the
  structuring head tags their span and the normalization head reads the
  value from it.
* ``class`` fields choose labels from a closed option list.  Options are
  rendered in the prompt after their field as ``[OPTION] label`` markers and
  scored against a (record, field) representation.

The text parsers here ground gold values of value fields on the spans they
were read from when training data is prepared.  This module is torch-free.
"""

from __future__ import annotations

import contextvars
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from datetime import date, datetime
from datetime import time as clock_time
from typing import Any

SPAN_KIND = "span"
CLASS_KIND = "class"

# Integer codes shared by processor tensors and the head.
FIELD_KIND_IDS = {SPAN_KIND: 0, CLASS_KIND: 1}

# Canonical type names after alias resolution.
VALUE_TYPES = frozenset({"number", "duration", "date", "time", "datetime"})
SPAN_TYPES = frozenset({"text", "list", "bool"}) | VALUE_TYPES
CLASS_TYPES = frozenset({"class"})
TEMPORAL_TYPES = frozenset({"date", "time", "datetime"})
TYPED_NAMES = SPAN_TYPES | CLASS_TYPES
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
        "unit",
        "decimals",
        "integer",
        "items",
        "dayfirst",
        "partial",
        # ``{"$type": "text", "$format": "date"}`` is read as ``{"$type": "date"}``.
        "format",
    }
)
DESCRIPTOR_OPTION_KEYS = frozenset(f"${key}" for key in SPEC_OPTION_KEYS)

_EPOCH_DATE = date(1970, 1, 1)
_EPOCH_DATETIME = datetime(1970, 1, 1)

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
_SUPERSCRIPTS = str.maketrans("⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹", "-+0123456789")
# Scientific notation after the coefficient: "e-7", "E+11", "× 10^-6",
# "x 10⁻⁸", "·10**3". Words rebuilt from a splitter can carry spaces
# ("10 ^ - 6", "7.9E + 11"), so single spaces are allowed inside.
_SCI_EXPONENT = (
    r"(?:[eE]\s?(?P<exp>[-+−]?\s?\d{1,3})(?![\d\w])"
    r"|\s?[×xX·*]\s?10\s?(?:(?:\^|\*\*)\s?(?P<exp_caret>[-+−]?\s?\d{1,3})(?!\d)"
    r"|(?P<exp_sup>[⁻⁺]?\s?[⁰¹²³⁴⁵⁶⁷⁸⁹](?:\s?[⁰¹²³⁴⁵⁶⁷⁸⁹])*)))"
)
NUMBER_RE = re.compile(
    r"(?P<num>" + _NUMBER_PATTERN + r")"
    r"(?:" + _SCI_EXPONENT + r"|\s?(?P<mult>k|thousand|mn|million|millions|bn|billion|"
    r"billions|trillion|[mbt])\b)?",
    re.IGNORECASE,
)


def number_match_exponent(match: re.Match) -> int | None:
    """The written power of ten of a scientific-notation match, else None."""

    groups = match.groupdict()
    raw = groups.get("exp") or groups.get("exp_caret") or groups.get("exp_sup")
    if raw is None:
        return None
    return int(re.sub(r"\s", "", raw.translate(_SUPERSCRIPTS)).replace("−", "-"))
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
    r"(?<![\d:.])\d{1,2}(?::\d{2}){1,2}(?:\.\d+)?(?:\s?[AaPp]\.?[Mm]\b)?(?!\d)"
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
    "%d-%b-%Y",
    "%d-%B-%Y",
    "%d-%b-%y",
    "%Y.%m.%d",
    "%d %b %y",
    "%b-%d-%Y",
    "%A %d %B %Y",
    "%A %B %d %Y",
    "%a %d %b %Y",
    "%a %b %d %Y",
)


def normalize_typed_name(type_name: Any) -> str:
    """Resolve aliases such as ``int`` or ``date-time`` to canonical names."""

    name = str(type_name or "text").strip().lower()
    return TYPE_ALIASES.get(name, name)


@dataclass
class TypedFieldSpec:
    """Typed description of one structuring field."""

    type: str = "text"
    description: str | None = None
    options: tuple[str, ...] = ()
    multi: bool = True
    unit: str | None = None
    decimals: int | None = None
    integer: bool = False
    items: str | None = None
    # Day/month order of ambiguous numeric dates; None detects per document.
    dayfirst: bool | None = None
    # Dates/times may be written partially ("May 2024" -> "2024-05"); False
    # asks for full values.
    partial: bool = True
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
        if self.items is not None:
            self.items = normalize_typed_name(self.items)
        self.partial = bool(self.partial)

    # -- classification ----------------------------------------------------

    @property
    def kind(self) -> str:
        return CLASS_KIND if self.type in CLASS_TYPES else SPAN_KIND

    @property
    def value_type(self) -> str | None:
        """Normalized value type of the field or of its list items."""

        if self.type in VALUE_TYPES:
            return self.type
        if self.type == "list" and self.items in VALUE_TYPES:
            return self.items
        return None

    @property
    def temporal(self) -> bool:
        return self.value_type in TEMPORAL_TYPES

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
            ("$unit", self.unit),
            ("$decimals", self.decimals),
            ("$items", self.items),
            ("$dayfirst", self.dayfirst),
        ):
            if value is not None:
                result[key] = value
        if self.integer:
            result["$integer"] = True
        if not self.partial:
            result["$partial"] = False
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
        fmt = values.get("format")
        if fmt is not None:
            # A text field with a format is a value field of that type.
            if type_name != "text":
                raise ValueError("$format applies to fields of type 'text'")
            if str(fmt).strip().lower() in {"int", "integer"}:
                values.setdefault("integer", True)
            type_name = normalize_typed_name(fmt)
            if type_name not in VALUE_TYPES:
                raise ValueError(f"Unsupported $format {fmt!r}")
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
            unit=values.get("unit"),
            decimals=values.get("decimals"),
            integer=bool(integer),
            items=values.get("items"),
            dayfirst=values.get("dayfirst"),
            partial=bool(values.get("partial", True)),
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
        if self.value_type in {"number", "duration"}:
            if self.unit:
                parts.append(str(self.unit))
            if self.integer:
                parts.append("integer")
            elif self.decimals is not None:
                parts.append(f"{self.decimals} decimals")
        return "( " + " , ".join(parts) + " )"

    # -- parsing -----------------------------------------------------------

    def encode(self, value: Any) -> float | None:
        """Map a JSON value to the real line (``None`` when impossible)."""

        if self.value_type is None:
            return None
        return encode_value(value, self.value_type, unit=self.unit)

    def parse_text(self, text: str, *, dayfirst: bool | None = None) -> float | None:
        """Parse a span copied from the text into the real line."""

        if self.value_type is None:
            return None
        return parse_text_value(
            text,
            self.value_type,
            unit=self.unit,
            dayfirst=self.dayfirst if self.dayfirst is not None else dayfirst,
        )

    def parse_in_context(
        self,
        span_text: str,
        context: str | None = None,
        span_start: int | None = None,
    ) -> float | None:
        """Parse a span using its document for missing information.

        The document decides day/month order, and a datetime span that
        holds only a clock time takes the date written closest to it.
        """

        if self.value_type is None:
            return None
        dayfirst = self.dayfirst if self.dayfirst is not None else detect_dayfirst(context)
        if (
            context
            and span_start is not None
            and self.value_type in {"time", "datetime"}
            and not re.search(r"[AaPp]\.?[Mm]\b", span_text)
        ):
            # A span that stopped before its meridiem: "3:00" + " PM".
            length = len(span_text)
            if context[span_start:span_start + length] != span_text:
                length = len(detokenize_span(span_text))
            span_end = span_start + length
            meridiem = re.match(r"\s?([AaPp]\.?[Mm])\b", context[span_end:span_end + 6])
            if meridiem and re.search(r"\d:\d{2}\s*$|\d\s*$", span_text):
                span_text = f"{span_text} {meridiem.group(1)}"
        value = self.parse_text(span_text, dayfirst=dayfirst)
        if self.value_type == "number" and context and span_start is not None:
            extended = _extend_number_span(span_text, context, span_start)
            if extended is not None:
                value = extended
            if value is not None and value > 0:
                value = _sign_from_context(value, span_text, context, span_start)
        if value is not None or self.value_type != "datetime" or not context:
            return value
        seconds = parse_text_value(span_text, "time")
        if seconds is None:
            return None
        position = span_start if span_start is not None else context.find(span_text)
        best = None
        for match in DATE_RE.finditer(context):
            day = parse_text_value(match.group(0), "date", dayfirst=dayfirst)
            if day is None:
                continue
            distance = abs(match.start() - max(position, 0))
            if best is None or distance < best[0]:
                best = (distance, day)
        if best is None:
            return None
        return best[1] * 86400.0 + seconds


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
        # Keep the wall-clock time the text shows ("11:42 EST" stays 11:42);
        # converting to UTC would make written times and labels disagree.
        result = result.replace(tzinfo=None)
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
    raise ValueError(f"Unsupported value type {value_type!r}")


def _parse_number_match(match: re.Match) -> float | None:
    raw = match.group("num").replace("−", "-")
    raw = re.sub(r"[,   ]", "", raw)
    try:
        number = float(raw)
    except ValueError:
        return None
    exponent = number_match_exponent(match)
    if exponent is not None:
        try:
            return float(Decimal(raw).scaleb(exponent))
        except (InvalidOperation, ValueError):
            return None
    multiplier = match.groupdict().get("mult")
    if multiplier:
        number *= _NUMBER_WORD_MULTIPLIERS.get(multiplier.lower(), 1.0)
    return number


# -- number text beyond plain digits ------------------------------------------------

_FRACTION_GLYPHS = {
    "½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3, "⅕": 0.2, "⅖": 0.4,
    "⅗": 0.6, "⅘": 0.8, "⅙": 1 / 6, "⅚": 5 / 6, "⅛": 0.125, "⅜": 0.375, "⅝": 0.625,
    "⅞": 0.875,
}
_GLYPH_CLASS = "[" + "".join(_FRACTION_GLYPHS) + "]"
_FRACTION_NOUNS = {
    "half": 1 / 2, "halves": 1 / 2, "third": 1 / 3, "thirds": 1 / 3, "quarter": 1 / 4,
    "quarters": 1 / 4, "fourth": 1 / 4, "fourths": 1 / 4, "fifth": 1 / 5, "fifths": 1 / 5,
    "eighth": 1 / 8, "eighths": 1 / 8,
}
_UNIT_WORDS_EN = {
    word: index for index, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen".split()
    )
}
_TENS_WORDS_EN = {
    word: 10 * (index + 2) for index, word in enumerate(
        "twenty thirty forty fifty sixty seventy eighty ninety".split()
    )
}
_SCALE_WORDS_EN = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
# Decimal comma: "1.382.216,45", "6.043,4", "285,8", "0,25". A lone comma
# before exactly three digits ("1,286") stays an English thousands separator.
_EUROPEAN_RE = re.compile(
    r"(?<![\w.,])(?P<sign>[-+−]?)"
    r"(?:(?P<grouped>\d{1,3}(?:[.   ]\d{3})+),(?P<gfrac>\d+)"
    r"|(?P<plain>\d+),(?P<pfrac>\d{1,2}|\d{4,}))(?![\d,]|\.\d)"
    r"|(?<![\w.,])(?P<sign2>[-+−]?)(?P<dotted>\d{1,3}(?:\.\d{3}){2,})(?![\d,.])"
)
_TAIL_FRACTION_RE = re.compile(
    r"\s*(?:(?P<glyph>" + _GLYPH_CLASS + r")"
    r"|\s(?P<num>\d{1,2})/(?P<den>\d{1,2})(?![\d/])"
    r"|\s+and\s+(?P<count>a|an|one|two|three|four|five|seven)\s+"
    r"(?P<noun>halves|half|thirds?|quarters?|fourths?|fifths?|eighths?)\b)",
    re.IGNORECASE,
)


def _european_value(match: re.Match) -> float:
    groups = match.groupdict()
    if groups.get("dotted"):
        number = float(groups["dotted"].replace(".", ""))
        sign = groups.get("sign2")
    else:
        whole = groups.get("grouped") or groups.get("plain")
        whole = re.sub(r"[.   ]", "", whole)
        number = float(f"{whole}.{groups.get('gfrac') or groups.get('pfrac')}")
        sign = groups.get("sign")
    return -number if sign in {"-", "−"} else number


def _parse_number_words(text: str) -> float | None:
    """English number words: "five thousand thirty-three", "one and a half",
    "minus twenty point five", "three quarters", "1.5 million" is not here."""

    tokens = re.findall(r"[a-z]+", text.lower().replace("-", " "))
    total = current = 0.0
    started = negative = False
    since_and = None  # count read after "and" (fractions: "two and three quarters")
    decimals = None  # digits after "point"
    pending_a = False
    for index, token in enumerate(tokens):
        following = tokens[index + 1] if index + 1 < len(tokens) else ""
        if not started and token in {"minus", "negative"}:
            negative = True
            continue
        if decimals is not None:
            if token in _UNIT_WORDS_EN and _UNIT_WORDS_EN[token] < 10:
                decimals += str(_UNIT_WORDS_EN[token])
                continue
            if token in _SCALE_WORDS_EN and decimals:
                total = (total + current + float(f"0.{decimals}")) * _SCALE_WORDS_EN[token]
                current, decimals = 0.0, None
                break
            break
        if token in _UNIT_WORDS_EN or token in _TENS_WORDS_EN:
            value = _UNIT_WORDS_EN.get(token, _TENS_WORDS_EN.get(token))
            current += value
            if since_and is not None:
                since_and += value
            started = True
        elif token in {"a", "an"} and (
            following in {"hundred", "half", "quarter", "third", "fourth", "fifth", "eighth"}
            or following in _SCALE_WORDS_EN
        ):
            # "a" is not added to the count read after "and": the fraction
            # below counts it as one ("fifteen and a half" = 15.5).
            pending_a = True
            started = True
        elif token == "hundred":  # bare "hundred and sixteen" is 116
            current = (current or 1.0) * 100
            started, pending_a = True, False
        elif token in _SCALE_WORDS_EN and (started or pending_a):
            total += (current or 1.0) * _SCALE_WORDS_EN[token]
            current, pending_a = 0.0, False
        elif token in _FRACTION_NOUNS and (started or pending_a):
            fraction = _FRACTION_NOUNS[token]
            if since_and is not None:
                count = since_and or 1.0
                current = current - since_and + count * fraction
            else:
                current = (current or 1.0) * fraction
            started, pending_a = True, False
            break
        elif token == "and" and started:
            since_and = 0.0
        elif token == "point" and started:
            decimals = ""
        elif started:
            break
    if not started:
        return None
    value = total + current
    if decimals:
        value += float(f"0.{decimals}")
    return -value if negative else value


def _parse_number_text(text: str) -> float | None:
    """A number written in ``text``: digits (English or European grouping,
    scientific notation, k/M/bn), fractions, or English words. A value
    wrapped in parentheses is negative, as in financial statements."""

    stripped = text.strip()
    wrapped = re.fullmatch(r"\(\s*([^()]+?)\s*\)", stripped)
    if wrapped:
        inner = _parse_number_text(wrapped.group(1))
        return -abs(inner) if inner is not None else None
    match = NUMBER_RE.search(text)
    european = _EUROPEAN_RE.search(text)
    if european and (match is None or european.start() <= match.start()):
        return _european_value(european)
    glyph = re.search(r"(?<![\d.])" + _GLYPH_CLASS, text)
    if match is None or (glyph and glyph.start() < match.start()):
        if glyph:
            return _FRACTION_GLYPHS[glyph.group(0)]
        return _parse_number_words(text)
    slash = re.match(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?![\d/])", text[match.start():])
    if slash and int(slash.group(2)) > 0 and not match.group("mult"):
        return int(slash.group(1)) / int(slash.group(2))
    words_first = re.search(r"[A-Za-z]+", text[:match.start()])
    if words_first and _parse_number_words(text[:match.start()]) is not None:
        return _parse_number_words(text)
    value = _parse_number_match(match)
    if value is None:
        return None
    if float(value).is_integer() and not match.group("mult") and number_match_exponent(match) is None:
        tail = _TAIL_FRACTION_RE.match(text, match.end())
        if tail:
            if tail.group("glyph"):
                fraction = _FRACTION_GLYPHS[tail.group("glyph")]
            elif tail.group("num"):
                fraction = int(tail.group("num")) / max(int(tail.group("den")), 1)
                fraction = fraction if fraction < 1 else 0.0
            else:
                count = tail.group("count").lower()
                count = 1 if count in {"a", "an", "one"} else _UNIT_WORDS_EN[count]
                fraction = count * _FRACTION_NOUNS[tail.group("noun").lower()]
            value = -(abs(value) + fraction) if value < 0 else value + fraction
    return value



def _sign_from_context(value: float, span_text: str, context: str, span_start: int) -> float:
    """Negative when the copied span sits inside accounting parentheses,
    "(395,327)" (either parenthesis may be inside the span or just outside
    it), or right after a minus sign, "-12.5" / "−12.5"."""

    span_end = span_start + len(span_text)
    if context[span_start:span_end] != span_text:
        return value
    core = span_text.strip()
    opens = core.startswith("(") or re.search(r"\(\s?[$€£]?$", context[max(0, span_start - 3):span_start])
    closes = core.endswith(")") or re.match(r"\s?\)", context[span_end:span_end + 2])
    if opens and closes:
        digits = core.strip("()$€£ ")
        if "%" in core or re.fullmatch(r"(?:19|20)\d{2}", digits):
            return value  # "(5%)", "(2023)": an aside, not a negative
        return -value
    if re.search(r"(?:^|[\s:(=])[-−]$", context[max(0, span_start - 2):span_start]):
        return -value
    return value


def _extend_number_span(span_text: str, context: str, span_start: int) -> float | None:
    """Value of a number the span cut short: "4.6" in "4.6 × 10^-4", "1" in
    "1½" / "1 and a quarter", "8.693" in "8.693,0". Multiplier words are not
    pulled in ("5" stays 5 in "5 million"): the span chose to stop there."""

    span_end = span_start + len(span_text)
    if context[span_start:span_end] != span_text or not re.search(r"\d", span_text):
        return None
    first_digit = span_start + re.search(r"[-−+.]?\d", span_text).start()
    window = context[first_digit:span_end + 32]
    covered = span_end - first_digit
    european = _EUROPEAN_RE.match(window)
    if european and european.end() > covered:
        return _european_value(european)
    match = NUMBER_RE.match(window)
    if match is None:
        return None
    if number_match_exponent(match) is not None and match.end() > covered:
        return _parse_number_match(match)
    if match.group("mult"):
        return None
    tail = _TAIL_FRACTION_RE.match(window, match.end())
    if match.end() > covered and not tail:
        return _parse_number_match(match)  # "1" of "1,286", "3" of "3.75"
    if tail and tail.end() > covered:
        return _parse_number_text(window[:tail.end()])
    return None


def _unit_seconds(unit: str | None) -> float | None:
    if not unit:
        return None
    return _SECONDS.get(str(unit).strip().lower())


_FOREIGN_MONTHS = {
    # German, Dutch, French, Spanish, Italian, Portuguese -> English.
    "januar": "January", "januari": "January", "janvier": "January", "enero": "January",
    "gennaio": "January", "janeiro": "January", "jänner": "January",
    "februar": "February", "februari": "February", "février": "February", "fevrier": "February",
    "febrero": "February", "febbraio": "February", "fevereiro": "February",
    "märz": "March", "maerz": "March", "maart": "March", "mars": "March", "marzo": "March",
    "março": "March", "marco": "March",
    "april": "April", "avril": "April", "abril": "April", "aprile": "April",
    "mai": "May", "mei": "May", "mayo": "May", "maggio": "May", "maio": "May",
    "juni": "June", "juin": "June", "junio": "June", "giugno": "June", "junho": "June",
    "juli": "July", "juillet": "July", "julio": "July", "luglio": "July", "julho": "July",
    "august": "August", "augustus": "August", "août": "August", "aout": "August",
    "agosto": "August",
    "september": "September", "septembre": "September", "septiembre": "September",
    "settembre": "September", "setembro": "September",
    "oktober": "October", "octobre": "October", "octubre": "October", "ottobre": "October",
    "outubro": "October",
    "november": "November", "novembre": "November", "noviembre": "November",
    "novembro": "November",
    "dezember": "December", "december": "December", "décembre": "December",
    "decembre": "December", "diciembre": "December", "dicembre": "December",
    "dezembro": "December",
}
_FOREIGN_MONTH_RE = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, _FOREIGN_MONTHS), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")


def _parse_date_text(text: str) -> date | None:
    parsed = _parse_date_exact(text)
    if parsed is not None:
        return parsed
    # A date inside a longer span: "2024-10-28T19:00", "Mon, 3 March 2024".
    match = _ISO_DATE_RE.search(text)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass
    normalized = _FOREIGN_MONTH_RE.sub(lambda m: _FOREIGN_MONTHS[m.group(1).lower()], text)
    for candidate in DATE_RE.finditer(normalized):
        if candidate.group(0).strip() != text.strip():
            parsed = _parse_date_exact(candidate.group(0))
            if parsed is not None:
                return parsed
    if normalized != text:
        return _parse_date_exact(normalized)
    return None


# Day-first preference for ambiguous numeric dates ("05/11/2024"); set for
# one parse by ``parse_text_value(..., dayfirst=...)``.
_DAYFIRST: contextvars.ContextVar[bool | None] = contextvars.ContextVar(
    "gliformer_typed_dayfirst", default=None
)
_MONTH_FIRST_FORMATS = ("%m/%d/%Y", "%m-%d-%Y")
_DAY_FIRST_FORMATS = ("%d/%m/%Y", "%d-%m-%Y")
_NUMERIC_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})(?!\d)")


def detect_dayfirst(text: str | None) -> bool | None:
    """Day/month order of a document's numeric dates, when it is decidable.

    ``True`` if some date can only be day-first (``25/03/2024``), ``False``
    if some date can only be month-first (``03/25/2024``), else ``None``.
    """

    if not text:
        return None
    day_first = month_first = False
    for match in _NUMERIC_DATE_RE.finditer(text):
        first, second = int(match.group(1)), int(match.group(2))
        if first > 12 >= second:
            day_first = True
        elif second > 12 >= first:
            month_first = True
    if day_first != month_first:
        return day_first
    return None


def _date_formats() -> tuple[str, ...]:
    if not _DAYFIRST.get():
        return _DATE_FORMATS
    ordered = [fmt for fmt in _DATE_FORMATS if fmt not in _DAY_FIRST_FORMATS]
    position = min(ordered.index(fmt) for fmt in _MONTH_FIRST_FORMATS)
    return tuple(ordered[:position] + list(_DAY_FIRST_FORMATS) + ordered[position:])


def _parse_date_exact(text: str) -> date | None:
    text = _FOREIGN_MONTH_RE.sub(lambda m: _FOREIGN_MONTHS[m.group(1).lower()], text)
    cleaned = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text.strip())
    cleaned = re.sub(r"\bof\s+", "", cleaned)
    cleaned = re.sub(r"\bSept\b", "Sep", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace(".", " ") if not re.fullmatch(
        r"\d{1,2}\.\d{1,2}\.\d{4}", cleaned
    ) else cleaned
    cleaned = re.sub(r"\s+", " ", cleaned).replace(" ,", ",").strip(" ,")
    cleaned = re.sub(r"^(\w+),\s", r"\1 ", cleaned)  # "Monday, 3 March 2024"
    for fmt in _date_formats():
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
        r"\s*(\d{1,2})(?::(\d{2}))?(?::(\d{2}))?(?:\.\d+)?\s*([AaPp])?\.?(?:[Mm]\.?)?\s*",
        text,
    )
    if not match or (match.group(2) is None and match.group(4) is None):
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    meridiem = (match.group(4) or "").lower()
    if meridiem and hours > 12:
        meridiem = ""  # "16:10 PM": the 24-hour clock already says it
    if meridiem:
        if hours < 1:
            return None
        hours = hours % 12 + (12 if meridiem == "p" else 0)
    if hours > 23 or minutes > 59 or seconds > 59:
        return None
    return hours * 3600.0 + minutes * 60.0 + seconds


def detokenize_span(text: str) -> str:
    """Undo word-splitting inside numbers, dates and times.

    Decoded spans are rebuilt by joining word tokens with spaces, so
    "11:20", "2024-10-28" and "14-OCT-2024" arrive as "11 : 20",
    "2024 - 10 - 28" and "14 - OCT - 2024".
    """

    text = re.sub(r"(?<=\d)\s*([:/])\s*(?=\d)", r"\1", text)
    text = re.sub(r"(?<=\w)\s*-\s*(?=\w)", "-", text)
    text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", text)
    text = re.sub(r"(?<=\d)\s*T\s*(?=\d)", "T", text)
    return text


def parse_text_value(
    text: str,
    value_type: str,
    *,
    unit: str | None = None,
    dayfirst: bool | None = None,
) -> float | None:
    """Parse free text (a tagged span) into a float of ``value_type``.

    ``dayfirst`` resolves numeric dates such as ``05/11/2024``; ``None``
    keeps month-first order.
    """

    token = _DAYFIRST.set(dayfirst)
    try:
        return _parse_text_value(text, value_type, unit=unit)
    finally:
        _DAYFIRST.reset(token)


def _parse_text_value(text: str, value_type: str, *, unit: str | None = None) -> float | None:

    if not isinstance(text, str) or not text.strip():
        return None
    # Span text rebuilt from words can carry spaces around punctuation
    # ("March 3 , 2024", "$ 1 , 250"); undo that before parsing.
    text = detokenize_span(text)
    if value_type == "number":
        # "8.693 , 0" / "285 , 8": a comma split off inside one number.
        text = re.sub(r"(?<=\d)\s+,\s+(?=\d)", ",", text)
    text = re.sub(r"\s+([,.;:)%])", r"\1", text.strip())
    text = re.sub(r"([($])\s+", r"\1", text)
    text = re.sub(r"(\d),\s+(\d{3})\b", r"\1,\2", text)
    text = re.sub(r"(\d)\.\s+(\d)", r"\1.\2", text)
    text = re.sub(r"\s+", " ", text)
    if value_type == "number":
        return _parse_number_text(text)
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
        try:
            iso = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            iso = None
        if iso is not None:
            return float((iso.replace(tzinfo=None) - _EPOCH_DATETIME).total_seconds())
        # A date and a clock time written apart in one span.
        time_match = TIME_RE.search(text)
        seconds = _parse_time_text(time_match.group(0)) if time_match else None
        if seconds is not None:
            remainder = text[: time_match.start()] + " " + text[time_match.end():]
            day = _parse_date_text(remainder.strip(" ,;"))
            if day is not None:
                base = datetime.combine(day, clock_time()) - _EPOCH_DATETIME
                return base.total_seconds() + seconds
        parsed = _coerce_datetime(text)
        if parsed is not None:
            return float((parsed - _EPOCH_DATETIME).total_seconds())
        return None
    if value_type == "time":
        seconds = _parse_time_text(text)
        if seconds is not None:
            return seconds
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            moment = None
        if moment is not None and ("T" in text or ":" in text):
            return moment.hour * 3600.0 + moment.minute * 60.0 + moment.second
        match = TIME_RE.search(text)
        return _parse_time_text(match.group(0)) if match else None
    raise ValueError(f"Unsupported value type {value_type!r}")


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


def find_anchor(
    text: str,
    spec: TypedFieldSpec,
    target_x: float,
    *,
    snippets: Iterable[str] = (),
) -> dict[str, Any] | None:
    """Find the span a value field's gold value was read from.

    Evidence snippets are searched first, then the whole text.  Only a span
    whose parsed value equals ``target_x`` qualifies.  Returns ``{"text",
    "start", "end", "x"}`` with character offsets and the parsed value ``x``.
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

    detected = spec.dayfirst if spec.dayfirst is not None else detect_dayfirst(text)
    orders = (detected,) if detected is not None else (None, True)

    def candidates(regions):
        for region_start, region_end in regions:
            region = text[region_start:region_end]
            for start, end in candidate_spans(region, value_type):
                span_text = region[start:end]
                parsed = None
                for dayfirst in orders:
                    parsed = spec.parse_text(span_text, dayfirst=dayfirst)
                    if matches(parsed):
                        break
                if parsed is None:
                    parsed = spec.parse_in_context(span_text, text, region_start + start)
                if parsed is not None:
                    yield span_text, region_start + start, region_start + end, parsed

    whole = [(0, len(text))]
    for regions in (evidence_regions, whole):
        for span_text, start, end, parsed in candidates(regions):
            if matches(parsed):
                return {"text": span_text, "start": start, "end": end, "x": parsed}
    return None


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
    "CLASS_TYPES",
    "FIELD_KIND_IDS",
    "SPAN_KIND",
    "SPAN_TYPES",
    "TEMPORAL_TYPES",
    "TypedFieldSpec",
    "VALUE_TYPES",
    "candidate_spans",
    "encode_value",
    "find_anchor",
    "is_typed_descriptor",
    "lookup_group_types",
    "normalize_typed_name",
    "parse_text_value",
    "spec_from_json_schema",
    "specs_from_dicts",
    "specs_from_json_schema",
    "specs_to_dicts",
]
