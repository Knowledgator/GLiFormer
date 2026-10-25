"""Rule-based parser that finds and reads values written in a word sequence.

It is used only to preprocess training inputs: when enabled
(``normalization_config.parser.enabled``), every number, date, time and
datetime it finds in a training text becomes an extra normalization span
with a silver label.  It never produces model outputs.  Mentions with more
than one plausible reading ("1,286", "5 m", "05/11/2024") are skipped, so
silver labels are spans the parser converts with certainty.

Spans are on the word grid (``start``/``end`` are inclusive word indices),
the same grid the model's span heads use.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Sequence

from ...processing.typed_fields import (
    _MONTHS,
    _SCALE_WORDS_EN,
    _TENS_WORDS_EN,
    _UNIT_WORDS_EN,
    DATE_RE,
    DATETIME_RE,
    NUMBER_RE,
    TIME_RE,
    _parse_date_text,
    _parse_number_match,
    _parse_number_words,
    _parse_time_text,
)
from .values import (
    FAMILY_DATE,
    FAMILY_DATETIME,
    FAMILY_TIME,
    TemporalParts,
    is_iso_value,
    temporal_parts,
    valid_parts,
)


@dataclass
class ParsedSpan:
    """A value found in the text: a number, calendar components, or both.

    A four-digit year written alone ("in 2021") carries both: the number
    2021 and a date with only the year written.
    """

    start: int  # first word index
    end: int  # last word index (inclusive)
    number: float | None = None
    family: int | None = None  # FAMILY_DATE / FAMILY_TIME / FAMILY_DATETIME
    parts: TemporalParts | None = None


# -- word grid ----------------------------------------------------------------


def compact_text(words: Sequence[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join words, gluing pieces of numbers ("1", ",", "250") together.

    Returns the text and each word's ``(start, end)`` character range in it.
    """

    pieces: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    previous = ""
    for index, word in enumerate(words):
        following = words[index + 1] if index + 1 < len(words) else ""
        glue = index > 0 and (
            (re.fullmatch(r"[.:/\-]", word) and re.search(r"\d$", previous))
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


def word_range(spans: list[tuple[int, int]], start: int, end: int) -> tuple[int, int] | None:
    """Inclusive word range covering characters ``[start, end)``."""

    first = last = None
    for index, (word_start, word_end) in enumerate(spans):
        if word_end <= start or word_start >= end:
            continue
        if first is None:
            first = index
        last = index
    return (first, last) if first is not None else None


# -- readers -------------------------------------------------------------------

# Silver labels are kept only for spans with one reading.
# "1,286" / "1.286": English grouping or a European decimal.
_ONE_GROUP_RE = re.compile(r"[-+−]?\d{1,3}[.,]\d{3}")
# "5 m", "3 b", "2 t": a multiplier or a unit (metres, bytes, tonnes).
_UNIT_LIKE_MULTIPLIERS = {"m", "b", "t"}
_NUMERIC_DATE_RE = re.compile(r"(\d{1,2})[/.-](\d{1,2})[/.-]\d{2,4}")


def ambiguous_number(match) -> bool:
    """Whether a number mention has more than one plausible value."""

    if _ONE_GROUP_RE.fullmatch(match.group("num").strip()):
        return True
    multiplier = match.groupdict().get("mult")
    return bool(multiplier) and multiplier in _UNIT_LIKE_MULTIPLIERS


def ambiguous_date(text: str) -> bool:
    """A numeric date whose day and month can be swapped ("05/11/2024")."""

    match = _NUMERIC_DATE_RE.search(text)
    if not match:
        return False
    first, second = int(match.group(1)), int(match.group(2))
    return first != second and first <= 12 and second <= 12


_MONTH_YEAR_RE = re.compile(r"^\s*" + _MONTHS + r"\s+\d{4}\s*$", re.IGNORECASE)
_YEAR_RE = re.compile(r"^(1[5-9]\d{2}|20\d{2}|21\d{2})$")
_NUMBER_CORE_WORDS = (
    set(_UNIT_WORDS_EN) | set(_TENS_WORDS_EN) | set(_SCALE_WORDS_EN) | {"hundred"}
)
_NUMBER_SCALE_STARTS = {"hundred", *_SCALE_WORDS_EN}


def read_date(text: str) -> TemporalParts | None:
    """Written date components of a date mention ("May 2024" has no day)."""

    parsed: date | None = _parse_date_text(text)
    if parsed is None:
        return None
    if _MONTH_YEAR_RE.match(text):
        return TemporalParts(parsed.year, parsed.month, None)
    return TemporalParts(parsed.year, parsed.month, parsed.day)


def read_time(text: str) -> TemporalParts | None:
    """Written clock components ("5 pm" is an hour, "5:30" adds minutes)."""

    seconds = _parse_time_text(text)
    if seconds is None:
        return None
    colons = text.count(":")
    seconds = int(seconds)
    hour, minute, second = seconds // 3600, seconds % 3600 // 60, seconds % 60
    return TemporalParts(
        None, None, None, hour,
        minute if colons >= 1 else None,
        second if colons >= 2 else None,
    )


def read_datetime(text: str) -> TemporalParts | None:
    """Components of an ISO datetime mention ("2024-05-18T14:30")."""

    match = re.match(
        r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2}))?", text.strip(),
    )
    if not match:
        return None
    year, month, day, hour, minute = (int(match.group(index)) for index in range(1, 6))
    second = int(match.group(6)) if match.group(6) is not None else None
    parts = TemporalParts(year, month, day, hour, minute, second)
    return parts if valid_parts(parts, FAMILY_DATETIME) else None


def _is_number_word(word: str) -> bool:
    parts = re.findall(r"[a-z]+", word.lower())
    return bool(parts) and all(part in _NUMBER_CORE_WORDS for part in parts) and not re.search(
        r"[^a-z\-]", word.lower()
    )


def word_number_spans(words: Sequence[str], covered: set[int]) -> list[ParsedSpan]:
    """Runs of English number words: "one hundred and twenty-five", "a hundred"."""

    found = []
    index, count = 0, len(words)
    while index < count:
        if index in covered or not _is_number_word(words[index]):
            index += 1
            continue
        last = probe = index
        while probe + 1 < count and probe + 1 not in covered:
            word = words[probe + 1]
            if _is_number_word(word):
                probe += 1
                last = probe
            elif word == "-" or (
                word.lower() == "and" and words[last].lower() in _NUMBER_SCALE_STARTS
            ):
                # "and" joins only after hundred/thousand/...: "one hundred and
                # five", not "two and three".
                probe += 1
            else:
                break
        start = index
        if (
            index > 0
            and index - 1 not in covered
            and words[index - 1].lower() in {"a", "an"}
            and words[index].lower() in _NUMBER_SCALE_STARTS
        ):
            start = index - 1
        value = _parse_number_words(" ".join(words[start:last + 1]))
        if value is not None and math.isfinite(value):
            found.append(ParsedSpan(start, last, number=float(value)))
        index = last + 1
    return found


_READERS = {FAMILY_DATE: read_date, FAMILY_TIME: read_time, FAMILY_DATETIME: read_datetime}


def gold_parts(value, family: int, *, partial: bool = True) -> TemporalParts | None:
    """Calendar components of a gold (or written) temporal value.

    ISO values ("2024-05", "2024-05-18T14:30", "14:30") keep their written
    precision. Text written as in a document ("May 2024", "5:30 pm") is read
    with the readers above, so unwritten components stay missing instead of
    defaulting to 1 or 0.
    """

    if isinstance(value, str) and not is_iso_value(value, family):
        reader = _READERS.get(family)
        read = reader(value) if reader is not None else None
        if read is not None and valid_parts(read, family, partial=partial):
            return read
    return temporal_parts(value, family, partial=partial)


# -- parser --------------------------------------------------------------------


class NormalizationParser:
    """Finds values in a word sequence and reads them.

    ``numbers``/``dates``/``times`` switch the kinds of mentions on and off;
    ``number_words`` also reads English number words; at most
    ``max_mentions`` spans are returned, in reading order.
    """

    def __init__(
        self,
        *,
        numbers: bool = True,
        dates: bool = True,
        times: bool = True,
        number_words: bool = False,
        max_mentions: int = 64,
    ):
        self.numbers = bool(numbers)
        self.dates = bool(dates)
        self.times = bool(times)
        self.number_words = bool(number_words)
        self.max_mentions = int(max_mentions)

    @classmethod
    def from_config(cls, parser_cfg) -> "NormalizationParser":
        return cls(
            numbers=parser_cfg.numbers,
            dates=parser_cfg.dates,
            times=parser_cfg.times,
            number_words=parser_cfg.number_words,
            max_mentions=parser_cfg.max_mentions,
        )

    def find(self, words: Sequence[str]) -> list[ParsedSpan]:
        if not words:
            return []
        text, spans = compact_text(list(words))
        found: dict[tuple[int, int], ParsedSpan] = {}
        covered: set[int] = set()

        def claim(match, family, parts):
            grid = word_range(spans, match.start(), match.end())
            if grid is None or parts is None or any(i in covered for i in range(grid[0], grid[1] + 1)):
                return
            found[grid] = ParsedSpan(*grid, family=family, parts=parts)
            covered.update(range(grid[0], grid[1] + 1))

        if self.dates and self.times:
            for match in DATETIME_RE.finditer(text):
                claim(match, FAMILY_DATETIME, read_datetime(match.group(0)))
        if self.dates:
            for match in DATE_RE.finditer(text):
                if ambiguous_date(match.group(0)):
                    # Covered, so its digits are not read as numbers either.
                    grid = word_range(spans, match.start(), match.end())
                    if grid is not None:
                        covered.update(range(grid[0], grid[1] + 1))
                    continue
                claim(match, FAMILY_DATE, read_date(match.group(0)))
        if self.times:
            for match in TIME_RE.finditer(text):
                claim(match, FAMILY_TIME, read_time(match.group(0)))
        if self.numbers:
            for match in NUMBER_RE.finditer(text):
                grid = word_range(spans, match.start(), match.end())
                if grid is None or any(i in covered for i in range(grid[0], grid[1] + 1)):
                    continue
                if ambiguous_number(match):
                    continue
                value = _parse_number_match(match)
                if value is None or not math.isfinite(value):
                    continue
                parsed = ParsedSpan(*grid, number=float(value))
                written = match.group(0).strip()
                if self.dates and _YEAR_RE.fullmatch(written):
                    # A year on its own is both a number and a year-only date.
                    parsed.family = FAMILY_DATE
                    parsed.parts = TemporalParts(int(written))
                found[grid] = parsed
                covered.update(range(grid[0], grid[1] + 1))
            if self.number_words:
                for parsed in word_number_spans(words, covered):
                    found[(parsed.start, parsed.end)] = parsed
                    covered.update(range(parsed.start, parsed.end + 1))
        ordered = sorted(found.values(), key=lambda item: (item.start, item.end))
        return ordered[: self.max_mentions]


__all__ = [
    "NormalizationParser",
    "ParsedSpan",
    "compact_text",
    "gold_parts",
    "read_date",
    "read_datetime",
    "read_time",
    "word_range",
]
