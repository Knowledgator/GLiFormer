"""Token-level text units (``config.text_unit == "token"``).

In word mode a words splitter cuts the text into words and the model pools
each word's subwords into one position.  In token mode every backbone subword
is a position of its own.  ``TokenSplitter`` yields those subwords with
their character offsets, in the same ``(piece, start, end)`` shape as a words
splitter, so span resolution and the decoders' character maps work unchanged.

A unit's piece is the text it covers (``text[start:end]``), not the
tokenizer's vocabulary string, so units can be located in the text and read
as text.  Offsets are trimmed of whitespace; ids that cover only whitespace
("▁", "Ġ", "ĊĊ") belong to no unit: the encoder still reads them but no span
can start or end on them.  Ids whose offsets overlap the previous unit (the
byte pieces of one character) join that unit.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

TOKEN_UNIT_MARKER = "_gliformer_text_unit"

Unit = tuple[str, int, int]


def text_unit(config) -> str:
    return str(getattr(config, "text_unit", "word") or "word").lower()


def is_token_level(config) -> bool:
    return text_unit(config) == "token"


def snaps_to_words(config) -> bool:
    """Whether decoded token spans are widened to whole words."""

    return is_token_level(config) and str(
        getattr(config, "span_boundary", "free") or "free"
    ).lower() == "word"


@dataclass(frozen=True)
class TokenUnitEncoding:
    """One text encoded by the backbone tokenizer, grouped into units."""

    text: str
    input_ids: tuple[int, ...]
    # Unit index of every id; -1 for ids that belong to no unit.
    id_units: tuple[int, ...]
    units: tuple[Unit, ...]
    # Index of the first id of every unit.
    unit_first_id: tuple[int, ...]

    def id_count(self, num_units: int) -> int:
        """Number of leading ids that encode the first ``num_units`` units."""

        if num_units >= len(self.units):
            return len(self.input_ids)
        if num_units <= 0:
            return 0
        return self.unit_first_id[num_units]

    @property
    def pieces(self) -> list[str]:
        return [piece for piece, _, _ in self.units]


def build_token_units(text: str, input_ids, offsets) -> TokenUnitEncoding:
    """Group tokenizer ids into units using their character offsets."""

    units: list[list[int]] = []  # [start, end]
    unit_first_id: list[int] = []
    id_units: list[int] = []
    for id_index, (start, end) in enumerate(offsets):
        start, end = int(start), int(end)
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end <= start:
            id_units.append(-1)
            continue
        if units and start < units[-1][1]:
            # Byte-level pieces of one character share its offsets.
            units[-1][1] = max(units[-1][1], end)
            id_units.append(len(units) - 1)
            continue
        units.append([start, end])
        unit_first_id.append(id_index)
        id_units.append(len(units) - 1)
    return TokenUnitEncoding(
        text=text,
        input_ids=tuple(int(i) for i in input_ids),
        id_units=tuple(id_units),
        units=tuple((text[s:e], s, e) for s, e in units),
        unit_first_id=tuple(unit_first_id),
    )


class TokenSplitter:
    """Words-splitter stand-in that yields backbone subword units.

    Encodings are cached by text: the same text is split while resolving
    labels and encoded again when the batch is tokenized.
    """

    def __init__(self, tokenizer, cache_size: int = 4096):
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError(
                "text_unit='token' needs a fast tokenizer: units come from its "
                "offset mapping."
            )
        self.tokenizer = tokenizer
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[str, TokenUnitEncoding] = OrderedDict()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        return state

    def encode(self, text: str) -> TokenUnitEncoding:
        text = "" if text is None else str(text)
        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            return cached
        if text:
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=False,
                verbose=False,
            )
            result = build_token_units(text, encoded["input_ids"], encoded["offset_mapping"])
        else:
            result = build_token_units(text, (), ())
        self._cache[text] = result
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result

    def __call__(self, text: str) -> Iterator[Unit]:
        yield from self.encode(text).units


def unit_word_ids(units: Sequence[Unit], words: Sequence[Unit]) -> list[int | None]:
    """Index of the word that holds each unit's first character."""

    starts = [start for _, start, _ in words]
    result: list[int | None] = []
    for _, start, _ in units:
        index = bisect.bisect_right(starts, start) - 1
        if 0 <= index < len(words) and words[index][1] <= start < max(words[index][2], words[index][1] + 1):
            result.append(index)
        else:
            result.append(None)
    return result


def word_snapped_maps(
    units: Sequence[Unit],
    words: Sequence[Unit],
) -> tuple[list[int], list[int]]:
    """Unit start/end character maps widened to the words around each unit."""

    starts = [start for _, start, _ in words]
    ends = [end for _, _, end in words]
    start_map: list[int] = []
    end_map: list[int] = []
    for _, start, end in units:
        index = bisect.bisect_right(starts, start) - 1
        if 0 <= index < len(words) and start < ends[index]:
            start_map.append(min(start, starts[index]))
        else:
            start_map.append(start)
        index = bisect.bisect_left(ends, end)
        if 0 <= index < len(words) and starts[index] < end:
            end_map.append(max(end, ends[index]))
        else:
            end_map.append(end)
    return start_map, end_map


def aligned_word_grid(words: Sequence[str], text: str) -> list[Unit] | None:
    """Character offsets of pre-split ``words`` in ``text``, or None if they do not line up."""

    grid: list[Unit] = []
    cursor = 0
    for word in words:
        word = str(word)
        if not word:
            grid.append((word, cursor, cursor))
            continue
        index = text.find(word, cursor)
        if index < 0:
            return None
        cursor = index + len(word)
        grid.append((word, index, cursor))
    return grid


def joined_word_grid(words: Sequence[str]) -> tuple[str, list[Unit]]:
    """Text made by joining ``words`` with spaces, and each word's offsets."""

    grid: list[Unit] = []
    cursor = 0
    for index, word in enumerate(words):
        word = str(word)
        if index:
            cursor += 1
        grid.append((word, cursor, cursor + len(word)))
        cursor += len(word)
    return " ".join(str(word) for word in words), grid


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class WordIndexConverter:
    """Rewrites word-index annotations as character offsets on a word grid."""

    def __init__(self, grid: Sequence[Unit]):
        self.grid = grid

    def char_range(self, start, end) -> tuple[int, int]:
        if not (_is_int(start) and _is_int(end)):
            return -1, -1
        if start < 0 or end < start or end >= len(self.grid):
            return -1, -1
        return self.grid[start][1], self.grid[end][2]

    def ner_entity(self, value):
        """Convert one NER entity whose integers are word indices."""

        if isinstance(value, dict):
            if "start" in value and "end" in value:
                converted = dict(value)
                converted["start"], converted["end"] = self.char_range(value["start"], value["end"])
                return converted
            return value
        if isinstance(value, (list, tuple)):
            if len(value) >= 4 and isinstance(value[0], str) and _is_int(value[1]) and _is_int(value[2]):
                start, end = self.char_range(value[1], value[2])
                return [value[0], start, end, *value[3:]]
            if len(value) >= 3 and _is_int(value[0]) and _is_int(value[1]):
                start, end = self.char_range(value[0], value[1])
                return [start, end, *value[2:]]
        return value

    def span_dicts(self, payload):
        """Convert every ``{"start": int, "end": int}`` dict in a nested payload."""

        if isinstance(payload, dict):
            converted = {key: self.span_dicts(value) for key, value in payload.items()}
            if _is_int(payload.get("start")) and _is_int(payload.get("end")):
                converted["start"], converted["end"] = self.char_range(payload["start"], payload["end"])
            return converted
        if isinstance(payload, list):
            return [self.span_dicts(value) for value in payload]
        return payload


class UnitPieces(list):
    """Unit pieces of one row that can read a unit span back from the text."""

    def __init__(self, pieces: Sequence[str], text: str):
        super().__init__(pieces)
        self.text = text
        self.grid = aligned_word_grid(list(pieces), text or "")

    def span_text(self, start: int, end: int) -> str | None:
        grid = self.grid
        if grid is None or not (0 <= start <= end < len(grid)):
            return None
        return self.text[grid[start][1]:grid[end][2]]


def unit_pieces_batch(tokens, texts):
    """Wrap a batch's unit pieces so decoders slice span text from the source."""

    if tokens is None or texts is None:
        return tokens
    return [
        UnitPieces(row, text) if row is not None else row
        for row, text in zip(tokens, texts, strict=True)
    ]
