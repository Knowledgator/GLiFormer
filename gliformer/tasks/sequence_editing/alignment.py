"""Edit plans: per-unit operations that turn a source text into a target.

A text is a list of units (words or subwords, ``(text, start, end)``). Each
unit gets one operation; slot 0 is a virtual BOS unit before the first one, so
text can be inserted at the start:

* ``KEEP``: unit unchanged;
* ``DELETE``: unit removed;
* ``REPLACE``: unit replaced by a content fragment;
* ``APPEND``: unit kept, a content fragment inserted after it.

A content fragment is a target substring plus ``space``: whether whitespace
comes before it. Units are compared together with their own space flag
("yesterday ." vs "yesterday." differ in the "." unit), so spacing changes
become edits. The first unit of a text always counts as spaced; spaces at the
start of the output are dropped.

``apply_edits`` rebuilds the text from a plan. Whitespace that is kept copies
the source gap, new whitespace is one space, so ``apply_edits(plan(s, t))``
equals ``t`` up to whitespace runs (see :func:`normalize_space`).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

KEEP, DELETE, REPLACE, APPEND = 0, 1, 2, 3
OP_NAMES = ("keep", "delete", "replace", "append")
NUM_OPS = 4
CONTENT_OPS = (REPLACE, APPEND)

Unit = tuple  # (text, start, end)
_SPACE = re.compile(r"\s+")


def normalize_space(text: str) -> str:
    return _SPACE.sub(" ", text).strip()


@dataclass
class Content:
    """A fragment the plan writes: its text and whether a space precedes it."""

    text: str
    space: bool = True


@dataclass
class EditPlan:
    """``ops[s]`` for slot ``s`` (0 = BOS, unit ``i`` = slot ``i + 1``)."""

    ops: list[int]
    contents: dict[int, Content] = field(default_factory=dict)

    @property
    def num_edits(self) -> int:
        return sum(1 for op in self.ops if op != KEEP)

    def is_identity(self) -> bool:
        return self.num_edits == 0


def unit_keys(text: str, units: Sequence[Unit]) -> list[tuple[bool, str]]:
    """``(space_before, unit_text)`` per unit; the first unit counts as spaced."""

    keys = []
    previous_end = None
    for unit in units:
        start, end = int(unit[1]), int(unit[2])
        spaced = previous_end is None or start > previous_end
        keys.append((spaced, text[start:end]))
        previous_end = end
    return keys


def plan_edits(
    source: str,
    source_units: Sequence[Unit],
    target: str,
    target_units: Sequence[Unit],
) -> EditPlan:
    """Align source and target units and express the target as an edit plan."""

    source_keys = unit_keys(source, source_units)
    target_keys = unit_keys(target, target_units)
    ops = [KEEP] * (len(source_units) + 1)
    ranges: dict[int, list[int]] = {}  # slot -> [first target unit, end target unit)

    matcher = difflib.SequenceMatcher(None, source_keys, target_keys, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            for i in range(i1, i2):
                ops[i + 1] = DELETE
            continue
        if tag == "replace":
            source_len, target_len = i2 - i1, j2 - j1
            pairs = min(source_len, target_len)
            for offset in range(pairs):
                slot = i1 + offset + 1
                end = j1 + offset + 1 if offset < pairs - 1 else j2
                ops[slot] = REPLACE
                ranges[slot] = [j1 + offset, end]
            for i in range(i1 + pairs, i2):
                ops[i + 1] = DELETE
            continue
        # insert: after unit i1 - 1 (slot i1; BOS when i1 == 0)
        slot = i1
        if ops[slot] == REPLACE and ranges[slot][1] == j1:
            ranges[slot][1] = j2
        elif ops[slot] == DELETE:
            ops[slot] = REPLACE
            ranges[slot] = [j1, j2]
        else:
            ops[slot] = APPEND
            ranges[slot] = [j1, j2]

    contents = {}
    for slot, (first, end) in ranges.items():
        start_char = int(target_units[first][1])
        end_char = int(target_units[end - 1][2])
        contents[slot] = Content(target[start_char:end_char], target_keys[first][0])
    return EditPlan(ops, contents)


def apply_edits(
    source: str,
    source_units: Sequence[Unit],
    ops: Sequence[int],
    contents: dict[int, Content],
) -> str:
    """Rebuild a text from per-slot ops and content fragments.

    Units past ``len(ops) - 1`` (e.g. cut off by truncation) are kept.
    """

    pieces: list[str] = []

    def emit(fragment: str, spaced: bool, gap: str = "") -> None:
        if not fragment:
            return
        if pieces and spaced:
            pieces.append(gap if gap and gap.isspace() else " ")
        pieces.append(fragment)

    def content_of(slot):
        content = contents.get(slot)
        return content if content is not None else Content("", True)

    if ops and ops[0] == APPEND:
        content = content_of(0)
        emit(content.text, content.space)
    previous_end = None
    for index, unit in enumerate(source_units):
        start, end = int(unit[1]), int(unit[2])
        gap = source[previous_end:start] if previous_end is not None else ""
        spaced = previous_end is None or start > previous_end
        previous_end = end
        slot = index + 1
        op = ops[slot] if slot < len(ops) else KEEP
        if op == KEEP or op == APPEND:
            emit(source[start:end], spaced, gap)
            if op == APPEND:
                content = content_of(slot)
                emit(content.text, content.space)
        elif op == REPLACE:
            content = content_of(slot)
            emit(content.text, content.space, gap)
    if not source_units:
        return source
    lead = source[: int(source_units[0][1])]
    tail = source[int(source_units[-1][2]):]
    return lead + "".join(pieces) + tail


def char_edits(
    source: str,
    source_units: Sequence[Unit],
    ops: Sequence[int],
    contents: dict[int, Content],
    scores: Iterable[float] | None = None,
) -> list[dict]:
    """Character-level description of a plan: one dict per non-KEEP slot."""

    scores = list(scores) if scores is not None else None
    edits = []
    for slot, op in enumerate(ops):
        if op == KEEP:
            continue
        if slot == 0:
            start = end = int(source_units[0][1]) if source_units else 0
        else:
            if slot - 1 >= len(source_units):
                continue
            unit = source_units[slot - 1]
            start, end = int(unit[1]), int(unit[2])
            if op == APPEND:
                start = end
        content = contents.get(slot)
        edit = {
            "op": OP_NAMES[op],
            "start": start,
            "end": end,
            "source": source[start:end],
            "replacement": content.text if content is not None else "",
        }
        if content is not None:
            edit["space"] = bool(content.space)
        if scores is not None and slot < len(scores):
            edit["score"] = float(scores[slot])
        edits.append(edit)
    return edits


__all__ = [
    "APPEND",
    "CONTENT_OPS",
    "Content",
    "DELETE",
    "EditPlan",
    "KEEP",
    "NUM_OPS",
    "OP_NAMES",
    "REPLACE",
    "apply_edits",
    "char_edits",
    "normalize_space",
    "plan_edits",
    "unit_keys",
]
