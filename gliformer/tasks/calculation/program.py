"""Calculation programs: format, validation and execution.

A program is a list of steps. Each step has an operation and two operand
ports; every port is a *set* of members::

    {"op": "filter_gt", "args": [[{"ref": 0}], [{"span": "3.50", "start": 812, "end": 816}]]}

Members are text spans (``{"span", "start", "end"}`` with character offsets,
``end`` exclusive, optional ``"value"``), earlier steps (``{"ref": k}`` with
``k`` smaller than the step index), or constants from :data:`CONSTANTS`
(``{"const": 100}``).  The last step is the result.

A port evaluates to a list: spans in text order, then referenced steps in
step order (a list result is spliced in), then constants in vocabulary order.
This canonical order is what the model can reproduce from unordered set
predictions, so programs must be written to be correct under it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence

CONSTANTS: tuple[float, ...] = (
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0,
    100.0, 1000.0, 1_000_000.0, -1.0,
)

# Operation name -> (number of ports used, needs numeric values).
OPS: dict[str, tuple[int, bool]] = {
    "copy": (1, True),
    "collect": (1, False),
    "sum": (1, True),
    "mean": (1, True),
    "median": (1, True),
    "min": (1, True),
    "max": (1, True),
    "count": (1, False),
    "product": (1, True),
    "sort_asc": (1, True),
    "sort_desc": (1, True),
    "diff_adjacent": (1, True),
    "filter_gt": (2, True),
    "filter_ge": (2, True),
    "filter_lt": (2, True),
    "filter_le": (2, True),
    "add": (2, True),
    "subtract": (2, True),
    "multiply": (2, True),
    "divide": (2, True),
}
OP_NAMES: tuple[str, ...] = tuple(OPS)
OP_TO_ID = {name: index for index, name in enumerate(OP_NAMES)}
NUM_PORTS = 2


class ProgramError(ValueError):
    """The program is malformed or cannot be evaluated."""


def constant_index(value: Any) -> Optional[int]:
    """Index of ``value`` in :data:`CONSTANTS` (``None`` when absent)."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    for index, constant in enumerate(CONSTANTS):
        if number == constant:
            return index
    return None


def member_kind(member: dict) -> str:
    if not isinstance(member, dict):
        raise ProgramError(f"member must be an object, got {member!r}")
    kinds = [key for key in ("span", "ref", "const") if key in member]
    if len(kinds) != 1:
        raise ProgramError(f"member needs exactly one of span/ref/const: {member!r}")
    return kinds[0]


def validate_program(steps: Sequence[dict]) -> None:
    """Raise :class:`ProgramError` unless ``steps`` is well formed."""

    if not isinstance(steps, (list, tuple)) or not steps:
        raise ProgramError("program must be a non-empty list of steps")
    for index, step in enumerate(steps):
        op = step.get("op") if isinstance(step, dict) else None
        if op not in OPS:
            raise ProgramError(f"step {index}: unknown op {op!r}")
        args = step.get("args")
        if not isinstance(args, (list, tuple)) or len(args) > NUM_PORTS:
            raise ProgramError(f"step {index}: args must be a list of <= {NUM_PORTS} ports")
        arity, _ = OPS[op]
        ports = list(args) + [[]] * (NUM_PORTS - len(args))
        for port_index, port in enumerate(ports):
            if not isinstance(port, (list, tuple)):
                raise ProgramError(f"step {index}: port {port_index} must be a list")
            if port_index >= arity and port:
                raise ProgramError(f"step {index}: {op} uses {arity} port(s)")
            if port_index < arity and not port:
                raise ProgramError(f"step {index}: port {port_index} of {op} is empty")
            for member in port:
                kind = member_kind(member)
                if kind == "ref":
                    ref = member["ref"]
                    if not isinstance(ref, int) or not 0 <= ref < index:
                        raise ProgramError(f"step {index}: ref {ref!r} must point to an earlier step")
                elif kind == "const":
                    if constant_index(member["const"]) is None:
                        raise ProgramError(f"step {index}: constant {member['const']!r} not in vocabulary")
                elif kind == "span":
                    start, end = member.get("start"), member.get("end")
                    if not isinstance(start, int) or not isinstance(end, int) or end <= start:
                        raise ProgramError(f"step {index}: span needs integer start < end")


@dataclass
class _Item:
    value: Optional[float]   # None: not a number (fine for count/collect)
    label: Any = None


def _span_key(member: dict) -> tuple:
    return (int(member.get("start", 0)), int(member.get("end", 0)))


def canonical_port(port: Iterable[dict]) -> list[dict]:
    """Members in evaluation order: spans by position, refs, constants."""

    port = list(port)
    spans = sorted((m for m in port if "span" in m), key=_span_key)
    refs = sorted((m for m in port if "ref" in m), key=lambda m: int(m["ref"]))
    consts = sorted(
        (m for m in port if "const" in m),
        key=lambda m: constant_index(m["const"]) or 0,
    )
    return spans + refs + consts


def _numbers(items: list[_Item], op: str) -> list[float]:
    values = [item.value for item in items]
    if any(value is None for value in values):
        raise ProgramError(f"{op}: operand without a numeric value")
    return [float(value) for value in values]


def _scalar_or_list(items: list[_Item], op: str):
    numbers = _numbers(items, op)
    if not numbers:
        raise ProgramError(f"{op}: empty operand")
    return numbers[0] if len(numbers) == 1 else numbers


def _binary(op: str, left, right):
    def apply(a: float, b: float) -> float:
        if op == "add":
            return a + b
        if op == "subtract":
            return a - b
        if op == "multiply":
            return a * b
        if b == 0:
            raise ProgramError("division by zero")
        return a / b

    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ProgramError(f"{op}: list lengths differ ({len(left)} vs {len(right)})")
        return [apply(a, b) for a, b in zip(left, right)]
    if isinstance(left, list):
        return [apply(a, right) for a in left]
    if isinstance(right, list):
        return [apply(left, b) for b in right]
    return apply(left, right)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def execute_program(
    steps: Sequence[dict],
    span_value: Optional[Callable[[dict], Optional[float]]] = None,
) -> float:
    """Evaluate ``steps`` and return the final scalar.

    ``span_value(member)`` gives a span's number; by default the member's own
    ``"value"`` is used. Raises :class:`ProgramError` when evaluation fails.
    """

    if span_value is None:
        span_value = lambda member: member.get("value")  # noqa: E731
    results: list[Any] = []
    for index, step in enumerate(steps):
        op = step["op"]
        arity, _ = OPS[op]
        args = list(step.get("args") or []) + [[]] * NUM_PORTS
        ports: list[list[_Item]] = []
        for port in args[:NUM_PORTS]:
            items: list[_Item] = []
            for member in canonical_port(port):
                if "span" in member:
                    value = span_value(member)
                    items.append(_Item(None if value is None else float(value), member))
                elif "ref" in member:
                    ref = int(member["ref"])
                    if not 0 <= ref < index:
                        raise ProgramError(f"step {index}: bad ref {ref}")
                    result = results[ref]
                    if isinstance(result, list):
                        items.extend(_Item(v) for v in result)
                    else:
                        items.append(_Item(result))
                else:
                    items.append(_Item(float(member["const"])))
            ports.append(items)
        first, second = ports[0], ports[1]
        if not first or (arity == 2 and not second):
            raise ProgramError(f"step {index}: {op} has an empty operand")
        if op == "copy":
            result = _scalar_or_list(first, op)
            if isinstance(result, list):
                raise ProgramError("copy: more than one member")
        elif op == "collect":
            result = [item.value for item in first]
        elif op == "count":
            result = float(len(first))
        elif op in {"sum", "mean", "median", "min", "max", "product"}:
            values = _numbers(first, op)
            if op == "sum":
                result = math.fsum(values)
            elif op == "mean":
                result = math.fsum(values) / len(values)
            elif op == "median":
                result = _median(values)
            elif op == "min":
                result = min(values)
            elif op == "max":
                result = max(values)
            else:
                result = math.prod(values)
        elif op in {"sort_asc", "sort_desc"}:
            result = sorted(_numbers(first, op), reverse=op == "sort_desc")
        elif op == "diff_adjacent":
            values = _numbers(first, op)
            if len(values) < 2:
                raise ProgramError("diff_adjacent: needs at least two values")
            result = [b - a for a, b in zip(values, values[1:])]
        elif op.startswith("filter_"):
            values = _numbers(first, op)
            threshold = _scalar_or_list(second, op)
            if isinstance(threshold, list):
                raise ProgramError(f"{op}: threshold must be one value")
            compare = {
                "filter_gt": lambda v: v > threshold,
                "filter_ge": lambda v: v >= threshold,
                "filter_lt": lambda v: v < threshold,
                "filter_le": lambda v: v <= threshold,
            }[op]
            result = [v for v in values if compare(v)]
        else:
            result = _binary(op, _scalar_or_list(first, op), _scalar_or_list(second, op))
        results.append(result)
    final = results[-1]
    if isinstance(final, list):
        if len(final) != 1:
            raise ProgramError("program result is a list")
        final = final[0]
    if final is None or not math.isfinite(float(final)):
        raise ProgramError("program result is not a finite number")
    return float(final)


def program_spans(steps: Sequence[dict]) -> list[dict]:
    """Unique span members of a program, in text order."""

    seen = {}
    for step in steps:
        for port in step.get("args") or []:
            for member in port:
                if "span" in member:
                    seen.setdefault(_span_key(member), member)
    return [seen[key] for key in sorted(seen)]


__all__ = [
    "CONSTANTS",
    "NUM_PORTS",
    "OPS",
    "OP_NAMES",
    "OP_TO_ID",
    "ProgramError",
    "canonical_port",
    "constant_index",
    "execute_program",
    "program_spans",
    "validate_program",
]
