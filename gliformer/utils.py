"""Small dependency-free utilities shared across GLiFormer modules."""

from collections.abc import Iterable
from typing import Any


def pair_2d(value: Any) -> tuple[int, int]:
    """Normalize a scalar or two-item iterable to an integer ``(h, w)`` pair."""

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        values = tuple(value)
        if len(values) != 2:
            raise ValueError(f"Expected an int or 2-item pair, got {value!r}")
        return int(values[0]), int(values[1])
    return int(value), int(value)


__all__ = ["pair_2d"]
