"""Bin support of the normalization head's bin number classifier (torch-free).

A number ``y`` of a field is mapped to a transformed coordinate ``z``:

* ``asinh``: ``z = asinh(y / s)`` for any sign, inverse ``y = s * sinh(z)``;
* ``log10``: ``z = log10(|y| / s)`` for ``|y| / s >= ZERO_EPSILON``. Zero is
  a separate class, and the sign is a separate prediction.

``s`` is the configured ``scale``; it never depends on the gold answer.

Bins are fixed, strictly increasing edges ``e_0 < ... < e_B`` in ``z``.
Interior bin ``i`` covers ``[e_i, e_{i+1})``. With ``unbounded`` support a
left tail ``(-inf, e_0)`` and a right tail ``[e_B, +inf)`` are extra
classes; each tail has an outward half-normal density with scale
``tail_scale``. With ``finite`` support, ``z == e_B`` falls in the last bin
and anything outside ``[e_0, e_B]`` has no class (it is masked, never
clamped).

Class layout: ``[zero]`` (log10 only), ``[left tail]`` (unbounded only),
the ``B`` interior bins, ``[right tail]`` (unbounded only).

Resolution limit: every value inside one bin receives the same hard-bin
supervision, so the decoded value can be off by up to half a bin. With
``asinh``, scale 1, and 1024 bins on ``[-20, 20]``, that is about 2%
relative error for large values (about 1% with 2048 bins). Tail classes say
only that a value lies beyond an edge, not where.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from collections.abc import Sequence
from statistics import NormalDist
from typing import Any

ZERO_EPSILON = 1e-12
_NORMAL = NormalDist()
_TINY = 1e-12


@dataclass(frozen=True)
class BinSupport:
    edges: tuple[float, ...]
    transform: str = "asinh"
    scale: float = 1.0
    support: str = "unbounded"
    tail_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.transform not in {"asinh", "log10"}:
            raise ValueError("transform must be 'asinh' or 'log10'")
        if self.support not in {"unbounded", "finite"}:
            raise ValueError("support must be 'unbounded' or 'finite'")
        if len(self.edges) < 2 or any(b <= a for a, b in zip(self.edges, self.edges[1:], strict=False)):
            raise ValueError("bin edges must be strictly increasing")
        if not self.scale > 0 or not self.tail_scale > 0:
            raise ValueError("scale and tail_scale must be positive")

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_config(cls, bins_cfg: Any) -> BinSupport:
        """Support from a ``NormalizationBinsConfig``."""

        edges = bins_cfg.edges
        if edges is None:
            low, high = bins_cfg.range
            edges = uniform_edges(int(bins_cfg.count), low, high)
        return cls(
            edges=tuple(float(value) for value in edges),
            transform=str(bins_cfg.transform),
            scale=float(bins_cfg.scale),
            support=str(bins_cfg.support),
            tail_scale=float(bins_cfg.tail_scale),
        )

    # -- layout ---------------------------------------------------------------

    @property
    def num_interior(self) -> int:
        return len(self.edges) - 1

    @property
    def has_tails(self) -> bool:
        return self.support == "unbounded"

    @property
    def has_zero(self) -> bool:
        return self.transform == "log10"

    @property
    def first_interior(self) -> int:
        return int(self.has_zero) + int(self.has_tails)

    @property
    def num_classes(self) -> int:
        return self.first_interior + self.num_interior + int(self.has_tails)

    # -- transform ------------------------------------------------------------

    def to_z(self, y: float) -> float | None:
        """Transformed coordinate; ``None`` for zero under ``log10``."""

        s = self.scale
        if self.transform == "asinh":
            return math.asinh(y / s)
        magnitude = abs(y) / s
        if magnitude < ZERO_EPSILON:
            return None
        return math.log10(magnitude)

    def from_z(self, z: float, *, negative: bool = False) -> float:
        s = self.scale
        if self.transform == "asinh":
            return s * math.sinh(z)
        value = s * 10.0 ** min(z, 300.0)
        return -value if negative else value

    def target_class(self, y: float) -> int | None:
        """Class index of ``y``, or ``None`` when outside a finite support."""

        z = self.to_z(y)
        if z is None:
            return 0  # zero class (log10)
        return self.z_class(z)

    def z_class(self, z: float) -> int | None:
        edges = self.edges
        position = bisect.bisect_right(edges, z)  # edges[p-1] <= z < edges[p]
        if self.has_tails:
            return int(self.has_zero) + position
        if position == 0:
            return None
        if position == len(edges):
            return None if z > edges[-1] else int(self.has_zero) + self.num_interior - 1
        return int(self.has_zero) + position - 1

    # -- pieces of the distribution in value space -----------------------------

    def _z_pieces(self, probs: Sequence[float]):
        """``(mass, kind, a, b)`` in increasing ``z``; kind in tail_l/bin/tail_r."""

        offset = int(self.has_zero)
        pieces = []
        if self.has_tails:
            pieces.append((float(probs[offset]), "tail_l", self.edges[0], self.edges[0]))
        start = offset + int(self.has_tails)
        for index in range(self.num_interior):
            pieces.append((float(probs[start + index]), "bin", self.edges[index], self.edges[index + 1]))
        if self.has_tails:
            pieces.append((float(probs[start + self.num_interior]), "tail_r", self.edges[-1], self.edges[-1]))
        return pieces

    def _piece_z(self, kind: str, a: float, b: float, u: float) -> float:
        """``z`` at within-piece CDF ``u`` (increasing in ``z``)."""

        u = min(max(u, _TINY), 1.0 - _TINY)
        sigma = self.tail_scale
        if kind == "bin":
            return a + u * (b - a)
        if kind == "tail_l":  # z = e0 - D, D ~ HalfNormal(sigma)
            return a - sigma * _NORMAL.inv_cdf(1.0 - u / 2.0)
        return b + sigma * _NORMAL.inv_cdf((1.0 + u) / 2.0)

    def _value_segments(self, probs: Sequence[float], negative_prob: float):
        """Ordered segments of the distribution along the real line.

        Each is ``(mass, kind, a, b, sign)``; sign -1 segments are mirrored
        log10 magnitudes, sign 0 is the zero atom.
        """

        total = sum(max(float(p), 0.0) for p in probs) or 1.0
        probs = [max(float(p), 0.0) / total for p in probs]
        pieces = self._z_pieces(probs)
        if self.transform == "asinh":
            return [(mass, kind, a, b, 1) for mass, kind, a, b in pieces]
        negative_prob = min(max(float(negative_prob), 0.0), 1.0)
        segments = [
            (mass * negative_prob, kind, a, b, -1) for mass, kind, a, b in reversed(pieces)
        ]
        segments.append((probs[0], "zero", 0.0, 0.0, 0))
        segments.extend((mass * (1.0 - negative_prob), kind, a, b, 1) for mass, kind, a, b in pieces)
        return segments

    def _segment_value(self, segment, u: float) -> float:
        _mass, kind, a, b, sign = segment
        if sign == 0:
            return 0.0
        if sign < 0:  # larger z = more negative; flip the within-piece CDF
            return self.from_z(self._piece_z(kind, a, b, 1.0 - u), negative=True)
        return self.from_z(self._piece_z(kind, a, b, u))

    # -- statistics -----------------------------------------------------------

    def quantile(
        self,
        probs: Sequence[float],
        level: float,
        *,
        negative_prob: float = 0.0,
    ) -> float:
        """Value at quantile ``level`` (0.5 is the median)."""

        segments = self._value_segments(probs, negative_prob)
        level = min(max(float(level), 0.0), 1.0)
        total = 0.0
        for segment in segments:
            mass = segment[0]
            if mass > 0 and total + mass >= level:
                return self._segment_value(segment, (level - total) / mass)
            total += mass
        last = next((segment for segment in reversed(segments) if segment[0] > 0), segments[-1])
        return self._segment_value(last, 1.0)


def uniform_edges(count: int, low: float, high: float) -> list[float]:
    """``count`` bins of equal width on ``[low, high]`` (``count + 1`` edges)."""

    step = (float(high) - float(low)) / int(count)
    edges = [float(low) + index * step for index in range(int(count))]
    edges.append(float(high))
    return edges


__all__ = ["BinSupport", "ZERO_EPSILON", "uniform_edges"]
