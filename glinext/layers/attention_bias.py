"""Configurable additive attention-bias strategies for anchor refinement."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"attention bias {name} must be finite")
    return value


def _coordinates(
    value: torch.Tensor | None,
    *,
    name: str,
    length: int,
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        raise ValueError(f"{name} coordinates are required by this attention bias")
    if value.dim() == 2:
        value = value.unsqueeze(0)
    if value.dim() != 3 or value.shape[-2] != length:
        raise ValueError(
            f"{name} coordinates must have shape (N, C), (1, N, C), or "
            f"(B, N, C); got {tuple(value.shape)} for length {length}"
        )
    return value.to(device=device, dtype=torch.float32)


def _broadcast_coordinate_batches(
    queries: torch.Tensor,
    keys: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = max(queries.shape[0], keys.shape[0])
    for name, value in (("query", queries), ("key", keys)):
        if value.shape[0] not in {1, batch_size}:
            raise ValueError(
                f"{name} coordinate batch must be 1 or {batch_size}, "
                f"got {value.shape[0]}"
            )
    if queries.shape[0] == 1 and batch_size != 1:
        queries = queries.expand(batch_size, -1, -1)
    if keys.shape[0] == 1 and batch_size != 1:
        keys = keys.expand(batch_size, -1, -1)
    return queries, keys


def _select_dimensions(
    coordinates: torch.Tensor,
    dimensions: Sequence[int] | None,
    width: int | None = None,
) -> torch.Tensor:
    if dimensions is not None:
        indices = torch.as_tensor(
            tuple(int(index) for index in dimensions),
            dtype=torch.long,
            device=coordinates.device,
        )
        if indices.numel() == 0:
            raise ValueError("attention bias coordinate dimensions cannot be empty")
        if (indices < 0).any() or (indices >= coordinates.shape[-1]).any():
            raise ValueError(
                "attention bias coordinate dimension is outside the supplied "
                f"width {coordinates.shape[-1]}"
            )
        coordinates = coordinates.index_select(-1, indices)
    if width is not None:
        if coordinates.shape[-1] < width:
            raise ValueError(
                f"attention bias requires {width} coordinate dimensions, "
                f"got {coordinates.shape[-1]}"
            )
        coordinates = coordinates[..., :width]
    return coordinates


class AttentionBias(nn.Module):
    """Base class and registry for additive self/cross-attention biases."""

    _registry: dict[str, type["AttentionBias"]] = {}
    config_fields: frozenset[str] = frozenset()

    def __init_subclass__(cls, bias_type: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if bias_type:
            AttentionBias._registry[bias_type] = cls

    @classmethod
    def strategy_class(cls, bias_type: str) -> type["AttentionBias"]:
        normalized = str(bias_type or "none").lower().replace("-", "_")
        strategy = cls._registry.get(normalized)
        if strategy is None:
            raise ValueError(
                f"Unknown attention bias {bias_type!r}. "
                f"Available: {sorted(cls._registry)}"
            )
        return strategy

    @classmethod
    def from_config(
        cls,
        config: str | Mapping[str, Any] | Sequence[Any] | None,
        *,
        num_heads: int,
    ) -> "AttentionBias":
        """Build one strategy or a sum of strategies from configuration."""

        if isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
            return CompositeAttentionBias(
                [cls.from_config(item, num_heads=num_heads) for item in config]
            )
        if config is None:
            bias_type = "none"
            params: dict[str, Any] = {}
        elif isinstance(config, str):
            bias_type = config
            params = {}
        elif isinstance(config, Mapping):
            raw = dict(config)
            bias_type = raw.pop("type", raw.pop("name", "none"))
            configured_params = raw.pop("params", {})
            if not isinstance(configured_params, Mapping):
                raise TypeError("attention bias params must be a mapping")
            params = dict(configured_params)
            params.update(raw)
        else:
            raise TypeError(
                "attention bias must be a string, mapping, sequence, or None"
            )

        strategy = cls.strategy_class(str(bias_type))
        unknown = set(params) - set(strategy.config_fields)
        if unknown:
            raise ValueError(
                f"Unsupported {bias_type!r} attention bias options: "
                f"{sorted(unknown)}. Available: {sorted(strategy.config_fields)}"
            )
        return strategy(num_heads=num_heads, **params)

    def forward(
        self,
        *,
        query_coordinates: torch.Tensor | None,
        key_coordinates: torch.Tensor | None,
        query_length: int,
        key_length: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_index: int = 0,
    ) -> torch.Tensor | None:
        raise NotImplementedError


class NoAttentionBias(AttentionBias, bias_type="none"):
    def __init__(self, num_heads: int, **kwargs):
        super().__init__()

    def forward(self, **kwargs):
        return None


class GaussianDistanceAttentionBias(
    AttentionBias,
    bias_type="gaussian_distance",
):
    """Gaussian locality prior over normalized 1D or 2D coordinates."""

    config_fields = frozenset(
        {
            "sigma",
            "weight",
            "units",
            "query_dimensions",
            "key_dimensions",
        }
    )

    def __init__(
        self,
        num_heads: int,
        sigma: float = 0.2,
        weight: float = 1.0,
        units: str = "normalized",
        query_dimensions: Sequence[int] | None = None,
        key_dimensions: Sequence[int] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.sigma = _finite("sigma", sigma)
        if self.sigma <= 0.0:
            raise ValueError("attention bias sigma must be positive")
        self.weight = _finite("weight", weight)
        if self.weight < 0.0:
            raise ValueError("attention bias weight must be non-negative")
        self.units = str(units).lower().replace("-", "_")
        if self.units not in {"normalized", "query_steps"}:
            raise ValueError(
                "Gaussian attention bias units must be 'normalized' or "
                "'query_steps'"
            )
        self.query_dimensions = (
            None if query_dimensions is None else tuple(query_dimensions)
        )
        self.key_dimensions = (
            None if key_dimensions is None else tuple(key_dimensions)
        )

    def forward(
        self,
        *,
        query_coordinates,
        key_coordinates,
        query_length,
        key_length,
        dtype,
        device,
        **kwargs,
    ):
        queries = _coordinates(
            query_coordinates,
            name="query",
            length=query_length,
            device=device,
        )
        keys = _coordinates(
            key_coordinates,
            name="key",
            length=key_length,
            device=device,
        )
        queries = _select_dimensions(queries, self.query_dimensions)
        keys = _select_dimensions(keys, self.key_dimensions)
        coordinate_width = min(queries.shape[-1], keys.shape[-1])
        queries = _select_dimensions(queries, None, coordinate_width)
        keys = _select_dimensions(keys, None, coordinate_width)
        queries, keys = _broadcast_coordinate_batches(queries, keys)
        delta = queries.unsqueeze(-2) - keys.unsqueeze(-3)
        if self.units == "query_steps":
            delta = delta * max(int(query_length), 1)
        squared_distance = delta.square().sum(dim=-1)
        bias = -0.5 * self.weight * squared_distance / (self.sigma**2)
        return bias.clamp_min(-10_000.0).to(dtype=dtype)


class LocalWindowAttentionBias(AttentionBias, bias_type="local_window"):
    """Apply an additive penalty outside a coordinate-space radius."""

    config_fields = frozenset(
        {
            "radius",
            "outside_bias",
            "units",
            "query_dimensions",
            "key_dimensions",
        }
    )

    def __init__(
        self,
        num_heads: int,
        radius: float = 1.0,
        outside_bias: float = -10_000.0,
        units: str = "query_steps",
        query_dimensions: Sequence[int] | None = None,
        key_dimensions: Sequence[int] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.radius = _finite("radius", radius)
        if self.radius < 0.0:
            raise ValueError("local-window radius must be non-negative")
        self.outside_bias = _finite("outside_bias", outside_bias)
        self.units = str(units).lower().replace("-", "_")
        if self.units not in {"normalized", "query_steps"}:
            raise ValueError(
                "Local-window attention bias units must be 'normalized' or "
                "'query_steps'"
            )
        self.query_dimensions = (
            None if query_dimensions is None else tuple(query_dimensions)
        )
        self.key_dimensions = (
            None if key_dimensions is None else tuple(key_dimensions)
        )

    def forward(
        self,
        *,
        query_coordinates,
        key_coordinates,
        query_length,
        key_length,
        dtype,
        device,
        **kwargs,
    ):
        queries = _select_dimensions(
            _coordinates(
                query_coordinates,
                name="query",
                length=query_length,
                device=device,
            ),
            self.query_dimensions,
        )
        keys = _select_dimensions(
            _coordinates(
                key_coordinates,
                name="key",
                length=key_length,
                device=device,
            ),
            self.key_dimensions,
        )
        width = min(queries.shape[-1], keys.shape[-1])
        queries, keys = _broadcast_coordinate_batches(
            queries[..., :width], keys[..., :width]
        )
        distance = (queries.unsqueeze(-2) - keys.unsqueeze(-3)).square().sum(
            dim=-1
        ).sqrt()
        if self.units == "query_steps":
            distance = distance * max(int(query_length), 1)
        output = torch.zeros_like(distance)
        return output.masked_fill(
            distance > self.radius,
            self.outside_bias,
        ).to(dtype=dtype)


class CausalAttentionBias(AttentionBias, bias_type="causal"):
    """Standard causal additive mask, intended for anchor self-attention."""

    config_fields = frozenset({"future_bias"})

    def __init__(
        self,
        num_heads: int,
        future_bias: float = -10_000.0,
        **kwargs,
    ):
        super().__init__()
        self.future_bias = _finite("future_bias", future_bias)

    def forward(
        self,
        *,
        query_length,
        key_length,
        dtype,
        device,
        **kwargs,
    ):
        rows = torch.arange(query_length, device=device).unsqueeze(1)
        columns = torch.arange(key_length, device=device).unsqueeze(0)
        return torch.zeros(
            query_length,
            key_length,
            dtype=dtype,
            device=device,
        ).masked_fill(columns > rows, self.future_bias)


class RelativeMLPAttentionBias(AttentionBias, bias_type="relative_mlp"):
    """Learned per-head additive bias from relative coordinates."""

    config_fields = frozenset(
        {
            "coordinate_dimensions",
            "hidden_size",
            "query_dimensions",
            "key_dimensions",
        }
    )

    def __init__(
        self,
        num_heads: int,
        coordinate_dimensions: int,
        hidden_size: int = 32,
        query_dimensions: Sequence[int] | None = None,
        key_dimensions: Sequence[int] | None = None,
        **kwargs,
    ):
        super().__init__()
        coordinate_dimensions = int(coordinate_dimensions)
        hidden_size = int(hidden_size)
        if coordinate_dimensions <= 0 or hidden_size <= 0:
            raise ValueError(
                "relative-MLP coordinate_dimensions and hidden_size must be positive"
            )
        self.coordinate_dimensions = coordinate_dimensions
        self.num_heads = int(num_heads)
        self.query_dimensions = (
            None if query_dimensions is None else tuple(query_dimensions)
        )
        self.key_dimensions = (
            None if key_dimensions is None else tuple(key_dimensions)
        )
        self.mlp = nn.Sequential(
            nn.Linear(coordinate_dimensions, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.num_heads),
        )

    def forward(
        self,
        *,
        query_coordinates,
        key_coordinates,
        query_length,
        key_length,
        dtype,
        device,
        **kwargs,
    ):
        queries = _select_dimensions(
            _coordinates(
                query_coordinates,
                name="query",
                length=query_length,
                device=device,
            ),
            self.query_dimensions,
            self.coordinate_dimensions,
        )
        keys = _select_dimensions(
            _coordinates(
                key_coordinates,
                name="key",
                length=key_length,
                device=device,
            ),
            self.key_dimensions,
            self.coordinate_dimensions,
        )
        queries, keys = _broadcast_coordinate_batches(queries, keys)
        delta = queries.unsqueeze(-2) - keys.unsqueeze(-3)
        parameter = next(self.mlp.parameters())
        bias = self.mlp(delta.to(dtype=parameter.dtype))
        return bias.permute(0, 3, 1, 2).to(dtype=dtype)


class CompositeAttentionBias(AttentionBias):
    """Sum multiple independent additive-bias strategies."""

    def __init__(self, biases: Sequence[AttentionBias]):
        super().__init__()
        self.biases = nn.ModuleList(biases)

    def forward(self, **kwargs):
        result = None
        for bias in self.biases:
            value = bias(**kwargs)
            if value is not None:
                result = value if result is None else result + value
        return result


# Short and legacy-friendly aliases.
AttentionBias._registry["gaussian"] = GaussianDistanceAttentionBias
AttentionBias._registry["window"] = LocalWindowAttentionBias


__all__ = [
    "AttentionBias",
    "NoAttentionBias",
    "GaussianDistanceAttentionBias",
    "LocalWindowAttentionBias",
    "CausalAttentionBias",
    "RelativeMLPAttentionBias",
    "CompositeAttentionBias",
]
