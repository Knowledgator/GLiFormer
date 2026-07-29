"""Configurable positional embeddings for spatial task heads."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


def _finite_float(name: str, value) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"position embedding {name} must be finite")
    return value


def _positive_temperature(value) -> float:
    value = _finite_float("temperature", value)
    if value <= 0.0:
        raise ValueError("position embedding temperature must be positive")
    return value


def _nonnegative_init_std(value) -> float:
    value = _finite_float("init_std", value)
    if value < 0.0:
        raise ValueError("position embedding init_std must be non-negative")
    return value


def normalized_grid_2d(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return flattened cell-centre coordinates in normalized ``(x, y)`` order."""
    if height <= 0 or width <= 0:
        raise ValueError(f"Spatial shape must be positive, got {(height, width)}")
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack(
        [(xx.flatten() + 0.5) / width, (yy.flatten() + 0.5) / height],
        dim=-1,
    )


def normalized_grid_1d(
    count: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return normalized temporal cell centres with shape ``(count, 1)``."""

    if count <= 0:
        raise ValueError(f"Temporal position count must be positive, got {count}")
    positions = torch.arange(count, device=device, dtype=dtype)
    return ((positions + 0.5) / count).unsqueeze(-1)


def covering_grid_2d(count: int) -> tuple[int, int]:
    """Return a compact row/column grid with at least ``count`` cells."""

    if count <= 0:
        raise ValueError(f"Grid count must be positive, got {count}")
    columns = max(int(math.sqrt(count)), 1)
    rows = (int(count) + columns - 1) // columns
    return rows, columns


class PositionEmbedding(nn.Module):
    """Base class for registry-backed positional embedding strategies."""

    _registry: dict[str, type[PositionEmbedding]] = {}
    requires_coordinates: bool = False
    requires_num_embeddings: bool = False
    requires_grid_size: bool = False
    coordinate_dimensions: int | None = None
    config_fields: frozenset[str] = frozenset()

    def __init_subclass__(cls, position_type: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if position_type:
            PositionEmbedding._registry[position_type] = cls

    @classmethod
    def from_config(
        cls,
        position_type: str | Mapping[str, Any],
        hidden_size: int,
        **kwargs,
    ) -> PositionEmbedding:
        if isinstance(position_type, Mapping):
            raw = dict(position_type)
            position_type = raw.pop("type", raw.pop("name", "none"))
            configured_params = raw.pop("params", {})
            if not isinstance(configured_params, Mapping):
                raise TypeError("position embedding params must be a mapping")
            params = dict(configured_params)
            params.update(raw)
            overlap = set(params) & set(kwargs)
            if overlap:
                raise ValueError(
                    "position embedding parameters were supplied twice: "
                    f"{sorted(overlap)}"
                )
            params.update(kwargs)
            kwargs = params
        strategy = cls.strategy_class(position_type)
        cls.validate_kwargs(position_type, kwargs)
        return strategy(hidden_size=hidden_size, **kwargs)

    @classmethod
    def validate_kwargs(cls, position_type: str, kwargs: dict) -> None:
        """Reject misspelled or strategy-incompatible constructor options."""

        strategy = cls.strategy_class(position_type)
        unknown = set(kwargs) - set(strategy.config_fields)
        if unknown:
            raise ValueError(
                f"Unsupported {position_type!r} position embedding options: "
                f"{sorted(unknown)}. Available: {sorted(strategy.config_fields)}"
            )
        # Validate shared numeric semantics while the owning task config is
        # being built, rather than failing later during model construction.
        if "scale" in kwargs:
            _finite_float("scale", kwargs["scale"])
        if "temperature" in kwargs:
            _positive_temperature(kwargs["temperature"])
        if "init_std" in kwargs:
            _nonnegative_init_std(kwargs["init_std"])
        for name in ("max_position", "min_frequency", "max_frequency"):
            if name in kwargs:
                value = _finite_float(name, kwargs[name])
                if value <= 0.0:
                    raise ValueError(
                        f"position embedding {name} must be positive"
                    )
        if (
            "min_frequency" in kwargs
            and "max_frequency" in kwargs
            and float(kwargs["max_frequency"]) < float(kwargs["min_frequency"])
        ):
            raise ValueError(
                "position embedding max_frequency must be at least "
                "min_frequency"
            )
        if (
            "frequency_spacing" in kwargs
            and str(kwargs["frequency_spacing"]).lower() not in {"linear", "log"}
        ):
            raise ValueError("frequency_spacing must be 'linear' or 'log'")

    @classmethod
    def strategy_class(cls, position_type: str) -> type[PositionEmbedding]:
        """Resolve a registered strategy without constructing it."""

        normalized = str(position_type or "none").lower().replace("-", "_")
        strategy = cls._registry.get(normalized)
        if strategy is None:
            raise ValueError(
                f"Unknown position embedding type {position_type!r}. "
                f"Available: {sorted(cls._registry)}"
            )
        return strategy

    def forward(
        self,
        coordinates: torch.Tensor | None = None,
        *,
        count: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor | None:
        raise NotImplementedError


class NoPositionEmbedding(PositionEmbedding, position_type="none"):
    def __init__(self, hidden_size: int, **kwargs):
        super().__init__()

    def forward(self, coordinates=None, **kwargs):
        return None


class Sine2DPositionEmbedding(PositionEmbedding, position_type="sine2d"):
    """Fixed DETR-style sine/cosine features for normalized 2D coordinates."""

    requires_coordinates = True
    coordinate_dimensions = 2
    config_fields = frozenset({"scale", "temperature"})

    def __init__(
        self,
        hidden_size: int,
        scale: float = 1.0,
        temperature: float = 10_000.0,
        **kwargs,
    ):
        super().__init__()
        if hidden_size % 4 != 0:
            raise ValueError(
                "sine2d positional embeddings require hidden_size divisible by 4, "
                f"got {hidden_size}"
            )
        self.hidden_size = int(hidden_size)
        self.scale = _finite_float("scale", scale)
        self.temperature = _positive_temperature(temperature)

    def forward(self, coordinates=None, *, dtype=None, device=None, **kwargs):
        if coordinates is None:
            raise ValueError("sine2d positional embeddings require coordinates")
        coordinates = coordinates.to(device=device or coordinates.device, dtype=torch.float32)
        half = self.hidden_size // 2
        dim_t = torch.arange(half, device=coordinates.device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / half)
        features = []
        for axis in range(2):
            pos = coordinates[..., axis : axis + 1] * (2 * math.pi) / dim_t
            encoded = torch.stack(
                (pos[..., 0::2].sin(), pos[..., 1::2].cos()),
                dim=-1,
            ).flatten(-2)
            features.append(encoded)
        output = self.scale * torch.cat(features, dim=-1)
        return output.to(dtype=dtype or coordinates.dtype)


class SineBBox2DPositionEmbedding(
    PositionEmbedding,
    position_type="sine_bbox2d",
):
    """Fixed sine/cosine features for normalized ``(cx, cy, w, h)`` boxes.

    Each box component owns one quarter of the output channels.  Requiring an
    even number of channels per component preserves paired sine/cosine features
    while retaining the complete box geometry rather than encoding only its
    centre.
    """

    requires_coordinates = True
    coordinate_dimensions = 4
    config_fields = frozenset({"scale", "temperature"})

    def __init__(
        self,
        hidden_size: int,
        scale: float = 1.0,
        temperature: float = 10_000.0,
        **kwargs,
    ):
        super().__init__()
        if hidden_size % 8 != 0:
            raise ValueError(
                "sine_bbox2d positional embeddings require hidden_size divisible "
                f"by 8, got {hidden_size}"
            )
        self.hidden_size = int(hidden_size)
        self.scale = _finite_float("scale", scale)
        self.temperature = _positive_temperature(temperature)

    def forward(self, coordinates=None, *, dtype=None, device=None, **kwargs):
        if coordinates is None or coordinates.shape[-1] != 4:
            raise ValueError(
                "sine_bbox2d positions require (..., 4) normalized "
                "(cx, cy, w, h) coordinates"
            )
        coordinates = coordinates.to(
            device=device or coordinates.device,
            dtype=torch.float32,
        )
        channels_per_component = self.hidden_size // 4
        dim_t = torch.arange(
            channels_per_component,
            device=coordinates.device,
            dtype=torch.float32,
        )
        dim_t = self.temperature ** (
            2
            * torch.div(dim_t, 2, rounding_mode="floor")
            / channels_per_component
        )
        features = []
        for component in range(4):
            angles = (
                coordinates[..., component : component + 1]
                * (2 * math.pi)
                / dim_t
            )
            encoded = torch.stack(
                (angles[..., 0::2].sin(), angles[..., 1::2].cos()),
                dim=-1,
            ).flatten(-2)
            features.append(encoded)
        output = self.scale * torch.cat(features, dim=-1)
        return output.to(dtype=dtype or coordinates.dtype)


class Linear2DPositionEmbedding(PositionEmbedding, position_type="linear2d"):
    """Learned linear projection of normalized 2D coordinates."""

    requires_coordinates = True
    coordinate_dimensions = 2
    config_fields = frozenset({"scale"})

    def __init__(self, hidden_size: int, scale: float = 1.0, **kwargs):
        super().__init__()
        self.projection = nn.Linear(2, hidden_size, bias=False)
        self.scale = _finite_float("scale", scale)

    def forward(self, coordinates=None, *, dtype=None, device=None, **kwargs):
        if coordinates is None:
            raise ValueError("linear2d positional embeddings require coordinates")
        parameter = self.projection.weight
        coordinates = coordinates.to(device=device or parameter.device, dtype=parameter.dtype)
        return (self.scale * self.projection(coordinates)).to(dtype=dtype or parameter.dtype)


class MLP2DPositionEmbedding(PositionEmbedding, position_type="mlp2d"):
    """Learned non-linear projection of normalized 2D coordinates."""

    requires_coordinates = True
    coordinate_dimensions = 2
    config_fields = frozenset({"scale"})

    def __init__(self, hidden_size: int, scale: float = 1.0, **kwargs):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.scale = _finite_float("scale", scale)

    def forward(self, coordinates=None, *, dtype=None, device=None, **kwargs):
        if coordinates is None:
            raise ValueError("mlp2d positional embeddings require coordinates")
        parameter = next(self.projection.parameters())
        coordinates = coordinates.to(device=device or parameter.device, dtype=parameter.dtype)
        return (self.scale * self.projection(coordinates)).to(dtype=dtype or parameter.dtype)


class LearnedIndexPositionEmbedding(PositionEmbedding, position_type="learned"):
    """Learned positional identity for a fixed number of query slots."""

    requires_num_embeddings = True
    config_fields = frozenset({"num_embeddings", "init_std", "scale"})

    def __init__(
        self,
        hidden_size: int,
        num_embeddings: int | None = None,
        init_std: float = 1.0,
        scale: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        if not num_embeddings or num_embeddings <= 0:
            raise ValueError("learned positional embeddings require num_embeddings > 0")
        init_std = _nonnegative_init_std(init_std)
        self.embedding = nn.Embedding(int(num_embeddings), hidden_size)
        nn.init.normal_(self.embedding.weight, std=init_std)
        self.scale = _finite_float("scale", scale)

    def forward(self, coordinates=None, *, count=None, dtype=None, device=None, **kwargs):
        if count is None:
            count = coordinates.shape[-2] if coordinates is not None else self.embedding.num_embeddings
        if count > self.embedding.num_embeddings:
            raise ValueError(
                f"Requested {count} learned positions, but only "
                f"{self.embedding.num_embeddings} are configured"
            )
        output = self.scale * self.embedding.weight[:count]
        return output.to(device=device or output.device, dtype=dtype or output.dtype)


class LearnedGrid2DPositionEmbedding(PositionEmbedding, position_type="learned_grid2d"):
    """Learned 2D table interpolated to the current feature-map resolution.

    Unlike an index table this strategy is resolution-flexible, which makes it
    suitable for local patch encoders and rectangular images.  The base grid is
    a parameter of the configuration rather than an assumption in the caller.
    """

    requires_grid_size = True
    coordinate_dimensions = 2
    config_fields = frozenset(
        {"grid_size", "init_std", "scale", "interpolation_mode"}
    )

    def __init__(
        self,
        hidden_size: int,
        grid_size,
        init_std: float = 0.02,
        scale: float = 1.0,
        interpolation_mode: str = "bicubic",
        **kwargs,
    ):
        super().__init__()
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)
        if not grid_size or len(grid_size) != 2:
            raise ValueError("learned_grid2d requires a two-item grid_size")
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        if min(self.grid_size) <= 0:
            raise ValueError("learned_grid2d grid_size values must be positive")
        init_std = _nonnegative_init_std(init_std)
        if interpolation_mode not in {"nearest", "bilinear", "bicubic"}:
            raise ValueError(
                "learned_grid2d interpolation_mode must be nearest, bilinear, or bicubic"
            )
        self.hidden_size = int(hidden_size)
        self.scale = _finite_float("scale", scale)
        self.interpolation_mode = interpolation_mode
        self.embedding = nn.Parameter(
            torch.empty(1, self.grid_size[0] * self.grid_size[1], self.hidden_size)
        )
        nn.init.trunc_normal_(self.embedding, std=init_std)

    def forward(
        self,
        coordinates=None,
        *,
        count=None,
        spatial_shape=None,
        dtype=None,
        device=None,
        **kwargs,
    ):
        if spatial_shape is None:
            if count is None or int(count) == self.embedding.shape[1]:
                spatial_shape = self.grid_size
            else:
                raise ValueError(
                    "learned_grid2d requires spatial_shape when the requested "
                    "count differs from its base grid"
                )
        height, width = (int(spatial_shape[0]), int(spatial_shape[1]))
        if height <= 0 or width <= 0:
            raise ValueError("learned_grid2d spatial_shape values must be positive")
        table = self.embedding.view(1, self.grid_size[0], self.grid_size[1], self.hidden_size)
        table = table.permute(0, 3, 1, 2)
        if (height, width) != self.grid_size:
            interpolate_kwargs = {
                "size": (height, width),
                "mode": self.interpolation_mode,
            }
            if self.interpolation_mode != "nearest":
                interpolate_kwargs["align_corners"] = False
            table = torch.nn.functional.interpolate(table, **interpolate_kwargs)
        output = self.scale * table.flatten(2).transpose(1, 2)
        return output.to(device=device or output.device, dtype=dtype or output.dtype)


class Sine1DPositionEmbedding(PositionEmbedding, position_type="sine1d"):
    """Fixed sine/cosine features for normalized temporal coordinates."""

    requires_coordinates = True
    coordinate_dimensions = 1
    config_fields = frozenset({"scale", "temperature"})

    def __init__(
        self,
        hidden_size: int,
        scale: float = 1.0,
        temperature: float = 10_000.0,
        **kwargs,
    ):
        super().__init__()
        if hidden_size % 2 != 0:
            raise ValueError(
                "sine1d positional embeddings require an even hidden_size"
            )
        self.hidden_size = int(hidden_size)
        self.scale = _finite_float("scale", scale)
        self.temperature = _positive_temperature(temperature)

    def forward(self, coordinates=None, *, dtype=None, device=None, **kwargs):
        if coordinates is None or coordinates.shape[-1] != 1:
            raise ValueError("sine1d positions require (..., 1) coordinates")
        coordinates = coordinates.to(
            device=device or coordinates.device,
            dtype=torch.float32,
        )
        dim_t = torch.arange(
            self.hidden_size,
            device=coordinates.device,
            dtype=torch.float32,
        )
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.hidden_size
        )
        angles = coordinates * (2 * math.pi) / dim_t
        output = torch.stack(
            (angles[..., 0::2].sin(), angles[..., 1::2].cos()),
            dim=-1,
        ).flatten(-2)
        return (self.scale * output).to(dtype=dtype or coordinates.dtype)


class FixedSinusoidal1DPositionEmbedding(
    PositionEmbedding,
    position_type="fixed_sinusoidal",
):
    """Parameter-free Transformer sinusoidal features for normalized 1D positions.

    ``coordinates`` remain normalized to ``[0, 1]`` so query slots and memory
    tokens with different sequence lengths share one coordinate system.  The
    configurable ``max_position`` maps that coordinate to the conventional
    Transformer position scale before applying the temperature schedule.
    """

    requires_coordinates = True
    coordinate_dimensions = 1
    config_fields = frozenset({"scale", "temperature", "max_position"})

    def __init__(
        self,
        hidden_size: int,
        scale: float = 1.0,
        temperature: float = 10_000.0,
        max_position: float = 512.0,
        **kwargs,
    ):
        super().__init__()
        if hidden_size % 2 != 0:
            raise ValueError(
                "fixed sinusoidal positional embeddings require an even "
                "hidden_size"
            )
        self.hidden_size = int(hidden_size)
        self.scale = _finite_float("scale", scale)
        self.temperature = _positive_temperature(temperature)
        self.max_position = _finite_float("max_position", max_position)
        if self.max_position <= 0.0:
            raise ValueError(
                "position embedding max_position must be positive"
            )
        pair_count = self.hidden_size // 2
        dimensions = torch.arange(
            pair_count,
            dtype=torch.float32,
        )
        inverse_wavelengths = self.temperature ** (
            -dimensions / pair_count
        )
        self.register_buffer(
            "inverse_wavelengths",
            inverse_wavelengths,
            persistent=False,
        )

    def forward(self, coordinates=None, *, dtype=None, device=None, **kwargs):
        if coordinates is None or coordinates.shape[-1] != 1:
            raise ValueError(
                "fixed sinusoidal positions require (..., 1) coordinates"
            )
        coordinates = coordinates.to(
            device=device or coordinates.device,
            dtype=torch.float32,
        )
        inverse_wavelengths = self.inverse_wavelengths.to(
            device=coordinates.device,
            dtype=torch.float32,
        )
        angles = coordinates * self.max_position * inverse_wavelengths
        output = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        return (self.scale * output).to(dtype=dtype or coordinates.dtype)


class Fourier1DPositionEmbedding(PositionEmbedding, position_type="fourier"):
    """Fixed multi-frequency Fourier features for normalized 1D positions.

    Frequencies are deterministic and stored as a non-persistent buffer, so
    this strategy introduces no trainable parameters and no checkpoint state.
    Log spacing supplies both document-scale and token-scale positional bands.
    """

    requires_coordinates = True
    coordinate_dimensions = 1
    config_fields = frozenset(
        {"scale", "min_frequency", "max_frequency", "frequency_spacing"}
    )

    def __init__(
        self,
        hidden_size: int,
        scale: float = 1.0,
        min_frequency: float = 1.0,
        max_frequency: float = 64.0,
        frequency_spacing: str = "log",
        **kwargs,
    ):
        super().__init__()
        if hidden_size % 2 != 0:
            raise ValueError(
                "fourier positional embeddings require an even hidden_size"
            )
        self.hidden_size = int(hidden_size)
        self.scale = _finite_float("scale", scale)
        min_frequency = _finite_float("min_frequency", min_frequency)
        max_frequency = _finite_float("max_frequency", max_frequency)
        if min_frequency <= 0.0 or max_frequency <= 0.0:
            raise ValueError(
                "position embedding frequencies must be positive"
            )
        if max_frequency < min_frequency:
            raise ValueError(
                "position embedding max_frequency must be at least "
                "min_frequency"
            )
        frequency_spacing = str(frequency_spacing).lower()
        if frequency_spacing not in {"linear", "log"}:
            raise ValueError(
                "frequency_spacing must be 'linear' or 'log'"
            )
        pair_count = self.hidden_size // 2
        if frequency_spacing == "linear":
            frequencies = torch.linspace(
                min_frequency,
                max_frequency,
                pair_count,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.logspace(
                math.log10(min_frequency),
                math.log10(max_frequency),
                pair_count,
                dtype=torch.float32,
            )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, coordinates=None, *, dtype=None, device=None, **kwargs):
        if coordinates is None or coordinates.shape[-1] != 1:
            raise ValueError("fourier positions require (..., 1) coordinates")
        coordinates = coordinates.to(
            device=device or coordinates.device,
            dtype=torch.float32,
        )
        frequencies = self.frequencies.to(
            device=coordinates.device,
            dtype=torch.float32,
        )
        angles = coordinates * (2.0 * math.pi) * frequencies
        output = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        return (self.scale * output).to(dtype=dtype or coordinates.dtype)


class Linear1DPositionEmbedding(PositionEmbedding, position_type="linear1d"):
    """Learned linear projection of normalized temporal coordinates."""

    requires_coordinates = True
    coordinate_dimensions = 1
    config_fields = frozenset({"scale"})

    def __init__(self, hidden_size: int, scale: float = 1.0, **kwargs):
        super().__init__()
        self.projection = nn.Linear(1, hidden_size, bias=False)
        self.scale = _finite_float("scale", scale)

    def forward(self, coordinates=None, *, dtype=None, device=None, **kwargs):
        if coordinates is None or coordinates.shape[-1] != 1:
            raise ValueError("linear1d positions require (..., 1) coordinates")
        parameter = self.projection.weight
        coordinates = coordinates.to(
            device=device or parameter.device,
            dtype=parameter.dtype,
        )
        return (self.scale * self.projection(coordinates)).to(
            dtype=dtype or parameter.dtype
        )


class MLP1DPositionEmbedding(PositionEmbedding, position_type="mlp1d"):
    """Learned non-linear projection of normalized temporal coordinates."""

    requires_coordinates = True
    coordinate_dimensions = 1
    config_fields = frozenset({"scale"})

    def __init__(self, hidden_size: int, scale: float = 1.0, **kwargs):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.scale = _finite_float("scale", scale)

    def forward(self, coordinates=None, *, dtype=None, device=None, **kwargs):
        if coordinates is None or coordinates.shape[-1] != 1:
            raise ValueError("mlp1d positions require (..., 1) coordinates")
        parameter = next(self.projection.parameters())
        coordinates = coordinates.to(
            device=device or parameter.device,
            dtype=parameter.dtype,
        )
        return (self.scale * self.projection(coordinates)).to(
            dtype=dtype or parameter.dtype
        )


def masked_normalized_grid_1d(
    mask: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return padding-independent cell-centre coordinates for valid positions."""

    if mask.dim() != 2:
        raise ValueError(
            f"1D coordinate mask must have shape (B, L), got {tuple(mask.shape)}"
        )
    valid = mask.bool()
    counts = valid.sum(dim=1, keepdim=True).clamp_min(1)
    ranks = valid.long().cumsum(dim=1).to(torch.float32) - 0.5
    coordinates = (ranks / counts).unsqueeze(-1)
    coordinates = torch.where(
        valid.unsqueeze(-1),
        coordinates,
        torch.zeros_like(coordinates),
    )
    return coordinates.to(dtype=dtype)


class RefinementPositionEncoding(nn.Module):
    """Independent query/memory positional conditioning for anchor refinement.

    The object owns the two position encoders but not modality geometry. Callers
    may supply arbitrary normalized coordinates. For one-dimensional sequences,
    padding-independent valid-rank coordinates are generated automatically when
    coordinates are omitted; fixed query slots similarly use normalized index
    centres by default.
    """

    _MEMORY_USAGES = frozenset({"none", "keys_only", "keys_and_values"})

    @staticmethod
    def _split_position_config(
        config: str | Mapping[str, Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        if config is None:
            return "none", {}
        if isinstance(config, str):
            return config, {}
        if not isinstance(config, Mapping):
            raise TypeError("position component must be a string, mapping, or None")
        raw = dict(config)
        position_type = raw.pop("type", raw.pop("name", "none"))
        configured_params = raw.pop("params", {})
        if not isinstance(configured_params, Mapping):
            raise TypeError("position embedding params must be a mapping")
        params = dict(configured_params)
        params.update(raw)
        return str(position_type), params

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | None,
        hidden_size: int,
        *,
        num_query_embeddings: int | None = None,
    ) -> "RefinementPositionEncoding":
        """Build query/memory position components from one strict object."""

        raw = dict(config or {})
        allowed = {"memory", "query", "memory_usage"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(
                "Unsupported refinement position options: "
                f"{sorted(unknown)}. Available: {sorted(allowed)}"
            )
        memory_type, memory_kwargs = cls._split_position_config(
            raw.get("memory")
        )
        query_type, query_kwargs = cls._split_position_config(raw.get("query"))
        return cls(
            hidden_size,
            memory_type=memory_type,
            query_type=query_type,
            memory_kwargs=memory_kwargs,
            query_kwargs=query_kwargs,
            num_query_embeddings=num_query_embeddings,
            memory_usage=raw.get("memory_usage", "keys_only"),
        )

    def __init__(
        self,
        hidden_size: int,
        *,
        memory_type: str = "none",
        query_type: str = "none",
        memory_kwargs: dict | None = None,
        query_kwargs: dict | None = None,
        num_query_embeddings: int | None = None,
        memory_usage: str = "keys_only",
    ):
        super().__init__()
        memory_kwargs = dict(memory_kwargs or {})
        query_kwargs = dict(query_kwargs or {})
        memory_strategy = PositionEmbedding.strategy_class(memory_type)
        query_strategy = PositionEmbedding.strategy_class(query_type)
        if num_query_embeddings is not None:
            for strategy, strategy_kwargs in (
                (memory_strategy, memory_kwargs),
                (query_strategy, query_kwargs),
            ):
                if issubclass(strategy, FixedSinusoidal1DPositionEmbedding):
                    strategy_kwargs.setdefault(
                        "max_position",
                        float(num_query_embeddings),
                    )
        if memory_strategy.requires_num_embeddings:
            raise ValueError(
                "Refinement memory positions must support variable sequence lengths"
            )
        if query_strategy.requires_num_embeddings:
            if num_query_embeddings is None:
                raise ValueError(
                    "Learned refinement query positions require "
                    "num_query_embeddings"
                )
            query_kwargs.setdefault("num_embeddings", int(num_query_embeddings))
        if query_strategy.requires_grid_size:
            if num_query_embeddings is None:
                raise ValueError(
                    "Grid refinement query positions require "
                    "num_query_embeddings"
                )
            query_kwargs.setdefault(
                "grid_size",
                covering_grid_2d(int(num_query_embeddings)),
            )
        memory_usage = str(memory_usage).lower().replace("-", "_")
        if memory_usage not in self._MEMORY_USAGES:
            raise ValueError(
                "refinement position memory_usage must be 'none', "
                "'keys_only', or 'keys_and_values'"
            )
        self.hidden_size = int(hidden_size)
        self.memory_strategy = memory_strategy
        self.query_strategy = query_strategy
        self.memory_embedding = PositionEmbedding.from_config(
            memory_type,
            hidden_size,
            **memory_kwargs,
        )
        self.query_embedding = PositionEmbedding.from_config(
            query_type,
            hidden_size,
            **query_kwargs,
        )
        self.memory_usage = memory_usage

    @property
    def memory_position_in_values(self) -> bool:
        return self.memory_usage == "keys_and_values"

    @staticmethod
    def _mask(
        mask: torch.Tensor | None,
        shape: tuple[int, int],
        *,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        if mask is None:
            return torch.ones(shape, dtype=torch.bool, device=device)
        if mask.shape != shape:
            raise ValueError(
                f"{name} mask must have shape {shape}, got {tuple(mask.shape)}"
            )
        return mask.to(device=device).bool()

    @staticmethod
    def _coordinate_batch(
        coordinates: torch.Tensor,
        *,
        batch_size: int,
        count: int,
        dimensions: int | None,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        if coordinates.dim() == 2:
            coordinates = coordinates.unsqueeze(0)
        if coordinates.dim() != 3:
            raise ValueError(
                f"{name} coordinates must have shape (N, C), (1, N, C), "
                f"or (B, N, C), got {tuple(coordinates.shape)}"
            )
        if coordinates.shape[0] not in {1, batch_size} or coordinates.shape[1] != count:
            raise ValueError(
                f"{name} coordinates must have batch 1 or {batch_size} and "
                f"length {count}, got {tuple(coordinates.shape)}"
            )
        if dimensions is not None and coordinates.shape[-1] != dimensions:
            raise ValueError(
                f"{name} coordinates require width {dimensions}, "
                f"got {coordinates.shape[-1]}"
            )
        return coordinates.to(device=device, dtype=torch.float32)

    def _default_memory_coordinates(
        self,
        mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.memory_strategy.requires_coordinates:
            return None
        if self.memory_strategy.coordinate_dimensions != 1:
            raise ValueError(
                "Explicit memory coordinates are required for non-1D "
                "refinement position encoders"
            )
        return masked_normalized_grid_1d(mask)

    def _default_query_coordinates(
        self,
        anchors: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.query_strategy.requires_coordinates:
            return None
        if self.query_strategy.coordinate_dimensions != 1:
            raise ValueError(
                "Explicit query coordinates are required for non-1D "
                "refinement position encoders"
            )
        return normalized_grid_1d(
            anchors.shape[1],
            device=anchors.device,
            dtype=torch.float32,
        ).unsqueeze(0)

    @staticmethod
    def _normalize_output(
        positions: torch.Tensor | None,
        *,
        batch_size: int,
        count: int,
        hidden_size: int,
        mask: torch.Tensor,
        name: str,
    ) -> torch.Tensor | None:
        if positions is None:
            return None
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        if positions.shape[0] == 1 and batch_size != 1:
            positions = positions.expand(batch_size, -1, -1)
        expected = (batch_size, count, hidden_size)
        if positions.shape != expected:
            raise ValueError(
                f"{name} positions must have shape {expected}, "
                f"got {tuple(positions.shape)}"
            )
        return positions * mask.unsqueeze(-1).to(dtype=positions.dtype)

    def memory_positions(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
        *,
        coordinates: torch.Tensor | None = None,
        spatial_shape: tuple[int, int] | None = None,
    ) -> torch.Tensor | None:
        batch_size, count, hidden_size = memory.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Memory hidden size must be {self.hidden_size}, got {hidden_size}"
            )
        if (
            self.memory_usage == "none"
            or isinstance(self.memory_embedding, NoPositionEmbedding)
        ):
            return None
        valid = self._mask(
            memory_mask,
            (batch_size, count),
            device=memory.device,
            name="memory",
        )
        if coordinates is None:
            coordinates = self._default_memory_coordinates(valid)
        elif self.memory_strategy.requires_coordinates:
            coordinates = self._coordinate_batch(
                coordinates,
                batch_size=batch_size,
                count=count,
                dimensions=self.memory_strategy.coordinate_dimensions,
                device=memory.device,
                name="memory",
            )
        positions = self.memory_embedding(
            coordinates,
            count=count,
            spatial_shape=spatial_shape,
            dtype=memory.dtype,
            device=memory.device,
        )
        return self._normalize_output(
            positions,
            batch_size=batch_size,
            count=count,
            hidden_size=hidden_size,
            mask=valid,
            name="memory",
        )

    def query_positions(
        self,
        anchors: torch.Tensor,
        query_mask: torch.Tensor | None = None,
        *,
        coordinates: torch.Tensor | None = None,
        spatial_shape: tuple[int, int] | None = None,
    ) -> torch.Tensor | None:
        batch_size, count, hidden_size = anchors.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Anchor hidden size must be {self.hidden_size}, got {hidden_size}"
            )
        if isinstance(self.query_embedding, NoPositionEmbedding):
            return None
        valid = self._mask(
            query_mask,
            (batch_size, count),
            device=anchors.device,
            name="query",
        )
        if coordinates is None:
            coordinates = self._default_query_coordinates(anchors)
        elif self.query_strategy.requires_coordinates:
            coordinates = self._coordinate_batch(
                coordinates,
                batch_size=batch_size,
                count=count,
                dimensions=self.query_strategy.coordinate_dimensions,
                device=anchors.device,
                name="query",
            )
        positions = self.query_embedding(
            coordinates,
            count=count,
            spatial_shape=spatial_shape,
            dtype=anchors.dtype,
            device=anchors.device,
        )
        return self._normalize_output(
            positions,
            batch_size=batch_size,
            count=count,
            hidden_size=hidden_size,
            mask=valid,
            name="query",
        )

    def forward(
        self,
        anchors: torch.Tensor,
        memory: torch.Tensor,
        *,
        query_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        query_coordinates: torch.Tensor | None = None,
        memory_coordinates: torch.Tensor | None = None,
        query_spatial_shape: tuple[int, int] | None = None,
        memory_spatial_shape: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return (
            self.query_positions(
                anchors,
                query_mask,
                coordinates=query_coordinates,
                spatial_shape=query_spatial_shape,
            ),
            self.memory_positions(
                memory,
                memory_mask,
                coordinates=memory_coordinates,
                spatial_shape=memory_spatial_shape,
            ),
        )

    def position_bucket_attention_bias(
        self,
        anchors: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_mask: torch.Tensor | None = None,
        sigma: float = 0.5,
        weight: float = 1.0,
    ) -> torch.Tensor:
        """Compatibility helper for the former text-only position object."""

        from .attention_bias import GaussianDistanceAttentionBias

        query_coordinates = normalized_grid_1d(
            anchors.shape[1],
            device=anchors.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        valid = self._mask(
            memory_mask,
            memory.shape[:2],
            device=memory.device,
            name="memory",
        )
        return GaussianDistanceAttentionBias(
            num_heads=1,
            sigma=sigma,
            weight=weight,
            units="query_steps",
        )(
            query_coordinates=query_coordinates,
            key_coordinates=masked_normalized_grid_1d(valid),
            query_length=anchors.shape[1],
            key_length=memory.shape[1],
            dtype=anchors.dtype,
            device=anchors.device,
        )


class AnchorRefinementPositionEmbeddings(nn.Module):
    """Build configurable 1D query and memory positions for anchor decoding.

    Memory coordinates are assigned by valid-token rank, which makes them
    independent of right padding. Query coordinates use absolute slot centres;
    for position-bucket anchors this is exactly the centre of each text bucket.
    The returned tensors are supplied to every cross-attention decoder layer.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        memory_type: str = "none",
        query_type: str = "none",
        memory_kwargs: dict | None = None,
        query_kwargs: dict | None = None,
        num_query_embeddings: int | None = None,
    ):
        super().__init__()
        memory_kwargs = dict(memory_kwargs or {})
        query_kwargs = dict(query_kwargs or {})
        memory_strategy = PositionEmbedding.strategy_class(memory_type)
        query_strategy = PositionEmbedding.strategy_class(query_type)
        for role, strategy in (
            ("memory", memory_strategy),
            ("query", query_strategy),
        ):
            if strategy.coordinate_dimensions not in {None, 1}:
                raise ValueError(
                    f"Anchor-refinement {role} positions must be one-dimensional"
                )
            if strategy.requires_grid_size:
                raise ValueError(
                    f"Anchor-refinement {role} positions cannot use a 2D grid"
                )
        if memory_strategy.requires_num_embeddings:
            raise ValueError(
                "Anchor-refinement memory positions must support variable "
                "sequence lengths"
            )
        if query_strategy.requires_num_embeddings:
            if num_query_embeddings is None:
                raise ValueError(
                    "Fixed-capacity query positions require "
                    "num_query_embeddings"
                )
            query_kwargs.setdefault(
                "num_embeddings",
                int(num_query_embeddings),
            )
        self.hidden_size = int(hidden_size)
        self.memory_requires_coordinates = memory_strategy.requires_coordinates
        self.memory_embedding = PositionEmbedding.from_config(
            memory_type,
            hidden_size,
            **memory_kwargs,
        )
        self.query_embedding = PositionEmbedding.from_config(
            query_type,
            hidden_size,
            **query_kwargs,
        )

    @staticmethod
    def _validate_mask(mask, expected_shape, name, device):
        if mask is None:
            return torch.ones(expected_shape, dtype=torch.bool, device=device)
        if mask.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(mask.shape)}"
            )
        return mask.to(device=device).bool()

    def memory_positions(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Encode valid token ranks in a normalized document coordinate system."""

        batch_size, token_count, hidden_size = memory.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Memory hidden size must be {self.hidden_size}, got {hidden_size}"
            )
        if isinstance(self.memory_embedding, NoPositionEmbedding):
            return None
        if token_count == 0:
            return memory.new_empty(memory.shape)
        valid = self._validate_mask(
            memory_mask,
            (batch_size, token_count),
            "memory_mask",
            memory.device,
        )
        coordinates = None
        if self.memory_requires_coordinates:
            valid_counts = valid.sum(dim=1, keepdim=True).clamp_min(1)
            valid_ranks = valid.long().cumsum(dim=1).to(torch.float32) - 0.5
            coordinates = (valid_ranks / valid_counts).unsqueeze(-1)
            coordinates = torch.where(
                valid.unsqueeze(-1),
                coordinates,
                torch.zeros_like(coordinates),
            )
        positions = self.memory_embedding(
            coordinates,
            count=token_count,
            dtype=memory.dtype,
            device=memory.device,
        )
        if positions is None:
            return None
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        if positions.shape[0] == 1 and batch_size != 1:
            positions = positions.expand(batch_size, -1, -1)
        if positions.shape != memory.shape:
            raise ValueError(
                "Anchor-refinement memory positions must have shape "
                "(batch, tokens, hidden_size)"
            )
        return positions * valid.unsqueeze(-1).to(positions.dtype)

    def query_positions(self, anchors: torch.Tensor) -> torch.Tensor | None:
        """Encode normalized slot centres shared by every batch item."""

        anchor_count, hidden_size = anchors.shape[1:]
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Anchor hidden size must be {self.hidden_size}, got {hidden_size}"
            )
        if isinstance(self.query_embedding, NoPositionEmbedding):
            return None
        if anchor_count == 0:
            return anchors.new_empty(1, 0, hidden_size)
        coordinates = normalized_grid_1d(
            anchor_count,
            device=anchors.device,
            dtype=torch.float32,
        )
        positions = self.query_embedding(
            coordinates,
            count=anchor_count,
            dtype=anchors.dtype,
            device=anchors.device,
        )
        if positions is None:
            return None
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        if positions.shape[-2:] != (anchor_count, hidden_size) or positions.shape[0] not in {
            1,
            anchors.shape[0],
        }:
            raise ValueError(
                "Anchor-refinement query positions must have shape "
                "(1, anchors, hidden_size) or (batch, anchors, hidden_size)"
            )
        return positions

    def position_bucket_attention_bias(
        self,
        anchors: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_mask: torch.Tensor | None = None,
        sigma: float = 0.5,
        weight: float = 1.0,
    ) -> torch.Tensor:
        """Return a Gaussian locality prior for relative-position buckets.

        Distances are measured in bucket widths rather than raw normalized
        coordinates.  A sigma of 0.5 therefore has the same meaning for ten
        or twenty slots. Padding is excluded by assigning memory coordinates
        from valid-token rank; the attention layer applies the hard padding
        mask after combining it with this finite additive bias.
        """

        sigma = float(sigma)
        weight = float(weight)
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("position-bucket attention sigma must be positive")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                "position-bucket attention bias weight must be non-negative"
            )
        if anchors.dim() != 3 or memory.dim() != 3:
            raise ValueError("anchors and memory must both be three-dimensional")
        batch_size, anchor_count, _ = anchors.shape
        if memory.shape[0] != batch_size:
            raise ValueError("anchors and memory must have the same batch size")
        token_count = memory.shape[1]
        if anchor_count == 0 or token_count == 0:
            return anchors.new_zeros(batch_size, anchor_count, token_count)

        valid = self._validate_mask(
            memory_mask,
            (batch_size, token_count),
            "memory_mask",
            memory.device,
        )
        valid_counts = valid.sum(dim=1, keepdim=True).clamp_min(1)
        memory_coordinates = (
            valid.long().cumsum(dim=1).to(torch.float32) - 0.5
        ) / valid_counts
        memory_coordinates = torch.where(
            valid,
            memory_coordinates,
            torch.zeros_like(memory_coordinates),
        )
        query_coordinates = normalized_grid_1d(
            anchor_count,
            device=anchors.device,
            dtype=torch.float32,
        ).squeeze(-1)

        distances_in_buckets = (
            memory_coordinates.unsqueeze(1)
            - query_coordinates.view(1, anchor_count, 1)
        ) * anchor_count
        bias = -0.5 * weight * (distances_in_buckets / sigma).square()
        # Extremely small user-provided sigmas should produce a hard locality
        # prior, not overflow to -inf and make an attention row undefined.
        return bias.clamp_min(-10_000.0).to(dtype=anchors.dtype)

    def forward(
        self,
        anchors: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return (
            self.query_positions(anchors),
            self.memory_positions(memory, memory_mask),
        )


# Friendly aliases accepted by configuration files.
PositionEmbedding._registry["sine_2d"] = Sine2DPositionEmbedding
PositionEmbedding._registry["linear_2d"] = Linear2DPositionEmbedding
PositionEmbedding._registry["mlp_2d"] = MLP2DPositionEmbedding
PositionEmbedding._registry["learned_grid_2d"] = LearnedGrid2DPositionEmbedding
PositionEmbedding._registry["grid2d"] = LearnedGrid2DPositionEmbedding
PositionEmbedding._registry["sine_1d"] = Sine1DPositionEmbedding
PositionEmbedding._registry["linear_1d"] = Linear1DPositionEmbedding
PositionEmbedding._registry["mlp_1d"] = MLP1DPositionEmbedding
