"""Configurable positional embeddings for spatial task heads."""

from __future__ import annotations

import math

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
        position_type: str,
        hidden_size: int,
        **kwargs,
    ) -> PositionEmbedding:
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


# Friendly aliases accepted by configuration files.
PositionEmbedding._registry["sine_2d"] = Sine2DPositionEmbedding
PositionEmbedding._registry["linear_2d"] = Linear2DPositionEmbedding
PositionEmbedding._registry["mlp_2d"] = MLP2DPositionEmbedding
PositionEmbedding._registry["learned_grid_2d"] = LearnedGrid2DPositionEmbedding
PositionEmbedding._registry["grid2d"] = LearnedGrid2DPositionEmbedding
PositionEmbedding._registry["sine_1d"] = Sine1DPositionEmbedding
PositionEmbedding._registry["linear_1d"] = Linear1DPositionEmbedding
PositionEmbedding._registry["mlp_1d"] = MLP1DPositionEmbedding
