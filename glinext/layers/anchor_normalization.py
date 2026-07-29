"""Independent normalization strategies for generated anchor representations.

Anchor normalization intentionally sits between anchor generation and optional
attention refinement.  It is separate from the residual normalization used by
the refinement decoder blocks.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _validated_eps(value: float) -> float:
    value = float(value)
    if not torch.isfinite(torch.tensor(value)) or value <= 0.0:
        raise ValueError("anchor normalization eps must be finite and positive")
    return value


def _anchor_mask(
    anchors: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if anchors.dim() != 3:
        raise ValueError(
            "anchors must have shape (batch, anchors, hidden_size), "
            f"got {tuple(anchors.shape)}"
        )
    if mask is None:
        return torch.ones(
            anchors.shape[:2],
            dtype=torch.bool,
            device=anchors.device,
        )
    if mask.shape != anchors.shape[:2]:
        raise ValueError(
            "anchor mask must match the first two anchor dimensions, "
            f"got {tuple(mask.shape)} for {tuple(anchors.shape)}"
        )
    return mask.to(device=anchors.device).bool()


class AnchorNormalizer(nn.Module):
    """Base class and registry for post-generation anchor normalization."""

    _registry: dict[str, type["AnchorNormalizer"]] = {}
    config_fields: frozenset[str] = frozenset()

    def __init_subclass__(
        cls,
        normalization_type: str | None = None,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)
        if normalization_type:
            AnchorNormalizer._registry[normalization_type] = cls

    @classmethod
    def strategy_class(cls, normalization_type: str) -> type["AnchorNormalizer"]:
        normalized = str(normalization_type or "none").lower().replace("-", "_")
        strategy = cls._registry.get(normalized)
        if strategy is None:
            raise ValueError(
                f"Unknown anchor normalization {normalization_type!r}. "
                f"Available: {sorted(cls._registry)}"
            )
        return strategy

    @classmethod
    def from_config(
        cls,
        config: str | Mapping[str, Any] | None,
        hidden_size: int,
        **legacy_kwargs,
    ) -> "AnchorNormalizer":
        """Construct a strategy from a name or ``{type, params}`` mapping."""

        if config is None:
            normalization_type = "none"
            params: dict[str, Any] = {}
        elif isinstance(config, str):
            normalization_type = config
            params = dict(legacy_kwargs)
        elif isinstance(config, Mapping):
            raw = dict(config)
            normalization_type = raw.pop("type", raw.pop("name", "none"))
            configured_params = raw.pop("params", {})
            if not isinstance(configured_params, Mapping):
                raise TypeError("anchor normalization params must be a mapping")
            params = dict(configured_params)
            # Allow concise ``{type: l2, eps: ...}`` spelling as well.
            params.update(raw)
            if legacy_kwargs:
                overlap = set(params) & set(legacy_kwargs)
                if overlap:
                    raise ValueError(
                        "anchor normalization parameters were supplied twice: "
                        f"{sorted(overlap)}"
                    )
                params.update(legacy_kwargs)
        else:
            raise TypeError(
                "anchor normalization must be a string, mapping, or None"
            )

        strategy = cls.strategy_class(str(normalization_type))
        unknown = set(params) - set(strategy.config_fields)
        if unknown:
            raise ValueError(
                f"Unsupported {normalization_type!r} anchor normalization "
                f"options: {sorted(unknown)}. Available: "
                f"{sorted(strategy.config_fields)}"
            )
        return strategy(hidden_size=hidden_size, **params)

    def forward(
        self,
        anchors: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        source_embeddings: torch.Tensor | None = None,
        source_mask: torch.Tensor | None = None,
        support: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _zero_invalid(
        anchors: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        valid = _anchor_mask(anchors, mask)
        return anchors.masked_fill(~valid.unsqueeze(-1), 0.0)


class NoAnchorNormalization(AnchorNormalizer, normalization_type="none"):
    """Identity strategy, preserving historical unnormalized checkpoints."""

    def __init__(self, hidden_size: int, **kwargs):
        super().__init__()

    def forward(self, anchors, mask=None, **kwargs):
        _anchor_mask(anchors, mask)
        return anchors


class LayerNormAnchorNormalization(
    AnchorNormalizer,
    normalization_type="layer_norm",
):
    """Apply trainable LayerNorm independently to every anchor."""

    config_fields = frozenset({"eps", "elementwise_affine"})

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(
            hidden_size,
            eps=_validated_eps(eps),
            elementwise_affine=bool(elementwise_affine),
        )

    def forward(self, anchors, mask=None, **kwargs):
        return self._zero_invalid(self.norm(anchors), mask)


class RMSNormAnchorNormalization(
    AnchorNormalizer,
    normalization_type="rms_norm",
):
    """Apply trainable per-anchor RMS normalization."""

    config_fields = frozenset({"eps", "elementwise_affine"})

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.eps = _validated_eps(eps)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def forward(self, anchors, mask=None, **kwargs):
        rms = anchors.float().square().mean(dim=-1, keepdim=True).add(
            self.eps
        ).sqrt()
        output = anchors / rms.to(dtype=anchors.dtype)
        if self.weight is not None:
            output = output * self.weight.to(dtype=output.dtype)
        return self._zero_invalid(output, mask)


class L2AnchorNormalization(AnchorNormalizer, normalization_type="l2"):
    """Parameter-free unit-L2 normalization for every valid anchor."""

    config_fields = frozenset({"eps"})

    def __init__(self, hidden_size: int, eps: float = 1e-12, **kwargs):
        super().__init__()
        self.eps = _validated_eps(eps)

    def forward(self, anchors, mask=None, **kwargs):
        output = F.normalize(anchors.float(), p=2.0, dim=-1, eps=self.eps)
        return self._zero_invalid(output.to(dtype=anchors.dtype), mask)


class CenterRMSAnchorNormalization(
    AnchorNormalizer,
    normalization_type="center_rms",
):
    """Remove shared context and restore each anchor to a stable RMS scale.

    When source features are supplied, their masked mean is the shared context.
    This exactly supports position-bucket anchors whose buckets may contain
    slightly different numbers of source tokens. Otherwise a support-weighted
    masked anchor mean is used.
    """

    config_fields = frozenset({"eps"})

    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs):
        super().__init__()
        self.eps = _validated_eps(eps)

    @staticmethod
    def _source_mean(
        anchors: torch.Tensor,
        valid: torch.Tensor,
        source_embeddings: torch.Tensor | None,
        source_mask: torch.Tensor | None,
        support: torch.Tensor | None,
    ) -> torch.Tensor:
        if source_embeddings is not None:
            if source_embeddings.dim() != 3 or (
                source_embeddings.shape[0] != anchors.shape[0]
                or source_embeddings.shape[-1] != anchors.shape[-1]
            ):
                raise ValueError(
                    "source embeddings must have shape "
                    "(batch, source_length, hidden_size)"
                )
            if source_mask is None:
                source_valid = torch.ones(
                    source_embeddings.shape[:2],
                    dtype=torch.bool,
                    device=source_embeddings.device,
                )
            else:
                if source_mask.shape != source_embeddings.shape[:2]:
                    raise ValueError(
                        "source mask must match the first two source dimensions"
                    )
                source_valid = source_mask.to(
                    device=source_embeddings.device
                ).bool()
            weights = source_valid.to(dtype=source_embeddings.dtype)
            numerator = (
                source_embeddings * weights.unsqueeze(-1)
            ).sum(dim=1, keepdim=True)
            denominator = weights.sum(dim=1, keepdim=True).clamp_min(1)
            return numerator / denominator.unsqueeze(-1)

        if support is None:
            weights = valid.to(dtype=anchors.dtype)
        else:
            if support.shape != anchors.shape[:2]:
                raise ValueError(
                    "anchor support must match the first two anchor dimensions"
                )
            weights = support.to(device=anchors.device, dtype=anchors.dtype)
            weights = torch.where(valid, weights, torch.zeros_like(weights))
        numerator = (anchors * weights.unsqueeze(-1)).sum(dim=1, keepdim=True)
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1)
        return numerator / denominator.unsqueeze(-1)

    def forward(
        self,
        anchors,
        mask=None,
        *,
        source_embeddings=None,
        source_mask=None,
        support=None,
    ):
        valid = _anchor_mask(anchors, mask)
        shared = self._source_mean(
            anchors,
            valid,
            source_embeddings,
            source_mask,
            support,
        )
        centered = anchors - shared
        # Centering a single valid anchor would erase it. There is no collapse
        # to correct in that case, so retain its original representation.
        multiple = valid.sum(dim=1, keepdim=True) > 1
        residual = torch.where(multiple.unsqueeze(-1), centered, anchors)
        residual = residual.masked_fill(~valid.unsqueeze(-1), 0.0)
        rms = residual.float().square().mean(dim=-1, keepdim=True).add(
            self.eps
        ).sqrt()
        output = residual / rms.to(dtype=residual.dtype)
        return output.masked_fill(~valid.unsqueeze(-1), 0.0)


# Friendly spellings accepted in old and hand-written configurations.
AnchorNormalizer._registry["layernorm"] = LayerNormAnchorNormalization
AnchorNormalizer._registry["rmsnorm"] = RMSNormAnchorNormalization


__all__ = [
    "AnchorNormalizer",
    "NoAnchorNormalization",
    "LayerNormAnchorNormalization",
    "RMSNormAnchorNormalization",
    "L2AnchorNormalization",
    "CenterRMSAnchorNormalization",
]
