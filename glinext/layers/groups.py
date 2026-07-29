"""Group/anchor generation layers for structuring tasks."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .attention_bias import AttentionBias, NoAttentionBias
from .mlp import create_mlp
from .position import (
    RefinementPositionEncoding,
    masked_normalized_grid_1d,
    normalized_grid_1d,
)
from .rotary import RotaryEmbedding, apply_rotary_pos_emb


def _select_query_tokens(
    context_embedding: torch.Tensor,
    token_emb: torch.Tensor,
    count_val: torch.Tensor | None,
    threshold: float,
    token_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select query features while keeping padded features semantically absent.

    The mask-free branches intentionally mirror the historical behavior. When
    a feature mask is supplied, valid features are compacted (threshold mode)
    or exclusively ranked (count mode). A zero sentinel is retained when a row
    has no selectable feature so recurrent/attention kernels still receive a
    non-empty, finite sequence; its returned anchor mask remains false.
    """

    batch_size, token_count, hidden_size = token_emb.shape
    device = token_emb.device
    similarity = torch.einsum("bld,bd->bl", token_emb, context_embedding)

    if token_mask is None:
        if count_val is not None:
            max_k = min(int(count_val.max().item()), token_count)
            _, topk_indices = torch.topk(similarity, k=max_k, dim=1)
            batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(topk_indices)
            selected = token_emb[batch_idx, topk_indices]
            selected_mask = torch.arange(max_k, device=device).unsqueeze(0) < count_val.unsqueeze(1)
            return selected, selected_mask
        return token_emb, similarity > threshold

    if token_mask.shape != token_emb.shape[:2]:
        raise ValueError(
            "token_mask must match the first two token dimensions, "
            f"got {tuple(token_mask.shape)} for {tuple(token_emb.shape)}"
        )
    valid_tokens = token_mask.to(device=device).bool()
    if valid_tokens.all():
        return _select_query_tokens(
            context_embedding,
            token_emb,
            count_val,
            threshold,
            token_mask=None,
        )

    if count_val is not None:
        requested = count_val.to(device=device)
        if requested.dim() == 0:
            requested = requested.expand(batch_size)
        requested = requested.long().clamp(min=0)
        requested_width = min(int(requested.max().item()), token_count)
        internal_width = max(requested_width, 1)
        if token_count == 0:
            selected = token_emb.new_zeros(batch_size, 1, hidden_size)
            selected_mask = torch.zeros(
                batch_size,
                1,
                dtype=torch.bool,
                device=device,
            )
            return selected, selected_mask

        masked_similarity = similarity.masked_fill(~valid_tokens, float("-inf"))
        _, topk_indices = torch.topk(
            masked_similarity,
            k=min(internal_width, token_count),
            dim=1,
        )
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(topk_indices)
        selected_valid = valid_tokens[batch_idx, topk_indices]
        selected_valid = selected_valid & (
            torch.arange(topk_indices.shape[1], device=device).unsqueeze(0) < requested.unsqueeze(1)
        )
        selected = token_emb[batch_idx, topk_indices]
        selected = torch.where(
            selected_valid.unsqueeze(-1),
            selected,
            torch.zeros_like(selected),
        )
        return selected, selected_valid

    # Preserve source order while removing padded tokens from the recurrent or
    # transformer sequence. Thresholding remains the semantic anchor mask.
    valid_counts = valid_tokens.sum(dim=1)
    internal_width = max(int(valid_counts.max().item()), 1)
    if token_count == 0:
        selected = token_emb.new_zeros(batch_size, 1, hidden_size)
        selected_mask = torch.zeros(
            batch_size,
            1,
            dtype=torch.bool,
            device=device,
        )
        return selected, selected_mask

    positions = torch.arange(token_count, device=device).unsqueeze(0).expand(batch_size, -1)
    compact_indices = positions.masked_fill(~valid_tokens, token_count).sort(dim=1).values
    compact_indices = compact_indices[:, :internal_width]
    compact_valid = torch.arange(internal_width, device=device).unsqueeze(
        0
    ) < valid_counts.unsqueeze(1)
    safe_indices = compact_indices.clamp(max=token_count - 1)
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(safe_indices)
    selected = token_emb[batch_idx, safe_indices]
    selected = torch.where(
        compact_valid.unsqueeze(-1),
        selected,
        torch.zeros_like(selected),
    )
    selected_similarity = similarity[batch_idx, safe_indices]
    selected_mask = compact_valid & (selected_similarity > threshold)
    return selected, selected_mask


def _safe_cross_attention_memory(token_emb, token_mask, memory_pos_emb):
    """Add an internal zero sentinel for rows whose memory is fully masked."""

    if token_mask is None:
        return token_emb, token_mask, memory_pos_emb
    if token_mask.shape != token_emb.shape[:2]:
        raise ValueError(
            "token_mask must match the first two token dimensions, "
            f"got {tuple(token_mask.shape)} for {tuple(token_emb.shape)}"
        )

    valid_tokens = token_mask.to(device=token_emb.device).bool()
    needs_sentinel = ~valid_tokens.any(dim=1, keepdim=True)
    if not needs_sentinel.any():
        return token_emb, valid_tokens, memory_pos_emb

    sentinel = token_emb.new_zeros(token_emb.shape[0], 1, token_emb.shape[-1])
    safe_token_emb = torch.cat([sentinel, token_emb], dim=1)
    safe_token_mask = torch.cat([needs_sentinel, valid_tokens], dim=1)
    if memory_pos_emb is not None:
        position_sentinel = memory_pos_emb.new_zeros(
            *memory_pos_emb.shape[:-2],
            1,
            memory_pos_emb.shape[-1],
        )
        memory_pos_emb = torch.cat([position_sentinel, memory_pos_emb], dim=-2)
    return safe_token_emb, safe_token_mask, memory_pos_emb


@dataclass(frozen=True)
class AnchorCrossAttentionMemory:
    """Prepared encoder memory shared by anchor decoder layers.

    Instances are produced by :meth:`AnchorCrossAttentionLayer.prepare_memory`.
    ``source_length`` records the caller-provided memory width so an additive
    attention bias can be padded when preparation had to prepend a safe
    sentinel for an all-masked batch row.
    """

    token_emb: torch.Tensor
    token_mask: torch.Tensor | None
    memory_pos_emb: torch.Tensor | None
    source_length: int
    coordinates: torch.Tensor | None = None
    spatial_shape: tuple[int, int] | None = None


def _apply_layer_scale(value: torch.Tensor, scale: nn.Parameter | None):
    return value if scale is None else value * scale


class _RMSNorm(nn.Module):
    """Small local RMSNorm implementation with a stable state-dict layout."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        rms = value.float().square().mean(dim=-1, keepdim=True).add(
            self.eps
        ).rsqrt()
        return (value * rms.to(dtype=value.dtype)) * self.weight.to(
            dtype=value.dtype
        )


def _refinement_norm(norm_type: str, hidden_size: int) -> nn.Module:
    norm_type = str(norm_type).lower().replace("-", "_")
    if norm_type in {"layer_norm", "layernorm"}:
        return nn.LayerNorm(hidden_size)
    if norm_type in {"rms_norm", "rmsnorm"}:
        return _RMSNorm(hidden_size)
    raise ValueError(
        "anchor refinement norm type must be 'layer_norm' or 'rms_norm'"
    )


class _AnchorCrossAttentionBlock(nn.Module):
    """Refine anchors with optional positional information in memory values."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        norm_first: bool = False,
        norm_type: str = "layer_norm",
        ffn_multiplier: int = 4,
        activation: str = "gelu",
        layer_scale_init: float | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.norm_first = norm_first
        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm0 = _refinement_norm(norm_type, hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = _refinement_norm(norm_type, hidden_size)
        ffn_multiplier = int(ffn_multiplier)
        if ffn_multiplier <= 0:
            raise ValueError("anchor refinement ffn_multiplier must be positive")
        activation = str(activation).lower()
        activation_layer: nn.Module
        if activation == "gelu":
            activation_layer = nn.GELU()
        elif activation == "relu":
            activation_layer = nn.ReLU()
        elif activation in {"silu", "swish"}:
            activation_layer = nn.SiLU()
        else:
            raise ValueError(
                "anchor refinement activation must be 'gelu', 'relu', or 'silu'"
            )
        ffn_size = hidden_size * ffn_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            activation_layer,
            nn.Dropout(dropout),
            nn.Linear(ffn_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm2 = _refinement_norm(norm_type, hidden_size)
        if layer_scale_init is None:
            self.register_parameter("self_attn_scale", None)
            self.register_parameter("cross_attn_scale", None)
            self.register_parameter("ffn_scale", None)
        else:
            scale = float(layer_scale_init)
            if not math.isfinite(scale) or scale < 0.0:
                raise ValueError(
                    "anchor refinement layer_scale_init must be finite and "
                    "non-negative"
                )
            self.self_attn_scale = nn.Parameter(torch.full((hidden_size,), scale))
            self.cross_attn_scale = nn.Parameter(torch.full((hidden_size,), scale))
            self.ffn_scale = nn.Parameter(torch.full((hidden_size,), scale))

    def _self_attention(
        self,
        anchor_rep: torch.Tensor,
        query_pos_emb: torch.Tensor | None,
        self_attention_mask: torch.Tensor | None,
        self_attention_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        q = anchor_rep if query_pos_emb is None else anchor_rep + query_pos_emb
        attention_mask = None
        key_padding_mask = self_attention_mask
        if self_attention_bias is not None:
            batch_size, query_length = anchor_rep.shape[:2]
            attention_mask = self._expand_cross_attention_bias(
                self_attention_bias,
                batch_size=batch_size,
                query_length=query_length,
                memory_length=query_length,
                source_length=query_length,
                dtype=anchor_rep.dtype,
                device=anchor_rep.device,
            )
            if key_padding_mask is not None:
                padding_bias = torch.zeros(
                    batch_size,
                    1,
                    1,
                    query_length,
                    dtype=anchor_rep.dtype,
                    device=anchor_rep.device,
                )
                padding_bias.masked_fill_(
                    key_padding_mask[:, None, None, :],
                    float("-inf"),
                )
                attention_mask = attention_mask + padding_bias.expand(
                    batch_size,
                    self.num_heads,
                    query_length,
                    query_length,
                ).reshape_as(attention_mask)
                key_padding_mask = None
        return self.self_attn(
            q,
            q,
            anchor_rep,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
        )[0]

    def _expand_cross_attention_bias(
        self,
        cross_attention_bias: torch.Tensor,
        *,
        batch_size: int,
        query_length: int,
        memory_length: int,
        source_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Expand additive bias to the ``B * num_heads`` MHA mask layout."""

        bias = cross_attention_bias.to(device=device, dtype=dtype)
        if bias.dim() not in (2, 3, 4):
            raise ValueError(
                "cross_attention_bias must have shape (A, L), (B, A, L), "
                "(B * H, A, L), or (B, H, A, L)"
            )
        if bias.shape[-2] != query_length:
            raise ValueError(
                "cross_attention_bias query dimension must match anchors, "
                f"got {bias.shape[-2]} for {query_length}"
            )
        if bias.shape[-1] == source_length and memory_length == source_length + 1:
            bias = torch.nn.functional.pad(bias, (1, 0))
        elif bias.shape[-1] != memory_length:
            raise ValueError(
                "cross_attention_bias memory dimension must match prepared "
                f"memory, got {bias.shape[-1]} for {memory_length}"
            )

        if bias.dim() == 2:
            bias = bias[None, None].expand(
                batch_size,
                self.num_heads,
                query_length,
                memory_length,
            )
        elif bias.dim() == 3:
            leading = bias.shape[0]
            if leading in (1, batch_size):
                bias = bias[:, None].expand(
                    batch_size,
                    self.num_heads,
                    query_length,
                    memory_length,
                )
            elif leading == batch_size * self.num_heads:
                return bias
            else:
                raise ValueError(
                    "3D cross_attention_bias must have leading dimension 1, "
                    f"B, or B * num_heads; got {leading}"
                )
        elif bias.dim() == 4:
            if bias.shape[0] not in (1, batch_size):
                raise ValueError("4D cross_attention_bias batch dimension must be 1 or B")
            if bias.shape[1] not in (1, self.num_heads):
                raise ValueError("4D cross_attention_bias head dimension must be 1 or num_heads")
            bias = bias.expand(
                batch_size,
                self.num_heads,
                query_length,
                memory_length,
            )
        return bias.reshape(
            batch_size * self.num_heads,
            query_length,
            memory_length,
        )

    def _cross_attention_masks(
        self,
        anchor_rep: torch.Tensor,
        token_emb: torch.Tensor,
        token_mask: torch.Tensor | None,
        cross_attention_bias: torch.Tensor | None,
        source_length: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        key_padding_mask = ~token_mask.bool() if token_mask is not None else None
        if cross_attention_bias is None:
            return None, key_padding_mask

        batch_size, query_length = anchor_rep.shape[:2]
        memory_length = token_emb.shape[1]
        attention_mask = self._expand_cross_attention_bias(
            cross_attention_bias,
            batch_size=batch_size,
            query_length=query_length,
            memory_length=memory_length,
            source_length=source_length,
            dtype=anchor_rep.dtype,
            device=anchor_rep.device,
        )
        if key_padding_mask is not None:
            padding_bias = torch.zeros(
                batch_size,
                1,
                1,
                memory_length,
                dtype=anchor_rep.dtype,
                device=anchor_rep.device,
            )
            padding_bias.masked_fill_(
                key_padding_mask[:, None, None, :],
                float("-inf"),
            )
            padding_bias = padding_bias.expand(
                batch_size,
                self.num_heads,
                query_length,
                memory_length,
            ).reshape_as(attention_mask)
            attention_mask = attention_mask + padding_bias
        # PyTorch requires attn_mask and key_padding_mask to have the same
        # dtype. Padding is already represented additively above, so passing a
        # second boolean mask is both redundant and warning-prone.
        return attention_mask, None

    def _cross_attention(
        self,
        anchor_rep: torch.Tensor,
        token_emb: torch.Tensor,
        token_mask: torch.Tensor | None,
        query_pos_emb: torch.Tensor | None,
        memory_pos_emb: torch.Tensor | None,
        memory_position_in_values: bool,
        cross_attention_bias: torch.Tensor | None,
        source_length: int,
    ) -> torch.Tensor:
        q_cross = anchor_rep if query_pos_emb is None else anchor_rep + query_pos_emb
        memory_key = token_emb if memory_pos_emb is None else token_emb + memory_pos_emb
        memory_value = (
            memory_key if memory_position_in_values and memory_pos_emb is not None else token_emb
        )
        attention_mask, key_padding_mask = self._cross_attention_masks(
            anchor_rep,
            token_emb,
            token_mask,
            cross_attention_bias,
            source_length,
        )
        return self.cross_attn(
            q_cross,
            memory_key,
            memory_value,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
        )[0]

    def forward(
        self,
        anchor_rep,
        token_emb,
        token_mask=None,
        query_mask=None,
        query_pos_emb=None,
        memory_pos_emb=None,
        memory_position_in_values: bool = False,
        self_attention_bias: torch.Tensor | None = None,
        cross_attention_bias: torch.Tensor | None = None,
        source_length: int | None = None,
    ):
        """Run one refinement block.

        ``memory_pos_emb`` always augments cross-attention keys. When
        ``memory_position_in_values`` is true, it also augments values so the
        refined anchors retain explicit memory-location information.
        """
        # Add slot positional embedding to Q and K only (not V) so each slot
        # maintains a unique identity through self-attention (DETR convention).
        valid_queries = None
        self_attention_mask = None
        if query_mask is not None:
            valid_queries = query_mask.to(device=anchor_rep.device).bool()
            if valid_queries.shape != anchor_rep.shape[:2]:
                raise ValueError("query_mask must match the first two anchor dimensions")
            safe_queries = valid_queries.clone()
            all_invalid = ~safe_queries.any(dim=1)
            if all_invalid.any():
                safe_queries[all_invalid, 0] = True
            self_attention_mask = ~safe_queries
        source_length = token_emb.shape[1] if source_length is None else source_length
        if self.norm_first:
            normalized = self.norm0(anchor_rep)
            sa_out = self._self_attention(
                normalized,
                query_pos_emb,
                self_attention_mask,
                self_attention_bias,
            )
            x = anchor_rep + _apply_layer_scale(sa_out, self.self_attn_scale)
            normalized = self.norm1(x)
            attn_out = self._cross_attention(
                normalized,
                token_emb,
                token_mask,
                query_pos_emb,
                memory_pos_emb,
                memory_position_in_values,
                cross_attention_bias,
                source_length,
            )
            x = x + _apply_layer_scale(attn_out, self.cross_attn_scale)
            ffn_out = self.ffn(self.norm2(x))
            x = x + _apply_layer_scale(ffn_out, self.ffn_scale)
        else:
            sa_out = self._self_attention(
                anchor_rep,
                query_pos_emb,
                self_attention_mask,
                self_attention_bias,
            )
            anchor_rep = self.norm0(anchor_rep + _apply_layer_scale(sa_out, self.self_attn_scale))
            attn_out = self._cross_attention(
                anchor_rep,
                token_emb,
                token_mask,
                query_pos_emb,
                memory_pos_emb,
                memory_position_in_values,
                cross_attention_bias,
                source_length,
            )
            x = self.norm1(anchor_rep + _apply_layer_scale(attn_out, self.cross_attn_scale))
            x = self.norm2(x + _apply_layer_scale(self.ffn(x), self.ffn_scale))
        if valid_queries is not None:
            x = x.masked_fill(~valid_queries.unsqueeze(-1), 0.0)
        return x


class PreNormAnchorRefinementBlock(_AnchorCrossAttentionBlock):
    """Anchor decoder block whose residual branches receive normalized inputs."""

    def __init__(self, hidden_size, num_heads, dropout, **kwargs):
        kwargs.pop("norm_first", None)
        super().__init__(
            hidden_size,
            num_heads,
            dropout,
            norm_first=True,
            **kwargs,
        )


class PostNormAnchorRefinementBlock(_AnchorCrossAttentionBlock):
    """Anchor decoder block that normalizes after every residual update."""

    def __init__(self, hidden_size, num_heads, dropout, **kwargs):
        kwargs.pop("norm_first", None)
        super().__init__(
            hidden_size,
            num_heads,
            dropout,
            norm_first=False,
            **kwargs,
        )


class AnchorCrossAttentionLayer(nn.Module):
    """Reusable self/cross-attention refinement for generated anchors.

    Positional encoding and attention-bias modules are independent components.
    They may be installed as defaults on this module or supplied per call, which
    allows a shared refinement core to retain modality-specific conditioning.
    """

    config_fields = frozenset(
        {
            "num_heads",
            "num_layers",
            "dropout",
            "norm_style",
            "norm_type",
            "ffn_multiplier",
            "activation",
            "layer_scale_init",
        }
    )

    @classmethod
    def from_config(
        cls,
        config: str | Mapping[str, Any] | None,
        hidden_size: int,
        **defaults,
    ) -> "AnchorCrossAttentionLayer | None":
        """Build the independent refinement core from a strict component spec."""

        if config is None:
            refinement_type = "cross_attention"
            params: dict[str, Any] = {}
        elif isinstance(config, str):
            refinement_type = config
            params = {}
        elif isinstance(config, Mapping):
            raw = dict(config)
            refinement_type = raw.pop(
                "type", raw.pop("name", "cross_attention")
            )
            configured_params = raw.pop("params", {})
            if not isinstance(configured_params, Mapping):
                raise TypeError("anchor refinement params must be a mapping")
            params = dict(configured_params)
            params.update(raw)
        else:
            raise TypeError(
                "anchor refinement must be a string, mapping, or None"
            )
        refinement_type = str(refinement_type).lower().replace("-", "_")
        if refinement_type in {"none", "disabled"}:
            return None
        if refinement_type not in {"cross_attention", "decoder"}:
            raise ValueError(
                f"Unknown anchor refinement type {refinement_type!r}. "
                "Available: ['cross_attention', 'none']"
            )
        aliases = {
            "heads": "num_heads",
            "layers": "num_layers",
            "norm": "norm_style",
        }
        for old_name, new_name in aliases.items():
            if old_name in params:
                if new_name in params:
                    raise ValueError(
                        f"anchor refinement specifies both {old_name!r} and "
                        f"{new_name!r}"
                    )
                params[new_name] = params.pop(old_name)
        unknown = set(params) - set(cls.config_fields)
        if unknown:
            raise ValueError(
                "Unsupported anchor refinement options: "
                f"{sorted(unknown)}. Available: {sorted(cls.config_fields)}"
            )
        for name in cls.config_fields:
            if name not in params and name in defaults:
                params[name] = defaults[name]
        if int(params.get("num_layers", 1)) <= 0:
            return None
        return cls(hidden_size, **params)

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        norm_first: bool = False,
        norm_style: str | None = None,
        norm_type: str = "layer_norm",
        ffn_multiplier: int = 4,
        activation: str = "gelu",
        layer_scale_init: float | None = None,
        position_encoding: RefinementPositionEncoding | None = None,
        self_attention_bias: AttentionBias | None = None,
        cross_attention_bias: AttentionBias | None = None,
    ):
        super().__init__()
        if (
            isinstance(num_heads, bool)
            or int(num_heads) != num_heads
            or num_heads <= 0
        ):
            raise ValueError("anchor refinement num_heads must be positive")
        if (
            isinstance(num_layers, bool)
            or int(num_layers) != num_layers
            or num_layers < 0
        ):
            raise ValueError(
                "anchor refinement num_layers must be non-negative"
            )
        dropout = float(dropout)
        if not math.isfinite(dropout) or not 0.0 <= dropout <= 1.0:
            raise ValueError(
                "anchor refinement dropout must be finite and in [0, 1]"
            )
        if norm_style is None:
            norm_style = "pre_norm" if norm_first else "post_norm"
        norm_style = str(norm_style).lower().replace("-", "_")
        if norm_style not in {"pre_norm", "post_norm"}:
            raise ValueError(
                "anchor refinement norm_style must be 'pre_norm' or 'post_norm'"
            )
        block_type = (
            PreNormAnchorRefinementBlock
            if norm_style == "pre_norm"
            else PostNormAnchorRefinementBlock
        )
        self.norm_style = norm_style
        self.num_heads = int(num_heads)
        self.position_encoding = position_encoding
        self.self_attention_bias = self_attention_bias or NoAttentionBias(
            num_heads=self.num_heads
        )
        self.cross_attention_bias = cross_attention_bias or NoAttentionBias(
            num_heads=self.num_heads
        )
        self.layers = nn.ModuleList(
            [
                block_type(
                    hidden_size,
                    num_heads,
                    dropout,
                    norm_type=norm_type,
                    ffn_multiplier=ffn_multiplier,
                    activation=activation,
                    layer_scale_init=layer_scale_init,
                )
                for _ in range(num_layers)
            ]
        )

    @property
    def num_layers(self) -> int:
        """Number of independently callable decoder layers."""

        return len(self.layers)

    def prepare_memory(
        self,
        token_emb: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        memory_pos_emb: torch.Tensor | None = None,
        *,
        memory_coordinates: torch.Tensor | None = None,
        memory_spatial_shape: tuple[int, int] | None = None,
        position_encoding: RefinementPositionEncoding | None = None,
    ) -> AnchorCrossAttentionMemory:
        """Prepare encoder memory once for one or more decoder layers."""

        source_length = token_emb.shape[1]
        conditioning = position_encoding or self.position_encoding
        if memory_coordinates is None:
            coordinate_dimensions = (
                conditioning.memory_strategy.coordinate_dimensions
                if conditioning is not None
                else None
            )
            if coordinate_dimensions in {None, 1}:
                valid = (
                    token_mask.to(device=token_emb.device).bool()
                    if token_mask is not None
                    else torch.ones(
                        token_emb.shape[:2],
                        dtype=torch.bool,
                        device=token_emb.device,
                    )
                )
                memory_coordinates = masked_normalized_grid_1d(valid)
        if memory_pos_emb is None and conditioning is not None:
            memory_pos_emb = conditioning.memory_positions(
                token_emb,
                token_mask,
                coordinates=memory_coordinates,
                spatial_shape=memory_spatial_shape,
            )
        token_emb, token_mask, memory_pos_emb = _safe_cross_attention_memory(
            token_emb,
            token_mask,
            memory_pos_emb,
        )
        return AnchorCrossAttentionMemory(
            token_emb=token_emb,
            token_mask=token_mask,
            memory_pos_emb=memory_pos_emb,
            source_length=source_length,
            coordinates=memory_coordinates,
            spatial_shape=memory_spatial_shape,
        )

    @staticmethod
    def _query_coordinates(
        anchor_rep: torch.Tensor,
        coordinates: torch.Tensor | None,
        conditioning: RefinementPositionEncoding | None,
    ) -> torch.Tensor | None:
        if coordinates is not None:
            return coordinates
        coordinate_dimensions = (
            conditioning.query_strategy.coordinate_dimensions
            if conditioning is not None
            else None
        )
        if coordinate_dimensions in {None, 1}:
            return normalized_grid_1d(
                anchor_rep.shape[1],
                device=anchor_rep.device,
                dtype=torch.float32,
            ).unsqueeze(0)
        return None

    @staticmethod
    def _configured_bias(
        module: AttentionBias,
        *,
        query_coordinates: torch.Tensor | None,
        key_coordinates: torch.Tensor | None,
        query_length: int,
        key_length: int,
        anchor_rep: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor | None:
        return module(
            query_coordinates=query_coordinates,
            key_coordinates=key_coordinates,
            query_length=query_length,
            key_length=key_length,
            dtype=anchor_rep.dtype,
            device=anchor_rep.device,
            layer_index=layer_index,
        )

    def forward_layer(
        self,
        layer_index: int,
        anchor_rep: torch.Tensor,
        memory: AnchorCrossAttentionMemory,
        query_mask: torch.Tensor | None = None,
        query_pos_emb: torch.Tensor | None = None,
        memory_position_in_values: bool | None = None,
        self_attention_bias: torch.Tensor | None = None,
        cross_attention_bias: torch.Tensor | None = None,
        *,
        query_coordinates: torch.Tensor | None = None,
        query_spatial_shape: tuple[int, int] | None = None,
        position_encoding: RefinementPositionEncoding | None = None,
        self_attention_bias_module: AttentionBias | None = None,
        cross_attention_bias_module: AttentionBias | None = None,
    ) -> torch.Tensor:
        """Run one decoder layer against prepared memory.

        Calling layers individually lets task heads derive a fresh query
        position or attention bias from the previous layer's predictions.
        """

        if not isinstance(memory, AnchorCrossAttentionMemory):
            raise TypeError("memory must be produced by prepare_memory()")
        try:
            layer = self.layers[layer_index]
        except IndexError as error:
            raise IndexError(
                f"layer_index {layer_index} is out of range for {self.num_layers} decoder layers"
            ) from error
        conditioning = position_encoding or self.position_encoding
        query_coordinates = self._query_coordinates(
            anchor_rep,
            query_coordinates,
            conditioning,
        )
        if query_pos_emb is None and conditioning is not None:
            query_pos_emb = conditioning.query_positions(
                anchor_rep,
                query_mask,
                coordinates=query_coordinates,
                spatial_shape=query_spatial_shape,
            )
        if memory_position_in_values is None:
            memory_position_in_values = bool(
                conditioning is not None
                and conditioning.memory_position_in_values
            )
        if self_attention_bias is None:
            self_attention_bias = self._configured_bias(
                self_attention_bias_module or self.self_attention_bias,
                query_coordinates=query_coordinates,
                key_coordinates=query_coordinates,
                query_length=anchor_rep.shape[1],
                key_length=anchor_rep.shape[1],
                anchor_rep=anchor_rep,
                layer_index=layer_index,
            )
        if cross_attention_bias is None:
            cross_attention_bias = self._configured_bias(
                cross_attention_bias_module or self.cross_attention_bias,
                query_coordinates=query_coordinates,
                key_coordinates=memory.coordinates,
                query_length=anchor_rep.shape[1],
                key_length=memory.source_length,
                anchor_rep=anchor_rep,
                layer_index=layer_index,
            )
        return layer(
            anchor_rep=anchor_rep,
            token_emb=memory.token_emb,
            token_mask=memory.token_mask,
            query_mask=query_mask,
            query_pos_emb=query_pos_emb,
            memory_pos_emb=memory.memory_pos_emb,
            memory_position_in_values=memory_position_in_values,
            self_attention_bias=self_attention_bias,
            cross_attention_bias=cross_attention_bias,
            source_length=memory.source_length,
        )

    def forward(
        self,
        anchor_rep: torch.Tensor,
        token_emb: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
        query_pos_emb: torch.Tensor | None = None,
        memory_pos_emb: torch.Tensor | None = None,
        memory_position_in_values: bool | None = None,
        self_attention_bias: torch.Tensor | None = None,
        cross_attention_bias: torch.Tensor | None = None,
        *,
        query_coordinates: torch.Tensor | None = None,
        memory_coordinates: torch.Tensor | None = None,
        query_spatial_shape: tuple[int, int] | None = None,
        memory_spatial_shape: tuple[int, int] | None = None,
        position_encoding: RefinementPositionEncoding | None = None,
        self_attention_bias_module: AttentionBias | None = None,
        cross_attention_bias_module: AttentionBias | None = None,
    ) -> torch.Tensor:
        """
        Args:
            anchor_rep: (B, A, D) anchor embeddings to refine
            token_emb: (B, L, D) token embeddings from encoder
            token_mask: (B, L) optional mask for valid token positions
            query_mask: (B, A) optional mask for valid anchor/query slots
            query_pos_emb: (B, A, D) or (1, A, D) slot positional embeddings
                added to Q (and K) in self-attention and Q in cross-attention.
                Prevents anchor collapse when anchors start with similar values.
            memory_pos_emb: (B, L, D) or (1, L, D) spatial positions added to
                cross-attention keys.
            memory_position_in_values: Also add ``memory_pos_emb`` to
                cross-attention values. Defaults to false, leaving values as
                unmodified content for backward compatibility.
            cross_attention_bias: Optional additive spatial bias with shape
                ``(A, L)``, ``(B, A, L)``, ``(B * H, A, L)``, or
                ``(B, H, A, L)``.

        Returns:
            refined: (B, A, D) refined anchor embeddings
        """
        if anchor_rep.shape[1] == 0:
            return anchor_rep
        memory = self.prepare_memory(
            token_emb,
            token_mask,
            memory_pos_emb,
            memory_coordinates=memory_coordinates,
            memory_spatial_shape=memory_spatial_shape,
            position_encoding=position_encoding,
        )
        for layer_index in range(self.num_layers):
            anchor_rep = self.forward_layer(
                layer_index,
                anchor_rep,
                memory,
                query_mask=query_mask,
                query_pos_emb=query_pos_emb,
                memory_position_in_values=memory_position_in_values,
                self_attention_bias=self_attention_bias,
                cross_attention_bias=cross_attention_bias,
                query_coordinates=query_coordinates,
                query_spatial_shape=query_spatial_shape,
                position_encoding=position_encoding,
                self_attention_bias_module=self_attention_bias_module,
                cross_attention_bias_module=cross_attention_bias_module,
            )
        return anchor_rep


class RotaryGroupRNN(nn.Module):
    def __init__(self, hidden_size, max_count=20, rope_base=10_000.0):
        """
        Initializes the module with a learned positional embedding for count steps and a GRU,
        enhanced with rotary position embeddings.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_count = max_count

        self.pos_embedding = nn.Embedding(max_count, hidden_size)
        self.rotary_embeddings = RotaryEmbedding(hidden_size, base=rope_base)

        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)

        self.projector = create_mlp(
            input_dim=hidden_size * 2,
            intermediate_dims=[hidden_size * 4],
            output_dim=hidden_size,
            dropout=0.0,
            activation="relu",
            add_layer_norm=False,
        )

    def forward(self, field_emb: torch.Tensor, count_val: torch.Tensor) -> torch.Tensor:
        """
        Args:
            field_emb (Tensor): Field embeddings of shape (M, hidden_size).
            count_val (int): Predicted count value (number of steps).
        Returns:
            Tensor: Count-aware structure embeddings of shape (count_val, M, hidden_size).
        """
        M, D = field_emb.shape
        device = field_emb.device

        min_count = min(count_val, self.max_count)
        base_indices = torch.arange(min_count, device=device)
        base_pos = self.pos_embedding(base_indices)

        if count_val > self.max_count:
            num_repeats = (count_val + self.max_count - 1) // self.max_count
            pos_seq = base_pos.repeat(num_repeats, 1)[:count_val, :]
        else:
            pos_seq = base_pos

        pos_seq = pos_seq.unsqueeze(0)

        position_ids = torch.arange(count_val, device=device).unsqueeze(0)

        cos, sin = self.rotary_embeddings(pos_seq, position_ids)
        pos_seq = apply_rotary_pos_emb(pos_seq, cos, sin)

        pos_seq = pos_seq.squeeze(0).unsqueeze(1).expand(-1, M, -1)

        h0 = field_emb.unsqueeze(0)

        output, _ = self.gru(pos_seq, h0)

        field_broadcast = field_emb.unsqueeze(0).expand_as(output)
        return self.projector(torch.cat([output, field_broadcast], dim=-1))


class QueryGroupRNN(nn.Module):
    """Similarity-based token selection + GRU anchor generation.

    Accepts batched context_embedding (B, D) and token_emb (B, L, D).
    Selects tokens by similarity with context, processes through GRU
    conditioned on context as initial hidden state.
    """

    def __init__(self, hidden_size, max_count=20, rope_base=10_000.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_count = max_count

        self.rotary_embeddings = RotaryEmbedding(hidden_size, base=rope_base)

        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)

        self.projector = create_mlp(
            input_dim=hidden_size * 2,
            intermediate_dims=[hidden_size * 4],
            output_dim=hidden_size,
            dropout=0.0,
            activation="relu",
            add_layer_norm=False,
        )

    def forward(
        self,
        context_embedding: torch.Tensor,
        token_emb: torch.Tensor,
        count_val: torch.Tensor = None,
        threshold: float = 0.5,
        token_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            context_embedding (Tensor): Batched context of shape (B, D).
            token_emb (Tensor): Token embeddings of shape (B, L, D).
            count_val (Tensor): Per-sample counts of shape (B,), or None.
            threshold (float): Similarity threshold when count_val is None.
        Returns:
            Tuple[Tensor, Tensor]: (output of shape (B, k, D), mask of shape (B, k))
        """
        B, _, _ = token_emb.shape
        device = context_embedding.device

        selected_token_emb, mask = _select_query_tokens(
            context_embedding,
            token_emb,
            count_val,
            threshold,
            token_mask,
        )

        # Apply rotary position embeddings
        seq_len = selected_token_emb.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_embeddings(selected_token_emb, position_ids)
        selected_token_emb = apply_rotary_pos_emb(selected_token_emb, cos, sin)

        # GRU: context as initial hidden state — (1, B, D)
        h0 = context_embedding.unsqueeze(0).contiguous()
        gru_input = selected_token_emb.transpose(0, 1)  # (seq_len, B, D)

        output, _ = self.gru(gru_input, h0)  # (seq_len, B, D)
        output = output.transpose(0, 1)  # (B, seq_len, D)

        # Concat with context and project
        context_broadcast = context_embedding.unsqueeze(1).expand(B, seq_len, -1)  # (B, seq_len, D)
        output = self.projector(torch.cat([output, context_broadcast], dim=-1))  # (B, seq_len, D)

        return output, mask


class QueryGroupTransformer(nn.Module):
    """Similarity-based token selection + Transformer anchor generation.

    Accepts batched context_embedding (B, D) and token_emb (B, L, D).
    Selects tokens by similarity with context, processes through Transformer
    with context prepended as a prefix token.
    """

    def __init__(self, hidden_size, num_heads, num_layers, dropout=0.1, max_count=20):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_count = max_count

        self.rotary_embeddings = RotaryEmbedding(hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.projector = create_mlp(
            input_dim=hidden_size * 2,
            intermediate_dims=[hidden_size * 4],
            output_dim=hidden_size,
            dropout=0.0,
            activation="relu",
            add_layer_norm=False,
        )

    def forward(
        self,
        context_embedding: torch.Tensor,
        token_emb: torch.Tensor,
        count_val: torch.Tensor = None,
        threshold: float = 0.5,
        token_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            context_embedding (Tensor): Batched context of shape (B, D).
            token_emb (Tensor): Token embeddings of shape (B, L, D).
            count_val (Tensor): Per-sample counts of shape (B,), or None.
            threshold (float): Similarity threshold when count_val is None.
        Returns:
            Tuple[Tensor, Tensor]: (output of shape (B, k, D), mask of shape (B, k))
        """
        B, _, _ = token_emb.shape
        device = context_embedding.device

        selected_token_emb, mask = _select_query_tokens(
            context_embedding,
            token_emb,
            count_val,
            threshold,
            token_mask,
        )

        # Apply rotary position embeddings
        seq_len = selected_token_emb.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_embeddings(selected_token_emb, position_ids)
        selected_token_emb = apply_rotary_pos_emb(selected_token_emb, cos, sin)

        # Prepend context as prefix token: (B, 1 + seq_len, D)
        context_token = context_embedding.unsqueeze(1)  # (B, 1, D)
        combined = torch.cat([context_token, selected_token_emb], dim=1)  # (B, 1 + seq_len, D)

        # Extend mask to cover the prepended context token (always valid)
        context_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        full_mask = torch.cat([context_mask, mask], dim=1)  # (B, 1 + seq_len)

        # Transformer encoder (expects seq-first format)
        transformer_output = self.transformer_encoder(
            combined.transpose(0, 1), src_key_padding_mask=~full_mask
        ).transpose(0, 1)  # (B, 1 + seq_len, D)

        # Strip the context prefix, keep only selected token outputs
        transformer_output = transformer_output[:, 1:, :]  # (B, seq_len, D)

        # Concat with context and project
        context_broadcast = context_embedding.unsqueeze(1).expand(B, seq_len, -1)  # (B, seq_len, D)
        output = self.projector(
            torch.cat([transformer_output, context_broadcast], dim=-1)
        )  # (B, seq_len, D)

        return output, mask
