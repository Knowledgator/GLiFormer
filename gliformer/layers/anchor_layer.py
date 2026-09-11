"""Configurable anchor acquisition layers.

All extraction tasks follow: anchor + child -> spans.
The anchor layer is a configurable abstraction that produces anchor representations
from different sources depending on the task and configuration.

All layers accept context_embedding of shape (B, D) — a single context vector per sample.

Strategies:
- ParentAnchorLayer: returns context as single anchor (NER, Classification default)
- FeatureAnchorLayer: returns sequence feature embeddings as anchors
- PositionBucketAnchorLayer: mean-pools equal positional regions into anchors
- TopKNormAnchorLayer: selects tokens with the largest embedding magnitudes
- TopKDistinctAnchorLayer: greedily selects mutually distinct tokens
- TopKParentAnchorLayer: selects tokens most similar to the parent embedding
- TopKDensityDistinctAnchorLayer: selects dense-cluster representatives that stay distinct
- FixedAnchorLayer: learnable nn.Embedding table conditioned on context
- FixedRNNAnchorLayer: learnable slots conditioned on context via GRU
- FixedTransformerAnchorLayer: learnable slots conditioned on context via transformer
- RotaryAnchorLayer: wraps RotaryGroupRNN
- QueryRNNAnchorLayer: wraps QueryGroupRNN
- QueryTransformerAnchorLayer: wraps QueryGroupTransformer
"""

from collections.abc import Mapping
from typing import Any, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .anchor_normalization import AnchorNormalizer
from .mlp import create_mlp
from .groups import RotaryGroupRNN, QueryGroupRNN, QueryGroupTransformer


class AnchorLayer(nn.Module):
    """Base class for anchor acquisition layers.

    All subclasses produce:
        anchors: (B, A, D) — anchor representations
        mask: (B, A) — boolean mask for valid anchors

    Uses registry-based polymorphism via __init_subclass__.
    """

    _registry: dict = {}
    config_fields: frozenset[str] = frozenset()

    def __init_subclass__(cls, anchor_mode: str = "", **kwargs):
        super().__init_subclass__(**kwargs)
        if anchor_mode:
            cls._registry[anchor_mode] = cls

    @classmethod
    def strategy_class(cls, anchor_mode: str) -> type["AnchorLayer"]:
        normalized = "rotary" if anchor_mode == "rnn" else str(anchor_mode)
        strategy = cls._registry.get(normalized)
        if strategy is None:
            raise ValueError(
                f"Unknown anchor_mode: {anchor_mode}. "
                f"Available: {sorted(cls._registry)}"
            )
        return strategy

    @classmethod
    def from_config(
        cls,
        anchor_mode: str | Mapping[str, Any],
        hidden_size: int,
        **kwargs,
    ) -> "AnchorLayer":
        """Construct from a legacy mode or strict ``{type, params}`` mapping."""

        strict = isinstance(anchor_mode, Mapping)
        if strict:
            raw = dict(anchor_mode)
            anchor_mode = raw.pop("type", raw.pop("name", "parent"))
            configured_params = raw.pop("params", {})
            if not isinstance(configured_params, Mapping):
                raise TypeError("anchor layer params must be a mapping")
            params = dict(configured_params)
            params.update(raw)
        else:
            params = {}
        if anchor_mode == "rnn":
            anchor_mode = "rotary"
        strategy = cls.strategy_class(str(anchor_mode))
        if strict:
            unknown = set(params) - set(strategy.config_fields)
            if unknown:
                raise ValueError(
                    f"Unsupported {anchor_mode!r} anchor layer options: "
                    f"{sorted(unknown)}. Available: "
                    f"{sorted(strategy.config_fields)}"
                )
            for name in strategy.config_fields:
                if name not in params and name in kwargs:
                    params[name] = kwargs[name]
            return strategy(hidden_size, **params)
        return strategy(hidden_size, **kwargs)

    def forward(
        self,
        context_embedding: torch.Tensor,
        feature_embeddings: Optional[torch.Tensor] = None,
        count: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        feature_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            context_embedding: (B, D) context vector per sample
            feature_embeddings: (B, L, D) text, vision, or audio token features
            count: (B,) or scalar — number of anchors to generate per sample
            threshold: float — similarity threshold (for query-based layers)
            feature_mask: (B, L) valid feature mask

        Returns:
            anchors: (B, A, D) anchor representations
            mask: (B, A) boolean mask for valid anchors
        """
        raise NotImplementedError


def _count_mask(B: int, num_slots: int, count: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Build boolean mask from per-sample counts, or all-True if count is None."""
    if count is not None:
        if count.dim() == 0:
            count = count.unsqueeze(0).expand(B)
        return torch.arange(num_slots, device=device).unsqueeze(0) < count.unsqueeze(1)
    return torch.ones(B, num_slots, dtype=torch.bool, device=device)


class ParentAnchorLayer(AnchorLayer, anchor_mode="parent"):
    """Context embedding as single anchor — simplest case.

    Used by NER and Classification where the anchor is the parent prompt embedding.
    Returns a single anchor per sample: context_embedding unsqueezed to (B, 1, D).
    """

    def __init__(self, hidden_size: int, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(
        self,
        context_embedding,
        feature_embeddings=None,
        count=None,
        threshold=0.5,
        feature_mask=None,
    ):
        B = context_embedding.shape[0]
        anchor = context_embedding.unsqueeze(1)  # (B, 1, D)
        mask = torch.ones(B, 1, dtype=torch.bool, device=context_embedding.device)
        return anchor, mask


class FeatureAnchorLayer(AnchorLayer, anchor_mode="features"):
    """Use input sequence features directly as anchor slots.

    This is useful for vision/audio tasks where each encoded patch or frame can
    serve as an anchor candidate. If ``feature_anchor_mlp`` is enabled in the
    task config, the features are passed through a small MLP before being used as
    anchors; otherwise the returned anchors are exactly the input features.
    """

    config_fields = frozenset(
        {"feature_mlp", "feature_mlp_hidden_multiplier", "dropout"}
    )

    def __init__(
        self,
        hidden_size: int,
        feature_mlp: bool = False,
        feature_mlp_hidden_multiplier: int = 1,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.feature_mlp = bool(feature_mlp)
        if self.feature_mlp:
            hidden_dim = hidden_size * max(int(feature_mlp_hidden_multiplier), 1)
            self.projector = create_mlp(
                input_dim=hidden_size,
                intermediate_dims=[hidden_dim],
                output_dim=hidden_size,
                dropout=dropout,
                activation="gelu",
                add_layer_norm=True,
            )

    def forward(
        self,
        context_embedding,
        feature_embeddings=None,
        count=None,
        threshold=0.5,
        feature_mask=None,
    ):
        if feature_embeddings is None:
            B = context_embedding.shape[0]
            return (
                torch.zeros(B, 0, context_embedding.shape[-1], device=context_embedding.device),
                torch.zeros(B, 0, dtype=torch.bool, device=context_embedding.device),
            )

        anchors = self.projector(feature_embeddings) if self.feature_mlp else feature_embeddings
        if feature_mask is None:
            mask = torch.ones(anchors.shape[:2], dtype=torch.bool, device=anchors.device)
        else:
            mask = feature_mask.to(device=anchors.device).bool()
        return anchors, mask


# Backward-friendly singular alias.
AnchorLayer._registry["feature"] = FeatureAnchorLayer


class _TokenSelectionAnchorLayer(AnchorLayer):
    """Parameter-free fixed-width anchor selection from sequence features."""

    config_fields = frozenset({"num_slots"})

    def __init__(self, hidden_size: int, num_slots: int = 10, **kwargs):
        super().__init__()
        if isinstance(num_slots, bool) or int(num_slots) != num_slots or num_slots <= 0:
            raise ValueError("num_slots must be a positive integer")
        self.hidden_size = hidden_size
        self.num_slots = int(num_slots)

    def _empty(self, context_embedding: torch.Tensor):
        batch_size = context_embedding.shape[0]
        return (
            context_embedding.new_zeros(batch_size, self.num_slots, self.hidden_size),
            torch.zeros(
                batch_size,
                self.num_slots,
                dtype=torch.bool,
                device=context_embedding.device,
            ),
        )

    def _valid_features(
        self,
        context_embedding: torch.Tensor,
        feature_embeddings: Optional[torch.Tensor],
        feature_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if feature_embeddings is None:
            return None
        if feature_embeddings.dim() != 3:
            raise ValueError(
                "feature_embeddings must have shape (B, L, D), "
                f"got {tuple(feature_embeddings.shape)}"
            )
        if feature_embeddings.shape[0] != context_embedding.shape[0]:
            raise ValueError(
                "context_embedding and feature_embeddings must have the same "
                "batch size"
            )
        if feature_embeddings.shape[-1] != self.hidden_size:
            raise ValueError(
                f"feature embedding size must be {self.hidden_size}, "
                f"got {feature_embeddings.shape[-1]}"
            )
        if feature_mask is None:
            return torch.ones(
                feature_embeddings.shape[:2],
                dtype=torch.bool,
                device=feature_embeddings.device,
            )
        if feature_mask.shape != feature_embeddings.shape[:2]:
            raise ValueError(
                "feature_mask must match the first two feature dimensions, "
                f"got {tuple(feature_mask.shape)} for "
                f"{tuple(feature_embeddings.shape)}"
            )
        return feature_mask.to(device=feature_embeddings.device).bool()

    def _finalize_selection(
        self,
        feature_embeddings: torch.Tensor,
        indices: torch.Tensor,
        selected_mask: torch.Tensor,
        count: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather selected tokens, pad to ``num_slots``, and apply masks."""

        batch_size, _, hidden_size = feature_embeddings.shape
        width = indices.shape[1]
        if width:
            anchors = feature_embeddings.gather(
                1, indices.unsqueeze(-1).expand(-1, -1, hidden_size)
            )
            anchors = torch.where(
                selected_mask.unsqueeze(-1), anchors, torch.zeros_like(anchors)
            )
        else:
            anchors = feature_embeddings.new_zeros(batch_size, 0, hidden_size)

        if width < self.num_slots:
            anchors = torch.cat(
                [
                    anchors,
                    feature_embeddings.new_zeros(
                        batch_size, self.num_slots - width, hidden_size
                    ),
                ],
                dim=1,
            )
            selected_mask = torch.cat(
                [
                    selected_mask,
                    torch.zeros(
                        batch_size,
                        self.num_slots - width,
                        dtype=torch.bool,
                        device=feature_embeddings.device,
                    ),
                ],
                dim=1,
            )

        if count is not None:
            if not torch.is_tensor(count):
                count = torch.as_tensor(count, device=feature_embeddings.device)
            else:
                count = count.to(device=feature_embeddings.device)
            selected_mask = selected_mask & _count_mask(
                batch_size, self.num_slots, count, feature_embeddings.device
            )
        anchors = torch.where(
            selected_mask.unsqueeze(-1), anchors, torch.zeros_like(anchors)
        )
        return anchors, selected_mask

    def _ranked_selection(
        self,
        feature_embeddings: torch.Tensor,
        valid: torch.Tensor,
        scores: torch.Tensor,
        count: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select the highest-scoring valid tokens without replacement."""

        width = min(self.num_slots, feature_embeddings.shape[1])
        if width == 0:
            indices = torch.empty(
                feature_embeddings.shape[0],
                0,
                dtype=torch.long,
                device=feature_embeddings.device,
            )
            selected_mask = valid[:, :0]
        else:
            scores = scores.masked_fill(~valid, -torch.inf)
            indices = scores.topk(width, dim=1).indices
            selected_mask = valid.gather(1, indices)
        return self._finalize_selection(
            feature_embeddings, indices, selected_mask, count
        )

    def _greedy_distinct_indices(
        self,
        normalized_features: torch.Tensor,
        valid: torch.Tensor,
        first_scores: torch.Tensor,
        quality: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedy farthest-point selection, optionally weighted by quality."""

        batch_size, sequence_length, _ = normalized_features.shape
        width = min(self.num_slots, sequence_length)
        if width == 0:
            return (
                torch.empty(
                    batch_size,
                    0,
                    dtype=torch.long,
                    device=normalized_features.device,
                ),
                valid[:, :0],
            )

        available = valid.clone()
        max_similarity = normalized_features.new_full(
            (batch_size, sequence_length), -1.0
        )
        selected_indices = []
        selection_masks = []

        for step in range(width):
            if step == 0:
                scores = first_scores
            elif quality is None:
                scores = -max_similarity
            else:
                novelty = ((1.0 - max_similarity) * 0.5).clamp(0.0, 1.0)
                scores = quality * novelty

            scores = scores.masked_fill(~available, -torch.inf)
            indices = scores.argmax(dim=1)
            is_selected = available.gather(1, indices.unsqueeze(1)).squeeze(1)
            selected_indices.append(indices)
            selection_masks.append(is_selected)

            selected = normalized_features.gather(
                1,
                indices.view(batch_size, 1, 1).expand(
                    -1, 1, normalized_features.shape[-1]
                ),
            ).squeeze(1)
            similarity = torch.einsum(
                "bld,bd->bl", normalized_features, selected
            ).clamp(-1.0, 1.0)
            max_similarity = torch.where(
                is_selected.unsqueeze(1),
                torch.maximum(max_similarity, similarity),
                max_similarity,
            )
            available.scatter_(1, indices.unsqueeze(1), False)

        return torch.stack(selected_indices, dim=1), torch.stack(
            selection_masks, dim=1
        )


class PositionBucketAnchorLayer(
    _TokenSelectionAnchorLayer, anchor_mode="position_buckets"
):
    """Mean-pool valid tokens in equal relative-position buckets.

    Encoder features can contain a large document-wide common component.  In
    that case raw bucket means are almost collinear even when the local token
    content differs, which makes fixed slots interchangeable under Hungarian
    matching.  ``center_rms`` removes the valid-token document mean and
    restores every non-empty bucket residual to a stable RMS scale.
    """

    _NORMALIZATIONS = frozenset({"none", "center_rms"})
    _NORMALIZATION_EPS = 1e-6
    config_fields = frozenset({"num_slots", "position_bucket_normalization"})

    def __init__(
        self,
        hidden_size: int,
        num_slots: int = 10,
        position_bucket_normalization: str = "none",
        **kwargs,
    ):
        super().__init__(hidden_size, num_slots=num_slots, **kwargs)
        normalization = str(position_bucket_normalization).lower().replace(
            "-", "_"
        )
        if normalization not in self._NORMALIZATIONS:
            raise ValueError(
                "position_bucket_normalization must be 'none' or "
                "'center_rms'"
            )
        self.normalization = normalization
        normalizer_options = (
            {"eps": self._NORMALIZATION_EPS}
            if normalization == "center_rms"
            else {}
        )
        self.normalizer = AnchorNormalizer.from_config(
            normalization, hidden_size, **normalizer_options
        )

    def _normalize_bucket_anchors(
        self,
        anchors: torch.Tensor,
        bucket_sums: torch.Tensor,
        bucket_counts: torch.Tensor,
        anchor_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.normalizer(
            anchors,
            anchor_mask,
            support=bucket_counts,
        )

    def forward(
        self,
        context_embedding,
        feature_embeddings=None,
        count=None,
        threshold=0.5,
        feature_mask=None,
    ):
        valid = self._valid_features(
            context_embedding, feature_embeddings, feature_mask
        )
        if valid is None:
            return self._empty(context_embedding)

        batch_size, sequence_length, hidden_size = feature_embeddings.shape
        if sequence_length == 0:
            return self._empty(context_embedding)

        valid_counts = valid.sum(dim=1)
        valid_ranks = valid.long().cumsum(dim=1) - 1
        bucket_indices = (
            valid_ranks.clamp_min(0)
            * self.num_slots
            // valid_counts.clamp_min(1).unsqueeze(1)
        ).clamp_max(self.num_slots - 1)

        bucket_sums = feature_embeddings.new_zeros(
            batch_size, self.num_slots, hidden_size
        )
        source = torch.where(
            valid.unsqueeze(-1),
            feature_embeddings,
            torch.zeros_like(feature_embeddings),
        )
        bucket_sums.scatter_add_(
            1, bucket_indices.unsqueeze(-1).expand(-1, -1, hidden_size), source
        )

        bucket_counts = feature_embeddings.new_zeros(batch_size, self.num_slots)
        bucket_counts.scatter_add_(
            1, bucket_indices, valid.to(feature_embeddings.dtype)
        )
        anchor_mask = bucket_counts > 0
        anchors = bucket_sums / bucket_counts.clamp_min(1).unsqueeze(-1)
        anchors = self._normalize_bucket_anchors(
            anchors,
            bucket_sums,
            bucket_counts,
            anchor_mask,
        )

        if count is not None:
            if not torch.is_tensor(count):
                count = torch.as_tensor(count, device=feature_embeddings.device)
            else:
                count = count.to(device=feature_embeddings.device)
            anchor_mask = anchor_mask & _count_mask(
                batch_size, self.num_slots, count, feature_embeddings.device
            )
        anchors = torch.where(
            anchor_mask.unsqueeze(-1), anchors, torch.zeros_like(anchors)
        )
        return anchors, anchor_mask


class TopKNormAnchorLayer(_TokenSelectionAnchorLayer, anchor_mode="topk_norm"):
    """Select tokens with the largest squared L2 embedding magnitude."""

    def forward(
        self,
        context_embedding,
        feature_embeddings=None,
        count=None,
        threshold=0.5,
        feature_mask=None,
    ):
        valid = self._valid_features(
            context_embedding, feature_embeddings, feature_mask
        )
        if valid is None:
            return self._empty(context_embedding)
        scores = feature_embeddings.detach().float().square().sum(dim=-1)
        return self._ranked_selection(feature_embeddings, valid, scores, count)


class TopKDistinctAnchorLayer(
    _TokenSelectionAnchorLayer, anchor_mode="topk_distinct"
):
    """Select a globally unusual seed, then cosine-distant token anchors."""

    def forward(
        self,
        context_embedding,
        feature_embeddings=None,
        count=None,
        threshold=0.5,
        feature_mask=None,
    ):
        valid = self._valid_features(
            context_embedding, feature_embeddings, feature_mask
        )
        if valid is None:
            return self._empty(context_embedding)
        if feature_embeddings.shape[1] == 0:
            return self._empty(context_embedding)

        normalized = F.normalize(feature_embeddings.detach().float(), dim=-1)
        normalized = torch.where(
            valid.unsqueeze(-1), normalized, torch.zeros_like(normalized)
        )
        centroid = normalized.sum(dim=1) / valid.sum(dim=1).clamp_min(1).unsqueeze(1)
        centroid = F.normalize(centroid, dim=-1)
        first_scores = -torch.einsum("bld,bd->bl", normalized, centroid)
        indices, selected_mask = self._greedy_distinct_indices(
            normalized, valid, first_scores
        )
        return self._finalize_selection(
            feature_embeddings, indices, selected_mask, count
        )


class TopKParentAnchorLayer(_TokenSelectionAnchorLayer, anchor_mode="topk_parent"):
    """Select tokens with the highest cosine similarity to the parent token."""

    def forward(
        self,
        context_embedding,
        feature_embeddings=None,
        count=None,
        threshold=0.5,
        feature_mask=None,
    ):
        valid = self._valid_features(
            context_embedding, feature_embeddings, feature_mask
        )
        if valid is None:
            return self._empty(context_embedding)

        features = F.normalize(feature_embeddings.detach().float(), dim=-1)
        parent_features = context_embedding.detach().to(
            device=feature_embeddings.device, dtype=torch.float32
        )
        parent = F.normalize(parent_features, dim=-1)
        scores = torch.einsum("bld,bd->bl", features, parent)
        return self._ranked_selection(feature_embeddings, valid, scores, count)


class TopKDensityDistinctAnchorLayer(
    _TokenSelectionAnchorLayer, anchor_mode="topk_density_distinct"
):
    """Select high average-cosine representatives that remain distinct."""

    def forward(
        self,
        context_embedding,
        feature_embeddings=None,
        count=None,
        threshold=0.5,
        feature_mask=None,
    ):
        valid = self._valid_features(
            context_embedding, feature_embeddings, feature_mask
        )
        if valid is None:
            return self._empty(context_embedding)
        if feature_embeddings.shape[1] == 0:
            return self._empty(context_embedding)

        normalized = F.normalize(feature_embeddings.detach().float(), dim=-1)
        normalized = torch.where(
            valid.unsqueeze(-1), normalized, torch.zeros_like(normalized)
        )
        # For unit vectors, x_i dot sum_j(x_j) is the sum of all cosine
        # similarities for token i. Subtracting x_i removes self-similarity,
        # giving an exact average signed-cosine density in O(B * L * D), without
        # materializing the O(B * L^2) pairwise matrix.
        normalized_sum = normalized.sum(dim=1, keepdim=True)
        neighbor_count = (valid.sum(dim=1) - 1).clamp_min(1).unsqueeze(1)
        density = (normalized * (normalized_sum - normalized)).sum(
            dim=-1
        ) / neighbor_count
        density = density.clamp_min(0.0)
        density = torch.where(valid, density, torch.zeros_like(density))
        max_density = density.amax(dim=1, keepdim=True)
        density_epsilon = torch.finfo(density.dtype).eps
        quality = torch.where(
            max_density > density_epsilon,
            density / max_density.clamp_min(density_epsilon),
            valid.to(density.dtype),
        )

        indices, selected_mask = self._greedy_distinct_indices(
            normalized, valid, quality, quality=quality
        )
        return self._finalize_selection(
            feature_embeddings, indices, selected_mask, count
        )


class FixedAnchorLayer(AnchorLayer, anchor_mode="fixed"):
    """Learnable fixed-size embedding table conditioned on context.

    Each anchor slot is a learnable embedding shifted by a gated projected
    context vector.  The configurable gate lets set-prediction heads disable
    that shared context component without changing the fixed-slot state-dict
    layout used by existing checkpoints.
    """

    config_fields = frozenset(
        {"num_slots", "context_gate_init", "context_gate_trainable"}
    )

    def __init__(
        self,
        hidden_size: int,
        num_slots: int = 10,
        context_gate_init: float = 0.1,
        context_gate_trainable: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.anchor_table = nn.Embedding(num_slots, hidden_size)
        nn.init.orthogonal_(self.anchor_table.weight)
        self.context_proj = nn.Linear(hidden_size, hidden_size)
        # The slots start as orthogonal directions, but the projected context is a
        # single vector added to every slot. If it dominates (its learned norm is
        # typically several times the unit-norm table rows), it rotates all slots
        # toward one direction → they become near-collinear → downstream
        # self-attention averages them into a single identical query, so every
        # anchor predicts the same class/box. Gate the context low so the distinct
        # per-slot identity survives; training can grow the gate if it helps.
        self.context_gate = nn.Parameter(
            torch.tensor(float(context_gate_init)),
            requires_grad=bool(context_gate_trainable),
        )

    def forward(
        self,
        context_embedding,
        feature_embeddings=None,
        count=None,
        threshold=0.5,
        feature_mask=None,
    ):
        B = context_embedding.shape[0]
        device = context_embedding.device

        anchors = self.anchor_table.weight.unsqueeze(0).expand(B, -1, -1)  # (B, num_slots, D)
        context = self.context_proj(context_embedding).unsqueeze(1)  # (B, 1, D)
        anchors = anchors + self.context_gate * context

        mask = _count_mask(B, self.num_slots, count, device)
        return anchors, mask


class FixedRNNAnchorLayer(AnchorLayer, anchor_mode="fixed_rnn"):
    """Learnable fixed slots conditioned on context via GRU.

    Fixed slot embeddings are fed as input sequence to a GRU whose initial hidden
    state is the context embedding. Output is concatenated with context and projected.
    """

    config_fields = frozenset({"num_slots"})

    def __init__(self, hidden_size: int, num_slots: int = 10, **kwargs):
        super().__init__()
        self.num_slots = num_slots
        self.anchor_table = nn.Embedding(num_slots, hidden_size)
        nn.init.orthogonal_(self.anchor_table.weight)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.projector = create_mlp(
            input_dim=hidden_size * 2,
            intermediate_dims=[hidden_size * 4],
            output_dim=hidden_size,
            dropout=0.,
            activation="relu",
            add_layer_norm=False,
        )

    def forward(
        self,
        context_embedding,
        feature_embeddings=None,
        count=None,
        threshold=0.5,
        feature_mask=None,
    ):
        B = context_embedding.shape[0]
        device = context_embedding.device

        # Fixed slots as GRU input: (B, num_slots, D)
        slots = self.anchor_table.weight.unsqueeze(0).expand(B, -1, -1)
        # Context as initial hidden state: (1, B, D)
        h0 = context_embedding.unsqueeze(0)

        output, _ = self.gru(slots, h0)  # (B, num_slots, D)

        # Concat with context and project
        context_broadcast = context_embedding.unsqueeze(1).expand_as(output)  # (B, num_slots, D)
        anchors = self.projector(torch.cat([output, context_broadcast], dim=-1))  # (B, num_slots, D)

        mask = _count_mask(B, self.num_slots, count, device)
        return anchors, mask


class FixedTransformerAnchorLayer(AnchorLayer, anchor_mode="fixed_transformer"):
    """Learnable fixed slots conditioned on context via transformer cross-attention.

    Fixed slot embeddings serve as queries in a transformer decoder that cross-attends
    to the context embedding (and optionally sequence feature embeddings).
    """

    config_fields = frozenset(
        {"num_slots", "num_heads", "num_layers", "dropout"}
    )

    def __init__(self, hidden_size: int, num_slots: int = 10, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.num_slots = num_slots
        self.anchor_table = nn.Embedding(num_slots, hidden_size)
        nn.init.orthogonal_(self.anchor_table.weight)
        self.context_proj = nn.Linear(hidden_size, hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(
        self,
        context_embedding,
        feature_embeddings=None,
        count=None,
        threshold=0.5,
        feature_mask=None,
    ):
        B = context_embedding.shape[0]
        device = context_embedding.device

        # Fixed slots as decoder queries: (B, num_slots, D)
        slots = self.anchor_table.weight.unsqueeze(0).expand(B, -1, -1)
        # Context as memory: (B, 1, D)
        memory = self.context_proj(context_embedding).unsqueeze(1)
        memory_key_padding_mask = None
        # Optionally include sequence features as additional memory
        if feature_embeddings is not None:
            memory = torch.cat([memory, feature_embeddings], dim=1)  # (B, 1+L, D)
            if feature_mask is not None:
                if feature_mask.shape != feature_embeddings.shape[:2]:
                    raise ValueError(
                        "feature_mask must match the first two feature dimensions, "
                        f"got {tuple(feature_mask.shape)} for "
                        f"{tuple(feature_embeddings.shape)}"
                    )
                context_valid = torch.ones(B, 1, dtype=torch.bool, device=device)
                memory_valid = torch.cat(
                    [context_valid, feature_mask.to(device=device).bool()],
                    dim=1,
                )
                memory_key_padding_mask = ~memory_valid

        anchors = self.transformer_decoder(
            slots,
            memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (B, num_slots, D)

        mask = _count_mask(B, self.num_slots, count, device)
        return anchors, mask


class RotaryAnchorLayer(AnchorLayer, anchor_mode="rotary"):
    """Wraps RotaryGroupRNN — rotary position-conditioned anchor generation."""

    config_fields = frozenset({"max_count"})

    def __init__(self, hidden_size: int, max_count: int = 20, **kwargs):
        super().__init__()
        self.groups_layer = RotaryGroupRNN(hidden_size=hidden_size, max_count=max_count)

    def forward(
        self,
        context_embedding,
        feature_embeddings=None,
        count=None,
        threshold=0.5,
        feature_mask=None,
    ):
        B, D = context_embedding.shape
        device = context_embedding.device

        outputs = []
        for b in range(B):
            c = count[b].item() if count is not None else 1
            c = max(int(c), 1)
            # (1, D) single context vector as field_emb for RotaryGroupRNN
            out = self.groups_layer(context_embedding[b].unsqueeze(0), c)  # (count, 1, D)
            outputs.append(out.squeeze(1))  # (count, D)

        max_instances = max(o.shape[0] for o in outputs)
        anchors = torch.zeros(B, max_instances, D, device=device)
        mask = torch.zeros(B, max_instances, dtype=torch.bool, device=device)
        for b, out in enumerate(outputs):
            n = out.shape[0]
            anchors[b, :n] = out
            mask[b, :n] = True

        return anchors, mask


# Backward compat: "rnn" alias for "rotary"
AnchorLayer._registry["rnn"] = RotaryAnchorLayer


class QueryRNNAnchorLayer(AnchorLayer, anchor_mode="query_rnn"):
    """Wraps QueryGroupRNN — similarity-based token selection + GRU."""

    config_fields = frozenset({"max_count"})

    def __init__(self, hidden_size: int, max_count: int = 20, **kwargs):
        super().__init__()
        self.groups_layer = QueryGroupRNN(hidden_size=hidden_size, max_count=max_count)

    def forward(
        self,
        context_embedding,
        feature_embeddings=None,
        count=None,
        threshold=0.5,
        feature_mask=None,
    ):
        if feature_embeddings is None:
            B = context_embedding.shape[0]
            device = context_embedding.device
            return (
                torch.zeros(B, 0, context_embedding.shape[-1], device=device),
                torch.zeros(B, 0, dtype=torch.bool, device=device),
            )
        return self.groups_layer(
            context_embedding,
            feature_embeddings,
            count_val=count,
            threshold=threshold,
            token_mask=feature_mask,
        )


class QueryTransformerAnchorLayer(AnchorLayer, anchor_mode="query_transformer"):
    """Wraps QueryGroupTransformer — similarity-based selection + Transformer."""

    config_fields = frozenset(
        {"num_heads", "num_layers", "dropout", "max_count"}
    )

    def __init__(self, hidden_size: int, num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.1, max_count: int = 20, **kwargs):
        super().__init__()
        self.groups_layer = QueryGroupTransformer(
            hidden_size=hidden_size, num_heads=num_heads,
            num_layers=num_layers, dropout=dropout, max_count=max_count,
        )

    def forward(
        self,
        context_embedding,
        feature_embeddings=None,
        count=None,
        threshold=0.5,
        feature_mask=None,
    ):
        if feature_embeddings is None:
            B = context_embedding.shape[0]
            device = context_embedding.device
            return (
                torch.zeros(B, 0, context_embedding.shape[-1], device=device),
                torch.zeros(B, 0, dtype=torch.bool, device=device),
            )
        return self.groups_layer(
            context_embedding,
            feature_embeddings,
            count_val=count,
            threshold=threshold,
            token_mask=feature_mask,
        )
