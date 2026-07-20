"""Configurable anchor acquisition layers.

All extraction tasks follow: anchor + child -> spans.
The anchor layer is a configurable abstraction that produces anchor representations
from different sources depending on the task and configuration.

All layers accept context_embedding of shape (B, D) — a single context vector per sample.

Strategies:
- ParentAnchorLayer: returns context as single anchor (NER, Classification default)
- FeatureAnchorLayer: returns sequence feature embeddings as anchors
- FixedAnchorLayer: learnable nn.Embedding table conditioned on context
- FixedRNNAnchorLayer: learnable slots conditioned on context via GRU
- FixedTransformerAnchorLayer: learnable slots conditioned on context via transformer
- RotaryAnchorLayer: wraps RotaryGroupRNN
- QueryRNNAnchorLayer: wraps QueryGroupRNN
- QueryTransformerAnchorLayer: wraps QueryGroupTransformer
"""

from typing import Optional, Tuple

import torch
from torch import nn

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

    def __init_subclass__(cls, anchor_mode: str = "", **kwargs):
        super().__init_subclass__(**kwargs)
        if anchor_mode:
            cls._registry[anchor_mode] = cls

    @classmethod
    def from_config(cls, anchor_mode: str, hidden_size: int, **kwargs) -> "AnchorLayer":
        """Factory method to create the appropriate anchor layer."""
        if anchor_mode not in cls._registry:
            raise ValueError(f"Unknown anchor_mode: {anchor_mode}. Available: {list(cls._registry.keys())}")
        return cls._registry[anchor_mode](hidden_size, **kwargs)

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


class FixedAnchorLayer(AnchorLayer, anchor_mode="fixed"):
    """Learnable fixed-size embedding table conditioned on context.

    Each anchor slot is a learnable embedding shifted by a gated projected
    context vector.  The configurable gate lets set-prediction heads disable
    that shared context component without changing the fixed-slot state-dict
    layout used by existing checkpoints.
    """

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
