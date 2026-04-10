"""Configurable anchor acquisition layers.

All extraction tasks follow: anchor + child -> spans.
The anchor layer is a configurable abstraction that produces anchor representations
from different sources depending on the task and configuration.

All layers accept context_embedding of shape (B, D) — a single context vector per sample.

Strategies:
- ParentAnchorLayer: returns context as single anchor (NER, Classification default)
- FixedAnchorLayer: learnable nn.Embedding table conditioned on context
- FixedLSTMAnchorLayer: learnable slots conditioned on context via GRU
- FixedTransformerAnchorLayer: learnable slots conditioned on context via transformer
- RotaryAnchorLayer: wraps RotaryGroupLSTM
- QueryLSTMAnchorLayer: wraps QueryGroupLSTM
- QueryTransformerAnchorLayer: wraps QueryGroupTransformer
"""

from typing import Optional, Tuple

import torch
from torch import nn

from .mlp import create_mlp
from .groups import RotaryGroupLSTM, QueryGroupLSTM, QueryGroupTransformer


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
        word_embeddings: Optional[torch.Tensor] = None,
        count: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            context_embedding: (B, D) context vector per sample
            word_embeddings: (B, L, D) text token embeddings (for query-based layers)
            count: (B,) or scalar — number of anchors to generate per sample
            threshold: float — similarity threshold (for query-based layers)

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

    def forward(self, context_embedding, word_embeddings=None, count=None, threshold=0.5):
        B = context_embedding.shape[0]
        anchor = context_embedding.unsqueeze(1)  # (B, 1, D)
        mask = torch.ones(B, 1, dtype=torch.bool, device=context_embedding.device)
        return anchor, mask


class FixedAnchorLayer(AnchorLayer, anchor_mode="fixed"):
    """Learnable fixed-size embedding table conditioned on context.

    Each anchor slot is a learnable embedding shifted by a projected context vector.
    """

    def __init__(self, hidden_size: int, num_slots: int = 10, **kwargs):
        super().__init__()
        self.num_slots = num_slots
        self.anchor_table = nn.Embedding(num_slots, hidden_size)
        self.context_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, context_embedding, word_embeddings=None, count=None, threshold=0.5):
        B = context_embedding.shape[0]
        device = context_embedding.device

        anchors = self.anchor_table.weight.unsqueeze(0).expand(B, -1, -1)  # (B, num_slots, D)
        context = self.context_proj(context_embedding).unsqueeze(1)  # (B, 1, D)
        anchors = anchors + context

        mask = _count_mask(B, self.num_slots, count, device)
        return anchors, mask


class FixedLSTMAnchorLayer(AnchorLayer, anchor_mode="fixed_lstm"):
    """Learnable fixed slots conditioned on context via GRU.

    Fixed slot embeddings are fed as input sequence to a GRU whose initial hidden
    state is the context embedding. Output is concatenated with context and projected.
    """

    def __init__(self, hidden_size: int, num_slots: int = 10, **kwargs):
        super().__init__()
        self.num_slots = num_slots
        self.anchor_table = nn.Embedding(num_slots, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.projector = create_mlp(
            input_dim=hidden_size * 2,
            intermediate_dims=[hidden_size * 4],
            output_dim=hidden_size,
            dropout=0.,
            activation="relu",
            add_layer_norm=False,
        )

    def forward(self, context_embedding, word_embeddings=None, count=None, threshold=0.5):
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
    to the context embedding (and optionally word embeddings).
    """

    def __init__(self, hidden_size: int, num_slots: int = 10, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.num_slots = num_slots
        self.anchor_table = nn.Embedding(num_slots, hidden_size)
        self.context_proj = nn.Linear(hidden_size, hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, context_embedding, word_embeddings=None, count=None, threshold=0.5):
        B = context_embedding.shape[0]
        device = context_embedding.device

        # Fixed slots as decoder queries: (B, num_slots, D)
        slots = self.anchor_table.weight.unsqueeze(0).expand(B, -1, -1)
        # Context as memory: (B, 1, D)
        memory = self.context_proj(context_embedding).unsqueeze(1)
        # Optionally include word embeddings as additional memory
        if word_embeddings is not None:
            memory = torch.cat([memory, word_embeddings], dim=1)  # (B, 1+L, D)

        anchors = self.transformer_decoder(slots, memory)  # (B, num_slots, D)

        mask = _count_mask(B, self.num_slots, count, device)
        return anchors, mask


class RotaryAnchorLayer(AnchorLayer, anchor_mode="rotary"):
    """Wraps RotaryGroupLSTM — rotary position-conditioned anchor generation."""

    def __init__(self, hidden_size: int, max_count: int = 20, **kwargs):
        super().__init__()
        self.groups_layer = RotaryGroupLSTM(hidden_size=hidden_size, max_count=max_count)

    def forward(self, context_embedding, word_embeddings=None, count=None, threshold=0.5):
        B, D = context_embedding.shape
        device = context_embedding.device

        outputs = []
        for b in range(B):
            c = count[b].item() if count is not None else 1
            c = max(int(c), 1)
            # (1, D) single context vector as field_emb for RotaryGroupLSTM
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


# Backward compat: "lstm" alias for "rotary"
AnchorLayer._registry["lstm"] = RotaryAnchorLayer


class QueryLSTMAnchorLayer(AnchorLayer, anchor_mode="query_lstm"):
    """Wraps QueryGroupLSTM — similarity-based token selection + GRU."""

    def __init__(self, hidden_size: int, max_count: int = 20, **kwargs):
        super().__init__()
        self.groups_layer = QueryGroupLSTM(hidden_size=hidden_size, max_count=max_count)

    def forward(self, context_embedding, word_embeddings=None, count=None, threshold=0.5):
        if word_embeddings is None:
            B = context_embedding.shape[0]
            device = context_embedding.device
            return (
                torch.zeros(B, 0, context_embedding.shape[-1], device=device),
                torch.zeros(B, 0, dtype=torch.bool, device=device),
            )
        return self.groups_layer(context_embedding, word_embeddings, count_val=count, threshold=threshold)


class QueryTransformerAnchorLayer(AnchorLayer, anchor_mode="query_transformer"):
    """Wraps QueryGroupTransformer — similarity-based selection + Transformer."""

    def __init__(self, hidden_size: int, num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.1, max_count: int = 20, **kwargs):
        super().__init__()
        self.groups_layer = QueryGroupTransformer(
            hidden_size=hidden_size, num_heads=num_heads,
            num_layers=num_layers, dropout=dropout, max_count=max_count,
        )

    def forward(self, context_embedding, word_embeddings=None, count=None, threshold=0.5):
        if word_embeddings is None:
            B = context_embedding.shape[0]
            device = context_embedding.device
            return (
                torch.zeros(B, 0, context_embedding.shape[-1], device=device),
                torch.zeros(B, 0, dtype=torch.bool, device=device),
            )
        return self.groups_layer(context_embedding, word_embeddings, count_val=count, threshold=threshold)
