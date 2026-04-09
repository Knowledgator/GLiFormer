"""Configurable anchor acquisition layers.

All extraction tasks follow: anchor + child -> spans.
The anchor layer is a configurable abstraction that produces anchor representations
from different sources depending on the task and configuration.

Strategies:
- ParentAnchorLayer: single parent embedding (NER, Classification default)
- FixedAnchorLayer: learnable nn.Embedding table (GLiNER2-style)
- RotaryAnchorLayer: wraps RotaryGroupLSTM
- QueryLSTMAnchorLayer: wraps QueryGroupLSTM
- QueryTransformerAnchorLayer: wraps QueryGroupTransformer
"""

from typing import Optional, Tuple

import torch
from torch import nn

from .groups import RotaryGroupLSTM, QueryGroupLSTM, QueryGroupTransformer


class AnchorLayer(nn.Module):
    """Base class for anchor acquisition layers.

    All subclasses produce:
        anchors: (B, A, D) — anchor representations
        mask: (B, A) — boolean mask for valid anchors
    """

    @classmethod
    def from_config(cls, anchor_mode: str, hidden_size: int, **kwargs) -> "AnchorLayer":
        """Factory method to create the appropriate anchor layer."""
        if anchor_mode == "parent":
            return ParentAnchorLayer(hidden_size)
        elif anchor_mode == "fixed":
            num_slots = kwargs.get("num_slots", 10)
            return FixedAnchorLayer(hidden_size, num_slots=num_slots)
        elif anchor_mode in ("lstm", "rotary"):
            max_count = kwargs.get("max_count", 20)
            return RotaryAnchorLayer(hidden_size, max_count=max_count)
        elif anchor_mode == "query_lstm":
            max_count = kwargs.get("max_count", 20)
            return QueryLSTMAnchorLayer(hidden_size, max_count=max_count)
        elif anchor_mode == "query_transformer":
            max_count = kwargs.get("max_count", 20)
            num_heads = kwargs.get("num_heads", 4)
            num_layers = kwargs.get("num_layers", 2)
            dropout = kwargs.get("dropout", 0.1)
            return QueryTransformerAnchorLayer(
                hidden_size, num_heads=num_heads, num_layers=num_layers,
                dropout=dropout, max_count=max_count,
            )
        else:
            raise ValueError(f"Unknown anchor_mode: {anchor_mode}")

    def forward(
        self,
        context_embedding: torch.Tensor,
        word_embeddings: Optional[torch.Tensor] = None,
        count: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            context_embedding: (B, C, D) prompt/field embeddings used as context
            word_embeddings: (B, L, D) text token embeddings (for query-based layers)
            count: (B,) or scalar — number of anchors to generate per sample
            threshold: float — similarity threshold (for query-based layers)

        Returns:
            anchors: (B, A, D) anchor representations
            mask: (B, A) boolean mask for valid anchors
        """
        raise NotImplementedError


class ParentAnchorLayer(AnchorLayer):
    """Single parent embedding as anchor — simplest case.

    Used by NER and Classification where the anchor is the parent prompt embedding.
    Returns a single anchor per sample (the mean-pooled prompt embedding).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, context_embedding, word_embeddings=None, count=None, threshold=0.5):
        B = context_embedding.shape[0]
        # Mean pool the context to get a single anchor per sample
        anchor = context_embedding.mean(dim=1, keepdim=True)  # (B, 1, D)
        mask = torch.ones(B, 1, dtype=torch.bool, device=anchor.device)
        return anchor, mask


class FixedAnchorLayer(AnchorLayer):
    """Learnable fixed-size embedding table for N anchor slots (GLiNER2-style).

    The number of anchors is a hyperparameter. Each anchor slot is a learnable
    embedding that is expanded across the batch.
    """

    def __init__(self, hidden_size: int, num_slots: int = 10):
        super().__init__()
        self.num_slots = num_slots
        self.anchor_table = nn.Embedding(num_slots, hidden_size)

    def forward(self, context_embedding, word_embeddings=None, count=None, threshold=0.5):
        B = context_embedding.shape[0]
        device = context_embedding.device

        anchors = self.anchor_table.weight.unsqueeze(0).expand(B, -1, -1)  # (B, num_slots, D)

        if count is not None:
            # Create mask based on count per sample
            max_count = anchors.shape[1]
            if count.dim() == 0:
                count = count.unsqueeze(0).expand(B)
            mask = torch.arange(max_count, device=device).unsqueeze(0) < count.unsqueeze(1)
        else:
            mask = torch.ones(B, self.num_slots, dtype=torch.bool, device=device)

        return anchors, mask


class RotaryAnchorLayer(AnchorLayer):
    """Wraps RotaryGroupLSTM — rotary position-conditioned anchor generation."""

    def __init__(self, hidden_size: int, max_count: int = 20):
        super().__init__()
        self.groups_layer = RotaryGroupLSTM(hidden_size=hidden_size, max_count=max_count)

    def forward(self, context_embedding, word_embeddings=None, count=None, threshold=0.5):
        B = context_embedding.shape[0]
        device = context_embedding.device

        if context_embedding.dim() == 3:
            outputs = []
            for b in range(B):
                c = count[b].item() if count is not None else 1
                c = max(int(c), 1)
                out = self.groups_layer(context_embedding[b], c)
                outputs.append(out)

            max_instances = max(o.shape[0] for o in outputs)
            D = outputs[0].shape[-1]
            anchors = torch.zeros(B, max_instances, D, device=device)
            mask = torch.zeros(B, max_instances, dtype=torch.bool, device=device)
            for b, out in enumerate(outputs):
                n = out.shape[0]
                anchors[b, :n] = out.mean(dim=1)
                mask[b, :n] = True
        else:
            c = count.item() if count is not None else 1
            c = max(int(c), 1)
            out = self.groups_layer(context_embedding, c)
            anchors = out.mean(dim=1).unsqueeze(0)
            mask = torch.ones(1, anchors.shape[1], dtype=torch.bool, device=device)

        return anchors, mask


class QueryLSTMAnchorLayer(AnchorLayer):
    """Wraps QueryGroupLSTM — similarity-based token selection + GRU."""

    def __init__(self, hidden_size: int, max_count: int = 20):
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
        field_emb = context_embedding.mean(dim=0) if context_embedding.dim() == 3 else context_embedding
        return self.groups_layer(field_emb, word_embeddings, count_val=count, threshold=threshold)


class QueryTransformerAnchorLayer(AnchorLayer):
    """Wraps QueryGroupTransformer — similarity-based selection + Transformer."""

    def __init__(self, hidden_size: int, num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.1, max_count: int = 20):
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
        field_emb = context_embedding.mean(dim=0) if context_embedding.dim() == 3 else context_embedding
        return self.groups_layer(field_emb, word_embeddings, count_val=count, threshold=threshold)
