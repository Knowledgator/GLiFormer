"""Query built from a record anchor and a field marker that reads the text.

Same layout as the structuring option scorer's ``_record_field_vectors``: an
MLP over ``[record, field, record * field]`` gives a query, which reads the
text through one cross-attention layer and an FFN (post-norm residuals).
"""

from __future__ import annotations

import torch
from torch import nn

from .mlp import create_mlp


class RecordFieldReadout(nn.Module):
    def __init__(self, hidden_size: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if heads < 1 or hidden_size % heads != 0:
            heads = 1
        self.query = create_mlp(
            input_dim=3 * hidden_size,
            intermediate_dims=[hidden_size],
            output_dim=hidden_size,
            dropout=dropout,
            activation="gelu",
        )
        self.readout = nn.MultiheadAttention(hidden_size, heads, dropout=dropout, batch_first=True)
        self.readout_norm = nn.LayerNorm(hidden_size)
        self.ffn = create_mlp(
            input_dim=hidden_size,
            intermediate_dims=[2 * hidden_size],
            output_dim=hidden_size,
            dropout=dropout,
            activation="gelu",
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        record: torch.Tensor,
        field: torch.Tensor,
        words: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> torch.Tensor:
        """``(N, D)`` records and fields, ``(N, W, D)`` text -> ``(N, D)``."""

        query = self.query(torch.cat([record, field, record * field], dim=-1)).unsqueeze(1)
        if words.shape[1] == 0:
            return query.squeeze(1)
        key_padding = ~word_mask.bool()
        empty = key_padding.all(dim=-1)
        if empty.any():
            key_padding = key_padding.clone()
            key_padding[empty, 0] = False
        attended, _ = self.readout(query, words, words, key_padding_mask=key_padding, need_weights=False)
        hidden = self.readout_norm(query + attended)
        hidden = self.ffn_norm(hidden + self.ffn(hidden))
        return hidden.squeeze(1)


__all__ = ["RecordFieldReadout"]
