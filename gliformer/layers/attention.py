"""Attention blocks and fusers."""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        return self.norm(x + self.dropout(attn_output))

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        attn_output, _ = self.cross_attn(query, key, value, attn_mask=mask)
        return self.norm(query + self.dropout(attn_output))

class Fuser(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            nn.ModuleList([
                SelfAttentionBlock(d_model, num_heads, dropout),
                CrossAttentionBlock(d_model, num_heads, dropout)
            ])
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, query, key, query_mask=None, key_mask=None):
        if query_mask is not None and key_mask is not None:
            self_attn_mask = query_mask.unsqueeze(1) * query_mask.unsqueeze(2)
            cross_attn_mask = query_mask.unsqueeze(-1) * key_mask.unsqueeze(1)
        else:
            self_attn_mask = None
            cross_attn_mask = None

        value = self.fc(key)

        for self_attn, cross_attn in self.layers:
            query = self_attn(query, mask=self_attn_mask)
            query = cross_attn(query, key, value, mask=cross_attn_mask)

        return query

class LayerwiseAttention(nn.Module):
    def __init__(self, num_layers, hidden_size, output_size=None):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size if output_size is not None else hidden_size

        # Squeeze operation
        self.squeeze = nn.Linear(hidden_size, 1)

        # Excitation operation
        self.W1 = nn.Linear(num_layers, num_layers // 2)
        self.W2 = nn.Linear(num_layers // 2, num_layers)

        # Final projection
        self.output_projection = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, encoder_outputs):
        # encoder_outputs is a list of tensors, each of shape [B, L, D]
        B, L, D = encoder_outputs[0].shape

        # Concatenate all layers
        U = torch.stack(encoder_outputs, dim=1)  # [B, K, L, D]

        # Squeeze operation
        Z = self.squeeze(U).squeeze(-1)  # [B, K, L]
        Z = Z.mean(dim=2)  # [B, K]

        # Excitation operation
        s = self.W2(F.relu(self.W1(Z)))  # [B, K]
        s = torch.sigmoid(s)  # [B, K]

        # Apply attention weights
        U_weighted = U * s.unsqueeze(-1).unsqueeze(-1)  # [B, K, L, D]

        # Sum across layers
        U_sum = U_weighted.sum(dim=1)  # [B, L, D]

        # Final projection
        output = self.output_projection(U_sum)  # [B, L, output_size]

        return output


class CrossModalTokenFusion(nn.Module):
    """Fuse non-text modality tokens into text word embeddings without changing word length."""

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        text_embeddings: torch.Tensor,
        modality_embeddings: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if modality_embeddings is None or modality_embeddings.shape[1] == 0:
            return text_embeddings
        key_padding_mask = None
        if modality_mask is not None:
            key_padding_mask = ~modality_mask.bool()
        attended, _ = self.attention(
            text_embeddings,
            modality_embeddings,
            modality_embeddings,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.norm(text_embeddings + self.dropout(attended))
