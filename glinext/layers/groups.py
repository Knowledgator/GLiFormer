"""Group/anchor generation layers for structuring tasks."""

import torch
from torch import nn
from .mlp import create_mlp
from .rotary import RotaryEmbedding, apply_rotary_pos_emb


class _AnchorCrossAttentionBlock(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm0 = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, anchor_rep, token_emb, token_mask=None, query_pos_emb=None):
        # Add slot positional embedding to Q and K only (not V) so each slot
        # maintains a unique identity through self-attention (DETR convention).
        q = anchor_rep if query_pos_emb is None else anchor_rep + query_pos_emb
        sa_out, _ = self.self_attn(q, q, anchor_rep)
        anchor_rep = self.norm0(anchor_rep + sa_out)
        key_padding_mask = ~token_mask.bool() if token_mask is not None else None
        q_cross = anchor_rep if query_pos_emb is None else anchor_rep + query_pos_emb
        attn_out, _ = self.cross_attn(
            q_cross, token_emb, token_emb, key_padding_mask=key_padding_mask,
        )
        x = self.norm1(anchor_rep + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x

class AnchorCrossAttentionLayer(nn.Module):
    """Pre-processing layer that refines anchor embeddings via cross-attention with token embeddings.

    Sits between anchor acquisition (AnchorLayer) and anchor modeling (AnchorModeling).
    Each layer applies: anchor attends to tokens → residual + norm → FFN → residual + norm.
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            _AnchorCrossAttentionBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        anchor_rep: torch.Tensor,
        token_emb: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        query_pos_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            anchor_rep: (B, A, D) anchor embeddings to refine
            token_emb: (B, L, D) token embeddings from encoder
            token_mask: (B, L) optional mask for valid token positions
            query_pos_emb: (B, A, D) or (1, A, D) slot positional embeddings
                added to Q (and K) in self- and cross-attention at every layer.
                Prevents anchor collapse when anchors start with similar values.

        Returns:
            refined: (B, A, D) refined anchor embeddings
        """
        for layer in self.layers:
            anchor_rep = layer(anchor_rep, token_emb, token_mask, query_pos_emb)
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
            dropout=0.,
            activation="relu",
            add_layer_norm=False
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
            dropout=0.,
            activation="relu",
            add_layer_norm=False
        )

    def forward(self, context_embedding: torch.Tensor, token_emb: torch.Tensor,
                count_val: torch.Tensor = None, threshold: float = 0.5) -> torch.Tensor:
        """
        Args:
            context_embedding (Tensor): Batched context of shape (B, D).
            token_emb (Tensor): Token embeddings of shape (B, L, D).
            count_val (Tensor): Per-sample counts of shape (B,), or None.
            threshold (float): Similarity threshold when count_val is None.
        Returns:
            Tuple[Tensor, Tensor]: (output of shape (B, k, D), mask of shape (B, k))
        """
        B, L, _ = token_emb.shape
        device = context_embedding.device

        # Per-sample similarity: (B, L)
        max_sim = torch.einsum("bld,bd->bl", token_emb, context_embedding)

        if count_val is not None:
            max_k = min(int(count_val.max().item()), L)
            _, topk_indices = torch.topk(max_sim, k=max_k, dim=1)  # (B, max_k)

            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(topk_indices)
            selected_token_emb = token_emb[batch_idx, topk_indices]  # (B, max_k, D)

            mask = torch.arange(max_k, device=device).unsqueeze(0) < count_val.unsqueeze(1)  # (B, max_k)
        else:
            mask = max_sim > threshold  # (B, L)
            selected_token_emb = token_emb  # (B, L, D)

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

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.projector = create_mlp(
            input_dim=hidden_size * 2,
            intermediate_dims=[hidden_size * 4],
            output_dim=hidden_size,
            dropout=0.,
            activation="relu",
            add_layer_norm=False
        )

    def forward(self, context_embedding: torch.Tensor, token_emb: torch.Tensor,
                count_val: torch.Tensor = None, threshold: float = 0.5) -> torch.Tensor:
        """
        Args:
            context_embedding (Tensor): Batched context of shape (B, D).
            token_emb (Tensor): Token embeddings of shape (B, L, D).
            count_val (Tensor): Per-sample counts of shape (B,), or None.
            threshold (float): Similarity threshold when count_val is None.
        Returns:
            Tuple[Tensor, Tensor]: (output of shape (B, k, D), mask of shape (B, k))
        """
        B, L, _ = token_emb.shape
        device = context_embedding.device

        # Per-sample similarity: (B, L)
        max_sim = torch.einsum("bld,bd->bl", token_emb, context_embedding)

        if count_val is not None:
            max_k = min(int(count_val.max().item()), L)
            _, topk_indices = torch.topk(max_sim, k=max_k, dim=1)  # (B, max_k)

            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(topk_indices)
            selected_token_emb = token_emb[batch_idx, topk_indices]  # (B, max_k, D)

            mask = torch.arange(max_k, device=device).unsqueeze(0) < count_val.unsqueeze(1)  # (B, max_k)
        else:
            mask = max_sim > threshold  # (B, L)
            selected_token_emb = token_emb  # (B, L, D)

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
        output = self.projector(torch.cat([transformer_output, context_broadcast], dim=-1))  # (B, seq_len, D)

        return output, mask
