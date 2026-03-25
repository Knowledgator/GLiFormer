"""Group/anchor generation layers for structuring tasks."""

import torch
from torch import nn

from .mlp import create_mlp
from .rotary import RotaryEmbedding, apply_rotary_pos_emb


class RotaryGroupLSTM(nn.Module):
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

    def forward(self, pc_emb: torch.Tensor, gold_count_val: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pc_emb (Tensor): Field embeddings of shape (M, hidden_size).
            gold_count_val (int): Predicted count value (number of steps).
        Returns:
            Tensor: Count-aware structure embeddings of shape (gold_count_val, M, hidden_size).
        """
        M, D = pc_emb.shape
        device = pc_emb.device

        min_count = min(gold_count_val, self.max_count)
        base_indices = torch.arange(min_count, device=device)
        base_pos = self.pos_embedding(base_indices)

        if gold_count_val > self.max_count:
            num_repeats = (gold_count_val + self.max_count - 1) // self.max_count
            pos_seq = base_pos.repeat(num_repeats, 1)[:gold_count_val, :]
        else:
            pos_seq = base_pos

        pos_seq = pos_seq.unsqueeze(0)

        position_ids = torch.arange(gold_count_val, device=device).unsqueeze(0)

        cos, sin = self.rotary_embeddings(pos_seq, position_ids)
        pos_seq = apply_rotary_pos_emb(pos_seq, cos, sin)

        pos_seq = pos_seq.squeeze(0).unsqueeze(1).expand(-1, M, -1)

        h0 = pc_emb.unsqueeze(0)

        output, _ = self.gru(pos_seq, h0)

        pc_broadcast = pc_emb.unsqueeze(0).expand_as(output)
        return self.projector(torch.cat([output, pc_broadcast], dim=-1))


class QueryGroupLSTM(nn.Module):
    def __init__(self, hidden_size, max_count=20, rope_base=10_000.0):
        """
        Like RotaryGroupLSTM but selects tokens from token_emb (B, L, D) based on
        similarity with pc_emb (M, D) instead of using learned positional embeddings.
        """
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

    def forward(self, pc_emb: torch.Tensor, token_emb: torch.Tensor, gold_count_val: torch.Tensor = None, threshold: float = 0.5) -> torch.Tensor:
        """
        Args:
            pc_emb (Tensor): Field embeddings of shape (M, D).
            token_emb (Tensor): Token embeddings of shape (B, L, D).
            gold_count_val (Tensor): Per-sample counts of shape (B,), or None.
            threshold (float): Similarity threshold when gold_count_val is None.
        Returns:
            Tuple[Tensor, Tensor]: (output of shape (B, k, D), mask of shape (B, k))
        """
        B, L, _ = token_emb.shape
        device = pc_emb.device

        # Similarity between each token and each field: (B, L, M)
        pc_token = torch.einsum("bld,md->blm", token_emb, pc_emb)

        # Max similarity across fields for each token: (B, L)
        max_sim = pc_token.max(dim=2).values

        if gold_count_val is not None:
            max_k = min(int(gold_count_val.max().item()), L)
            _, topk_indices = torch.topk(max_sim, k=max_k, dim=1)  # (B, max_k)

            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(topk_indices)
            selected_token_emb = token_emb[batch_idx, topk_indices]  # (B, max_k, D)

            mask = torch.arange(max_k, device=device).unsqueeze(0) < gold_count_val.unsqueeze(1)  # (B, max_k)
        else:
            mask = max_sim > threshold  # (B, L)
            selected_token_emb = token_emb  # (B, L, D)

        # Apply rotary position embeddings
        seq_len = selected_token_emb.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_embeddings(selected_token_emb, position_ids)
        selected_token_emb = apply_rotary_pos_emb(selected_token_emb, cos, sin)

        # GRU: use mean-pooled pc_emb as initial hidden state
        # GRU expects input (seq, batch, D) and h0 (1, batch, D)
        h0 = pc_emb.mean(dim=0, keepdim=True).unsqueeze(0).expand(1, B, -1).contiguous()  # (1, B, D)
        gru_input = selected_token_emb.transpose(0, 1)  # (seq_len, B, D)

        output, _ = self.gru(gru_input, h0)  # (seq_len, B, D)
        output = output.transpose(0, 1)  # (B, seq_len, D)

        # Concat with mean-pooled pc_emb and project
        pc_broadcast = pc_emb.mean(dim=0, keepdim=True).unsqueeze(0).expand(B, seq_len, -1)  # (B, seq_len, D)
        output = self.projector(torch.cat([output, pc_broadcast], dim=-1))  # (B, seq_len, D)

        return output, mask


class QueryGroupTransformer(nn.Module):
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

    def forward(self, pc_emb: torch.Tensor, token_emb: torch.Tensor, gold_count_val: torch.Tensor = None, threshold: float = 0.5) -> torch.Tensor:
        """
        Args:
            pc_emb (Tensor): Field embeddings of shape (M, D).
            token_emb (Tensor): Token embeddings of shape (B, L, D).
            gold_count_val (Tensor): Per-sample counts of shape (B,), or None.
            threshold (float): Similarity threshold when gold_count_val is None.
        Returns:
            Tuple[Tensor, Tensor]: (output of shape (B, k, D), mask of shape (B, k))
        """
        M = pc_emb.shape[0]
        B, L, _ = token_emb.shape
        device = pc_emb.device

        # Similarity between each token and each field: (B, L, M)
        pc_token = torch.einsum("bld,md->blm", token_emb, pc_emb)

        # Max similarity across fields for each token: (B, L)
        max_sim = pc_token.max(dim=2).values

        if gold_count_val is not None:
            # Select top-k tokens based on max similarity with fields
            max_k = min(int(gold_count_val.max().item()), L)
            _, topk_indices = torch.topk(max_sim, k=max_k, dim=1)  # (B, max_k)

            # Gather selected tokens per batch
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(topk_indices)
            selected_token_emb = token_emb[batch_idx, topk_indices]  # (B, max_k, D)

            # Mask for variable counts per sample
            mask = torch.arange(max_k, device=device).unsqueeze(0) < gold_count_val.unsqueeze(1)  # (B, max_k)
        else:
            # Select tokens based on threshold — keep all tokens but mask below threshold
            mask = max_sim > threshold  # (B, L)
            selected_token_emb = token_emb  # (B, L, D)

        # Apply rotary position embeddings
        seq_len = selected_token_emb.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_embeddings(selected_token_emb, position_ids)
        selected_token_emb = apply_rotary_pos_emb(selected_token_emb, cos, sin)

        # Prepend pc_emb as context tokens: (B, M + seq_len, D)
        pc_expanded = pc_emb.unsqueeze(0).expand(B, -1, -1)  # (B, M, D)
        combined = torch.cat([pc_expanded, selected_token_emb], dim=1)  # (B, M + seq_len, D)

        # Extend mask to cover the prepended pc_emb tokens (always valid)
        pc_mask = torch.ones(B, M, dtype=torch.bool, device=device)
        full_mask = torch.cat([pc_mask, mask], dim=1)  # (B, M + seq_len)

        # Transformer encoder (expects seq-first format)
        transformer_output = self.transformer_encoder(
            combined.transpose(0, 1), src_key_padding_mask=~full_mask
        ).transpose(0, 1)  # (B, M + seq_len, D)

        # Strip the M prefix tokens, keep only selected token outputs
        transformer_output = transformer_output[:, M:, :]  # (B, seq_len, D)

        # Concat with mean-pooled pc_emb and project
        pc_broadcast = pc_emb.mean(dim=0, keepdim=True).unsqueeze(0).expand(B, seq_len, -1)  # (B, seq_len, D)
        output = self.projector(torch.cat([transformer_output, pc_broadcast], dim=-1))  # (B, seq_len, D)

        return output, mask
