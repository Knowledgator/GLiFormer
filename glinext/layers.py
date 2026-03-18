from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers.activations import ACT2FN

from .config import GLiNextConfig


def create_mlp(input_dim, intermediate_dims, output_dim, dropout=0.1, activation="gelu", add_layer_norm=False):
    """
    Creates a multi-layer perceptron (MLP) with specified dimensions and activation functions.
    """
    activation_mapping = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU
    }
    layers = []
    in_dim = input_dim
    for dim in intermediate_dims:
        layers.append(nn.Linear(in_dim, dim))
        if add_layer_norm:
            layers.append(nn.LayerNorm(dim))
        layers.append(activation_mapping[activation]())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)

class LstmSeq2SeqEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0., bidirectional=False):
        super(LstmSeq2SeqEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=bidirectional,
                            batch_first=True)

    def forward(self, x, mask, hidden=None):
        # Packing the input sequence
        lengths = mask.sum(dim=1).cpu()
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Passing packed sequence through LSTM
        packed_output, hidden = self.lstm(packed_x, hidden)

        # Unpacking the output sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        return output

class FeaturesProjector(nn.Module):
    def __init__(self, config: GLiNextConfig):
        super().__init__()

        self.linear_1 = nn.Linear(config.encoder_config.hidden_size, config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.dropout = nn.Dropout(config.dropout)
        self.linear_2 = nn.Linear(config.hidden_size, config.encoder_config.hidden_size, bias=True)

    def forward(self, features):
        hidden_states = self.linear_1(features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class PairRepLayer(nn.Module):
    """Creates pair representations from head and tail entity embeddings.

    Supports multiple combination strategies:
      - 'concat_proj': concatenate + linear projection (default, same as original)
      - 'bilinear': element-wise product of independently projected head/tail
      - 'additive': sum of independently projected head/tail + nonlinearity
      - 'mlp': deeper MLP on concatenated head/tail
    """

    def __init__(self, hidden_size: int, pair_rep_type: str = "concat_proj", dropout: float = 0.1):
        super().__init__()
        self.pair_rep_type = pair_rep_type

        if pair_rep_type == "concat_proj":
            self.layer = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, hidden_size),
            )
        elif pair_rep_type == "bilinear":
            self.head_proj = nn.Linear(hidden_size, hidden_size)
            self.tail_proj = nn.Linear(hidden_size, hidden_size)
        elif pair_rep_type == "additive":
            self.head_proj = nn.Linear(hidden_size, hidden_size)
            self.tail_proj = nn.Linear(hidden_size, hidden_size)
            self.out = nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        elif pair_rep_type == "mlp":
            self.layer = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size * 4),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(hidden_size * 4, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),
            )
        else:
            raise ValueError(f"Unknown pair_rep_type: {pair_rep_type}")

    def forward(self, head_rep: torch.Tensor, tail_rep: torch.Tensor) -> torch.Tensor:
        """Combine head and tail representations into a pair representation.

        Args:
            head_rep: (B, N, D) head entity embeddings.
            tail_rep: (B, N, D) tail entity embeddings.

        Returns:
            pair_rep: (B, N, D)
        """
        if self.pair_rep_type in ("concat_proj", "mlp"):
            return self.layer(torch.cat([head_rep, tail_rep], dim=-1))
        elif self.pair_rep_type == "bilinear":
            return self.head_proj(head_rep) * self.tail_proj(tail_rep)
        elif self.pair_rep_type == "additive":
            return self.out(self.head_proj(head_rep) + self.tail_proj(tail_rep))


class PromptRelationExtractor(nn.Module):
    """Relation extraction via prompt-guided source and target entity selection.

    Instead of first predicting an adjacency matrix and then classifying pairs,
    this module uses relation prompt embeddings to:
      1. Score each entity as a potential source for each relation type.
      2. Combine source entity + relation prompt to score potential targets.

    This produces a dense (B, E, E, C) score tensor that is then thresholded
    to select entity pairs, yielding the same output format as the adjacency-based
    approach.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.source_proj = nn.Linear(hidden_size, hidden_size)
        self.source_rel_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.target_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        entity_rep: torch.Tensor,
        rel_prompts: torch.Tensor,
        entity_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score all (source, relation, target) triples.

        Args:
            entity_rep: (B, E, D) entity embeddings.
            rel_prompts: (B, C, D) relation prompt embeddings.
            entity_mask: (B, E) optional mask for valid entities.

        Returns:
            scores: (B, E, E, C) — scores[b, src, tgt, rel].
        """
        B, E, D = entity_rep.shape
        C = rel_prompts.size(1)

        # Combine each source entity with each relation type
        entity_exp = entity_rep.unsqueeze(2).expand(B, E, C, D)
        rel_exp = rel_prompts.unsqueeze(1).expand(B, E, C, D)
        combined = self.source_rel_proj(
            torch.cat([entity_exp, rel_exp], dim=-1)
        )  # (B, E, C, D)

        # Score each target entity
        tgt_proj = self.target_proj(entity_rep)  # (B, E, D)
        # (B, E_src, C, D) @ (B, E_tgt, D)^T → (B, E_src, C, E_tgt)
        scores = torch.einsum("becd,bfd->becf", combined, tgt_proj)

        # Rearrange to (B, E_src, E_tgt, C)
        scores = scores.permute(0, 1, 3, 2)

        # Mask invalid entities and self-loops
        if entity_mask is not None:
            m = entity_mask.float()
            scores = scores * m[:, :, None, None] * m[:, None, :, None]

        diag_mask = ~torch.eye(E, device=scores.device, dtype=torch.bool)
        scores = scores * diag_mask[None, :, :, None].float()

        return scores


class AnchoredSpanScorer(nn.Module):
    """Scores text positions for span extraction conditioned on anchor representations.

    Unified mechanism for anchor-based tasks:
    - NER: anchor=parent (broadcast), child=entity_type -> spans
    - Relations (anchor mode): anchor=entity span, child=relation_type -> target spans
    - Structuring: anchor=instance rep (from groups layer), child=field_type -> value spans

    For each (anchor, child) pair, produces start/inside/end scores per text position.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size

        self.anchor_proj = nn.Linear(hidden_size, hidden_size)
        self.gate_proj = nn.Linear(hidden_size * 2, hidden_size)

        # 3-channel projections for start/end/inside scoring
        self.word_channel_proj = nn.Linear(hidden_size, hidden_size * 3)
        self.child_channel_proj = nn.Linear(hidden_size, hidden_size * 3)

    def forward(
        self,
        anchor_rep: torch.Tensor,
        child_rep: torch.Tensor,
        word_embs: torch.Tensor,
        child_mask: Optional[torch.Tensor] = None,
        word_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            anchor_rep: (B, A, D) anchor representations (instances/entities/parents)
            child_rep: (B, C, D) child/field representations
            word_embs: (B, L, D) text token embeddings
            child_mask: (B, C) optional mask for valid children
            word_mask: (B, L) optional mask for valid text positions

        Returns:
            scores: (B, A, L, C, 3) span scores — start/end/inside per anchor per child
        """
        B, A, D = anchor_rep.shape
        C = child_rep.shape[1]
        L = word_embs.shape[1]

        # Condition word embeddings on each anchor via gating
        anchor_proj = self.anchor_proj(anchor_rep)  # (B, A, D)
        word_exp = word_embs.unsqueeze(1).expand(B, A, L, D)
        anchor_exp = anchor_proj.unsqueeze(2).expand(B, A, L, D)

        gate = torch.sigmoid(self.gate_proj(
            torch.cat([word_exp, anchor_exp], dim=-1)
        ))  # (B, A, L, D)
        conditioned = word_exp + gate * anchor_exp  # (B, A, L, D)

        # Project to 3 channels: (B, A, L, 3, D)
        word_3ch = self.word_channel_proj(conditioned).view(B, A, L, 3, D)

        # Project children to 3 channels: (B, C, 3, D)
        child_3ch = self.child_channel_proj(child_rep).view(B, C, 3, D)

        # Score: (B, A, L, C, 3)
        scores = torch.einsum("baltd,bctd->balct", word_3ch, child_3ch)

        if word_mask is not None:
            scores = scores * word_mask[:, None, :, None, None].float()
        if child_mask is not None:
            scores = scores * child_mask[:, None, None, :, None].float()

        return scores


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


class RotaryEmbedding(nn.Module):
    """
    Standard RoPE (rotary) embedding that returns (cos, sin) tensors for a batch of position_ids.
    """
    inv_freq: torch.Tensor

    def __init__(
        self,
        dim: int,
        base: float = 10_000.0,
        attention_scaling: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RotaryEmbedding requires even head_dim; got {dim}")

        self.dim = dim
        self.base = float(base)
        self.attention_scaling = float(attention_scaling)

        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x_like: torch.Tensor, position_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos = position_ids[:, None, :].float()

        device_type = x_like.device.type if (isinstance(x_like.device.type, str) and x_like.device.type != "mps") else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv.float() @ pos.float()).transpose(1, 2)
            emb = torch.cat([freqs, freqs], dim=-1)

            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x_like.dtype, device=x_like.device), sin.to(dtype=x_like.dtype, device=x_like.device)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate last-dim halves: (x1, x2) -> (-x2, x1)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to q. Shapes: q: [bs, seq, head_dim], cos/sin: [bs, seq, head_dim]."""
    return (q * cos) + (rotate_half(q) * sin)


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
