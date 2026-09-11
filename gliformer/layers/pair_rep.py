"""Pair representation and prompt relation extraction layers."""

from typing import Optional

import torch
from torch import nn


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
