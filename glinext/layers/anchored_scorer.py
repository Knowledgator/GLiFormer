"""Anchored span scorer for unified anchor+child->spans extraction."""

from typing import Optional

import torch
from torch import nn


class AnchoredSpanScorer(nn.Module):
    """Scores text positions for span extraction given fused anchor-child representations.

    Takes already-fused representations (from AnchorModeling) and word embeddings,
    produces start/inside/end scores per text position. Follows GLiNER Scorer pattern:
    bilinear interaction + MLP -> 3 scores.

    Used by:
    - NER (anchored mode): fused = anchor_modeling(parent, entity_types)
    - Relations: fused = anchor_modeling(entity_span, relation_types)
    - Structuring: fused = anchor_modeling(instance_rep, field_types)
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size

        self.proj_token = nn.Linear(hidden_size, hidden_size * 2)
        self.proj_label = nn.Linear(hidden_size, hidden_size * 2)
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 4),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, 3),  # start, end, inside
        )

    def forward(
        self,
        fused_rep: torch.Tensor,
        word_embs: torch.Tensor,
        word_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score fused anchor-child representations against word embeddings.

        Args:
            fused_rep: (B, N, D) pre-fused anchor-child representations
            word_embs: (B, L, D) text token embeddings
            word_mask: (B, L) optional mask for valid text positions

        Returns:
            scores: (B, N, L, 3) span scores — start/end/inside per fused rep
        """
        batch_size, num_classes, hidden_size = fused_rep.shape
        seq_len = word_embs.shape[1]

        # Project and split into two components for bilinear interaction
        # Shape: (batch_size, seq_len, 1, 2, hidden_size)
        token_proj = self.proj_token(word_embs).view(batch_size, seq_len, 1, 2, hidden_size)
        # Shape: (batch_size, 1, num_classes, 2, hidden_size)
        label_proj = self.proj_label(fused_rep).view(batch_size, 1, num_classes, 2, hidden_size)

        # Expand for pairwise computation
        # Shape: (2, batch_size, seq_len, num_classes, hidden_size)
        token_proj = token_proj.expand(-1, -1, num_classes, -1, -1).permute(3, 0, 1, 2, 4)
        label_proj = label_proj.expand(-1, seq_len, -1, -1, -1).permute(3, 0, 1, 2, 4)

        # Concatenate: [token_proj_1, label_proj_1, token_proj_2 * label_proj_2]
        # Shape: (batch_size, seq_len, num_classes, hidden_size * 3)
        cat = torch.cat([token_proj[0], label_proj[0], token_proj[1] * label_proj[1]], dim=-1)

        # Compute final scores: (batch_size, seq_len, num_classes, 3)
        scores = self.out_mlp(cat)

        # Permute to (B, N, L, 3) to match expected output layout
        scores = scores.permute(0, 2, 1, 3)

        if word_mask is not None:
            scores = scores * word_mask[:, None, :, None].float()

        return scores