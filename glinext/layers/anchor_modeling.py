"""Anchor modeling layers — post-anchor processing that fuses anchor + child representations.

After anchor acquisition (via AnchorLayer), the anchor_modeling layer controls how
anchor and child representations are combined before scoring. This is the configurable
"anchor modeling" step described in CLAUDE.md.

Strategies:
- LinearAnchorModeling: linear projection of concatenated anchor + child
- LSTMAnchorModeling: recurrent processing (GLiNER2-style)
- MLPAnchorModeling: multi-layer perceptron fusion
"""

import torch
from torch import nn

from .mlp import create_mlp


class AnchorModeling(nn.Module):
    """Base class for anchor modeling layers.

    Takes anchor reps (B, A, D) and child reps (B, C, D) and produces
    fused representations (B, A, C, D) ready for scoring.
    """

    @classmethod
    def from_config(cls, modeling_type: str, hidden_size: int, **kwargs) -> "AnchorModeling":
        """Factory method to create the appropriate modeling layer."""
        if modeling_type == "lstm":
            return LSTMAnchorModeling(hidden_size, **kwargs)
        elif modeling_type == "mlp":
            dropout = kwargs.get("dropout", 0.1)
            return MLPAnchorModeling(hidden_size, dropout=dropout)
        else:  # "linear" (default)
            return LinearAnchorModeling(hidden_size)

    def forward(
        self,
        anchor_rep: torch.Tensor,
        child_rep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor_rep: (B, A, D) anchor representations
            child_rep: (B, C, D) child/class representations

        Returns:
            fused: (B, A, C, D) fused representations for scoring
        """
        raise NotImplementedError


class LinearAnchorModeling(AnchorModeling):
    """Linear projection of anchor + child concatenation."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, anchor_rep, child_rep):
        B, A, D = anchor_rep.shape
        C = child_rep.shape[1]

        # Expand and concatenate: (B, A, C, 2D) -> (B, A, C, D)
        anchor_exp = anchor_rep.unsqueeze(2).expand(B, A, C, D)
        child_exp = child_rep.unsqueeze(1).expand(B, A, C, D)
        combined = torch.cat([anchor_exp, child_exp], dim=-1)
        return self.proj(combined)

class MLPAnchorModeling(AnchorModeling):
    """MLP fusion of anchor + child representations."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = create_mlp(
            hidden_size * 2, [hidden_size * 2], hidden_size,
            dropout=dropout, activation="gelu",
        )

    def forward(self, anchor_rep, child_rep):
        B, A, D = anchor_rep.shape
        C = child_rep.shape[1]

        anchor_exp = anchor_rep.unsqueeze(2).expand(B, A, C, D)
        child_exp = child_rep.unsqueeze(1).expand(B, A, C, D)
        combined = torch.cat([anchor_exp, child_exp], dim=-1)
        return self.mlp(combined)
    
class LSTMAnchorModeling(AnchorModeling):
    """Recurrent processing of anchor-child pairs (GLiNER2-style).

    Uses child representations as h0 (initial hidden state) for the GRU,
    with anchor representations as the input sequence. This follows the
    GLiNER2 pattern where the child embedding conditions the recurrence.
    """

    def __init__(self, hidden_size: int, **kwargs):
        super().__init__()
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        self.proj_out = create_mlp(
            hidden_size * 2, [hidden_size * 2], hidden_size,
            dropout=0., activation="gelu",
        )

    def forward(self, anchor_rep, child_rep):
        B, A, D = anchor_rep.shape
        C = child_rep.shape[1]

        # h0: child embeddings as initial hidden state — (1, B*C, D)
        h0 = child_rep.reshape(B * C, D).unsqueeze(0)

        # Input sequence: anchors expanded over children — (A, B*C, D)
        # For each child, we run the GRU over the full anchor sequence
        anchor_seq = anchor_rep.unsqueeze(2).expand(B, A, C, D)
        anchor_seq = anchor_seq.permute(1, 0, 2, 3).reshape(A, B * C, D)

        gru_out, _ = self.gru(anchor_seq, h0)  # (A, B*C, D)

        # Concatenate GRU output with original child embeddings and project
        child_broadcast = child_rep.unsqueeze(0).expand(A, B, C, D).reshape(A, B * C, D)
        fused = self.proj_out(torch.cat([gru_out, child_broadcast], dim=-1))

        # Reshape back to (B, A, C, D)
        return fused.reshape(A, B, C, D).permute(1, 0, 2, 3)

