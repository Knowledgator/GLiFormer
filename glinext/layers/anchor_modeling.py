"""Anchor modeling layers — post-anchor processing that fuses anchor + child representations.

After anchor acquisition (via AnchorLayer), the anchor_modeling layer controls how
anchor and child representations are combined before scoring. This is the configurable
"anchor modeling" step described in CLAUDE.md.

Strategies:
- LinearAnchorModeling: linear projection of concatenated anchor + child
- RNNAnchorModeling: recurrent processing (GLiNER2-style)
- MLPAnchorModeling: multi-layer perceptron fusion
- TransformerAnchorModeling: self-attention over anchor sequence conditioned on child reps
"""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .mlp import create_mlp


class AnchorModeling(nn.Module):
    """Base class for anchor modeling layers.

    Takes anchor reps (B, A, D) and child reps (B, C, D) and produces
    fused representations (B, A, C, D) ready for scoring.

    Uses registry-based polymorphism via __init_subclass__.
    """

    _registry: dict = {}
    config_fields: frozenset[str] = frozenset()

    def __init_subclass__(cls, modeling_type: str = "", **kwargs):
        super().__init_subclass__(**kwargs)
        if modeling_type:
            cls._registry[modeling_type] = cls

    @classmethod
    def strategy_class(cls, modeling_type: str) -> type["AnchorModeling"]:
        strategy = cls._registry.get(str(modeling_type))
        if strategy is None:
            raise ValueError(
                f"Unknown modeling_type: {modeling_type}. "
                f"Available: {sorted(cls._registry)}"
            )
        return strategy

    @classmethod
    def from_config(
        cls,
        modeling_type: str | Mapping[str, Any],
        hidden_size: int,
        **kwargs,
    ) -> "AnchorModeling":
        """Construct from a legacy name or strict ``{type, params}`` mapping."""

        strict = isinstance(modeling_type, Mapping)
        if strict:
            raw = dict(modeling_type)
            modeling_type = raw.pop("type", raw.pop("name", "linear"))
            configured_params = raw.pop("params", {})
            if not isinstance(configured_params, Mapping):
                raise TypeError("anchor modeling params must be a mapping")
            params = dict(configured_params)
            params.update(raw)
        else:
            params = {}
        strategy = cls.strategy_class(str(modeling_type))
        if strict:
            unknown = set(params) - set(strategy.config_fields)
            if unknown:
                raise ValueError(
                    f"Unsupported {modeling_type!r} anchor modeling options: "
                    f"{sorted(unknown)}. Available: "
                    f"{sorted(strategy.config_fields)}"
                )
            for name in strategy.config_fields:
                if name not in params and name in kwargs:
                    params[name] = kwargs[name]
            return strategy(hidden_size, **params)
        return strategy(hidden_size, **kwargs)

    @staticmethod
    def _mask_output(
        output: torch.Tensor,
        anchor_mask: torch.Tensor | None,
        child_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if anchor_mask is not None:
            output = output * anchor_mask[:, :, None, None].to(output.dtype)
        if child_mask is not None:
            output = output * child_mask[:, None, :, None].to(output.dtype)
        return output

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


class IdentityAnchorModeling(AnchorModeling, modeling_type="identity"):
    """Compatibility strategy that leaves child/label representations unchanged."""

    def __init__(self, hidden_size: int, **kwargs):
        super().__init__()

    def forward(
        self,
        anchor_rep,
        child_rep,
        *,
        anchor_mask=None,
        child_mask=None,
    ):
        batch_size, anchor_count, _ = anchor_rep.shape
        output = child_rep.unsqueeze(1).expand(
            batch_size,
            anchor_count,
            child_rep.shape[1],
            child_rep.shape[2],
        )
        return self._mask_output(output, anchor_mask, child_mask)


class LinearAnchorModeling(AnchorModeling, modeling_type="linear"):
    """Linear projection of anchor + child concatenation."""

    def __init__(self, hidden_size: int, **kwargs):
        super().__init__()
        self.proj = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        anchor_rep,
        child_rep,
        *,
        anchor_mask=None,
        child_mask=None,
    ):
        B, A, D = anchor_rep.shape
        C = child_rep.shape[1]
        if A == 0 or C == 0:
            return anchor_rep.new_zeros(B, A, C, D)

        # Expand and concatenate: (B, A, C, 2D) -> (B, A, C, D)
        anchor_exp = anchor_rep.unsqueeze(2).expand(B, A, C, D)
        child_exp = child_rep.unsqueeze(1).expand(B, A, C, D)
        combined = torch.cat([anchor_exp, child_exp], dim=-1)
        return self._mask_output(
            self.proj(combined),
            anchor_mask,
            child_mask,
        )

class MLPAnchorModeling(AnchorModeling, modeling_type="mlp"):
    """MLP fusion of anchor + child representations."""

    config_fields = frozenset({"dropout"})

    def __init__(self, hidden_size: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.mlp = create_mlp(
            hidden_size * 2, [hidden_size * 2], hidden_size,
            dropout=dropout, activation="gelu",
        )

    def forward(
        self,
        anchor_rep,
        child_rep,
        *,
        anchor_mask=None,
        child_mask=None,
    ):
        B, A, D = anchor_rep.shape
        C = child_rep.shape[1]

        anchor_exp = anchor_rep.unsqueeze(2).expand(B, A, C, D)
        child_exp = child_rep.unsqueeze(1).expand(B, A, C, D)
        combined = torch.cat([anchor_exp, child_exp], dim=-1)
        return self._mask_output(
            self.mlp(combined),
            anchor_mask,
            child_mask,
        )
    
class RNNAnchorModeling(AnchorModeling, modeling_type="rnn"):
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

    def forward(
        self,
        anchor_rep,
        child_rep,
        *,
        anchor_mask=None,
        child_mask=None,
    ):
        B, A, D = anchor_rep.shape
        C = child_rep.shape[1]
        if A == 0 or C == 0:
            return anchor_rep.new_zeros(B, A, C, D)

        # h0: child embeddings as initial hidden state — (1, B*C, D)
        h0 = child_rep.reshape(B * C, D).unsqueeze(0)

        # Input sequence: anchors expanded over children — (A, B*C, D)
        # For each child, we run the GRU over the full anchor sequence
        anchor_seq = anchor_rep.unsqueeze(2).expand(B, A, C, D)
        anchor_seq = anchor_seq.permute(1, 0, 2, 3).reshape(A, B * C, D)

        if anchor_mask is None:
            gru_out, _ = self.gru(anchor_seq, h0)  # (A, B*C, D)
        else:
            if anchor_mask.shape != (B, A):
                raise ValueError("anchor_mask must have shape (B, A)")
            hidden = h0
            outputs = []
            expanded_mask = anchor_mask.bool()[:, :, None].expand(B, A, C)
            expanded_mask = expanded_mask.permute(1, 0, 2).reshape(A, B * C)
            for anchor_index in range(A):
                candidate, candidate_hidden = self.gru(
                    anchor_seq[anchor_index : anchor_index + 1],
                    hidden,
                )
                valid = expanded_mask[anchor_index].view(1, B * C, 1)
                hidden = torch.where(valid, candidate_hidden, hidden)
                outputs.append(
                    torch.where(valid, candidate, torch.zeros_like(candidate))
                )
            gru_out = torch.cat(outputs, dim=0) if outputs else anchor_seq

        # Concatenate GRU output with original child embeddings and project
        child_broadcast = child_rep.unsqueeze(0).expand(A, B, C, D).reshape(A, B * C, D)
        fused = self.proj_out(torch.cat([gru_out, child_broadcast], dim=-1))

        # Reshape back to (B, A, C, D)
        output = fused.reshape(A, B, C, D).permute(1, 0, 2, 3)
        return self._mask_output(output, anchor_mask, child_mask)


class TransformerAnchorModeling(AnchorModeling, modeling_type="transformer"):
    """Transformer-based processing of anchor-child pairs.

    Uses child representations as conditioning: concatenates child embedding
    to each anchor position, applies self-attention over the anchor sequence,
    then projects back. Analogous to RNNAnchorModeling but replaces the GRU
    with a transformer encoder for parallel, attention-based fusion.
    """

    config_fields = frozenset({"num_layers", "num_heads", "dropout"})

    def __init__(self, hidden_size: int, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1, **kwargs):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj_in = nn.Linear(hidden_size * 2, hidden_size)
        self.proj_out = create_mlp(
            hidden_size * 2, [hidden_size * 2], hidden_size,
            dropout=0., activation="gelu",
        )

    def forward(
        self,
        anchor_rep,
        child_rep,
        *,
        anchor_mask=None,
        child_mask=None,
    ):
        B, A, D = anchor_rep.shape
        C = child_rep.shape[1]
        if A == 0 or C == 0:
            return anchor_rep.new_zeros(B, A, C, D)

        # Expand anchor and child: (B, A, C, D)
        anchor_exp = anchor_rep.unsqueeze(2).expand(B, A, C, D)
        child_exp = child_rep.unsqueeze(1).expand(B, A, C, D)

        # Concatenate and project to D: (B, A, C, 2D) -> (B, A, C, D)
        combined = self.proj_in(torch.cat([anchor_exp, child_exp], dim=-1))

        # Reshape to run transformer over anchor dim per child: (B*C, A, D)
        combined = combined.permute(0, 2, 1, 3).reshape(B * C, A, D)

        # Self-attention over anchor sequence
        padding_mask = None
        if anchor_mask is not None:
            if anchor_mask.shape != (B, A):
                raise ValueError("anchor_mask must have shape (B, A)")
            safe_mask = anchor_mask.bool().clone()
            all_invalid = ~safe_mask.any(dim=1)
            if all_invalid.any() and A:
                safe_mask[all_invalid, 0] = True
            padding_mask = ~safe_mask[:, None, :].expand(B, C, A).reshape(
                B * C,
                A,
            )
        trans_out = self.transformer(
            combined,
            src_key_padding_mask=padding_mask,
        )  # (B*C, A, D)

        # Concatenate with original child embeddings and project
        child_broadcast = child_rep.unsqueeze(2).expand(B, C, A, D).reshape(B * C, A, D)
        fused = self.proj_out(torch.cat([trans_out, child_broadcast], dim=-1))  # (B*C, A, D)

        # Reshape back to (B, A, C, D)
        output = fused.reshape(B, C, A, D).permute(0, 2, 1, 3)
        return self._mask_output(output, anchor_mask, child_mask)
