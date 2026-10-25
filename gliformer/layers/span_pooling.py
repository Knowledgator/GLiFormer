"""Span poolers: turn ``[start, end]`` text-unit spans into vectors.

Spans index text units (words, or subword tokens with ``text_unit="token"``).
Three poolers share one call signature:

* ``marker`` (default): MLPs on the start and end unit states, concatenated
  and projected. Same module and parameter names as GLiNER's
  ``SpanRepLayer(span_mode="token_level")``, so existing checkpoints load.
* ``mean``: mean of the span's subword states (or unit states when no subword
  map is given), then the marker's projection shape (MLP, ReLU, MLP), which
  keeps its output on the marker's scale.
* ``tokens``: keeps the span's subword states (the first ``max_tokens``).
  ``encode`` returns them with a mask so later layers can cross-attend over
  them; the span vector is the marker vector plus what learned queries read
  from those tokens, scaled by a learned gate that starts at 0. A fresh
  ``tokens`` pooler therefore outputs exactly the marker vector, and the
  token part grows gradually; heads score spans with raw dot products, so
  a sudden change of scale saturates their logits.

Subword states come from a :class:`SpanTokenSource`: the encoder output and a
map from each subword to its 1-based unit id (0 for prompt, special and
padding tokens). The map must mark every subword of a word, not only the first
one (see ``token_words_mask`` in the processor). Without a source, the unit
states themselves are used as the tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from gliner.modeling.layers import create_projection_layer
from gliner.modeling.span_rep import SpanRepLayer
from torch import nn

SPAN_POOLERS = ("marker", "mean", "tokens")
# Poolers that read subword states (and so need the full subword map).
TOKEN_POOLERS = frozenset({"mean", "tokens"})


@dataclass
class SpanTokenSource:
    """Subword states for the span rows.

    ``tokens`` is ``(B_enc, T, D)``; ``token_units`` is ``(B_enc, T)`` with the
    1-based unit id of each subword. ``rows`` (``(B,)``) maps each span row to
    its encoder row; ``None`` means row ``b`` reads encoder row ``b``.
    """

    tokens: torch.Tensor
    token_units: torch.Tensor
    rows: Optional[torch.Tensor] = None


@dataclass
class SpanPoolerOutput:
    vectors: torch.Tensor  # (B, S, D)
    tokens: Optional[torch.Tensor] = None  # (B, S, K, D), "tokens" pooler only
    token_mask: Optional[torch.Tensor] = None  # (B, S, K) True = real token


def _unit_source(units: torch.Tensor) -> SpanTokenSource:
    """Use the unit states as tokens (unit ``i`` has id ``i + 1``)."""

    ids = torch.arange(1, units.shape[1] + 1, device=units.device)
    return SpanTokenSource(units, ids.unsqueeze(0).expand(units.shape[0], -1))


def _span_bounds(span_idx: torch.Tensor, width: int):
    start = span_idx[..., 0].clamp(0, max(width - 1, 0))
    end = torch.maximum(span_idx[..., 1].clamp(0, max(width - 1, 0)), start)
    return start, end


def _inside(source: SpanTokenSource, start, end) -> torch.Tensor:
    """``(B, S, T)``: subword ``t`` belongs to span ``s``."""

    units = source.token_units
    if source.rows is not None:
        units = units.index_select(0, source.rows.to(units.device))
    units = units.long()
    return (units[:, None, :] >= (start + 1)[..., None]) & (
        units[:, None, :] <= (end + 1)[..., None]
    )


def _row_tokens(source: SpanTokenSource) -> torch.Tensor:
    if source.rows is None:
        return source.tokens
    return source.tokens.index_select(0, source.rows.to(source.tokens.device))


def gather_span_tokens(
    units: torch.Tensor,
    span_idx: torch.Tensor,
    max_tokens: int,
    source: Optional[SpanTokenSource] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The first ``max_tokens`` subword states of each span.

    Returns ``(B, S, K, D)`` states and a ``(B, S, K)`` mask. Every span keeps
    at least one unmasked slot (its first token, or position 0 when the span
    has no subwords left after truncation) so attention over it is defined.
    """

    source = source or _unit_source(units)
    start, end = _span_bounds(span_idx, units.shape[1])
    inside = _inside(source, start, end)
    batch, spans, length = inside.shape
    dim = source.tokens.shape[-1]
    keep = max(1, min(int(max_tokens), length))
    if spans == 0 or length == 0:
        empty = source.tokens.new_zeros(batch, spans, keep, dim)
        return empty, torch.zeros(batch, spans, keep, dtype=torch.bool, device=units.device)
    # Rank inside positions by order of appearance, earliest first.
    order = torch.arange(length, 0, -1, device=inside.device, dtype=torch.float32)
    _, positions = (inside.float() * order).topk(keep, dim=-1)
    mask = inside.gather(-1, positions)
    if source.rows is None:
        flat_rows = torch.arange(batch, device=positions.device)
    else:
        flat_rows = source.rows.to(positions.device)
    flat = (flat_rows[:, None, None] * source.tokens.shape[1] + positions).reshape(-1)
    tokens = source.tokens.reshape(-1, dim).index_select(0, flat).reshape(batch, spans, keep, dim)
    empty = ~mask.any(dim=-1)
    mask[..., 0] = mask[..., 0] | empty
    return tokens.to(units.dtype), mask


class _PoolerMixin:
    pooler: str = "marker"
    uses_tokens: bool = False

    def encode(self, units, span_idx, *, token_source=None) -> SpanPoolerOutput:
        return SpanPoolerOutput(self(units, span_idx, token_source=token_source))


class MarkerSpanPooler(_PoolerMixin, SpanRepLayer):
    """Start/end marker; state-dict compatible with ``SpanRepLayer(token_level)``."""

    pooler = "marker"

    def __init__(self, hidden_size: int, dropout: float, max_width: int = 12, **_):
        super().__init__(
            hidden_size=hidden_size,
            max_width=max_width,
            span_mode="token_level",
            dropout=dropout,
        )

    def forward(self, units, span_idx, word_mask=None, *, token_source=None, **kwargs):
        return super().forward(units, span_idx, word_mask, **kwargs)


class MeanSpanPooler(_PoolerMixin, nn.Module):
    """Mean of the span's subword (or unit) states, then a projection."""

    pooler = "mean"
    uses_tokens = True

    def __init__(self, hidden_size: int, dropout: float, **_):
        super().__init__()
        self.project = create_projection_layer(hidden_size, dropout)
        self.out_project = create_projection_layer(hidden_size, dropout)

    def forward(self, units, span_idx, word_mask=None, *, token_source=None, **_):
        source = token_source or _unit_source(units)
        start, end = _span_bounds(span_idx, units.shape[1])
        inside = _inside(source, start, end)
        tokens = _row_tokens(source)
        weights = inside.to(tokens.dtype)
        total = torch.bmm(weights, tokens)
        mean = total / weights.sum(-1, keepdim=True).clamp(min=1.0)
        return self.out_project(self.project(mean.to(units.dtype)).relu())


class TokensSpanPooler(_PoolerMixin, nn.Module):
    """Keeps the span's subword states; queries attend over them for a vector."""

    pooler = "tokens"
    uses_tokens = True

    def __init__(
        self,
        hidden_size: int,
        dropout: float,
        *,
        max_tokens: int = 32,
        queries: int = 4,
        heads: int = 8,
        **_,
    ):
        super().__init__()
        if heads < 1 or hidden_size % heads:
            heads = 1
        self.max_tokens = int(max_tokens)
        self.marker = MarkerSpanPooler(hidden_size, dropout)
        self.queries = nn.Parameter(torch.randn(int(queries), hidden_size) * 0.02)
        self.attention = nn.MultiheadAttention(hidden_size, heads, dropout=dropout, batch_first=True)
        self.token_norm = nn.LayerNorm(hidden_size)
        self.token_proj = nn.Linear(int(queries) * hidden_size, hidden_size)
        self.token_gate = nn.Parameter(torch.zeros(()))

    def encode(self, units, span_idx, *, token_source=None) -> SpanPoolerOutput:
        base = self.marker(units, span_idx)
        tokens, mask = gather_span_tokens(units, span_idx, self.max_tokens, token_source)
        batch, spans, keep, dim = tokens.shape
        if spans == 0:
            return SpanPoolerOutput(base, tokens, mask)
        query = self.queries[None].to(base.dtype) + base.reshape(batch * spans, 1, dim)
        keys = self.token_norm(tokens.reshape(batch * spans, keep, dim))
        attended, _ = self.attention(
            query,
            keys,
            keys,
            key_padding_mask=~mask.reshape(batch * spans, keep),
            need_weights=False,
        )
        mixed = self.token_proj(attended.reshape(batch, spans, -1))
        return SpanPoolerOutput(base + self.token_gate.to(base.dtype) * mixed, tokens, mask)

    def forward(self, units, span_idx, word_mask=None, *, token_source=None, **_):
        return self.encode(units, span_idx, token_source=token_source).vectors


_POOLERS = {
    "marker": MarkerSpanPooler,
    "mean": MeanSpanPooler,
    "tokens": TokensSpanPooler,
}


def build_span_pooler(task_cfg, hidden_size: int, dropout: float, max_width: int = 12):
    """The pooler a head config asks for (``marker`` when it names none)."""

    name = str(getattr(task_cfg, "span_pooler", "marker") or "marker").lower()
    if name not in _POOLERS:
        raise ValueError(f"span_pooler must be one of {list(SPAN_POOLERS)}, got {name!r}")
    return _POOLERS[name](
        hidden_size,
        dropout,
        max_width=max_width,
        max_tokens=getattr(task_cfg, "span_max_tokens", 32),
        queries=getattr(task_cfg, "span_pooler_queries", 4),
        heads=getattr(task_cfg, "span_pooler_heads", 8),
    )


SPAN_HEAD_CONFIGS = ("ner_config", "structuring_config", "joint_relex_config", "open_relex_config")


def config_needs_token_map(config) -> bool:
    """A head reads subword states, so the processor must map every subword."""

    for attr in SPAN_HEAD_CONFIGS:
        head_cfg = getattr(config, attr, None)
        if head_cfg is not None and str(getattr(head_cfg, "span_pooler", "marker")).lower() in TOKEN_POOLERS:
            return True
    normalization = getattr(config, "normalization_config", None)
    return normalization is not None and getattr(normalization, "span_encoder", "pooled") == "token_attention"
