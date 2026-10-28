"""Normalization head: reads normalized values (numbers, dates) from spans.

The head works on spans of the input text:

* in training, on the labelled spans the processor builds (gold value fields
  of the structuring data, plus parser spans when enabled);
* at inference, on every span the ``ner`` and ``structuring`` heads picked
  (``normalization_config.span_sources``).

Each span becomes one vector through a shared span pooler
(``span_encoder``: ``marker``, ``mean`` or ``tokens``).  With ``token_queries`` every output (the number, each
significant digit, each calendar component) has its own learned query that
reads the span's subword states (kept by the ``tokens`` pooler).  Number heads (``number.heads``: regression,
bins, sci_digits) read a number from it; calendar classifiers read date, time
and datetime components, each with a "null" class for components that are
not written.  Numbers are decoded in the forward pass (in eval mode), since
the digit head decodes greedily with its own weights.  Spans without a target of some kind (a text span, or a number span
for the calendar classifiers) carry padded labels and add no loss.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from gliner.modeling.utils import extract_spans_from_tokens
from torch import nn

from ...layers.mlp import create_mlp
from ...layers.span_pooling import SpanTokenSource, build_span_pooler, gather_span_tokens
from .. import TaskHead, TaskHeadOutput, text_token_states
from .numbers import build_number_heads
from .values import IGNORE, NULL, YEAR_DIGITS

# Classes per calendar component: the values, then "null".
MONTH_CLASSES = 13
DAY_CLASSES = 32
HOUR_CLASSES = 25
MINUTE_CLASSES = 61
SECOND_CLASSES = 61
# Value offsets: months and days are written from 1, clock parts from 0.
_COMPONENT_LAYOUT = {
    "month": (MONTH_CLASSES, 1),
    "day": (DAY_CLASSES, 1),
    "hour": (HOUR_CLASSES, 0),
    "minute": (MINUTE_CLASSES, 0),
    "second": (SECOND_CLASSES, 0),
}


def component_targets(codes: torch.Tensor, classes: int, offset: int) -> torch.Tensor:
    """Class indices for component codes: value - offset, null last, else ignored."""

    targets = torch.full_like(codes, IGNORE)
    written = codes >= 0
    targets = torch.where(written, codes - offset, targets)
    targets = torch.where(codes == NULL, torch.full_like(codes, classes - 1), targets)
    # Out-of-range values (bad labels) are not trained.
    bad = written & ((codes - offset < 0) | (codes - offset >= classes - 1))
    return targets.masked_fill(bad, IGNORE)


def year_targets(codes: torch.Tensor, first: int, last: int) -> torch.Tensor:
    """Year classes: in-range offset, then out of range, then null."""

    span = last - first + 1
    targets = torch.full_like(codes, IGNORE)
    written = codes >= 0
    in_range = written & (codes >= first) & (codes <= last)
    targets = torch.where(in_range, codes - first, targets)
    targets = torch.where(written & ~in_range, torch.full_like(codes, span), targets)
    return torch.where(codes == NULL, torch.full_like(codes, span + 1), targets)


def year_digit_targets(codes: torch.Tensor) -> torch.Tensor:
    """Four digit classes of written years 1-9999 (else ignored)."""

    written = (codes >= 1) & (codes < 10 ** YEAR_DIGITS)
    magnitude = codes.clamp(min=0, max=10 ** YEAR_DIGITS - 1)
    powers = [10 ** power for power in range(YEAR_DIGITS - 1, -1, -1)]
    digits = torch.stack([magnitude // power % 10 for power in powers], dim=-1)
    return digits.masked_fill(~written.unsqueeze(-1), IGNORE)


def _masked_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor | None:
    """Mean cross-entropy over targets that are not ``IGNORE``; None if none."""

    flat = targets.reshape(-1)
    if not (flat != IGNORE).any():
        return None
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), flat, ignore_index=IGNORE,
    )


class SpanTokenQueries(nn.Module):
    """One learned query per output, reading the span's subword states.

    Each query starts as its learned embedding plus the span vector, then
    goes through pre-norm transformer decoder layers: self-attention among
    the span's queries (the digits see each other) and cross-attention over
    the span's subwords. Returns ``(B, S, Q, D)``, normalized.
    """

    def __init__(self, hidden_size: int, dropout: float, names, *, heads: int = 8, layers: int = 2):
        super().__init__()
        if heads < 1 or hidden_size % heads:
            heads = 1
        self.names = list(names)
        self.index = {name: slot for slot, name in enumerate(self.names)}
        self.queries = nn.Parameter(torch.randn(len(self.names), hidden_size) * 0.02)
        self.token_norm = nn.LayerNorm(hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                hidden_size, heads, dim_feedforward=2 * hidden_size, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(int(layers))
        ])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, vectors, tokens, token_mask):
        batch, spans, dim = vectors.shape
        keep = tokens.shape[2]
        hidden = self.queries[None].to(vectors.dtype) + vectors.reshape(batch * spans, 1, dim)
        memory = self.token_norm(tokens.reshape(batch * spans, keep, dim).to(vectors.dtype))
        padding = ~token_mask.reshape(batch * spans, keep)
        for layer in self.layers:
            hidden = layer(hidden, memory, memory_key_padding_mask=padding)
        return self.norm(hidden).reshape(batch, spans, len(self.names), dim)

    def take(self, reads, name):
        return reads[..., self.index[name], :]

    def take_prefix(self, reads, prefix):
        slots = [slot for name, slot in self.index.items() if name.startswith(prefix)]
        return reads[..., slots, :] if slots else None


class TemporalClassifier(nn.Module):
    """Year, month, day, hour, minute and second classes per span."""

    def __init__(self, hidden_size: int, dropout: float, *, year_bounds, year_digits: bool):
        super().__init__()
        self.first, self.last = (int(value) for value in year_bounds)
        self.year_span = max(self.last - self.first + 1, 0)
        self.hidden = create_mlp(
            input_dim=hidden_size,
            intermediate_dims=[hidden_size],
            output_dim=hidden_size,
            dropout=dropout,
            activation="gelu",
        )
        # In-range years, "out of range", "null".
        self.year = nn.Linear(hidden_size, self.year_span + 2)
        # Four digit classes read years outside the class range.
        self.year_digits = bool(year_digits)
        if self.year_digits:
            self.digits = nn.Linear(hidden_size, YEAR_DIGITS * 10)
        self.components = nn.ModuleDict({
            name: nn.Linear(hidden_size, classes) for name, (classes, _) in _COMPONENT_LAYOUT.items()
        })

    def forward(self, vectors: torch.Tensor, parts: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """``parts`` (token queries) gives each component its own vector."""

        if parts is None:
            hidden = self.hidden(vectors)
            per_part = dict.fromkeys(("year", *self.components), hidden)
        else:
            per_part = {name: self.hidden(parts[name]) for name in ("year", *self.components)}
        logits = {"year": self.year(per_part["year"])}
        if self.year_digits:
            logits["year_digits"] = self.digits(per_part["year"]).unflatten(-1, (YEAR_DIGITS, 10))
        for name, layer in self.components.items():
            logits[name] = layer(per_part[name])
        return logits

    def loss(self, logits: dict[str, torch.Tensor], codes: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Mean over components that have targets in this batch."""

        terms = [_masked_ce(logits["year"], year_targets(codes["year"], self.first, self.last))]
        if self.year_digits:
            terms.append(_masked_ce(logits["year_digits"], year_digit_targets(codes["year"])))
        for name, (classes, offset) in _COMPONENT_LAYOUT.items():
            terms.append(_masked_ce(logits[name], component_targets(codes[name], classes, offset)))
        terms = [term for term in terms if term is not None]
        if not terms:
            return None
        return torch.stack(terms).mean()


class NormalizationHead(TaskHead):
    """Reads numbers and calendar components from spans."""

    name = "normalization"
    dependencies = ["ner", "structuring", "calculation"]

    def __init__(self, config, hidden_size: int, dropout: float):
        super().__init__()
        cfg = config.normalization_config
        self.cfg = cfg
        self.loss_coef = float(cfg.loss_coef)
        self.span_pooler = build_span_pooler(
            SimpleNamespace(
                span_pooler=cfg.span_encoder,
                span_max_tokens=cfg.max_span_tokens,
                span_pooler_queries=cfg.span_queries,
                span_pooler_heads=cfg.span_attention_heads,
            ),
            hidden_size,
            dropout,
            max_width=cfg.max_span_width,
        )
        self.token_queries = None
        if cfg.token_queries:
            names = ["number", "year", *_COMPONENT_LAYOUT]
            if "sci_digits" in cfg.number.heads:
                names += [f"digit_{position}" for position in range(cfg.number.sci_digits.digits)]
            self.token_queries = SpanTokenQueries(
                hidden_size,
                dropout,
                names,
                heads=cfg.span_attention_heads,
                layers=cfg.token_query_layers,
            )
        self.temporal = TemporalClassifier(
            hidden_size,
            dropout,
            year_bounds=cfg.temporal.year_bounds(),
            year_digits=cfg.temporal.year_digits,
        )
        self.temporal_loss_coef = float(cfg.temporal.loss_coef)
        self.number_heads = build_number_heads(hidden_size, dropout, cfg.number)
        self.number_loss_coef = float(cfg.number.loss_coef)

    @classmethod
    def from_config(cls, config, **kwargs):
        if getattr(config, "normalization_config", None) is None:
            return None
        return cls(config, hidden_size=config.hidden_size, dropout=config.dropout)

    # -- inference spans ---------------------------------------------------

    def _source_spans(self, output, name, threshold):
        """``(span_idx, span_mask, batch_origin)`` of one dependency, or None."""

        if output is None:
            return None
        extra = output.extra or {}
        origin = extra.get("batch_origin")
        if name == "structuring":
            spans, mask = extra.get("entity_spans"), extra.get("entity_mask")
        elif name == "calculation":
            # Operand spans of every calculation query, rows = text rows.
            spans, mask = extra.get("operand_spans"), extra.get("operand_mask")
        else:
            if output.logits is None or output.logits.dim() != 4:
                return None
            spans, mask = extract_spans_from_tokens(output.logits, labels=None, threshold=threshold)
        if spans is None or mask is None:
            return None
        if origin is None:
            origin = torch.arange(spans.shape[0], device=spans.device)
        return spans, mask.bool(), origin

    def collect_spans(self, dependency_outputs, batch_size, width, threshold, device):
        """Unique spans per text row from the configured source heads."""

        rows: list[dict[tuple[int, int], None]] = [{} for _ in range(batch_size)]
        sources = list(self.cfg.span_sources)
        if "calculation" in dependency_outputs and "calculation" not in sources:
            # Calculation operands always need values.
            sources.append("calculation")
        for name in sources:
            found = self._source_spans(dependency_outputs.get(name), name, threshold)
            if found is None:
                continue
            spans, mask, origin = found
            for flat, row in enumerate(origin.tolist()):
                if not 0 <= row < batch_size:
                    continue
                for (start, end), keep in zip(spans[flat].tolist(), mask[flat].tolist()):
                    if not keep or start < 0 or end < start or end >= width:
                        continue
                    if end - start + 1 > self.cfg.max_span_width:
                        continue
                    if len(rows[row]) < self.cfg.max_inference_spans:
                        rows[row][(start, end)] = None
        count = max((len(row) for row in rows), default=0)
        span_idx = torch.zeros(batch_size, count, 2, dtype=torch.long, device=device)
        span_mask = torch.zeros(batch_size, count, dtype=torch.bool, device=device)
        for row, spans in enumerate(rows):
            for slot, (start, end) in enumerate(sorted(spans)):
                span_idx[row, slot, 0], span_idx[row, slot, 1] = start, end
                span_mask[row, slot] = True
        return span_idx, span_mask

    # -- forward -----------------------------------------------------------

    def _unused_parameters_loss(self, reference: torch.Tensor) -> torch.Tensor:
        # Keeps every parameter in the graph for DDP when a batch has no target.
        return sum(parameter.sum() for parameter in self.parameters()) * 0.0 + reference.sum() * 0.0

    @staticmethod
    def _token_source(shared, words, batch):
        """Subword states of the text rows, or ``None`` (the unit states are used)."""

        tokens = text_token_states(shared) if getattr(shared, "token_embeds", None) is not None else None
        token_words = batch.get("span_token_words")
        if tokens is None or token_words is None or tokens.dim() != 3:
            return None
        if tokens.shape[0] != words.shape[0] or token_words.shape != tokens.shape[:2]:
            return None
        return SpanTokenSource(tokens, token_words.to(tokens.device))

    def _encode_spans(self, shared, words, span_idx, batch):
        """Span vectors, plus the span's subword states when the pooler keeps them."""

        source = self._token_source(shared, words, batch) if self.span_pooler.uses_tokens else None
        output = self.span_pooler.encode(words, span_idx, token_source=source)
        return output.vectors, output.tokens, output.token_mask

    def forward(self, shared, dependency_outputs, flat_inputs=None, **batch):
        words = shared.words_embedding
        batch_size, width = words.shape[:2]
        span_idx = batch.get("normalization_span_idx")
        supervised = span_idx is not None
        if supervised:
            span_idx = span_idx.to(words.device).long()
            span_mask = batch["normalization_span_mask"].to(words.device).bool()
        else:
            threshold = batch.get("threshold")
            span_idx, span_mask = self.collect_spans(
                dependency_outputs or {}, batch_size, width,
                0.5 if threshold is None else float(threshold), words.device,
            )
        extra = {"span_idx": span_idx, "span_mask": span_mask}
        if span_idx.shape[1] == 0:
            loss = self._unused_parameters_loss(words) if (self.training and supervised) else None
            return TaskHeadOutput(loss=loss, extra=extra)

        vectors, tokens, token_mask = self._encode_spans(shared, words, span_idx, batch)
        parts = number_vectors = digit_vectors = None
        if self.token_queries is not None:
            if tokens is None:
                tokens, token_mask = gather_span_tokens(
                    words, span_idx, self.cfg.max_span_tokens, self._token_source(shared, words, batch),
                )
            reads = self.token_queries(vectors, tokens, token_mask)
            parts = {name: self.token_queries.take(reads, name) for name in ("year", *_COMPONENT_LAYOUT)}
            number_vectors = self.token_queries.take(reads, "number")
            digit_vectors = self.token_queries.take_prefix(reads, "digit_")
        temporal = self.temporal(vectors, parts)
        extra["temporal"] = temporal
        number_outputs = {
            kind: head(
                vectors if number_vectors is None else number_vectors,
                **({"digit_vectors": digit_vectors} if kind == "sci_digits" and digit_vectors is not None else {}),
            )
            for kind, head in self.number_heads.items()
        }

        loss = None
        if supervised:
            codes = {
                name: batch[f"normalization_{name}"].to(words.device).long().masked_fill(~span_mask, IGNORE)
                for name in ("year", "month", "day", "hour", "minute", "second")
            }
            temporal_loss = self.temporal.loss(temporal, codes)
            if temporal_loss is not None:
                extra["temporal_loss"] = temporal_loss.detach()
                loss = self.temporal_loss_coef * temporal_loss
            number_values = batch.get("normalization_number_value")
            if number_values is not None and number_outputs:
                number_values = number_values.to(words.device, torch.float64)
                number_mask = batch["normalization_number_mask"].to(words.device).bool() & span_mask
                for kind, head in self.number_heads.items():
                    number_loss, stats = head.loss(number_outputs[kind], number_values, number_mask)
                    for name, value in stats.items():
                        extra[f"number_{kind}_{name}"] = value
                    if number_loss is not None:
                        extra[f"number_{kind}_loss"] = number_loss.detach()
                        weighted = self.number_loss_coef * number_loss
                        loss = weighted if loss is None else loss + weighted
            if self.training:
                unused = self._unused_parameters_loss(vectors)
                loss = unused if loss is None else loss + unused
        if not self.training and number_outputs:
            values, confidence = {}, {}
            for kind, head in self.number_heads.items():
                values[kind], confidence[kind] = head.decode(number_outputs[kind])
            extra["number"] = values
            extra["number_confidence"] = confidence
        return TaskHeadOutput(loss=loss, extra=extra)


__all__ = [
    "NormalizationHead",
    "SpanTokenQueries",
    "TemporalClassifier",
    "component_targets",
    "year_digit_targets",
    "year_targets",
]
