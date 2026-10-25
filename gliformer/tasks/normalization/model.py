"""Normalization head: reads normalized values (numbers, dates) from spans.

The head works on spans of the input text:

* in training, on the labelled spans the processor builds (gold value fields
  of the structuring data, plus parser spans when enabled);
* at inference, on every span the ``ner`` and ``structuring`` heads picked
  (``normalization_config.span_sources``).

Each span becomes one vector (``span_encoder``): ``pooled`` mixes the start,
end and mean word states; ``token_attention`` adds learned queries that read
the span's subword tokens.  Number heads (``number.heads``: regression,
bins, sci_digits) read a number from it; calendar classifiers read date, time
and datetime components, each with a "null" class for components that are
not written.  Numbers are decoded in the forward pass (in eval mode), since
the digit head decodes greedily with its own weights.  Spans without a target of some kind (a text span, or a number span
for the calendar classifiers) carry padded labels and add no loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from gliner.modeling.utils import extract_spans_from_tokens
from torch import nn

from .. import TaskHead, TaskHeadOutput
from ...layers.mlp import create_mlp
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


class SpanEncoder(nn.Module):
    """One vector per span from word (and optionally subword) states."""

    def __init__(
        self,
        hidden_size: int,
        dropout: float,
        *,
        mode: str = "pooled",
        queries: int = 4,
        heads: int = 8,
        max_tokens: int = 32,
    ):
        super().__init__()
        self.mode = mode
        self.max_tokens = int(max_tokens)
        self.pooled = create_mlp(
            input_dim=3 * hidden_size,
            intermediate_dims=[hidden_size],
            output_dim=hidden_size,
            dropout=dropout,
            activation="gelu",
        )
        if mode == "token_attention":
            if heads < 1 or hidden_size % heads:
                heads = 1
            self.queries = nn.Parameter(torch.randn(int(queries), hidden_size) * 0.02)
            self.attention = nn.MultiheadAttention(
                hidden_size, heads, dropout=dropout, batch_first=True,
            )
            self.token_proj = nn.Linear(int(queries) * hidden_size, hidden_size)
            self.norm = nn.LayerNorm(hidden_size)

    @staticmethod
    def _gather(states: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """``(B, W, D)`` at ``(B, S)`` positions -> ``(B, S, D)``."""

        return states.gather(1, index.unsqueeze(-1).expand(-1, -1, states.shape[-1]))

    def forward(self, words, word_mask, span_idx, *, token_embeds=None, token_words=None):
        width = words.shape[1]
        start = span_idx[..., 0].clamp(0, max(width - 1, 0))
        end = span_idx[..., 1].clamp(0, max(width - 1, 0))
        end = torch.maximum(end, start)
        masked = words * word_mask.unsqueeze(-1).to(words.dtype)
        cumulative = F.pad(masked.cumsum(dim=1), (0, 0, 1, 0))
        total = self._gather(cumulative, end + 1) - self._gather(cumulative, start)
        mean = total / (end - start + 1).unsqueeze(-1).to(words.dtype)
        vectors = self.pooled(torch.cat([self._gather(words, start), self._gather(words, end), mean], -1))
        if self.mode != "token_attention":
            return vectors
        if token_embeds is None or token_words is None or token_embeds.shape[0] != words.shape[0]:
            raise ValueError(
                "span_encoder='token_attention' needs subword states aligned with the text rows"
            )
        return self._attend_tokens(vectors, start, end, token_embeds, token_words)

    def _attend_tokens(self, vectors, start, end, token_embeds, token_words):
        batch, spans, dim = vectors.shape
        if spans == 0:
            return vectors
        length = token_embeds.shape[1]
        words = token_words.long()  # 1-based word id per subword, 0 elsewhere
        inside = (words[:, None, :] >= (start + 1)[..., None]) & (
            words[:, None, :] <= (end + 1)[..., None]
        )  # (B, S, T)
        keep = min(self.max_tokens, length)
        order = torch.arange(length, 0, -1, device=words.device, dtype=torch.float32)
        _, positions = (inside.float() * order).topk(keep, dim=-1)  # earliest first
        valid = inside.gather(-1, positions)
        tokens = token_embeds.gather(
            1, positions.reshape(batch, spans * keep, 1).expand(-1, -1, dim),
        ).reshape(batch * spans, keep, dim)
        padding = ~valid.reshape(batch * spans, keep)
        # A span whose subwords were all cut off still needs one key.
        padding[padding.all(dim=-1), 0] = False
        query = (self.queries[None].to(vectors.dtype) + vectors.reshape(batch * spans, 1, dim))
        attended, _ = self.attention(
            query, tokens.to(vectors.dtype), tokens.to(vectors.dtype),
            key_padding_mask=padding, need_weights=False,
        )
        mixed = self.token_proj(attended.reshape(batch, spans, -1))
        return self.norm(vectors + mixed)


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

    def forward(self, vectors: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.hidden(vectors)
        logits = {"year": self.year(hidden)}
        if self.year_digits:
            logits["year_digits"] = self.digits(hidden).unflatten(-1, (YEAR_DIGITS, 10))
        for name, layer in self.components.items():
            logits[name] = layer(hidden)
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
    dependencies = ["ner", "structuring"]

    def __init__(self, config, hidden_size: int, dropout: float):
        super().__init__()
        cfg = config.normalization_config
        self.cfg = cfg
        self.loss_coef = float(cfg.loss_coef)
        self.span_encoder = SpanEncoder(
            hidden_size,
            dropout,
            mode=cfg.span_encoder,
            queries=cfg.span_queries,
            heads=cfg.span_attention_heads,
            max_tokens=cfg.max_span_tokens,
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
        for name in self.cfg.span_sources:
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

    def forward(self, shared, dependency_outputs, flat_inputs=None, **batch):
        words = shared.words_embedding
        word_mask = shared.mask
        if word_mask is None:
            word_mask = torch.ones(words.shape[:2], dtype=torch.bool, device=words.device)
        word_mask = word_mask.bool()
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

        vectors = self.span_encoder(
            words,
            word_mask,
            span_idx,
            token_embeds=shared.token_embeds,
            token_words=batch.get("normalization_words_mask"),
        )
        temporal = self.temporal(vectors)
        extra["temporal"] = temporal
        number_outputs = {kind: head(vectors) for kind, head in self.number_heads.items()}

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
    "SpanEncoder",
    "TemporalClassifier",
    "component_targets",
    "year_digit_targets",
    "year_targets",
]
