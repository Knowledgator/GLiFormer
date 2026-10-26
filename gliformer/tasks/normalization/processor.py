"""Training targets of the normalization head.

Labelled spans come from two sources:

* gold value fields of the structuring data: number, duration, date, time
  and datetime fields (and lists of them) whose normalized gold value was
  grounded on a span that converts to it exactly;
* the rule-based parser, when ``normalization_config.parser.enabled``: every
  value it finds in the text becomes a span with a silver label.

A gold label wins over a silver one on the same span; a silver label still
fills a target the gold label does not have (a year written alone is both a
number and a year-only date).

Label tensors, per text row (``B`` rows, ``S`` spans):

* ``normalization_span_idx`` ``(B, S, 2)``: inclusive word start/end;
* ``normalization_span_mask`` ``(B, S)``;
* ``normalization_number_value`` ``(B, S)`` float64 and
  ``normalization_number_mask`` ``(B, S)``;
* ``normalization_temporal_family`` ``(B, S)``: date/time/datetime, or -1;
* ``normalization_{year,month,day,hour,minute,second}`` ``(B, S)``:
  component codes (``values.NULL`` not written, ``values.IGNORE`` no target).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


import torch

from .. import TaskProcessor
from ...processing.token_units import is_token_level
from ...processing._structuring_hierarchy import (
    TYPED_ANCHORS_KEY,
    is_internal_instance_key,
)
from .parser import NormalizationParser, gold_parts
from .values import (
    COMPONENTS,
    FAMILY_NONE,
    FAMILY_NUMBER,
    IGNORE,
    TEMPORAL_FAMILIES,
    VALUE_TYPE_FAMILY,
    component_codes,
    is_iso_value,
)

@dataclass
class SpanTarget:
    """Targets of one labelled span (either part may be missing)."""

    start: int
    end: int
    number: float | None = None
    family: int = FAMILY_NONE
    codes: dict[str, int] = field(default_factory=lambda: dict.fromkeys(COMPONENTS, IGNORE))

    def merge_missing(self, other: "SpanTarget") -> None:
        """Fill the targets this span lacks from ``other``."""

        if self.number is None and other.number is not None:
            self.number = other.number
        if self.family == FAMILY_NONE and other.family != FAMILY_NONE:
            self.family = other.family
            self.codes = dict(other.codes)


class NormalizationProcessor(TaskProcessor):
    """Builds normalization label tensors from structuring gold and the parser."""

    task_name = "normalization"
    tensor_prefix = "normalization"

    def __init__(self, config, tokenizer=None, words_splitter=None, **kwargs):
        super().__init__(config, tokenizer, words_splitter, **kwargs)
        self.task_config = config.normalization_config
        if self.task_config is None:
            raise ValueError("normalization_config is required for normalization")
        parser_cfg = self.task_config.parser
        self.parser = NormalizationParser.from_config(parser_cfg) if parser_cfg.enabled else None
        self.words_splitter = words_splitter
        # The parser reads words. In token mode it runs on the configured
        # word splitter and its spans are mapped onto the token units.
        self.word_splitter = None
        if is_token_level(config):
            from gliner.data_processing import WordsSplitter

            self.word_splitter = WordsSplitter(config.words_splitter_type)
        # Set by the data processor: the structuring processor whose resolved
        # records hold the gold value spans.
        self.structuring_processor = None
        # Set by the data processor when the model has a calculation head:
        # gold operand spans of calculation programs are number targets.
        self.calculation_processor = None

    def get_classes_mapping(self, batch_list, **kwargs):
        return None

    # -- spans -----------------------------------------------------------------

    def _words(self, item) -> list[str]:
        structuring = self.structuring_processor
        if structuring is not None:
            _spans, words = structuring._tokenize_text(item)
            if words is not None:
                return list(words)
        return list(item.get("tokenized_text") or ())

    def _gold_targets(self, item) -> list[SpanTarget]:
        structuring = self.structuring_processor
        if structuring is None or not getattr(structuring, "typed_enabled", False):
            return []
        data = item.get(structuring.data_key)
        if not isinstance(data, dict):
            return []
        meta = item.get(structuring.meta_key) or {}
        names = {group.get("data_key"): group.get("name") for group in meta.get("groups", [])}
        targets = []
        for data_key, instances in data.items():
            if not isinstance(instances, list):
                continue
            raw_types = structuring._raw_group_types(item, data_key, names.get(data_key, data_key))
            specs = {label: structuring._spec(raw) for label, raw in raw_types.items()}
            for instance in instances:
                if not isinstance(instance, dict):
                    continue
                gold_by_label = instance.get(TYPED_ANCHORS_KEY) or {}
                for label, value in instance.items():
                    if is_internal_instance_key(label):
                        continue
                    spec = specs.get(label)
                    if spec is None or spec.value_type is None:
                        continue
                    family = VALUE_TYPE_FAMILY[spec.value_type]
                    gold_by_text = gold_by_label.get(label) or {}
                    for span in value if isinstance(value, list) else [value]:
                        if not isinstance(span, dict) or "start" not in span:
                            continue
                        gold = gold_by_text.get(span.get("text"))
                        target = self._target_from_gold(
                            int(span["start"]), int(span["end"]), family, gold, spec,
                        )
                        if target is not None:
                            targets.append(target)
        return targets

    @staticmethod
    def _target_from_gold(start, end, family, gold, spec) -> SpanTarget | None:
        """Target of a gold span, only when the gold value is normalized.

        A gold value that is just the span's text ("4.3 billion", "May 2024")
        says nothing beyond what a parser would guess, so it is no target.
        """

        if gold is None:
            return None
        if family == FAMILY_NUMBER and (isinstance(gold, str) or isinstance(gold, bool)):
            return None
        if family != FAMILY_NUMBER and isinstance(gold, str) and not is_iso_value(gold, family):
            return None
        target = SpanTarget(start, end)
        if family == FAMILY_NUMBER:
            number = spec.encode(gold)
            if number is None or not math.isfinite(number):
                return None
            target.number = float(number)
            return target
        parts = gold_parts(gold, family, partial=spec.partial)
        if parts is None:
            return None
        target.family = family
        target.codes = component_codes(parts, family)
        return target

    def _units_with_spans(self, item):
        structuring = self.structuring_processor
        if structuring is not None:
            units, _ = structuring._tokenize_text(item)
            return units
        text = item.get("text") or ""
        if not text or self.words_splitter is None:
            return None
        return list(self.words_splitter(text))

    def _parsed_spans(self, item):
        """Parser hits as ``(parsed, start_unit, end_unit)``."""

        if self.word_splitter is None:
            for parsed in self.parser.find(self._words(item)):
                yield parsed, parsed.start, parsed.end
            return
        text = item.get("text") or ""
        units = self._units_with_spans(item)
        if not text or not units:
            return
        from ..span_processor import SpanProcessor

        words = list(self.word_splitter(text))
        starts, ends = SpanProcessor._build_token_char_maps(units)
        for parsed in self.parser.find([word for word, _, _ in words]):
            if not (0 <= parsed.start <= parsed.end < len(words)):
                continue
            span = SpanProcessor._match_to_token_range(
                words[parsed.start][1], words[parsed.end][2], units, starts, ends,
            )
            if span is not None:
                yield parsed, span[0], span[1]

    def _silver_targets(self, item) -> list[SpanTarget]:
        if self.parser is None:
            return []
        targets = []
        for parsed, start, end in self._parsed_spans(item):
            target = SpanTarget(start, end, number=parsed.number)
            if parsed.family in TEMPORAL_FAMILIES and parsed.parts is not None:
                target.family = parsed.family
                target.codes = component_codes(parsed.parts, parsed.family)
            targets.append(target)
        return targets

    def item_targets(self, item, *, limit: int | None = None) -> list[SpanTarget]:
        """Labelled spans of one row in reading order, within ``limit`` words."""

        cfg = self.task_config
        merged: dict[tuple[int, int], SpanTarget] = {}
        for target in self._gold_targets(item):
            key = (target.start, target.end)
            if key in merged:
                merged[key].merge_missing(target)
            else:
                merged[key] = target
        if self.calculation_processor is not None:
            for start, end, value in self.calculation_processor.operand_targets(item):
                key = (start, end)
                target = SpanTarget(start, end, number=value)
                if key in merged:
                    merged[key].merge_missing(target)
                else:
                    merged[key] = target
        for target in self._silver_targets(item):
            key = (target.start, target.end)
            if key in merged:
                merged[key].merge_missing(target)
            else:
                merged[key] = target
        result = []
        for (start, end), target in sorted(merged.items()):
            if start < 0 or end < start or end - start + 1 > cfg.max_span_width:
                continue
            if limit is not None and end >= limit:
                continue
            if target.number is None and target.family == FAMILY_NONE:
                continue
            result.append(target)
        return result[: cfg.max_spans]

    # -- tensors ---------------------------------------------------------------

    def create_labels(self, batch_list, classes_mapping, max_seq_len=0, **kwargs):
        sequence_lengths = kwargs.get("sequence_lengths")
        rows = []
        for index, item in enumerate(batch_list):
            limit = None
            if isinstance(sequence_lengths, torch.Tensor) and index < sequence_lengths.shape[0]:
                limit = int(sequence_lengths.reshape(sequence_lengths.shape[0], -1)[index, 0])
            elif max_seq_len:
                limit = int(max_seq_len)
            rows.append(self.item_targets(item, limit=limit))
        width = max((len(row) for row in rows), default=0)
        if width == 0:
            return None
        batch = len(rows)
        span_idx = torch.zeros(batch, width, 2, dtype=torch.long)
        span_mask = torch.zeros(batch, width, dtype=torch.bool)
        number_value = torch.zeros(batch, width, dtype=torch.float64)
        number_mask = torch.zeros(batch, width, dtype=torch.bool)
        family = torch.full((batch, width), FAMILY_NONE, dtype=torch.long)
        components = {name: torch.full((batch, width), IGNORE, dtype=torch.long) for name in COMPONENTS}
        for row, targets in enumerate(rows):
            for slot, target in enumerate(targets):
                span_idx[row, slot, 0] = target.start
                span_idx[row, slot, 1] = target.end
                span_mask[row, slot] = True
                if target.number is not None:
                    number_value[row, slot] = target.number
                    number_mask[row, slot] = True
                family[row, slot] = target.family
                for name in COMPONENTS:
                    components[name][row, slot] = target.codes[name]
        result = {
            "normalization_span_idx": span_idx,
            "normalization_span_mask": span_mask,
            "normalization_number_value": number_value,
            "normalization_number_mask": number_mask,
            "normalization_temporal_family": family,
        }
        result.update({f"normalization_{name}": tensor for name, tensor in components.items()})
        return result


__all__ = ["NormalizationProcessor", "SpanTarget"]
