"""Decode normalization head outputs into values per span.

``decode`` returns, per text, one entry per span the head read::

    {"start": 3, "end": 5, "number": 3, "date": "2024-03-03", "time": ..., "datetime": ...}

``number`` comes from the ``decode_with`` number head; with several heads,
``numbers`` holds each head's value.

Calendar values are decoded jointly over valid dates and written patterns,
so a span reads as ``"2024-05"`` when only the month and year are written.
``span_values`` gives the same values for one ``(row, start, end)`` span, for
decoders that pick spans themselves (the structuring decoder).
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .. import TaskDecoder
from .calendar import decode_calendar
from .values import (
    FAMILY_DATE,
    FAMILY_DATETIME,
    FAMILY_NAMES,
    FAMILY_NUMBER,
    FAMILY_TIME,
    VALUE_TYPE_FAMILY,
    TemporalParts,
    compose_partial,
)

_TEMPORAL = (FAMILY_DATE, FAMILY_TIME, FAMILY_DATETIME)


class NormalizationDecoder(TaskDecoder):
    """Values of the spans the normalization head read."""

    def __init__(self, config):
        super().__init__(config)
        cfg = config.normalization_config
        self.year_bounds = cfg.temporal.year_bounds()
        self.year_digits = bool(cfg.temporal.year_digits)
        self.number_heads = list(cfg.number.heads)
        self.decode_with = cfg.number.decode_with

    # -- one span ------------------------------------------------------------

    @staticmethod
    def _probs(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits.detach().float().cpu(), dim=-1)

    def _fallback_year(self, temporal, row, slot) -> int | None:
        if not self.year_digits or "year_digits" not in temporal:
            return None
        digits = temporal["year_digits"][row, slot].argmax(dim=-1).tolist()
        year = 0
        for digit in digits:
            year = year * 10 + int(digit)
        return year if year >= 1 else None

    def temporal_value(self, temporal, row: int, slot: int, family: int, *, partial: bool = True):
        """ISO text (reduced precision when partial) and its probability."""

        probs = {
            name: self._probs(temporal[name][row, slot])
            for name in ("year", "month", "day", "hour", "minute", "second")
        }
        components, confidence = decode_calendar(
            family,
            year_probs=probs["year"],
            month_probs=probs["month"],
            day_probs=probs["day"],
            hour_probs=probs["hour"],
            minute_probs=probs["minute"],
            second_probs=probs["second"],
            year_range=self.year_bounds,
            fallback_year=self._fallback_year(temporal, row, slot),
            partial=partial,
        )
        return compose_partial(family, TemporalParts(**components)), confidence

    @staticmethod
    def _clean_number(value: float) -> float | int:
        if math.isfinite(value) and value.is_integer() and abs(value) < 2 ** 53:
            return int(value)
        return float(value)

    def slot_values(self, model_output, row: int, slot: int) -> dict[str, Any]:
        temporal = getattr(model_output, "normalization_temporal", None) or {}
        numbers = getattr(model_output, "normalization_number", None) or {}
        values: dict[str, Any] = {}
        if numbers:
            by_head = {
                kind: self._clean_number(float(numbers[kind][row, slot]))
                for kind in self.number_heads if kind in numbers
            }
            if self.decode_with in by_head:
                values["number"] = by_head[self.decode_with]
            if len(by_head) > 1:
                values["numbers"] = by_head
        if temporal:
            for family in _TEMPORAL:
                values[FAMILY_NAMES[family]], _ = self.temporal_value(temporal, row, slot, family)
        return values

    def span_values(self, model_output, row: int, start: int, end: int) -> dict[str, Any] | None:
        """Values of the span ``(start, end)`` of text ``row``, if the head read it."""

        slot = self._slot(model_output, row, start, end)
        return None if slot is None else self.slot_values(model_output, row, slot)

    def _slot(self, model_output, row: int, start: int, end: int) -> int | None:
        span_idx = getattr(model_output, "normalization_span_idx", None)
        span_mask = getattr(model_output, "normalization_span_mask", None)
        if span_idx is None or span_mask is None or row >= span_idx.shape[0]:
            return None
        matches = (
            (span_idx[row, :, 0] == start) & (span_idx[row, :, 1] == end) & span_mask[row].bool()
        ).nonzero(as_tuple=False)
        return int(matches[0, 0]) if matches.numel() else None

    def field_value(self, model_output, row: int, start: int, end: int, spec) -> Any:
        """Normalized value of a span for a typed value field, or ``None``.

        ``spec`` is the field's ``TypedFieldSpec``: numbers follow its
        ``$integer``/``$decimals`` rounding; dates and times are full values
        when ``$partial`` is false.
        """

        value_type = getattr(spec, "value_type", None)
        if value_type is None:
            return None
        slot = self._slot(model_output, row, start, end)
        if slot is None:
            return None
        family = VALUE_TYPE_FAMILY[value_type]
        if family == FAMILY_NUMBER:
            numbers = getattr(model_output, "normalization_number", None) or {}
            if self.decode_with not in numbers:
                return None
            number = float(numbers[self.decode_with][row, slot])
            if not math.isfinite(number):
                return None
            if spec.integer:
                return int(round(number))
            if spec.decimals is not None:
                return round(number, spec.decimals)
            return self._clean_number(number)
        temporal = getattr(model_output, "normalization_temporal", None) or {}
        if not temporal:
            return None
        value, _confidence = self.temporal_value(temporal, row, slot, family, partial=spec.partial)
        return value

    # -- whole batch ---------------------------------------------------------

    def decode(self, model_output, classes_mapping=None, **kwargs):
        span_idx = getattr(model_output, "normalization_span_idx", None)
        span_mask = getattr(model_output, "normalization_span_mask", None)
        if span_idx is None or span_mask is None:
            return []
        results = []
        for row in range(span_idx.shape[0]):
            entries = []
            for slot in range(span_idx.shape[1]):
                if not bool(span_mask[row, slot]):
                    continue
                start, end = (int(value) for value in span_idx[row, slot].tolist())
                entries.append({"start": start, "end": end, **self.slot_values(model_output, row, slot)})
            results.append(entries)
        if not any(results):
            return []
        return results

    def map_results(
        self,
        task_results: list,
        valid_to_orig_idx,
        all_start_maps,
        all_end_maps,
        valid_texts,
        num_original: int,
        **kwargs,
    ) -> list:
        """Word spans -> character offsets and span text in the original order."""

        output = [[] for _ in range(num_original)]
        for valid_i, entries in enumerate(task_results):
            start_map, end_map = all_start_maps[valid_i], all_end_maps[valid_i]
            text = valid_texts[valid_i]
            mapped = []
            for entry in entries:
                start, end = entry["start"], entry["end"]
                if start >= len(start_map) or end >= len(end_map):
                    continue
                start_char, end_char = start_map[start], end_map[end]
                mapped.append({
                    **entry,
                    "start": start_char,
                    "end": end_char,
                    "text": text[start_char:end_char],
                })
            output[valid_to_orig_idx[valid_i]] = mapped
        return output


__all__ = ["NormalizationDecoder"]
