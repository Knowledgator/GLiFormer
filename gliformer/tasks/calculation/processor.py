"""Inputs and training targets of the calculation head.

Every structuring field marked ``"$calculate": true`` renders a ``[CALC]``
marker after its description. Gold programs come from the row's
``calculation`` key, nested like ``structuring``::

    "calculation": {"$root": {"answer": {"steps": [...]}}}

A schema whose records are a list takes a list of the same length. Program
format: :mod:`gliformer.tasks.calculation.program`.

Label-free tensors (training and inference), per flat structuring group
``BN`` and calc field ``Kc`` (prompt order):

* ``calculation_marker_index`` ``(BN, Kc)``: ordinal of the field's ``[CALC]``
  marker among the row's markers;
* ``calculation_marker_mask`` ``(BN, Kc)``, ``calculation_field_idx`` ``(BN, Kc)``;
* ``calculation_marker_list``: the same ordinals as a Python list per group.

Training tensors, one row per supervised query ``Q`` = (group, gold record,
calc field), ``S`` operand spans, ``K`` slots, ``C`` constants:

* ``calculation_query_group`` / ``_record`` / ``_marker`` ``(Q,)``;
* ``calculation_bio_labels`` ``(Q, L, 3)`` start/end/inside of operand spans;
* ``calculation_span_idx`` ``(Q, S, 2)`` inclusive unit spans, ``_span_mask``;
* ``calculation_op`` ``(Q, K)`` op ids (-100 past the program);
* ``calculation_active`` ``(Q, K)`` 1 for program steps;
* ``calculation_port_spans`` ``(Q, K, 2, S)``, ``_port_refs`` ``(Q, K, 2, K)``,
  ``_port_consts`` ``(Q, K, 2, C)``: operand-set targets;
* ``calculation_query_list`` / ``_span_list``: Python lists of the query
  triples and gold spans, read by the head without a device round trip.
"""

from __future__ import annotations

import copy
from typing import Any

import torch

from .. import TaskProcessor
from ...processing._structuring_hierarchy import calc_field_layout
from ..span_processor import SpanProcessor
from .program import (
    CONSTANTS,
    NUM_PORTS,
    OP_TO_ID,
    ProgramError,
    constant_index,
    validate_program,
)

RESOLVED_FLAG = "_gliformer_calculation_resolved"
DATA_KEY = "calculation"


def iter_program_spans(steps):
    for step in steps:
        for port in step.get("args") or []:
            for member in port:
                if isinstance(member, dict) and "span" in member:
                    yield member


class CalculationProcessor(TaskProcessor):
    """Marker indices and program targets for the calculation head."""

    task_name = "calculation"
    tensor_prefix = "calculation"

    def __init__(self, config, tokenizer=None, words_splitter=None, **kwargs):
        super().__init__(config, tokenizer, words_splitter, **kwargs)
        self.task_config = config.calculation_config
        if self.task_config is None:
            raise ValueError("calculation_config is required for calculation")
        self.words_splitter = words_splitter
        # Set by the data processor: resolves text units and field specs.
        self.structuring_processor = None

    def get_classes_mapping(self, batch_list, **kwargs):
        return None

    # -- spans -------------------------------------------------------------

    def _units(self, item):
        structuring = self.structuring_processor
        if structuring is not None:
            units, _ = structuring._tokenize_text(item)
            return units
        text = item.get("text") or ""
        if not text or self.words_splitter is None:
            return None
        return list(self.words_splitter(text))

    def resolve_spans(self, item):
        """Map every span member's character offsets onto text units.

        A program with a span that does not land on units, or that fails
        validation, is dropped (the field then gets no calculation target).
        """

        if item.get(RESOLVED_FLAG):
            return
        programs = item.get(DATA_KEY)
        if not isinstance(programs, dict) or not programs:
            return
        units = self._units(item)
        if not units:
            item[DATA_KEY] = {}
            item[RESOLVED_FLAG] = True
            return
        starts, ends = SpanProcessor._build_token_char_maps(units)
        text = item.get("text") or ""
        resolved: dict[str, Any] = {}
        for schema, records in programs.items():
            is_list = isinstance(records, list)
            out_records = []
            for record in records if is_list else [records]:
                out = {}
                for label, entry in (record or {}).items() if isinstance(record, dict) else ():
                    steps = entry.get("steps") if isinstance(entry, dict) else entry
                    steps = self._resolve_program(steps, text, units, starts, ends)
                    if steps is not None:
                        out[label] = {"steps": steps}
                out_records.append(out)
            resolved[schema] = out_records if is_list else out_records[0]
        item[DATA_KEY] = resolved
        item[RESOLVED_FLAG] = True

    @staticmethod
    def _resolve_program(steps, text, units, starts, ends):
        try:
            validate_program(steps)
        except ProgramError:
            return None
        steps = copy.deepcopy(steps)
        for member in iter_program_spans(steps):
            start, end = int(member["start"]), int(member["end"])
            if text and text[start:end] != member["span"]:
                return None
            span = SpanProcessor._match_to_token_range(start, end, units, starts, ends)
            if span is None:
                return None
            member["unit"] = [int(span[0]), int(span[1])]
        return steps

    def operand_targets(self, item):
        """``(start_unit, end_unit, value)`` of every resolved operand span."""

        programs = item.get(DATA_KEY)
        if not isinstance(programs, dict):
            return []
        found = {}
        for records in programs.values():
            for record in records if isinstance(records, list) else [records]:
                for entry in (record or {}).values():
                    for member in iter_program_spans(entry.get("steps") or []):
                        unit, value = member.get("unit"), member.get("value")
                        if unit is None or value is None:
                            continue
                        found.setdefault((int(unit[0]), int(unit[1])), float(value))
        return [(start, end, value) for (start, end), value in sorted(found.items())]

    # -- label-free inputs -------------------------------------------------

    def _flat_groups(self, classes_mapping):
        structuring = self.structuring_processor
        if structuring is None or classes_mapping is None:
            return []
        return list(structuring._flat_structuring_iter(classes_mapping))

    def _layout(self, struct_item):
        structuring = self.structuring_processor
        factory = getattr(structuring, "_spec", None)
        return calc_field_layout(struct_item, spec_factory=factory)

    def prepare_typed_inputs(self, classes_mapping, use_labels_encoder=False, texts=None):
        """``[CALC]`` marker ordinals of every calc field (train and inference)."""

        if use_labels_encoder:
            return {}
        groups = self._flat_groups(classes_mapping)
        layouts = [(batch_idx, self._layout(item)) for _flat, batch_idx, _g, item in groups]
        width = max((len(layout) for _, layout in layouts), default=0)
        if width == 0:
            return {}
        marker_index = torch.zeros(len(layouts), width, dtype=torch.long)
        marker_mask = torch.zeros(len(layouts), width, dtype=torch.bool)
        field_idx = torch.zeros(len(layouts), width, dtype=torch.long)
        counts: dict[int, int] = {}
        ordinals: list[list[int]] = []
        for group, (batch_idx, layout) in enumerate(layouts):
            ordinals.append([])
            for slot, (_label, field_id, _spec) in enumerate(layout):
                ordinals[group].append(counts.get(batch_idx, 0))
                marker_index[group, slot] = counts.get(batch_idx, 0)
                counts[batch_idx] = counts.get(batch_idx, 0) + 1
                marker_mask[group, slot] = True
                field_idx[group, slot] = field_id
        return {
            "calculation_marker_index": marker_index,
            "calculation_marker_mask": marker_mask,
            "calculation_field_idx": field_idx,
            # Host copy of the marker ordinals (``marker_index`` where
            # ``marker_mask``): a Python list stays on the CPU when the
            # trainer moves the batch, so the head reads it without a sync.
            "calculation_marker_list": ordinals,
        }

    # -- training targets --------------------------------------------------

    @staticmethod
    def _schema_programs(programs, struct_item):
        """Programs of one schema group (multi-level roots are renamed)."""

        name = getattr(struct_item, "name", None)
        for key in (getattr(struct_item, "data_key", None), name, f"${name}" if name else None):
            if key is not None and key in programs:
                return programs[key]
        return next(iter(programs.values())) if len(programs) == 1 else None

    def _queries(self, batch_list, classes_mapping, limits):
        """Supervised ``(group, record, marker slot, steps)`` in flat order."""

        queries = []
        for flat, batch_idx, _group, struct_item in self._flat_groups(classes_mapping):
            if batch_idx >= len(batch_list):
                continue
            programs = batch_list[batch_idx].get(DATA_KEY)
            if not isinstance(programs, dict):
                continue
            records = self._schema_programs(programs, struct_item)
            if records is None:
                continue
            records = records if isinstance(records, list) else [records]
            layout = self._layout(struct_item)
            limit = limits[batch_idx] if batch_idx < len(limits) else None
            for record_idx, record in enumerate(records):
                if not isinstance(record, dict):
                    continue
                for slot, (label, _field_id, _spec) in enumerate(layout):
                    entry = record.get(label)
                    steps = entry.get("steps") if isinstance(entry, dict) else None
                    if not steps or len(steps) > self.task_config.num_slots:
                        continue
                    spans = list(iter_program_spans(steps))
                    if any("unit" not in member for member in spans):
                        continue
                    if limit is not None and any(member["unit"][1] >= limit for member in spans):
                        continue
                    if any(
                        member["unit"][1] - member["unit"][0] + 1 > self.task_config.max_span_width
                        for member in spans
                    ):
                        continue
                    queries.append((flat, record_idx, slot, steps))
        return queries

    def create_labels(self, batch_list, classes_mapping, max_seq_len=0, **kwargs):
        sequence_lengths = kwargs.get("sequence_lengths")
        limits = []
        for index in range(len(batch_list)):
            if isinstance(sequence_lengths, torch.Tensor) and index < sequence_lengths.shape[0]:
                limits.append(int(sequence_lengths.reshape(sequence_lengths.shape[0], -1)[index, 0]))
            else:
                limits.append(int(max_seq_len) if max_seq_len else None)
        queries = self._queries(batch_list, classes_mapping, limits)
        if not queries:
            return None
        num_slots = self.task_config.num_slots
        num_consts = len(CONSTANTS)
        length = max(int(max_seq_len or 0), 1)
        span_lists = []
        for _flat, _record, _slot, steps in queries:
            unique = sorted({tuple(member["unit"]) for member in iter_program_spans(steps)})
            span_lists.append(unique)
        width = max(max((len(spans) for spans in span_lists), default=0), 1)
        count = len(queries)
        group = torch.zeros(count, dtype=torch.long)
        record = torch.zeros(count, dtype=torch.long)
        marker = torch.zeros(count, dtype=torch.long)
        bio = torch.zeros(count, length, 3, dtype=torch.float32)
        span_idx = torch.zeros(count, width, 2, dtype=torch.long)
        span_mask = torch.zeros(count, width, dtype=torch.bool)
        op = torch.full((count, num_slots), -100, dtype=torch.long)
        active = torch.zeros(count, num_slots, dtype=torch.float32)
        port_spans = torch.zeros(count, num_slots, NUM_PORTS, width, dtype=torch.float32)
        port_refs = torch.zeros(count, num_slots, NUM_PORTS, num_slots, dtype=torch.float32)
        port_consts = torch.zeros(count, num_slots, NUM_PORTS, num_consts, dtype=torch.float32)
        for row, ((flat, record_idx, slot, steps), spans) in enumerate(zip(queries, span_lists)):
            group[row], record[row], marker[row] = flat, record_idx, slot
            position = {span: index for index, span in enumerate(spans)}
            for index, (start, end) in enumerate(spans):
                span_idx[row, index, 0], span_idx[row, index, 1] = start, end
                span_mask[row, index] = True
                if end < length:
                    bio[row, start, 0] = 1.0
                    bio[row, end, 1] = 1.0
                    bio[row, start:end + 1, 2] = 1.0
            for step_index, step in enumerate(steps):
                op[row, step_index] = OP_TO_ID[step["op"]]
                active[row, step_index] = 1.0
                for port_index, port in enumerate(step.get("args") or []):
                    for member in port:
                        if "span" in member:
                            port_spans[row, step_index, port_index, position[tuple(member["unit"])]] = 1.0
                        elif "ref" in member:
                            port_refs[row, step_index, port_index, int(member["ref"])] = 1.0
                        else:
                            port_consts[row, step_index, port_index, constant_index(member["const"])] = 1.0
        return {
            "calculation_query_group": group,
            "calculation_query_record": record,
            "calculation_query_marker": marker,
            "calculation_bio_labels": bio,
            "calculation_span_idx": span_idx,
            "calculation_span_mask": span_mask,
            "calculation_op": op,
            "calculation_active": active,
            "calculation_port_spans": port_spans,
            "calculation_port_refs": port_refs,
            "calculation_port_consts": port_consts,
            # Host copies for the head (Python lists are not moved to the
            # device): ``(group, record, marker)`` per query and its gold
            # operand spans (``span_idx`` where ``span_mask``).
            "calculation_query_list": [[flat, record_idx, slot] for flat, record_idx, slot, _ in queries],
            "calculation_span_list": [[list(span) for span in spans] for spans in span_lists],
        }


__all__ = ["CalculationProcessor", "iter_program_spans"]
