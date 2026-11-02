"""Decode calculation slots into programs and run them.

Per query (structuring record slot x ``$calculate`` field) the active slots
form a prefix; each slot's operation is its argmax, and each operand port
takes the pool members whose probability is above ``port_threshold``. A port
an operation needs but left empty takes its single best member; ports that
must hold one value (``copy``, a filter threshold) keep only the best one.

Operand spans are read as numbers by the normalization head; the program then
runs in float64 (:func:`~gliformer.tasks.calculation.program.execute_program`).
A program that fails (no value read for an operand, division by zero, ...)
gives ``None``.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch

from .. import TaskDecoder
from .program import CONSTANTS, NUM_PORTS, OP_NAMES, OPS, ProgramError, execute_program

# Ports that must hold exactly one member, per op.
_SINGLE = {
    "copy": (0,),
    "filter_gt": (1,),
    "filter_ge": (1,),
    "filter_lt": (1,),
    "filter_le": (1,),
}


class CalculationDecoder(TaskDecoder):
    """Programs and values of the calculation head's queries."""

    def __init__(self, config):
        super().__init__(config)
        cfg = config.calculation_config
        self.port_threshold = float(cfg.port_threshold)
        self.active_threshold = float(cfg.active_threshold)
        self.normalization = None
        if getattr(config, "normalization_config", None) is not None:
            from ..normalization.decoder import NormalizationDecoder

            self.normalization = NormalizationDecoder(config)

    @classmethod
    def from_config(cls, config):
        return cls(config)

    # -- programs ------------------------------------------------------------

    def _port_members(self, data, query, slot, port, single):
        """Members of one port: (kind, index, probability), best first when single."""

        spans = data["port_span_probs"][query, slot, port]
        refs = data["port_ref_probs"][query, slot, port, :slot]
        consts = data["port_const_probs"][query, slot, port]
        span_mask = data["operand_mask"][query]
        pool = []
        for index in range(spans.shape[0]):
            if bool(span_mask[index]):
                pool.append(("span", index, float(spans[index])))
        pool.extend(("ref", index, float(refs[index])) for index in range(refs.shape[0]))
        pool.extend(("const", index, float(consts[index])) for index in range(consts.shape[0]))
        if not pool:
            return []
        chosen = [member for member in pool if member[2] > self.port_threshold]
        if not chosen or single:
            chosen = [max(chosen or pool, key=lambda member: member[2])]
        return chosen

    def program(self, data, query: int) -> tuple[list[dict], float]:
        """Program of one query (members use unit offsets) and its confidence."""

        active = data["active_probs"][query]
        steps, confidence = [], 1.0
        for slot in range(active.shape[0]):
            if float(active[slot]) <= self.active_threshold:
                break
            op_probs = data["op_probs"][query, slot]
            op_id = int(op_probs.argmax())
            op = OP_NAMES[op_id]
            arity, _ = OPS[op]
            args = []
            for port in range(NUM_PORTS):
                if port >= arity:
                    args.append([])
                    continue
                members = []
                for kind, index, _prob in self._port_members(
                    data, query, slot, port, port in _SINGLE.get(op, ()),
                ):
                    if kind == "span":
                        start, end = (int(v) for v in data["operand_spans"][query, index].tolist())
                        members.append({"span": "", "start": start, "end": end})
                    elif kind == "ref":
                        members.append({"ref": index})
                    else:
                        members.append({"const": CONSTANTS[index]})
                args.append(members)
            steps.append({"op": op, "args": args})
            confidence = min(confidence, float(active[slot]) * float(op_probs[op_id]))
        return steps, (confidence if steps else 0.0)

    def run(self, model_output, row: int, steps: list[dict]) -> Optional[float]:
        if not steps:
            return None

        def span_value(member):
            if self.normalization is None:
                return None
            return self.normalization.number_value(model_output, row, member["start"], member["end"])

        try:
            return execute_program(steps, span_value=span_value)
        except (ProgramError, OverflowError, ValueError):
            return None

    @staticmethod
    def format_value(value: Optional[float], spec) -> Any:
        if value is None or not math.isfinite(value):
            return None
        if spec is not None and getattr(spec, "integer", False):
            return int(round(value))
        if spec is not None and getattr(spec, "decimals", None) is not None:
            return int(round(value)) if spec.decimals == 0 else round(value, spec.decimals)
        if value.is_integer() and abs(value) < 2 ** 53:
            return int(value)
        return value

    def query_results(self, model_output) -> list[dict]:
        """One entry per query: group, anchor, field id, row, program, value."""

        data = getattr(model_output, "calculation", None)
        if not data:
            return []
        data = {
            key: value.detach().float().cpu() if torch.is_tensor(value) and value.is_floating_point()
            else value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in data.items()
        }
        results = []
        for query in range(int(data["query_group"].shape[0])):
            row = int(data["query_row"][query])
            steps, confidence = self.program(data, query)
            results.append({
                "group": int(data["query_group"][query]),
                "anchor": int(data["query_anchor"][query]),
                "marker": int(data["query_marker"][query]),
                "field_id": int(data["query_field"][query]),
                "row": row,
                "steps": steps,
                "value": self.run(model_output, row, steps),
                "score": confidence,
            })
        return results

    # -- whole batch ---------------------------------------------------------

    def decode(self, model_output, classes_mapping=None, **kwargs):
        results = self.query_results(model_output)
        if not results:
            return []
        rows = max(len(kwargs.get("texts") or ()), max(result["row"] for result in results) + 1)
        output = [[] for _ in range(rows)]
        for result in results:
            output[result["row"]].append(result)
        return output

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
        output = [[] for _ in range(num_original)]
        for valid_i, entries in enumerate(task_results):
            if valid_i >= len(all_start_maps):
                continue
            start_map, end_map = all_start_maps[valid_i], all_end_maps[valid_i]
            text = valid_texts[valid_i]
            mapped = []
            for entry in entries:
                mapped.append({
                    **entry,
                    "steps": map_program(entry["steps"], start_map, end_map, text),
                })
            output[valid_to_orig_idx[valid_i]] = mapped
        return output


def map_program(steps, start_map, end_map, text):
    """Unit offsets -> character offsets and span text."""

    mapped = []
    for step in steps:
        args = []
        for port in step["args"]:
            members = []
            for member in port:
                if "span" in member:
                    start, end = member["start"], member["end"]
                    if start < len(start_map) and end < len(end_map):
                        start_char, end_char = start_map[start], end_map[end]
                        members.append({
                            "span": text[start_char:end_char] if isinstance(text, str) else "",
                            "start": start_char,
                            "end": end_char,
                        })
                else:
                    members.append(dict(member))
            args.append(members)
        mapped.append({"op": step["op"], "args": args})
    return mapped


__all__ = ["CalculationDecoder", "map_program"]
