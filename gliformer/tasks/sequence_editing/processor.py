"""Inputs and training targets of the sequence editing head.

A row asks for edits with its ``sequence_editing`` key: an instruction
string, one group dict, or a list of them (one ``[EDIT] <instruction>``
prompt section per group)::

    "sequence_editing": {"instruction": "fix grammar", "target": "She went home."}
    "sequence_editing": {"instruction": "redact names",
                         "edits": [{"start": 5, "end": 9, "replacement": "[NAME]"}]}

``target`` is the edited text; ``edits`` (character offsets into ``text``)
are applied to build it. The target is aligned to the row's text units
(words or subwords, see ``alignment.plan_edits``).

Label-free tensors (training and inference), per flat group ``BN``:

* ``seq_edit_group_row`` ``(BN,)``: batch row of the group;
* ``seq_edit_marker`` ``(BN,)``: ordinal of its ``[EDIT]`` marker in the row.

Training tensors, ``L`` = units + 1 (slot 0 = BOS), ``E`` = content edits,
``M`` = ``max_content_len``:

* ``seq_edit_op`` ``(BN, L)``: KEEP/DELETE/REPLACE/APPEND, -100 = ignored
  (padding, truncated units, groups without a target);
* ``seq_edit_pos`` ``(E, 2)``: (group, slot) of every REPLACE/APPEND;
* ``seq_edit_length`` ``(E,)``: content length - 1 (-100 if over ``M``);
* ``seq_edit_tokens`` ``(E, M)``: content subword ids, -100 past the length;
* ``seq_edit_space`` ``(E,)``: 1 if whitespace precedes the content, -100 at BOS.

With ``content_mode: tags`` the content tensors are replaced by
``seq_edit_tag`` ``(E,)``: index of the content in ``tag_vocab``. Edits
whose content is not in the vocabulary get op label -100.
"""

from __future__ import annotations

from typing import Any

import torch

from .. import TaskProcessor
from .alignment import CONTENT_OPS, plan_edits
from .tags import tag_key

DATA_KEY = "sequence_editing"
PLAN_KEY = "_gliformer_seq_edit_plan"
IGNORE = -100


def normalize_groups(value) -> list[dict]:
    """``sequence_editing`` value -> list of group dicts (may be empty)."""

    if value is None:
        return []
    if isinstance(value, (str, dict)):
        value = [value]
    groups = []
    for entry in value:
        if isinstance(entry, str):
            groups.append({"instruction": entry})
        elif isinstance(entry, dict):
            groups.append(entry)
    return groups


def instruction_of(group: dict) -> str:
    text = group.get("instruction", group.get("prompt"))
    return str(text).strip() if text else ""


def apply_char_edits(text: str, edits) -> str | None:
    """Apply ``[{"start", "end", "replacement"}]`` to ``text``; None if they overlap."""

    spans = sorted(
        (int(edit["start"]), int(edit["end"]), str(edit.get("replacement", "")))
        for edit in edits
    )
    out, cursor = [], 0
    for start, end, replacement in spans:
        if start < cursor or end < start or end > len(text):
            return None
        out.append(text[cursor:start])
        out.append(replacement)
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def target_of(text: str, group: dict) -> str | None:
    target = group.get("target", group.get("output"))
    if isinstance(target, str):
        return target
    edits = group.get("edits")
    if isinstance(edits, list):
        return apply_char_edits(text, edits)
    return None


class SequenceEditingProcessor(TaskProcessor):
    """Prompt sections, marker indices and edit targets."""

    task_name = "sequence_editing"

    def __init__(self, config, tokenizer=None, words_splitter=None, **kwargs):
        super().__init__(config, tokenizer, words_splitter, **kwargs)
        self.task_config = config.sequence_editing_config
        if self.task_config is None:
            raise ValueError("sequence_editing_config is required for sequence_editing")
        self.tokenizer = tokenizer
        self.words_splitter = words_splitter
        self.edit_token = getattr(config, "edit_token", None) or "[EDIT]"
        self.tag_mode = self.task_config.content_mode == "tags"
        self.tag_index = (
            {tag: index for index, tag in enumerate(self.task_config.tag_vocab)} if self.tag_mode else {}
        )

    # -- prompt --------------------------------------------------------------

    def get_classes_mapping(self, batch_list, **kwargs):
        """Instructions per row, in prompt order."""

        return [
            [instruction_of(group) for group in normalize_groups(item.get(DATA_KEY))]
            for item in batch_list
        ]

    @staticmethod
    def _row_instructions(classes_mapping, batch_idx):
        mapping = getattr(classes_mapping, "sequence_editing_mapping", None) or []
        return mapping[batch_idx] if batch_idx < len(mapping) else []

    def contribute_prompt(self, classes_mapping, batch_idx, use_labels_encoder=False):
        prompt = []
        for instruction in self._row_instructions(classes_mapping, batch_idx):
            prompt.append(self.edit_token)
            prompt.extend(instruction.split())
        return prompt

    def prepare_typed_inputs(self, classes_mapping, use_labels_encoder=False, texts=None):
        rows, markers = [], []
        mapping = getattr(classes_mapping, "sequence_editing_mapping", None) or []
        for row, instructions in enumerate(mapping):
            for ordinal in range(len(instructions)):
                rows.append(row)
                markers.append(ordinal)
        if not rows:
            return {}
        return {
            "seq_edit_group_row": torch.tensor(rows, dtype=torch.long),
            "seq_edit_marker": torch.tensor(markers, dtype=torch.long),
        }

    def contribute_inference_input(self, item: dict[str, Any], **kwargs):
        instructions = kwargs.get("sequence_editing")
        if instructions is not None and DATA_KEY not in item:
            item[DATA_KEY] = normalize_groups(instructions)

    # -- targets ---------------------------------------------------------------

    def _units(self, text: str):
        if self.words_splitter is None:
            return []
        return [tuple(unit) for unit in self.words_splitter(text)]

    def _content_ids(self, fragment: str) -> list[int]:
        return list(self.tokenizer(fragment, add_special_tokens=False)["input_ids"])

    def plan_for(self, text: str, group: dict):
        """Cached ``{"ops", "contents": {slot: (ids, space)}, "pieces"}`` or None."""

        cached = group.get(PLAN_KEY)
        if cached is not None and cached.get("text") == text and "tags" in cached:
            return cached
        target = target_of(text, group)
        if target is None or not text:
            return None
        source_units = self._units(text)
        if not source_units:
            return None
        plan = plan_edits(text, source_units, target, self._units(target))
        contents = {
            slot: (self._content_ids(content.text), bool(content.space))
            for slot, content in plan.contents.items()
        }
        cached = {
            "text": text,
            "pieces": [unit[0] for unit in source_units],
            "ops": list(plan.ops),
            "contents": contents,
            "tags": {slot: tag_key(content, slot) for slot, content in plan.contents.items()},
        }
        group[PLAN_KEY] = cached
        return cached

    def resolve_spans(self, item):
        text = item.get("text") or ""
        for group in normalize_groups(item.get(DATA_KEY)):
            self.plan_for(text, group)

    def create_labels(self, batch_list, classes_mapping, max_seq_len=0, sequence_lengths=None, **kwargs):
        groups = []
        for row, item in enumerate(batch_list):
            for group in normalize_groups(item.get(DATA_KEY)):
                groups.append((row, item, group))
        if not groups:
            return None
        num_units = max(
            [len(item.get("tokenized_text") or []) for item in batch_list] + [int(max_seq_len or 0), 1]
        )
        max_content = self.task_config.max_content_len
        ops = torch.full((len(groups), num_units + 1), IGNORE, dtype=torch.long)
        positions, lengths, tokens, spaces, tags = [], [], [], [], []
        for flat, (row, item, group) in enumerate(groups):
            text = item.get("text") or ""
            plan = self.plan_for(text, group)
            if plan is None:
                continue
            pieces = list(item.get("tokenized_text") or [])
            if pieces and pieces != plan["pieces"]:
                # The batch's units differ from the splitter's (pre-split
                # input); the plan does not apply.
                continue
            available = len(plan["pieces"])
            if sequence_lengths is not None:
                available = min(available, int(sequence_lengths[row].reshape(-1)[0]))
            available = min(available, num_units)
            slots = available + 1
            ops[flat, :slots] = torch.tensor(plan["ops"][:slots], dtype=torch.long)
            if self.tag_mode:
                for slot, tag in plan["tags"].items():
                    if slot >= slots or plan["ops"][slot] not in CONTENT_OPS:
                        continue
                    index = self.tag_index.get(tag)
                    if index is None:
                        ops[flat, slot] = IGNORE
                        continue
                    positions.append((flat, slot))
                    tags.append(index)
                continue
            for slot, (ids, space) in plan["contents"].items():
                if slot >= slots or plan["ops"][slot] not in CONTENT_OPS or not ids:
                    if slot < slots and not ids:
                        ops[flat, slot] = IGNORE
                    continue
                positions.append((flat, slot))
                lengths.append(len(ids) - 1 if len(ids) <= max_content else IGNORE)
                row_tokens = [IGNORE] * max_content
                row_tokens[: min(len(ids), max_content)] = ids[:max_content]
                tokens.append(row_tokens)
                spaces.append(int(space) if slot > 0 else IGNORE)
        result = {"seq_edit_op": ops}
        if self.tag_mode:
            result.update(
                seq_edit_pos=torch.tensor(positions, dtype=torch.long).reshape(-1, 2),
                seq_edit_tag=torch.tensor(tags, dtype=torch.long),
            )
        elif positions:
            result.update(
                seq_edit_pos=torch.tensor(positions, dtype=torch.long),
                seq_edit_length=torch.tensor(lengths, dtype=torch.long),
                seq_edit_tokens=torch.tensor(tokens, dtype=torch.long),
                seq_edit_space=torch.tensor(spaces, dtype=torch.long),
            )
        else:
            result.update(
                seq_edit_pos=torch.zeros((0, 2), dtype=torch.long),
                seq_edit_length=torch.zeros((0,), dtype=torch.long),
                seq_edit_tokens=torch.zeros((0, max_content), dtype=torch.long),
                seq_edit_space=torch.zeros((0,), dtype=torch.long),
            )
        return result


__all__ = ["SequenceEditingProcessor", "normalize_groups", "target_of", "apply_char_edits"]
