"""Turns sequence editing head outputs into edited texts and edit lists."""

from __future__ import annotations

from typing import Sequence

from .. import TaskDecoder
from .alignment import APPEND, DELETE, KEEP, REPLACE, Content, apply_edits, char_edits
from .tags import tag_content


def _collapse_repeats(ids: list[int]) -> list[int]:
    out = []
    for token in ids:
        if not out or out[-1] != token:
            out.append(token)
    return out


class SequenceEditingDecoder(TaskDecoder):
    """``decode`` dict of the head -> one result per flat group."""

    def __init__(self, config, tokenizer=None):
        super().__init__(config)
        self.tokenizer = tokenizer
        edit_cfg = getattr(config, "sequence_editing_config", None)
        self.tag_vocab = list(edit_cfg.tag_vocab or []) if edit_cfg is not None else []

    def content_text(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(
            list(ids), skip_special_tokens=True, clean_up_tokenization_spaces=False,
        ).strip()

    def group_plan(self, decode: dict, group: int, num_units: int):
        """``(ops, contents, scores)`` of one group, ``len(ops) == num_units + 1``."""

        slots = num_units + 1
        ops = decode["ops"][group, :slots].tolist()
        ops += [KEEP] * (slots - len(ops))
        probs = decode["op_probs"][group, : len(ops)]
        scores = [float(probs[slot, op]) for slot, op in enumerate(ops[: probs.shape[0]])]
        contents: dict[int, Content] = {}
        positions = decode.get("edit_pos")
        if positions is not None and positions.numel():
            for index in (positions[:, 0] == group).nonzero(as_tuple=False).flatten().tolist():
                slot = int(positions[index, 1])
                if slot >= slots:
                    continue
                if "edit_tag" in decode:
                    contents[slot] = tag_content(self.tag_vocab[int(decode["edit_tag"][index])])
                    continue
                length = int(decode["edit_length"][index])
                ids = _collapse_repeats(decode["edit_tokens"][index, :length].tolist())
                text = self.content_text(ids)
                if not text:
                    # Nothing to write: replace becomes delete, append keeps.
                    ops[slot] = KEEP if ops[slot] == APPEND else DELETE
                    continue
                contents[slot] = Content(text, bool(float(decode["edit_space"][index]) >= 0.5))
        for slot, op in enumerate(ops):
            if op in (REPLACE, APPEND) and slot not in contents:
                ops[slot] = KEEP if op == APPEND else DELETE
        return ops, contents, scores

    def decode_groups(self, decode: dict, texts: Sequence[str], units: Sequence[Sequence[tuple]]):
        """Per flat group: ``{"row", "text", "edits", "ops"}``."""

        results = []
        if decode is None:
            return results
        rows = decode["group_row"].tolist()
        for group, row in enumerate(rows):
            row_units = list(units[row])
            ops, contents, scores = self.group_plan(decode, group, len(row_units))
            source = texts[row]
            results.append(
                {
                    "row": row,
                    "text": apply_edits(source, row_units, ops, contents),
                    "edits": char_edits(source, row_units, ops, contents, scores),
                    "ops": ops,
                }
            )
        return results

    def decode(self, model_output, classes_mapping=None, **kwargs):
        return getattr(model_output, "sequence_editing", None)


__all__ = ["SequenceEditingDecoder"]
