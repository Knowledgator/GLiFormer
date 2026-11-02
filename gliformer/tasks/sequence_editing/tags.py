"""Tag vocabulary of the ``tags`` content mode (GECToR-style).

Instead of writing content with learned slots, the head classifies every
REPLACE/APPEND unit into one of a fixed list of content strings, the most
frequent fragments of the training data. A tag carries the space bit as a
leading space: ``" the"`` = whitespace before "the", ``","`` = none.

    PYTHONPATH=. python -m gliformer.tasks.sequence_editing.tags \\
        --data data/grammar_correction/train_ready.jsonl \\
        --out data/grammar_correction/tag_vocab_10k.json --size 10000
"""

from __future__ import annotations

import argparse
import collections
import json
from typing import Iterable

from .alignment import Content, plan_edits


def tag_key(content: Content, slot: int) -> str:
    """Tag string of a content fragment; the space bit does not matter at BOS."""

    spaced = True if slot == 0 else bool(content.space)
    return (" " if spaced else "") + content.text


def tag_content(tag: str) -> Content:
    if tag.startswith(" "):
        return Content(tag[1:], True)
    return Content(tag, False)


def count_tags(pairs: Iterable[tuple[str, str]], splitter) -> collections.Counter:
    counts = collections.Counter()
    for source, target in pairs:
        if not source or target is None:
            continue
        units = [tuple(unit) for unit in splitter(source)]
        if not units:
            continue
        plan = plan_edits(source, units, target, [tuple(unit) for unit in splitter(target)])
        for slot, content in plan.contents.items():
            if content.text:
                counts[tag_key(content, slot)] += 1
    return counts


def build_tag_vocab(pairs, splitter, size: int, min_count: int = 2) -> list[str]:
    counts = count_tags(pairs, splitter)
    return [tag for tag, count in counts.most_common(size) if count >= min_count]


def read_pairs(path: str):
    from .processor import normalize_groups, target_of

    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            if "input" in row:
                yield row["input"], row["output"]
                continue
            text = row.get("text") or ""
            for group in normalize_groups(row.get("sequence_editing")):
                yield text, target_of(text, group)


def main():
    from gliner.data_processing.tokenizer import WordsSplitter

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--min_count", type=int, default=2)
    parser.add_argument("--splitter", default="whitespace")
    args = parser.parse_args()
    counts = count_tags(read_pairs(args.data), WordsSplitter(args.splitter))
    vocab = [tag for tag, count in counts.most_common(args.size) if count >= args.min_count]
    total = sum(counts.values())
    covered = sum(counts[tag] for tag in vocab)
    with open(args.out, "w") as handle:
        json.dump(vocab, handle, ensure_ascii=False, indent=0)
    print(f"{len(vocab)} tags, {covered / max(total, 1):.1%} of {total} content edits covered")


if __name__ == "__main__":
    main()
