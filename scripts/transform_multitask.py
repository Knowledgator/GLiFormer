#!/usr/bin/env python3
"""Transform raw_multitask_gemini_lite.jsonl into GLiNExT multi-task format.

Each input record (id ending in _relex or _json) is expanded into ``--n``
multi-task examples. Every output example includes:
  - tokenized_text and text
  - one or more classification ontologies (sampled)
  - one embedding pair (positive=1.0 or negative=0.0)
  - extraction (NER + relations from item) when input is *_relex
  - structuring (schemas derived from item.extracted) when input is *_json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


# Matches gliner.data_processing.tokenizer.WhitespaceTokenSplitter.
_TOKEN_RE = re.compile(r"\w+(?:[-_]\w+)*|\S")


def tokenize(text: str) -> List[str]:
    return [m.group() for m in _TOKEN_RE.finditer(text or "")]


def flatten_for_structuring(value: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested dict into dot-notation string / list-of-string fields.

    - dict → recurse with dotted key
    - list of primitives → list of strings (None dropped)
    - list of dicts at sub-level → skipped (only top-level lists become schemas)
    - scalar → coerced to str
    """
    out: Dict[str, Any] = {}
    if not isinstance(value, dict):
        return out
    for k, v in value.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_for_structuring(v, key))
        elif isinstance(v, list):
            if not v:
                continue
            cleaned = [str(x) for x in v if isinstance(x, (str, int, float, bool))]
            if cleaned and len(cleaned) == len(v):
                out[key] = cleaned
        elif v is None:
            continue
        else:
            out[key] = str(v)
    return out


def build_structuring(extracted: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Build the GLiNExT ``structuring`` dict from a free-form extracted JSON.

    - top-level key with list[dict]  → schema = key, instances = flattened dicts
    - top-level key with dict        → schema = key, single flattened instance
    - top-level scalar / list values → grouped under the ``metadata`` schema
    """
    structuring: Dict[str, List[Dict[str, Any]]] = {}
    metadata: Dict[str, Any] = {}
    if not isinstance(extracted, dict):
        return structuring

    for key, val in extracted.items():
        if isinstance(val, list) and val and all(isinstance(x, dict) for x in val):
            instances = [flatten_for_structuring(x) for x in val]
            instances = [x for x in instances if x]
            if instances:
                structuring[str(key)] = instances
        elif isinstance(val, dict):
            inst = flatten_for_structuring(val)
            if inst:
                structuring[str(key)] = [inst]
        elif isinstance(val, list):
            cleaned = [str(x) for x in val if isinstance(x, (str, int, float, bool))]
            if cleaned and len(cleaned) == len(val):
                metadata[str(key)] = cleaned
        elif val is None:
            continue
        else:
            metadata[str(key)] = str(val)

    if metadata:
        structuring["metadata"] = [metadata]
    return structuring


def sample_classification(ontologies: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    if not ontologies:
        return []
    k = rng.randint(1, len(ontologies))
    chosen = rng.sample(ontologies, k)
    out: List[Dict[str, Any]] = []
    for o in chosen:
        labels = list(o.get("labels") or [])
        if not labels:
            continue
        out.append({
            "name": o.get("ontology") or "topic",
            "all_labels": labels,
            "true_labels": list(o.get("true_positives") or []),
        })
    return out


def sample_embedding_pair(
    text: str,
    positives: List[str],
    negatives: List[str],
    rng: random.Random,
) -> List[List[Any]]:
    """Pick one paraphrase pair: positive=1.0 / negative=0.0."""
    pool: List[tuple[str, str]] = []
    if positives:
        pool.append(("pos", rng.choice(positives)))
    if negatives:
        pool.append(("neg", rng.choice(negatives)))
    if not pool or not text:
        return []
    kind, other = rng.choice(pool)
    score = 1.0 if kind == "pos" else 0.0
    return [[tokenize(text), tokenize(other), score]]


def select_extraction(item: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    """Build the ``extraction`` field from item.ner / item.relations.

    - if no NER spans, return [].
    - if relations exist, include them with probability ~0.5 (NER alone otherwise).
      Relations are indices into the NER list, so they cannot ship without NER.
    """
    ner = list(item.get("ner") or [])
    rels = list(item.get("relations") or [])
    if not ner:
        return []
    entry: Dict[str, Any] = {"name": None, "ner": ner}
    if rels and rng.random() < 0.5:
        entry["relations"] = rels
    return [entry]


def select_structuring(
    structuring_full: Dict[str, List[Dict[str, Any]]],
    rng: random.Random,
) -> Dict[str, List[Dict[str, Any]]]:
    if not structuring_full:
        return {}
    schemas = list(structuring_full.keys())
    k = rng.randint(1, len(schemas))
    chosen = rng.sample(schemas, k)
    return {s: structuring_full[s] for s in chosen}


def make_combo(
    record: Dict[str, Any],
    structuring_full: Dict[str, List[Dict[str, Any]]],
    kind: str,
    rng: random.Random,
) -> Dict[str, Any] | None:
    item = record.get("item") or {}
    raw_text = record.get("text") or ""

    if kind == "relex":
        tokenized = list(item.get("tokenized_text") or tokenize(raw_text))
        text = raw_text or " ".join(tokenized)
    else:
        text = item.get("text") or raw_text
        tokenized = tokenize(text)

    if not tokenized:
        return None

    out: Dict[str, Any] = {"tokenized_text": tokenized, "text": text}

    cls = sample_classification(record.get("classification") or [], rng)
    if cls:
        out["classification"] = cls

    emb = sample_embedding_pair(
        text,
        record.get("positives") or [],
        record.get("negatives") or [],
        rng,
    )
    if emb:
        out["embedding"] = emb

    if kind == "relex":
        extraction = select_extraction(item, rng)
        if extraction:
            out["extraction"] = extraction
    elif kind == "json":
        sampled = select_structuring(structuring_full, rng)
        if sampled:
            out["structuring"] = sampled

    # require at least one task field besides text/tokenized_text
    if not any(k in out for k in ("classification", "embedding", "extraction", "structuring")):
        return None
    return out


def transform(input_path: Path, output_path: Path, n: int, seed: int) -> None:
    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = skipped_records = skipped_combos = 0

    with input_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped_records += 1
                continue

            rid = str(record.get("id", ""))
            if rid.endswith("_relex"):
                kind = "relex"
                structuring_full: Dict[str, List[Dict[str, Any]]] = {}
            elif rid.endswith("_json"):
                kind = "json"
                extracted = (record.get("item") or {}).get("extracted") or {}
                if isinstance(extracted, str):
                    try:
                        extracted = json.loads(extracted)
                    except json.JSONDecodeError:
                        extracted = {}
                structuring_full = build_structuring(extracted) if isinstance(extracted, dict) else {}
            else:
                skipped_records += 1
                continue

            for _ in range(n):
                combo = make_combo(record, structuring_full, kind, rng)
                if combo is None:
                    skipped_combos += 1
                    continue
                fout.write(json.dumps(combo, ensure_ascii=False) + "\n")
                written += 1

    print(
        f"wrote {written} examples to {output_path} "
        f"(skipped {skipped_records} input lines, {skipped_combos} empty combos)",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/home/ingvar/GLiNext/gliner-multitask-data/data/raw_multitask_gemini_lite.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/ingvar/GLiNext/GLiNext/data/multitask_gemini_lite.jsonl"),
    )
    parser.add_argument("-n", "--num-combinations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    transform(args.input, args.output, args.num_combinations, args.seed)


if __name__ == "__main__":
    main()
