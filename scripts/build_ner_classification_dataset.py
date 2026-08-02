#!/usr/bin/env python3
"""Build a mixed GLiNExT NER + text-classification JSONL dataset.

By default this combines the converted ``bio_post_ner`` records with the
``knowledgator/gliclass-v3-logic-dataset`` train split. Classification labels
are cleaned, repaired so every positive is a candidate, capped while retaining
all positives, and shuffled to avoid positional label leakage.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NER_INPUT = REPO_ROOT / "data" / "bio_post_ner_glinext.jsonl"
DEFAULT_CLASSIFICATION_INPUT = REPO_ROOT / "data" / "gliclass_v3_logic_train.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "bio_post_ner_gliclass_logic.jsonl"
CLASSIFICATION_DATASET_ID = "knowledgator/gliclass-v3-logic-dataset"
CLASSIFICATION_REVISION = "ee8e07d42c1f95a4421ae78498eec23aec74f1d7"


@dataclass
class BuildStats:
    ner_records: int = 0
    ner_records_dropped: int = 0
    classification_records_read: int = 0
    classification_records_written: int = 0
    classification_records_dropped: int = 0
    candidate_sets_repaired: int = 0
    candidate_labels_removed: int = 0

    @property
    def total_records(self) -> int:
        return self.ner_records + self.classification_records_written


def iter_json_records(path: Path) -> Iterable[dict[str, Any]]:
    """Yield object records from a JSON array or JSONL file."""
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8-sig") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {path} at line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Expected an object in {path} at line {line_number}."
                    )
                yield record
        return

    with path.open(encoding="utf-8-sig") as input_file:
        records = json.load(input_file)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}.")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Expected an object at index {index} in {path}.")
        yield record


def iter_parquet_records(path: Path) -> Iterable[dict[str, Any]]:
    """Yield only the classification columns from a Parquet file."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required; install the project data dependencies."
        ) from exc

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(
        columns=["text", "true_labels", "all_labels"],
    ):
        yield from batch.to_pylist()


def _clean_labels(value: Any, *, field: str, index: int) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"Classification record {index} has non-list {field}: {value!r}"
        )
    labels = []
    seen = set()
    for raw_label in value:
        if not isinstance(raw_label, str):
            raise ValueError(
                f"Classification record {index} has a non-string {field} label: "
                f"{raw_label!r}"
            )
        label = raw_label.strip()
        if label and label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def convert_classification_record(
    record: dict[str, Any],
    *,
    index: int,
    group_name: str,
    max_candidates: int,
    rng: random.Random,
) -> tuple[dict[str, Any] | None, bool, int]:
    """Convert one GLiClass row, returning row, repair flag, and labels removed."""
    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Classification record {index} has empty or invalid text.")

    true_labels = _clean_labels(record.get("true_labels"), field="true_labels", index=index)
    original_candidates = _clean_labels(
        record.get("all_labels"), field="all_labels", index=index,
    )
    true_set = set(true_labels)
    repaired = not true_set.issubset(original_candidates)

    negative_labels = [
        label for label in original_candidates if label not in true_set
    ]
    rng.shuffle(negative_labels)

    if max_candidates > 0:
        if len(true_labels) > max_candidates:
            raise ValueError(
                f"Classification record {index} has {len(true_labels)} positive "
                f"labels, exceeding max_candidates={max_candidates}."
            )
        negative_labels = negative_labels[: max_candidates - len(true_labels)]

    candidates = [*true_labels, *negative_labels]
    rng.shuffle(candidates)
    if not candidates:
        return None, repaired, 0

    original_unique_count = len(set(original_candidates) | true_set)
    removed = max(0, original_unique_count - len(candidates))
    converted = {
        "text": text,
        "classification": [
            {
                "name": group_name,
                "all_labels": candidates,
                "true_labels": true_labels,
            }
        ],
        "source": {
            "dataset": CLASSIFICATION_DATASET_ID,
            "revision": CLASSIFICATION_REVISION,
            "split": "train",
            "index": index,
        },
    }
    return converted, repaired, removed


def build_dataset(
    ner_input: Path,
    classification_input: Path,
    output: Path,
    *,
    group_name: str = "logic",
    max_candidates: int = 100,
    seed: int = 42,
    shuffle: bool = True,
) -> BuildStats:
    """Build and write the combined task-pure dataset."""
    if not group_name.strip():
        raise ValueError("group_name must not be empty.")
    if max_candidates < 0:
        raise ValueError("max_candidates must be non-negative.")
    if output.resolve() in {ner_input.resolve(), classification_input.resolve()}:
        raise ValueError("Output must differ from both inputs.")

    stats = BuildStats()
    combined = []
    for index, record in enumerate(iter_json_records(ner_input)):
        extraction = record.get("extraction")
        if not isinstance(extraction, list):
            raise ValueError(
                f"NER record {index} is not in GLiNExT extraction format."
            )
        has_supervision = any(
            isinstance(group, dict)
            and isinstance(group.get("ner"), list)
            and bool(group["ner"])
            for group in extraction
        )
        if not has_supervision:
            stats.ner_records_dropped += 1
            continue
        combined.append(record)
        stats.ner_records += 1

    label_rng = random.Random(seed)
    for index, record in enumerate(iter_parquet_records(classification_input)):
        stats.classification_records_read += 1
        converted, repaired, removed = convert_classification_record(
            record,
            index=index,
            group_name=group_name,
            max_candidates=max_candidates,
            rng=label_rng,
        )
        stats.candidate_sets_repaired += int(repaired)
        stats.candidate_labels_removed += removed
        if converted is None:
            stats.classification_records_dropped += 1
            continue
        combined.append(converted)
        stats.classification_records_written += 1

    if shuffle:
        random.Random(seed).shuffle(combined)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as output_file:
        for record in combined:
            json.dump(record, output_file, ensure_ascii=False, separators=(",", ":"))
            output_file.write("\n")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ner-input", type=Path, default=DEFAULT_NER_INPUT)
    parser.add_argument(
        "--classification-input",
        type=Path,
        default=DEFAULT_CLASSIFICATION_INPUT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--group-name", default="logic")
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    stats = build_dataset(
        args.ner_input,
        args.classification_input,
        args.output,
        group_name=args.group_name,
        max_candidates=args.max_candidates,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )
    print(
        f"Wrote {stats.total_records} rows to {args.output}: "
        f"{stats.ner_records} NER + {stats.classification_records_written} "
        f"classification ({stats.ner_records_dropped} empty NER rows and "
        f"{stats.classification_records_dropped} empty classification rows dropped, "
        f"{stats.candidate_sets_repaired} candidate sets repaired, "
        f"{stats.candidate_labels_removed} excess candidates removed)."
    )


if __name__ == "__main__":
    main()
