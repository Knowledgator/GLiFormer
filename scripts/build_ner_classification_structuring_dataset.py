#!/usr/bin/env python3
"""Build a task-pure GLiNExT NER + classification + structuring JSONL dataset."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, TextIO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NER_CLASSIFICATION_INPUT = (
    REPO_ROOT / "data" / "bio_post_ner_gliclass_logic.jsonl"
)
DEFAULT_STRUCTURING_INPUT = (
    REPO_ROOT / "data" / "structuring_synthetic.cleaned.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "data" / "bio_post_ner_gliclass_logic_structuring.jsonl"
)


@dataclass
class BuildStats:
    ner_records: int = 0
    classification_records: int = 0
    structuring_records: int = 0
    max_structuring_schemas: int = 0
    max_structuring_instances: int = 0
    max_structuring_fields: int = 0

    @property
    def total_records(self) -> int:
        return self.ner_records + self.classification_records + self.structuring_records


def _read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as input_file:
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
            yield line_number, record


def _validate_ner_classification_record(
    record: dict[str, Any],
    *,
    path: Path,
    line_number: int,
) -> str:
    has_ner = isinstance(record.get("extraction"), list)
    has_classification = isinstance(record.get("classification"), list)
    if has_ner == has_classification or "structuring" in record:
        raise ValueError(
            f"Expected one task-pure NER or classification record in {path} "
            f"at line {line_number}."
        )
    if has_ner and not record["extraction"]:
        raise ValueError(f"Empty extraction list in {path} at line {line_number}.")
    if has_ner and not any(
        isinstance(group, dict)
        and isinstance(group.get("ner"), list)
        and bool(group["ner"])
        for group in record["extraction"]
    ):
        raise ValueError(
            f"NER record has no supervised spans in {path} at line {line_number}."
        )
    if has_classification and not record["classification"]:
        raise ValueError(
            f"Empty classification list in {path} at line {line_number}."
        )
    return "ner" if has_ner else "classification"


def _validate_structuring_record(
    record: dict[str, Any],
    *,
    path: Path,
    line_number: int,
) -> tuple[int, int, int]:
    text = record.get("text")
    tokenized_text = record.get("tokenized_text")
    structuring = record.get("structuring")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Empty or invalid text in {path} at line {line_number}.")
    if not isinstance(tokenized_text, list) or not tokenized_text:
        raise ValueError(
            f"Empty or invalid tokenized_text in {path} at line {line_number}."
        )
    if not isinstance(structuring, dict) or not structuring:
        raise ValueError(
            f"Empty or invalid structuring object in {path} at line {line_number}."
        )
    if "extraction" in record or "classification" in record:
        raise ValueError(
            f"Structuring row is not task-pure in {path} at line {line_number}."
        )

    max_instances = 0
    max_fields = 0
    for schema_name, instances in structuring.items():
        if not isinstance(schema_name, str) or not schema_name.strip():
            raise ValueError(
                f"Invalid structuring schema name in {path} at line {line_number}."
            )
        if not isinstance(instances, list) or not instances:
            raise ValueError(
                f"Schema {schema_name!r} has no instances in {path} at line "
                f"{line_number}."
            )
        max_instances = max(max_instances, len(instances))
        for instance in instances:
            if not isinstance(instance, dict) or not instance:
                raise ValueError(
                    f"Schema {schema_name!r} has an invalid instance in {path} "
                    f"at line {line_number}."
                )
            if any(not isinstance(field, str) or not field for field in instance):
                raise ValueError(
                    f"Schema {schema_name!r} has an invalid field name in {path} "
                    f"at line {line_number}."
                )
            max_fields = max(max_fields, len(instance))
    return len(structuring), max_instances, max_fields


def _write_record(output_file: TextIO, record: dict[str, Any]) -> None:
    json.dump(record, output_file, ensure_ascii=False, separators=(",", ":"))
    output_file.write("\n")


def build_dataset(
    ner_classification_input: Path,
    structuring_input: Path,
    output: Path,
    *,
    structuring_per_base: int = 3,
) -> BuildStats:
    """Validate and interleave the three tasks into one JSONL file.

    The existing NER/classification file is already deterministically shuffled.
    Interleaving three structuring rows per base row keeps all three tasks spread
    through almost the entire output while using constant memory. ``train.py``
    performs its own example-level shuffle before training.
    """
    if structuring_per_base <= 0:
        raise ValueError("structuring_per_base must be positive.")
    resolved_output = output.resolve()
    if resolved_output in {
        ner_classification_input.resolve(),
        structuring_input.resolve(),
    }:
        raise ValueError("Output must differ from both inputs.")

    stats = BuildStats()
    structuring_records = _read_jsonl(structuring_input)
    output.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)

            def write_structuring(line_number: int, record: dict[str, Any]) -> None:
                schemas, instances, fields = _validate_structuring_record(
                    record,
                    path=structuring_input,
                    line_number=line_number,
                )
                stats.structuring_records += 1
                stats.max_structuring_schemas = max(
                    stats.max_structuring_schemas, schemas,
                )
                stats.max_structuring_instances = max(
                    stats.max_structuring_instances, instances,
                )
                stats.max_structuring_fields = max(
                    stats.max_structuring_fields, fields,
                )
                _write_record(output_file, record)

            for line_number, record in _read_jsonl(ner_classification_input):
                task = _validate_ner_classification_record(
                    record,
                    path=ner_classification_input,
                    line_number=line_number,
                )
                if task == "ner":
                    stats.ner_records += 1
                else:
                    stats.classification_records += 1
                _write_record(output_file, record)

                for _ in range(structuring_per_base):
                    try:
                        struct_line, struct_record = next(structuring_records)
                    except StopIteration:
                        break
                    write_structuring(struct_line, struct_record)

            for struct_line, struct_record in structuring_records:
                write_structuring(struct_line, struct_record)

        temporary_path.chmod(0o644)
        temporary_path.replace(output)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ner-classification-input",
        type=Path,
        default=DEFAULT_NER_CLASSIFICATION_INPUT,
    )
    parser.add_argument(
        "--structuring-input",
        type=Path,
        default=DEFAULT_STRUCTURING_INPUT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--structuring-per-base", type=int, default=3)
    args = parser.parse_args()

    stats = build_dataset(
        args.ner_classification_input,
        args.structuring_input,
        args.output,
        structuring_per_base=args.structuring_per_base,
    )
    print(
        f"Wrote {stats.total_records} rows to {args.output}: "
        f"{stats.ner_records} NER + {stats.classification_records} "
        f"classification + {stats.structuring_records} structuring. "
        f"Structuring maxima: {stats.max_structuring_schemas} schemas, "
        f"{stats.max_structuring_instances} instances, "
        f"{stats.max_structuring_fields} fields."
    )


if __name__ == "__main__":
    main()
