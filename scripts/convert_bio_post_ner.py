#!/usr/bin/env python3
"""Convert legacy GLiNER NER rows into GLiNExT extraction JSONL.

The source records use the legacy GLiNER shape::

    {"tokenized_text": [...], "ner": [[start, end, label], ...]}

GLiNExT keeps the same inclusive token offsets but nests NER annotations in an
``extraction`` group. Extra source fields such as ``metadata`` and
``negatives`` are preserved.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "bio_post_ner.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "bio_post_ner_glinext.jsonl"


@dataclass
class ConversionStats:
    records_read: int = 0
    records_written: int = 0
    records_dropped: int = 0
    spans_written: int = 0
    spans_dropped: int = 0


def load_records(path: Path) -> Iterable[dict[str, Any]]:
    """Load a JSON array or yield non-empty JSONL records."""
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
                        f"Expected an object in {path} at line {line_number}, "
                        f"got {type(record).__name__}."
                    )
                yield record
        return

    with path.open(encoding="utf-8-sig") as input_file:
        records = json.load(input_file)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}.")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                f"Expected an object at index {index} in {path}, "
                f"got {type(record).__name__}."
            )
        yield record


def _valid_span(span: Any, token_count: int) -> bool:
    return (
        isinstance(span, (list, tuple))
        and len(span) == 3
        and isinstance(span[0], int)
        and not isinstance(span[0], bool)
        and isinstance(span[1], int)
        and not isinstance(span[1], bool)
        and isinstance(span[2], str)
        and bool(span[2])
        and 0 <= span[0] <= span[1] < token_count
    )


def convert_record(
    record: dict[str, Any],
    *,
    index: int,
    group_name: str,
    invalid_spans: str,
) -> tuple[dict[str, Any], int]:
    """Convert one legacy row and return it with its dropped-span count."""
    tokens = record.get("tokenized_text")
    spans = record.get("ner")
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise ValueError(f"Record {index} has invalid or missing tokenized_text.")
    if not tokens:
        raise ValueError(f"Record {index} has empty tokenized_text.")
    if not isinstance(spans, list):
        raise ValueError(f"Record {index} has invalid or missing ner spans.")
    if "extraction" in record:
        raise ValueError(f"Record {index} already contains an extraction field.")

    valid_spans = []
    dropped = 0
    for span_index, span in enumerate(spans):
        if not _valid_span(span, len(tokens)):
            if invalid_spans == "error":
                raise ValueError(
                    f"Invalid span in record {index} at span index {span_index}: {span!r}"
                )
            dropped += 1
            continue
        valid_spans.append([span[0], span[1], span[2]])

    valid_spans.sort(key=lambda span: (span[0], span[1], span[2]))
    all_labels = sorted(
        {span[2] for span in valid_spans},
        key=lambda label: (label.casefold(), label),
    )

    converted = {key: value for key, value in record.items() if key != "ner"}
    converted["extraction"] = [
        {
            "name": group_name,
            "all_labels": all_labels,
            "ner": valid_spans,
        }
    ]
    converted["_glinext_extraction_spans_resolved"] = True
    return converted, dropped


def convert_file(
    input_path: Path,
    output_path: Path,
    *,
    group_name: str = "entities",
    invalid_spans: str = "drop",
) -> ConversionStats:
    """Convert all records to newline-delimited GLiNExT JSON."""
    if invalid_spans not in {"drop", "error"}:
        raise ValueError("invalid_spans must be 'drop' or 'error'.")
    if not group_name.strip():
        raise ValueError("group_name must not be empty.")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = ConversionStats()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            for index, record in enumerate(load_records(input_path)):
                stats.records_read += 1
                converted, dropped = convert_record(
                    record,
                    index=index,
                    group_name=group_name,
                    invalid_spans=invalid_spans,
                )
                stats.spans_dropped += dropped
                if not converted["extraction"][0]["ner"]:
                    stats.records_dropped += 1
                    continue
                json.dump(
                    converted,
                    output_file,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                output_file.write("\n")
                stats.records_written += 1
                stats.spans_written += len(converted["extraction"][0]["ner"])
        temporary_path.chmod(0o644)
        temporary_path.replace(output_path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--group-name", default="entities")
    parser.add_argument(
        "--invalid-spans",
        choices=("drop", "error"),
        default="drop",
        help="Drop malformed/out-of-range spans or stop at the first one.",
    )
    args = parser.parse_args()

    stats = convert_file(
        args.input,
        args.output,
        group_name=args.group_name,
        invalid_spans=args.invalid_spans,
    )
    print(
        f"Converted {stats.records_written}/{stats.records_read} records to "
        f"{args.output} ({stats.spans_written} spans written, "
        f"{stats.spans_dropped} invalid spans and "
        f"{stats.records_dropped} zero-supervision records dropped)."
    )


if __name__ == "__main__":
    main()
