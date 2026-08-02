#!/usr/bin/env python3
"""Convert the annotated top-1000 extraction set to GLiNExT structuring JSONL.

The source contains ``text`` plus a free-form ``extracted`` JSON value.  A
GLiNExT structuring training/evaluation row only needs these fields::

    {"text": "...", "structuring": {"schema": [{"field": "value"}]}}

Consequently, optional source metadata (``_source``, templates, pre-tokenized
text, and schema metadata) is deliberately not copied.  Missing annotation
values are also omitted: GLiNExT learns extractive spans, so supervising a
``null`` or a value absent from the source text would create an invalid target.

Nested dictionaries are flattened with dotted field names.  Nested lists of
objects cannot be represented as one GLiNExT field and are skipped; top-level
lists of objects become repeated instances of their top-level schema.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "top_1000_en_annoated.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "top_1000_en_glinext.jsonl"
_NULL_STRINGS = {"", "null", "none"}


@dataclass
class ConversionStats:
    rows_seen: int = 0
    rows_written: int = 0
    rows_skipped_invalid: int = 0
    rows_skipped_empty: int = 0
    values_written: int = 0
    values_dropped_null: int = 0
    values_dropped_ungrounded: int = 0
    values_dropped_unsupported: int = 0
    extracted_types: Counter[str] = field(default_factory=Counter)


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield object rows from *path*, reporting malformed input precisely."""

    with path.open("r", encoding="utf-8-sig") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected an object in {path} at line {line_number}, "
                    f"got {type(row).__name__}."
                )
            yield line_number, row


def parse_extracted(value: Any) -> Any:
    """Parse the source's object-or-JSON-string ``extracted`` field."""

    # Some source shards encode the annotation once as JSONL and once again
    # as a JSON string.  Permit repeated string encoding defensively.
    for _ in range(2):
        if not isinstance(value, str):
            break
        stripped = value.strip()
        if not stripped:
            return None
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON string in extracted: {exc.msg}") from exc
    return value


def _scalar_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.casefold() in _NULL_STRINGS:
            return None
        return stripped
    if isinstance(value, (int, float)):
        return str(value)
    return None


def _grounded(value: str, folded_text: str) -> bool:
    return value.casefold() in folded_text


def normalize_field_value(
    value: Any,
    *,
    folded_text: str,
    require_grounding: bool,
    stats: ConversionStats,
) -> str | list[str] | None:
    """Return a GLiNExT scalar/list field value, or ``None`` when unusable."""

    if isinstance(value, list):
        normalized: list[str] = []
        for item in value:
            scalar = _scalar_text(item)
            if scalar is None:
                if item is None or (
                    isinstance(item, str)
                    and item.strip().casefold() in _NULL_STRINGS
                ):
                    stats.values_dropped_null += 1
                else:
                    stats.values_dropped_unsupported += 1
                continue
            if require_grounding and not _grounded(scalar, folded_text):
                stats.values_dropped_ungrounded += 1
                continue
            normalized.append(scalar)
        if not normalized:
            return None
        stats.values_written += len(normalized)
        return normalized

    scalar = _scalar_text(value)
    if scalar is None:
        if value is None or (
            isinstance(value, str)
            and value.strip().casefold() in _NULL_STRINGS
        ):
            stats.values_dropped_null += 1
        else:
            stats.values_dropped_unsupported += 1
        return None
    if require_grounding and not _grounded(scalar, folded_text):
        stats.values_dropped_ungrounded += 1
        return None
    stats.values_written += 1
    return scalar


def flatten_instance(
    value: Any,
    *,
    folded_text: str,
    require_grounding: bool,
    stats: ConversionStats,
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten one object to GLiNExT's flat field/value representation."""

    if not isinstance(value, dict):
        stats.values_dropped_unsupported += 1
        return {}

    flattened: dict[str, Any] = {}
    for raw_key, child in value.items():
        key_part = str(raw_key).strip()
        if not key_part:
            stats.values_dropped_unsupported += 1
            continue
        field_name = f"{prefix}.{key_part}" if prefix else key_part
        if isinstance(child, dict):
            flattened.update(
                flatten_instance(
                    child,
                    folded_text=folded_text,
                    require_grounding=require_grounding,
                    stats=stats,
                    prefix=field_name,
                )
            )
            continue

        normalized = normalize_field_value(
            child,
            folded_text=folded_text,
            require_grounding=require_grounding,
            stats=stats,
        )
        if normalized is not None:
            flattened[field_name] = normalized
    return flattened


def _unique_schema_name(preferred: str, existing: dict[str, Any]) -> str:
    if preferred not in existing:
        return preferred
    index = 2
    while f"{preferred}_{index}" in existing:
        index += 1
    return f"{preferred}_{index}"


def build_structuring(
    extracted: Any,
    text: str,
    *,
    root_schema: str = "record",
    require_grounding: bool = True,
    stats: ConversionStats | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Map free-form extracted JSON to GLiNExT structuring schemas."""

    stats = stats or ConversionStats()
    folded_text = text.casefold()

    if isinstance(extracted, list):
        instances = [
            flatten_instance(
                item,
                folded_text=folded_text,
                require_grounding=require_grounding,
                stats=stats,
            )
            for item in extracted
            if isinstance(item, dict)
        ]
        instances = [instance for instance in instances if instance]
        return {root_schema: instances} if instances else {}

    if not isinstance(extracted, dict):
        return {}

    structuring: dict[str, list[dict[str, Any]]] = {}
    root_fields: dict[str, Any] = {}
    for raw_schema_name, value in extracted.items():
        schema_name = str(raw_schema_name).strip()
        if not schema_name:
            stats.values_dropped_unsupported += 1
            continue

        if isinstance(value, list) and any(isinstance(item, dict) for item in value):
            instances = [
                flatten_instance(
                    item,
                    folded_text=folded_text,
                    require_grounding=require_grounding,
                    stats=stats,
                )
                for item in value
                if isinstance(item, dict)
            ]
            instances = [instance for instance in instances if instance]
            if instances:
                structuring[schema_name] = instances
            continue

        if isinstance(value, dict):
            instance = flatten_instance(
                value,
                folded_text=folded_text,
                require_grounding=require_grounding,
                stats=stats,
            )
            if instance:
                structuring[schema_name] = [instance]
            continue

        normalized = normalize_field_value(
            value,
            folded_text=folded_text,
            require_grounding=require_grounding,
            stats=stats,
        )
        if normalized is not None:
            root_fields[schema_name] = normalized

    if root_fields:
        metadata_schema = _unique_schema_name("metadata", structuring)
        structuring[metadata_schema] = [root_fields]
    return structuring


def convert_record(
    row: dict[str, Any],
    *,
    root_schema: str = "record",
    require_grounding: bool = True,
    stats: ConversionStats | None = None,
) -> dict[str, Any] | None:
    """Convert one source row, returning only mandatory GLiNExT fields."""

    stats = stats or ConversionStats()
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    extracted = parse_extracted(row.get("extracted"))
    stats.extracted_types[type(extracted).__name__] += 1
    structuring = build_structuring(
        extracted,
        text,
        root_schema=root_schema,
        require_grounding=require_grounding,
        stats=stats,
    )
    if not structuring:
        return None
    return {"text": text, "structuring": structuring}


def _write_jsonl_row(output_file: TextIO, row: dict[str, Any]) -> None:
    json.dump(row, output_file, ensure_ascii=False, separators=(",", ":"))
    output_file.write("\n")


def convert_file(
    input_path: Path,
    output_path: Path,
    *,
    root_schema: str = "record",
    require_grounding: bool = True,
) -> ConversionStats:
    """Convert a JSONL file atomically and return conversion statistics."""

    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ.")
    if not root_schema.strip():
        raise ValueError("root_schema must not be empty.")

    stats = ConversionStats()
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
            for line_number, row in read_jsonl(input_path):
                stats.rows_seen += 1
                try:
                    converted = convert_record(
                        row,
                        root_schema=root_schema,
                        require_grounding=require_grounding,
                        stats=stats,
                    )
                except ValueError as exc:
                    stats.rows_skipped_invalid += 1
                    print(
                        f"warning: skipping {input_path}:{line_number}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                if converted is None:
                    stats.rows_skipped_empty += 1
                    continue
                _write_jsonl_row(output_file, converted)
                stats.rows_written += 1

        temporary_path.chmod(0o644)
        temporary_path.replace(output_path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--root-schema",
        default="record",
        help="Schema name for the rare source rows whose extracted root is a list.",
    )
    parser.add_argument(
        "--keep-ungrounded",
        action="store_true",
        help="Keep values absent from the source text (not recommended for span training).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        stats = convert_file(
            args.input,
            args.output,
            root_schema=args.root_schema,
            require_grounding=not args.keep_ungrounded,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Wrote {stats.rows_written}/{stats.rows_seen} rows to {args.output}; "
        f"skipped invalid={stats.rows_skipped_invalid}, "
        f"empty={stats.rows_skipped_empty}. Values: kept={stats.values_written}, "
        f"null={stats.values_dropped_null}, "
        f"ungrounded={stats.values_dropped_ungrounded}, "
        f"unsupported={stats.values_dropped_unsupported}.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
