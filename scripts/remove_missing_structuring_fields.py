#!/usr/bin/env python3
"""Clean structuring JSONL rows by removing fields whose values are absent.

The script streams an input JSONL file and writes a cleaned JSONL file. For each
row it checks every scalar leaf value under ``structuring`` against the row's
``text`` field.

Default behavior:
  * if more than 50% of scalar field values are missing from text, drop the row;
  * otherwise, remove only the individual field value that is missing. Valid
    values for the same field in other records are preserved.

By default matching is exact substring matching, which matches the analysis
previously run on ``data/structuring_synthetic.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: Any) -> str:
    return WHITESPACE_RE.sub(" ", str(value)).strip().casefold()


def iter_scalar_fields(
    value: Any,
    path: tuple[str | int, ...] = (),
) -> Iterable[tuple[tuple[str | int, ...], Any]]:
    """Yield ``(path, scalar_value)`` pairs for scalar leaves in nested JSON."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from iter_scalar_fields(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_scalar_fields(child, (*path, index))
    else:
        yield path, value


def remove_leaf_paths(
    value: Any,
    paths_to_remove: set[tuple[str | int, ...]],
    path: tuple[str | int, ...] = (),
) -> Any:
    """Return a copy of ``value`` without the selected scalar leaf paths."""
    if isinstance(value, dict):
        cleaned = {}
        for key, child in value.items():
            child_path = (*path, str(key))
            if child_path in paths_to_remove:
                continue
            cleaned[key] = remove_leaf_paths(
                child, paths_to_remove, child_path,
            )
        return cleaned
    if isinstance(value, list):
        cleaned = []
        for index, child in enumerate(value):
            child_path = (*path, index)
            if child_path in paths_to_remove:
                continue
            cleaned.append(remove_leaf_paths(
                child, paths_to_remove, child_path,
            ))
        return cleaned
    return value


def prune_empty_containers(value: Any) -> Any:
    """Remove empty dicts/lists left after field removal."""
    if isinstance(value, dict):
        pruned = {}
        for key, child in value.items():
            cleaned_child = prune_empty_containers(child)
            if cleaned_child == {} or cleaned_child == []:
                continue
            pruned[key] = cleaned_child
        return pruned
    if isinstance(value, list):
        pruned_list = []
        for child in value:
            cleaned_child = prune_empty_containers(child)
            if cleaned_child == {} or cleaned_child == []:
                continue
            pruned_list.append(cleaned_child)
        return pruned_list
    return value


def has_scalar_fields(value: Any) -> bool:
    for path, scalar_value in iter_scalar_fields(value):
        if path and scalar_value is not None and str(scalar_value) != "":
            return True
    return False


def structuring_instance_count(value: Any) -> int:
    """Return the largest schema-record count represented by a row."""

    if isinstance(value, list):
        return len(value)
    if not isinstance(value, dict):
        return 0
    return max(
        (len(instances) for instances in value.values()
         if isinstance(instances, list)),
        default=0,
    )


def finalize_cleaned_row(
    row: dict[str, Any],
    structuring: Any,
    stats: Counter[str],
    *,
    total_values: int,
    max_instances: int | None,
) -> tuple[dict[str, Any] | None, Counter[str]]:
    """Apply capacity/metadata checks and classify a retained row."""

    instance_count = structuring_instance_count(structuring)
    if max_instances is not None and instance_count > max_instances:
        stats["rows_dropped_over_capacity"] += 1
        stats["rows_dropped"] += 1
        stats["values_removed_by_dropped_rows"] += total_values
        return None, stats

    metadata_needs_update = (
        "n_objects_extracted" in row
        and row.get("n_objects_extracted") != instance_count
    )
    if structuring != row.get("structuring") or metadata_needs_update:
        cleaned = deepcopy(row)
        cleaned["structuring"] = structuring
        if "n_objects_extracted" in cleaned:
            cleaned["n_objects_extracted"] = instance_count
        if metadata_needs_update:
            stats["metadata_counts_updated"] += 1
        stats["rows_written_cleaned"] += 1
        return cleaned, stats

    stats["rows_written_unchanged"] += 1
    return row, stats


def value_is_present(value: Any, text: str, normalized_text: str | None, match_mode: str) -> bool:
    candidate = str(value)
    if match_mode == "exact":
        return candidate in text
    if normalized_text is None:
        normalized_text = normalize_text(text)
    return normalize_text(candidate) in normalized_text


def clean_row(
    row: dict[str, Any],
    *,
    drop_row_threshold: float,
    match_mode: str,
    max_instances: int | None = None,
) -> tuple[dict[str, Any] | None, Counter[str]]:
    """Clean one JSON row and return ``(cleaned_row, stats)``.

    ``cleaned_row`` is ``None`` when the row should be dropped.
    """
    stats: Counter[str] = Counter(rows_seen=1)
    structuring = row.get("structuring")
    if not isinstance(structuring, (dict, list)):
        stats["rows_without_structuring"] += 1
        return row, stats

    text = row.get("text") or ""
    normalized_text = normalize_text(text) if match_mode == "normalized" else None
    missing_paths: set[tuple[str | int, ...]] = set()
    total_values = 0
    missing_values = 0

    for path, value in iter_scalar_fields(structuring):
        if value is None or str(value) == "":
            continue
        total_values += 1
        if not value_is_present(value, text, normalized_text, match_mode):
            missing_values += 1
            if path:
                missing_paths.add(path)

    stats["values_seen"] += total_values
    stats["values_missing"] += missing_values
    stats["field_values_marked_for_removal"] += len(missing_paths)

    if total_values == 0:
        stats["rows_without_scalar_values"] += 1
        stats["rows_dropped_empty_after_cleaning"] += 1
        stats["rows_dropped"] += 1
        stats["values_removed_by_dropped_rows"] += total_values
        return None, stats

    pruned_structuring = prune_empty_containers(structuring)
    if not has_scalar_fields(pruned_structuring):
        stats["rows_dropped_empty_after_cleaning"] += 1
        stats["rows_dropped"] += 1
        stats["values_removed_by_dropped_rows"] += total_values
        return None, stats

    if not missing_paths:
        return finalize_cleaned_row(
            row,
            pruned_structuring,
            stats,
            total_values=total_values,
            max_instances=max_instances,
        )

    missing_ratio = missing_values / total_values
    if missing_ratio > drop_row_threshold:
        stats["rows_dropped"] += 1
        stats["values_removed_by_dropped_rows"] += total_values
        return None, stats

    cleaned = deepcopy(row)
    cleaned["structuring"] = prune_empty_containers(
        remove_leaf_paths(cleaned["structuring"], missing_paths)
    )
    if not has_scalar_fields(cleaned["structuring"]):
        stats["rows_dropped_empty_after_cleaning"] += 1
        stats["rows_dropped"] += 1
        stats["values_removed_by_dropped_rows"] += total_values
        return None, stats
    return finalize_cleaned_row(
        row,
        cleaned["structuring"],
        stats,
        total_values=total_values,
        max_instances=max_instances,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove structuring fields whose values are absent from row text.",
    )
    parser.add_argument("input", type=Path, help="Input JSONL file.")
    parser.add_argument("output", type=Path, help="Output JSONL file.")
    parser.add_argument(
        "--drop-row-threshold",
        type=float,
        default=0.5,
        help="Drop a row when missing scalar field values / total scalar field values is greater than this value. Default: 0.5.",
    )
    parser.add_argument(
        "--match-mode",
        choices=("exact", "normalized"),
        default="exact",
        help="Use exact substring matching or case-insensitive whitespace-normalized matching. Default: exact.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Drop rows whose largest schema exceeds this record capacity.",
    )
    parser.add_argument(
        "--keep-duplicate-texts",
        action="store_true",
        help="Keep repeated passage text instead of retaining only its first row.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute stats without writing the output file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.drop_row_threshold <= 1:
        print("--drop-row-threshold must be between 0 and 1", file=sys.stderr)
        return 2
    if args.max_instances is not None and args.max_instances <= 0:
        print("--max-instances must be positive", file=sys.stderr)
        return 2

    totals: Counter[str] = Counter()
    seen_texts: set[bytes] = set()
    output_handle = None
    try:
        if not args.dry_run:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output_handle = args.output.open("w", encoding="utf-8")

        with args.input.open("r", encoding="utf-8") as input_handle:
            for line_number, line in enumerate(input_handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Line {line_number} is not a JSON object")

                cleaned_row, stats = clean_row(
                    row,
                    drop_row_threshold=args.drop_row_threshold,
                    match_mode=args.match_mode,
                    max_instances=args.max_instances,
                )
                if cleaned_row is not None and not args.keep_duplicate_texts:
                    row_text = cleaned_row.get("text")
                    if isinstance(row_text, str):
                        fingerprint = hashlib.sha256(
                            row_text.encode("utf-8")
                        ).digest()
                        if fingerprint in seen_texts:
                            stats["rows_written_cleaned"] = 0
                            stats["rows_written_unchanged"] = 0
                            stats["rows_dropped_duplicate_text"] += 1
                            stats["rows_dropped"] += 1
                            cleaned_row = None
                        else:
                            seen_texts.add(fingerprint)
                totals.update(stats)
                if cleaned_row is not None and output_handle is not None:
                    output_handle.write(json.dumps(cleaned_row, ensure_ascii=False) + "\n")
                    totals["rows_written"] += 1
    finally:
        if output_handle is not None:
            output_handle.close()

    rows_seen = totals["rows_seen"]
    values_seen = totals["values_seen"]
    values_missing = totals["values_missing"]
    missing_pct = (100 * values_missing / values_seen) if values_seen else 0.0

    print(f"rows_seen={rows_seen}")
    print(f"rows_written={totals['rows_written'] if not args.dry_run else rows_seen - totals['rows_dropped']}")
    print(f"rows_dropped={totals['rows_dropped']}")
    print(f"rows_dropped_empty_after_cleaning={totals['rows_dropped_empty_after_cleaning']}")
    print(f"rows_dropped_over_capacity={totals['rows_dropped_over_capacity']}")
    print(f"rows_dropped_duplicate_text={totals['rows_dropped_duplicate_text']}")
    print(f"rows_written_unchanged={totals['rows_written_unchanged']}")
    print(f"rows_written_cleaned={totals['rows_written_cleaned']}")
    print(f"values_seen={values_seen}")
    print(f"values_missing={values_missing}")
    print(f"values_missing_pct={missing_pct:.2f}")
    print(f"field_values_marked_for_removal={totals['field_values_marked_for_removal']}")
    print(f"metadata_counts_updated={totals['metadata_counts_updated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
