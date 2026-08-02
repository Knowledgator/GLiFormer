#!/usr/bin/env python3
"""Build a grounded, deduplicated GLiNext multi-level structuring dataset.

The raw files are expected to contain ``text`` and ``extracted`` fields.
``extracted`` may already be JSON or may be a JSON-encoded string.  The output
contains only GLiNext's public training fields::

    {"text": "...", "structuring": {...}}

An example is retained only when every non-empty scalar annotation can be
resolved as a case-insensitive exact substring of its text and its structure
contains at least one list with an object member.  Nulls and empty strings are
treated as missing annotations rather than spans to ground.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "structuring"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "structuring_multi_level.jsonl"


def parse_structuring(row: dict[str, Any]) -> dict[str, Any] | list[Any] | None:
    """Return a JSON object/list from a raw row, or ``None`` when unusable."""

    value = row.get("extracted")
    if value is None and "output" in row:
        value = row["output"]
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    return value if isinstance(value, (dict, list)) else None


def iter_scalar_values(
    value: Any,
    path: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Yield non-null, non-empty scalar leaves and their JSON paths."""

    if isinstance(value, dict):
        for key, child in value.items():
            yield from iter_scalar_values(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_scalar_values(child, path + (str(index),))
    elif value is not None and value != "":
        yield path, value


def first_ungrounded_value(
    text: str,
    structuring: dict[str, Any] | list[Any],
) -> tuple[tuple[str, ...], Any] | None:
    """Return the first leaf that GLiNext cannot ground in ``text``."""

    folded_text = text.casefold()
    found_scalar = False
    for path, value in iter_scalar_values(structuring):
        found_scalar = True
        if str(value).casefold() not in folded_text:
            return path, value
    if not found_scalar:
        return (), None
    return None


def contains_object_list(value: Any) -> bool:
    """Whether ``value`` contains a list with at least one dictionary member."""

    if isinstance(value, dict):
        return any(contains_object_list(child) for child in value.values())
    if isinstance(value, list):
        return any(isinstance(child, dict) for child in value) or any(
            contains_object_list(child) for child in value
        )
    return False


def canonical_example_digest(
    text: str,
    structuring: dict[str, Any] | list[Any],
) -> bytes:
    """Hash semantic example content while ignoring dictionary key order."""

    encoded_text = text.encode("utf-8")
    encoded_structuring = json.dumps(
        structuring,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(encoded_text).to_bytes(8, "big"))
    digest.update(encoded_text)
    digest.update(encoded_structuring)
    return digest.digest()


def iter_input_paths(input_dir: Path, output: Path) -> Iterable[Path]:
    """Yield source JSONL files without ever treating the output as input."""

    output = output.resolve()
    for path in sorted(input_dir.glob("*.jsonl")):
        if path.resolve() != output:
            yield path


def build_dataset(
    input_dir: Path,
    output: Path,
    *,
    force: bool = False,
    supplement: Path | None = None,
    supplement_count: int = 0,
    seed: int = 42,
) -> Counter:
    """Convert and atomically write the filtered dataset."""

    input_dir = input_dir.resolve()
    output = output.resolve()
    supplement = supplement.resolve() if supplement is not None else None
    if supplement_count < 0:
        raise ValueError("supplement_count must be non-negative")
    if supplement_count and supplement is None:
        raise ValueError("supplement_count requires a supplement path")
    if supplement is not None and supplement == output:
        raise ValueError("Supplement path must not be the output path")
    input_paths = list(iter_input_paths(input_dir, output))
    if not input_paths:
        raise FileNotFoundError(f"No JSONL inputs found in {input_dir}")
    if output.exists() and not force:
        raise FileExistsError(f"Output already exists (use --force): {output}")
    if any(path.resolve() == output for path in input_paths):
        raise ValueError("Output path must not overwrite a raw input file")

    output.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    seen: set[bytes] = set()
    written_identities: set[bytes] = set()

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as destination:
        temporary_path = Path(destination.name)
        try:
            for input_path in input_paths:
                with input_path.open(encoding="utf-8-sig") as source:
                    for line_number, line in enumerate(source, 1):
                        stats["input_rows"] += 1
                        row = json.loads(line)
                        if not isinstance(row, dict) or not isinstance(
                            row.get("text"), str
                        ):
                            raise ValueError(
                                f"Expected an object with string text at "
                                f"{input_path}:{line_number}"
                            )

                        structuring = parse_structuring(row)
                        if structuring is None:
                            stats["invalid_structuring"] += 1
                            continue

                        text = row["text"]
                        identity = canonical_example_digest(text, structuring)
                        if identity in seen:
                            stats["duplicates_removed"] += 1
                            continue
                        seen.add(identity)

                        if first_ungrounded_value(text, structuring) is not None:
                            stats["ungrounded_examples_removed"] += 1
                            continue
                        if not contains_object_list(structuring):
                            stats["single_level_examples_removed"] += 1
                            continue

                        destination.write(
                            json.dumps(
                                {"text": text, "structuring": structuring},
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                        )
                        destination.write("\n")
                        written_identities.add(identity)
                        stats["base_output_rows"] += 1
                        stats["output_rows"] += 1

            if supplement_count:
                if not supplement.is_file():
                    raise FileNotFoundError(
                        f"Supplement JSONL does not exist: {supplement}"
                    )
                rng = random.Random(seed)
                supplement_seen: set[bytes] = set()
                sample: list[dict[str, Any]] = []
                eligible_count = 0
                with supplement.open(encoding="utf-8-sig") as source:
                    for line_number, line in enumerate(source, 1):
                        stats["supplement_input_rows"] += 1
                        row = json.loads(line)
                        text = row.get("text") if isinstance(row, dict) else None
                        structuring = (
                            row.get("structuring")
                            if isinstance(row, dict)
                            else None
                        )
                        if not isinstance(text, str) or not isinstance(
                            structuring, (dict, list)
                        ):
                            stats["supplement_invalid_rows"] += 1
                            continue

                        identity = canonical_example_digest(text, structuring)
                        if identity in written_identities:
                            stats["supplement_overlap_removed"] += 1
                            continue
                        if identity in supplement_seen:
                            stats["supplement_duplicates_removed"] += 1
                            continue
                        supplement_seen.add(identity)

                        if first_ungrounded_value(text, structuring) is not None:
                            stats["supplement_ungrounded_removed"] += 1
                            continue
                        if not contains_object_list(structuring):
                            stats["supplement_single_level_removed"] += 1
                            continue

                        eligible_count += 1
                        candidate = {"text": text, "structuring": structuring}
                        if len(sample) < supplement_count:
                            sample.append(candidate)
                        else:
                            index = rng.randrange(eligible_count)
                            if index < supplement_count:
                                sample[index] = candidate

                if len(sample) != supplement_count:
                    raise ValueError(
                        f"Requested {supplement_count} supplement rows, but "
                        f"only {eligible_count} unique grounded rows are eligible"
                    )
                stats["supplement_eligible_rows"] = eligible_count
                for row in sample:
                    destination.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                    destination.write("\n")
                    stats["supplement_rows_added"] += 1
                    stats["output_rows"] += 1

            destination.flush()
            os.fsync(destination.fileno())
            os.chmod(temporary_path, 0o644)
            os.replace(temporary_path, output)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    stats["input_files"] = len(input_paths)
    stats["output_bytes"] = output.stat().st_size
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--supplement",
        type=Path,
        help="Optional GLiNext JSONL source for deterministic augmentation.",
    )
    parser.add_argument(
        "--supplement-count",
        type=int,
        default=0,
        help="Number of unique grounded supplement rows to add.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for supplement reservoir sampling (default: 42).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing derived output; raw inputs remain protected.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_dataset(
        args.input_dir,
        args.output,
        force=args.force,
        supplement=args.supplement,
        supplement_count=args.supplement_count,
        seed=args.seed,
    )
    print(json.dumps(dict(stats), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
