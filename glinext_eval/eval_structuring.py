#!/usr/bin/env python3
"""Evaluate a GLiNExT checkpoint on a structuring JSONL dataset.

Unlike a causal language model, GLiNExT does not consume a rendered JSON
template or generate JSON text.  This evaluator infers the minimal per-example
``structures`` argument from fields that are actually present in the target,
calls the native ``model.structure`` API, and computes JSON structure, value,
accuracy, precision, recall, and F1 metrics comparable to the generative
evaluation script.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def load_jsonl(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    """Load object rows from JSONL, warning and continuing on invalid lines."""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping invalid JSON at line %d: %s", line_number, exc)
                continue
            if not isinstance(record, dict):
                LOGGER.warning("Skipping non-object JSON at line %d", line_number)
                continue
            records.append(record)
            if max_samples is not None and len(records) >= max_samples:
                break
    LOGGER.info("Loaded %d examples from %s", len(records), path)
    return records


def try_parse_json(value: str) -> tuple[Any, bool]:
    """Parse strings that are themselves complete JSON scalar/container values."""

    if not isinstance(value, str):
        return value, False
    stripped = value.strip()
    if not stripped:
        return value, False
    if (
        stripped[0] in '{["'
        or stripped in {"true", "false", "null"}
        or stripped.lstrip("-").replace(".", "", 1).isdigit()
    ):
        try:
            return json.loads(stripped), True
        except (json.JSONDecodeError, ValueError):
            pass
    return value, False


def normalize_value(value: Any) -> Any:
    """Recursively normalize nested JSON strings and scalar encodings."""

    if isinstance(value, str):
        parsed, success = try_parse_json(value)
        return normalize_value(parsed) if success else value
    if isinstance(value, dict):
        return {str(key): normalize_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [normalize_value(child) for child in value]
    return value


_MISSING = object()


def _prune_optional(value: Any, *, root: bool = False) -> Any:
    """Remove null/empty optional values while retaining meaningful scalars."""

    if value is None:
        return _MISSING
    if isinstance(value, str) and value.strip().casefold() in {"", "null", "none"}:
        return _MISSING
    if isinstance(value, dict):
        cleaned = {}
        for key, child in value.items():
            normalized = _prune_optional(child)
            if normalized is not _MISSING:
                cleaned[str(key)] = normalized
        if cleaned or root:
            return cleaned
        return _MISSING
    if isinstance(value, list):
        cleaned_list = []
        for child in value:
            normalized = _prune_optional(child)
            if normalized is not _MISSING:
                cleaned_list.append(normalized)
        return cleaned_list if cleaned_list or root else _MISSING
    return value


def canonicalize_json(value: Any) -> Any:
    """Normalize scalar encodings and discard optional null/empty fields."""

    cleaned = _prune_optional(normalize_value(value), root=True)
    return {} if cleaned is _MISSING else cleaned


def _schema_value_kind(value: Any) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return "scalar"


def _union_schema_exemplar(values: list[Any]) -> Any:
    """Build one deterministic placeholder exemplar from observed values.

    Dictionary keys retain first-occurrence order. Lists retain one exemplar
    for each encountered member kind, while dictionaries within a list are
    merged recursively. This is enough to expose every field path to the
    multi-level structuring processor without leaking target values into the
    inference prompt.
    """

    if not values:
        return ""

    kinds = [_schema_value_kind(value) for value in values]
    if all(kind == "object" for kind in kinds):
        values_by_key: dict[str, list[Any]] = {}
        for value in values:
            for raw_key, child in value.items():
                key = str(raw_key)
                values_by_key.setdefault(key, []).append(child)
        return {
            key: _union_schema_exemplar(children)
            for key, children in values_by_key.items()
        }

    if all(kind == "array" for kind in kinds):
        members = [member for value in values for member in value]
        if not members:
            return []

        members_by_kind: dict[str, list[Any]] = {}
        for member in members:
            kind = _schema_value_kind(member)
            members_by_kind.setdefault(kind, []).append(member)
        return [
            _union_schema_exemplar(kind_members)
            for kind_members in members_by_kind.values()
        ]

    # A single inference exemplar cannot express a scalar/object union at one
    # JSON key. Follow first occurrence deterministically; list-member unions
    # are handled above, where multiple representative members are legal.
    first_kind = kinds[0]
    same_kind = [
        value for value, kind in zip(values, kinds, strict=True)
        if kind == first_kind
    ]
    if first_kind in {"object", "array"}:
        return _union_schema_exemplar(same_kind)
    return ""


def _is_flat_record(record: dict[str, Any]) -> bool:
    """Return whether the historical field-name-only schema is sufficient."""

    return all(
        not isinstance(value, dict | list)
        for value in record.values()
    )


def _is_named_schema_envelope(value: Any) -> bool:
    """Recognize the historical ``{schema: [record, ...]}`` target shape."""

    return bool(value) and isinstance(value, dict) and all(
        isinstance(instances, list)
        and instances
        and all(isinstance(instance, dict) for instance in instances)
        for instances in value.values()
    )


def infer_structures(structuring: Any) -> Any:
    """Infer a native flat or multi-level ``structures`` argument.

    Historical named schemas containing only scalar fields keep their compact
    ``{schema: [field, ...]}`` representation. Nested named schemas receive a
    one-record union exemplar. Arbitrary object roots use the public ``$root``
    discriminator, and list roots retain their list shape.
    """

    canonical = canonicalize_json(structuring)
    if isinstance(canonical, list):
        return _union_schema_exemplar([canonical]) if canonical else []
    if not isinstance(canonical, dict) or not canonical:
        return {}

    if not _is_named_schema_envelope(canonical):
        return {"$root": _union_schema_exemplar([canonical])}

    structures: dict[str, Any] = {}
    for raw_schema_name, instances in canonical.items():
        schema_name = str(raw_schema_name)
        if all(_is_flat_record(instance) for instance in instances):
            fields: list[str] = []
            seen: set[str] = set()
            for instance in instances:
                for raw_field_name in instance:
                    field_name = str(raw_field_name)
                    if field_name not in seen:
                        seen.add(field_name)
                        fields.append(field_name)
            if fields:
                structures[schema_name] = fields
            continue

        structures[schema_name] = [
            _union_schema_exemplar(list(instances))
        ]
    return structures


def prepare_test_data(path: Path, max_samples: int | None = 30) -> list[dict[str, Any]]:
    """Load minimal GLiNExT rows and infer their native inference schemas."""

    prepared: list[dict[str, Any]] = []
    for record in load_jsonl(path, max_samples=max_samples):
        text = record.get("text")
        solution = canonicalize_json(record.get("structuring"))
        if not isinstance(text, str) or not text.strip():
            LOGGER.warning("Skipping example with missing text")
            continue
        if not isinstance(solution, dict | list) or not solution:
            LOGGER.warning("Skipping example with missing structuring target")
            continue
        structures = infer_structures(solution)
        if not structures:
            LOGGER.warning("Skipping example with no inferable target fields")
            continue
        prepared.append(
            {"text": text, "structures": structures, "solution": solution}
        )
    return prepared


def values_match(source: Any, target: Any) -> bool:
    """Compare scalar leaves with the same flexible coercion as the reference."""

    if source == target:
        return True
    if target is None:
        return source is None or (
            isinstance(source, str)
            and source.strip().casefold() in {"", "null", "none"}
        )
    if source is None:
        return False
    if isinstance(target, bool):
        if isinstance(source, str):
            candidates = {"true", "1", "yes"} if target else {"false", "0", "no"}
            return source.strip().casefold() in candidates
        if isinstance(source, int | float):
            return bool(source) == target
        return False
    if isinstance(target, int | float) and not isinstance(target, bool):
        if isinstance(source, int | float) and not isinstance(source, bool):
            return abs(source - target) < 1e-9
        if isinstance(source, str):
            try:
                parsed = float(source) if any(c in source.casefold() for c in ".e") else int(source)
            except (TypeError, ValueError):
                return False
            return abs(parsed - target) < 1e-9
        return False
    if isinstance(target, str):
        if isinstance(source, str):
            return source.strip().casefold() == target.strip().casefold()
        return str(source) == target
    return False


def flatten_json(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten JSON to dot paths, including numeric list indices."""

    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        if not value:
            flattened[prefix] = {}
        for key, child in value.items():
            child_key = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_json(child, child_key))
    elif isinstance(value, list):
        if not value:
            flattened[prefix] = []
        for index, child in enumerate(value):
            child_key = f"{prefix}.{index}" if prefix else str(index)
            flattened.update(flatten_json(child, child_key))
    else:
        flattened[prefix] = value
    return flattened


def compute_json_f1(source: Any, target: Any) -> tuple[float, float, float]:
    """Compute value-aware precision/recall/F1 over flattened JSON leaves."""

    source_flat = flatten_json(canonicalize_json(source))
    target_flat = flatten_json(canonicalize_json(target))
    if not source_flat and not target_flat:
        return 1.0, 1.0, 1.0
    if not source_flat or not target_flat:
        return 0.0, 0.0, 0.0
    correct = sum(
        1
        for key, source_value in source_flat.items()
        if key in target_flat and values_match(source_value, target_flat[key])
    )
    precision = correct / len(source_flat)
    recall = correct / len(target_flat)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _structure_paths(value: Any, prefix: str = "$") -> set[str]:
    """Collect key/index paths, allowing structure F1 to inspect GLiNExT lists."""

    paths: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}"
            paths.add(path)
            paths.update(_structure_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{prefix}[{index}]"
            paths.add(path)
            paths.update(_structure_paths(child, path))
    return paths


def json_structure_f1(source: Any, target: Any) -> float:
    """Compute F1 over all dict-key and list-index paths."""

    source_paths = _structure_paths(canonicalize_json(source))
    target_paths = _structure_paths(canonicalize_json(target))
    if not source_paths and not target_paths:
        return 1.0
    if not source_paths or not target_paths:
        return 0.0
    overlap = len(source_paths & target_paths)
    precision = overlap / len(source_paths)
    recall = overlap / len(target_paths)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def count_elements(value: Any) -> int:
    if isinstance(value, dict):
        return sum((count_elements(child) for child in value.values()), start=0) or 1
    if isinstance(value, list):
        return sum((count_elements(child) for child in value), start=0) or 1
    return 1


def compute_json_accuracy(source: Any, target: Any) -> tuple[int, int]:
    """Return correct and compared leaf counts, penalizing missing/extra data."""

    source = canonicalize_json(source)
    target = canonicalize_json(target)
    if isinstance(target, dict):
        if not isinstance(source, dict):
            return 0, count_elements(target)
        if not target and not source:
            return 1, 1
        correct = total = 0
        source_keys = set(source)
        target_keys = set(target)
        for key in target_keys - source_keys:
            total += count_elements(target[key])
        total += len(source_keys - target_keys)
        for key in source_keys & target_keys:
            child_correct, child_total = compute_json_accuracy(source[key], target[key])
            correct += child_correct
            total += child_total
        return correct, total
    if isinstance(target, list):
        if not isinstance(source, list):
            return 0, count_elements(target)
        if not target and not source:
            return 1, 1
        correct = total = 0
        for index in range(max(len(source), len(target))):
            if index < len(source) and index < len(target):
                child_correct, child_total = compute_json_accuracy(source[index], target[index])
                correct += child_correct
                total += child_total
            elif index < len(target):
                total += count_elements(target[index])
            else:
                total += 1
        return correct, total
    return (1, 1) if values_match(source, target) else (0, 1)


def json_accuracy(source: Any, target: Any) -> float:
    correct, total = compute_json_accuracy(source, target)
    return correct / total if total else 1.0


@dataclass
class EvalMetrics:
    json_consistency: float = 0.0
    json_structure_f1: float = 0.0
    json_accuracy: float = 0.0
    json_f1: float = 0.0
    json_f1_precision: float = 0.0
    json_f1_recall: float = 0.0
    num_samples: int = 0
    num_valid_json: int = 0
    num_failed_inference: int = 0

    def add(self, metrics: dict[str, Any]) -> None:
        for name in (
            "json_consistency",
            "json_structure_f1",
            "json_accuracy",
            "json_f1",
            "json_f1_precision",
            "json_f1_recall",
        ):
            setattr(self, name, getattr(self, name) + float(metrics[name]))
        self.num_samples += 1
        self.num_valid_json += int(bool(metrics["valid_json"]))
        self.num_failed_inference += int(bool(metrics.get("inference_failed")))

    def average(self) -> None:
        if not self.num_samples:
            return
        for name in (
            "json_consistency",
            "json_structure_f1",
            "json_accuracy",
            "json_f1",
            "json_f1_precision",
            "json_f1_recall",
        ):
            setattr(self, name, getattr(self, name) / self.num_samples)


def evaluate_single(prediction: Any, solution: Any) -> dict[str, Any]:
    """Evaluate one native GLiNExT structuring prediction."""

    valid_json = isinstance(prediction, dict | list)
    prediction = canonicalize_json(prediction) if valid_json else {}
    solution = canonicalize_json(solution)
    precision, recall, f1 = compute_json_f1(prediction, solution)
    return {
        # Native GLiNExT returns one already-decoded JSON container, rather
        # than text that may contain zero or several JSON values.
        "json_consistency": float(valid_json),
        "json_structure_f1": json_structure_f1(prediction, solution),
        "json_accuracy": json_accuracy(prediction, solution),
        "json_f1": f1,
        "json_f1_precision": precision,
        "json_f1_recall": recall,
        "valid_json": valid_json,
        "inference_failed": False,
    }


def _progress(items: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    try:
        from tqdm import tqdm
    except ImportError:
        return iter(items)
    return iter(tqdm(items, desc="Evaluating"))


def evaluate(
    model: Any,
    test_data: list[dict[str, Any]],
    *,
    threshold: float = 0.5,
    objectness_threshold: float | None = None,
    batch_size: int = 8,
    structuring_dedup: bool = True,
    save_predictions: Path | None = None,
    fail_fast: bool = False,
) -> EvalMetrics:
    """Run native GLiNExT inference and aggregate evaluation metrics."""

    totals = EvalMetrics()
    results: list[dict[str, Any]] = []
    for item in _progress(test_data):
        error: str | None = None
        try:
            prediction = model.structure(
                item["text"],
                item["structures"],
                threshold=threshold,
                objectness_threshold=objectness_threshold,
                batch_size=batch_size,
                structuring_dedup=structuring_dedup,
            )
            metrics = evaluate_single(prediction, item["solution"])
        except Exception as exc:
            if fail_fast:
                raise
            error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Inference failed for one example")
            prediction = None
            metrics = evaluate_single(prediction, item["solution"])
            metrics["inference_failed"] = True
        totals.add(metrics)
        results.append(
            {
                "text": item["text"],
                "structures": item["structures"],
                "solution": item["solution"],
                "prediction": prediction,
                "metrics": metrics,
                **({"error": error} if error else {}),
            }
        )

    totals.average()
    if save_predictions is not None:
        save_predictions.parent.mkdir(parents=True, exist_ok=True)
        with save_predictions.open("w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=2, ensure_ascii=False)
        LOGGER.info("Predictions saved to %s", save_predictions)
    return totals


def load_model(
    model_path: str,
    *,
    device: str,
    dtype: str | None,
    local_files_only: bool,
) -> Any:
    """Load a GLiNExT checkpoint and place it on the requested device."""

    import torch

    from glinext import GLiNExT

    resolved_dtype = getattr(torch, dtype) if dtype else None
    model = GLiNExT.from_pretrained(
        model_path,
        map_location="cpu",
        dtype=resolved_dtype,
        local_files_only=local_files_only,
        load_tokenizer=True,
    )
    model = model.to(device)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-path", "--model_path", required=True)
    parser.add_argument(
        "--test-data-path",
        "--test_data_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--max-samples",
        "--max_samples",
        type=int,
        default=30,
        help="Number of examples to evaluate; use 0 for the entire file (default: 30).",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--objectness-threshold", type=float, default=None)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=8)
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda when available, otherwise cpu).",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default=None,
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-structuring-dedup", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--save-predictions", "--save_predictions", type=Path)
    parser.add_argument("--output-file", "--output_file", type=Path)
    return parser.parse_args()


def _print_metrics(metrics: EvalMetrics) -> None:
    samples = metrics.num_samples
    valid_percent = 100 * metrics.num_valid_json / samples if samples else 0.0
    print("\n" + "=" * 50)
    print("GLiNExT STRUCTURING EVALUATION")
    print("=" * 50)
    print(f"Number of samples:        {samples}")
    print(f"Valid JSON predictions:   {metrics.num_valid_json} ({valid_percent:.1f}%)")
    print(f"Inference failures:       {metrics.num_failed_inference}")
    print("-" * 50)
    print(f"JSON Consistency:         {metrics.json_consistency:.4f}")
    print(f"JSON Structure F1:        {metrics.json_structure_f1:.4f}")
    print(f"JSON Accuracy:            {metrics.json_accuracy:.4f}")
    print(f"JSON F1 Score:            {metrics.json_f1:.4f}")
    print(f"  - Precision:            {metrics.json_f1_precision:.4f}")
    print(f"  - Recall:               {metrics.json_f1_recall:.4f}")
    print("=" * 50)


def main() -> int:
    args = parse_args()
    if args.max_samples < 0:
        raise SystemExit("--max-samples must be non-negative")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if not 0 <= args.threshold <= 1:
        raise SystemExit("--threshold must be between 0 and 1")
    if args.objectness_threshold is not None and not 0 <= args.objectness_threshold <= 1:
        raise SystemExit("--objectness-threshold must be between 0 and 1")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    max_samples = args.max_samples or None
    test_data = prepare_test_data(args.test_data_path, max_samples=max_samples)
    LOGGER.info("Prepared %d test examples", len(test_data))
    LOGGER.info("Loading GLiNExT model from %s on %s", args.model_path, device)
    model = load_model(
        args.model_path,
        device=device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    metrics = evaluate(
        model,
        test_data,
        threshold=args.threshold,
        objectness_threshold=args.objectness_threshold,
        batch_size=args.batch_size,
        structuring_dedup=not args.no_structuring_dedup,
        save_predictions=args.save_predictions,
        fail_fast=args.fail_fast,
    )
    _print_metrics(metrics)

    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        output = asdict(metrics)
        output["config"] = {
            "model_path": args.model_path,
            "test_data_path": str(args.test_data_path),
            "max_samples": args.max_samples,
            "threshold": args.threshold,
            "objectness_threshold": args.objectness_threshold,
            "batch_size": args.batch_size,
            "device": device,
        }
        with args.output_file.open("w", encoding="utf-8") as output_file:
            json.dump(output, output_file, indent=2, ensure_ascii=False)
        LOGGER.info("Metrics saved to %s", args.output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
