#!/usr/bin/env python3
"""Evaluate a GLiFormer checkpoint on a structuring JSONL dataset.

Unlike a causal language model, GLiFormer does not consume a rendered JSON
template or generate JSON text.  This evaluator infers the minimal per-example
``structures`` argument from fields that are actually present in the target,
calls the native ``model.structure`` API, and computes JSON structure, value,
accuracy, precision, recall, and F1 metrics comparable to the generative
evaluation script.

Metrics are reported under three progressively more forgiving policies so that
two artefacts of set-prediction structuring can be read off directly instead of
being charged to the model as content errors:

``positional``
    Records are compared in emission order, exactly as the generative script
    does.  A structuring head predicts an unordered set of record anchors, so
    any permutation of otherwise perfect records is penalised here.
``order-insensitive``
    Predicted records are first assigned to gold records by optimal leaf
    overlap.  The gap against ``positional`` is the cost of record ordering
    alone.
``boundary-tolerant``
    As above, but a value also matches when the two spans differ only by a
    unit or descriptor affix (``"13.8"`` vs ``"13.8 g/dl"``, ``"c-19"`` vs
    ``"bottle c-19"``).  The gap against ``order-insensitive`` is the cost of
    span boundary conventions alone.

The ``positional``/exact numbers stay the headline figures, so results remain
comparable with previously recorded runs.
"""

from __future__ import annotations

import argparse
import json
import random
import logging
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

RECORD_MATCHING_POLICY = (
    "records compared in emission order (headline), and additionally under an "
    "optimal assignment of predicted records to gold records by leaf overlap; "
    "predictions left unassigned keep distinct positions and still cost precision"
)
VALUE_MATCHING_POLICY = (
    "exact after scalar coercion (headline), and additionally boundary-tolerant, "
    "where two spans also match when they differ only by a leading or trailing "
    "unit/descriptor affix carrying no independent value"
)


def load_jsonl(path: Path, max_samples: int | None = None, *, seed: int = 42) -> list[dict[str, Any]]:
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
    if max_samples is not None and max_samples < len(records):
        records = random.Random(seed).sample(records, max_samples)
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


def prepare_test_data(path: Path, max_samples: int | None = 30, *, seed: int = 42) -> list[dict[str, Any]]:
    """Load minimal GLiFormer rows and infer their native inference schemas."""

    prepared: list[dict[str, Any]] = []
    for record in load_jsonl(path):
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
    if max_samples is not None and max_samples < len(prepared):
        prepared = random.Random(seed).sample(prepared, max_samples)
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


# Affix characters that can legitimately separate a value from a unit or a
# descriptor. Sentence punctuation is excluded: a residue containing it means
# the span ran into neighbouring text rather than over a unit.
_RESIDUE_REJECTED_CHARS = frozenset(";:,()[]{}<>|\"\u201c\u201d")
_MAX_RESIDUE_CHARS = 16
_MAX_RESIDUE_WORDS = 2


def boundary_residue(source: Any, target: Any) -> str | None:
    """Return the affix by which two otherwise identical spans differ.

    ``None`` means the two values are not a boundary variant of one another.
    The residue must look like a unit or a descriptor rather than dropped
    content: at most two words, free of sentence punctuation, and carrying no
    digits of its own unless it opens with a unit symbol (``\u00d710^9/l``,
    ``/month``). ``"12"`` vs ``"12 34"`` is therefore not a boundary variant,
    while ``"12"`` vs ``"12 km"`` is.
    """

    if source is None or target is None:
        return None
    if isinstance(source, dict | list) or isinstance(target, dict | list):
        return None
    left = " ".join(str(source).split()).casefold()
    right = " ".join(str(target).split()).casefold()
    if not left or not right or left == right:
        return None

    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    if longer.startswith(shorter):
        residue = longer[len(shorter):]
    elif longer.endswith(shorter):
        residue = longer[:-len(shorter)]
    else:
        return None

    residue = residue.strip()
    if not residue or len(residue) > _MAX_RESIDUE_CHARS:
        return None
    if len(residue.split()) > _MAX_RESIDUE_WORDS:
        return None
    if _RESIDUE_REJECTED_CHARS & set(residue):
        return None
    if any(character.isdigit() for character in residue) and residue[0].isalnum():
        return None
    return residue


class BoundaryTolerantMatcher:
    """``values_match``, extended to accept unit/descriptor-only affixes.

    Instances are callable so they can be passed wherever ``values_match`` is,
    and they record which residues they forgave, which is what turns the
    boundary-tolerant column into an actionable diagnostic.
    """

    def __init__(self) -> None:
        self.residues: Counter[str] = Counter()

    def __call__(self, source: Any, target: Any) -> bool:
        if values_match(source, target):
            return True
        residue = boundary_residue(source, target)
        if residue is None:
            return False
        self.residues[residue] += 1
        return True


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


def _leaf_map(value: Any) -> dict[str, list[Any]]:
    """Collect leaves by path, collapsing list indices to a single bucket.

    Indices are dropped so that two records still look similar when a nested
    array inside them is ordered differently - the ordering of those inner
    arrays is settled by the same alignment, one level down.
    """

    leaves: dict[str, list[Any]] = {}

    def visit(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            for child in node:
                visit(child, f"{path}[]")
        else:
            leaves.setdefault(path, []).append(node)

    visit(value, "")
    return leaves


def _record_similarity(source: Any, target: Any) -> int:
    """Count leaves shared by two records, under boundary-tolerant matching.

    Alignment deliberately uses the most forgiving value comparison available:
    a record whose values all trail a unit is still that record, and pairing it
    correctly is what lets the stricter columns attribute the loss to values
    rather than to ordering.
    """

    source_leaves = _leaf_map(source)
    matched = 0
    for path, target_values in _leaf_map(target).items():
        available = list(source_leaves.get(path, ()))
        for target_value in target_values:
            for index, source_value in enumerate(available):
                if values_match(source_value, target_value) or (
                    boundary_residue(source_value, target_value) is not None
                ):
                    matched += 1
                    available.pop(index)
                    break
    return matched


def _assign_records(
    predictions: list[Any],
    targets: list[Any],
) -> dict[int, int]:
    """Map gold record index -> predicted record index by maximum overlap.

    Pairs sharing no leaf at all are left unassigned: an entirely unrelated
    prediction should not consume a gold slot and mask a missing record.
    """

    if not predictions or not targets:
        return {}
    scores = [
        [_record_similarity(prediction, target) for target in targets]
        for prediction in predictions
    ]
    try:
        import numpy
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        pairs = sorted(
            (
                (scores[source][target], source, target)
                for source in range(len(predictions))
                for target in range(len(targets))
            ),
            reverse=True,
        )
        assignment: dict[int, int] = {}
        used_predictions: set[int] = set()
        for score, source, target in pairs:
            if score <= 0 or source in used_predictions or target in assignment:
                continue
            used_predictions.add(source)
            assignment[target] = source
        return assignment

    rows, columns = linear_sum_assignment(-numpy.asarray(scores, dtype=float))
    return {
        int(column): int(row)
        for row, column in zip(rows, columns, strict=True)
        if scores[row][column] > 0
    }


def align_records(prediction: Any, solution: Any) -> tuple[Any, Any]:
    """Rewrite record lists so matching records share a path on both sides.

    Every aligned list becomes a dict keyed by the *gold* index, which
    ``flatten_json`` renders identically to a list index, so the existing
    positional metrics can be reused unchanged. Predictions that matched
    nothing are keyed past the end of the gold list, keeping them countable
    against precision instead of silently disappearing.
    """

    if isinstance(prediction, dict) and isinstance(solution, dict):
        aligned_prediction: dict[str, Any] = {}
        aligned_solution: dict[str, Any] = dict(solution)
        for key in solution:
            if key in prediction:
                child_prediction, child_solution = align_records(
                    prediction[key], solution[key]
                )
                aligned_prediction[key] = child_prediction
                aligned_solution[key] = child_solution
        for key in prediction:
            if key not in solution:
                aligned_prediction[key] = prediction[key]
        return aligned_prediction, aligned_solution

    if isinstance(prediction, list) and isinstance(solution, list):
        assignment = _assign_records(prediction, solution)
        target_of_prediction = {
            source: target for target, source in assignment.items()
        }
        aligned_prediction = {}
        aligned_solution = {str(index): item for index, item in enumerate(solution)}
        spare_index = len(solution)
        for index, item in enumerate(prediction):
            target = target_of_prediction.get(index)
            if target is None:
                aligned_prediction[str(spare_index)] = item
                spare_index += 1
                continue
            child_prediction, child_solution = align_records(item, solution[target])
            aligned_prediction[str(target)] = child_prediction
            aligned_solution[str(target)] = child_solution
        return aligned_prediction, aligned_solution

    return prediction, solution


def compute_json_f1(
    source: Any,
    target: Any,
    match: Callable[[Any, Any], bool] = values_match,
) -> tuple[float, float, float]:
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
        if key in target_flat and match(source_value, target_flat[key])
    )
    precision = correct / len(source_flat)
    recall = correct / len(target_flat)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _structure_paths(value: Any, prefix: str = "$") -> set[str]:
    """Collect key/index paths, allowing structure F1 to inspect GLiFormer lists."""

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


def compute_json_accuracy(
    source: Any,
    target: Any,
    match: Callable[[Any, Any], bool] = values_match,
) -> tuple[int, int]:
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
            child_correct, child_total = compute_json_accuracy(
                source[key], target[key], match
            )
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
                child_correct, child_total = compute_json_accuracy(
                    source[index], target[index], match
                )
                correct += child_correct
                total += child_total
            elif index < len(target):
                total += count_elements(target[index])
            else:
                total += 1
        return correct, total
    return (1, 1) if match(source, target) else (0, 1)


def json_accuracy(
    source: Any,
    target: Any,
    match: Callable[[Any, Any], bool] = values_match,
) -> float:
    correct, total = compute_json_accuracy(source, target, match)
    return correct / total if total else 1.0


def count_gold_records(solution: Any) -> int:
    """Number of gold record instances, i.e. dicts emitted inside a list.

    Nested records count too, since the head anchors those separately as well.
    A bare flat record with no enclosing list is one record.
    """

    total = 0

    def visit(node: Any, inside_list: bool) -> None:
        nonlocal total
        if isinstance(node, dict):
            if inside_list:
                total += 1
            for child in node.values():
                visit(child, False)
        elif isinstance(node, list):
            for child in node:
                visit(child, True)

    visit(canonicalize_json(solution), False)
    return total or 1


def text_length_words(text: Any) -> int:
    return len(str(text or "").split())


def json_depth(solution: Any, level: int = 0) -> int:
    """Nesting depth of the gold target, counting dict levels only.

    A flat ``{schema: [record, ...]}`` target is depth 2, and each further level
    of child records adds one; list nesting on its own does not, since a list is
    how one level holds its instances rather than a level of its own.
    """

    if isinstance(solution, dict):
        return max(
            (json_depth(child, level + 1) for child in solution.values()),
            default=level,
        )
    if isinstance(solution, list):
        return max((json_depth(child, level) for child in solution), default=level)
    return level


DEFAULT_RECORD_BUCKETS = (1, 2, 3, 5, 10)
DEFAULT_LENGTH_BUCKETS = (64, 128, 256, 512)
DEFAULT_DEPTH_BUCKETS = (2, 3, 4, 5)


@dataclass(frozen=True)
class BreakdownSpec:
    """One bucketed view of a run: ordered ranges over a per-example key."""

    name: str
    header: str
    key: Callable[[dict[str, Any]], int]
    thresholds: tuple[int, ...]
    start: int = 0

    @property
    def labels(self) -> list[str]:
        """Human-readable range per threshold, plus an open-ended last bucket."""

        labels: list[str] = []
        lower = self.start
        for upper in self.thresholds:
            labels.append(str(upper) if lower >= upper else f"{lower}-{upper}")
            lower = upper + 1
        labels.append(f"{lower}+")
        return labels

    def bucket_of(self, item: dict[str, Any]) -> str:
        value = self.key(item)
        labels = self.labels
        for threshold, label in zip(self.thresholds, labels, strict=False):
            if value <= threshold:
                return label
        return labels[-1]


def default_breakdowns(
    record_buckets: Sequence[int] = DEFAULT_RECORD_BUCKETS,
    length_buckets: Sequence[int] = DEFAULT_LENGTH_BUCKETS,
    depth_buckets: Sequence[int] = DEFAULT_DEPTH_BUCKETS,
) -> list[BreakdownSpec]:
    """The breakdowns reported by default: gold size, input length, nesting."""

    return [
        BreakdownSpec(
            name="gold_record_count",
            header="gold record count",
            key=lambda item: count_gold_records(item["solution"]),
            thresholds=tuple(record_buckets),
            start=1,
        ),
        BreakdownSpec(
            name="text_length_words",
            header="text length (words)",
            key=lambda item: text_length_words(item.get("text")),
            thresholds=tuple(length_buckets),
            start=0,
        ),
        BreakdownSpec(
            name="gold_json_depth",
            header="gold JSON depth",
            key=lambda item: json_depth(item["solution"]),
            thresholds=tuple(depth_buckets),
            start=2,
        ),
    ]


_VARIANT_FIELDS = (
    "json_structure_f1",
    "json_accuracy",
    "json_f1",
    "json_f1_precision",
    "json_f1_recall",
)


@dataclass
class VariantMetrics:
    """One column of the report: the same metrics under a looser policy."""

    json_structure_f1: float = 0.0
    json_accuracy: float = 0.0
    json_f1: float = 0.0
    json_f1_precision: float = 0.0
    json_f1_recall: float = 0.0

    def add(self, metrics: dict[str, Any]) -> None:
        for name in _VARIANT_FIELDS:
            setattr(self, name, getattr(self, name) + float(metrics[name]))

    def average(self, num_samples: int) -> None:
        if not num_samples:
            return
        for name in _VARIANT_FIELDS:
            setattr(self, name, getattr(self, name) / num_samples)


@dataclass
class EvalMetrics:
    # Headline fields stay the strict positional numbers, so recorded runs and
    # downstream readers keep their meaning.
    json_consistency: float = 0.0
    json_structure_f1: float = 0.0
    json_accuracy: float = 0.0
    json_f1: float = 0.0
    json_f1_precision: float = 0.0
    json_f1_recall: float = 0.0
    num_samples: int = 0
    num_valid_json: int = 0
    num_failed_inference: int = 0
    order_insensitive: VariantMetrics = field(default_factory=VariantMetrics)
    boundary_tolerant: VariantMetrics = field(default_factory=VariantMetrics)
    boundary_repairs: int = 0
    boundary_residues: dict[str, int] = field(default_factory=dict)
    # breakdown name -> bucket label -> the same metrics over that bucket only.
    breakdowns: dict[str, dict[str, "EvalMetrics"]] = field(default_factory=dict)

    def add(self, metrics: dict[str, Any]) -> None:
        for name in ("json_consistency", *_VARIANT_FIELDS):
            setattr(self, name, getattr(self, name) + float(metrics[name]))
        variants = metrics.get("variants") or {}
        # A caller still emitting the pre-variant dict shape (any external
        # evaluator sharing evaluate_single) collapses to the strict column
        # rather than failing.
        self.order_insensitive.add(variants.get("order_insensitive", metrics))
        self.boundary_tolerant.add(variants.get("boundary_tolerant", metrics))
        self.boundary_repairs += int(metrics.get("boundary_repairs", 0))
        for residue, count in (metrics.get("boundary_residues") or {}).items():
            self.boundary_residues[residue] = (
                self.boundary_residues.get(residue, 0) + int(count)
            )
        self.num_samples += 1
        self.num_valid_json += int(bool(metrics["valid_json"]))
        self.num_failed_inference += int(bool(metrics.get("inference_failed")))

    def average(self) -> None:
        if not self.num_samples:
            return
        for name in ("json_consistency", *_VARIANT_FIELDS):
            setattr(self, name, getattr(self, name) / self.num_samples)
        self.order_insensitive.average(self.num_samples)
        self.boundary_tolerant.average(self.num_samples)
        for buckets in self.breakdowns.values():
            for bucket in buckets.values():
                bucket.average()


def _score(
    prediction: Any,
    solution: Any,
    match: Callable[[Any, Any], bool],
) -> dict[str, float]:
    """Score one prediction/solution pair under a single matching policy."""

    precision, recall, f1 = compute_json_f1(prediction, solution, match)
    return {
        "json_structure_f1": json_structure_f1(prediction, solution),
        "json_accuracy": json_accuracy(prediction, solution, match),
        "json_f1": f1,
        "json_f1_precision": precision,
        "json_f1_recall": recall,
    }


def evaluate_single(prediction: Any, solution: Any) -> dict[str, Any]:
    """Evaluate one native GLiFormer structuring prediction.

    The strict positional scores stay at the top level. ``variants`` adds the
    same scores once records have been aligned, and again with boundary-only
    value differences forgiven, so the report can attribute loss to record
    ordering and to span boundaries separately from real content errors.
    """

    valid_json = isinstance(prediction, dict | list)
    prediction = canonicalize_json(prediction) if valid_json else {}
    solution = canonicalize_json(solution)

    aligned_prediction, aligned_solution = align_records(prediction, solution)
    order_insensitive = _score(aligned_prediction, aligned_solution, values_match)

    # The matcher is shared by both scoring passes below, so the residue tally
    # is snapshotted after the leaf-level pass alone; letting the accuracy pass
    # add to it would count every forgiven leaf twice.
    matcher = BoundaryTolerantMatcher()
    precision, recall, f1 = compute_json_f1(
        aligned_prediction, aligned_solution, matcher
    )
    residues = dict(matcher.residues)
    boundary_tolerant = {
        "json_structure_f1": json_structure_f1(aligned_prediction, aligned_solution),
        "json_accuracy": json_accuracy(aligned_prediction, aligned_solution, matcher),
        "json_f1": f1,
        "json_f1_precision": precision,
        "json_f1_recall": recall,
    }

    return {
        # Native GLiFormer returns one already-decoded JSON container, rather
        # than text that may contain zero or several JSON values.
        "json_consistency": float(valid_json),
        **_score(prediction, solution, values_match),
        "variants": {
            "order_insensitive": order_insensitive,
            "boundary_tolerant": boundary_tolerant,
        },
        "boundary_repairs": sum(residues.values()),
        "boundary_residues": residues,
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
    breakdowns: Sequence[BreakdownSpec] | None = None,
    skip_inference_error: Callable[[Exception], bool] | None = None,
) -> EvalMetrics:
    """Run native GLiFormer inference and aggregate evaluation metrics.

    ``breakdowns`` adds bucketed copies of the same aggregation; pass an empty
    sequence to skip them and ``None`` for :func:`default_breakdowns`.
    ``skip_inference_error`` optionally excludes selected failures from metrics
    while recording them in saved predictions. ``fail_fast`` takes precedence.
    """

    specs = default_breakdowns() if breakdowns is None else list(breakdowns)
    totals = EvalMetrics()
    # Seeded up front so buckets print in threshold order, not first-seen order.
    for spec in specs:
        totals.breakdowns[spec.name] = {label: EvalMetrics() for label in spec.labels}
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
            if skip_inference_error is not None and skip_inference_error(exc):
                LOGGER.warning("Skipping example after API error: %s", exc)
                results.append({
                    "text": item["text"], "structures": item["structures"],
                    "solution": item["solution"], "prediction": None, "metrics": None,
                    "skipped": True, "skip_reason": "transient_api_error",
                    "error": str(exc), "status_code": getattr(exc, "status_code", None),
                })
                continue
            error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Inference failed for one example")
            prediction = None
            metrics = evaluate_single(prediction, item["solution"])
            metrics["inference_failed"] = True
        totals.add(metrics)
        for spec in specs:
            totals.breakdowns[spec.name][spec.bucket_of(item)].add(metrics)
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
    """Load a GLiFormer checkpoint and place it on the requested device."""

    import torch

    from gliformer import GLiFormer

    resolved_dtype = getattr(torch, dtype) if dtype else None
    model = GLiFormer.from_pretrained(
        model_path,
        map_location="cpu",
        dtype=resolved_dtype,
        local_files_only=local_files_only,
        load_tokenizer=True,
    )
    model = model.to(device)
    model.eval()
    return model


def _int_thresholds(value: str) -> tuple[int, ...]:
    """Parse and validate a comma-separated list of ascending bucket bounds."""

    try:
        thresholds = tuple(
            int(part) for part in value.split(",") if part.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {value!r}"
        ) from exc
    if not thresholds:
        raise argparse.ArgumentTypeError("expected at least one bound")
    if any(bound < 0 for bound in thresholds):
        raise argparse.ArgumentTypeError("bounds must be non-negative")
    if any(a >= b for a, b in zip(thresholds, thresholds[1:], strict=False)):
        raise argparse.ArgumentTypeError("bounds must be strictly ascending")
    return thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42, help="Random sampling seed (default: 42).")
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
    parser.add_argument(
        "--record-buckets",
        "--record_buckets",
        type=_int_thresholds,
        default=DEFAULT_RECORD_BUCKETS,
        metavar="N,N,...",
        help=(
            "Ascending inclusive upper bounds for the gold-record-count "
            f"breakdown (default: {','.join(map(str, DEFAULT_RECORD_BUCKETS))})."
        ),
    )
    parser.add_argument(
        "--length-buckets",
        "--length_buckets",
        type=_int_thresholds,
        default=DEFAULT_LENGTH_BUCKETS,
        metavar="N,N,...",
        help=(
            "Ascending inclusive upper bounds, in words, for the text-length "
            f"breakdown (default: {','.join(map(str, DEFAULT_LENGTH_BUCKETS))})."
        ),
    )
    parser.add_argument(
        "--depth-buckets",
        "--depth_buckets",
        type=_int_thresholds,
        default=DEFAULT_DEPTH_BUCKETS,
        metavar="N,N,...",
        help=(
            "Ascending inclusive upper bounds for the gold-JSON-depth "
            "breakdown, where a flat target is depth 2 (default: "
            f"{','.join(map(str, DEFAULT_DEPTH_BUCKETS))})."
        ),
    )
    parser.add_argument(
        "--no-breakdown",
        action="store_true",
        help="Report only the corpus-level metrics, without bucketed tables.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-structuring-dedup", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--save-predictions", "--save_predictions", type=Path)
    parser.add_argument("--output-file", "--output_file", type=Path)
    return parser.parse_args()


def _print_breakdowns(
    metrics: EvalMetrics,
    specs: Sequence[BreakdownSpec],
    width: int,
) -> None:
    """Print per-bucket JSON F1 tables, skipping buckets with no examples."""

    for spec in specs:
        buckets = metrics.breakdowns.get(spec.name) or {}
        populated = [
            (label, bucket)
            for label, bucket in buckets.items()
            if bucket.num_samples
        ]
        if not populated:
            continue
        print("-" * width)
        print(f"JSON F1 by {spec.header}")
        print(
            f"{'bucket':<14s}{'n':>6s}{'P':>10s}{'R':>10s}{'F1':>10s}"
            f"{'order-free':>12s}{'+boundary':>12s}"
        )
        for label, bucket in populated:
            print(
                f"{label:<14s}{bucket.num_samples:>6d}"
                f"{bucket.json_f1_precision:>10.4f}"
                f"{bucket.json_f1_recall:>10.4f}"
                f"{bucket.json_f1:>10.4f}"
                f"{bucket.order_insensitive.json_f1:>12.4f}"
                f"{bucket.boundary_tolerant.json_f1:>12.4f}"
            )


def _print_metrics(
    metrics: EvalMetrics,
    breakdowns: Sequence[BreakdownSpec] = (),
    *,
    title: str = "GLiFormer STRUCTURING EVALUATION",
) -> None:
    samples = metrics.num_samples
    valid_percent = 100 * metrics.num_valid_json / samples if samples else 0.0
    width = 74
    print("\n" + "=" * width)
    print(title)
    print("=" * width)
    print(f"Number of samples:        {samples}")
    print(f"Valid JSON predictions:   {metrics.num_valid_json} ({valid_percent:.1f}%)")
    print(f"Inference failures:       {metrics.num_failed_inference}")
    print(f"JSON Consistency:         {metrics.json_consistency:.4f}")
    print("-" * width)
    print(f"{'':26s}{'positional':>14s}{'order-free':>14s}{'+boundary':>14s}")
    ordered = metrics.order_insensitive
    tolerant = metrics.boundary_tolerant
    rows = (
        ("JSON Structure F1", "json_structure_f1"),
        ("JSON Accuracy", "json_accuracy"),
        ("JSON F1 Score", "json_f1"),
        ("  - Precision", "json_f1_precision"),
        ("  - Recall", "json_f1_recall"),
    )
    for label, name in rows:
        print(
            f"{label:26s}"
            f"{getattr(metrics, name):>14.4f}"
            f"{getattr(ordered, name):>14.4f}"
            f"{getattr(tolerant, name):>14.4f}"
        )
    print("-" * width)
    ordering_cost = ordered.json_f1 - metrics.json_f1
    boundary_cost = tolerant.json_f1 - ordered.json_f1
    print("Attribution of the JSON F1 gap")
    print(f"  record ordering:        {ordering_cost:+.4f}")
    print(f"  span boundaries:        {boundary_cost:+.4f}")
    print(f"  residual (content):     {1.0 - tolerant.json_f1:.4f}")
    if metrics.boundary_residues:
        top = sorted(
            metrics.boundary_residues.items(),
            key=lambda item: (-item[1], item[0]),
        )[:8]
        summary = ", ".join(f"{residue!r}x{count}" for residue, count in top)
        print(
            f"  boundary repairs:       {metrics.boundary_repairs} "
            f"leaves; most common affixes: {summary}"
        )
    _print_breakdowns(metrics, breakdowns, width)
    print("=" * width)


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

    breakdowns: list[BreakdownSpec] = (
        []
        if args.no_breakdown
        else default_breakdowns(
            args.record_buckets, args.length_buckets, args.depth_buckets
        )
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    max_samples = args.max_samples or None
    test_data = prepare_test_data(args.test_data_path, max_samples=max_samples, seed=args.seed)
    LOGGER.info("Prepared %d test examples", len(test_data))
    LOGGER.info("Loading GLiFormer model from %s on %s", args.model_path, device)
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
        breakdowns=breakdowns,
    )
    _print_metrics(metrics, breakdowns)

    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        output = asdict(metrics)
        output["record_matching"] = RECORD_MATCHING_POLICY
        output["value_matching"] = VALUE_MATCHING_POLICY
        output["config"] = {
            "model_path": args.model_path,
            "test_data_path": str(args.test_data_path),
            "max_samples": args.max_samples, "seed": args.seed,
            "threshold": args.threshold,
            "objectness_threshold": args.objectness_threshold,
            "batch_size": args.batch_size,
            "device": device,
            "record_buckets": list(args.record_buckets),
            "length_buckets": list(args.length_buckets),
            "depth_buckets": list(args.depth_buckets),
        }
        with args.output_file.open("w", encoding="utf-8") as output_file:
            json.dump(output, output_file, indent=2, ensure_ascii=False)
        LOGGER.info("Metrics saved to %s", args.output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
