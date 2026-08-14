#!/usr/bin/env python3
"""Evaluate a GLiNExT checkpoint on the ``data/NER`` benchmarks.

The benchmark files contain character-level entity spans, and GLiNExT returns
character-level spans from ``predict_entities``.  Scores are therefore strict
entity-level micro precision, recall, and F1: a prediction is correct only when
its example, start offset, end offset, and entity type all match the gold data.
"""

# Console output is the primary report produced by this CLI.
# ruff: noqa: T201

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_MODEL = PROJECT_ROOT / "logs" / "ner" / "checkpoint-5000"
DEFAULT_NER_DATA = WORKSPACE_ROOT / "data" / "NER"

NESTED_DATASET_MARKERS = ("ace", "genia", "corpus")
ZERO_SHOT_DATASETS = {
    "mit-movie",
    "mit-restaurant",
    "CrossNER_AI",
    "CrossNER_literature",
    "CrossNER_music",
    "CrossNER_politics",
    "CrossNER_science",
}


@dataclass(frozen=True)
class NERMetrics:
    """Strict entity-level micro metrics for one dataset."""

    samples: int
    gold_entities: int
    predicted_entities: int
    true_positives: int
    precision: float
    recall: float
    f1: float


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate GLiNExT on data/NER datasets and report per-dataset F1.")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Local checkpoint path or Hugging Face model ID.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_NER_DATA,
        help="Directory containing one subdirectory per NER dataset.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("test", "dev", "train"),
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional dataset names to evaluate; names containing spaces must be quoted.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Entity confidence threshold.")
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=12,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, for example 'cuda:0' or 'cpu' (default: auto).",
    )
    parser.add_argument(
        "--ner-mode",
        choices=("auto", "flat", "nested"),
        default="auto",
        help="NER decoding mode. Auto treats ACE, GENIA, and Corpus datasets as nested.",
    )
    parser.add_argument(
        "--include-samples",
        action="store_true",
        help="Also evaluate dataset directories whose names contain 'sample_'.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Evaluate at most this many examples from each dataset (useful for smoke tests).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for a JSON copy of the report.",
    )
    return parser


def load_dataset(dataset_dir: Path, split: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Load one benchmark split and its candidate entity types."""
    split_path = dataset_dir / f"{split}.json"
    labels_path = dataset_dir / "labels.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    with split_path.open(encoding="utf-8") as input_file:
        records = json.load(input_file)
    with labels_path.open(encoding="utf-8") as input_file:
        labels = json.load(input_file)

    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError(f"{split_path} must contain a JSON list of objects")
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise ValueError(f"{labels_path} must contain a JSON list of strings")

    return records, [label.casefold() for label in labels]


def discover_datasets(
    data_dir: Path,
    requested: Sequence[str] | None = None,
    include_samples: bool = False,
) -> list[Path]:
    """Return selected benchmark directories in deterministic name order."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"NER data directory does not exist: {data_dir}")

    requested_names = set(requested or ())
    paths = []
    for path in sorted(data_dir.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_dir():
            continue
        if requested_names and path.name not in requested_names:
            continue
        if not include_samples and "sample_" in path.name:
            continue
        if (path / "labels.json").is_file():
            paths.append(path)

    if requested_names:
        found = {path.name for path in paths}
        missing = sorted(requested_names - found)
        if missing:
            raise ValueError(f"Dataset(s) not found or excluded: {', '.join(missing)}")
    if not paths:
        raise ValueError(f"No NER datasets found in {data_dir}")
    return paths


def uses_flat_ner(dataset_name: str, mode: str = "auto") -> bool:
    if mode == "flat":
        return True
    if mode == "nested":
        return False
    normalized_name = dataset_name.casefold()
    return not any(marker in normalized_name for marker in NESTED_DATASET_MARKERS)


def _gold_entity_key(entity: dict[str, Any]) -> tuple[int, int, str]:
    try:
        start, end = entity["pos"]
        label = entity["type"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid gold entity: {entity!r}") from exc
    return int(start), int(end), str(label).casefold()


def _predicted_entity_key(entity: dict[str, Any]) -> tuple[int, int, str]:
    try:
        start = entity["start"]
        end = entity["end"]
        label = entity.get("label", entity.get("type", entity.get("entity_type")))
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid predicted entity: {entity!r}") from exc
    if label is None:
        raise ValueError(f"Predicted entity has no label: {entity!r}")
    return int(start), int(end), str(label).casefold()


def compute_metrics(
    gold_entities: Iterable[Iterable[dict[str, Any]]],
    predicted_entities: Iterable[Iterable[dict[str, Any]]],
) -> NERMetrics:
    """Compute strict micro metrics, keeping examples separate."""
    gold_batches = list(gold_entities)
    prediction_batches = list(predicted_entities)
    if len(gold_batches) != len(prediction_batches):
        raise ValueError(f"Gold/prediction sample count mismatch: {len(gold_batches)} != {len(prediction_batches)}")

    true_positives = 0
    gold_total = 0
    predicted_total = 0
    for gold, predicted in zip(gold_batches, prediction_batches, strict=True):
        gold_set = {_gold_entity_key(entity) for entity in gold}
        predicted_set = {_predicted_entity_key(entity) for entity in predicted}
        true_positives += len(gold_set & predicted_set)
        gold_total += len(gold_set)
        predicted_total += len(predicted_set)

    precision = true_positives / predicted_total if predicted_total else 0.0
    recall = true_positives / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return NERMetrics(
        samples=len(gold_batches),
        gold_entities=gold_total,
        predicted_entities=predicted_total,
        true_positives=true_positives,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def evaluate_dataset(
    model: Any,
    records: list[dict[str, Any]],
    labels: list[str],
    *,
    threshold: float,
    batch_size: int,
    flat_ner: bool,
) -> NERMetrics:
    """Run GLiNExT inference and score one loaded dataset."""
    all_gold: list[list[dict[str, Any]]] = []
    all_predictions: list[list[dict[str, Any]]] = []
    for batch in _batched(records, batch_size):
        try:
            texts = [str(record["sentence"]) for record in batch]
            gold = [record["entities"] for record in batch]
        except KeyError as exc:
            raise ValueError(f"Dataset row is missing required key {exc.args[0]!r}") from exc

        predictions = model.predict_entities(
            texts=texts,
            entities=labels,
            flat_ner=flat_ner,
            threshold=threshold,
            batch_size=batch_size,
        )
        if len(predictions) != len(batch):
            raise RuntimeError(
                f"GLiNExT returned an unexpected number of prediction lists: {len(predictions)} for {len(batch)} inputs"
            )
        all_gold.extend(gold)
        all_predictions.extend(predictions)

    return compute_metrics(all_gold, all_predictions)


def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    import torch  # noqa: PLC0415 - keep --help and metric-only imports lightweight

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_model(model_id: str, device: str) -> Any:
    from glinext import GLiNExT  # noqa: PLC0415 - importing torch is intentionally deferred

    model = GLiNExT.from_pretrained(
        model_id,
        load_tokenizer=True,
        map_location=device,
    )
    model.to(device)
    model.eval()
    return model


def _mean_f1(results: dict[str, NERMetrics], names: set[str] | None = None) -> float | None:
    scores = [metrics.f1 for name, metrics in results.items() if names is None or name in names]
    return sum(scores) / len(scores) if scores else None


def print_report(results: dict[str, NERMetrics]) -> None:
    name_width = max(20, *(len(name) for name in results))
    print()
    print(f"{'Dataset':<{name_width}}  {'Samples':>8}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")
    print("-" * (name_width + 46))
    for name, metrics in results.items():
        print(
            f"{name:<{name_width}}  {metrics.samples:>8,d}  "
            f"{metrics.precision:>9.2%}  {metrics.recall:>9.2%}  {metrics.f1:>9.2%}"
        )

    macro_f1 = _mean_f1(results)
    standard_f1 = _mean_f1(results, set(results) - ZERO_SHOT_DATASETS)
    zero_shot_f1 = _mean_f1(results, ZERO_SHOT_DATASETS)
    print("-" * (name_width + 46))
    if macro_f1 is not None:
        print(f"{'Macro average':<{name_width}}  {'':>8}  {'':>10}  {'':>10}  {macro_f1:>9.2%}")
    if standard_f1 is not None and zero_shot_f1 is not None:
        print(f"{'Standard average':<{name_width}}  {'':>8}  {'':>10}  {'':>10}  {standard_f1:>9.2%}")
        print(f"{'Zero-shot average':<{name_width}}  {'':>8}  {'':>10}  {'':>10}  {zero_shot_f1:>9.2%}")


def save_report(
    output_path: Path,
    results: dict[str, NERMetrics],
    *,
    model_id: str,
    split: str,
    threshold: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_id,
        "split": split,
        "threshold": threshold,
        "macro_f1": _mean_f1(results),
        "standard_macro_f1": _mean_f1(results, set(results) - ZERO_SHOT_DATASETS),
        "zero_shot_macro_f1": _mean_f1(results, ZERO_SHOT_DATASETS),
        "datasets": {name: asdict(metrics) for name, metrics in results.items()},
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")


def main() -> None:
    args = create_parser().parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be at least 1")

    dataset_paths = discover_datasets(
        args.data,
        requested=args.datasets,
        include_samples=args.include_samples,
    )
    device = resolve_device(args.device)
    print(f"Loading {args.model} on {device} ...")
    model = load_model(args.model, device)

    results: dict[str, NERMetrics] = {}
    for index, dataset_path in enumerate(dataset_paths, start=1):
        records, labels = load_dataset(dataset_path, args.split)
        if args.max_samples is not None:
            records = records[: args.max_samples]
        flat_ner = uses_flat_ner(dataset_path.name, args.ner_mode)
        mode = "flat" if flat_ner else "nested"
        print(
            f"[{index}/{len(dataset_paths)}] {dataset_path.name}: "
            f"{len(records):,} samples, {len(labels)} labels, {mode} NER"
        )
        metrics = evaluate_dataset(
            model,
            records,
            labels,
            threshold=args.threshold,
            batch_size=args.batch_size,
            flat_ner=flat_ner,
        )
        results[dataset_path.name] = metrics
        print(f"  F1={metrics.f1:.2%}  P={metrics.precision:.2%}  R={metrics.recall:.2%}")

    print_report(results)
    if args.output is not None:
        save_report(
            args.output,
            results,
            model_id=args.model,
            split=args.split,
            threshold=args.threshold,
        )
        print(f"\nSaved JSON report to {args.output}")


if __name__ == "__main__":
    main()
