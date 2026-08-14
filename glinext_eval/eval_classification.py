#!/usr/bin/env python3
"""Evaluate GLiNExT on the zero-shot classification suite used by GLiClass.

This script mirrors ``GLiClass/test_gliclass.py`` while using GLiNExT's native
``classify`` API.  Every benchmark is evaluated as single-label classification
and receives accuracy, micro F1, macro F1, weighted F1, and prediction coverage.
"""

# Console output is the primary report produced by this CLI.
# ruff: noqa: T201

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    """Loading and column information for one classification benchmark."""

    name: str
    config: str | None = None
    text_column: str = "text"
    label_column: str = "label_text"
    classes: tuple[str, ...] | None = None


BENCHMARKS = (
    DatasetSpec("SetFit/CR"),
    DatasetSpec("SetFit/sst2"),
    DatasetSpec(
        "SetFit/sst5",
        classes=("very negative", "negative", "neutral", "positive", "very positive"),
    ),
    # DatasetSpec("stanfordnlp/imdb", label_column="label", classes=("negative", "positive")),
    # DatasetSpec("SetFit/20_newsgroups"),
    DatasetSpec("SetFit/enron_spam"),
    DatasetSpec("AmazonScience/massive", config="en-US", text_column="utt", label_column="intent"),
    DatasetSpec("PolyAI/banking77", label_column="label"),
    DatasetSpec(
        "takala/financial_phrasebank",
        config="sentences_allagree",
        text_column="sentence",
        label_column="label",
    ),
    DatasetSpec("ag_news", label_column="label"),
    DatasetSpec("dair-ai/emotion", label_column="label"),
    DatasetSpec("MoritzLaurer/cap_sotu", label_column="labels"),
    DatasetSpec("cornell-movie-review-data/rotten_tomatoes", label_column="label"),
)
BENCHMARK_BY_NAME = {benchmark.name: benchmark for benchmark in BENCHMARKS}


@dataclass(frozen=True)
class PreparedDataset:
    """Normalized text, candidate classes, and string gold labels."""

    name: str
    split: str
    texts: list[str]
    classes: list[str]
    true_labels: list[str]


@dataclass(frozen=True)
class ClassificationMetrics:
    """Single-label classification metrics for one dataset."""

    samples: int
    correct: int
    covered: int
    accuracy: float
    micro_f1: float
    macro_f1: float
    weighted_f1: float
    coverage: float


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate GLiNExT on the GLiClass zero-shot classification suite.")
    parser.add_argument("--model", required=True, help="Local GLiNExT checkpoint or Hugging Face model ID.")
    parser.add_argument(
        "--api-key",
        "--api_key",
        "--token",
        dest="token",
        help="Optional Hugging Face access token.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional benchmark IDs to run. The default is the full GLiClass suite.",
    )
    parser.add_argument(
        "--split",
        help="Force a split for every dataset. By default, test is preferred, then train.",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=8,
        help="GLiNExT inference batch size.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-6,
        help="Top-class threshold. The near-zero default gives top-1 behavior.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device such as 'cuda:0' or 'cpu' (default: auto).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Evaluate at most this many examples per dataset.",
    )
    parser.add_argument("--cache-dir", type=Path, help="Optional Hugging Face datasets/model cache directory.")
    parser.add_argument("--output", type=Path, help="Optional path for a JSON copy of the report.")
    return parser


def select_benchmarks(requested: Sequence[str] | None) -> list[DatasetSpec]:
    if not requested:
        return list(BENCHMARKS)

    unknown = sorted(set(requested) - set(BENCHMARK_BY_NAME))
    if unknown:
        raise ValueError(f"Unknown benchmark(s): {', '.join(unknown)}")
    return [BENCHMARK_BY_NAME[name] for name in requested]


def _select_split(dataset: Any, requested_split: str | None) -> tuple[Any, str]:
    """Select a DatasetDict split using the reference evaluator's preference."""
    if requested_split is not None:
        try:
            return dataset[requested_split], requested_split
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Dataset does not contain requested split {requested_split!r}") from exc

    if hasattr(dataset, "keys"):
        available = set(dataset.keys())
        for split_name in ("test", "train", "validation"):
            if split_name in available:
                return dataset[split_name], split_name
    return dataset, "dataset"


def _feature_class_names(dataset_split: Any, label_column: str) -> list[str] | None:
    features = getattr(dataset_split, "features", None)
    if features is None or label_column not in features:
        return None
    names = getattr(features[label_column], "names", None)
    if names is None:
        return None
    return [str(name) for name in names]


def _normalize_true_labels(raw_labels: Sequence[Any], classes: list[str]) -> list[str]:
    normalized = []
    for label in raw_labels:
        if isinstance(label, Integral):
            label_index = int(label)
            if not 0 <= label_index < len(classes):
                raise ValueError(f"Label index {label_index} is outside the {len(classes)} candidate classes")
            normalized.append(classes[label_index])
        elif isinstance(label, str):
            normalized.append(label)
        else:
            raise ValueError(f"Expected scalar string/integer label, got {label!r}")
    return normalized


def prepare_dataset(
    dataset: Any,
    spec: DatasetSpec,
    *,
    requested_split: str | None = None,
    max_samples: int | None = None,
) -> PreparedDataset:
    """Normalize a loaded Hugging Face dataset for GLiNExT inference."""
    dataset_split, split_name = _select_split(dataset, requested_split)
    try:
        raw_texts = list(dataset_split[spec.text_column])
        raw_labels = list(dataset_split[spec.label_column])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{spec.name} must provide columns {spec.text_column!r} and {spec.label_column!r}") from exc

    if not raw_texts:
        raise ValueError(f"{spec.name}/{split_name} contains no examples")
    if len(raw_texts) != len(raw_labels):
        raise ValueError(f"{spec.name} text/label column lengths do not match")

    if spec.classes is not None:
        classes = list(spec.classes)
    else:
        classes = _feature_class_names(dataset_split, spec.label_column) or list(
            dict.fromkeys(str(label) for label in raw_labels)
        )

    if max_samples is not None:
        raw_texts = raw_texts[:max_samples]
        raw_labels = raw_labels[:max_samples]

    true_labels = _normalize_true_labels(raw_labels, classes)
    unknown_labels = sorted(set(true_labels) - set(classes))
    if unknown_labels:
        raise ValueError(f"{spec.name} contains labels absent from its candidate classes: {unknown_labels}")

    return PreparedDataset(
        name=spec.name,
        split=split_name,
        texts=[str(text) for text in raw_texts],
        classes=classes,
        true_labels=true_labels,
    )


def load_benchmark(
    spec: DatasetSpec,
    *,
    token: str | None,
    cache_dir: Path | None,
) -> Any:
    from datasets import load_dataset  # noqa: PLC0415 - keep metric-only imports lightweight

    kwargs: dict[str, Any] = {}
    if token is not None:
        kwargs["token"] = token
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if spec.config is None:
        return load_dataset(spec.name, **kwargs)
    return load_dataset(spec.name, spec.config, **kwargs)


def _prediction_label(result: Any) -> str | None:
    """Extract the top class name across supported GLiNExT output shapes."""
    while isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
        result = result[0]
    if isinstance(result, dict):
        candidates = [result]
    elif isinstance(result, list):
        candidates = [candidate for candidate in result if isinstance(candidate, dict)]
    else:
        candidates = []
    if not candidates:
        return None

    prediction = max(candidates, key=lambda candidate: float(candidate.get("score", 0.0)))
    label = prediction.get("class_name", prediction.get("label"))
    return str(label) if label is not None else None


def predict_labels(
    model: Any,
    texts: Sequence[str],
    classes: list[str],
    *,
    threshold: float,
    batch_size: int,
) -> list[str | None]:
    """Run native single-label GLiNExT classification in bounded batches."""
    predictions: list[str | None] = []
    for start in range(0, len(texts), batch_size):
        text_batch = list(texts[start : start + batch_size])
        batch_results = model.classify(
            texts=text_batch,
            classes=classes,
            threshold=threshold,
            multi_label=False,
            batch_size=batch_size,
        )
        if len(batch_results) != len(text_batch):
            raise RuntimeError(
                f"GLiNExT returned {len(batch_results)} outputs for {len(text_batch)} classification inputs"
            )
        predictions.extend(_prediction_label(result) for result in batch_results)
    return predictions


def compute_metrics(true_labels: Sequence[str], predictions: Sequence[str | None]) -> ClassificationMetrics:
    """Compute sklearn-compatible single-label F1 averages without sklearn."""
    if len(true_labels) != len(predictions):
        raise ValueError(f"Gold/prediction length mismatch: {len(true_labels)} != {len(predictions)}")
    if not true_labels:
        raise ValueError("Cannot evaluate an empty label sequence")

    labels = sorted(set(true_labels) | {prediction for prediction in predictions if prediction is not None})
    per_label_f1 = []
    supports = []
    for label in labels:
        pairs = zip(true_labels, predictions, strict=True)
        true_positives = sum(
            gold == label and predicted == label for gold, predicted in pairs
        )
        pairs = zip(true_labels, predictions, strict=True)
        false_positives = sum(
            gold != label and predicted == label for gold, predicted in pairs
        )
        pairs = zip(true_labels, predictions, strict=True)
        false_negatives = sum(
            gold == label and predicted != label for gold, predicted in pairs
        )
        support = sum(gold == label for gold in true_labels)
        denominator = 2 * true_positives + false_positives + false_negatives
        per_label_f1.append(2 * true_positives / denominator if denominator else 0.0)
        supports.append(support)

    correct = sum(
        gold == predicted
        for gold, predicted in zip(true_labels, predictions, strict=True)
    )
    covered = sum(prediction is not None for prediction in predictions)
    accuracy = correct / len(true_labels)
    macro_f1 = sum(per_label_f1) / len(per_label_f1) if per_label_f1 else 0.0
    weighted_f1 = sum(
        score * support
        for score, support in zip(per_label_f1, supports, strict=True)
    ) / len(true_labels)
    return ClassificationMetrics(
        samples=len(true_labels),
        correct=correct,
        covered=covered,
        accuracy=accuracy,
        micro_f1=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        coverage=covered / len(true_labels),
    )


def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    import torch  # noqa: PLC0415 - keep --help and metric-only imports lightweight

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_model(model_id: str, *, device: str, token: str | None, cache_dir: Path | None) -> Any:
    from glinext import GLiNExT  # noqa: PLC0415 - defer the torch-heavy import

    model = GLiNExT.from_pretrained(
        model_id,
        token=token,
        cache_dir=cache_dir,
        load_tokenizer=True,
        map_location=device,
    )
    model.to(device)
    model.eval()
    return model


def print_report(results: dict[str, ClassificationMetrics]) -> None:
    name_width = max(24, *(len(name) for name in results))
    print()
    print(
        f"{'Dataset':<{name_width}}  {'Samples':>8}  {'Accuracy':>9}  {'Micro F1':>9}  "
        f"{'Macro F1':>9}  {'Weighted':>9}  {'Coverage':>9}"
    )
    print("-" * (name_width + 72))
    for name, metrics in results.items():
        print(
            f"{name:<{name_width}}  {metrics.samples:>8,d}  {metrics.accuracy:>8.2%}  "
            f"{metrics.micro_f1:>8.2%}  {metrics.macro_f1:>8.2%}  "
            f"{metrics.weighted_f1:>8.2%}  {metrics.coverage:>8.2%}"
        )
    average_macro_f1 = sum(metrics.macro_f1 for metrics in results.values()) / len(results)
    print("-" * (name_width + 72))
    print(f"{'Average macro F1':<{name_width}}  {'':>8}  {'':>9}  {'':>9}  {average_macro_f1:>8.2%}")


def save_report(
    output_path: Path,
    results: dict[str, ClassificationMetrics],
    *,
    model_id: str,
    threshold: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_id,
        "threshold": threshold,
        "average_macro_f1": sum(metrics.macro_f1 for metrics in results.values()) / len(results),
        "datasets": {name: asdict(metrics) for name, metrics in results.items()},
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")


def main() -> None:
    args = create_parser().parse_args()
    if not 0.0 < args.threshold <= 1.0:
        raise ValueError("--threshold must be greater than 0 and no greater than 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be at least 1")

    benchmarks = select_benchmarks(args.datasets)
    device = resolve_device(args.device)
    print(f"Loading {args.model} on {device} ...")
    model = load_model(args.model, device=device, token=args.token, cache_dir=args.cache_dir)

    results: dict[str, ClassificationMetrics] = {}
    for index, spec in enumerate(benchmarks, start=1):
        print(f"[{index}/{len(benchmarks)}] Loading {spec.name} ...")
        dataset = load_benchmark(
            spec,
            token=args.token,
            cache_dir=args.cache_dir,
        )
        prepared = prepare_dataset(
            dataset,
            spec,
            requested_split=args.split,
            max_samples=args.max_samples,
        )
        print(
            f"  Classifying {len(prepared.texts):,} {prepared.split} examples "
            f"against {len(prepared.classes)} classes ..."
        )
        predictions = predict_labels(
            model,
            prepared.texts,
            prepared.classes,
            threshold=args.threshold,
            batch_size=args.batch_size,
        )
        metrics = compute_metrics(prepared.true_labels, predictions)
        results[spec.name] = metrics
        print(
            f"  macro F1={metrics.macro_f1:.2%}  micro F1={metrics.micro_f1:.2%}  weighted F1={metrics.weighted_f1:.2%}"
        )

    print_report(results)
    if args.output is not None:
        save_report(args.output, results, model_id=args.model, threshold=args.threshold)
        print(f"\nSaved JSON report to {args.output}")


if __name__ == "__main__":
    main()
