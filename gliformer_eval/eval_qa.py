#!/usr/bin/env python3
"""Evaluate a GLiFormer checkpoint on extractive question answering (SQuAD).

GLiFormer has no QA head, so extractive QA is posed as NER over a prompted text:
the question is prepended to the passage and a single entity type -- ``answer``
by default -- is requested::

    Who founded Amazon?
    Amazon was founded in 1994 by Jeff Bezos in Bellevue, Washington ...

The highest-scoring predicted span that starts inside the passage becomes the
answer; spans that start inside the prepended question are discarded unless
``--allow-question-spans`` is set.

Reported scores are the official SQuAD metrics -- exact match and token-level
F1 after the standard normalization (lowercase, drop punctuation and articles,
collapse whitespace), maximized over each example's gold answers.  A strict
character-offset exact match is reported alongside them, because GLiFormer
returns offsets and SQuAD provides ``answer_start``.  ``rajpurkar/squad_v2``
also works: examples without a gold answer count as correct when the model
predicts nothing above ``--threshold``.
"""

# Console output is the primary report produced by this CLI.
# ruff: noqa: T201

from __future__ import annotations

import argparse
import json
import random
import re
import string
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DATASET_ID = "rajpurkar/squad"
DEFAULT_LABEL = "answer"
DEFAULT_SEPARATOR = "\n"


@dataclass(frozen=True)
class QAExample:
    """One question, its passage, and the gold answers with their offsets."""

    example_id: str
    question: str
    context: str
    answers: tuple[str, ...]
    answer_spans: tuple[tuple[int, int], ...]

    @property
    def is_answerable(self) -> bool:
        return bool(self.answers)


@dataclass(frozen=True)
class QAPrediction:
    """The selected answer span, with offsets relative to the passage.

    ``start`` is negative both when nothing was predicted and, with
    ``--allow-question-spans``, when the span starts inside the question.
    """

    text: str
    score: float
    start: int
    end: int

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass(frozen=True)
class QAMetrics:
    """SQuAD metrics for one evaluated split."""

    samples: int
    answerable: int
    unanswerable: int
    answered: int
    coverage: float
    exact_match: float
    f1: float
    span_exact_match: float
    has_answer_exact_match: float | None
    has_answer_f1: float | None
    has_answer_span_exact_match: float | None
    no_answer_accuracy: float | None
    mean_score: float


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate GLiFormer on extractive QA by prepending the question to the passage.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random sampling seed (default: 42).")
    parser.add_argument(
        "--model",
        required=True,
        help="Local GLiFormer checkpoint path or Hugging Face model ID.",
    )
    parser.add_argument(
        "--dataset",
        default=DATASET_ID,
        help=f"Hugging Face SQuAD-style dataset ID (default: {DATASET_ID}).",
    )
    parser.add_argument(
        "--config",
        help="Optional dataset configuration name.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split to evaluate (default: validation).",
    )
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help=f"Entity type requested from the NER head (default: {DEFAULT_LABEL!r}).",
    )
    parser.add_argument(
        "--separator",
        default=DEFAULT_SEPARATOR,
        help=r"Separator inserted between question and passage; escapes such as '\n' are decoded (default: '\n').",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Span confidence threshold.",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=16,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device such as 'cuda:0' or 'cpu' (default: auto).",
    )
    parser.add_argument(
        "--allow-question-spans",
        action="store_true",
        help="Keep predicted spans that start inside the prepended question instead of discarding them.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Evaluate only the first N examples, useful for smoke tests.",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=0,
        help="Print the first N scored examples for inspection.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional Hugging Face model and dataset cache directory.",
    )
    parser.add_argument(
        "--token",
        "--api-key",
        "--api_key",
        dest="token",
        help="Optional Hugging Face access token.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the model and dataset only from local caches.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for a JSON copy of the evaluation report.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Optional path for a JSONL dump of per-example predictions and scores.",
    )
    return parser


def decode_separator(raw_separator: str) -> str:
    r"""Turn a shell-supplied separator such as ``'\n'`` into a real newline."""
    try:
        return raw_separator.encode("utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return raw_separator


# ── Dataset loading ──────────────────────────────────────────────────────────


def _gold_spans(context: str, answers: Sequence[str], answer_starts: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Return verified character spans for the gold answers.

    Offsets that do not line up with the passage are re-resolved by searching
    the passage, and dropped when the answer text is absent altogether.
    """
    spans: list[tuple[int, int]] = []
    for index, answer in enumerate(answers):
        start = int(answer_starts[index]) if index < len(answer_starts) else -1
        if start < 0 or context[start : start + len(answer)] != answer:
            start = context.find(answer)
        if start < 0:
            continue
        spans.append((start, start + len(answer)))
    return tuple(dict.fromkeys(spans))


def _load_hf_split(
    dataset_id: str,
    split: str,
    *,
    config: str | None,
    cache_dir: Path | None,
    token: str | None,
    local_files_only: bool,
) -> Any:
    """Load one split, tolerating dataset-viewer metadata this install cannot parse.

    ``load_dataset`` prefers the dataset viewer's exported infos over the repo's
    own YAML metadata.  Those infos are serialized by whatever ``datasets``
    release the viewer runs, so a newer feature type -- ``{"_type": "List"}``,
    written where older releases wrote ``Sequence`` -- makes an older local
    install raise ``TypeError`` before it ever reads a row.  ``rajpurkar/squad_v2``
    hits this on ``datasets`` 3.3.0.  The repo YAML describes the same columns,
    so retry with the export disabled instead of failing the evaluation.
    """
    from datasets import DownloadConfig, load_dataset  # noqa: PLC0415 - optional data dependency
    from datasets import config as datasets_config  # noqa: PLC0415 - optional data dependency

    download_config = DownloadConfig(local_files_only=True) if local_files_only else None
    load_kwargs = {
        "split": split,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "token": token,
        "download_config": download_config,
    }
    try:
        return load_dataset(dataset_id, config, **load_kwargs)
    except TypeError:
        previous = datasets_config.USE_PARQUET_EXPORT
        datasets_config.USE_PARQUET_EXPORT = False
        try:
            dataset = load_dataset(dataset_id, config, **load_kwargs)
        finally:
            datasets_config.USE_PARQUET_EXPORT = previous
        print(f"Note: loaded {dataset_id} from its repo metadata; the exported dataset infos were unreadable.")
        return dataset


def load_qa_split(
    dataset_id: str,
    split: str,
    *,
    config: str | None = None,
    cache_dir: Path | None = None,
    token: str | None = None,
    local_files_only: bool = False,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[QAExample]:
    """Load and validate a SQuAD-style split from Hugging Face Datasets."""
    dataset = _load_hf_split(
        dataset_id,
        split,
        config=config,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    required_columns = {"question", "context", "answers"}
    missing_columns = required_columns - set(dataset.column_names)
    if missing_columns:
        raise ValueError(f"{dataset_id}/{split} is missing required column(s): {', '.join(sorted(missing_columns))}")
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(random.Random(seed).sample(range(len(dataset)), max_samples))

    examples: list[QAExample] = []
    for index, row in enumerate(dataset):
        answers = row["answers"]
        if not isinstance(answers, dict) or "text" not in answers:
            raise ValueError(f"{dataset_id}/{split} row {index} has no SQuAD-style 'answers' mapping: {answers!r}")
        context = str(row["context"])
        answer_texts = tuple(str(answer) for answer in answers["text"])
        examples.append(
            QAExample(
                example_id=str(row.get("id", index)),
                question=str(row["question"]).strip(),
                context=context,
                answers=answer_texts,
                answer_spans=_gold_spans(context, answer_texts, answers.get("answer_start", ())),
            )
        )
    if not examples:
        raise ValueError(f"{dataset_id}/{split} contains no examples")
    return examples


# ── Prompting and prediction ─────────────────────────────────────────────────


def build_input_text(example: QAExample, separator: str) -> tuple[str, int]:
    """Prepend the question to the passage and report where the passage starts."""
    prefix = f"{example.question}{separator}"
    return f"{prefix}{example.context}", len(prefix)


def select_prediction(
    entities: Iterable[dict[str, Any]],
    prefix_length: int,
    *,
    allow_question_spans: bool,
) -> QAPrediction:
    """Pick the highest-scoring span, with offsets rebased onto the passage."""
    best: dict[str, Any] | None = None
    for entity in entities:
        start = int(entity["start"])
        if not allow_question_spans and start < prefix_length:
            continue
        if best is None or float(entity["score"]) > float(best["score"]):
            best = entity
    if best is None:
        return QAPrediction(text="", score=0.0, start=-1, end=-1)
    return QAPrediction(
        text=str(best["text"]),
        score=float(best["score"]),
        start=int(best["start"]) - prefix_length,
        end=int(best["end"]) - prefix_length,
    )


def _batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _progress(items: Sequence[Any], total: int) -> Iterable[Any]:
    try:
        from tqdm import tqdm  # noqa: PLC0415 - optional progress dependency
    except ImportError:
        return iter(items)
    return iter(tqdm(items, total=total, desc="Answering", unit="batch"))


def predict_answers(
    model: Any,
    examples: Sequence[QAExample],
    *,
    label: str,
    separator: str,
    threshold: float,
    batch_size: int,
    allow_question_spans: bool,
) -> list[QAPrediction]:
    """Run NER over question-prefixed passages and select one answer each."""
    prompts = [build_input_text(example, separator) for example in examples]
    predictions: list[QAPrediction] = []
    batches = list(_batched(prompts, batch_size))
    for batch in _progress(batches, len(batches)):
        batch_entities = model.predict_entities(
            texts=[text for text, _ in batch],
            entities=[label],
            flat_ner=True,
            threshold=threshold,
            batch_size=batch_size,
        )
        if len(batch_entities) != len(batch):
            raise RuntimeError(
                f"GLiFormer returned {len(batch_entities)} prediction lists for {len(batch)} inputs",
            )
        for entities, (_, prefix_length) in zip(batch_entities, batch, strict=True):
            predictions.append(
                select_prediction(entities, prefix_length, allow_question_spans=allow_question_spans)
            )
    return predictions


# ── SQuAD metrics ────────────────────────────────────────────────────────────

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = frozenset(string.punctuation)


def normalize_answer(text: str) -> str:
    """Official SQuAD normalization: lowercase, drop punctuation and articles."""
    lowered = text.lower()
    unpunctuated = "".join(character for character in lowered if character not in _PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", unpunctuated).split())


def exact_match_score(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1_score(prediction: str, gold: str) -> float:
    """Token-overlap F1 between two normalized answer strings."""
    predicted_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not predicted_tokens or not gold_tokens:
        # Matches the official script: two empty answers agree, one does not.
        return float(predicted_tokens == gold_tokens)
    overlap = sum((Counter(predicted_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_example(example: QAExample, prediction: QAPrediction) -> tuple[float, float, float]:
    """Return ``(exact_match, f1, span_exact_match)`` for one example."""
    if not example.is_answerable:
        abstained = float(prediction.is_empty)
        return abstained, abstained, abstained
    if prediction.is_empty:
        return 0.0, 0.0, 0.0
    exact_match = max(exact_match_score(prediction.text, gold) for gold in example.answers)
    f1 = max(token_f1_score(prediction.text, gold) for gold in example.answers)
    span_exact_match = float((prediction.start, prediction.end) in example.answer_spans)
    return exact_match, f1, span_exact_match


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_metrics(
    examples: Sequence[QAExample],
    predictions: Sequence[QAPrediction],
) -> tuple[QAMetrics, list[tuple[float, float, float]]]:
    """Aggregate per-example scores, keeping the answerable subset separate."""
    if len(examples) != len(predictions):
        raise ValueError(f"Example/prediction count mismatch: {len(examples)} != {len(predictions)}")

    scores = [score_example(example, prediction) for example, prediction in zip(examples, predictions, strict=True)]
    answerable = [index for index, example in enumerate(examples) if example.is_answerable]
    unanswerable = [index for index, example in enumerate(examples) if not example.is_answerable]
    answered = [prediction for prediction in predictions if not prediction.is_empty]

    metrics = QAMetrics(
        samples=len(examples),
        answerable=len(answerable),
        unanswerable=len(unanswerable),
        answered=len(answered),
        coverage=len(answered) / len(examples),
        exact_match=_mean([score[0] for score in scores]),
        f1=_mean([score[1] for score in scores]),
        span_exact_match=_mean([score[2] for score in scores]),
        has_answer_exact_match=_mean([scores[index][0] for index in answerable]) if answerable else None,
        has_answer_f1=_mean([scores[index][1] for index in answerable]) if answerable else None,
        has_answer_span_exact_match=_mean([scores[index][2] for index in answerable]) if answerable else None,
        no_answer_accuracy=_mean([scores[index][0] for index in unanswerable]) if unanswerable else None,
        mean_score=_mean([prediction.score for prediction in answered]),
    )
    return metrics, scores


# ── Model loading and reporting ──────────────────────────────────────────────


def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    import torch  # noqa: PLC0415 - keep --help and metric-only imports lightweight

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_model(
    model_id: str,
    *,
    device: str,
    cache_dir: Path | None = None,
    token: str | None = None,
    local_files_only: bool = False,
) -> Any:
    """Load a NER-capable GLiFormer checkpoint."""
    from gliformer import GLiFormer  # noqa: PLC0415 - defer the torch-heavy import

    model = GLiFormer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
        load_tokenizer=True,
        map_location=device,
    )
    if getattr(model.config, "ner_config", None) is None:
        raise ValueError(f"GLiFormer checkpoint {model_id!r} does not enable a NER head")
    model.to(device)
    model.eval()
    return model


def print_report(metrics: QAMetrics, *, dataset: str, split: str, label: str) -> None:
    rows: list[tuple[str, str]] = [
        ("Dataset", f"{dataset} [{split}]"),
        ("Entity label", label),
        ("Samples", f"{metrics.samples:,d}"),
        ("Answerable", f"{metrics.answerable:,d}"),
        ("Unanswerable", f"{metrics.unanswerable:,d}"),
        ("Answered", f"{metrics.answered:,d} ({metrics.coverage:.2%})"),
        ("Exact match", f"{metrics.exact_match:.2%}"),
        ("Token F1", f"{metrics.f1:.2%}"),
        ("Span exact match", f"{metrics.span_exact_match:.2%}"),
    ]
    if metrics.unanswerable:
        rows.extend(
            [
                ("HasAns exact match", f"{metrics.has_answer_exact_match:.2%}"),
                ("HasAns token F1", f"{metrics.has_answer_f1:.2%}"),
                ("NoAns accuracy", f"{metrics.no_answer_accuracy:.2%}"),
            ]
        )
    rows.append(("Mean answer score", f"{metrics.mean_score:.3f}"))

    label_width = max(len(name) for name, _ in rows)
    print()
    print(f"{'Metric':<{label_width}}  Value")
    print("-" * (label_width + 24))
    for name, value in rows:
        print(f"{name:<{label_width}}  {value}")


def print_examples(
    examples: Sequence[QAExample],
    predictions: Sequence[QAPrediction],
    scores: Sequence[tuple[float, float, float]],
    count: int,
) -> None:
    print()
    print(f"First {min(count, len(examples))} example(s):")
    for example, prediction, score in list(zip(examples, predictions, scores, strict=True))[:count]:
        gold = " | ".join(example.answers) if example.answers else "<no answer>"
        predicted = prediction.text if not prediction.is_empty else "<no answer>"
        print(f"  Q: {example.question}")
        print(f"    gold: {gold}")
        print(f"    pred: {predicted}  (score={prediction.score:.3f}, EM={score[0]:.0f}, F1={score[1]:.2f})")


def save_report(
    output_path: Path,
    metrics: QAMetrics,
    *,
    model_id: str,
    dataset: str,
    config: str | None,
    split: str,
    label: str,
    separator: str,
    threshold: float,
    allow_question_spans: bool,
    seed: int = 42,
    max_samples: int | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_id,
        "seed": seed, "max_samples": max_samples,
        "dataset": dataset,
        "config": config,
        "split": split,
        "label": label,
        "separator": separator,
        "threshold": threshold,
        "allow_question_spans": allow_question_spans,
        "metrics": asdict(metrics),
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")


def save_predictions(
    output_path: Path,
    examples: Sequence[QAExample],
    predictions: Sequence[QAPrediction],
    scores: Sequence[tuple[float, float, float]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for example, prediction, score in zip(examples, predictions, scores, strict=True):
            row = {
                "id": example.example_id,
                "question": example.question,
                "gold_answers": list(example.answers),
                "gold_spans": [list(span) for span in example.answer_spans],
                "prediction": prediction.text,
                "prediction_span": [prediction.start, prediction.end],
                "score": prediction.score,
                "exact_match": score[0],
                "f1": score[1],
                "span_exact_match": score[2],
            }
            json.dump(row, output_file, ensure_ascii=False)
            output_file.write("\n")


def main() -> None:
    args = create_parser().parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be at least 1")
    if args.show_examples < 0:
        raise ValueError("--show-examples must not be negative")
    if not args.label.strip():
        raise ValueError("--label must not be empty")

    separator = decode_separator(args.separator)
    print(f"Loading {args.dataset} [{args.split}] ...")
    examples = load_qa_split(
        args.dataset,
        args.split,
        config=args.config,
        cache_dir=args.cache_dir,
        token=args.token,
        local_files_only=args.local_files_only,
        max_samples=args.max_samples, seed=args.seed,
    )
    device = resolve_device(args.device)
    print(f"Loading {args.model} on {device} ...")
    model = load_model(
        args.model,
        device=device,
        cache_dir=args.cache_dir,
        token=args.token,
        local_files_only=args.local_files_only,
    )

    print(f"Answering {len(examples):,} question(s) as {args.label!r} spans ...")
    predictions = predict_answers(
        model,
        examples,
        label=args.label,
        separator=separator,
        threshold=args.threshold,
        batch_size=args.batch_size,
        allow_question_spans=args.allow_question_spans,
    )
    metrics, scores = compute_metrics(examples, predictions)

    if args.show_examples:
        print_examples(examples, predictions, scores, args.show_examples)
    print_report(metrics, dataset=args.dataset, split=args.split, label=args.label)

    if args.output is not None:
        save_report(
            args.output,
            metrics,
            model_id=args.model,
            dataset=args.dataset,
            config=args.config,
            split=args.split,
            label=args.label,
            separator=separator,
            threshold=args.threshold,
            allow_question_spans=args.allow_question_spans, seed=args.seed, max_samples=args.max_samples,
        )
        print(f"\nSaved JSON report to {args.output}")
    if args.predictions is not None:
        save_predictions(args.predictions, examples, predictions, scores)
        print(f"Saved per-example predictions to {args.predictions}")


if __name__ == "__main__":
    main()
