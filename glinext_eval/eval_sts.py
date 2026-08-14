#!/usr/bin/env python3
"""Evaluate GLiNext sentence embeddings on the STS Benchmark.

The ``sentence-transformers/stsb`` dataset contains ``sentence1``,
``sentence2``, and a human similarity ``score`` normalized to ``[0, 1]``.
This evaluator embeds both sentences with GLiNext, calculates pairwise model
similarities, and reports Pearson and Spearman correlation coefficients.
"""

# Console output is the primary report produced by this CLI.
# ruff: noqa: T201

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

if TYPE_CHECKING:
    import torch


DATASET_ID = "sentence-transformers/stsb"
SIMILARITY_FUNCTIONS = ("auto", "cosine", "dot", "l2")


@dataclass(frozen=True)
class CorrelationResult:
    """Pearson and Spearman results for one STS split."""

    dataset: str
    split: str
    samples: int
    similarity: str
    pearson: float
    pearson_pvalue: float
    spearman: float
    spearman_pvalue: float
    gold_mean: float
    prediction_mean: float
    prediction_std: float


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate GLiNext embeddings on sentence-transformers/stsb.")
    parser.add_argument(
        "--model",
        required=True,
        help="Local GLiNext checkpoint path or Hugging Face model ID.",
    )
    parser.add_argument(
        "--dataset",
        default=DATASET_ID,
        help=f"Hugging Face STS dataset ID (default: {DATASET_ID}).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=32,
        help="Sentence embedding batch size.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device such as 'cuda:0' or 'cpu' (default: auto).",
    )
    parser.add_argument(
        "--similarity",
        choices=SIMILARITY_FUNCTIONS,
        default="auto",
        help="Pair scoring function. Auto uses the checkpoint embedding configuration.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Evaluate only the first N examples, useful for smoke tests.",
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
    return parser


def load_sts_split(
    dataset_id: str,
    split: str,
    *,
    cache_dir: Path | None = None,
    token: str | None = None,
    local_files_only: bool = False,
) -> tuple[list[str], list[str], np.ndarray]:
    """Load and validate an STS split from Hugging Face Datasets."""
    from datasets import DownloadConfig, load_dataset  # noqa: PLC0415 - optional data dependency

    download_config = DownloadConfig(local_files_only=True) if local_files_only else None
    dataset = load_dataset(
        dataset_id,
        split=split,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        token=token,
        download_config=download_config,
    )
    required_columns = {"sentence1", "sentence2", "score"}
    missing_columns = required_columns - set(dataset.column_names)
    if missing_columns:
        raise ValueError(f"{dataset_id}/{split} is missing required column(s): {', '.join(sorted(missing_columns))}")

    sentences1 = [str(sentence) for sentence in dataset["sentence1"]]
    sentences2 = [str(sentence) for sentence in dataset["sentence2"]]
    scores = np.asarray(dataset["score"], dtype=np.float64)
    validate_sts_data(sentences1, sentences2, scores)
    return sentences1, sentences2, scores


def validate_sts_data(
    sentences1: Sequence[str],
    sentences2: Sequence[str],
    scores: Sequence[float] | np.ndarray,
) -> None:
    """Validate aligned sentence pairs and finite correlation targets."""
    if len(sentences1) != len(sentences2) or len(sentences1) != len(scores):
        raise ValueError(
            f"STS column length mismatch: sentence1={len(sentences1)}, sentence2={len(sentences2)}, score={len(scores)}"
        )
    if len(scores) < 2:
        raise ValueError("STS evaluation requires at least two sentence pairs")
    if not np.isfinite(np.asarray(scores, dtype=np.float64)).all():
        raise ValueError("STS gold scores must all be finite")


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
    """Load an embedding-capable GLiNext checkpoint."""
    from glinext import GLiNExT  # noqa: PLC0415 - defer the torch-heavy import

    model = GLiNExT.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
        load_tokenizer=True,
        map_location=device,
    )
    if getattr(model.config, "embedding_config", None) is None:
        raise ValueError(f"GLiNext checkpoint {model_id!r} does not enable an embedding head")
    model.to(device)
    model.eval()
    return model


def resolve_similarity(model: Any, requested_similarity: str) -> str:
    """Resolve ``auto`` from the checkpoint and validate the result."""
    if requested_similarity != "auto":
        return requested_similarity

    embedding_config = getattr(model.config, "embedding_config", None)
    similarity = getattr(embedding_config, "similarity_fn", "cosine")
    if similarity not in SIMILARITY_FUNCTIONS[1:]:
        raise ValueError(
            f"Unsupported checkpoint similarity_fn {similarity!r}; "
            f"choose one of {', '.join(SIMILARITY_FUNCTIONS[1:])} explicitly"
        )
    return similarity


def embed_sentence_pairs(
    model: Any,
    sentences1: Sequence[str],
    sentences2: Sequence[str],
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed both STS columns in one call and restore pair alignment."""
    import torch  # noqa: PLC0415 - used only during model inference

    validate_sts_data(sentences1, sentences2, np.zeros(len(sentences1)))
    sentence_count = len(sentences1)
    embeddings = model.embed_text(
        [*sentences1, *sentences2],
        batch_size=batch_size,
    )
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise RuntimeError(
            "GLiNext embed_text() must return a rank-2 torch.Tensor, got "
            f"{type(embeddings).__name__} with shape {getattr(embeddings, 'shape', None)}"
        )
    if embeddings.shape[0] != sentence_count * 2:
        raise RuntimeError(
            f"GLiNext returned {embeddings.shape[0]} embeddings for {sentence_count * 2} input sentences"
        )
    return embeddings[:sentence_count], embeddings[sentence_count:]


def compute_similarities(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor,
    similarity: str,
) -> np.ndarray:
    """Calculate pairwise similarity using GLiNext embedding-head semantics."""
    import torch  # noqa: PLC0415 - used only during model inference
    from torch.nn import functional  # noqa: PLC0415

    if embeddings1.shape != embeddings2.shape or embeddings1.ndim != 2:
        raise ValueError(f"Embedding shape mismatch: {tuple(embeddings1.shape)} != {tuple(embeddings2.shape)}")
    if similarity == "cosine":
        predictions = functional.cosine_similarity(embeddings1.float(), embeddings2.float(), dim=-1)
    elif similarity == "dot":
        predictions = (embeddings1.float() * embeddings2.float()).sum(dim=-1)
    elif similarity == "l2":
        predictions = -torch.linalg.vector_norm(embeddings1.float() - embeddings2.float(), dim=-1)
    else:
        raise ValueError(f"Unsupported similarity function: {similarity!r}")

    prediction_array = predictions.detach().cpu().numpy().astype(np.float64, copy=False)
    if not np.isfinite(prediction_array).all():
        raise ValueError("GLiNext produced non-finite STS similarity predictions")
    return prediction_array


def compute_correlations(
    gold_scores: Sequence[float] | np.ndarray,
    predictions: Sequence[float] | np.ndarray,
    *,
    dataset: str = DATASET_ID,
    split: str = "test",
    similarity: str = "cosine",
) -> CorrelationResult:
    """Compute Pearson and Spearman coefficients and two-sided p-values."""
    gold = np.asarray(gold_scores, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    if gold.shape != predicted.shape or gold.ndim != 1:
        raise ValueError(f"Gold/prediction shape mismatch: {gold.shape} != {predicted.shape}")
    if gold.size < 2:
        raise ValueError("Correlation requires at least two observations")
    if not np.isfinite(gold).all() or not np.isfinite(predicted).all():
        raise ValueError("Gold scores and predictions must all be finite")
    if np.ptp(gold) == 0 or np.ptp(predicted) == 0:
        raise ValueError("Pearson and Spearman correlations are undefined for constant inputs")

    pearson = pearsonr(gold, predicted)
    spearman = spearmanr(gold, predicted)
    return CorrelationResult(
        dataset=dataset,
        split=split,
        samples=int(gold.size),
        similarity=similarity,
        pearson=float(pearson.statistic),
        pearson_pvalue=float(pearson.pvalue),
        spearman=float(spearman.statistic),
        spearman_pvalue=float(spearman.pvalue),
        gold_mean=float(gold.mean()),
        prediction_mean=float(predicted.mean()),
        prediction_std=float(predicted.std()),
    )


def evaluate(
    model: Any,
    sentences1: Sequence[str],
    sentences2: Sequence[str],
    gold_scores: Sequence[float] | np.ndarray,
    *,
    dataset: str = DATASET_ID,
    split: str = "test",
    similarity: str = "auto",
    batch_size: int = 32,
) -> CorrelationResult:
    """Embed an STS split, calculate similarities, and correlate with gold."""
    validate_sts_data(sentences1, sentences2, gold_scores)
    resolved_similarity = resolve_similarity(model, similarity)
    embeddings1, embeddings2 = embed_sentence_pairs(
        model,
        sentences1,
        sentences2,
        batch_size=batch_size,
    )
    predictions = compute_similarities(embeddings1, embeddings2, resolved_similarity)
    return compute_correlations(
        gold_scores,
        predictions,
        dataset=dataset,
        split=split,
        similarity=resolved_similarity,
    )


def print_report(result: CorrelationResult) -> None:
    print()
    print("GLiNext STS evaluation")
    print(f"Dataset:    {result.dataset}")
    print(f"Split:      {result.split}")
    print(f"Samples:    {result.samples:,}")
    print(f"Similarity: {result.similarity}")
    print(f"Pearson:    {result.pearson:.6f}  (p={result.pearson_pvalue:.3g})")
    print(f"Spearman:   {result.spearman:.6f}  (p={result.spearman_pvalue:.3g})")


def save_report(output_path: Path, result: CorrelationResult, *, model_id: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model_id, **asdict(result)}
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")


def main() -> None:
    args = create_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.max_samples is not None and args.max_samples < 2:
        raise ValueError("--max-samples must be at least 2")

    sentences1, sentences2, gold_scores = load_sts_split(
        args.dataset,
        args.split,
        cache_dir=args.cache_dir,
        token=args.token,
        local_files_only=args.local_files_only,
    )
    if args.max_samples is not None:
        sentences1 = sentences1[: args.max_samples]
        sentences2 = sentences2[: args.max_samples]
        gold_scores = gold_scores[: args.max_samples]

    device = resolve_device(args.device)
    print(f"Loading {args.model} on {device} ...")
    model = load_model(
        args.model,
        device=device,
        cache_dir=args.cache_dir,
        token=args.token,
        local_files_only=args.local_files_only,
    )
    print(f"Evaluating {len(gold_scores):,} pairs from {args.dataset}/{args.split} ...")
    result = evaluate(
        model,
        sentences1,
        sentences2,
        gold_scores,
        dataset=args.dataset,
        split=args.split,
        similarity=args.similarity,
        batch_size=args.batch_size,
    )
    print_report(result)

    if args.output is not None:
        save_report(args.output, result, model_id=args.model)
        print(f"Saved JSON report to {args.output}")


if __name__ == "__main__":
    main()
