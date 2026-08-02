#!/usr/bin/env python3
"""Convert Kwash67/layoutlmv3-invoice-dataset to GLiNExT layout format.

The source dataset is already preprocessed for LayoutLMv3 and stores parquet
rows with ``input_ids``, ``attention_mask``, ``bbox``, ``labels``, and large
``pixel_values`` tensors. This converter deliberately reads only the text,
layout, and BIO labels, then writes GLiNExT layout JSON arrays:

    data/layoutlmv3_invoice_train.json
    data/layoutlmv3_invoice_valid.json
    data/layoutlmv3_invoice_test.json

Rows contain word-level ``tokenized_text``/``bboxes`` plus GLiNExT
``extraction`` spans. The source labels are token indices after LayoutLMv3
subword tokenization, so the converter merges continuation subtokens into the
preceding labeled word and marks the output spans as already resolved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ID = "Kwash67/layoutlmv3-invoice-dataset"
TOKENIZER_NAME = "microsoft/layoutlmv3-base"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PREFIX = REPO_ROOT / "data" / "layoutlmv3_invoice"
PARQUET_COLUMNS = ["input_ids", "attention_mask", "bbox", "labels"]


def download_file(repo_id: str, filename: str, revision: str | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required to download this dataset.") from exc

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
        )
    )


def list_dataset_files(repo_id: str, revision: str | None) -> list[str]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required to list this dataset.") from exc

    return HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision)


def load_label_info(repo_id: str, revision: str | None, label_info: Path | None) -> dict[str, Any]:
    path = label_info or download_file(repo_id, "label_info.json", revision)
    with open(path, encoding="utf-8") as fin:
        info = json.load(fin)
    if "id2label" not in info:
        raise ValueError(f"label metadata missing id2label: {path}")
    return info


def load_tokenizer(tokenizer_name: str):
    # Some local environments ship a minimal wcwidth module. Importing
    # transformers can register a torch.dynamo atexit hook that expects
    # wcwidth.wcswidth to exist, so provide the small compatibility surface.
    try:
        import wcwidth

        if not hasattr(wcwidth, "wcswidth"):
            wcwidth.wcswidth = lambda text: sum(  # type: ignore[attr-defined]
                max(wcwidth.wcwidth(char), 0) for char in str(text)
            )
    except Exception:
        pass

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers is required to decode LayoutLMv3 input_ids.") from exc

    return AutoTokenizer.from_pretrained(tokenizer_name)


def label_names_without_bio(labels: Iterable[str]) -> list[str]:
    names = set()
    for label in labels:
        if label == "O" or label.startswith("B-") or label.startswith("I-"):
            if label.startswith(("B-", "I-")):
                names.add(label[2:])
            continue
        names.add(label)
    return sorted(names)


def normalize_bbox(raw_bbox: Any, *, clamp: bool) -> list[int]:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raise ValueError(f"expected bbox with 4 coordinates, got {raw_bbox!r}")
    bbox = [int(round(float(value))) for value in raw_bbox]
    if clamp:
        bbox = [max(0, min(1000, value)) for value in bbox]
    return bbox


def decode_piece(tokenizer, ids: list[int]) -> str:
    text = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
    text = text.replace("\u00a0", " ").strip()
    if text:
        return text
    tokens = tokenizer.convert_ids_to_tokens(ids)
    return "".join(str(token).lstrip("Ġ") for token in tokens).strip()


def merge_subwords(row: dict[str, Any], tokenizer, id2label: dict[int, str], *, clamp_bboxes: bool):
    input_ids = row["input_ids"]
    attention_mask = row["attention_mask"]
    bboxes = row["bbox"]
    labels = row["labels"]

    words: list[str] = []
    word_bboxes: list[list[int]] = []
    word_labels: list[str] = []

    current_ids: list[int] = []
    current_bbox: list[int] | None = None
    current_label: str | None = None

    def flush_current() -> None:
        nonlocal current_ids, current_bbox, current_label
        if current_ids and current_bbox is not None and current_label is not None:
            text = decode_piece(tokenizer, current_ids)
            if text:
                words.append(text)
                word_bboxes.append(current_bbox)
                word_labels.append(current_label)
        current_ids = []
        current_bbox = None
        current_label = None

    for token_id, mask, bbox, label_id in zip(input_ids, attention_mask, bboxes, labels):
        if int(mask) == 0:
            break
        label_id = int(label_id)
        if label_id == -100:
            if current_ids:
                current_ids.append(int(token_id))
            continue

        flush_current()
        current_ids = [int(token_id)]
        current_bbox = normalize_bbox(bbox, clamp=clamp_bboxes)
        current_label = id2label.get(label_id, "O")

    flush_current()
    return words, word_bboxes, word_labels


def bio_to_spans(labels: list[str]) -> list[list[Any]]:
    spans: list[list[Any]] = []
    start: int | None = None
    active_label: str | None = None

    def close(end_idx: int) -> None:
        nonlocal start, active_label
        if start is not None and active_label is not None:
            spans.append([start, end_idx, active_label])
        start = None
        active_label = None

    for idx, raw_label in enumerate(labels):
        if raw_label == "O" or not raw_label:
            close(idx - 1)
            continue

        if raw_label.startswith("B-"):
            close(idx - 1)
            start = idx
            active_label = raw_label[2:]
            continue

        if raw_label.startswith("I-"):
            entity_label = raw_label[2:]
            if active_label == entity_label and start is not None:
                continue
            close(idx - 1)
            start = idx
            active_label = entity_label
            continue

        close(idx - 1)
        start = idx
        active_label = raw_label

    close(len(labels) - 1)
    return spans


def convert_row(
    row: dict[str, Any],
    tokenizer,
    id2label: dict[int, str],
    all_labels: list[str],
    *,
    group_name: str,
    include_layout: bool,
    clamp_bboxes: bool,
    split: str,
    index: int,
) -> dict[str, Any] | None:
    words, bboxes, word_labels = merge_subwords(
        row,
        tokenizer,
        id2label,
        clamp_bboxes=clamp_bboxes,
    )
    if not words:
        return None

    item: dict[str, Any] = {
        "tokenized_text": words,
        "text": " ".join(words),
        "bboxes": bboxes,
        "extraction": [
            {
                "name": group_name,
                "all_labels": all_labels,
                "ner": bio_to_spans(word_labels),
            }
        ],
        "_glinext_extraction_spans_resolved": True,
        "source": {
            "dataset": REPO_ID,
            "split": split,
            "index": index,
        },
    }
    if include_layout:
        item["layout"] = [
            {"word": word, "bbox": bbox}
            for word, bbox in zip(words, bboxes)
        ]
    return item


def iter_parquet_rows(path: Path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required to read the source parquet files.") from exc

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(columns=PARQUET_COLUMNS):
        for row in batch.to_pylist():
            yield row


def split_files(repo_id: str, revision: str | None, local_parquet: list[Path]) -> dict[str, list[Path]]:
    if local_parquet:
        return {"custom": local_parquet}

    files = list_dataset_files(repo_id, revision)
    split_to_files: dict[str, list[str]] = {}
    for filename in files:
        if not filename.startswith("data/") or not filename.endswith(".parquet"):
            continue
        split = Path(filename).name.split("-", 1)[0]
        split_to_files.setdefault(split, []).append(filename)

    if not split_to_files:
        raise ValueError(f"no parquet files found in {repo_id}")

    return {
        split: [download_file(repo_id, filename, revision) for filename in sorted(filenames)]
        for split, filenames in sorted(split_to_files.items())
    }


def output_path(prefix: Path, split: str) -> Path:
    if prefix.suffix:
        return prefix
    return prefix.with_name(f"{prefix.name}_{split}.json")


def convert_split(
    *,
    split: str,
    files: list[Path],
    output: Path,
    tokenizer,
    id2label: dict[int, str],
    all_labels: list[str],
    group_name: str,
    include_layout: bool,
    clamp_bboxes: bool,
    pretty: bool,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    seen = 0

    with open(output, "w", encoding="utf-8") as fout:
        fout.write("[")
        first = True
        for path in files:
            for row in iter_parquet_rows(path):
                item = convert_row(
                    row,
                    tokenizer,
                    id2label,
                    all_labels,
                    group_name=group_name,
                    include_layout=include_layout,
                    clamp_bboxes=clamp_bboxes,
                    split=split,
                    index=seen,
                )
                seen += 1
                if item is None:
                    continue
                if not first:
                    fout.write(",")
                    if pretty:
                        fout.write("\n")
                json.dump(item, fout, ensure_ascii=False, indent=2 if pretty else None)
                first = False
                written += 1
        fout.write("]\n")

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--tokenizer-name", default=TOKENIZER_NAME)
    parser.add_argument(
        "--local-parquet",
        type=Path,
        nargs="*",
        default=[],
        help="Optional local parquet files. If omitted, all HF split shards are downloaded.",
    )
    parser.add_argument("--label-info", type=Path, default=None)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output prefix; writes <prefix>_<split>.json unless a .json path is provided.",
    )
    parser.add_argument("--group-name", default="invoice")
    parser.add_argument("--no-layout-list", action="store_true")
    parser.add_argument("--no-clamp-bboxes", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_info = load_label_info(args.repo_id, args.revision, args.label_info)
    raw_id2label = label_info["id2label"]
    id2label = {int(idx): str(label) for idx, label in raw_id2label.items()}
    all_labels = label_names_without_bio(id2label.values())
    tokenizer = load_tokenizer(args.tokenizer_name)

    total = 0
    for split, files in split_files(args.repo_id, args.revision, args.local_parquet).items():
        out = output_path(args.output_prefix, split)
        count = convert_split(
            split=split,
            files=files,
            output=out,
            tokenizer=tokenizer,
            id2label=id2label,
            all_labels=all_labels,
            group_name=args.group_name,
            include_layout=not args.no_layout_list,
            clamp_bboxes=not args.no_clamp_bboxes,
            pretty=args.pretty,
        )
        total += count
        print(f"Wrote {count} rows to {out}", file=sys.stderr)

    print(f"Wrote {total} rows total with {len(all_labels)} labels", file=sys.stderr)


if __name__ == "__main__":
    main()
