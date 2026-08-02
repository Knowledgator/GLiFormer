#!/usr/bin/env python3
"""Convert Ihor/GLiNER-FUNSD into GLiNExT layout training format.

The source dataset stores GLiNER-style rows with word tokens, LayoutLM-style
bounding boxes, token-index NER spans, and image paths. This script rewrites
those rows into the GLiNExT layout schema:

    {
      "tokenized_text": [...],
      "text": "...",
      "bboxes": [[x0, y0, x1, y1], ...],
      "layout": [{"word": "...", "bbox": [...]}, ...],
      "image": "data/gliner_funsd_images/0000971160.png",
      "extraction": [{"name": "funsd", "all_labels": [...], "ner": [...]}],
      "_glinext_extraction_spans_resolved": true
    }

The resolved flag is important: the FUNSD spans are token indices, while the
GLiNExT resolver treats bare integer spans as character offsets unless this
flag is already set.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ID = "Ihor/GLiNER-FUNSD"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "layout_train.json"
DEFAULT_IMAGES_DIR = REPO_ROOT / "data" / "gliner_funsd_images"


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def normalize_bbox(raw_bbox: Any, *, clamp: bool) -> list[int]:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raise ValueError(f"expected bbox with 4 coordinates, got {raw_bbox!r}")
    bbox = [int(round(float(value))) for value in raw_bbox]
    if clamp:
        bbox = [max(0, min(1000, value)) for value in bbox]
    return bbox


def normalize_ner(raw_ner: Any, *, num_tokens: int) -> list[list[Any]]:
    spans: list[list[Any]] = []
    for span in raw_ner or []:
        if isinstance(span, dict):
            start = span.get("start", span.get("token_start"))
            end = span.get("end", span.get("token_end"))
            label = span.get("label", span.get("type", span.get("entity_type")))
        elif isinstance(span, (list, tuple)) and len(span) >= 3:
            start, end, label = span[0], span[1], span[-1]
        else:
            continue

        try:
            start_idx = int(start)
            end_idx = int(end)
        except (TypeError, ValueError):
            continue
        if label is None or start_idx < 0 or end_idx < start_idx or end_idx >= num_tokens:
            continue
        spans.append([start_idx, end_idx, str(label)])

    return sorted(spans, key=lambda item: (item[0], item[1], item[2]))


def source_image_repo_path(raw_path: Any) -> str | None:
    if raw_path is None:
        return None
    path = str(raw_path).replace("\\", "/")
    if not path:
        return None
    if path.startswith("images/"):
        return path
    marker = "/images/"
    if marker in path:
        return "images/" + path.rsplit(marker, 1)[1]
    return "images/" + Path(path).name


def download_file(repo_id: str, filename: str, revision: str | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required to download the source dataset. "
            "Install the project dependencies or pass --input-json."
        ) from exc

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
        )
    )


def local_image_path(
    *,
    raw_path: Any,
    repo_id: str,
    revision: str | None,
    images_dir: Path,
    download_images: bool,
    absolute_image_paths: bool,
) -> str | None:
    repo_path = source_image_repo_path(raw_path)
    if repo_path is None:
        return None

    dst = images_dir / Path(repo_path).name
    if download_images:
        src = download_file(repo_id, repo_path, revision)
        images_dir.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or src.stat().st_size != dst.stat().st_size:
            shutil.copy2(src, dst)

    if absolute_image_paths:
        return dst.resolve().as_posix()
    return repo_relative(dst)


def convert_record(
    record: dict[str, Any],
    *,
    all_labels: list[str],
    group_name: str,
    include_layout: bool,
    clamp_bboxes: bool,
    image: str | None,
) -> dict[str, Any]:
    words = record.get("tokenized_text") or record.get("words") or record.get("tokens")
    bboxes = record.get("bboxes") or record.get("word_bboxes") or record.get("bbox")
    if not isinstance(words, list) or not isinstance(bboxes, list):
        raise ValueError("record is missing tokenized_text/words and bboxes")
    if len(words) != len(bboxes):
        raise ValueError(f"words/bboxes length mismatch: {len(words)} != {len(bboxes)}")

    tokens = [str(word) for word in words]
    norm_bboxes = [normalize_bbox(bbox, clamp=clamp_bboxes) for bbox in bboxes]
    ner = normalize_ner(record.get("ner"), num_tokens=len(tokens))

    item: dict[str, Any] = {
        "tokenized_text": tokens,
        "text": " ".join(tokens),
        "bboxes": norm_bboxes,
        "extraction": [
            {
                "name": group_name,
                "all_labels": all_labels,
                "ner": ner,
            }
        ],
        "_glinext_extraction_spans_resolved": True,
    }
    if include_layout:
        item["layout"] = [
            {"word": word, "bbox": bbox}
            for word, bbox in zip(tokens, norm_bboxes)
        ]
    if image is not None:
        item["image"] = image
        item["image_path"] = image
    return item


def collect_labels(records: list[dict[str, Any]]) -> list[str]:
    labels = {
        span[-1]
        for record in records
        for span in normalize_ner(record.get("ner"), num_tokens=len(record.get("tokenized_text") or []))
    }
    return sorted(labels)


def convert_dataset(
    records: list[dict[str, Any]],
    *,
    repo_id: str,
    revision: str | None,
    images_dir: Path,
    download_images: bool,
    absolute_image_paths: bool,
    include_layout: bool,
    clamp_bboxes: bool,
    group_name: str,
) -> list[dict[str, Any]]:
    all_labels = collect_labels(records)
    converted: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        try:
            image = local_image_path(
                raw_path=record.get("image_path") or record.get("image"),
                repo_id=repo_id,
                revision=revision,
                images_dir=images_dir,
                download_images=download_images,
                absolute_image_paths=absolute_image_paths,
            )
            converted.append(
                convert_record(
                    record,
                    all_labels=all_labels,
                    group_name=group_name,
                    include_layout=include_layout,
                    clamp_bboxes=clamp_bboxes,
                    image=image,
                )
            )
        except Exception as exc:
            raise ValueError(f"failed to convert record {idx}") from exc
    return converted


def load_source_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    source_path = args.input_json
    if source_path is None:
        source_path = download_file(args.repo_id, "data.json", args.revision)
    with open(source_path, encoding="utf-8") as fin:
        records = json.load(fin)
    if not isinstance(records, list):
        raise ValueError(f"expected a JSON array in {source_path}")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Optional local GLiNER-FUNSD data.json. If omitted, it is downloaded.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--group-name", default="funsd")
    parser.add_argument("--no-download-images", action="store_true")
    parser.add_argument("--absolute-image-paths", action="store_true")
    parser.add_argument("--no-layout-list", action="store_true")
    parser.add_argument("--no-clamp-bboxes", action="store_true")
    parser.add_argument("--pretty", action="store_true", help="Write indented JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_source_records(args)
    converted = convert_dataset(
        records,
        repo_id=args.repo_id,
        revision=args.revision,
        images_dir=args.images_dir,
        download_images=not args.no_download_images,
        absolute_image_paths=args.absolute_image_paths,
        include_layout=not args.no_layout_list,
        clamp_bboxes=not args.no_clamp_bboxes,
        group_name=args.group_name,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fout:
        json.dump(
            converted,
            fout,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
        )
        fout.write("\n")

    labels = sorted({
        span[-1]
        for item in converted
        for group in item.get("extraction", [])
        for span in group.get("ner", [])
    })
    print(
        f"Wrote {len(converted)} rows, {len(labels)} labels to {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
