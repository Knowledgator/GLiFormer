#!/usr/bin/env python3
"""Convert winvoker/lvis to GLiNExT vision format.

Default behavior converts the LVIS validation split into:

    data/lvis_validation.json

Each output row is shaped for GLiNExT vision processors:

    {
      "image": "data/lvis_images/validation/val2017/000000000139.jpg",
      "height": 426,
      "width": 640,
      "image_classification": [
        {
          "name": "lvis",
          "all_labels": ["person", ...],
          "true_labels": ["person", ...]
        }
      ],
      "object_detection": [
        {
          "name": "lvis",
          "all_labels": ["person", ...],
          "objects": [
            {
              "label": "person",
              "bbox": [x1, y1, x2, y2]
            }
          ]
        }
      ],
      "segmentation": [
        {
          "name": "lvis",
          "all_labels": ["person", ...],
          "objects": [
            {
              "label": "person",
              "bbox": [x1, y1, x2, y2],
              "mask": "data/lvis_masks/validation/000000000139_000000.png"
            }
          ]
        }
      ]
    }

LVIS boxes are COCO-style ``xywh``; GLiNExT expects ``xyxy``. The classification
group uses unique labels from image objects as ``true_labels``. By default, each
task group's ``all_labels`` contains those true labels plus 30 deterministic
random negative labels from the complete LVIS category list. Polygon annotations
are rasterized into small full-image binary masks by default. Use
``--no-rasterize-masks`` to skip mask generation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data"
DEFAULT_DOWNLOAD_DIR = REPO_ROOT / "data" / "lvis_raw"
DEFAULT_IMAGES_DIR = REPO_ROOT / "data" / "lvis_images"
DEFAULT_MASKS_DIR = REPO_ROOT / "data" / "lvis_masks"

SPLITS = {
    "train": {
        "image_folder": "train2017",
        "images_url": "http://images.cocodataset.org/zips/train2017.zip",
        "annotations_url": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",
        "annotation_file": "lvis_v1_train.json",
        "hf_name": "train",
    },
    "validation": {
        "image_folder": "val2017",
        "images_url": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations_url": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip",
        "annotation_file": "lvis_v1_val.json",
        "hf_name": "validation",
    },
    "test": {
        "image_folder": "test2017",
        "images_url": "http://images.cocodataset.org/zips/test2017.zip",
        "annotations_url": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_image_info_test_dev.json.zip",
        "annotation_file": "lvis_v1_image_info_test_dev.json",
        "hf_name": "test",
    },
}


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def download_url(url: str, output: Path) -> None:
    if output.exists() and output.stat().st_size > 0:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading {url} -> {output}")
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(output)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def extract_zip(zip_path: Path, extract_dir: Path, expected_path: Path) -> None:
    if expected_path.exists():
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    log(f"Extracting {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)


def prepare_split_files(
    *,
    split: str,
    download_dir: Path,
    images_dir: Path,
    annotations_json: Path | None,
    image_root: Path | None,
    skip_images: bool,
) -> tuple[Path, Path | None]:
    spec = SPLITS[split]

    if annotations_json is None:
        ann_zip = download_dir / split / Path(spec["annotations_url"]).name
        ann_dir = download_dir / split / "annotations"
        annotations_json = ann_dir / spec["annotation_file"]
        download_url(spec["annotations_url"], ann_zip)
        extract_zip(ann_zip, ann_dir, annotations_json)

    if image_root is not None:
        return annotations_json, image_root

    if skip_images:
        return annotations_json, None

    img_zip = download_dir / split / Path(spec["images_url"]).name
    image_root = images_dir / split / spec["image_folder"]
    download_url(spec["images_url"], img_zip)
    extract_zip(img_zip, images_dir / split, image_root)
    return annotations_json, image_root


def category_map(annotation_data: dict[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    for category in annotation_data.get("categories", []):
        if not isinstance(category, dict):
            continue
        cat_id = category.get("id")
        name = category.get("name")
        if cat_id is None or name is None:
            continue
        out[int(cat_id)] = str(name)
    return out


def xywh_to_xyxy(bbox: Any) -> list[float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    x, y, w, h = [float(v) for v in bbox]
    if w <= 0 or h <= 0:
        return None
    return [x, y, x + w, y + h]


def normalize_polygons(segmentation: Any) -> list[list[float]]:
    if not isinstance(segmentation, list):
        return []
    if segmentation and all(isinstance(v, (int, float)) for v in segmentation):
        return [[float(v) for v in segmentation]]
    polygons = []
    for polygon in segmentation:
        if isinstance(polygon, list) and len(polygon) >= 6:
            try:
                polygons.append([float(v) for v in polygon])
            except (TypeError, ValueError):
                continue
    return polygons


def rasterize_polygons(
    polygons: list[list[float]],
    *,
    width: int,
    height: int,
    mask_size: int,
    output: Path,
) -> str | None:
    if not polygons or width <= 0 or height <= 0:
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.new("L", (mask_size, mask_size), 0)
    draw = ImageDraw.Draw(mask)
    x_scale = mask_size / max(width, 1)
    y_scale = mask_size / max(height, 1)
    for polygon in polygons:
        if len(polygon) < 6:
            continue
        points = [
            (polygon[i] * x_scale, polygon[i + 1] * y_scale)
            for i in range(0, len(polygon) - 1, 2)
        ]
        if len(points) >= 3:
            draw.polygon(points, outline=255, fill=255)
    if not mask.getbbox():
        return None
    mask.save(output)
    return repo_relative(output)


def build_image_index(annotation_data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    images = {}
    for image in annotation_data.get("images", []):
        image_id = image.get("id")
        if image_id is None:
            continue
        file_name = image.get("file_name")
        if not file_name and image.get("coco_url"):
            file_name = str(image["coco_url"]).rsplit("/", 1)[-1]
        images[int(image_id)] = {
            "id": int(image_id),
            "file_name": file_name,
            "height": int(image.get("height") or 0),
            "width": int(image.get("width") or 0),
            "coco_url": image.get("coco_url"),
        }
    return images


def build_annotation_index(annotation_data: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotation_data.get("annotations", []):
        image_id = annotation.get("image_id")
        if image_id is None:
            continue
        grouped[int(image_id)].append(annotation)
    return grouped


def labels_for_image(
    objects: list[dict[str, Any]],
    all_labels: list[str],
    label_scope: str,
    negative_count: int,
    seed: int,
) -> list[str]:
    positive_labels = positive_labels_for_image(objects)
    if label_scope == "all":
        return all_labels
    labels = list(positive_labels)
    if label_scope == "sampled" and negative_count > 0:
        positives = set(positive_labels)
        negatives = [label for label in all_labels if label not in positives]
        rng = random.Random(seed)
        rng.shuffle(negatives)
        labels.extend(negatives[:negative_count])
    return labels


def positive_labels_for_image(objects: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(obj["label"] for obj in objects if obj.get("label")))


def convert_annotations(
    *,
    annotation_data: dict[str, Any],
    image_root: Path | None,
    output: Path,
    masks_dir: Path,
    split: str,
    task: str,
    label_scope: str,
    rasterize_masks: bool,
    mask_size: int,
    max_examples: int | None,
    allow_missing_images: bool,
    absolute_image_paths: bool,
    include_polygons: bool,
    pretty: bool,
    negative_labels: int = 30,
    classification_negative_labels: int | None = None,
) -> dict[str, int]:
    categories = category_map(annotation_data)
    all_labels = [categories[cat_id] for cat_id in sorted(categories)]
    images = build_image_index(annotation_data)
    annotations = build_annotation_index(annotation_data)

    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_missing_images = 0
    written_objects = 0
    written_masks = 0

    with open(output, "w", encoding="utf-8") as fout:
        fout.write("[")
        first = True

        for source_index, image_id in enumerate(sorted(images)):
            if max_examples is not None and written >= max_examples:
                break
            image = images[image_id]
            file_name = image.get("file_name")
            if not file_name:
                continue

            image_path = (image_root / file_name) if image_root is not None else Path(file_name)
            if image_root is not None and not image_path.exists() and not allow_missing_images:
                skipped_missing_images += 1
                continue

            objects = []
            for obj_index, annotation in enumerate(annotations.get(image_id, [])):
                bbox = xywh_to_xyxy(annotation.get("bbox"))
                if bbox is None:
                    continue
                label = categories.get(int(annotation.get("category_id", -1)))
                if label is None:
                    continue

                obj: dict[str, Any] = {
                    "label": label,
                    "bbox": bbox,
                    "lvis_annotation_id": annotation.get("id"),
                    "lvis_category_id": annotation.get("category_id"),
                }

                polygons = normalize_polygons(annotation.get("segmentation"))
                if include_polygons and polygons:
                    obj["segmentation"] = polygons
                if rasterize_masks and task in {"segmentation", "both"} and polygons:
                    mask_path = masks_dir / split / f"{int(image_id):012d}_{obj_index:06d}.png"
                    rel_mask = rasterize_polygons(
                        polygons,
                        width=image["width"],
                        height=image["height"],
                        mask_size=mask_size,
                        output=mask_path,
                    )
                    if rel_mask is not None:
                        obj["mask"] = rel_mask
                        written_masks += 1
                objects.append(obj)

            positive_labels = positive_labels_for_image(objects)
            label_group = labels_for_image(
                objects,
                all_labels,
                label_scope,
                negative_labels,
                seed=int(image_id),
            )
            classification_label_group = label_group
            if classification_negative_labels is not None:
                classification_label_group = labels_for_image(
                    objects,
                    all_labels,
                    "sampled",
                    max(0, classification_negative_labels),
                    seed=int(image_id),
                )
            row: dict[str, Any] = {
                "id": image["id"],
                "image": image_path.resolve().as_posix() if absolute_image_paths else repo_relative(image_path),
                "height": image["height"],
                "width": image["width"],
                "image_classification": [
                    {
                        "name": "lvis",
                        "all_labels": classification_label_group,
                        "true_labels": positive_labels,
                    }
                ],
                "source": {
                    "dataset": "winvoker/lvis",
                    "split": split,
                    "index": source_index,
                    "coco_url": image.get("coco_url"),
                },
            }
            if task in {"object_detection", "both"}:
                row["object_detection"] = [{
                    "name": "lvis",
                    "all_labels": label_group,
                    "objects": objects,
                }]
            if task in {"segmentation", "both"}:
                row["segmentation"] = [{
                    "name": "lvis",
                    "all_labels": label_group,
                    "objects": objects,
                }]

            if not first:
                fout.write(",")
                if pretty:
                    fout.write("\n")
            json.dump(row, fout, ensure_ascii=False, indent=2 if pretty else None)
            first = False
            written += 1
            written_objects += len(objects)

        fout.write("]\n")

    return {
        "rows": written,
        "objects": written_objects,
        "masks": written_masks,
        "labels": len(all_labels),
        "skipped_missing_images": skipped_missing_images,
    }


def default_output(split: str) -> Path:
    return DEFAULT_OUTPUT_DIR / f"lvis_{split}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--split",
        choices=sorted(SPLITS),
        default="validation",
        help="LVIS split to convert. Default: validation.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--masks-dir", type=Path, default=DEFAULT_MASKS_DIR)
    parser.add_argument("--annotations-json", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument(
        "--task",
        choices=["object_detection", "segmentation", "both"],
        default="both",
        help="Task group to add to each row. Default: both.",
    )
    parser.add_argument(
        "--label-scope",
        choices=["image", "sampled", "all"],
        default="sampled",
        help=(
            "Use only image-positive labels, true labels plus sampled negatives, "
            "or all LVIS labels in each prompt. Default: sampled."
        ),
    )
    parser.add_argument(
        "--negative-labels",
        type=int,
        default=30,
        help=(
            "When --label-scope=sampled, add this many deterministic negative "
            "LVIS labels to each task group. Default: 30."
        ),
    )
    parser.add_argument(
        "--classification-negative-labels",
        type=int,
        default=None,
        help=(
            "Optionally sample this many negative labels for image classification "
            "without changing detection or segmentation label scope."
        ),
    )
    parser.add_argument("--mask-size", type=int, default=128)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--absolute-image-paths", action="store_true")
    parser.add_argument("--no-rasterize-masks", action="store_true")
    parser.add_argument("--include-polygons", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or default_output(args.split)
    annotations_json, image_root = prepare_split_files(
        split=args.split,
        download_dir=args.download_dir,
        images_dir=args.images_dir,
        annotations_json=args.annotations_json,
        image_root=args.image_root,
        skip_images=args.skip_images,
    )
    with open(annotations_json, encoding="utf-8") as fin:
        annotation_data = json.load(fin)

    stats = convert_annotations(
        annotation_data=annotation_data,
        image_root=image_root,
        output=output,
        masks_dir=args.masks_dir,
        split=args.split,
        task=args.task,
        label_scope=args.label_scope,
        rasterize_masks=not args.no_rasterize_masks,
        mask_size=args.mask_size,
        max_examples=args.max_examples,
        allow_missing_images=args.allow_missing_images,
        absolute_image_paths=args.absolute_image_paths,
        include_polygons=args.include_polygons,
        pretty=args.pretty,
        negative_labels=max(0, args.negative_labels),
        classification_negative_labels=(
            None
            if args.classification_negative_labels is None
            else max(0, args.classification_negative_labels)
        ),
    )
    log(
        "Wrote {rows} rows, {objects} objects, {masks} masks, {labels} labels to {output}"
        .format(output=output, **stats)
    )
    if stats["skipped_missing_images"]:
        log(f"Skipped {stats['skipped_missing_images']} rows with missing local images.")


if __name__ == "__main__":
    main()
