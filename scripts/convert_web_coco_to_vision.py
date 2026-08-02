#!/usr/bin/env python3
"""Convert a YOLO polygon dataset into GLiNExT vision JSON.

The Web-COCO source stores one normalized segmentation polygon per label line::

    class_id x1 y1 x2 y2 ...

GLiNExT object detection expects ``xyxy`` boxes.  This converter derives the
tight absolute-pixel box around every polygon and also creates an image
classification group.  Every category found among an image's objects is added
to that group's ``true_labels``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "data" / "web_coco"


def repo_relative(path: Path) -> str:
    """Return a repository-relative path when possible, otherwise absolute."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_class_names(config_path: Path) -> list[str]:
    with config_path.open(encoding="utf-8") as fin:
        config = yaml.safe_load(fin)
    if not isinstance(config, dict):
        raise ValueError(f"Dataset config must be a mapping: {config_path}")

    raw_names = config.get("names")
    if isinstance(raw_names, dict):
        try:
            names = [str(raw_names[index]) for index in sorted(raw_names, key=int)]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid class-name mapping in {config_path}") from exc
    elif isinstance(raw_names, list):
        names = [str(name) for name in raw_names]
    else:
        raise ValueError(f"Dataset config has no valid 'names' list: {config_path}")

    if not names or any(not name for name in names):
        raise ValueError(f"Dataset class names must be non-empty: {config_path}")
    if len(set(names)) != len(names):
        raise ValueError(f"Dataset class names must be unique: {config_path}")

    declared_count = config.get("nc")
    if declared_count is not None and int(declared_count) != len(names):
        raise ValueError(
            f"Dataset config declares nc={declared_count}, but contains {len(names)} names"
        )
    return names


def image_files(image_dir: Path) -> list[Path]:
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in supported
    )
    stems = [path.stem for path in images]
    if len(stems) != len(set(stems)):
        raise ValueError(f"Image stems are not unique in {image_dir}")
    return images


def polygon_to_object(
    line: str,
    *,
    class_names: list[str],
    width: int,
    height: int,
    label_path: Path,
    line_number: int,
    instance_index: int,
) -> tuple[dict[str, Any], bool]:
    parts = line.split()
    if len(parts) < 7 or len(parts) % 2 == 0:
        raise ValueError(
            f"{label_path}:{line_number}: expected class id and at least 3 xy points"
        )

    try:
        class_id = int(parts[0])
    except ValueError as exc:
        raise ValueError(
            f"{label_path}:{line_number}: invalid class id {parts[0]!r}"
        ) from exc
    if not 0 <= class_id < len(class_names):
        raise ValueError(
            f"{label_path}:{line_number}: class id {class_id} is outside "
            f"[0, {len(class_names) - 1}]"
        )

    try:
        coordinates = [float(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError(
            f"{label_path}:{line_number}: polygon contains a non-numeric coordinate"
        ) from exc
    if any(not math.isfinite(value) for value in coordinates):
        raise ValueError(f"{label_path}:{line_number}: polygon contains a non-finite value")

    was_clamped = any(value < 0.0 or value > 1.0 for value in coordinates)
    coordinates = [min(1.0, max(0.0, value)) for value in coordinates]
    xs = coordinates[0::2]
    ys = coordinates[1::2]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{label_path}:{line_number}: polygon has a zero-area box")

    bbox = [
        round(x1 * width, 6),
        round(y1 * height, 6),
        round(x2 * width, 6),
        round(y2 * height, 6),
    ]
    return (
        {
            "label": class_names[class_id],
            "bbox": bbox,
            "yolo_class_id": class_id,
            "yolo_instance_index": instance_index,
        },
        was_clamped,
    )


def read_objects(
    label_path: Path,
    *,
    class_names: list[str],
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], int]:
    objects: list[dict[str, Any]] = []
    clamped_polygons = 0
    with label_path.open(encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            obj, was_clamped = polygon_to_object(
                line,
                class_names=class_names,
                width=width,
                height=height,
                label_path=label_path,
                line_number=line_number,
                instance_index=len(objects),
            )
            objects.append(obj)
            clamped_polygons += int(was_clamped)
    return objects, clamped_polygons


def build_row(
    *,
    image_path: Path,
    label_path: Path,
    class_names: list[str],
    split: str,
    index: int,
    absolute_image_paths: bool,
) -> tuple[dict[str, Any], int]:
    with Image.open(image_path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Image has invalid dimensions: {image_path}")

    objects, clamped_polygons = read_objects(
        label_path,
        class_names=class_names,
        width=width,
        height=height,
    )
    true_labels = list(dict.fromkeys(obj["label"] for obj in objects))
    group_name = "web_coco"
    image_value = image_path.resolve().as_posix() if absolute_image_paths else repo_relative(image_path)

    return (
        {
            "id": image_path.stem,
            "image": image_value,
            "height": height,
            "width": width,
            "image_classification": [
                {
                    "name": group_name,
                    "all_labels": class_names,
                    "true_labels": true_labels,
                }
            ],
            "object_detection": [
                {
                    "name": group_name,
                    "all_labels": class_names,
                    "objects": objects,
                }
            ],
            "source": {
                "dataset": "web_coco",
                "split": split,
                "index": index,
                "label": repo_relative(label_path),
            },
        },
        clamped_polygons,
    )


def convert_split(
    *,
    dataset_dir: Path,
    split: str,
    output: Path,
    class_names: list[str],
    absolute_image_paths: bool = False,
    pretty: bool = False,
) -> dict[str, int]:
    image_dir = dataset_dir / "images" / split
    label_dir = dataset_dir / "labels" / split
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {label_dir}")

    images = image_files(image_dir)
    image_stems = {path.stem for path in images}
    label_paths = {path.stem: path for path in label_dir.glob("*.txt")}
    missing_labels = sorted(image_stems - set(label_paths))
    orphan_labels = sorted(set(label_paths) - image_stems)
    if missing_labels:
        raise FileNotFoundError(
            f"{len(missing_labels)} images have no label file; first: {missing_labels[0]}"
        )
    if orphan_labels:
        raise FileNotFoundError(
            f"{len(orphan_labels)} label files have no image; first: {orphan_labels[0]}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    rows = objects = positive_rows = clamped_polygons = 0
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as fout:
            temporary_path = Path(fout.name)
            fout.write("[")
            for index, image_path in enumerate(images):
                row, row_clamped = build_row(
                    image_path=image_path,
                    label_path=label_paths[image_path.stem],
                    class_names=class_names,
                    split=split,
                    index=index,
                    absolute_image_paths=absolute_image_paths,
                )
                if rows:
                    fout.write(",\n" if pretty else ",")
                json.dump(row, fout, ensure_ascii=False, indent=2 if pretty else None)
                row_objects = row["object_detection"][0]["objects"]
                objects += len(row_objects)
                positive_rows += int(bool(row_objects))
                clamped_polygons += row_clamped
                rows += 1
            fout.write("]\n")
        temporary_path.chmod(0o644)
        temporary_path.replace(output)
    except BaseException:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise

    return {
        "rows": rows,
        "objects": objects,
        "positive_rows": positive_rows,
        "empty_rows": rows - positive_rows,
        "labels": len(class_names),
        "clamped_polygons": clamped_polygons,
    }


def selected_splits(values: Iterable[str] | None) -> list[str]:
    return list(dict.fromkeys(values or ("train", "val")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "val"),
        help="Split to convert; repeat for both. Default: train and val.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="Output directory. Default: data/.",
    )
    parser.add_argument("--absolute-image-paths", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names = load_class_names(args.dataset_dir / "config.yaml")
    for split in selected_splits(args.split):
        output = args.output_dir / f"web_coco_{split}.json"
        stats = convert_split(
            dataset_dir=args.dataset_dir,
            split=split,
            output=output,
            class_names=class_names,
            absolute_image_paths=args.absolute_image_paths,
            pretty=args.pretty,
        )
        print(
            "Wrote {rows} rows ({positive_rows} positive, {empty_rows} empty), "
            "{objects} objects, and {labels} labels to {output}; "
            "clamped polygons: {clamped_polygons}".format(output=output, **stats),
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()
