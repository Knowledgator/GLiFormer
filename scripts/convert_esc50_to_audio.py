#!/usr/bin/env python3
"""Convert ashraq/esc50 to GLiNExT audio-classification format.

The Hugging Face dataset has one ``train`` split with ESC-50's original
``fold`` column. By default this script uses fold 5 as validation and writes:

    data/esc50_train.json
    data/esc50_validation.json
    data/esc50_audio/*.wav

Each output row is shaped for ``GLiNextAudioProcessor``:

    {
      "audio": "data/esc50_audio/1-100032-A-0.wav",
      "sample_rate": 44100,
      "duration": 5.0,
      "audio_classification": [
        {
          "name": "esc50",
          "all_labels": ["dog", "rooster", ...],
          "true_labels": ["dog"]
        }
      ]
    }
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import wave
from pathlib import Path
from typing import Any


REPO_ID = "ashraq/esc50"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data"
DEFAULT_AUDIO_DIR = REPO_ROOT / "data" / "esc50_audio"
PARQUET_COLUMNS = ["filename", "fold", "target", "category", "esc10", "src_file", "take", "audio"]


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def list_dataset_parquets(repo_id: str, revision: str | None) -> list[str]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required to list ESC-50 parquet files.") from exc

    return sorted(
        filename
        for filename in HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision)
        if filename.startswith("data/") and filename.endswith(".parquet")
    )


def download_file(repo_id: str, filename: str, revision: str | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required to download ESC-50.") from exc

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
        )
    )


def iter_parquet_rows(path: Path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required to read ESC-50 parquet shards.") from exc

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(columns=PARQUET_COLUMNS):
        for row in batch.to_pylist():
            yield row


def audio_info(wav_bytes: bytes) -> tuple[int | None, float | None]:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
            sample_rate = int(handle.getframerate())
            frames = int(handle.getnframes())
            duration = frames / sample_rate if sample_rate else None
    except wave.Error:
        return None, None
    return sample_rate, duration


def write_audio(row: dict[str, Any], audio_dir: Path) -> tuple[Path | None, int | None, float | None]:
    audio = row.get("audio") or {}
    wav_bytes = audio.get("bytes")
    if not wav_bytes:
        return None, None, None

    filename = row.get("filename") or audio.get("path")
    if not filename:
        filename = f"{row.get('fold', 'unknown')}-{row.get('target', 'unknown')}-{row.get('src_file', 'unknown')}.wav"
    filename = Path(str(filename)).name
    output = audio_dir / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists() or output.stat().st_size != len(wav_bytes):
        with open(output, "wb") as fout:
            fout.write(wav_bytes)

    sample_rate, duration = audio_info(wav_bytes)
    return output, sample_rate, duration


def collect_rows(parquet_files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in parquet_files:
        rows.extend(iter_parquet_rows(path))
    return rows


def label_list(rows: list[dict[str, Any]]) -> list[str]:
    pairs = []
    for row in rows:
        if row.get("category") is None:
            continue
        target = row.get("target")
        target_key = int(target) if target is not None else 10**9
        pairs.append((target_key, str(row["category"])))
    return list(dict.fromkeys(label for _, label in sorted(pairs)))


def convert_row(
    row: dict[str, Any],
    *,
    audio_dir: Path,
    all_labels: list[str],
    group_name: str,
    absolute_audio_paths: bool,
) -> dict[str, Any] | None:
    audio_path, sample_rate, duration = write_audio(row, audio_dir)
    if audio_path is None:
        return None

    category = str(row.get("category") or "")
    if not category:
        return None

    item: dict[str, Any] = {
        "audio": audio_path.resolve().as_posix() if absolute_audio_paths else repo_relative(audio_path),
        "audio_classification": [
            {
                "name": group_name,
                "all_labels": all_labels,
                "true_labels": [category],
            }
        ],
        "source": {
            "dataset": REPO_ID,
            "filename": row.get("filename"),
            "fold": int(row["fold"]) if row.get("fold") is not None else None,
            "target": int(row["target"]) if row.get("target") is not None else None,
            "category": category,
            "esc10": bool(row["esc10"]) if row.get("esc10") is not None else None,
            "src_file": int(row["src_file"]) if row.get("src_file") is not None else None,
            "take": row.get("take"),
        },
    }
    if sample_rate is not None:
        item["sample_rate"] = sample_rate
    if duration is not None:
        item["duration"] = duration
        item["audio_duration"] = duration
    return item


def write_json(path: Path, rows: list[dict[str, Any]], *, pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(rows, fout, ensure_ascii=False, indent=2 if pretty else None)
        fout.write("\n")


def convert_dataset(
    *,
    parquet_files: list[Path],
    output_dir: Path,
    audio_dir: Path,
    validation_fold: int | None,
    group_name: str,
    absolute_audio_paths: bool,
    pretty: bool,
) -> dict[str, int]:
    source_rows = collect_rows(parquet_files)
    all_labels = label_list(source_rows)

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    skipped = 0

    for row in source_rows:
        item = convert_row(
            row,
            audio_dir=audio_dir,
            all_labels=all_labels,
            group_name=group_name,
            absolute_audio_paths=absolute_audio_paths,
        )
        if item is None:
            skipped += 1
            continue

        if validation_fold is not None and int(row.get("fold", -1)) == validation_fold:
            validation_rows.append(item)
        else:
            train_rows.append(item)

    if validation_fold is None:
        write_json(output_dir / "esc50_train.json", train_rows, pretty=pretty)
    else:
        write_json(output_dir / "esc50_train.json", train_rows, pretty=pretty)
        write_json(output_dir / "esc50_validation.json", validation_rows, pretty=pretty)

    return {
        "source_rows": len(source_rows),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "labels": len(all_labels),
        "skipped": skipped,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--local-parquet",
        type=Path,
        nargs="*",
        default=[],
        help="Optional local ESC-50 parquet shards. If omitted, shards are downloaded.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument(
        "--validation-fold",
        type=int,
        default=5,
        choices=[1, 2, 3, 4, 5],
        help="Fold held out as validation. Default: 5.",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Write all rows to esc50_train.json and skip esc50_validation.json.",
    )
    parser.add_argument("--group-name", default="esc50")
    parser.add_argument("--absolute-audio-paths", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.local_parquet:
        parquet_files = args.local_parquet
    else:
        parquet_files = [
            download_file(args.repo_id, filename, args.revision)
            for filename in list_dataset_parquets(args.repo_id, args.revision)
        ]
    if not parquet_files:
        raise SystemExit("no ESC-50 parquet files found")

    validation_fold = None if args.no_validation else args.validation_fold
    stats = convert_dataset(
        parquet_files=parquet_files,
        output_dir=args.output_dir,
        audio_dir=args.audio_dir,
        validation_fold=validation_fold,
        group_name=args.group_name,
        absolute_audio_paths=args.absolute_audio_paths,
        pretty=args.pretty,
    )
    log(
        "Wrote {train_rows} train rows, {validation_rows} validation rows, "
        "{labels} labels ({skipped} skipped).".format(**stats)
    )
    log(f"Audio files: {args.audio_dir}")
    log(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
