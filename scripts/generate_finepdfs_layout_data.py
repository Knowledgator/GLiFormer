#!/usr/bin/env python3
"""Generate GLiNExT layout rows from codelion/finepdfs-10M PDFs.

The pipeline:
  1. Samples random rows from a Hugging Face dataset.
  2. Asynchronously downloads and parses source PDFs with bounded workers.
  3. Collects prepared layout rows into batches.
  4. Processes batches through an OpenAI-compatible LLM endpoint in parallel.
  5. Flushes completed JSONL batches to the final dataset.

Example:
    python scripts/generate_finepdfs_layout_data.py \\
        --num-samples 100 \\
        --model Qwen/Qwen2.5-14B-Instruct \\
        --base-url http://localhost:8000/v1 \\
        --api-key EMPTY \\
        --output data/finepdfs_layout.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import hashlib
import json
import os
import random
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_REPO_ID = "codelion/finepdfs-10M"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "finepdfs_layout.jsonl"
DEFAULT_PDFS_DIR = REPO_ROOT / "data" / "finepdfs_pdfs"
DEFAULT_IMAGES_DIR = REPO_ROOT / "data" / "finepdfs_images"
DEFAULT_BASE_URL = "http://localhost:8000/v1"


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sanitize_label(value: Any, *, fallback: str = "label") -> str:
    label = re.sub(r"[^0-9A-Za-z]+", "_", str(value or "").strip().lower())
    label = re.sub(r"_+", "_", label).strip("_")
    return label or fallback


def stable_pdf_name(row: dict[str, Any], url: str, id_column: str) -> str:
    raw_id = row.get(id_column) or Path(urllib.parse.urlparse(url).path).stem or url
    prefix = re.sub(r"[^0-9A-Za-z._-]+", "_", str(raw_id)).strip("._-")[:80] or "pdf"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}.pdf"


def download_pdf(
    url: str,
    destination: Path,
    *,
    timeout: float,
    overwrite: bool,
    user_agent: str,
) -> Path:
    if destination.exists() and not overwrite and destination.stat().st_size > 0:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to download {url}: {exc}") from exc

    if not data.startswith(b"%PDF"):
        stripped = data.lstrip()
        if not stripped.startswith(b"%PDF"):
            raise RuntimeError(f"downloaded content is not a PDF: {url}")
    destination.write_bytes(data)
    return destination


def count_pdf_pages(pdf_path: Path, password: str | None = None) -> int:
    from glinext.processing.pdf import GLiNextPDFProcessor

    fitz = GLiNextPDFProcessor._import_fitz()
    doc = fitz.open(str(pdf_path))
    try:
        if doc.needs_pass and not doc.authenticate(password or ""):
            raise ValueError("PDF is password-protected and authentication failed.")
        return len(doc)
    finally:
        doc.close()


def choose_pages(
    *,
    num_pages: int,
    rng: random.Random,
    all_pages_probability: float,
    max_pages_per_document: int,
) -> tuple[list[int], str]:
    if num_pages <= 0:
        return [], "empty"

    use_all = rng.random() < all_pages_probability
    if use_all:
        pages = list(range(num_pages))
        mode = "all"
    else:
        pages = [rng.randrange(num_pages)]
        mode = "single"

    if max_pages_per_document > 0 and len(pages) > max_pages_per_document:
        pages = sorted(rng.sample(pages, max_pages_per_document))
        mode = f"{mode}_capped"
    return pages, mode


def tensor_to_png(tensor: Any, output_path: Path) -> str:
    import torch
    from PIL import Image

    value = torch.as_tensor(tensor).detach().cpu()
    if value.ndim != 3:
        raise ValueError(f"expected page image tensor with shape C,H,W, got {tuple(value.shape)}")
    if value.shape[0] in (1, 3, 4):
        value = value.permute(1, 2, 0)
    value = value.clamp(0, 1)
    if value.shape[-1] == 1:
        array = (value[..., 0].numpy() * 255).astype("uint8")
        image = Image.fromarray(array, mode="L")
    else:
        array = (value[..., :3].numpy() * 255).astype("uint8")
        image = Image.fromarray(array, mode="RGB")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return repo_relative(output_path)


def parse_pdf_layout(
    pdf_path: Path,
    *,
    pages: list[int],
    include_images: bool,
    image_prefix: str,
    images_dir: Path,
    password: str | None,
    extract_tables: bool,
    image_dpi: int,
) -> dict[str, Any] | None:
    from glinext.processing.pdf import GLiNextPDFProcessor

    processor = GLiNextPDFProcessor(image_dpi=image_dpi)
    page_rows = processor(
        pdf_path,
        pages=pages,
        password=password,
        add_image_token=include_images,
        return_pixel_values=include_images,
        return_word_bboxes=True,
        return_page_ids=True,
        split_pages=True,
        extract_tables=extract_tables,
    )

    tokens: list[str] = []
    bboxes: list[list[int]] = []
    page_ids: list[int] = []
    image_paths: list[str] = []
    image_page_ids: list[int] = []

    for row in page_rows:
        row_tokens = [str(token) for token in row.get("tokenized_text") or []]
        row_bboxes = row.get("bboxes") or []
        if len(row_tokens) != len(row_bboxes):
            raise ValueError(
                f"PDF parser produced token/bbox mismatch on page {row.get('page')}: "
                f"{len(row_tokens)} != {len(row_bboxes)}"
            )
        page_id = int(row.get("page", len(image_page_ids)))
        tokens.extend(row_tokens)
        bboxes.extend([[int(v) for v in bbox] for bbox in row_bboxes])
        page_ids.extend([page_id for _ in row_tokens])

        if include_images and row.get("pixel_values") is not None:
            image_path = images_dir / f"{image_prefix}-page-{page_id}.png"
            image_paths.append(tensor_to_png(row["pixel_values"], image_path))
            image_page_ids.append(page_id)

    if not tokens:
        return None

    item: dict[str, Any] = {
        "tokenized_text": tokens,
        "bboxes": bboxes,
    }
    if len(set(page_ids)) > 1:
        item["page_ids"] = page_ids
    if image_paths:
        if len(image_paths) == 1:
            item["image"] = image_paths[0]
        else:
            item["image"] = image_paths
            item["image_page_ids"] = image_page_ids
    return item


def prompt_text(tokens: list[str], max_tokens: int) -> str:
    return " ".join(tokens[:max_tokens])


def build_annotation_prompt(tokens: list[str], *, max_tokens: int) -> list[dict[str, str]]:
    system = "Return one JSON object only. No markdown."
    user = f"""OCR text:
{prompt_text(tokens, max_tokens)}

Extract useful entities and classify the document.

Rules:
- Entity spans must be exact contiguous text copied from the OCR text.
- Do not return token indices or character offsets.
- Entity labels must be concise snake_case.
- Classification true_labels must be included in all_labels.

JSON shape:
{{
  "entities": [
    {{"span": "copied entity text", "label": "entity_type"}}
  ],
  "classification": [
    {{"name": "document_type", "all_labels": ["invoice", "report"], "true_labels": ["report"]}}
  ]
}}"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_openai_compatible_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_completion_tokens: int,
    timeout: float,
) -> str:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_completion_tokens,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = json.loads(response.read().decode("utf-8"))
    try:
        return raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"unexpected chat completion response: {raw!r}") from exc


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    decoder = json.JSONDecoder()
    for idx, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"LLM output did not contain a JSON object: {text[:200]!r}")


def _normalize_span_piece(value: Any) -> str:
    return re.sub(r"(^\W+|\W+$)", "", str(value).lower())


def find_token_span(tokens: list[str], mention: Any) -> tuple[int, int] | None:
    words = str(mention or "").strip().split()
    if not words:
        return None
    max_start = len(tokens) - len(words)
    lowered_tokens = [token.lower() for token in tokens]
    lowered_words = [word.lower() for word in words]
    for start in range(max_start + 1):
        if lowered_tokens[start:start + len(words)] == lowered_words:
            return start, start + len(words) - 1

    normalized_tokens = [_normalize_span_piece(token) for token in tokens]
    normalized_words = [_normalize_span_piece(word) for word in words]
    for start in range(max_start + 1):
        if normalized_tokens[start:start + len(words)] == normalized_words:
            return start, start + len(words) - 1
    return None


def normalize_annotation(
    annotation: dict[str, Any],
    *,
    tokens: list[str],
    visible_token_count: int,
) -> tuple[list[dict[str, Any]], list[list[Any]], list[str]]:
    raw_entities = annotation.get("entities") or annotation.get("ner") or []
    if not isinstance(raw_entities, list):
        raw_entities = []

    spans: list[list[Any]] = []
    seen_spans: set[tuple[int, int, str]] = set()
    for entity in raw_entities:
        if not isinstance(entity, dict):
            continue
        mention = entity.get("span", entity.get("text", entity.get("mention", entity.get("entity"))))
        aligned = find_token_span(tokens[:visible_token_count], mention)
        if aligned is None:
            continue
        start_idx, end_idx = aligned
        label = sanitize_label(
            entity.get("type", entity.get("label", entity.get("entity_type"))),
            fallback="entity",
        )
        key = (start_idx, end_idx, label)
        if key in seen_spans:
            continue
        seen_spans.add(key)
        spans.append([start_idx, end_idx, label])

    all_labels = sorted({span[2] for span in spans})

    raw_classification = annotation.get("classification") or annotation.get("classifications") or []
    if isinstance(raw_classification, dict):
        raw_classification = [raw_classification]
    if not isinstance(raw_classification, list):
        raw_classification = []

    classifications: list[dict[str, Any]] = []
    for idx, group in enumerate(raw_classification):
        if not isinstance(group, dict):
            continue
        name = sanitize_label(group.get("name", f"classification_{idx}"), fallback=f"classification_{idx}")
        all_group_labels = [
            sanitize_label(label, fallback="label")
            for label in group.get("all_labels", group.get("labels", [])) or []
            if label is not None
        ]
        true_labels = [
            sanitize_label(label, fallback="label")
            for label in group.get("true_labels", group.get("labels_true", [])) or []
            if label is not None
        ]
        if not all_group_labels:
            all_group_labels = list(true_labels)
        all_group_labels = list(dict.fromkeys(all_group_labels))
        true_labels = [label for label in dict.fromkeys(true_labels) if label in all_group_labels]
        if all_group_labels:
            classifications.append({
                "name": name,
                "all_labels": all_group_labels,
                "true_labels": true_labels,
            })

    return classifications, sorted(spans, key=lambda item: (item[0], item[1], item[2])), all_labels


def annotate_item(
    item: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_completion_tokens: int,
    max_prompt_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    tokens = item["tokenized_text"]
    messages = build_annotation_prompt(tokens, max_tokens=max_prompt_tokens)
    raw = call_openai_compatible_chat(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        timeout=timeout,
    )
    annotation = extract_json_object(raw)
    visible_token_count = min(len(tokens), max_prompt_tokens)
    classification, ner, all_labels = normalize_annotation(
        annotation,
        tokens=tokens,
        visible_token_count=visible_token_count,
    )
    item["extraction"] = [{
        "name": "entities",
        "ner": ner,
        "all_labels": all_labels,
    }]
    if classification:
        item["classification"] = classification
    item["_glinext_extraction_spans_resolved"] = True
    return item


def sample_dataset_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The `datasets` package is required. Install the data extra or run "
            "`pip install datasets`."
        ) from exc

    candidate_count = max(args.num_samples, args.num_samples * args.sample_attempts_factor)
    dataset = load_dataset(args.repo_id, split=args.split, streaming=args.streaming)
    if args.streaming:
        yield from dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer).take(candidate_count)
        return

    dataset = dataset.shuffle(seed=args.seed)
    limit = min(candidate_count, len(dataset))
    for row in dataset.select(range(limit)):
        yield dict(row)


@dataclass
class ProcessStats:
    written: int = 0
    attempted: int = 0
    skipped: int = 0
    prepared: int = 0
    llm_failed: int = 0


@dataclass
class PreparedResult:
    candidate_index: int
    item: dict[str, Any] | None
    error: str | None = None


@dataclass
class BatchResult:
    batch_index: int
    rows: list[dict[str, Any]]
    failed: int = 0


def prepare_row(row: dict[str, Any], args: argparse.Namespace, rng: random.Random) -> dict[str, Any] | None:
    url = str(row.get(args.url_column) or "").strip()
    if not url:
        raise ValueError(f"row has no URL column {args.url_column!r}")

    pdf_name = stable_pdf_name(row, url, args.id_column)
    if args.store_pdfs:
        pdf_path = download_pdf(
            url,
            args.pdfs_dir / pdf_name,
            timeout=args.download_timeout,
            overwrite=args.overwrite_pdfs,
            user_agent=args.user_agent,
        )
        return prepare_downloaded_pdf(row, args, rng, url, pdf_name, pdf_path, pdf_stored=True)

    with tempfile.TemporaryDirectory(prefix="finepdfs_pdf_") as tmp_dir:
        pdf_path = download_pdf(
            url,
            Path(tmp_dir) / pdf_name,
            timeout=args.download_timeout,
            overwrite=True,
            user_agent=args.user_agent,
        )
        return prepare_downloaded_pdf(row, args, rng, url, pdf_name, pdf_path, pdf_stored=False)


def prepare_downloaded_pdf(
    row: dict[str, Any],
    args: argparse.Namespace,
    rng: random.Random,
    url: str,
    pdf_name: str,
    pdf_path: Path,
    *,
    pdf_stored: bool,
) -> dict[str, Any] | None:
    num_pages = count_pdf_pages(pdf_path, password=args.password)
    pages, page_mode = choose_pages(
        num_pages=num_pages,
        rng=rng,
        all_pages_probability=args.all_pages_probability,
        max_pages_per_document=args.max_pages_per_document,
    )
    if not pages:
        return None

    include_images = rng.random() < args.image_probability
    image_prefix = Path(pdf_name).stem
    item = parse_pdf_layout(
        pdf_path,
        pages=pages,
        include_images=include_images,
        image_prefix=image_prefix,
        images_dir=args.images_dir,
        password=args.password,
        extract_tables=args.extract_tables,
        image_dpi=args.image_dpi,
    )
    if item is None:
        return None

    source = {
        "dataset": args.repo_id,
        "id": row.get(args.id_column),
        "url": url,
        "pages": pages,
        "page_selection": page_mode,
        "images_included": include_images,
    }
    if pdf_stored:
        source["pdf"] = repo_relative(pdf_path)
    item["_source"] = source
    return item


def process_row(row: dict[str, Any], args: argparse.Namespace, rng: random.Random) -> dict[str, Any] | None:
    item = prepare_row(row, args, rng)
    if item is None:
        return None

    annotate_item(
        item,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        timeout=args.llm_timeout,
    )
    return item


async def prepare_row_async(
    candidate_index: int,
    row: dict[str, Any],
    args: argparse.Namespace,
    row_seed: int,
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
) -> PreparedResult:
    async with semaphore:
        try:
            rng = random.Random(row_seed)
            loop = asyncio.get_running_loop()
            item = await loop.run_in_executor(
                executor,
                functools.partial(prepare_row, row, args, rng),
            )
            return PreparedResult(candidate_index=candidate_index, item=item)
        except Exception as exc:
            return PreparedResult(candidate_index=candidate_index, item=None, error=str(exc))


async def annotate_item_async(
    item: dict[str, Any],
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
) -> dict[str, Any]:
    async with semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            functools.partial(
                annotate_item,
                item,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                max_completion_tokens=args.max_completion_tokens,
                max_prompt_tokens=args.max_prompt_tokens,
                timeout=args.llm_timeout,
            ),
        )


async def annotate_batch_async(
    batch_index: int,
    batch: list[dict[str, Any]],
    args: argparse.Namespace,
    llm_semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
) -> BatchResult:
    tasks = [
        asyncio.create_task(annotate_item_async(item, args, llm_semaphore, executor))
        for item in batch
    ]
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    rows: list[dict[str, Any]] = []
    failed = 0
    for result in results:
        if isinstance(result, Exception):
            failed += 1
            _log(f"skip LLM item in batch {batch_index}: {result}")
            continue
        rows.append(result)
    return BatchResult(batch_index=batch_index, rows=rows, failed=failed)


def write_rows(rows: list[dict[str, Any]], output: Path, output_format: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        with open(output, "w", encoding="utf-8") as fout:
            json.dump(rows, fout, ensure_ascii=False, indent=2)
            fout.write("\n")
        return

    with open(output, "w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False))
            fout.write("\n")


def write_jsonl_batch(rows: list[dict[str, Any]], fout: Any) -> None:
    for row in rows:
        fout.write(json.dumps(row, ensure_ascii=False))
        fout.write("\n")
    fout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--sample-attempts-factor", type=int, default=4)
    parser.add_argument("--strict-num-samples", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Prepared layout rows per LLM/write batch.")
    parser.add_argument("--prepare-concurrency", type=int, default=4,
                        help="Concurrent PDF download/layout preparation workers.")
    parser.add_argument("--llm-concurrency", type=int, default=8,
                        help="Maximum concurrent OpenAI-compatible chat requests.")
    parser.add_argument("--batch-concurrency", type=int, default=2,
                        help="Maximum LLM batches processed in parallel.")
    parser.add_argument("--max-queued-batches", type=int, default=4,
                        help="Backpressure limit for prepared batches waiting for LLM.")

    parser.add_argument("--url-column", default="url")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--pdfs-dir", type=Path, default=DEFAULT_PDFS_DIR)
    parser.add_argument("--store-pdfs", dest="store_pdfs", action="store_true", default=True,
                        help="Keep downloaded PDFs in --pdfs-dir. Enabled by default.")
    parser.add_argument("--no-store-pdfs", dest="store_pdfs", action="store_false",
                        help="Download PDFs to temporary files and delete them after parsing.")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-format", choices=("jsonl", "json"), default="jsonl")
    parser.add_argument("--overwrite-pdfs", action="store_true")
    parser.add_argument("--download-timeout", type=float, default=45.0)
    parser.add_argument("--user-agent", default="GLiNExT finepdfs data generator")

    parser.add_argument("--all-pages-probability", type=float, default=0.25)
    parser.add_argument("--max-pages-per-document", type=int, default=10)
    parser.add_argument("--image-probability", type=float, default=0.5)
    parser.add_argument("--image-dpi", type=int, default=144)
    parser.add_argument("--password", default=None)
    parser.add_argument("--extract-tables", action="store_true")

    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-completion-tokens", type=int, default=2048)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be positive")
    for name in ("all_pages_probability", "image_probability"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.sample_attempts_factor <= 0:
        raise SystemExit("--sample-attempts-factor must be positive")
    for name in (
        "batch_size",
        "prepare_concurrency",
        "llm_concurrency",
        "batch_concurrency",
        "max_queued_batches",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    return args


async def generate_dataset_async(args: argparse.Namespace) -> ProcessStats:
    stats = ProcessStats()
    row_iter = iter(sample_dataset_rows(args))
    prepare_semaphore = asyncio.Semaphore(args.prepare_concurrency)
    llm_semaphore = asyncio.Semaphore(args.llm_concurrency)
    prepare_executor = ThreadPoolExecutor(max_workers=args.prepare_concurrency)
    llm_executor = ThreadPoolExecutor(max_workers=args.llm_concurrency)

    prepare_tasks: set[asyncio.Task[PreparedResult]] = set()
    batch_tasks: dict[asyncio.Task[BatchResult], int] = {}
    pending_batches: deque[tuple[int, list[dict[str, Any]]]] = deque()
    prepared_batch: list[dict[str, Any]] = []
    json_rows: list[dict[str, Any]] = []
    exhausted = False
    next_batch_index = 1
    queued_backlog_limit = args.batch_size * args.max_queued_batches

    def queued_prepared_count() -> int:
        return len(prepared_batch) + sum(len(batch) for _, batch in pending_batches)

    def inflight_llm_count() -> int:
        return sum(batch_tasks.values())

    def potential_output_count() -> int:
        return (
            stats.written
            + inflight_llm_count()
            + queued_prepared_count()
            + len(prepare_tasks)
        )

    def enqueue_batch(batch: list[dict[str, Any]]) -> None:
        nonlocal next_batch_index
        if not batch:
            return
        pending_batches.append((next_batch_index, batch))
        next_batch_index += 1

    def queue_ready_batches(*, force_partial: bool = False) -> None:
        while len(prepared_batch) >= args.batch_size:
            enqueue_batch(prepared_batch[:args.batch_size])
            del prepared_batch[:args.batch_size]
        if force_partial and prepared_batch:
            enqueue_batch(list(prepared_batch))
            prepared_batch.clear()

    def schedule_prepare_tasks() -> None:
        nonlocal exhausted
        while (
            not exhausted
            and len(prepare_tasks) < args.prepare_concurrency
            and queued_prepared_count() < queued_backlog_limit
            and potential_output_count() < args.num_samples
        ):
            try:
                row = next(row_iter)
            except StopIteration:
                exhausted = True
                break
            stats.attempted += 1
            candidate_index = stats.attempted
            row_seed = args.seed + candidate_index * 1_000_003
            task = asyncio.create_task(
                prepare_row_async(
                    candidate_index,
                    dict(row),
                    args,
                    row_seed,
                    prepare_semaphore,
                    prepare_executor,
                )
            )
            prepare_tasks.add(task)

    def dispatch_llm_batches() -> None:
        while pending_batches and len(batch_tasks) < args.batch_concurrency:
            remaining_capacity = (
                args.num_samples - stats.written - inflight_llm_count()
            )
            if remaining_capacity <= 0:
                break
            batch_index, batch = pending_batches.popleft()
            llm_batch = batch[:remaining_capacity]
            remainder = batch[remaining_capacity:]
            if remainder:
                pending_batches.appendleft((batch_index, remainder))
            task = asyncio.create_task(
                annotate_batch_async(
                    batch_index,
                    llm_batch,
                    args,
                    llm_semaphore,
                    llm_executor,
                )
            )
            batch_tasks[task] = len(llm_batch)
            _log(f"LLM batch {batch_index} started with {len(llm_batch)} items")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_f = None
    if args.output_format == "jsonl":
        out_f = open(args.output, "w", encoding="utf-8")

    _log(
        "pipeline: "
        f"prepare_concurrency={args.prepare_concurrency}, "
        f"batch_size={args.batch_size}, "
        f"batch_concurrency={args.batch_concurrency}, "
        f"llm_concurrency={args.llm_concurrency}"
    )

    try:
        while (
            stats.written < args.num_samples
            and (
                not exhausted
                or prepare_tasks
                or batch_tasks
                or pending_batches
                or prepared_batch
            )
        ):
            schedule_prepare_tasks()
            queue_ready_batches(
                force_partial=(
                    not prepare_tasks
                    and prepared_batch
                    and (exhausted or potential_output_count() >= args.num_samples)
                )
            )
            dispatch_llm_batches()

            wait_tasks: set[asyncio.Task[Any]] = set(prepare_tasks)
            wait_tasks.update(batch_tasks)
            if not wait_tasks:
                if exhausted:
                    break
                schedule_prepare_tasks()
                continue

            done, _ = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task in prepare_tasks:
                    prepare_tasks.remove(task)
                    result = task.result()
                    if result.error is not None:
                        stats.skipped += 1
                        _log(f"skip candidate {result.candidate_index}: {result.error}")
                        continue
                    if result.item is None:
                        stats.skipped += 1
                        continue
                    prepared_batch.append(result.item)
                    stats.prepared += 1
                    queue_ready_batches()
                    continue

                batch_tasks.pop(task)
                result = task.result()
                stats.llm_failed += result.failed
                stats.skipped += result.failed
                remaining = args.num_samples - stats.written
                rows_to_write = result.rows[:remaining]
                if args.output_format == "jsonl":
                    assert out_f is not None
                    write_jsonl_batch(rows_to_write, out_f)
                else:
                    json_rows.extend(rows_to_write)
                stats.written += len(rows_to_write)
                _log(
                    f"wrote LLM batch {result.batch_index}: "
                    f"+{len(rows_to_write)} rows "
                    f"({stats.written}/{args.num_samples}); "
                    f"llm_failed={result.failed}"
                )

        if args.output_format == "json":
            write_rows(json_rows, args.output, args.output_format)
    finally:
        unfinished_tasks = list(prepare_tasks) + list(batch_tasks)
        for task in unfinished_tasks:
            task.cancel()
        if unfinished_tasks:
            await asyncio.gather(*unfinished_tasks, return_exceptions=True)
        if out_f is not None:
            out_f.close()
        prepare_executor.shutdown(wait=True, cancel_futures=True)
        llm_executor.shutdown(wait=True, cancel_futures=True)

    return stats


def main() -> None:
    args = parse_args()
    stats = asyncio.run(generate_dataset_async(args))

    if args.strict_num_samples and stats.written < args.num_samples:
        raise SystemExit(
            f"only generated {stats.written}/{args.num_samples} rows after "
            f"{stats.attempted} attempts"
        )

    _log(
        f"done: wrote {stats.written} rows to {args.output}; "
        f"attempted={stats.attempted}, prepared={stats.prepared}, "
        f"skipped={stats.skipped}, llm_failed={stats.llm_failed}"
    )


if __name__ == "__main__":
    main()
