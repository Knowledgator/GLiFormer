"""PDF preprocessing helpers for GLiFormer layout inference."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

PageSelection = list[int] | tuple[int, ...] | None
EMPTY_TABLE_CELL = ""
CELL_NEW_LINE = "<br>"


def _is_bbox(value: Any) -> bool:
    return (
        isinstance(value, list | tuple)
        and len(value) == 4
        and all(isinstance(v, int | float) for v in value)
    )


def _normalize_bbox(bbox: Any) -> list[int]:
    values = torch.as_tensor(bbox, dtype=torch.long).view(-1)
    if values.numel() != 4:
        raise ValueError(f"bbox entries must contain 4 coordinates, got {bbox!r}")
    return [int(v) for v in values.tolist()]


def _scale_box_to_1000(box: Sequence[float], width: float, height: float) -> list[int]:
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    x0, y0, x1, y1 = [float(v) for v in box]
    return [
        int(max(0, min(1000, round(1000 * x0 / width)))),
        int(max(0, min(1000, round(1000 * y0 / height)))),
        int(max(0, min(1000, round(1000 * x1 / width)))),
        int(max(0, min(1000, round(1000 * y1 / height)))),
    ]


def _bbox_contains_word(bbox: Sequence[float], word: dict[str, Any], tolerance: float = 1.0) -> bool:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    cx = (float(word["x0"]) + float(word["x1"])) / 2
    cy = (float(word["top"]) + float(word["bottom"])) / 2
    return x0 - tolerance <= cx <= x1 + tolerance and y0 - tolerance <= cy <= y1 + tolerance


class PDFTableProcessor:
    """Convert pdfplumber table objects into markdown-like tokens and boxes."""

    def __init__(self, page_width: float, page_height: float, y_tolerance: float = 3.0):
        self.page_width = float(page_width)
        self.page_height = float(page_height)
        self.y_tolerance = float(y_tolerance)

    @classmethod
    def get_markdown_table(cls, table: list[list[str | None]]) -> str:
        lines = []
        for row in table:
            cells = [
                cell.replace("\n", CELL_NEW_LINE) if cell else EMPTY_TABLE_CELL
                for cell in row
            ]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    @staticmethod
    def _sort_key_from_bbox(bbox: Sequence[float]) -> tuple[float, float]:
        return float(bbox[1]), float(bbox[0])

    @staticmethod
    def _sort_key_from_word(word: dict[str, Any]) -> tuple[float, float]:
        return float(word.get("top", 0)), float(word.get("x0", 0))

    @staticmethod
    def _extract_words_in_bbox(
        bbox: Sequence[float],
        words: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        remaining = []
        inside = []
        for word in words:
            if _bbox_contains_word(bbox, word):
                inside.append(word)
            else:
                remaining.append(word)
        return remaining, sorted(inside, key=PDFTableProcessor._sort_key_from_word)

    def _scale_bbox(self, bbox: Sequence[float]) -> list[int]:
        return _scale_box_to_1000(bbox, self.page_width, self.page_height)

    def _scale_word_bbox(self, word: dict[str, Any]) -> list[int]:
        return self._scale_bbox([word["x0"], word["top"], word["x1"], word["bottom"]])

    def _pipe_bbox(self, x: float, y0: float, y1: float) -> list[int]:
        width = max(self.page_width / 1000, 1.0)
        return self._scale_bbox([x, y0, min(x + width, self.page_width), y1])

    def _line_groups(self, words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        lines: list[list[dict[str, Any]]] = []
        for word in sorted(words, key=self._sort_key_from_word):
            if not lines:
                lines.append([word])
                continue
            current_top = float(lines[-1][0].get("top", 0))
            if abs(float(word.get("top", 0)) - current_top) <= self.y_tolerance:
                lines[-1].append(word)
            else:
                lines.append([word])
        return [sorted(line, key=lambda item: float(item.get("x0", 0))) for line in lines]

    def _append_cell_words(
        self,
        tokens: list[str],
        bboxes: list[list[int]],
        cell_bbox: Sequence[float],
        cell_words: list[dict[str, Any]],
    ) -> None:
        for line_idx, line in enumerate(self._line_groups(cell_words)):
            if line_idx > 0:
                tokens.append(CELL_NEW_LINE)
                bboxes.append(self._scale_bbox(cell_bbox))
            for word in line:
                text = str(word.get("text", "")).strip()
                if not text:
                    continue
                tokens.append(text.replace("\n", CELL_NEW_LINE))
                bboxes.append(self._scale_word_bbox(word))

    def extract_table(
        self,
        table: Any,
        words: list[dict[str, Any]],
    ) -> tuple[list[str], list[list[int]], list[dict[str, Any]]]:
        remaining_words, table_words = self._extract_words_in_bbox(table.bbox, words)
        tokens: list[str] = []
        bboxes: list[list[int]] = []

        for row in table.rows:
            cells = list(getattr(row, "cells", []) or [])
            cells_with_bbox = [cell for cell in cells if cell is not None]
            if not cells_with_bbox:
                continue
            row_bbox = getattr(row, "bbox", None) or [
                min(cell[0] for cell in cells_with_bbox),
                min(cell[1] for cell in cells_with_bbox),
                max(cell[2] for cell in cells_with_bbox),
                max(cell[3] for cell in cells_with_bbox),
            ]
            row_words = [word for word in table_words if _bbox_contains_word(row_bbox, word)]

            first_x = float(cells_with_bbox[0][0])
            tokens.append("|")
            bboxes.append(self._pipe_bbox(first_x, row_bbox[1], row_bbox[3]))

            for cell in cells:
                if cell is None:
                    continue
                _, cell_words = self._extract_words_in_bbox(cell, row_words)
                self._append_cell_words(tokens, bboxes, cell, cell_words)
                tokens.append("|")
                bboxes.append(self._pipe_bbox(float(cell[2]), cell[1], cell[3]))

        return tokens, bboxes, remaining_words

    def extract_page(
        self,
        tables: list[Any],
        words: list[dict[str, Any]],
    ) -> tuple[list[str], list[list[int]]]:
        blocks = [
            {
                "bbox": table.bbox,
                "table": table,
            }
            for table in tables
        ]
        remaining_words = words
        table_blocks = []
        for block in sorted(blocks, key=lambda item: self._sort_key_from_bbox(item["bbox"])):
            table_tokens, table_bboxes, remaining_words = self.extract_table(
                block["table"],
                remaining_words,
            )
            if table_tokens:
                table_blocks.append({
                    "sort_key": self._sort_key_from_bbox(block["bbox"]),
                    "tokens": table_tokens,
                    "bboxes": table_bboxes,
                })

        word_blocks = [
            {
                "sort_key": self._sort_key_from_word(word),
                "tokens": [str(word["text"])],
                "bboxes": [self._scale_word_bbox(word)],
            }
            for word in remaining_words
            if str(word.get("text", "")).strip()
        ]

        tokens = []
        bboxes = []
        for block in sorted([*word_blocks, *table_blocks], key=lambda item: item["sort_key"]):
            tokens.extend(block["tokens"])
            bboxes.extend(block["bboxes"])
        return tokens, bboxes


class GLiFormerPDFProcessor:
    """Convert PDFs or pre-extracted PDF fields into layout inference rows.

    The output is a list of dictionaries consumable by ``GLiFormerLayoutProcessor``:
    each row contains ``tokenized_text`` plus optional ``bboxes`` and
    ``pixel_values``. Bounding boxes are expected in LayoutLM-style 0-1000 page
    coordinates; boxes extracted with PyMuPDF are normalized into that range.
    """

    def __init__(self, image_dpi: int = 144):
        self.image_dpi = int(image_dpi)

    def __call__(
        self,
        pdf_path: str | Path,
        words: list[str] | None = None,
        bbox: list[list[int]] | None = None,
        pixel_values: Any | None = None,
        pages: PageSelection = None,
        password: str | None = None,
        add_image_token: bool = True,
        return_pixel_values: bool | None = None,
        return_word_bboxes: bool = True,
        return_page_ids: bool = False,
        split_pages: bool = True,
        extract_tables: bool = False,
        table_settings: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        page_numbers = self._normalize_pages(pages)
        if return_pixel_values is None:
            return_pixel_values = bool(add_image_token)

        if words is not None or bbox is not None:
            if extract_tables:
                raise ValueError(
                    "extract_tables=True requires PDF text extraction; omit pre-extracted "
                    "words/bbox so table regions can be detected and de-duplicated."
                )
            rows = self._from_prefields(
                pdf_path=pdf_path,
                words=words,
                bbox=bbox,
                pixel_values=pixel_values,
                page_numbers=page_numbers,
                password=password,
                return_pixel_values=return_pixel_values,
                return_word_bboxes=return_word_bboxes,
                return_page_ids=return_page_ids,
            )
            return rows if split_pages else self._combine_pages(rows)

        rows = self._from_pdf(
            pdf_path=pdf_path,
            page_numbers=page_numbers,
            password=password,
            pixel_values=pixel_values,
            return_pixel_values=return_pixel_values,
            return_word_bboxes=return_word_bboxes,
            return_page_ids=return_page_ids,
            extract_tables=extract_tables,
            table_settings=table_settings,
        )
        return rows if split_pages else self._combine_pages(rows)

    @staticmethod
    def _normalize_pages(pages: PageSelection) -> list[int] | None:
        if pages is None:
            return None
        return [int(page) for page in pages]

    @staticmethod
    def _is_nested_words(words: Any) -> bool:
        return (
            isinstance(words, list | tuple)
            and bool(words)
            and isinstance(words[0], list | tuple)
            and not isinstance(words[0], str)
        )

    @staticmethod
    def _is_nested_bbox(bbox: Any) -> bool:
        return (
            isinstance(bbox, list | tuple)
            and bool(bbox)
            and isinstance(bbox[0], list | tuple)
            and not _is_bbox(bbox[0])
        )

    @classmethod
    def _normalize_words_pages(cls, words: Any | None) -> list[list[str]] | None:
        if words is None:
            return None
        if cls._is_nested_words(words):
            return [[str(token) for token in page_words] for page_words in words]
        return [[str(token) for token in words]]

    @classmethod
    def _normalize_bbox_pages(cls, bbox: Any | None) -> list[list[list[int]]] | None:
        if bbox is None:
            return None
        if cls._is_nested_bbox(bbox):
            return [[_normalize_bbox(box) for box in page_boxes] for page_boxes in bbox]
        return [[_normalize_bbox(box) for box in bbox]]

    @staticmethod
    def _normalize_pixel_pages(pixel_values: Any | None) -> list[Any] | None:
        if pixel_values is None:
            return None
        if isinstance(pixel_values, torch.Tensor):
            if pixel_values.dim() == 4:
                return [pixel_values[i] for i in range(pixel_values.shape[0])]
            return [pixel_values]
        if isinstance(pixel_values, list | tuple):
            return list(pixel_values)
        return [pixel_values]

    @staticmethod
    def _infer_prefield_page_count(
        words_pages: list[list[str]] | None,
        bbox_pages: list[list[list[int]]] | None,
        pixel_pages: list[Any] | None,
        page_numbers: list[int] | None,
    ) -> int:
        counts = [
            len(value)
            for value in (words_pages, bbox_pages, pixel_pages, page_numbers)
            if value is not None
        ]
        return max(counts, default=1)

    @staticmethod
    def _value_for_page(values: list[Any] | None, idx: int, name: str) -> Any:
        if values is None:
            return None
        if len(values) == 1:
            return values[0]
        if idx >= len(values):
            raise ValueError(f"{name} has {len(values)} page entries, but page index {idx} was requested.")
        return values[idx]

    def _from_prefields(
        self,
        pdf_path: str | Path,
        words: Any | None,
        bbox: Any | None,
        pixel_values: Any | None,
        page_numbers: list[int] | None,
        password: str | None,
        return_pixel_values: bool,
        return_word_bboxes: bool,
        return_page_ids: bool,
    ) -> list[dict[str, Any]]:
        words_pages = self._normalize_words_pages(words)
        bbox_pages = self._normalize_bbox_pages(bbox) if return_word_bboxes else None
        pixel_pages = self._normalize_pixel_pages(pixel_values) if return_pixel_values else None
        count = self._infer_prefield_page_count(words_pages, bbox_pages, pixel_pages, page_numbers)
        page_numbers = page_numbers or list(range(count))
        rendered_pixel_pages = None

        rows = []
        for idx in range(count):
            page_words = self._value_for_page(words_pages, idx, "words")
            page_bbox = self._value_for_page(bbox_pages, idx, "bbox")
            if page_words is None:
                raise ValueError("words are required when using pre-extracted PDF fields.")
            if return_word_bboxes and page_bbox is None:
                raise ValueError("bbox is required when return_word_bboxes=True.")
            if return_word_bboxes and len(page_words) != len(page_bbox):
                raise ValueError(
                    f"PDF words and bbox must have the same length for page {idx}, "
                    f"got {len(page_words)} and {len(page_bbox)}."
                )

            row = {
                "pdf_path": str(pdf_path),
                "page": int(self._value_for_page(page_numbers, idx, "pages")),
                "tokenized_text": page_words,
                "text": " ".join(page_words),
            }
            if return_word_bboxes:
                row["bboxes"] = page_bbox
            if return_page_ids:
                row["page_ids"] = [row["page"] for _ in page_words]
            page_pixels = self._value_for_page(pixel_pages, idx, "pixel_values")
            if page_pixels is None and return_pixel_values:
                if rendered_pixel_pages is None:
                    rendered_pixel_pages = self._render_selected_pages(pdf_path, page_numbers, password)
                page_pixels = self._value_for_page(rendered_pixel_pages, idx, "pixel_values")
            if page_pixels is not None:
                row["pixel_values"] = page_pixels
            rows.append(row)
        return rows

    @staticmethod
    def _combine_pages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        combined = {
            "pdf_path": rows[0].get("pdf_path"),
            "pages": [int(row.get("page", idx)) for idx, row in enumerate(rows)],
        }
        tokens: list[str] = []
        bboxes: list[list[int]] = []
        page_ids: list[int] = []
        pixel_values = []
        image_page_ids = []
        has_bboxes = any("bboxes" in row for row in rows)
        has_page_ids = any("page_ids" in row for row in rows)
        for idx, row in enumerate(rows):
            page = int(row.get("page", idx))
            row_tokens = list(row.get("tokenized_text") or [])
            tokens.extend(row_tokens)
            if has_bboxes:
                bboxes.extend(row.get("bboxes") or [[0, 0, 0, 0] for _ in row_tokens])
            if has_page_ids:
                page_ids.extend(row.get("page_ids") or [page for _ in row_tokens])
            if row.get("pixel_values") is not None:
                pixel_values.append(row["pixel_values"])
                image_page_ids.append(page)
        combined["tokenized_text"] = tokens
        combined["text"] = " ".join(tokens)
        if has_bboxes:
            combined["bboxes"] = bboxes
        if has_page_ids:
            combined["page_ids"] = page_ids
        if pixel_values:
            try:
                combined["pixel_values"] = torch.stack([torch.as_tensor(value) for value in pixel_values])
            except Exception:
                combined["pixel_values"] = pixel_values
            combined["image_page_ids"] = image_page_ids
        return [combined]

    @staticmethod
    def _import_fitz():
        try:
            import fitz  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional env.
            raise ImportError(
                "PDF extraction requires PyMuPDF. Install it with `pip install pymupdf` "
                "or pass pre-extracted `words` and `bbox` to parse_pdf."
            ) from exc
        return fitz

    @staticmethod
    def _import_pdfplumber():
        try:
            import pdfplumber  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional env.
            raise ImportError(
                "PDF table extraction requires pdfplumber. Install it with "
                "`pip install pdfplumber` or call parse_pdf with extract_tables=False."
            ) from exc
        return pdfplumber

    @staticmethod
    def _page_words(page: Any) -> tuple[list[str], list[list[int]]]:
        raw_words = page.get_text("words")
        raw_words = sorted(raw_words, key=lambda word: (word[5], word[6], word[7]))
        width = float(page.rect.width)
        height = float(page.rect.height)
        words = []
        bboxes = []
        for word in raw_words:
            text = str(word[4]).strip()
            if not text:
                continue
            words.append(text)
            bboxes.append(_scale_box_to_1000(word[:4], width, height))
        return words, bboxes

    def _render_page(self, fitz: Any, page: Any) -> torch.Tensor:
        zoom = max(self.image_dpi, 1) / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        data = torch.frombuffer(pixmap.samples, dtype=torch.uint8).clone()
        data = data.to(dtype=torch.float).view(pixmap.height, pixmap.width, pixmap.n)
        if data.shape[-1] > 3:
            data = data[..., :3]
        if data.numel() and data.max() > 1:
            data = data / 255.0
        return data.permute(2, 0, 1).contiguous()

    def _render_selected_pages(
        self,
        pdf_path: str | Path,
        page_numbers: list[int],
        password: str | None,
    ) -> list[torch.Tensor]:
        fitz = self._import_fitz()
        doc = fitz.open(str(pdf_path))
        try:
            if doc.needs_pass and not doc.authenticate(password or ""):
                raise ValueError("PDF is password-protected and authentication failed.")
            rendered = []
            for page_number in page_numbers:
                if page_number < 0 or page_number >= len(doc):
                    raise IndexError(f"PDF page index {page_number} is out of range for {len(doc)} pages.")
                rendered.append(self._render_page(fitz, doc[page_number]))
            return rendered
        finally:
            doc.close()

    def _from_pdf(
        self,
        pdf_path: str | Path,
        page_numbers: list[int] | None,
        password: str | None,
        pixel_values: Any | None,
        return_pixel_values: bool,
        return_word_bboxes: bool,
        return_page_ids: bool,
        extract_tables: bool,
        table_settings: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        fitz = self._import_fitz()
        doc = fitz.open(str(pdf_path))
        try:
            if doc.needs_pass and not doc.authenticate(password or ""):
                raise ValueError("PDF is password-protected and authentication failed.")

            selected_pages = page_numbers or list(range(len(doc)))
            pixel_pages = self._normalize_pixel_pages(pixel_values) if return_pixel_values else None
            table_pages = (
                self._extract_pdfplumber_page_layout(
                    pdf_path,
                    selected_pages,
                    password,
                    table_settings=table_settings,
                )
                if extract_tables
                else {}
            )
            rows = []
            for idx, page_number in enumerate(selected_pages):
                if page_number < 0 or page_number >= len(doc):
                    raise IndexError(f"PDF page index {page_number} is out of range for {len(doc)} pages.")
                page = doc[page_number]
                if page_number in table_pages:
                    page_words, page_bbox = table_pages[page_number]
                else:
                    page_words, page_bbox = self._page_words(page)
                row = {
                    "pdf_path": str(pdf_path),
                    "page": int(page_number),
                    "tokenized_text": page_words,
                    "text": " ".join(page_words),
                }
                if return_word_bboxes:
                    row["bboxes"] = page_bbox
                if return_page_ids:
                    row["page_ids"] = [int(page_number) for _ in page_words]
                if return_pixel_values:
                    page_pixels = self._value_for_page(pixel_pages, idx, "pixel_values")
                    row["pixel_values"] = page_pixels if page_pixels is not None else self._render_page(fitz, page)
                rows.append(row)
            return rows
        finally:
            doc.close()

    def _extract_pdfplumber_page_layout(
        self,
        pdf_path: str | Path,
        page_numbers: list[int],
        password: str | None,
        table_settings: dict[str, Any] | None,
    ) -> dict[int, tuple[list[str], list[list[int]]]]:
        pdfplumber = self._import_pdfplumber()
        pages: dict[int, tuple[list[str], list[list[int]]]] = {}
        with pdfplumber.open(str(pdf_path), password=password) as pdf:
            for page_number in page_numbers:
                if page_number < 0 or page_number >= len(pdf.pages):
                    raise IndexError(f"PDF page index {page_number} is out of range for {len(pdf.pages)} pages.")
                page = pdf.pages[page_number]
                words = page.extract_words() or []
                tables = page.find_tables(table_settings=table_settings or {}) or []
                processor = PDFTableProcessor(page.width, page.height)
                pages[page_number] = processor.extract_page(tables, words)
        return pages
