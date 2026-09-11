from types import SimpleNamespace

import pytest
import torch

from gliformer.processing.pdf import (
    CELL_NEW_LINE,
    GLiFormerPDFProcessor,
    PDFTableProcessor,
    _normalize_bbox,
    _scale_box_to_1000,
)


def _prefield_values():
    words = [["Alpha", "Beta"], ["Gamma"]]
    bboxes = [
        [[0, 0, 100, 100], [100, 0, 200, 100]],
        [[0, 100, 100, 200]],
    ]
    pixels = torch.arange(24, dtype=torch.float).reshape(2, 3, 2, 2)
    return words, bboxes, pixels


def test_prefields_preserve_multiple_pages_without_opening_pdf():
    words, bboxes, pixels = _prefield_values()
    processor = GLiFormerPDFProcessor()

    rows = processor(
        "not-opened.pdf",
        words=words,
        bbox=bboxes,
        pixel_values=pixels,
        pages=[2, 5],
        return_page_ids=True,
    )

    assert [row["page"] for row in rows] == [2, 5]
    assert [row["tokenized_text"] for row in rows] == words
    assert [row["text"] for row in rows] == ["Alpha Beta", "Gamma"]
    assert [row["bboxes"] for row in rows] == bboxes
    assert [row["page_ids"] for row in rows] == [[2, 2], [5]]
    assert torch.equal(rows[0]["pixel_values"], pixels[0])
    assert torch.equal(rows[1]["pixel_values"], pixels[1])


def test_combining_prefield_pages_flattens_layout_and_stacks_images():
    words, bboxes, pixels = _prefield_values()

    rows = GLiFormerPDFProcessor()(
        "not-opened.pdf",
        words=words,
        bbox=bboxes,
        pixel_values=pixels,
        pages=[2, 5],
        return_page_ids=True,
        split_pages=False,
    )

    assert len(rows) == 1
    combined = rows[0]
    assert combined["pages"] == [2, 5]
    assert combined["tokenized_text"] == ["Alpha", "Beta", "Gamma"]
    assert combined["text"] == "Alpha Beta Gamma"
    assert combined["bboxes"] == [*bboxes[0], *bboxes[1]]
    assert combined["page_ids"] == [2, 2, 5]
    assert combined["image_page_ids"] == [2, 5]
    assert torch.equal(combined["pixel_values"], pixels)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"words": None, "bbox": [[0, 0, 1, 1]]}, "words are required"),
        ({"words": ["word"], "bbox": None}, "bbox is required"),
        (
            {"words": ["one", "two"], "bbox": [[0, 0, 1, 1]]},
            "must have the same length",
        ),
    ],
)
def test_prefields_validate_required_and_aligned_values(kwargs, error):
    with pytest.raises(ValueError, match=error):
        GLiFormerPDFProcessor()(
            "not-opened.pdf",
            add_image_token=False,
            **kwargs,
        )


def test_prefields_reject_incomplete_page_entries():
    with pytest.raises(ValueError, match="words has 2 page entries"):
        GLiFormerPDFProcessor()(
            "not-opened.pdf",
            words=[["one"], ["two"]],
            bbox=[
                [[0, 0, 1, 1]],
                [[0, 0, 1, 1]],
                [[0, 0, 1, 1]],
            ],
            pages=[0, 1, 2],
            add_image_token=False,
        )


def test_prefields_reject_table_extraction():
    with pytest.raises(ValueError, match="extract_tables=True requires PDF text extraction"):
        GLiFormerPDFProcessor()(
            "not-opened.pdf",
            words=["word"],
            bbox=[[0, 0, 1, 1]],
            extract_tables=True,
            add_image_token=False,
        )


def test_bbox_normalization_and_scaling_are_bounded():
    assert _normalize_bbox(torch.tensor([1.9, 2.1, 3.8, 4.2])) == [1, 2, 3, 4]
    assert _scale_box_to_1000([-10, 25, 120, 75], width=100, height=50) == [
        0,
        500,
        1000,
        1000,
    ]
    assert _scale_box_to_1000([0, 0, 1, 1], width=0, height=0) == [0, 0, 1000, 1000]

    with pytest.raises(ValueError, match="4 coordinates"):
        _normalize_bbox([1, 2, 3])


def test_markdown_table_normalizes_multiline_and_empty_cells():
    table = [["first\nline", None], ["value", ""]]

    assert PDFTableProcessor.get_markdown_table(table) == (
        f"| first{CELL_NEW_LINE}line |  |\n| value |  |"
    )


def test_table_extraction_preserves_reading_order_and_removes_table_words():
    rows = [
        SimpleNamespace(
            bbox=[0, 10, 100, 20],
            cells=[[0, 10, 50, 20], [50, 10, 100, 20]],
        ),
        SimpleNamespace(
            bbox=[0, 20, 100, 40],
            cells=[[0, 20, 50, 40], [50, 20, 100, 40]],
        ),
    ]
    table = SimpleNamespace(bbox=[0, 10, 100, 40], rows=rows)
    words = [
        {"text": "Footer", "x0": 0, "top": 50, "x1": 20, "bottom": 55},
        {"text": "B", "x0": 60, "top": 12, "x1": 70, "bottom": 18},
        {"text": "Heading", "x0": 0, "top": 0, "x1": 30, "bottom": 5},
        {"text": "D", "x0": 10, "top": 30, "x1": 20, "bottom": 36},
        {"text": "A", "x0": 10, "top": 12, "x1": 20, "bottom": 18},
        {"text": "E", "x0": 60, "top": 22, "x1": 70, "bottom": 28},
        {"text": "C", "x0": 10, "top": 22, "x1": 20, "bottom": 28},
    ]

    tokens, bboxes = PDFTableProcessor(100, 100).extract_page([table], words)

    assert tokens == [
        "Heading",
        "|",
        "A",
        "|",
        "B",
        "|",
        "|",
        "C",
        CELL_NEW_LINE,
        "D",
        "|",
        "E",
        "|",
        "Footer",
    ]
    assert len(bboxes) == len(tokens)
    assert all(0 <= coordinate <= 1000 for bbox in bboxes for coordinate in bbox)


def test_page_words_sort_and_scale_pymupdf_tuples():
    page = SimpleNamespace(
        rect=SimpleNamespace(width=200, height=100),
        get_text=lambda _: [
            (100, 50, 220, 110, "Second", 1, 0, 0),
            (0, 0, 20, 10, "First", 0, 0, 0),
            (20, 0, 30, 10, "   ", 0, 0, 1),
        ],
    )

    words, bboxes = GLiFormerPDFProcessor._page_words(page)

    assert words == ["First", "Second"]
    assert bboxes == [[0, 0, 100, 100], [500, 500, 1000, 1000]]
