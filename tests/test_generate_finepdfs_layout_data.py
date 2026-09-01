import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_finepdfs_layout_data.py"
SPEC = importlib.util.spec_from_file_location("generate_finepdfs_layout_data", SCRIPT_PATH)
finepdfs = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = finepdfs
SPEC.loader.exec_module(finepdfs)


def test_extract_json_object_from_fenced_output():
    output = """```json
{"entities": [], "classification": []}
```"""

    assert finepdfs.extract_json_object(output) == {"entities": [], "classification": []}


def test_normalize_annotation_sanitizes_and_filters_spans():
    annotation = {
        "entities": [
            {"span": "42", "label": "Invoice Number"},
            {"span": "Missing Value", "label": "bad"},
            {"span": "Total Due", "label": "Amount Due"},
        ],
        "classification": [
            {
                "name": "Document Type",
                "all_labels": ["Invoice", "Memo"],
                "true_labels": ["Invoice", "Not In Pool"],
            }
        ],
    }

    classification, ner, labels = finepdfs.normalize_annotation(
        annotation,
        tokens=["Invoice", "42", "Total", "Due"],
        visible_token_count=4,
    )

    assert ner == [[1, 1, "invoice_number"], [2, 3, "amount_due"]]
    assert labels == ["amount_due", "invoice_number"]
    assert classification == [
        {
            "name": "document_type",
            "all_labels": ["invoice", "memo"],
            "true_labels": ["invoice"],
        }
    ]


def test_annotation_prompt_does_not_request_offsets():
    prompt = finepdfs.build_annotation_prompt(["Invoice", "42"], max_tokens=10)[1]["content"]

    assert '"span"' in prompt
    assert '"label"' in prompt
    assert "Do not return token indices or character offsets." in prompt
    assert '"start"' not in prompt
    assert '"end"' not in prompt


def test_normalize_annotation_ignores_llm_offsets_and_aligns_span_text():
    annotation = {
        "entities": [
            {"start": 0, "end": 0, "span": "Total Due", "label": "Amount Due"},
        ],
    }

    _, ner, _ = finepdfs.normalize_annotation(
        annotation,
        tokens=["Invoice", "42", "Total", "Due"],
        visible_token_count=4,
    )

    assert ner == [[2, 3, "amount_due"]]


def test_parse_pdf_layout_combines_pages_and_saves_images(monkeypatch, tmp_path):
    class DummyProcessor:
        def __init__(self, image_dpi):
            assert image_dpi == 100

        def __call__(self, *args, **kwargs):
            assert kwargs["pages"] == [0, 2]
            assert kwargs["return_page_ids"] is True
            return [
                {
                    "page": 0,
                    "tokenized_text": ["Invoice", "42"],
                    "bboxes": [[10, 20, 30, 40], [35, 20, 55, 40]],
                    "pixel_values": torch.ones(3, 4, 4),
                },
                {
                    "page": 2,
                    "tokenized_text": ["Total"],
                    "bboxes": [[10, 50, 30, 70]],
                    "pixel_values": torch.zeros(3, 4, 4),
                },
            ]

    monkeypatch.setattr(finepdfs, "repo_relative", lambda path: Path(path).as_posix())
    monkeypatch.setitem(
        __import__("sys").modules,
        "glinext.processing.pdf",
        type("Module", (), {"GLiNextPDFProcessor": DummyProcessor}),
    )

    item = finepdfs.parse_pdf_layout(
        tmp_path / "doc.pdf",
        pages=[0, 2],
        include_images=True,
        image_prefix="doc",
        images_dir=tmp_path / "images",
        password=None,
        extract_tables=False,
        image_dpi=100,
    )

    assert item["tokenized_text"] == ["Invoice", "42", "Total"]
    assert item["bboxes"] == [[10, 20, 30, 40], [35, 20, 55, 40], [10, 50, 30, 70]]
    assert item["page_ids"] == [0, 0, 2]
    assert item["image_page_ids"] == [0, 2]
    assert len(item["image"]) == 2
    assert Path(item["image"][0]).exists()


def test_parse_pdf_layout_rejects_token_bbox_mismatch(monkeypatch, tmp_path):
    class DummyProcessor:
        def __init__(self, image_dpi):
            pass

        def __call__(self, *args, **kwargs):
            return [{"page": 0, "tokenized_text": ["A"], "bboxes": []}]

    monkeypatch.setitem(
        __import__("sys").modules,
        "glinext.processing.pdf",
        type("Module", (), {"GLiNextPDFProcessor": DummyProcessor}),
    )

    with pytest.raises(ValueError, match="token/bbox mismatch"):
        finepdfs.parse_pdf_layout(
            tmp_path / "doc.pdf",
            pages=[0],
            include_images=False,
            image_prefix="doc",
            images_dir=tmp_path / "images",
            password=None,
            extract_tables=False,
            image_dpi=100,
        )


def _prepare_row_args(tmp_path, *, store_pdfs=True):
    return SimpleNamespace(
        url_column="url",
        id_column="id",
        pdfs_dir=tmp_path / "pdfs",
        overwrite_pdfs=False,
        download_timeout=45.0,
        user_agent="test",
        store_pdfs=store_pdfs,
        password=None,
        all_pages_probability=0.0,
        max_pages_per_document=10,
        image_probability=0.0,
        images_dir=tmp_path / "images",
        extract_tables=False,
        image_dpi=100,
        repo_id="test/repo",
    )


def test_prepare_row_can_skip_storing_pdf(monkeypatch, tmp_path):
    downloaded_paths = []
    parsed_paths = []

    def fake_download_pdf(url, destination, **kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"%PDF test")
        downloaded_paths.append(destination)
        return destination

    def fake_count_pdf_pages(pdf_path, **kwargs):
        assert pdf_path.exists()
        return 1

    def fake_parse_pdf_layout(pdf_path, **kwargs):
        assert pdf_path.exists()
        parsed_paths.append(pdf_path)
        return {"tokenized_text": ["A"], "bboxes": [[0, 0, 1, 1]]}

    monkeypatch.setattr(finepdfs, "download_pdf", fake_download_pdf)
    monkeypatch.setattr(finepdfs, "count_pdf_pages", fake_count_pdf_pages)
    monkeypatch.setattr(finepdfs, "parse_pdf_layout", fake_parse_pdf_layout)

    args = _prepare_row_args(tmp_path, store_pdfs=False)
    item = finepdfs.prepare_row(
        {"id": "doc", "url": "https://example.test/doc.pdf"},
        args,
        finepdfs.random.Random(13),
    )

    assert item["_source"]["url"] == "https://example.test/doc.pdf"
    assert "pdf" not in item["_source"]
    assert downloaded_paths[0].parent != args.pdfs_dir
    assert parsed_paths[0] == downloaded_paths[0]
    assert not downloaded_paths[0].exists()


def test_prepare_row_stores_pdf_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(finepdfs, "repo_relative", lambda path: Path(path).as_posix())

    def fake_download_pdf(url, destination, **kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"%PDF test")
        return destination

    monkeypatch.setattr(finepdfs, "download_pdf", fake_download_pdf)
    monkeypatch.setattr(finepdfs, "count_pdf_pages", lambda *args, **kwargs: 1)
    monkeypatch.setattr(
        finepdfs,
        "parse_pdf_layout",
        lambda *args, **kwargs: {"tokenized_text": ["A"], "bboxes": [[0, 0, 1, 1]]},
    )

    args = _prepare_row_args(tmp_path)
    item = finepdfs.prepare_row(
        {"id": "doc", "url": "https://example.test/doc.pdf"},
        args,
        finepdfs.random.Random(13),
    )

    pdf_path = Path(item["_source"]["pdf"])
    assert pdf_path.parent == args.pdfs_dir
    assert pdf_path.exists()
