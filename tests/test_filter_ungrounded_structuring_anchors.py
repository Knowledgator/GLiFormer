import json
from dataclasses import asdict

from glinext.config import StructuringHeadConfig
from glinext.tasks.structuring.processor import StructuringProcessor
from scripts.filter_ungrounded_structuring_anchors import (
    filter_dataset,
    find_empty_grounding_anchors,
)
from tests.conftest import FakeWordsSplitter, make_config


def make_processor():
    config = make_config(
        structuring_config=asdict(StructuringHeadConfig(
            multi_level=True,
        )),
    )
    return StructuringProcessor(
        config,
        words_splitter=FakeWordsSplitter(),
    )


def test_repeated_value_without_repeated_evidence_creates_empty_anchor():
    row = {
        "text": "Alice",
        "structuring": {
            "person": [
                {"name": "Alice"},
                {"name": "Alice"},
            ],
        },
    }

    issues = find_empty_grounding_anchors(row, make_processor())

    assert len(issues) == 1
    assert issues[0]["fields"] == {"name": ["Alice"]}


def test_record_with_another_grounded_field_is_not_empty():
    row = {
        "text": "Alice Engineer",
        "structuring": {
            "person": [
                {"name": "Alice"},
                {"name": "Alice", "role": "Engineer"},
            ],
        },
    }

    assert find_empty_grounding_anchors(row, make_processor()) == []


def test_filter_dataset_is_atomic_and_reports_removed_rows(tmp_path):
    rows = [
        {
            "text": "Alice",
            "structuring": {"person": [{"name": "Alice"}]},
        },
        {
            "text": "Alice",
            "structuring": {
                "person": [{"name": "Alice"}, {"name": "Alice"}],
            },
        },
    ]
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    report_path = tmp_path / "removed.jsonl"
    input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    stats = filter_dataset(
        input_path,
        output_path,
        report_path,
        make_processor(),
    )

    assert stats["rows_seen"] == 2
    assert stats["rows_written"] == 1
    assert stats["rows_removed"] == 1
    assert json.loads(output_path.read_text()) == rows[0]
    report = json.loads(report_path.read_text())
    assert report["line_number"] == 2
    assert len(report["issues"]) == 1
