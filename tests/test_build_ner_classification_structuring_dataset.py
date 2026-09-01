import json

import pytest

from scripts.build_ner_classification_structuring_dataset import build_dataset


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_build_dataset_interleaves_and_counts_all_tasks(tmp_path):
    base_path = tmp_path / "base.jsonl"
    structuring_path = tmp_path / "structuring.jsonl"
    output_path = tmp_path / "combined.jsonl"
    _write_jsonl(
        base_path,
        [
            {
                "tokenized_text": ["Ada"],
                "extraction": [
                    {"name": "entities", "ner": [[0, 0, "person"]]},
                ],
            },
            {
                "text": "A premise",
                "classification": [
                    {
                        "name": "logic",
                        "all_labels": ["yes", "no"],
                        "true_labels": ["yes"],
                    },
                ],
            },
        ],
    )
    _write_jsonl(
        structuring_path,
        [
            {
                "text": f"Person {index}",
                "tokenized_text": ["Person", str(index)],
                "structuring": {
                    "person": [{"name": f"Person {index}", "id": str(index)}],
                },
            }
            for index in range(3)
        ],
    )

    stats = build_dataset(
        base_path,
        structuring_path,
        output_path,
        structuring_per_base=2,
    )
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]

    assert stats.total_records == 5
    assert stats.ner_records == 1
    assert stats.classification_records == 1
    assert stats.structuring_records == 3
    assert stats.max_structuring_schemas == 1
    assert stats.max_structuring_instances == 1
    assert stats.max_structuring_fields == 2
    assert [next(key for key in ("extraction", "classification", "structuring") if key in row)
            for row in rows] == [
                "extraction", "structuring", "structuring", "classification", "structuring",
            ]


def test_build_dataset_rejects_non_task_pure_structuring_row(tmp_path):
    base_path = tmp_path / "base.jsonl"
    structuring_path = tmp_path / "structuring.jsonl"
    output_path = tmp_path / "combined.jsonl"
    _write_jsonl(
        base_path,
        [{
            "tokenized_text": ["Ada"],
            "extraction": [{"name": "entities", "ner": [[0, 0, "person"]]}],
        }],
    )
    _write_jsonl(
        structuring_path,
        [{
            "text": "Ada",
            "tokenized_text": ["Ada"],
            "structuring": {"person": [{"name": "Ada"}]},
            "classification": [],
        }],
    )

    with pytest.raises(ValueError, match="not task-pure"):
        build_dataset(base_path, structuring_path, output_path)

    assert not output_path.exists()


def test_build_dataset_rejects_ner_row_without_supervised_spans(tmp_path):
    base_path = tmp_path / "base.jsonl"
    structuring_path = tmp_path / "structuring.jsonl"
    output_path = tmp_path / "combined.jsonl"
    _write_jsonl(
        base_path,
        [{"tokenized_text": ["Ada"], "extraction": [{"name": "entities", "ner": []}]}],
    )
    _write_jsonl(structuring_path, [])

    with pytest.raises(ValueError, match="no supervised spans"):
        build_dataset(base_path, structuring_path, output_path)

    assert not output_path.exists()
