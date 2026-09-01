import json
from pathlib import Path

from scripts.build_multitask_dataset import (
    _iter_json_array,
    build_dataset,
    has_task_supervision,
    iter_records,
)


def test_iter_json_array_streams_across_small_chunks(tmp_path):
    path = tmp_path / "records.json"
    records = [
        {"id": 1, "text": "a value longer than one chunk"},
        {"id": 2, "nested": {"items": [1, 2, 3]}},
    ]
    path.write_text(json.dumps(records), encoding="utf-8")

    assert list(_iter_json_array(path, chunk_size=7)) == records


def test_build_dataset_samples_each_source_and_appends_complete_data(tmp_path):
    sampled_array = tmp_path / "sampled.json"
    sampled_jsonl = tmp_path / "sampled.jsonl"
    complete = tmp_path / "complete.jsonl"
    output = tmp_path / "multitask.json"

    sampled_array.write_text(
        json.dumps(
            [
                {
                    "array": index,
                    "classification": [{"all_labels": ["yes", "no"], "true_labels": []}],
                }
                for index in range(6)
            ]
            + [{"array": "unlabeled", "classification": []}]
        ),
        encoding="utf-8",
    )
    sampled_jsonl.write_text(
        "".join(
            json.dumps(
                {
                    "jsonl": index,
                    "embedding": [["candidate", 1.0]],
                    "text": "root",
                }
            )
            + "\n"
            for index in range(5)
        ),
        encoding="utf-8",
    )
    complete_records = [
        {
            "complete": 1,
            "text": "grounded value",
            "structuring": {"schema": [{"field": "value"}]},
        },
        {"complete": 2, "extraction": [{"ner": [[0, 0, "entity"]]}]},
    ]
    complete.write_text(
        "".join(
            json.dumps(record) + "\n" for record in [*complete_records, {"complete": "unlabeled"}]
        ),
        encoding="utf-8",
    )

    stats = build_dataset(
        [sampled_array, sampled_jsonl],
        [complete],
        output,
        sample_size=3,
        seed=17,
    )
    records = list(iter_records(output))

    assert stats.total_written == 8
    assert [(source.available, source.eligible, source.written) for source in stats.sources] == [
        (7, 6, 3),
        (5, 5, 3),
        (3, 2, 2),
    ]
    assert sum("array" in record for record in records) == 3
    assert sum("jsonl" in record for record in records) == 3
    assert records[-2:] == complete_records

    second_output = Path(tmp_path / "multitask-second.json")
    build_dataset(
        [sampled_array, sampled_jsonl],
        [complete],
        second_output,
        sample_size=3,
        seed=17,
    )
    assert second_output.read_bytes() == output.read_bytes()


def test_has_task_supervision_requires_a_usable_label_space():
    assert not has_task_supervision({"text": "empty"})
    assert not has_task_supervision({"extraction": [{"ner": []}]})
    assert not has_task_supervision({"classification": []})
    assert not has_task_supervision({"structuring": {}})
    assert not has_task_supervision({"text": "document", "structuring": {"schema": []}})
    assert not has_task_supervision(
        {
            "text": "document",
            "structuring": {"schema": [{"field": None}]},
        }
    )
    assert not has_task_supervision(
        {
            "text": "document",
            "structuring": {"schema": [{"field": "missing"}]},
        }
    )
    assert not has_task_supervision({"text": "root", "embedding": []})

    assert has_task_supervision({"classification": [{"all_labels": ["a"], "true_labels": []}]})
    assert has_task_supervision({"extraction": [{"ner": [], "all_labels": ["entity"]}]})
    assert has_task_supervision({"extraction": [{"ner": [[0, 0, "entity"]]}]})
    assert has_task_supervision(
        {
            "text": "document contains a value",
            "structuring": {"schema": [{"field": "value"}]},
        }
    )
    assert has_task_supervision({"text": "root", "embedding": [["candidate", 1.0]]})
