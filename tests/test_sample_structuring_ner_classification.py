import json
import random

from scripts.sample_structuring_ner_classification import (
    build_dataset,
    convert_classification_group,
    convert_ner_group,
)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_ner_conversion_grounds_repeated_values_at_distinct_offsets():
    text = "Ada is active. Bob is active."
    converted = convert_ner_group(
        text,
        "person",
        [
            {"name": "Ada", "status": "active"},
            {"name": "Bob", "status": "active"},
        ],
        ("name", "status", "unrelated"),
        max_candidates=3,
        rng=random.Random(4),
    )

    entities = converted["extraction"][0]["ner"]
    assert [entity["text"] for entity in entities] == [
        "Ada",
        "active",
        "Bob",
        "active",
    ]
    status_spans = [
        (entity["start"], entity["end"])
        for entity in entities
        if entity["label"] == "status"
    ]
    assert len(status_spans) == len(set(status_spans)) == 2


def test_classification_conversion_caps_candidates_and_keeps_positives():
    converted = convert_classification_group(
        "Ada is active.",
        "person",
        [{"name": "Ada", "status": "active"}],
        ("age", "name", "status", "title"),
        max_candidates=3,
        rng=random.Random(8),
    )

    group = converted["classification"][0]
    assert set(group["true_labels"]) == {"name", "status"}
    assert set(group["true_labels"]).issubset(group["all_labels"])
    assert len(group["all_labels"]) == 3


def test_build_dataset_samples_task_pure_rows_deterministically(tmp_path):
    input_path = tmp_path / "structuring.jsonl"
    output_path = tmp_path / "sampled.jsonl"
    second_output_path = tmp_path / "sampled-second.jsonl"
    rows = [
        {
            "text": f"Person {index} is active in City {index}.",
            "structuring": {
                "person": [
                    {
                        "name": f"Person {index}",
                        "status": "active",
                        "address": {"city": f"City {index}"},
                    }
                ]
            },
        }
        for index in range(10)
    ]
    _write_jsonl(input_path, rows)

    stats = build_dataset(
        input_path,
        output_path,
        seed=19,
        max_candidates=4,
    )
    build_dataset(
        input_path,
        second_output_path,
        seed=19,
        max_candidates=4,
    )
    sampled = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert stats.records_read == stats.records_written == 10
    assert stats.ner_records > 0
    assert stats.classification_records > 0
    assert stats.ner_records + stats.classification_records == 10
    assert output_path.read_bytes() == second_output_path.read_bytes()
    assert all(
        ("extraction" in row) != ("classification" in row)
        and "structuring" not in row
        for row in sampled
    )
    assert all(
        len(next(iter(row.get("extraction", row.get("classification"))))["all_labels"])
        <= 4
        for row in sampled
    )
