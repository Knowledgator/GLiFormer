import json
import random

from scripts.build_ner_classification_dataset import (
    build_dataset,
    convert_classification_record,
)


def test_classification_conversion_repairs_and_deduplicates_candidates():
    converted, repaired, removed = convert_classification_record(
        {
            "text": "A premise",
            "true_labels": ["true", "true", "missing"],
            "all_labels": ["false", "true", "false"],
        },
        index=3,
        group_name="logic",
        max_candidates=100,
        rng=random.Random(7),
    )

    group = converted["classification"][0]
    assert repaired is True
    assert removed == 0
    assert group["true_labels"] == ["true", "missing"]
    assert set(group["all_labels"]) == {"true", "missing", "false"}
    assert len(group["all_labels"]) == len(set(group["all_labels"]))


def test_classification_cap_keeps_every_positive():
    converted, _, removed = convert_classification_record(
        {
            "text": "A premise",
            "true_labels": ["positive-1", "positive-2"],
            "all_labels": [
                "positive-1", "negative-1", "negative-2", "negative-3",
            ],
        },
        index=0,
        group_name="logic",
        max_candidates=3,
        rng=random.Random(2),
    )

    candidates = converted["classification"][0]["all_labels"]
    assert len(candidates) == 3
    assert {"positive-1", "positive-2"}.issubset(candidates)
    assert removed == 2


def test_classification_conversion_drops_rows_without_any_labels():
    converted, repaired, removed = convert_classification_record(
        {"text": "A premise", "true_labels": [], "all_labels": []},
        index=0,
        group_name="logic",
        max_candidates=100,
        rng=random.Random(1),
    )

    assert converted is None
    assert repaired is False
    assert removed == 0


def test_build_dataset_combines_and_shuffles_rows(tmp_path, monkeypatch):
    ner_input = tmp_path / "ner.jsonl"
    classification_input = tmp_path / "classification.parquet"
    output = tmp_path / "combined.jsonl"
    ner_row = {
        "tokenized_text": ["Ada"],
        "extraction": [{"name": "entities", "ner": [[0, 0, "person"]]}],
    }
    ner_input.write_text(json.dumps(ner_row) + "\n", encoding="utf-8")
    classification_input.touch()
    monkeypatch.setattr(
        "scripts.build_ner_classification_dataset.iter_parquet_records",
        lambda _: iter([
            {
                "text": "A classification example",
                "true_labels": ["yes"],
                "all_labels": ["yes", "no"],
            }
        ]),
    )

    stats = build_dataset(
        ner_input,
        classification_input,
        output,
        seed=9,
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert stats.ner_records == 1
    assert stats.classification_records_written == 1
    assert len(rows) == 2
    assert sum("extraction" in row for row in rows) == 1
    assert sum("classification" in row for row in rows) == 1


def test_build_dataset_drops_ner_rows_without_supervised_spans(tmp_path, monkeypatch):
    ner_input = tmp_path / "ner.jsonl"
    classification_input = tmp_path / "classification.parquet"
    output = tmp_path / "combined.jsonl"
    rows = [
        {
            "tokenized_text": ["Ada"],
            "extraction": [{"name": "entities", "ner": [[0, 0, "person"]]}],
        },
        {
            "tokenized_text": ["empty"],
            "extraction": [{"name": "entities", "ner": []}],
        },
    ]
    ner_input.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    classification_input.touch()
    monkeypatch.setattr(
        "scripts.build_ner_classification_dataset.iter_parquet_records",
        lambda _: iter(()),
    )

    stats = build_dataset(ner_input, classification_input, output)

    assert stats.ner_records == 1
    assert stats.ner_records_dropped == 1
    assert len(output.read_text().splitlines()) == 1
