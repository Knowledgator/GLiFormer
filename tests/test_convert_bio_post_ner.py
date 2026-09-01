import json

import pytest

from scripts.convert_bio_post_ner import convert_file, convert_record


def test_convert_record_wraps_legacy_ner_and_preserves_metadata():
    converted, dropped = convert_record(
        {
            "tokenized_text": ["New", "York", "hired", "Ada"],
            "ner": [[3, 3, "person"], [0, 1, "location"]],
            "metadata": {"source": "fixture"},
            "negatives": ["organization"],
        },
        index=0,
        group_name="entities",
        invalid_spans="drop",
    )

    assert dropped == 0
    assert "ner" not in converted
    assert converted["metadata"] == {"source": "fixture"}
    assert converted["negatives"] == ["organization"]
    assert converted["_glinext_extraction_spans_resolved"] is True
    assert converted["extraction"] == [{
        "name": "entities",
        "all_labels": ["location", "person"],
        "ner": [[0, 1, "location"], [3, 3, "person"]],
    }]


def test_convert_record_drops_invalid_spans():
    converted, dropped = convert_record(
        {
            "tokenized_text": ["one", "two"],
            "ner": [[0, 0, "valid"], [2, 2, "out-of-range"], [1, 0, "reversed"]],
        },
        index=4,
        group_name="entities",
        invalid_spans="drop",
    )

    assert dropped == 2
    assert converted["extraction"][0]["ner"] == [[0, 0, "valid"]]


def test_convert_record_can_fail_on_invalid_spans():
    with pytest.raises(ValueError, match=r"record 4 at span index 0"):
        convert_record(
            {"tokenized_text": ["one"], "ner": [[1, 1, "invalid"]]},
            index=4,
            group_name="entities",
            invalid_spans="error",
        )


def test_convert_json_array_to_jsonl(tmp_path):
    input_path = tmp_path / "source.json"
    output_path = tmp_path / "converted.jsonl"
    input_path.write_text(
        json.dumps([
            {"tokenized_text": ["Ada"], "ner": [[0, 0, "person"]]},
            {"tokenized_text": ["Paris"], "ner": [[0, 0, "location"]]},
        ]),
        encoding="utf-8",
    )

    stats = convert_file(input_path, output_path)
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert stats.records_written == 2
    assert stats.spans_written == 2
    assert stats.spans_dropped == 0
    assert rows[0]["extraction"][0]["ner"] == [[0, 0, "person"]]
    assert rows[1]["extraction"][0]["all_labels"] == ["location"]


def test_convert_file_drops_records_without_valid_spans(tmp_path):
    input_path = tmp_path / "source.json"
    output_path = tmp_path / "converted.jsonl"
    input_path.write_text(
        json.dumps([
            {"tokenized_text": ["Ada"], "ner": [[0, 0, "person"]]},
            {"tokenized_text": ["bad"], "ner": [[1, 1, "out-of-range"]]},
            {"tokenized_text": ["empty"], "ner": []},
        ]),
        encoding="utf-8",
    )

    stats = convert_file(input_path, output_path)
    rows = output_path.read_text(encoding="utf-8").splitlines()

    assert stats.records_read == 3
    assert stats.records_written == 1
    assert stats.records_dropped == 2
    assert stats.spans_dropped == 1
    assert len(rows) == 1


def test_convert_file_does_not_replace_output_when_input_is_missing(tmp_path):
    input_path = tmp_path / "missing.json"
    output_path = tmp_path / "converted.jsonl"
    output_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Input dataset not found"):
        convert_file(input_path, output_path)

    assert output_path.read_text(encoding="utf-8") == "existing\n"
