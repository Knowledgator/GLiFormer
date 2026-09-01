import json

from scripts.convert_top_1000_to_glinext import (
    ConversionStats,
    build_structuring,
    convert_file,
    convert_record,
    parse_extracted,
)


def test_parse_extracted_accepts_object_and_json_string():
    extracted = {"people": [{"name": "Alice"}]}

    assert parse_extracted(extracted) is extracted
    assert parse_extracted(json.dumps(extracted)) == extracted


def test_convert_record_emits_only_required_glinext_fields():
    row = {
        "_source": "sft",
        "text": "Alice joined Acme in 2024.",
        "extracted": {
            "people": [
                {
                    "name": "Alice",
                    "employer": {"name": "Acme", "missing": None},
                    "year": 2024,
                    "optional": "null",
                    "hallucinated": "Other Corp",
                }
            ]
        },
    }

    converted = convert_record(row)

    assert set(converted) == {"text", "structuring"}
    assert converted["structuring"] == {
        "people": [
            {
                "name": "Alice",
                "employer.name": "Acme",
                "year": "2024",
            }
        ]
    }


def test_build_structuring_maps_root_list_to_repeated_records():
    stats = ConversionStats()

    result = build_structuring(
        [{"name": "A"}, {"name": "B", "unused": None}],
        "A then B",
        root_schema="item",
        stats=stats,
    )

    assert result == {"item": [{"name": "A"}, {"name": "B"}]}
    assert stats.values_dropped_null == 1


def test_build_structuring_keeps_grounded_members_of_scalar_lists():
    result = build_structuring(
        {"record": {"aliases": ["Alpha", "invented", None]}},
        "Alpha is listed here.",
    )

    assert result == {"record": [{"aliases": ["Alpha"]}]}


def test_top_level_scalars_use_non_colliding_metadata_schema():
    result = build_structuring(
        {"metadata": {"author": "Alice"}, "title": "Report"},
        "Report by Alice",
    )

    assert result == {
        "metadata": [{"author": "Alice"}],
        "metadata_2": [{"title": "Report"}],
    }


def test_convert_file_handles_mixed_source_encodings_atomically(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "output.jsonl"
    rows = [
        {
            "text": "Alice",
            "extracted": {"person": {"name": "Alice"}},
        },
        {
            "text": "Bob",
            "extracted": json.dumps({"person": {"name": "Bob"}}),
        },
        {"text": "No annotation", "extracted": None},
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    stats = convert_file(source, output)
    converted = [json.loads(line) for line in output.read_text().splitlines()]

    assert stats.rows_seen == 3
    assert stats.rows_written == 2
    assert stats.rows_skipped_empty == 1
    assert [row["structuring"]["person"][0]["name"] for row in converted] == [
        "Alice",
        "Bob",
    ]
