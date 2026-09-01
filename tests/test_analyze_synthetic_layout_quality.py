import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "data_prod" / "analyze_synthetic_layout_quality.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_synthetic_layout_quality", SCRIPT_PATH)
analyzer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _valid_record(image_path):
    tokens = ["Account", "Summary", "USD", "42", "decorative"]
    bboxes = [
        [10, 10, 40, 20],
        [45, 10, 90, 20],
        [103, 11, 130, 21],
        [134, 11, 148, 21],
        [7, 971, 53, 981],
    ]
    return {
        "id": "example-1",
        "tokenized_text": tokens,
        "text": " ".join(tokens),
        "bboxes": bboxes,
        "layout": [{"word": token, "bbox": bboxes[index]} for index, token in enumerate(tokens)],
        "page_ids": [0, 0, 0, 0, 0],
        "blocks": [
            {
                "name": "account_summary",
                "bbox": [10, 10, 190, 40],
                "token_span": [0, 3],
                "anchor_bbox": [10, 10, 90, 30],
                "value_bbox": [100, 10, 190, 30],
                "style": {"requested_font_size": 20, "font_size": 10},
            }
        ],
        "extraction": [
            {
                "name": "layout_blocks",
                "all_labels": ["account_summary"],
                "ner": [[0, 3, "account_summary"]],
            }
        ],
        "classification": [
            {
                "name": "document_category",
                "all_labels": ["Finance"],
                "true_labels": ["Finance"],
            },
            {
                "name": "document_type",
                "all_labels": ["Invoice"],
                "true_labels": ["Invoice"],
            },
        ],
        "_source": {
            "category": "Finance",
            "document_type": "Invoice",
            "page_count": 1,
            "layout_dependency": {
                "score": 0.82,
                "passes_heuristic_gate": True,
                "sequence_nearest_anchor_accuracy": 0.25,
                "geometric_nearest_anchor_accuracy": 1.0,
            },
        },
        "image": str(image_path),
    }


def test_quality_metrics_cover_layout_ner_images_and_rejections(tmp_path):
    image_path = tmp_path / "example.png"
    image_path.write_bytes(b"not decoded by the API-free existence check")
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus_path, [_valid_record(image_path)])
    rejection_path = tmp_path / "corpus.jsonl.rejections.jsonl"
    _write_jsonl(
        rejection_path,
        [
            {
                "category": "Finance",
                "document_type": "Receipt",
                "reason": "InvalidResponseError: render: content for 'total' does not fit its block",
            },
            {
                "category": "Finance",
                "document_type": "Statement",
                "reason": "InvalidResponseError: render: content for 'fees' does not fit its block",
            },
        ],
    )
    schema_path = tmp_path / "schemas.json"
    schema_path.write_text(
        json.dumps({"Finance": {"Invoice": [["document_header", True]]}}),
        encoding="utf-8",
    )

    report = analyzer.analyze_corpus(corpus_path, schema_path=schema_path)

    assert report["structural"]["valid"] is True
    assert report["corpus"]["acceptance_rate"] == 1 / 3
    assert report["ner"]["o_tokens"] == 1
    assert report["ner"]["o_token_ratio"] == 0.2
    assert report["ner"]["span_lengths"]["median"] == 4
    assert report["text_leakage"]["exact_label_leakage_spans"] == 1
    assert report["anchor_value_layout"]["pairs"] == 1
    assert report["anchor_value_layout"]["edge_distance"]["median"] == 10
    assert report["anchor_value_layout"]["relations"] == {"value_right_of_anchor": 1}
    assert report["layout_dependency"]["score"]["median"] == 0.82
    assert report["layout_dependency"]["gate_pass_ratio"] == 1.0
    assert report["layout"]["font_shrunk_blocks"] == 1
    assert report["layout"]["severe_font_shrink_blocks"] == 1
    assert (
        report["layout"]["block_coordinate_roundness"][
            "boxes_all_coordinates_divisible_by_10_ratio"
        ]
        == 1.0
    )
    assert report["images"] == {
        "records_with_images": 1,
        "references": 1,
        "existing": 1,
        "missing": 0,
        "external_unverified": 0,
    }
    assert report["coverage"]["category_coverage_ratio"] == 1.0
    assert report["coverage"]["document_type_coverage_ratio"] == 1.0
    family = "InvalidResponseError: render: content for '<block>' does not fit its block"
    assert report["rejections"]["reason_families"] == {family: 2}


def test_structural_mismatches_make_the_audit_fail(tmp_path):
    corpus_path = tmp_path / "broken.jsonl"
    record = _valid_record(tmp_path / "missing.png")
    record["bboxes"] = record["bboxes"][:-1]
    record["extraction"][0]["ner"] = [[3, 99, "account_summary"]]
    _write_jsonl(corpus_path, [record])

    report = analyzer.analyze_corpus(corpus_path, schema_path=None)

    assert report["structural"]["valid"] is False
    assert report["alignment"]["token_bbox_length_mismatches"] == 1
    assert report["alignment"]["invalid_spans"] == 1
    assert report["images"]["missing"] == 1
    assert analyzer.main([str(corpus_path), "--schemas", str(tmp_path / "none.json")]) == 1


def test_json_cli_output_is_machine_readable(tmp_path, capsys):
    image_path = tmp_path / "example.png"
    image_path.touch()
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus_path, [_valid_record(image_path)])

    exit_code = analyzer.main(
        [str(corpus_path), "--schemas", str(tmp_path / "none.json"), "--json"]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["structural"]["valid"] is True
    assert output["corpus"]["accepted_records"] == 1


def test_multipage_layout_words_must_keep_page_ids(tmp_path):
    image_path = tmp_path / "example.png"
    image_path.touch()
    corpus_path = tmp_path / "corpus.jsonl"
    record = _valid_record(image_path)
    record["_source"]["page_count"] = 2
    record["page_ids"][-1] = 1
    _write_jsonl(corpus_path, [record])

    report = analyzer.analyze_corpus(corpus_path, schema_path=None)

    assert report["structural"]["valid"] is False
    assert report["alignment"]["multipage_layout_records_missing_pages"] == 1

    for index, item in enumerate(record["layout"]):
        item["page"] = record["page_ids"][index]
    _write_jsonl(corpus_path, [record])
    repaired = analyzer.analyze_corpus(corpus_path, schema_path=None)

    assert repaired["structural"]["valid"] is True
    assert repaired["alignment"]["multipage_layout_records_missing_pages"] == 0
