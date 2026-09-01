import json

import numpy as np
import pytest

from scripts.build_multitask_dataset import iter_records
from scripts.filter_embedding_dataset import (
    FilterSettings,
    classify_pair,
    filter_dataset,
)


class FakeTeacher:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def encode(self, sentences, **kwargs):
        assert kwargs["normalize_embeddings"]
        return np.asarray([self.embeddings[sentence] for sentence in sentences], dtype=np.float32)


def test_classify_pair_applies_binary_thresholds_and_preserves_continuous_scores():
    settings = FilterSettings(
        min_positive_similarity=0.4,
        max_negative_similarity=0.2,
    )

    assert classify_pair(1.0, 0.4, settings) == (True, "positive", None)
    assert classify_pair(1.0, 0.39, settings)[0] is False
    assert classify_pair(-1.0, 0.2, settings) == (True, "negative", None)
    assert classify_pair(0.0, 0.21, settings)[0] is False
    assert classify_pair(0.76, -0.5, settings) == (True, "continuous", None)


def test_classify_pair_can_filter_continuous_scores():
    settings = FilterSettings(max_continuous_error=0.1)

    assert classify_pair(0.7, 0.61, settings) == (True, "continuous", None)
    assert classify_pair(0.7, 0.59, settings)[0] is False


def test_filter_dataset_keeps_teacher_consistent_pairs_and_writes_audit(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    rejected_path = tmp_path / "rejected.jsonl"
    report_path = tmp_path / "report.json"
    records = [
        {
            "text": "anchor",
            "embedding": [
                ["same", 1.0],
                ["same", -1.0],
                ["different", -1.0],
                ["different", 1.0],
            ],
        },
        {
            "text": "continuous anchor",
            "embedding": [["continuous candidate", 0.76]],
        },
    ]
    input_path.write_text(json.dumps(records), encoding="utf-8")
    teacher = FakeTeacher(
        {
            "anchor": [1.0, 0.0],
            "same": [1.0, 0.0],
            "different": [0.0, 1.0],
            "continuous anchor": [1.0, 0.0],
            "continuous candidate": [-1.0, 0.0],
        }
    )

    stats = filter_dataset(
        input_path,
        output_path,
        teacher,
        settings=FilterSettings(
            min_positive_similarity=0.4,
            max_negative_similarity=0.2,
            item_batch_size=1,
            encode_batch_size=4,
        ),
        model_name="fake",
        rejected_output=rejected_path,
        report_output=report_path,
        progress_every=0,
    )

    assert list(iter_records(output_path)) == [
        {
            "text": "anchor",
            "embedding": [["same", 1.0], ["different", -1.0]],
        },
        records[1],
    ]
    rejected = [json.loads(line) for line in rejected_path.read_text().splitlines()]
    assert [(row["candidate"], row["label"]) for row in rejected] == [
        ("same", -1.0),
        ("different", 1.0),
    ]
    assert stats.items_read == stats.items_written == 2
    assert stats.pairs_read == 5
    assert stats.pairs_kept == 3
    assert stats.pairs_rejected == 2
    assert stats.continuous.kept == 1
    assert json.loads(report_path.read_text())["pairs"]["rejected"] == 2


def test_filter_dataset_rejects_an_in_place_output(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must be different"):
        filter_dataset(path, path, FakeTeacher({}), progress_every=0)
