"""Tests for NER processor."""

import pytest
import torch

from glinext.tasks.ner.processor import NERProcessor
from glinext.mappings import (
    BaseClassMapping, ExtractionItemMapping, ExtractionClassMapping, BatchClassesMapping,
    CatClassMapping, StructuringClassMapping, OpenRelexClassMapping,
)
from tests.conftest import make_config, make_batch_classes_mapping, make_extraction_mapping, FakeWordsSplitter


@pytest.fixture
def ner_proc():
    config = make_config()
    return NERProcessor(config, words_splitter=FakeWordsSplitter())


class TestGetClassesMapping:
    def test_basic(self, ner_proc, ner_item):
        mapping = ner_proc.get_classes_mapping([ner_item])
        assert len(mapping) == 1
        assert len(mapping[0].items) == 1
        ner_map = mapping[0].items[0].ner_class_to_id
        assert "person" in ner_map.class_to_id
        assert "location" in ner_map.class_to_id
        assert ner_map.name == "entities"

    def test_empty_extraction(self, ner_proc):
        mapping = ner_proc.get_classes_mapping([{"text": "hello"}])
        assert len(mapping) == 1
        assert len(mapping[0].items) == 0

    def test_multiple_items(self, ner_proc, ner_item):
        batch = [ner_item, ner_item]
        mapping = ner_proc.get_classes_mapping(batch)
        assert len(mapping) == 2
        assert len(mapping[0].items) == 1
        assert len(mapping[1].items) == 1

    def test_with_relations(self, ner_proc, joint_relex_item):
        mapping = ner_proc.get_classes_mapping([joint_relex_item])
        ext_item = mapping[0].items[0]
        assert ext_item.rel_class_to_id is not None
        # NER processor extracts rel[-1] from relations, which is the tail_id (int)
        # for format [head_id, rel_type, tail_id]
        assert 1 in ext_item.rel_class_to_id.class_to_id

    def test_negative_sampling(self, ner_proc, ner_item):
        negatives = ["org", "event", "date"]
        mapping = ner_proc.get_classes_mapping(
            [ner_item], ner_negatives=negatives, sample_neg=2,
        )
        ner_map = mapping[0].items[0].ner_class_to_id
        # Should have original + up to sample_neg negatives
        assert len(ner_map.class_to_id) >= 2  # at least person, location

    def test_shuffle_labels(self, ner_proc, ner_item):
        # Just check it doesn't crash; order is random
        mapping = ner_proc.get_classes_mapping([ner_item], shuffle_labels=True)
        assert len(mapping[0].items[0].ner_class_to_id.class_to_id) == 2


class TestContributePrompt:
    def test_basic(self, ner_proc, ner_item):
        mapping = ner_proc.get_classes_mapping([ner_item])
        classes_mapping = make_batch_classes_mapping(
            extraction_mappings=mapping,
        )
        prompt = ner_proc.contribute_prompt(classes_mapping, batch_idx=0)
        assert "[P]" in prompt
        assert "[SEP]" in prompt
        assert any("[ENT]" in tok for tok in prompt)
        assert "entities" in prompt

    def test_with_labels_encoder(self, ner_proc, ner_item):
        mapping = ner_proc.get_classes_mapping([ner_item])
        classes_mapping = make_batch_classes_mapping(extraction_mappings=mapping)
        prompt = ner_proc.contribute_prompt(classes_mapping, batch_idx=0, use_labels_encoder=True)
        # Should NOT include entity labels when using labels encoder
        assert not any("[ENT]" in tok for tok in prompt)
        assert "[P]" in prompt


class TestResolveSpans:
    def test_text_based(self, ner_proc, ner_item_text_spans):
        ner_proc.resolve_spans(ner_item_text_spans)
        ner = ner_item_text_spans["extraction"][0]["ner"]
        assert len(ner) == 2
        # Should be resolved to token indices
        assert all(isinstance(e[0], int) for e in ner)
        assert ner[0] == [0, 0, "person"]
        assert ner[1] == [3, 4, "location"]

    def test_pre_resolved(self, ner_proc, ner_item):
        original_ner = [list(e) for e in ner_item["extraction"][0]["ner"]]
        ner_proc.resolve_spans(ner_item)
        # Pre-resolved should stay the same
        assert ner_item["extraction"][0]["ner"] == original_ner

    def test_no_extraction(self, ner_proc):
        item = {"text": "hello"}
        ner_proc.resolve_spans(item)  # should not crash

    def test_sorting(self, ner_proc):
        item = {
            "text": "a b c d e",
            "extraction": [
                {
                    "ner": [[3, 4, "X"], [0, 0, "Y"]],
                }
            ],
        }
        ner_proc.resolve_spans(item)
        ner = item["extraction"][0]["ner"]
        assert ner[0][0] <= ner[1][0]  # sorted by start


class TestCreateLabels:
    def test_basic(self, ner_proc, ner_item):
        ext_mapping = ner_proc.get_classes_mapping([ner_item])
        classes_mapping = make_batch_classes_mapping(extraction_mappings=ext_mapping)
        result = ner_proc.create_labels([ner_item], classes_mapping, max_seq_len=10)

        assert "ner_labels" in result
        assert "ner_batch_idx" in result
        labels = result["ner_labels"]
        # Shape: (total_groups, max_seq_len, max_classes+1, 3)
        assert labels.shape[0] == 1
        assert labels.shape[1] == 10
        assert labels.shape[3] == 3

        # Check parent class (index 0) is marked at entity positions
        assert labels[0, 0, 0, 0] == 1.0  # "John" start at token 0, parent start
        assert labels[0, 0, 0, 1] == 1.0  # "John" end at token 0, parent end
        assert labels[0, 0, 0, 2] == 1.0  # "John" inside

    def test_empty_groups(self, ner_proc):
        classes_mapping = make_batch_classes_mapping()
        result = ner_proc.create_labels([{"text": "hello"}], classes_mapping, max_seq_len=5)
        assert result is None

    def test_multi_batch(self, ner_proc, ner_item):
        batch = [ner_item, ner_item]
        ext_mapping = ner_proc.get_classes_mapping(batch)
        classes_mapping = make_batch_classes_mapping(
            extraction_mappings=ext_mapping, batch_size=2,
        )
        result = ner_proc.create_labels(batch, classes_mapping, max_seq_len=10)
        assert result["ner_labels"].shape[0] == 2
        assert result["ner_batch_idx"].tolist() == [0, 1]

    def test_out_of_range_spans(self, ner_proc):
        item = {
            "text": "short",
            "extraction": [{"ner": [[0, 0, "X"], [100, 100, "Y"]]}],
        }
        ext_mapping = ner_proc.get_classes_mapping([item])
        classes_mapping = make_batch_classes_mapping(extraction_mappings=ext_mapping)
        result = ner_proc.create_labels([item], classes_mapping, max_seq_len=5)
        # The span (100, 100) should be skipped
        assert result is not None


class TestPrepareSpanIdx:
    def test_disabled_by_default(self, ner_proc):
        ner = [[0, 0, "person"]]
        classes_to_id = {"person": 0}
        span_idx, span_label = ner_proc.prepare_span_idx(ner, classes_to_id, num_tokens=5)
        assert span_idx is None
        assert span_label is None

    def test_enabled(self):
        config = make_config(ner_config={"represent_spans": True, "neg_spans_ratio": 0.0})
        proc = NERProcessor(config, words_splitter=FakeWordsSplitter())
        ner = [[0, 0, "person"], [3, 4, "location"]]
        classes_to_id = {"person": 0, "location": 1}
        span_idx, span_label = proc.prepare_span_idx(ner, classes_to_id, num_tokens=5)
        assert span_idx is not None
        assert span_idx.shape == (2, 2)
        assert span_label.tolist() == [0, 1]

    def test_with_negative_spans(self):
        config = make_config(ner_config={"represent_spans": True, "neg_spans_ratio": 1.0})
        proc = NERProcessor(config, words_splitter=FakeWordsSplitter())
        ner = [[0, 0, "person"], [3, 4, "location"]]
        classes_to_id = {"person": 0, "location": 1}
        span_idx, span_label = proc.prepare_span_idx(ner, classes_to_id, num_tokens=10)
        # Should have 2 positives + ~2 negatives
        assert span_idx.shape[0] >= 2
        # Negative spans have label 0
        assert (span_label == 0).sum() >= 0

    def test_empty_ner(self):
        config = make_config(ner_config={"represent_spans": True})
        proc = NERProcessor(config, words_splitter=FakeWordsSplitter())
        span_idx, span_label = proc.prepare_span_idx([], {}, num_tokens=5)
        assert span_idx.shape == (0, 2)