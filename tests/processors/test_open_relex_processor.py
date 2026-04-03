"""Tests for open relation extraction processor."""

import pytest
import torch

from glinext.tasks.open_relex.processor import OpenRelexProcessor
from tests.conftest import make_config, make_batch_classes_mapping, FakeWordsSplitter


@pytest.fixture
def open_proc():
    config = make_config(open_relex_config={})
    return OpenRelexProcessor(config, words_splitter=FakeWordsSplitter())


class TestGetClassesMapping:
    def test_basic(self, open_proc, open_relex_item):
        mapping = open_proc.get_classes_mapping([open_relex_item])
        assert len(mapping) == 1
        assert len(mapping[0].items) == 1
        rel_map = mapping[0].items[0].rel_class_to_id
        assert "lives_in" in rel_map.class_to_id
        assert mapping[0].items[0].name == "relations"

    def test_multiple_relations(self, open_proc):
        item = {
            "text": "test",
            "open_relex": [
                {
                    "name": "rels",
                    "relations": [
                        {"relation": "works_at", "head": "A", "tail": "B"},
                        {"relation": "lives_in", "head": "A", "tail": "C"},
                        {"relation": "works_at", "head": "D", "tail": "E"},  # duplicate type
                    ],
                }
            ],
        }
        mapping = open_proc.get_classes_mapping([item])
        rel_map = mapping[0].items[0].rel_class_to_id
        assert len(rel_map.class_to_id) == 2  # unique types only

    def test_empty(self, open_proc):
        mapping = open_proc.get_classes_mapping([{"text": "hello"}])
        assert len(mapping) == 1
        assert len(mapping[0].items) == 0


class TestContributePrompt:
    def test_basic(self, open_proc, open_relex_item):
        mapping = open_proc.get_classes_mapping([open_relex_item])
        classes_mapping = make_batch_classes_mapping(open_relex_mappings=mapping)
        prompt = open_proc.contribute_prompt(classes_mapping, batch_idx=0)
        assert "[P]" in prompt
        assert "[SEP]" in prompt
        assert any("[REL]" in tok for tok in prompt)
        assert "relations" in prompt

    def test_with_labels_encoder(self, open_proc, open_relex_item):
        mapping = open_proc.get_classes_mapping([open_relex_item])
        classes_mapping = make_batch_classes_mapping(open_relex_mappings=mapping)
        prompt = open_proc.contribute_prompt(classes_mapping, batch_idx=0, use_labels_encoder=True)
        assert not any("[REL]" in tok for tok in prompt)


class TestResolveSpans:
    def test_string_mentions(self, open_proc):
        item = {
            "text": "John lives in New York",
            "open_relex": [
                {
                    "relations": [
                        {"relation": "lives_in", "head": "John", "tail": "New York"},
                    ],
                }
            ],
        }
        open_proc.resolve_spans(item)
        rel = item["open_relex"][0]["relations"][0]
        assert isinstance(rel["head"], dict)
        assert rel["head"]["start"] == 0
        assert rel["head"]["end"] == 0
        assert rel["tail"]["start"] == 3
        assert rel["tail"]["end"] == 4

    def test_dict_without_indices(self, open_proc):
        item = {
            "text": "John lives in New York",
            "open_relex": [
                {
                    "relations": [
                        {"relation": "lives_in",
                         "head": {"text": "John"},
                         "tail": {"text": "New York"}},
                    ],
                }
            ],
        }
        open_proc.resolve_spans(item)
        rel = item["open_relex"][0]["relations"][0]
        assert rel["head"]["start"] == 0
        assert rel["tail"]["start"] == 3

    def test_pre_resolved(self, open_proc, open_relex_item):
        open_proc.resolve_spans(open_relex_item)
        rel = open_relex_item["open_relex"][0]["relations"][0]
        # Already has start/end, should not be overwritten
        assert rel["head"]["start"] == 0
        assert rel["tail"]["start"] == 3

    def test_empty(self, open_proc):
        item = {"text": "hello"}
        open_proc.resolve_spans(item)  # should not crash


class TestCreateLabels:
    def test_basic(self, open_proc, open_relex_item):
        mapping = open_proc.get_classes_mapping([open_relex_item])
        classes_mapping = make_batch_classes_mapping(open_relex_mappings=mapping)
        result = open_proc.create_labels([open_relex_item], classes_mapping, max_seq_len=10)

        assert result is not None
        assert "open_rel_labels" in result
        assert "open_rel_mask" in result
        assert "open_rel_batch_idx" in result
        assert "open_rel_count" in result

        labels = result["open_rel_labels"]
        # Shape: (total_groups, max_anchors, max_rel_classes, max_seq_len, 2, 3)
        assert labels.dim() == 6
        assert labels.shape[0] == 1
        assert labels.shape[4] == 2  # head/tail
        assert labels.shape[5] == 3  # start/inside/end

        assert result["open_rel_mask"][0] == True
        assert result["open_rel_count"][0] == 1

        # Check head span (token 0) is marked
        rel_map = mapping[0].items[0].rel_class_to_id
        rel_idx = rel_map.class_to_id["lives_in"]
        assert labels[0, 0, rel_idx, 0, 0, 0] == 1.0  # head start
        assert labels[0, 0, rel_idx, 0, 0, 1] == 1.0  # head end
        # Check tail span (tokens 3-4)
        assert labels[0, 0, rel_idx, 3, 1, 0] == 1.0  # tail start
        assert labels[0, 0, rel_idx, 4, 1, 1] == 1.0  # tail end

    def test_empty(self, open_proc):
        classes_mapping = make_batch_classes_mapping()
        result = open_proc.create_labels([{"text": "hello"}], classes_mapping, max_seq_len=5)
        assert result is None

    def test_multiple_anchors(self, open_proc):
        item = {
            "text": "A B C D E",
            "open_relex": [
                {
                    "name": "rels",
                    "relations": [
                        {"relation": "r1",
                         "head": {"text": "A", "start": 0, "end": 0},
                         "tail": {"text": "B", "start": 1, "end": 1}},
                        {"relation": "r1",
                         "head": {"text": "C", "start": 2, "end": 2},
                         "tail": {"text": "D", "start": 3, "end": 3}},
                    ],
                }
            ],
        }
        mapping = open_proc.get_classes_mapping([item])
        classes_mapping = make_batch_classes_mapping(open_relex_mappings=mapping)
        result = open_proc.create_labels([item], classes_mapping, max_seq_len=10)
        assert result["open_rel_count"][0] == 2
        assert result["open_rel_labels"].shape[1] == 2  # 2 anchors