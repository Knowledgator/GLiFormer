"""Tests for structuring processor."""

import pytest
import torch

from glinext.tasks.structuring.processor import StructuringProcessor
from glinext.model import GLiNExTModel
from tests.conftest import make_config, make_batch_classes_mapping, FakeWordsSplitter


@pytest.fixture
def struct_proc():
    config = make_config(structuring_config={})
    return StructuringProcessor(config, words_splitter=FakeWordsSplitter())


class TestGetClassesMapping:
    def test_basic(self, struct_proc, structuring_item):
        mapping = struct_proc.get_classes_mapping([structuring_item])
        assert len(mapping) == 1
        assert len(mapping[0].items) == 1
        field_map = mapping[0].items[0].field_class_to_id
        assert "name" in field_map.class_to_id
        assert "location" in field_map.class_to_id
        assert mapping[0].items[0].name == "person"

    def test_multiple_schemas(self, struct_proc):
        item = {
            "text": "test",
            "structuring": {
                "person": [{"name": "John"}],
                "address": [{"city": "NYC", "zip": "10001"}],
            },
        }
        mapping = struct_proc.get_classes_mapping([item])
        assert len(mapping[0].items) == 2

    def test_empty(self, struct_proc):
        mapping = struct_proc.get_classes_mapping([{"text": "hello"}])
        assert len(mapping) == 1
        assert len(mapping[0].items) == 0

    def test_multiple_instances_merge_fields(self, struct_proc):
        item = {
            "text": "test",
            "structuring": {
                "person": [
                    {"name": "John"},
                    {"name": "Jane", "age": "30"},
                ],
            },
        }
        mapping = struct_proc.get_classes_mapping([item])
        field_map = mapping[0].items[0].field_class_to_id
        assert "name" in field_map.class_to_id
        assert "age" in field_map.class_to_id


class TestContributePrompt:
    def test_basic(self, struct_proc, structuring_item):
        mapping = struct_proc.get_classes_mapping([structuring_item])
        classes_mapping = make_batch_classes_mapping(structuring_mappings=mapping)
        prompt = struct_proc.contribute_prompt(classes_mapping, batch_idx=0)
        assert "[P]" in prompt
        assert "[SEP]" in prompt
        assert any("[CHILD]" in tok for tok in prompt)
        assert "person" in prompt

    def test_with_labels_encoder(self, struct_proc, structuring_item):
        mapping = struct_proc.get_classes_mapping([structuring_item])
        classes_mapping = make_batch_classes_mapping(structuring_mappings=mapping)
        prompt = struct_proc.contribute_prompt(classes_mapping, batch_idx=0, use_labels_encoder=True)
        assert not any("[CHILD]" in tok for tok in prompt)


class TestResolveSpans:
    def test_raw_values(self, struct_proc):
        item = {
            "text": "John lives in New York",
            "structuring": {
                "person": [{"name": "John", "location": "New York"}]
            },
        }
        struct_proc.resolve_spans(item)
        inst = item["structuring"]["person"][0]
        assert isinstance(inst["name"], dict)
        assert inst["name"]["start"] == 0
        assert inst["name"]["end"] == 0
        assert inst["location"]["start"] == 3
        assert inst["location"]["end"] == 4

    def test_dict_without_indices(self, struct_proc):
        item = {
            "text": "John lives in New York",
            "structuring": {
                "person": [{"name": {"text": "John"}}]
            },
        }
        struct_proc.resolve_spans(item)
        assert item["structuring"]["person"][0]["name"]["start"] == 0

    def test_list_values(self, struct_proc):
        item = {
            "text": "A B C",
            "structuring": {
                "schema": [{"items": ["A", "C"]}]
            },
        }
        struct_proc.resolve_spans(item)
        items = item["structuring"]["schema"][0]["items"]
        assert isinstance(items, list)
        assert all(isinstance(v, dict) for v in items)
        assert items[0]["start"] == 0
        assert items[1]["start"] == 2


class TestCreateLabels:
    def test_basic(self, struct_proc, structuring_item):
        mapping = struct_proc.get_classes_mapping([structuring_item])
        classes_mapping = make_batch_classes_mapping(structuring_mappings=mapping)
        result = struct_proc.create_labels([structuring_item], classes_mapping, max_seq_len=10)

        assert result is not None
        assert "structuring_labels" in result
        assert "structuring_mask" in result
        assert "structuring_batch_idx" in result
        assert "structuring_count" in result

        labels = result["structuring_labels"]
        # Shape: (total_groups, max_instances, max_seq_len, max_fields, 3)
        assert labels.dim() == 5
        assert labels.shape[0] == 1
        assert labels.shape[4] == 3  # start/end/inside

        assert result["structuring_mask"][0] == True
        assert result["structuring_count"][0] == 1

        # Check "name" field ("John" at token 0)
        field_map = mapping[0].items[0].field_class_to_id
        name_id = field_map.class_to_id["name"]
        assert labels[0, 0, 0, name_id, 0] == 1.0  # start
        assert labels[0, 0, 0, name_id, 1] == 1.0  # end
        assert labels[0, 0, 0, name_id, 2] == 1.0  # inside

    def test_multiple_instances(self, struct_proc):
        item = {
            "text": "A B C",
            "structuring": {
                "schema": [
                    {"f": {"text": "A", "start": 0, "end": 0}},
                    {"f": {"text": "C", "start": 2, "end": 2}},
                ],
            },
        }
        mapping = struct_proc.get_classes_mapping([item])
        classes_mapping = make_batch_classes_mapping(structuring_mappings=mapping)
        result = struct_proc.create_labels([item], classes_mapping, max_seq_len=5)
        assert result["structuring_count"][0] == 2
        assert result["structuring_labels"].shape[1] == 2  # 2 instances

    def test_empty(self, struct_proc):
        classes_mapping = make_batch_classes_mapping()
        result = struct_proc.create_labels([{"text": "hello"}], classes_mapping, max_seq_len=5)
        assert result is None

    def test_list_values_create_multiple_positive_spans(self, struct_proc):
        item = {
            "text": "A B C",
            "structuring": {
                "schema": [
                    {"items": [
                        {"text": "A", "start": 0, "end": 0},
                        {"text": "C", "start": 2, "end": 2},
                    ]},
                ],
            },
        }
        mapping = struct_proc.get_classes_mapping([item])
        classes_mapping = make_batch_classes_mapping(structuring_mappings=mapping)
        result = struct_proc.create_labels([item], classes_mapping, max_seq_len=5)

        field_id = mapping[0].items[0].field_class_to_id.class_to_id["items"]
        labels = result["structuring_labels"]
        assert labels[0, 0, 0, field_id, 0] == 1.0
        assert labels[0, 0, 2, field_id, 0] == 1.0
        assert labels[0, 0, 0, field_id, 1] == 1.0
        assert labels[0, 0, 2, field_id, 1] == 1.0


class TestCreateSpanLabels:
    def test_disabled_by_default(self, struct_proc, structuring_item):
        mapping = struct_proc.get_classes_mapping([structuring_item])
        classes_mapping = make_batch_classes_mapping(structuring_mappings=mapping)
        result = struct_proc.create_span_labels([structuring_item], classes_mapping, max_seq_len=10)
        assert result is None

    def test_enabled(self):
        config = make_config(structuring_config={"represent_spans": True, "neg_spans_ratio": 0.0})
        proc = StructuringProcessor(config, words_splitter=FakeWordsSplitter())
        item = {
            "text": "A B C",
            "structuring": {
                "schema": [
                    {"f": {"text": "A", "start": 0, "end": 0}},
                ]
            },
        }
        mapping = proc.get_classes_mapping([item])
        classes_mapping = make_batch_classes_mapping(structuring_mappings=mapping)
        result = proc.create_span_labels([item], classes_mapping, max_seq_len=5)
        assert result is not None
        assert "structuring_span_idx" in result
        assert "structuring_span_labels" in result
        assert "structuring_span_mask" in result

    def test_list_values_create_multiple_positive_span_labels(self):
        config = make_config(structuring_config={"represent_spans": True, "neg_spans_ratio": 0.0})
        proc = StructuringProcessor(config, words_splitter=FakeWordsSplitter())
        item = {
            "text": "A B C",
            "structuring": {
                "schema": [
                    {"items": [
                        {"text": "A", "start": 0, "end": 0},
                        {"text": "C", "start": 2, "end": 2},
                    ]},
                ]
            },
        }
        mapping = proc.get_classes_mapping([item])
        classes_mapping = make_batch_classes_mapping(structuring_mappings=mapping)
        result = proc.create_span_labels([item], classes_mapping, max_seq_len=5)

        field_id = mapping[0].items[0].field_class_to_id.class_to_id["items"]
        assert result is not None
        assert result["structuring_span_mask"][0, 0] == True
        assert result["structuring_span_mask"][0, 1] == True
        assert torch.equal(result["structuring_span_idx"][0, :2], torch.tensor([[0, 0], [2, 2]]))
        assert result["structuring_span_labels"][0, 0, 0, field_id] == 1.0
        assert result["structuring_span_labels"][0, 1, 0, field_id] == 1.0


class TestModelStructuringCountBridge:
    def test_uses_tail_count_predictions_for_structuring_groups_regression(self):
        model = object.__new__(GLiNExTModel)
        model.config = make_config(count_config={"mode": "regression"})

        flat_inputs_map = {
            "count": type("Flat", (), {"batch_origin": torch.tensor([0, 0, 0, 1])})(),
            "structuring": type("Flat", (), {"batch_origin": torch.tensor([0, 1])})(),
        }
        count_logits = torch.tensor([[1.0], [1.0], [2.6], [0.4]])

        result = model._predict_structuring_counts_from_count_head(count_logits, flat_inputs_map)
        assert torch.equal(result, torch.tensor([3, 0]))

    def test_uses_tail_count_predictions_for_structuring_groups_classification(self):
        model = object.__new__(GLiNExTModel)
        model.config = make_config(count_config={"mode": "classification", "max_count": 5})

        flat_inputs_map = {
            "count": type("Flat", (), {"batch_origin": torch.tensor([0, 0, 0])})(),
            "structuring": type("Flat", (), {"batch_origin": torch.tensor([0])})(),
        }
        count_logits = torch.tensor([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
        ])

        result = model._predict_structuring_counts_from_count_head(count_logits, flat_inputs_map)
        assert torch.equal(result, torch.tensor([2]))
