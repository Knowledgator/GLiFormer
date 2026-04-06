"""Tests for count processor."""

import pytest
import torch

from glinext.tasks.count.processor import CountProcessor
from glinext.processing.mappings import (
    BaseClassMapping, CatClassMapping, ExtractionItemMapping, ExtractionClassMapping,
    StructuringItemMapping, StructuringClassMapping,
)
from tests.conftest import make_config, make_batch_classes_mapping


@pytest.fixture
def count_proc():
    config = make_config(count_config={})
    return CountProcessor(config)


class TestGetClassesMapping:
    def test_returns_none(self, count_proc):
        assert count_proc.get_classes_mapping([{}]) is None


class TestCreateLabels:
    def test_cat_only(self, count_proc):
        cat_mapping = [CatClassMapping(cat_class_to_id=[
            BaseClassMapping(class_to_id={"a": 0, "b": 1}),
        ])]
        classes_mapping = make_batch_classes_mapping(cat_mappings=cat_mapping)
        batch = [{"text": "test"}]
        result = count_proc.create_labels(batch, classes_mapping)
        assert result is not None
        assert result["count_targets"].shape == (1,)
        assert result["count_targets"][0] == 1.0

    def test_extraction_only(self, count_proc):
        ext_mapping = [ExtractionClassMapping(items=[
            ExtractionItemMapping(ner_class_to_id=BaseClassMapping(class_to_id={"X": 0})),
        ])]
        classes_mapping = make_batch_classes_mapping(extraction_mappings=ext_mapping)
        batch = [{"text": "test"}]
        result = count_proc.create_labels(batch, classes_mapping)
        assert result["count_targets"].shape == (1,)
        assert result["count_targets"][0] == 1.0

    def test_structuring_counts_instances(self, count_proc):
        struct_mapping = [StructuringClassMapping(items=[
            StructuringItemMapping(
                field_class_to_id=BaseClassMapping(class_to_id={"name": 0}),
                name="person",
            ),
        ])]
        classes_mapping = make_batch_classes_mapping(structuring_mappings=struct_mapping)
        batch = [{
            "text": "test",
            "structuring": {
                "person": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
            },
        }]
        result = count_proc.create_labels(batch, classes_mapping)
        assert result["count_targets"][0] == 3.0

    def test_mixed(self, count_proc):
        cat_mapping = [CatClassMapping(cat_class_to_id=[
            BaseClassMapping(class_to_id={"a": 0}),
        ])]
        ext_mapping = [ExtractionClassMapping(items=[
            ExtractionItemMapping(ner_class_to_id=BaseClassMapping(class_to_id={"X": 0})),
        ])]
        struct_mapping = [StructuringClassMapping(items=[
            StructuringItemMapping(
                field_class_to_id=BaseClassMapping(class_to_id={"f": 0}),
                name="s",
            ),
        ])]
        classes_mapping = make_batch_classes_mapping(
            cat_mappings=cat_mapping,
            extraction_mappings=ext_mapping,
            structuring_mappings=struct_mapping,
        )
        batch = [{"text": "test", "structuring": {"s": [{"f": "v1"}, {"f": "v2"}]}}]
        result = count_proc.create_labels(batch, classes_mapping)
        # 1 cat + 1 extraction + 1 structuring = 3 total
        assert result["count_targets"].shape == (3,)
        assert result["count_targets"][0] == 1.0  # cat
        assert result["count_targets"][1] == 1.0  # extraction
        assert result["count_targets"][2] == 2.0  # structuring (2 instances)

    def test_empty(self, count_proc):
        classes_mapping = make_batch_classes_mapping()
        result = count_proc.create_labels([{"text": "test"}], classes_mapping)
        assert result is None

    def test_gold_count_is_clone(self, count_proc):
        cat_mapping = [CatClassMapping(cat_class_to_id=[
            BaseClassMapping(class_to_id={"a": 0}),
        ])]
        classes_mapping = make_batch_classes_mapping(cat_mappings=cat_mapping)
        result = count_proc.create_labels([{"text": "test"}], classes_mapping)
        # Modify one, other should not change
        result["count_targets"][0] = 999.0
        assert result["gold_count_val"][0] == 1.0