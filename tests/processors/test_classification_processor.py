"""Tests for classification processor."""

import pytest
import torch

from glinext.tasks.classification.processor import ClassificationProcessor
from glinext.processing.mappings import (
    BaseClassMapping, CatClassMapping, BatchClassesMapping,
    ExtractionClassMapping, StructuringClassMapping, OpenRelexClassMapping,
)
from tests.conftest import make_config, make_batch_classes_mapping


@pytest.fixture
def cls_proc():
    config = make_config(classification_config={})
    return ClassificationProcessor(config)


class TestGetClassesMapping:
    def test_basic(self, cls_proc, classification_item):
        mapping = cls_proc.get_classes_mapping([classification_item])
        assert len(mapping) == 1
        assert len(mapping[0].cat_class_to_id) == 1
        cat_map = mapping[0].cat_class_to_id[0]
        assert "positive" in cat_map.class_to_id
        assert "negative" in cat_map.class_to_id
        assert "neutral" in cat_map.class_to_id
        assert cat_map.name == "sentiment"

    def test_empty(self, cls_proc):
        mapping = cls_proc.get_classes_mapping([{"text": "hello"}])
        assert len(mapping) == 1
        assert len(mapping[0].cat_class_to_id) == 0

    def test_multiple_groups(self, cls_proc):
        item = {
            "text": "test",
            "classification": [
                {"name": "sent", "all_labels": ["pos", "neg"], "true_labels": ["pos"]},
                {"name": "topic", "all_labels": ["a", "b", "c"], "true_labels": ["a", "b"]},
            ],
        }
        mapping = cls_proc.get_classes_mapping([item])
        assert len(mapping[0].cat_class_to_id) == 2

    def test_negative_sampling(self, cls_proc, classification_item):
        mapping = cls_proc.get_classes_mapping(
            [classification_item], cat_negatives=["extra1", "extra2"], sample_neg=1,
        )
        cat_map = mapping[0].cat_class_to_id[0]
        # Should have original 3 + up to 1 negative
        assert len(cat_map.class_to_id) >= 3


class TestContributePrompt:
    def test_basic(self, cls_proc, classification_item):
        mapping = cls_proc.get_classes_mapping([classification_item])
        classes_mapping = make_batch_classes_mapping(cat_mappings=mapping)
        prompt = cls_proc.contribute_prompt(classes_mapping, batch_idx=0)
        assert "[P]" in prompt
        assert "[SEP]" in prompt
        assert any("[CAT]" in tok for tok in prompt)
        assert "sentiment" in prompt

    def test_with_labels_encoder(self, cls_proc, classification_item):
        mapping = cls_proc.get_classes_mapping([classification_item])
        classes_mapping = make_batch_classes_mapping(cat_mappings=mapping)
        prompt = cls_proc.contribute_prompt(classes_mapping, batch_idx=0, use_labels_encoder=True)
        assert not any("[CAT]" in tok for tok in prompt)


class TestCreateLabels:
    def test_basic(self, cls_proc, classification_item):
        mapping = cls_proc.get_classes_mapping([classification_item])
        classes_mapping = make_batch_classes_mapping(cat_mappings=mapping)
        result = cls_proc.create_labels([classification_item], classes_mapping)

        assert "cat_labels" in result
        assert "cat_batch_idx" in result
        labels = result["cat_labels"]
        assert labels.shape[0] == 1  # 1 group
        assert labels.shape[1] == 3  # 3 classes

        # "positive" should be 1.0
        cat_map = mapping[0].cat_class_to_id[0]
        pos_idx = cat_map.class_to_id["positive"]
        assert labels[0, pos_idx] == 1.0
        # Others should be 0.0
        neg_idx = cat_map.class_to_id["negative"]
        assert labels[0, neg_idx] == 0.0

    def test_multi_label(self, cls_proc):
        item = {
            "text": "test",
            "classification": [
                {
                    "name": "tags",
                    "all_labels": ["a", "b", "c"],
                    "true_labels": ["a", "c"],
                }
            ],
        }
        mapping = cls_proc.get_classes_mapping([item])
        classes_mapping = make_batch_classes_mapping(cat_mappings=mapping)
        result = cls_proc.create_labels([item], classes_mapping)
        labels = result["cat_labels"]
        cat_map = mapping[0].cat_class_to_id[0]
        assert labels[0, cat_map.class_to_id["a"]] == 1.0
        assert labels[0, cat_map.class_to_id["b"]] == 0.0
        assert labels[0, cat_map.class_to_id["c"]] == 1.0

    def test_empty(self, cls_proc):
        classes_mapping = make_batch_classes_mapping()
        result = cls_proc.create_labels([{"text": "hello"}], classes_mapping)
        assert result is None

    def test_batch_idx(self, cls_proc, classification_item):
        batch = [classification_item, classification_item]
        mapping = cls_proc.get_classes_mapping(batch)
        classes_mapping = make_batch_classes_mapping(cat_mappings=mapping, batch_size=2)
        result = cls_proc.create_labels(batch, classes_mapping)
        assert result["cat_batch_idx"].tolist() == [0, 1]