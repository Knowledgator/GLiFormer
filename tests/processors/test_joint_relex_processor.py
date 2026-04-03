"""Tests for joint relation extraction processor."""

import pytest
import torch

from glinext.tasks.joint_relex.processor import JointRelexProcessor
from glinext.tasks.ner.processor import NERProcessor
from tests.conftest import make_config, make_batch_classes_mapping, FakeWordsSplitter


@pytest.fixture
def relex_proc():
    config = make_config(joint_relex_config={})
    return JointRelexProcessor(config, words_splitter=FakeWordsSplitter())


@pytest.fixture
def ner_proc():
    config = make_config()
    return NERProcessor(config, words_splitter=FakeWordsSplitter())


class TestInheritance:
    def test_is_ner_processor(self, relex_proc):
        assert isinstance(relex_proc, NERProcessor)

    def test_get_classes_mapping_returns_none(self, relex_proc, joint_relex_item):
        assert relex_proc.get_classes_mapping([joint_relex_item]) is None

    def test_contribute_prompt_empty(self, relex_proc, joint_relex_item):
        # NER processor builds the prompt with REL tokens already embedded
        # Joint relex should return empty
        assert relex_proc.contribute_prompt(None, 0) == []


class TestCreateLabels:
    def test_basic(self, relex_proc, ner_proc):
        """Test with manually constructed mapping to avoid NER's rel[-1] format issue."""
        item = {
            "text": "John lives in New York",
            "extraction": [
                {
                    "name": "entities",
                    "ner": [
                        [0, 0, "person"],
                        [3, 4, "location"],
                    ],
                    "relations": [
                        [0, "lives_in", 1],
                    ],
                }
            ],
        }
        # Build extraction mapping manually with correct rel_class_to_id
        from glinext.mappings import BaseClassMapping, ExtractionItemMapping, ExtractionClassMapping
        ext_mapping = [ExtractionClassMapping(items=[
            ExtractionItemMapping(
                ner_class_to_id=BaseClassMapping(class_to_id={"person": 0, "location": 1}, name="entities"),
                rel_class_to_id=BaseClassMapping(class_to_id={"lives_in": 0}, name="entities"),
            )
        ])]
        classes_mapping = make_batch_classes_mapping(extraction_mappings=ext_mapping)
        result = relex_proc.create_labels([item], classes_mapping)

        assert result is not None
        assert "rel_labels" in result
        assert "rel_mask" in result
        assert "rel_batch_idx" in result

        labels = result["rel_labels"]
        # Shape: (total_groups, max_entities, max_entities, max_rel_classes)
        assert labels.shape[0] == 1
        assert labels.shape[3] >= 1

        # Check the relation (entity 0 -> entity 1, "lives_in")
        assert labels[0, 0, 1, 0] == 1.0  # head=0, tail=1, rel_class=0
        assert result["rel_mask"][0] == True

    def test_no_relations(self, relex_proc, ner_proc, ner_item):
        batch = [ner_item]
        ext_mapping = ner_proc.get_classes_mapping(batch)
        classes_mapping = make_batch_classes_mapping(extraction_mappings=ext_mapping)
        result = relex_proc.create_labels(batch, classes_mapping)
        assert result is None

    def test_empty_extraction(self, relex_proc):
        classes_mapping = make_batch_classes_mapping()
        result = relex_proc.create_labels([{"text": "hello"}], classes_mapping)
        assert result is None

    def test_batch_idx(self, relex_proc):
        from glinext.mappings import BaseClassMapping, ExtractionItemMapping, ExtractionClassMapping
        item = {
            "text": "A B",
            "extraction": [{"ner": [[0, 0, "X"]], "relations": [[0, "r", 0]]}],
        }
        ext_mapping = [ExtractionClassMapping(items=[
            ExtractionItemMapping(
                ner_class_to_id=BaseClassMapping(class_to_id={"X": 0}),
                rel_class_to_id=BaseClassMapping(class_to_id={"r": 0}),
            )
        ])] * 2
        classes_mapping = make_batch_classes_mapping(extraction_mappings=ext_mapping, batch_size=2)
        result = relex_proc.create_labels([item, item], classes_mapping)
        assert result["rel_batch_idx"].tolist() == [0, 1]