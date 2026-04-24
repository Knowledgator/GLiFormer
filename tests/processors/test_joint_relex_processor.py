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
        from glinext.processing.mappings import BaseClassMapping, ExtractionItemMapping, ExtractionClassMapping
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
        assert "rel_pair_mask" in result
        assert "rel_mask" in result
        assert "rel_batch_idx" in result

        labels = result["rel_labels"]
        # Shape: (total_groups, max_entities, max_entities, max_rel_classes)
        assert labels.shape[0] == 1
        assert labels.shape[3] >= 1

        # Check the relation (entity 0 -> entity 1, "lives_in")
        assert labels[0, 0, 1, 0] == 1.0  # head=0, tail=1, rel_class=0
        assert result["rel_pair_mask"][0, 0, 1] == 1.0
        assert result["rel_mask"][0] == True

    def test_samples_no_relation_pairs_like_gliner_relex(self, relex_proc):
        from glinext.processing.mappings import BaseClassMapping, ExtractionItemMapping, ExtractionClassMapping

        item = {
            "text": "A B C",
            "extraction": [{
                "ner": [[0, 0, "T"], [1, 1, "T"], [2, 2, "T"]],
                "relations": [[0, "r", 1]],
            }],
        }
        ext_mapping = [ExtractionClassMapping(items=[
            ExtractionItemMapping(
                ner_class_to_id=BaseClassMapping(class_to_id={"T": 0}),
                rel_class_to_id=BaseClassMapping(class_to_id={"r": 0}),
            )
        ])]
        classes_mapping = make_batch_classes_mapping(extraction_mappings=ext_mapping)

        result = relex_proc.create_labels(
            [item],
            classes_mapping,
            add_random_negatives=False,
        )

        pair_mask = result["rel_pair_mask"][0]
        rel_labels = result["rel_labels"][0]
        assert pair_mask[0, 1] == 1.0  # positive pair
        assert pair_mask[1, 0] == 1.0  # reversed no-relation pair
        assert rel_labels[0, 1, 0] == 1.0
        assert rel_labels[1, 0].sum().item() == 0.0

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
        from glinext.processing.mappings import BaseClassMapping, ExtractionItemMapping, ExtractionClassMapping
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


class TestBugRegressions:
    """End-to-end checks that the four data-prep bugs stay fixed.

    Each test uses NERProcessor's resolve_spans / get_classes_mapping pipeline
    together with JointRelexProcessor.create_labels so that the relation
    label tensor reflects the full preprocessing path — the same path the
    data collator uses during training.
    """

    def test_rel_type_collected_by_name_not_tail_id(self, ner_proc):
        # Bug 2: rel_labels must key on the relation string, not the tail_id.
        item = {
            "text": "John lives in New York",
            "extraction": [{
                "name": "entities",
                "ner": [[0, 0, "person"], [3, 4, "location"]],
                "relations": [[0, "lives_in", 1]],
            }],
        }
        mapping = ner_proc.get_classes_mapping([item])
        rel_map = mapping[0].items[0].rel_class_to_id.class_to_id
        assert "lives_in" in rel_map
        assert 1 not in rel_map  # tail_id must not leak in

    def test_sort_preserves_relation_targets(self, relex_proc, ner_proc):
        # Bug 1: sorting ner by (start, end) must remap head_id/tail_id so
        # the relation still points at the same entities.
        # "New York" appears *before* "John" in the text — the raw ner list
        # orders person first, so the sort will reverse them.
        item = {
            "text": "New York hosts John",
            "extraction": [{
                "name": "entities",
                "ner": [[3, 3, "person"], [0, 1, "location"]],
                "relations": [[0, "lives_in", 1]],  # person lives_in location
            }],
        }
        ner_proc.resolve_spans(item)
        ner = item["extraction"][0]["ner"]
        relations = item["extraction"][0]["relations"]
        # After sort: [[0,1,"location"], [3,3,"person"]]
        assert ner[0][2] == "location" and ner[1][2] == "person"
        # Relation must be remapped: person (now idx 1) lives_in location (now idx 0)
        h, r, t = relations[0]
        assert r == "lives_in"
        assert ner[h][2] == "person"
        assert ner[t][2] == "location"

    def test_repeated_mention_does_not_duplicate(self, ner_proc):
        # Bug 3: one input entity → one resolved span, even if the mention
        # text appears several times.
        item = {
            "text": "John met John at the park",
            "extraction": [{
                "name": "entities",
                "ner": [["John", "person"]],
            }],
        }
        ner_proc.resolve_spans(item)
        ner = item["extraction"][0]["ner"]
        assert len(ner) == 1

    def test_mask_is_contiguous_when_entities_truncated(self, relex_proc, ner_proc):
        # Bug 4: when max_seq_len drops a middle entity, remaining valid
        # entities must compact to 0..n-1 (no mask holes) and the relation
        # label tensor must index the compacted positions.
        item = {
            "text": "a b c d e f g h",
            "extraction": [{
                "name": "entities",
                "ner": [
                    [0, 0, "T"],      # keep
                    [5, 5, "T"],      # drop: start >= max_seq_len=5
                    [2, 2, "T"],      # keep
                ],
                "relations": [[0, "r", 2]],  # first → third
            }],
        }
        mapping = ner_proc.get_classes_mapping([item])
        classes_mapping = make_batch_classes_mapping(extraction_mappings=mapping)
        result = relex_proc.create_labels([item], classes_mapping, max_seq_len=5)
        assert result is not None

        span_mask = result["rel_span_mask"][0]
        # Two valid entities, positions 0 and 1 (no hole at old position 1)
        assert span_mask[0].item() is True
        assert span_mask[1].item() is True
        assert span_mask[2].item() is False

        # Relation head/tail ids must be remapped to the new positions.
        rel_labels = result["rel_labels"][0]  # (E, E, C_rel)
        assert rel_labels[0, 1].sum().item() == 1.0  # exactly one relation set
        assert rel_labels[0, 2].sum().item() == 0.0  # old (0,2) must be empty
