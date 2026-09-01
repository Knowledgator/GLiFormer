"""Focused tests for the entity-first open-relex processor."""

import torch

from glinext.processing.mappings import (
    BatchClassesMapping,
    CatClassMapping,
    ExtractionClassMapping,
)
from glinext.processing.processor import GLiNextTextProcessor
from glinext.tasks.open_relex.processor import OpenRelexProcessor
from tests.conftest import FakeWordsSplitter, make_config


class _TokenizerStub:
    unk_token = "[UNK]"
    pad_token = "[PAD]"


def _processor(*, num_fixed_slots=11):
    config = make_config(
        default_ner_config=False,
        open_relex_config={"num_fixed_slots": num_fixed_slots},
    )
    return OpenRelexProcessor(
        config,
        words_splitter=FakeWordsSplitter(),
    )


def _classes_mapping(open_mapping):
    batch_size = len(open_mapping)
    return BatchClassesMapping(
        cat_mapping=[CatClassMapping([]) for _ in range(batch_size)],
        extraction_mapping=[
            ExtractionClassMapping() for _ in range(batch_size)
        ],
        open_relex_mapping=open_mapping,
    )


def _directed_pair_item():
    return {
        "text": "Alice Acme",
        "open_relex": [
            {
                "name": "relations",
                "relations": [
                    {
                        "relation": "works_at",
                        "source": "Alice",
                        "target": "Acme",
                    },
                    # A second type on the same ordered endpoint pair must be
                    # merged into the same gold set element.
                    {
                        "relation": "founded",
                        "source": "Alice",
                        "target": "Acme",
                    },
                    # Repeating an identical triple also must not grow G.
                    {
                        "relation": "works_at",
                        "source": "Alice",
                        "target": "Acme",
                    },
                    # Reversing source and target is a distinct directed pair.
                    {
                        "relation": "works_at",
                        "source": "Acme",
                        "target": "Alice",
                    },
                ],
            }
        ],
    }


def _joint_extraction_item():
    return {
        "text": "John lives in New York",
        "extraction": [
            {
                "name": "relations",
                "description": "directed factual relations",
                "ner": [
                    {
                        "text": "John",
                        "start": 0,
                        "end": 4,
                        "label": "person",
                    },
                    {
                        "text": "New York",
                        "start": 14,
                        "end": 22,
                        "label": "location",
                    },
                ],
                "relations": [[0, "lives_in", 1]],
                "all_labels": ["person", "location"],
                "all_rel_labels": ["lives_in", "works_at"],
            }
        ],
    }


def _resolved_relation_group(name, label, source_idx, target_idx):
    return {
        "name": name,
        "all_labels": [label],
        "relations": [
            {
                "relation": label,
                "source": {
                    "text": name,
                    "start": source_idx,
                    "end": source_idx,
                },
                "target": {
                    "text": name,
                    "start": target_idx,
                    "end": target_idx,
                },
            }
        ],
    }


def test_joint_extraction_format_builds_entity_and_relation_targets():
    processor = _processor()
    item = _joint_extraction_item()

    processor.resolve_spans(item)
    assert item["extraction"][0]["ner"] == [
        [0, 0, "person"],
        [3, 4, "location"],
    ]

    open_mapping = processor.get_classes_mapping([item])
    relation_mapping = open_mapping[0].items[0].rel_class_to_id
    assert list(relation_mapping.class_to_id) == ["lives_in", "works_at"]
    assert relation_mapping.description == "directed factual relations"

    result = processor.create_labels(
        [item],
        _classes_mapping(open_mapping),
        max_seq_len=5,
    )

    lives_in = relation_mapping.class_to_id["lives_in"]
    works_at = relation_mapping.class_to_id["works_at"]
    assert result["open_rel_count"].tolist() == [1]
    assert result["open_rel_span_idx"].tolist() == [
        [[0, 0], [3, 4]]
    ]
    assert result["open_rel_span_mask"].tolist() == [[True, True]]
    assert result["open_rel_labels"][0, 0, lives_in].item() == 1.0
    assert result["open_rel_labels"][0, 0, works_at].item() == 0.0
    assert result[
        "open_rel_assignment_labels"
    ][0, 0, :, 0].tolist() == [1.0, 0.0]
    assert result[
        "open_rel_assignment_labels"
    ][0, 0, :, 1].tolist() == [0.0, 1.0]

    # Endpoint BIO supervision is relation-conditioned. Entity type labels do
    # not leak into the relation class axis.
    entity_labels = result["open_rel_entity_labels"]
    assert torch.equal(entity_labels[0, 0, lives_in], torch.ones(3))
    assert entity_labels[0, 3:5, lives_in].tolist() == [
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    assert not entity_labels[:, :, works_at].any()


def test_orchestrator_routes_extraction_relations_to_open_without_ner_head():
    config = make_config(
        default_ner_config=False,
        open_relex_config={"num_fixed_slots": 7},
    )
    processor = GLiNextTextProcessor(
        config,
        tokenizer=_TokenizerStub(),
        words_splitter=FakeWordsSplitter(),
    )
    item = _joint_extraction_item()

    raw_batch = processor.collate_raw_batch([item])
    mapping = raw_batch["classes_mapping"]
    label_items = processor._build_label_batch_list(raw_batch)
    labels = processor.create_open_rel_labels(
        label_items,
        mapping,
        max_seq_len=5,
    )

    assert set(processor.task_processors) == {"open_relex"}
    assert mapping.total_extraction_groups() == 0
    assert mapping.total_open_relex_groups() == 1
    assert labels is not None
    assert labels["open_rel_count"].tolist() == [1]
    assert labels["open_rel_span_idx"].tolist() == [
        [[0, 0], [3, 4]]
    ]


def test_entity_pair_and_assignment_shapes_merge_directed_pairs():
    processor = _processor(num_fixed_slots=11)
    item = _directed_pair_item()
    open_mapping = processor.get_classes_mapping([item])
    classes_mapping = _classes_mapping(open_mapping)

    result = processor.create_labels(
        [item],
        classes_mapping,
        max_seq_len=2,
    )

    # G is the two unique directed pairs, not four annotation rows and not
    # the configured eleven query slots. E contains two unique boundaries.
    assert result["open_rel_entity_labels"].shape == (1, 2, 2, 3)
    assert result["open_rel_labels"].shape == (1, 2, 2)
    assert result["open_rel_assignment_labels"].shape == (
        1,
        2,
        2,
        2,
    )
    assert result["open_rel_span_idx"].shape == (1, 2, 2)
    assert result["open_rel_span_mask"].tolist() == [[True, True]]
    assert result["open_rel_span_idx"].tolist() == [
        [[0, 0], [1, 1]]
    ]
    assert result["open_rel_mask"].tolist() == [True]
    assert result["open_rel_batch_idx"].tolist() == [0]
    assert result["open_rel_count"].tolist() == [2]

    relation_to_id = (
        open_mapping[0].items[0].rel_class_to_id.class_to_id
    )
    works_at = relation_to_id["works_at"]
    founded = relation_to_id["founded"]
    relation_labels = result["open_rel_labels"]
    assignments = result["open_rel_assignment_labels"]

    # Sorted pair 0 is Alice -> Acme and carries both relation types.
    assert relation_labels[0, 0, works_at].item() == 1.0
    assert relation_labels[0, 0, founded].item() == 1.0
    assert assignments[0, 0, :, 0].tolist() == [1.0, 0.0]
    assert assignments[0, 0, :, 1].tolist() == [0.0, 1.0]

    # Pair 1 is Acme -> Alice. Direction changes the endpoint assignments,
    # while the unrelated relation class remains negative.
    assert relation_labels[0, 1, works_at].item() == 1.0
    assert relation_labels[0, 1, founded].item() == 0.0
    assert assignments[0, 1, :, 0].tolist() == [0.0, 1.0]
    assert assignments[0, 1, :, 1].tolist() == [1.0, 0.0]


def test_entity_bio_marks_both_endpoints_for_each_relation_class():
    processor = _processor()
    item = _directed_pair_item()
    open_mapping = processor.get_classes_mapping([item])
    result = processor.create_labels(
        [item],
        _classes_mapping(open_mapping),
        max_seq_len=2,
    )
    labels = result["open_rel_entity_labels"]

    # Both one-token endpoints have start, end, and inside targets for both
    # classes that occur on Alice -> Acme.
    assert torch.equal(labels[0, 0], torch.ones(2, 3))
    assert torch.equal(labels[0, 1], torch.ones(2, 3))


def test_all_negative_groups_keep_zero_targets_for_objectness_training():
    processor = _processor()
    item = {
        "text": "Alice Acme",
        "open_relex": [
            {
                "name": "relations",
                "relations": [],
                "all_labels": ["works_at"],
            }
        ],
    }
    open_mapping = processor.get_classes_mapping([item])

    result = processor.create_labels(
        [item],
        _classes_mapping(open_mapping),
        max_seq_len=2,
    )

    assert result is not None
    assert result["open_rel_entity_labels"].shape == (1, 2, 1, 3)
    assert result["open_rel_labels"].shape == (1, 1, 1)
    assert result["open_rel_assignment_labels"].shape == (
        1,
        1,
        1,
        2,
    )
    assert not result["open_rel_entity_labels"].any()
    assert not result["open_rel_labels"].any()
    assert not result["open_rel_assignment_labels"].any()
    assert result["open_rel_span_mask"].tolist() == [[False]]
    assert result["open_rel_count"].tolist() == [0]


def test_label_encoder_uses_canonical_open_namespace():
    processor = _processor()
    item = _directed_pair_item()
    classes_mapping = _classes_mapping(
        processor.get_classes_mapping([item])
    )

    def labels_tokenizer(labels, **kwargs):
        assert labels == ["works_at", "founded"]
        assert kwargs["padding"] == "longest"
        return {
            "input_ids": torch.tensor([[1], [2]]),
            "attention_mask": torch.ones(2, 1, dtype=torch.long),
        }

    encoded = processor.prepare_label_encoder_inputs(
        classes_mapping,
        labels_tokenizer,
    )

    assert set(encoded) == {
        "open_rel_labels_input_ids",
        "open_rel_labels_attention_mask",
        "open_rel_labels_group_size",
    }
    assert encoded["open_rel_labels_group_size"].tolist() == [2]


def test_exposes_open_relation_augmentation_groups():
    processor = _processor()
    item = _directed_pair_item()
    mapping = processor.get_classes_mapping([item])
    classes_mapping = _classes_mapping(mapping)

    groups = processor.get_augmentable_label_groups(
        [item], classes_mapping,
    )

    assert len(groups) == 1
    assert groups[0].task == "open_relex"
    assert set(groups[0].positive_labels) == {"works_at", "founded"}
    groups[0].replace_labels(["acquired", "founded", "works_at"])
    assert list(
        mapping[0].items[0].rel_class_to_id.class_to_id
    ) == ["acquired", "founded", "works_at"]


def test_labels_align_after_leading_group_without_relation_schema():
    processor = _processor()
    item = {
        "text": "Alice Acme",
        "open_relex": [
            {"name": "ignored", "relations": []},
            _resolved_relation_group("kept", "works_at", 0, 1),
        ],
        "_glinext_open_relex_spans_resolved": True,
    }
    open_mapping = processor.get_classes_mapping([item])

    assert [
        mapped.name for mapped in open_mapping[0].items
    ] == ["kept"]
    augmentation_groups = processor.get_augmentable_label_groups(
        [item], _classes_mapping(open_mapping),
    )
    assert [group.parent_name for group in augmentation_groups] == ["kept"]
    assert augmentation_groups[0].positive_labels == frozenset({"works_at"})

    result = processor.create_labels(
        [item],
        _classes_mapping(open_mapping),
        max_seq_len=2,
    )

    assert result["open_rel_count"].tolist() == [1]
    assert result["open_rel_span_idx"].tolist() == [
        [[0, 0], [1, 1]]
    ]
    assert result["open_rel_labels"][0, 0, 0].item() == 1.0


def test_labels_align_after_middle_group_without_relation_schema():
    processor = _processor()
    item = {
        "text": "Alice Acme Globex",
        "open_relex": [
            _resolved_relation_group("first", "works_at", 0, 1),
            {"name": "ignored", "relations": []},
            _resolved_relation_group("second", "supplies", 1, 2),
        ],
        "_glinext_open_relex_spans_resolved": True,
    }
    open_mapping = processor.get_classes_mapping([item])

    assert [
        mapped.name for mapped in open_mapping[0].items
    ] == ["first", "second"]
    augmentation_groups = processor.get_augmentable_label_groups(
        [item], _classes_mapping(open_mapping),
    )
    assert [group.parent_name for group in augmentation_groups] == [
        "first",
        "second",
    ]
    assert [group.positive_labels for group in augmentation_groups] == [
        frozenset({"works_at"}),
        frozenset({"supplies"}),
    ]

    result = processor.create_labels(
        [item],
        _classes_mapping(open_mapping),
        max_seq_len=3,
    )

    assert result["open_rel_count"].tolist() == [1, 1]
    assert result["open_rel_span_idx"].tolist() == [
        [[0, 0], [1, 1]],
        [[1, 1], [2, 2]],
    ]
    assert result["open_rel_labels"][:, 0, 0].tolist() == [1.0, 1.0]


def test_public_head_tail_mentions_resolve_to_token_spans():
    processor = _processor()
    item = {
        "text": "Alice joined Acme",
        "open_relex": [
            {
                "all_labels": ["works_at"],
                "relations": [
                    {
                        "relation": "works_at",
                        "head": "Alice",
                        "tail": {"text": "Acme"},
                    }
                ],
            }
        ],
    }

    processor.resolve_spans(item)
    relation = item["open_relex"][0]["relations"][0]

    assert relation["head"]["start"] == 0
    assert relation["head"]["end"] == 0
    assert relation["tail"]["start"] == 2
    assert relation["tail"]["end"] == 2


def test_inference_input_and_empty_result_use_canonical_namespace():
    processor = _processor()
    item = {}

    processor.contribute_inference_input(item, relations=["works_at"])

    assert item["open_relex"][0]["all_labels"] == ["works_at"]
    assert processor.empty_inference_result(
        2,
        relations=["works_at"],
    ) == {"open_relex": [[], []]}
