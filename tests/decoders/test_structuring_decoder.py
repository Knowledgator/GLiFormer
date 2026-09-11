"""Tests for Structuring decoder."""

from dataclasses import dataclass

import pytest
import torch

from gliformer.outputs import GLiFormerOutput
from gliformer.processing.mappings import (
    BaseClassMapping,
    BatchClassesMapping,
    CatClassMapping,
    ExtractionClassMapping,
    OpenRelexClassMapping,
    StructuringClassMapping,
    StructuringItemMapping,
)
from gliformer.processing.structuring_decoder import (
    align_structuring_anchors,
)
from gliformer.tasks.structuring.decoder import StructuringDecoder
from tests.conftest import make_config


@dataclass
class FakeModelOutput:
    structuring_entity_logits: torch.Tensor | None = None
    structuring_field_logits: torch.Tensor | None = None
    structuring_logits: torch.Tensor | None = None
    structuring_batch_origin: torch.Tensor | None = None
    structuring_anchor_mask: torch.Tensor | None = None
    structuring_objectness_logits: torch.Tensor | None = None
    structuring_anchor_relation_scores: torch.Tensor | None = None
    structuring_span_idx: torch.Tensor | None = None
    structuring_span_mask: torch.Tensor | None = None
    batch_size: int | None = None

    def __post_init__(self):
        if self.structuring_batch_origin is None and self.batch_size is None:
            logits = self.structuring_logits
            if logits is not None:
                BN = logits.shape[0]
                self.structuring_batch_origin = torch.arange(BN)
                self.batch_size = BN


@pytest.fixture
def decoder():
    config = make_config()
    return StructuringDecoder.from_config(config)


def _make_field_mapping(fields, batch_size=1):
    # class_to_id uses 0-indexed values; get_reverse_mapping produces {0: name, 1: age, ...}
    field_map = BaseClassMapping(class_to_id={f: i for i, f in enumerate(fields)})
    item = StructuringItemMapping(field_class_to_id=field_map, name="schema")
    return BatchClassesMapping(
        cat_mapping=[CatClassMapping(cat_class_to_id=[]) for _ in range(batch_size)],
        extraction_mapping=[ExtractionClassMapping() for _ in range(batch_size)],
        structuring_mapping=[
            StructuringClassMapping(items=[item])
            for _ in range(batch_size)
        ],
        open_relex_mapping=[OpenRelexClassMapping() for _ in range(batch_size)],
    )


def _make_multi_level_mapping(output_mode="schemas"):
    field_map = BaseClassMapping(
        class_to_id={"name": 0, "children.value": 1},
        name="catalog",
    )
    item = StructuringItemMapping(
        field_class_to_id=field_map,
        name="catalog",
        data_key="catalog",
        multi_level=True,
        hierarchy=[
            {
                "path": [],
                "parent_path": None,
                "parent_field_path": [],
                "fields": [{
                    "label": "name",
                    "path": ["name"],
                    "local_path": ["name"],
                }],
            },
            {
                "path": ["children"],
                "parent_path": [],
                "parent_field_path": ["children"],
                "fields": [{
                    "label": "children.value",
                    "path": ["children", "value"],
                    "local_path": ["value"],
                }],
            },
        ],
    )
    structuring_mapping = StructuringClassMapping(
        items=[item],
        output_mode=output_mode,
        multi_level=True,
    )
    return BatchClassesMapping(
        cat_mapping=[CatClassMapping(cat_class_to_id=[])],
        extraction_mapping=[ExtractionClassMapping()],
        structuring_mapping=[structuring_mapping],
        open_relex_mapping=[OpenRelexClassMapping()],
    )


def _make_three_level_project_item():
    fields = {
        "project_name": 0,
        "owner": 1,
        "milestones.milestone_name": 2,
        "milestones.due_date": 3,
        "milestones.tasks.task_name": 4,
        "milestones.tasks.assignee": 5,
        "milestones.tasks.status": 6,
    }
    return StructuringItemMapping(
        field_class_to_id=BaseClassMapping(
            class_to_id=fields, name="project",
        ),
        name="project",
        multi_level=True,
        hierarchy=[
            {
                "path": [],
                "parent_path": None,
                "parent_field_path": [],
                "fields": [
                    {
                        "label": "project_name",
                        "local_path": ["project_name"],
                    },
                    {"label": "owner", "local_path": ["owner"]},
                ],
            },
            {
                "path": ["milestones"],
                "parent_path": [],
                "parent_field_path": ["milestones"],
                "fields": [
                    {
                        "label": "milestones.milestone_name",
                        "local_path": ["milestone_name"],
                    },
                    {
                        "label": "milestones.due_date",
                        "local_path": ["due_date"],
                    },
                ],
            },
            {
                "path": ["milestones", "tasks"],
                "parent_path": ["milestones"],
                "parent_field_path": ["tasks"],
                "fields": [
                    {
                        "label": "milestones.tasks.task_name",
                        "local_path": ["task_name"],
                    },
                    {
                        "label": "milestones.tasks.assignee",
                        "local_path": ["assignee"],
                    },
                    {
                        "label": "milestones.tasks.status",
                        "local_path": ["status"],
                    },
                ],
            },
        ],
    )


def _field(label, token_index, score=0.9):
    return {
        "field": label,
        "start": token_index,
        "end": token_index,
        "score": score,
    }


def _entity_first_output(
    batch_groups,
    anchor_count,
    class_count,
    spans,
    **kwargs,
):
    """Build canonical membership, field, and entity-span tensors.

    ``spans`` contains ``(batch, anchor, start, end, class[, score])`` tuples.
    Repeated boundaries share one entity while retaining every field and
    anchor target associated with that entity.
    """

    entity_ids = [{} for _ in range(batch_groups)]
    for spec in spans:
        batch_idx, _anchor_idx, start, end, _class_idx = spec[:5]
        boundary = (start, end)
        entity_ids[batch_idx].setdefault(
            boundary,
            len(entity_ids[batch_idx]),
        )
    entity_count = max((len(ids) for ids in entity_ids), default=0)
    # Keep a non-empty rectangular entity axis for empty-record tests.
    entity_count = max(entity_count, 1)
    membership_logits = torch.full(
        (batch_groups, anchor_count, entity_count),
        -10.0,
    )
    field_logits = torch.full(
        (batch_groups, entity_count, class_count),
        -10.0,
    )
    span_idx = torch.zeros(batch_groups, entity_count, 2, dtype=torch.long)
    span_mask = torch.zeros(batch_groups, entity_count, dtype=torch.bool)

    for batch_idx, ids in enumerate(entity_ids):
        for boundary, entity_idx in ids.items():
            span_idx[batch_idx, entity_idx] = torch.tensor(boundary)
            span_mask[batch_idx, entity_idx] = True
    for spec in spans:
        batch_idx, anchor_idx, start, end, class_idx = spec[:5]
        score = float(spec[5]) if len(spec) > 5 else 5.0
        entity_idx = entity_ids[batch_idx][(start, end)]
        membership_logits[batch_idx, anchor_idx, entity_idx] = score
        field_logits[batch_idx, entity_idx, class_idx] = score

    return FakeModelOutput(
        structuring_field_logits=field_logits,
        structuring_logits=membership_logits,
        structuring_span_idx=span_idx,
        structuring_span_mask=span_mask,
        structuring_anchor_mask=torch.ones(
            batch_groups,
            anchor_count,
            dtype=torch.bool,
        ),
        **kwargs,
    )


def test_structuring_decoder_joins_ner_fields_with_anchor_membership():
    config = make_config(
        default_ner_config=False,
        structuring_config={"num_fixed_slots": 2},
    )
    decoder = StructuringDecoder.from_config(config)
    membership_logits = torch.full((1, 2, 2), -10.0)
    membership_logits[0, 0, 0] = 5.0
    membership_logits[0, 1, 1] = 5.0
    field_logits = torch.full((1, 2, 2), -10.0)
    field_logits[0, 0, 0] = 4.0
    field_logits[0, 1, 1] = 4.0
    output = GLiFormerOutput(
        batch_size=1,
        structuring_entity_logits=torch.zeros(1, 3, 2, 3),
        structuring_field_logits=field_logits,
        structuring_logits=membership_logits,
        structuring_batch_origin=torch.tensor([0]),
        structuring_anchor_mask=torch.ones(1, 2, dtype=torch.bool),
        structuring_span_idx=torch.tensor([[[0, 0], [2, 2]]]),
        structuring_span_mask=torch.ones(1, 2, dtype=torch.bool),
    )

    result = decoder.decode(
        output,
        classes_mapping=_make_field_mapping(["name", "amount"]),
    )

    assert len(result[0][0]) == 2
    assert result[0][0][0][0]["field"] == "name"
    assert result[0][0][0][0]["start"] == 0
    assert result[0][0][1][0]["field"] == "amount"
    assert result[0][0][1][0]["start"] == 2


def test_structuring_decoder_applies_anchor_objectness():
    config = make_config(
        default_ner_config=False,
        structuring_config={"num_fixed_slots": 2},
    )
    decoder = StructuringDecoder.from_config(config)
    output = GLiFormerOutput(
        batch_size=1,
        structuring_entity_logits=torch.zeros(1, 1, 1, 3),
        structuring_field_logits=torch.full((1, 1, 1), 5.0),
        structuring_logits=torch.full((1, 2, 1), 5.0),
        structuring_batch_origin=torch.tensor([0]),
        structuring_anchor_mask=torch.ones(1, 2, dtype=torch.bool),
        structuring_objectness_logits=torch.tensor([[5.0, -5.0]]),
        structuring_span_idx=torch.tensor([[[0, 0]]]),
        structuring_span_mask=torch.ones(1, 1, dtype=torch.bool),
    )

    result = decoder.decode(
        output,
        classes_mapping=_make_field_mapping(["value"]),
    )

    assert len(result[0][0]) == 1


def test_structuring_decoder_rejects_misaligned_shapes():
    config = make_config(
        default_ner_config=False,
        structuring_config={"num_fixed_slots": 2},
    )
    decoder = StructuringDecoder.from_config(config)
    output = GLiFormerOutput(
        batch_size=1,
        structuring_field_logits=torch.zeros(1, 2, 1),
        structuring_logits=torch.zeros(1, 2, 2),
        structuring_batch_origin=torch.tensor([0]),
        structuring_anchor_mask=torch.ones(1, 1, dtype=torch.bool),
        structuring_span_idx=torch.zeros(1, 2, 2, dtype=torch.long),
        structuring_span_mask=torch.ones(1, 2, dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="anchor_mask must have shape"):
        decoder.decode(
            output,
            classes_mapping=_make_field_mapping(["value"]),
        )


def test_hierarchy_mode_preserves_anchor_indices_and_aligns_graph():
    config = make_config(structuring_config={
        "multi_level": True,
        "anchor_objectness": True,
        "anchor_relations_threshold": 0.5,
    })
    decoder = StructuringDecoder.from_config(config)
    relation_scores = torch.zeros(1, 3, 3)
    relation_scores[0, 0, 2] = 0.9
    # These high invalid edges must not create cycles or reverse the hierarchy.
    relation_scores[0, 2, 0] = 0.99
    relation_scores[0, 0, 0] = 0.99
    output = _entity_first_output(
        1,
        3,
        2,
        [
            (0, 0, 0, 0, 0),
            # Anchor 1 is intentionally removed by objectness.
            (0, 1, 0, 0, 0),
            (0, 2, 1, 1, 1),
        ],
        structuring_objectness_logits=torch.tensor(
            [[5.0, -5.0, 5.0]]
        ),
        structuring_anchor_relation_scores=relation_scores,
    )
    mapping = _make_multi_level_mapping()

    decoded = decoder.decode(
        output,
        classes_mapping=mapping,
        texts=[["Root", "Child"]],
        threshold=0.5,
    )
    diagnostics = []
    result = decoder.map_results(
        decoded,
        valid_to_orig_idx=[0],
        all_start_maps=[[0, 5]],
        all_end_maps=[[4, 10]],
        valid_texts=["Root Child"],
        num_original=1,
        all_classes_mappings=[(mapping, 0)],
        anchor_diagnostics_output=diagnostics,
    )

    assert result == [{
        "catalog": [{
            "name": "Root",
            "children": [{"value": "Child"}],
        }],
    }]
    assert diagnostics[0]["summary"] == {
        "schema_group_count": 1,
        "activated_anchor_count": 2,
        "logical_anchor_count": 2,
        "selected_connection_count": 1,
        "raw_relation_connection_count": 2,
    }
    group_diagnostics = diagnostics[0]["groups"][0]
    assert group_diagnostics["schema"] == "catalog"
    assert group_diagnostics["active_anchor_ids"] == [0, 2]
    assert group_diagnostics["raw_relation_connections"] == [
        {"parent_anchor_id": 0, "child_anchor_id": 2, "score": 0.9},
        {"parent_anchor_id": 2, "child_anchor_id": 0, "score": 0.99},
    ]
    assert group_diagnostics["connections"] == [{
        "parent_logical_anchor_id": 0,
        "child_logical_anchor_id": 2,
        "parent_anchor_id": 0,
        "child_anchor_id": 2,
        "parent_path": [],
        "child_path": ["children"],
        "selection_reason": "model_relation",
        "raw_score": 0.9,
        "effective_score": 0.9,
    }]


def test_anchor_alignment_splits_subtree_collapsed_onto_one_anchor():
    text = "Atlas Maya Prototype August Build Omar active"
    tokens = text.split()
    starts = []
    cursor = 0
    for token in tokens:
        starts.append(cursor)
        cursor += len(token) + 1
    ends = [start + len(token) for start, token in zip(starts, tokens, strict=False)]
    payload = {
        "mapping": _make_three_level_project_item(),
        "nodes": [{
            "anchor_index": 7,
            "fields": [
                _field("project_name", 0),
                _field("owner", 1),
                _field("milestones.milestone_name", 2),
                _field("milestones.due_date", 3),
                _field("milestones.tasks.task_name", 4),
                _field("milestones.tasks.assignee", 5),
                _field("milestones.tasks.status", 6),
            ],
        }],
        "relation_scores": torch.zeros(8, 8),
    }

    diagnostics = {}
    result = align_structuring_anchors(
        payload, starts, ends, text, diagnostics=diagnostics,
    )

    assert result == [{
        "project_name": "Atlas",
        "owner": "Maya",
        "milestones": [{
            "milestone_name": "Prototype",
            "due_date": "August",
            "tasks": [{
                "task_name": "Build",
                "assignee": "Omar",
                "status": "active",
            }],
        }],
    }]
    assert diagnostics["active_anchor_count"] == 1
    assert diagnostics["active_anchor_ids"] == [7]
    assert len(diagnostics["logical_nodes"]) == 3
    assert [
        connection["selection_reason"]
        for connection in diagnostics["connections"]
    ] == ["same_physical_anchor", "same_physical_anchor"]
    assert {
        (
            connection["parent_anchor_id"],
            connection["child_anchor_id"],
        )
        for connection in diagnostics["connections"]
    } == {(7, 7)}


def test_collapsed_subtree_attaches_separate_child_by_text_order():
    text = "Atlas Maya Prototype August Build Omar active Create Lena complete"
    tokens = text.split()
    starts = []
    cursor = 0
    for token in tokens:
        starts.append(cursor)
        cursor += len(token) + 1
    ends = [start + len(token) for start, token in zip(starts, tokens, strict=False)]
    relations = torch.zeros(8, 8)
    relations[7, 3] = 0.2
    payload = {
        "mapping": _make_three_level_project_item(),
        "nodes": [
            {
                "anchor_index": 7,
                "fields": [
                    _field("project_name", 0, 0.95),
                    _field("owner", 1),
                    _field("milestones.milestone_name", 2),
                    _field("milestones.due_date", 3),
                    _field("milestones.tasks.task_name", 4),
                    _field("milestones.tasks.assignee", 5),
                    _field("milestones.tasks.status", 6),
                ],
            },
            {
                "anchor_index": 3,
                "fields": [
                    # A weak copy on a mixed anchor must not create another
                    # root when the same occurrence has a stronger owner.
                    _field("project_name", 0, 0.55),
                    _field("milestones.tasks.task_name", 7),
                    _field("milestones.tasks.assignee", 8),
                    _field("milestones.tasks.status", 9),
                ],
            },
        ],
        "relation_scores": relations,
    }

    result = align_structuring_anchors(
        payload, starts, ends, text,
    )

    assert len(result) == 1
    assert result[0]["project_name"] == "Atlas"
    assert result[0]["owner"] == "Maya"
    assert result[0]["milestones"][0]["tasks"] == [
        {"task_name": "Build", "assignee": "Omar", "status": "active"},
        {
            "task_name": "Create",
            "assignee": "Lena",
            "status": "complete",
        },
    ]


def test_collapsed_decoder_merges_complementary_sibling_fragments():
    text = (
        "Atlas Maya Prototype August Build Omar active Create Lena complete "
        "Beacon Noah Pilot September Recruit Priya planned"
    )
    tokens = text.split()
    starts = []
    cursor = 0
    for token in tokens:
        starts.append(cursor)
        cursor += len(token) + 1
    ends = [start + len(token) for start, token in zip(starts, tokens, strict=False)]
    payload = {
        "mapping": _make_three_level_project_item(),
        "nodes": [
            {
                "anchor_index": 7,
                "fields": [
                    _field("project_name", 0),
                    _field("owner", 1),
                    _field("milestones.milestone_name", 2),
                    _field("milestones.due_date", 3),
                    _field("milestones.tasks.task_name", 4),
                    _field("milestones.tasks.assignee", 5),
                    _field("milestones.tasks.status", 6),
                ],
            },
            {
                "anchor_index": 3,
                "fields": [
                    _field("milestones.tasks.task_name", 7),
                    _field("milestones.tasks.assignee", 8),
                    _field("milestones.tasks.status", 9),
                    _field("project_name", 10, 0.55),
                    _field("milestones.milestone_name", 12),
                ],
            },
            {
                "anchor_index": 6,
                "fields": [
                    _field("project_name", 10, 0.95),
                    _field("owner", 11),
                    _field("milestones.due_date", 13),
                    _field("milestones.tasks.task_name", 14),
                    _field("milestones.tasks.assignee", 15),
                    _field("milestones.tasks.status", 16),
                ],
            },
        ],
        "relation_scores": torch.zeros(8, 8),
    }

    result = align_structuring_anchors(
        payload, starts, ends, text,
    )

    assert result == [
        {
            "project_name": "Atlas",
            "owner": "Maya",
            "milestones": [{
                "milestone_name": "Prototype",
                "due_date": "August",
                "tasks": [
                    {
                        "task_name": "Build",
                        "assignee": "Omar",
                        "status": "active",
                    },
                    {
                        "task_name": "Create",
                        "assignee": "Lena",
                        "status": "complete",
                    },
                ],
            }],
        },
        {
            "project_name": "Beacon",
            "owner": "Noah",
            "milestones": [{
                "milestone_name": "Pilot",
                "due_date": "September",
                "tasks": [{
                    "task_name": "Recruit",
                    "assignee": "Priya",
                    "status": "planned",
                }],
            }],
        },
    ]
    assert [list(project) for project in result] == [
        ["project_name", "owner", "milestones"],
        ["project_name", "owner", "milestones"],
    ]
    assert [
        list(project["milestones"][0]) for project in result
    ] == [
        ["milestone_name", "due_date", "tasks"],
        ["milestone_name", "due_date", "tasks"],
    ]
    assert [
        list(task)
        for project in result
        for task in project["milestones"][0]["tasks"]
    ] == [
        ["task_name", "assignee", "status"],
        ["task_name", "assignee", "status"],
        ["task_name", "assignee", "status"],
    ]


def test_multi_level_relations_rescue_fieldless_container_from_objectness():
    config = make_config(structuring_config={
        "multi_level": True,
        "anchor_objectness": True,
        "anchor_relations_threshold": 0.5,
    })
    decoder = StructuringDecoder.from_config(config)
    item_mapping = StructuringItemMapping(
        field_class_to_id=BaseClassMapping(
            class_to_id={"name": 0, "children.items.value": 1},
            name="catalog",
        ),
        name="catalog",
        multi_level=True,
        hierarchy=[
            {
                "path": [],
                "parent_path": None,
                "parent_field_path": [],
                "fields": [{
                    "label": "name", "path": ["name"],
                    "local_path": ["name"],
                }],
            },
            {
                "path": ["children"],
                "parent_path": [],
                "parent_field_path": ["children"],
                "fields": [],
            },
            {
                "path": ["children", "items"],
                "parent_path": ["children"],
                "parent_field_path": ["items"],
                "fields": [{
                    "label": "children.items.value",
                    "path": ["children", "items", "value"],
                    "local_path": ["value"],
                }],
            },
        ],
    )
    mapping = BatchClassesMapping(
        cat_mapping=[CatClassMapping(cat_class_to_id=[])],
        extraction_mapping=[ExtractionClassMapping()],
        structuring_mapping=[StructuringClassMapping(
            items=[item_mapping], output_mode="schemas", multi_level=True,
        )],
    )
    relations = torch.zeros(1, 3, 3)
    relations[0, 0, 1] = 0.9
    relations[0, 1, 2] = 0.9
    output = _entity_first_output(
        1,
        3,
        2,
        [(0, 0, 0, 0, 0), (0, 2, 1, 1, 1)],
        structuring_objectness_logits=torch.tensor([[5.0, -5.0, 5.0]]),
        structuring_anchor_relation_scores=relations,
    )

    decoded = decoder.decode(
        output,
        classes_mapping=mapping,
        texts=[["Root", "Leaf"]],
    )
    result = decoder.map_results(
        decoded,
        valid_to_orig_idx=[0],
        all_start_maps=[[0, 5]],
        all_end_maps=[[4, 9]],
        valid_texts=["Root Leaf"],
        num_original=1,
    )

    assert result == [{"catalog": [{
        "name": "Root",
        "children": [{"items": [{"value": "Leaf"}]}],
    }]}]


def test_multi_level_filters_empty_records_at_every_nested_level():
    text = "Atlas Prototype Build"
    payload = {
        "mapping": _make_three_level_project_item(),
        "nodes": [
            {
                "anchor_index": 0,
                "fields": [_field("project_name", 0)],
            },
            {"anchor_index": 1, "fields": []},
            {"anchor_index": 2, "fields": []},
            {
                "anchor_index": 3,
                "fields": [_field("milestones.milestone_name", 1)],
            },
            {"anchor_index": 4, "fields": []},
            {
                "anchor_index": 5,
                "fields": [_field("milestones.tasks.task_name", 2)],
            },
        ],
        "relation_scores": torch.tensor([
            [0.0, 0.9, 0.0, 0.9, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.9, 0.9],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]),
    }

    result = align_structuring_anchors(
        payload,
        [0, 6, 16],
        [5, 15, 21],
        text,
    )

    assert result == [{
        "project_name": "Atlas",
        "owner": None,
        "milestones": [{
            "milestone_name": "Prototype",
            "due_date": None,
            "tasks": [{
                "task_name": "Build",
                "assignee": None,
                "status": None,
            }],
        }],
    }]


def test_root_only_schema_relations_do_not_bypass_objectness():
    decoder = StructuringDecoder.from_config(
        make_config(structuring_config={
            "multi_level": True,
            "anchor_objectness": True,
        })
    )
    mapping = StructuringItemMapping(
        field_class_to_id=BaseClassMapping(
            class_to_id={"ticket_id": 0}, name="support_ticket",
        ),
        name="support_ticket",
        multi_level=True,
        hierarchy=[{
            "path": [],
            "parent_path": None,
            "parent_field_path": [],
            "fields": [],
        }],
    )
    raw_mask = torch.ones(1, 2, dtype=torch.bool)
    objectness = torch.tensor([[5.0, -5.0]])
    relations = torch.zeros(1, 2, 2)
    relations[0, 0, 1] = 0.99

    resolved = decoder._resolve_anchor_mask(
        raw_mask,
        objectness,
        0.5,
        relation_scores=relations,
        expected_shape=(1, 2),
    )
    rescued = decoder._rescue_nested_relation_anchors(
        resolved,
        raw_mask,
        relations,
        [{"mapping": mapping}],
    )

    assert rescued.tolist() == [[True, False]]


def test_structuring_decoder_rejects_short_anchor_mask():
    config = make_config(structuring_config={"multi_level": True})
    decoder = StructuringDecoder.from_config(config)
    output = FakeModelOutput(
        structuring_field_logits=torch.zeros(1, 1, 1),
        structuring_logits=torch.zeros(1, 2, 1),
        structuring_anchor_mask=torch.ones(1, 1, dtype=torch.bool),
        structuring_span_idx=torch.zeros(1, 1, 2, dtype=torch.long),
        structuring_span_mask=torch.ones(1, 1, dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="structuring_anchor_mask"):
        decoder.decode(output, texts=[["value"]])


def test_hierarchy_mode_uses_shared_anchor_alignment():
    config = make_config(
        default_ner_config=False,
        structuring_config={
            "multi_level": True,
            "num_fixed_slots": 2,
            "anchor_relations_threshold": 0.5,
        },
    )
    decoder = StructuringDecoder.from_config(config)
    membership_logits = torch.full((1, 2, 2), -10.0)
    membership_logits[0, 0, 0] = 5.0
    membership_logits[0, 1, 1] = 5.0
    field_logits = torch.full((1, 2, 2), -10.0)
    field_logits[0, 0, 0] = 5.0
    field_logits[0, 1, 1] = 5.0
    relation_scores = torch.zeros(1, 2, 2)
    relation_scores[0, 0, 1] = 0.8
    output = GLiFormerOutput(
        batch_size=1,
        structuring_entity_logits=torch.zeros(1, 2, 2, 3),
        structuring_field_logits=field_logits,
        structuring_logits=membership_logits,
        structuring_batch_origin=torch.tensor([0]),
        structuring_anchor_mask=torch.ones(1, 2, dtype=torch.bool),
        structuring_anchor_relation_scores=relation_scores,
        structuring_span_idx=torch.tensor([[[0, 0], [1, 1]]]),
        structuring_span_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    mapping = _make_multi_level_mapping()

    decoded = decoder.decode(
        output,
        classes_mapping=mapping,
        texts=[["Root", "Child"]],
        threshold=0.5,
    )
    result = decoder.map_results(
        decoded,
        valid_to_orig_idx=[0],
        all_start_maps=[[0, 5]],
        all_end_maps=[[4, 10]],
        valid_texts=["Root Child"],
        num_original=1,
        all_classes_mappings=[(mapping, 0)],
    )

    assert result[0]["catalog"][0]["children"] == [{"value": "Child"}]


def test_multi_level_relation_does_not_revive_rejected_empty_anchors():
    config = make_config(
        default_ner_config=False,
        structuring_config={
            "multi_level": True,
            "anchor_objectness": True,
            "num_fixed_slots": 2,
        },
    )
    decoder = StructuringDecoder.from_config(config)
    relations = torch.zeros(1, 2, 2)
    relations[0, 0, 1] = 0.9
    output = GLiFormerOutput(
        batch_size=1,
        structuring_entity_logits=torch.zeros(1, 1, 2, 3),
        structuring_field_logits=torch.full((1, 1, 2), -10.0),
        structuring_logits=torch.full((1, 2, 1), -10.0),
        structuring_batch_origin=torch.tensor([0]),
        structuring_anchor_mask=torch.ones(1, 2, dtype=torch.bool),
        structuring_objectness_logits=torch.full((1, 2), -10.0),
        structuring_anchor_relation_scores=relations,
        structuring_span_idx=torch.tensor([[[0, 0]]]),
        structuring_span_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    mapping = _make_multi_level_mapping()

    decoded = decoder.decode(
        output, classes_mapping=mapping, texts=[["unused"]]
    )
    result = decoder.map_results(
        decoded,
        valid_to_orig_idx=[0],
        all_start_maps=[[0]],
        all_end_maps=[[6]],
        valid_texts=["unused"],
        num_original=1,
    )

    assert result == [{"catalog": []}]


def test_hierarchy_formatting_applies_required_fields_to_roots():
    config = make_config(structuring_config={"multi_level": True})
    decoder = StructuringDecoder.from_config(config)
    mapping = _make_multi_level_mapping()
    relation_scores = torch.zeros(1, 2, 2)
    relation_scores[0, 0, 1] = 0.9
    output = _entity_first_output(
        1,
        2,
        2,
        [(0, 1, 0, 0, 1)],
        structuring_anchor_relation_scores=relation_scores,
    )

    decoded = decoder.decode(
        output,
        classes_mapping=mapping,
        texts=[["Child"]],
    )
    result = decoder.map_results(
        decoded,
        valid_to_orig_idx=[0],
        all_start_maps=[[0]],
        all_end_maps=[[5]],
        valid_texts=["Child"],
        num_original=1,
        structures={
            "catalog": {
                "fields": ["name"],
                "required_fields": ["name"],
                "children": {"children": ["value"]},
            },
        },
    )

    assert result == [{"catalog": []}]


def test_multi_level_required_fields_support_nested_paths_and_children():
    decoder = StructuringDecoder.from_config(
        make_config(structuring_config={"multi_level": True})
    )
    spec = {
        "fields": {"seller": {"name": ""}},
        "required_fields": ["seller.name"],
        "children": {
            "items": {
                "fields": ["sku"],
                "required_fields": ["sku"],
            },
        },
    }

    assert decoder._filter_nested_required_fields(
        {"seller": {"name": None}, "items": [{"sku": "A"}]},
        spec,
    ) is None
    assert decoder._filter_nested_required_fields(
        {
            "seller": {"name": "Seller"},
            "items": [{"sku": None}, {"sku": "A"}],
        },
        spec,
    ) == {
        "seller": {"name": "Seller"},
        "items": [{"sku": "A"}],
    }


def test_hierarchy_formatting_unwraps_raw_root_object():
    config = make_config(structuring_config={"multi_level": True})
    decoder = StructuringDecoder.from_config(config)
    mapping = _make_multi_level_mapping(output_mode="object")
    output = _entity_first_output(
        1,
        1,
        2,
        [(0, 0, 0, 0, 0)],
        structuring_anchor_relation_scores=torch.zeros(1, 1, 1),
    )

    decoded = decoder.decode(
        output,
        classes_mapping=mapping,
        texts=[["Root"]],
    )
    result = decoder.map_results(
        decoded,
        valid_to_orig_idx=[0],
        all_start_maps=[[0]],
        all_end_maps=[[4]],
        valid_texts=["Root"],
        num_original=1,
        all_classes_mappings=[(mapping, 0)],
    )

    assert result == [{"name": "Root"}]


def test_hierarchy_formatting_preserves_arrays_and_empty_containers():
    config = make_config(structuring_config={"multi_level": True})
    decoder = StructuringDecoder.from_config(config)
    item_mapping = StructuringItemMapping(
        field_class_to_id=BaseClassMapping(
            class_to_id={"name": 0, "styles": 1, "tags": 2},
            name="root",
        ),
        name="root",
        multi_level=True,
        hierarchy=[{
            "path": [],
            "parent_path": None,
            "parent_field_path": [],
            "containers": [
                {"local_path": ["meta"], "kind": "object"},
                {"local_path": ["tags"], "kind": "array", "rank": 1},
            ],
            "fields": [
                {
                    "label": "name",
                    "path": ["name"],
                    "local_path": ["name"],
                    "shape": {"kind": "scalar"},
                },
                {
                    "label": "styles",
                    "path": ["styles"],
                    "local_path": ["styles"],
                    "shape": {"kind": "array", "rank": 1},
                },
                {
                    "label": "tags",
                    "path": ["tags"],
                    "local_path": ["tags"],
                    "shape": {"kind": "array", "rank": 1},
                },
            ],
        }],
    )
    mapping = BatchClassesMapping(
        cat_mapping=[CatClassMapping(cat_class_to_id=[])],
        extraction_mapping=[ExtractionClassMapping()],
        structuring_mapping=[StructuringClassMapping(
            items=[item_mapping], output_mode="object", multi_level=True,
        )],
    )
    output = _entity_first_output(
        1,
        1,
        3,
        [(0, 0, 0, 0, 0), (0, 0, 1, 1, 1)],
        structuring_anchor_relation_scores=torch.zeros(1, 1, 1),
    )

    decoded = decoder.decode(
        output,
        classes_mapping=mapping,
        texts=[["Root", "classical Latin"]],
    )
    result = decoder.map_results(
        decoded,
        valid_to_orig_idx=[0],
        all_start_maps=[[0, 5]],
        all_end_maps=[[4, 20]],
        valid_texts=["Root classical Latin"],
        num_original=1,
    )

    assert result == [{
        "meta": {},
        "tags": [],
        "name": "Root",
        "styles": ["classical Latin"],
    }]


def test_hierarchy_mode_drops_selected_empty_record_by_default():
    config = make_config(structuring_config={
        "multi_level": True,
        "anchor_objectness": True,
    })
    decoder = StructuringDecoder.from_config(config)
    item_mapping = StructuringItemMapping(
        field_class_to_id=BaseClassMapping(
            class_to_id={"unused": 0}, name="empty",
        ),
        name="empty",
        multi_level=True,
        hierarchy=[{
            "path": [],
            "parent_path": None,
            "parent_field_path": [],
            "containers": [
                {"local_path": ["meta"], "kind": "object"},
                {"local_path": ["tags"], "kind": "array", "rank": 1},
            ],
            "fields": [{
                "label": "unused",
                "path": ["unused"],
                "local_path": ["unused"],
                "shape": {"kind": "scalar"},
            }],
        }],
    )
    mapping = BatchClassesMapping(
        cat_mapping=[CatClassMapping(cat_class_to_id=[])],
        extraction_mapping=[ExtractionClassMapping()],
        structuring_mapping=[StructuringClassMapping(
            items=[item_mapping], output_mode="schemas", multi_level=True,
        )],
    )
    output = _entity_first_output(
        1,
        1,
        1,
        [],
        structuring_objectness_logits=torch.tensor([[5.0]]),
        structuring_anchor_relation_scores=torch.zeros(1, 1, 1),
    )

    decoded = decoder.decode(
        output, classes_mapping=mapping, texts=[["unused"]]
    )
    result = decoder.map_results(
        decoded,
        valid_to_orig_idx=[0],
        all_start_maps=[[0]],
        all_end_maps=[[6]],
        valid_texts=["unused"],
        num_original=1,
    )

    assert result == [{"empty": []}]

    preserved = decoder.decode(
        output,
        classes_mapping=mapping,
        texts=[["unused"]],
        preserve_empty_records=True,
    )
    preserved_result = decoder.map_results(
        preserved,
        valid_to_orig_idx=[0],
        all_start_maps=[[0]],
        all_end_maps=[[6]],
        valid_texts=["unused"],
        num_original=1,
    )
    assert preserved_result == [{"empty": [{
        "meta": {}, "tags": [], "unused": None,
    }]}]


def test_hierarchy_formatting_uses_evidence_for_union_paths():
    config = make_config(structuring_config={"multi_level": True})
    decoder = StructuringDecoder.from_config(config)
    item_mapping = StructuringItemMapping(
        field_class_to_id=BaseClassMapping(
            class_to_id={"value": 0, "value.name": 1},
            name="catalog",
        ),
        name="catalog",
        multi_level=True,
        hierarchy=[{
            "path": [],
            "parent_path": None,
            "parent_field_path": [],
            "containers": [
                {"local_path": ["value"], "kind": "object"},
            ],
            "fields": [
                {
                    "label": "value",
                    "path": ["value"],
                    "local_path": ["value"],
                    "shape": {"kind": "scalar"},
                },
                {
                    "label": "value.name",
                    "path": ["value", "name"],
                    "local_path": ["value", "name"],
                    "shape": {"kind": "scalar"},
                },
            ],
        }],
    )
    mapping = BatchClassesMapping(
        cat_mapping=[CatClassMapping(cat_class_to_id=[])],
        extraction_mapping=[ExtractionClassMapping()],
        structuring_mapping=[StructuringClassMapping(
            items=[item_mapping], output_mode="schemas", multi_level=True,
        )],
    )
    output = _entity_first_output(
        1,
        2,
        2,
        [(0, 0, 0, 0, 0), (0, 1, 1, 1, 1)],
        structuring_anchor_relation_scores=torch.zeros(1, 2, 2),
    )

    decoded = decoder.decode(
        output,
        classes_mapping=mapping,
        texts=[["Scalar", "Nested"]],
    )
    result = decoder.map_results(
        decoded,
        valid_to_orig_idx=[0],
        all_start_maps=[[0, 7]],
        all_end_maps=[[6, 13]],
        valid_texts=["Scalar Nested"],
        num_original=1,
    )

    assert result == [{"catalog": [
        {"value": "Scalar"},
        {"value": {"name": "Nested"}},
    ]}]


def test_multi_level_scalar_field_keeps_highest_confidence_occurrence():
    decoder = StructuringDecoder.from_config(
        make_config(structuring_config={"multi_level": True})
    )
    item_mapping = StructuringItemMapping(
        field_class_to_id=BaseClassMapping(
            class_to_id={"status": 0}, name="ticket",
        ),
        name="ticket",
        multi_level=True,
        hierarchy=[{
            "path": [],
            "parent_path": None,
            "parent_field_path": [],
            "fields": [{
                "label": "status",
                "path": ["status"],
                "local_path": ["status"],
                "shape": {"kind": "scalar"},
            }],
        }],
    )
    mapping = BatchClassesMapping(
        cat_mapping=[CatClassMapping(cat_class_to_id=[])],
        extraction_mapping=[ExtractionClassMapping()],
        structuring_mapping=[StructuringClassMapping(
            items=[item_mapping], output_mode="schemas", multi_level=True,
        )],
    )
    output = _entity_first_output(
        1,
        1,
        1,
        [
            (0, 0, 0, 0, 0, 3.0),
            (0, 0, 2, 2, 0, 6.0),
        ],
    )

    decoded = decoder.decode(
        output,
        classes_mapping=mapping,
        texts=[["active", "then", "closed"]],
    )
    result = decoder.map_results(
        decoded,
        valid_to_orig_idx=[0],
        all_start_maps=[[0, 7, 12]],
        all_end_maps=[[6, 11, 18]],
        valid_texts=["active then closed"],
        num_original=1,
    )

    assert result == [{"ticket": [{"status": "closed"}]}]


def test_hierarchy_formatting_preserves_raw_root_list_shape():
    config = make_config(structuring_config={"multi_level": True})
    decoder = StructuringDecoder.from_config(config)
    mapping = _make_multi_level_mapping(output_mode="list")
    output = _entity_first_output(
        1,
        2,
        2,
        [(0, 0, 0, 0, 0), (0, 1, 1, 1, 0)],
        structuring_anchor_relation_scores=torch.zeros(1, 2, 2),
    )

    decoded = decoder.decode(
        output,
        classes_mapping=mapping,
        texts=[["First", "Second"]],
    )
    result = decoder.map_results(
        decoded,
        valid_to_orig_idx=[0],
        all_start_maps=[[0, 6]],
        all_end_maps=[[5, 12]],
        valid_texts=["First Second"],
        num_original=1,
        all_classes_mappings=[(mapping, 0)],
        structures=[{"name": ""}],
    )

    assert result == [[{"name": "First"}, {"name": "Second"}]]


def test_multi_level_root_list_uses_list_shape_for_filtered_empty_texts():
    config = make_config(structuring_config={"multi_level": True})
    decoder = StructuringDecoder.from_config(config)

    result = decoder.map_results(
        [],
        valid_to_orig_idx=[],
        all_start_maps=[],
        all_end_maps=[],
        valid_texts=[],
        num_original=2,
        structures=[{"name": ""}],
    )

    assert result == [[], []]


def test_hierarchy_formatting_distinguishes_literal_dotted_keys():
    config = make_config(structuring_config={"multi_level": True})
    decoder = StructuringDecoder.from_config(config)
    item_mapping = StructuringItemMapping(
        field_class_to_id=BaseClassMapping(
            class_to_id={"a.b": 0, r"a\.b": 1},
            name="root",
        ),
        name="root",
        multi_level=True,
        hierarchy=[{
            "path": [],
            "parent_path": None,
            "parent_field_path": [],
            "fields": [
                {
                    "label": "a.b",
                    "path": ["a", "b"],
                    "local_path": ["a", "b"],
                },
                {
                    "label": r"a\.b",
                    "path": ["a.b"],
                    "local_path": ["a.b"],
                },
            ],
        }],
    )
    mapping = BatchClassesMapping(
        cat_mapping=[CatClassMapping(cat_class_to_id=[])],
        extraction_mapping=[ExtractionClassMapping()],
        structuring_mapping=[StructuringClassMapping(
            items=[item_mapping], output_mode="object", multi_level=True,
        )],
    )
    output = _entity_first_output(
        1,
        1,
        2,
        [(0, 0, 0, 0, 0), (0, 0, 1, 1, 1)],
        structuring_anchor_relation_scores=torch.zeros(1, 1, 1),
    )

    decoded = decoder.decode(
        output, classes_mapping=mapping, texts=[["Nested", "Literal"]]
    )
    result = decoder.map_results(
        decoded,
        valid_to_orig_idx=[0],
        all_start_maps=[[0, 7]],
        all_end_maps=[[6, 14]],
        valid_texts=["Nested Literal"],
        num_original=1,
    )

    assert result == [{"a": {"b": "Nested"}, "a.b": "Literal"}]


def test_hierarchy_formatting_orders_mixed_array_members_by_text_position():
    config = make_config(structuring_config={"multi_level": True})
    decoder = StructuringDecoder.from_config(config)
    item_mapping = StructuringItemMapping(
        field_class_to_id=BaseClassMapping(
            class_to_id={"values": 0, "values.name": 1},
            name="root",
        ),
        name="root",
        multi_level=True,
        hierarchy=[
            {
                "path": [],
                "parent_path": None,
                "parent_field_path": [],
                "fields": [{
                    "label": "values",
                    "path": ["values"],
                    "local_path": ["values"],
                }],
            },
            {
                "path": ["values"],
                "parent_path": [],
                "parent_field_path": ["values"],
                "fields": [{
                    "label": "values.name",
                    "path": ["values", "name"],
                    "local_path": ["name"],
                }],
            },
        ],
    )
    mapping = BatchClassesMapping(
        cat_mapping=[CatClassMapping(cat_class_to_id=[])],
        extraction_mapping=[ExtractionClassMapping()],
        structuring_mapping=[StructuringClassMapping(
            items=[item_mapping], output_mode="object", multi_level=True,
        )],
    )
    relation_scores = torch.zeros(1, 2, 2)
    relation_scores[0, 0, 1] = 0.9
    output = _entity_first_output(
        1,
        2,
        2,
        [
            (0, 0, 0, 0, 0),
            (0, 1, 1, 1, 1),
            (0, 0, 2, 2, 0),
        ],
        structuring_anchor_relation_scores=relation_scores,
    )

    decoded = decoder.decode(
        output,
        classes_mapping=mapping,
        texts=[["first", "middle", "last"]],
    )
    result = decoder.map_results(
        decoded,
        valid_to_orig_idx=[0],
        all_start_maps=[[0, 6, 13]],
        all_end_maps=[[5, 12, 17]],
        valid_texts=["first middle last"],
        num_original=1,
    )

    assert result == [{
        "values": ["first", {"name": "middle"}, "last"],
    }]
