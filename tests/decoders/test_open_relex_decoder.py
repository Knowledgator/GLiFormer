"""Focused tests for entity-first open-relex decoding."""

from types import SimpleNamespace

import torch

from gliformer.processing.mappings import (
    BaseClassMapping,
    BatchClassesMapping,
    CatClassMapping,
    ExtractionClassMapping,
    OpenRelexClassMapping,
    OpenRelexItemMapping,
)
from gliformer.tasks.open_relex.decoder import OpenRelexDecoder
from tests.conftest import make_config


def _decoder():
    config = make_config(
        default_ner_config=False,
        open_relex_config={"num_fixed_slots": 2},
    )
    return OpenRelexDecoder.from_config(config)


def _item_mapping(*labels):
    return OpenRelexItemMapping(
        rel_class_to_id=BaseClassMapping(
            {label: index for index, label in enumerate(labels)}
        )
    )


def _mapping(*open_groups):
    return BatchClassesMapping(
        cat_mapping=[CatClassMapping([])],
        extraction_mapping=[ExtractionClassMapping()],
        open_relex_mapping=[
            OpenRelexClassMapping(items=list(open_groups))
        ],
    )


def _output(
    logits,
    assignments,
    spans,
    *,
    span_mask=None,
    anchor_mask=None,
    objectness=None,
    batch_origin=None,
    batch_size=None,
):
    if span_mask is None:
        span_mask = torch.ones(spans.shape[:2], dtype=torch.bool)
    if batch_origin is None:
        batch_origin = torch.arange(logits.shape[0])
    if batch_size is None:
        batch_size = (
            int(batch_origin.max().item()) + 1
            if batch_origin.numel()
            else 0
        )
    return SimpleNamespace(
        open_rel_logits=logits,
        open_rel_assignment_logits=assignments,
        open_rel_span_idx=spans,
        open_rel_span_mask=span_mask,
        open_rel_anchor_mask=anchor_mask,
        open_rel_objectness_logits=objectness,
        open_rel_batch_origin=batch_origin,
        batch_size=batch_size,
    )


def test_decoder_returns_empty_when_logits_are_absent():
    decoder = _decoder()

    assert decoder.decode(SimpleNamespace(open_rel_logits=None)) == []


def test_decoder_selects_exactly_one_argmax_entity_per_role():
    decoder = _decoder()
    logits = torch.full((1, 1, 1), 8.0)
    assignments = torch.full((1, 1, 4, 2), -8.0)
    # Multiple valid entities clear the threshold for each role. Only the
    # role-wise argmax must be emitted, and a masked higher score is ignored.
    assignments[0, 0, 0, 0] = 9.0
    assignments[0, 0, 1, 0] = 7.0
    assignments[0, 0, 2, 1] = 9.0
    assignments[0, 0, 1, 1] = 7.0
    assignments[0, 0, 3, :] = 20.0
    spans = torch.tensor([[[0, 0], [1, 1], [2, 2], [3, 3]]])
    output = _output(
        logits,
        assignments,
        spans,
        span_mask=torch.tensor([[True, True, True, False]]),
    )

    decoded = decoder.decode(
        output,
        classes_mapping=_mapping(_item_mapping("works_at")),
        texts=[["Alice", "near", "Acme", "masked"]],
    )

    assert len(decoded[0][0]) == 1
    triple = decoded[0][0][0]
    assert triple["relation"] == "works_at"
    assert triple["head"] == {
        "start": 0,
        "end": 0,
        "text": "Alice",
    }
    assert triple["tail"] == {
        "start": 2,
        "end": 2,
        "text": "Acme",
    }


def test_decoder_preserves_relation_direction_across_slots():
    decoder = _decoder()
    logits = torch.full((1, 2, 1), 8.0)
    assignments = torch.full((1, 2, 2, 2), -8.0)
    assignments[0, 0, 0, 0] = 9.0
    assignments[0, 0, 1, 1] = 9.0
    assignments[0, 1, 1, 0] = 9.0
    assignments[0, 1, 0, 1] = 9.0
    spans = torch.tensor([[[0, 0], [1, 1]]])

    decoded = decoder.decode(
        _output(logits, assignments, spans),
        classes_mapping=_mapping(_item_mapping("related_to")),
        texts=[["Alice", "Acme"]],
    )

    directions = [
        (triple["head"]["text"], triple["tail"]["text"])
        for triple in decoded[0][0]
    ]
    assert directions == [("Alice", "Acme"), ("Acme", "Alice")]


def test_duplicate_slots_collapse_to_one_public_triple():
    decoder = _decoder()
    logits = torch.full((1, 2, 1), 8.0)
    assignments = torch.full((1, 2, 2, 2), -8.0)
    assignments[:, :, 0, 0] = 9.0
    assignments[:, :, 1, 1] = 9.0
    spans = torch.tensor([[[0, 0], [1, 1]]])

    decoded = decoder.decode(
        _output(logits, assignments, spans),
        classes_mapping=_mapping(_item_mapping("related_to")),
        texts=[["Alice", "Acme"]],
    )

    assert len(decoded[0][0]) == 1
    assert decoded[0][0][0]["head"]["text"] == "Alice"
    assert decoded[0][0][0]["tail"]["text"] == "Acme"


def test_objectness_gates_relation_slots_before_endpoint_decoding():
    decoder = _decoder()
    logits = torch.full((1, 2, 1), 8.0)
    assignments = torch.full((1, 2, 2, 2), -8.0)
    assignments[0, 0, 0, 0] = 9.0
    assignments[0, 0, 1, 1] = 9.0
    assignments[0, 1, 1, 0] = 9.0
    assignments[0, 1, 0, 1] = 9.0
    spans = torch.tensor([[[0, 0], [1, 1]]])
    output = _output(
        logits,
        assignments,
        spans,
        anchor_mask=torch.ones(1, 2, dtype=torch.bool),
        objectness=torch.tensor([[-8.0, 8.0]]),
    )

    decoded = decoder.decode(
        output,
        classes_mapping=_mapping(_item_mapping("related_to")),
        texts=[["Alice", "Acme"]],
        objectness_threshold=0.5,
    )

    assert len(decoded[0][0]) == 1
    assert decoded[0][0][0]["head"]["text"] == "Acme"
    assert decoded[0][0][0]["tail"]["text"] == "Alice"


def test_flat_groups_use_canonical_open_relation_mapping():
    decoder = _decoder()
    logits = torch.full((2, 1, 1), 8.0)
    assignments = torch.full((2, 1, 2, 2), -8.0)
    assignments[:, 0, 0, 0] = 9.0
    assignments[:, 0, 1, 1] = 9.0
    spans = torch.tensor(
        [
            [[0, 0], [1, 1]],
            [[2, 2], [3, 3]],
        ]
    )
    output = _output(
        logits,
        assignments,
        spans,
        batch_origin=torch.tensor([0, 0]),
        batch_size=1,
    )
    classes_mapping = _mapping(
        _item_mapping("first_relation"),
        _item_mapping("second_relation"),
    )

    decoded = decoder.decode(
        output,
        classes_mapping=classes_mapping,
        texts=[["A", "B", "C", "D"]],
    )

    assert len(decoded) == 1
    assert len(decoded[0]) == 2
    assert decoded[0][0][0]["relation"] == "first_relation"
    assert decoded[0][1][0]["relation"] == "second_relation"
    assert decoded[0][0][0]["head"]["text"] == "A"
    assert decoded[0][1][0]["head"]["text"] == "C"


def test_map_results_converts_each_directed_endpoint_to_character_offsets():
    decoder = _decoder()
    token_results = [
        [
            [
                {
                    "head": {"start": 0, "end": 0, "text": "Alice"},
                    "tail": {"start": 2, "end": 2, "text": "Bob"},
                    "relation": "knows",
                    "score": 0.9,
                }
            ]
        ]
    ]

    mapped = decoder.map_results(
        token_results,
        valid_to_orig_idx=[1],
        all_start_maps=[[0, 6, 10]],
        all_end_maps=[[5, 9, 13]],
        valid_texts=["Alice met Bob"],
        num_original=2,
    )

    assert mapped[0] == []
    assert mapped[1] == [
        {
            "head": {"start": 0, "end": 5, "text": "Alice"},
            "tail": {"start": 10, "end": 13, "text": "Bob"},
            "relation": "knows",
            "score": 0.9,
        }
    ]
