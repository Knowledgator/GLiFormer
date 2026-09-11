"""Focused tests for the shared structuring decoding component."""

from types import SimpleNamespace

import pytest
import torch

from gliformer.processing.mappings import (
    BaseClassMapping,
    StructuringClassMapping,
    StructuringItemMapping,
)
from gliformer.processing.structuring_decoder import (
    StructuringDecoder as BaseStructuringDecoder,
)
from gliformer.processing.structuring_decoder import (
    StructuringDecoderComponent,
    resolve_structuring_decoder,
)
from gliformer.processing.structuring_processor import (
    StructuringProcessor as BaseStructuringProcessor,
)
from gliformer.tasks.structuring.decoder import StructuringDecoder
from gliformer.tasks.structuring.processor import StructuringProcessor
from tests.conftest import make_config


def _anchor_entries():
    return [
        {
            "anchor_index": 2,
            "fields": [{"field": "name", "text": "Ada"}],
            "presence_is_reliable": True,
        },
        {
            "anchor_index": 5,
            "fields": [],
            "presence_is_reliable": False,
        },
    ]


def test_canonical_adapters_extend_shared_processing_implementations():
    assert StructuringDecoder.__mro__[1] is BaseStructuringDecoder
    assert StructuringProcessor.__mro__[1] is BaseStructuringProcessor


def test_decoder_component_resolver_supports_bool_and_mapping_specs():
    flat = resolve_structuring_decoder(False)
    multi = resolve_structuring_decoder(
        {
            "type": "hierarchy",
            "params": {"relation_threshold": 0.73},
        }
    )

    assert isinstance(flat, StructuringDecoderComponent)
    assert isinstance(multi, StructuringDecoderComponent)
    assert flat.multi_level is False
    assert multi.multi_level is True
    assert multi.relation_threshold == pytest.approx(0.73)


def test_one_component_preserves_flat_and_hierarchy_payloads():
    entries = _anchor_entries()
    component = StructuringDecoderComponent("flat")
    relation_scores = torch.tensor([[0.0, 0.9], [0.0, 0.0]])
    mapping = SimpleNamespace(multi_level=True)

    flat_payload = component.finalize_group(entries)
    multi_payload = component.finalize_group(
        entries,
        relation_scores=relation_scores,
        mapping=mapping,
        output_mode="list",
        preserve_empty_records=True,
    )

    assert flat_payload == [entries[0]["fields"]]
    assert multi_payload["nodes"] == entries
    assert torch.equal(multi_payload["relation_scores"], relation_scores)
    assert multi_payload["mapping"] is mapping
    assert multi_payload["output_mode"] == "list"
    assert multi_payload["preserve_empty_records"] is True


class _RecordingComponent(StructuringDecoderComponent):
    def __init__(self):
        super().__init__("flat")
        self.calls = []

    def finalize_group(self, entries, **kwargs):
        self.calls.append(entries)
        return super().finalize_group(entries, **kwargs)


def _entry_shape(entries):
    return [
        {
            "keys": set(entry),
            "anchor_index": entry["anchor_index"],
            "has_fields": bool(entry["fields"]),
            "presence_is_reliable": entry["presence_is_reliable"],
        }
        for entry in entries
    ]


def test_entity_first_decoder_finalizes_neutral_anchor_entries():
    decoder = StructuringDecoder.from_config(
        make_config(structuring_config={"num_fixed_slots": 2})
    )
    component = _RecordingComponent()
    decoder.component = component
    decoder.decode(
        SimpleNamespace(
            batch_size=1,
            structuring_logits=torch.tensor([[[10.0], [-10.0]]]),
            structuring_field_logits=torch.tensor([[[10.0]]]),
            structuring_span_idx=torch.tensor([[[0, 0]]]),
            structuring_span_mask=torch.ones(1, 1, dtype=torch.bool),
            structuring_anchor_mask=torch.ones(1, 2, dtype=torch.bool),
            structuring_batch_origin=torch.tensor([0]),
        ),
        threshold=0.5,
        texts=[["word"]],
    )

    expected = [
        {
            "keys": {
                "anchor_index",
                "fields",
                "presence_is_reliable",
            },
            "anchor_index": 0,
            "has_fields": True,
            "presence_is_reliable": False,
        },
        {
            "keys": {
                "anchor_index",
                "fields",
                "presence_is_reliable",
            },
            "anchor_index": 1,
            "has_fields": False,
            "presence_is_reliable": False,
        },
    ]
    assert _entry_shape(component.calls[0]) == expected


def test_decoder_reads_structure_mode_component_spec():
    decoder = StructuringDecoder.from_config(
        make_config(
            structuring_config={
                "anchor_relations_threshold": 0.41,
                "structure_mode": {
                    "type": "multi_level",
                    "decoder": {
                        "type": "multi_level",
                        "params": {"relation_threshold": 0.67},
                    },
                },
            }
        )
    )

    assert decoder.multi_level is True
    assert isinstance(
        decoder.component,
        StructuringDecoderComponent,
    )
    assert decoder.component.multi_level is True
    assert decoder.component.relation_threshold == pytest.approx(0.67)


def test_decoder_applies_legacy_and_explicit_mode_options():
    legacy = StructuringDecoder.from_config(
        make_config(
            structuring_config={
                "structure_mode": {
                    "type": "multi_level",
                    "relation_threshold": 0.12,
                },
            }
        )
    )
    explicit = StructuringDecoder.from_config(
        make_config(
            structuring_config={
                "structure_mode": {
                    "type": "multi_level",
                    "decoder_options": {"relation_threshold": 0.23},
                    "decoder": {
                        "params": {"relation_threshold": 0.34},
                    },
                },
            }
        )
    )

    assert legacy.component.relation_threshold == pytest.approx(0.12)
    # The closest component-level setting wins over mode-level defaults.
    assert explicit.component.relation_threshold == pytest.approx(0.34)


def test_builtin_components_reject_unknown_mode_options():
    with pytest.raises(TypeError, match="unexpected keyword"):
        StructuringProcessor(
            make_config(
                structuring_config={
                    "structure_mode": {
                        "processor_options": {"unknown": True},
                    },
                }
            )
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        StructuringDecoder.from_config(
            make_config(
                structuring_config={
                    "structure_mode": {
                        "decoder_options": {"unknown": True},
                    },
                }
            )
        )


def _schema_mapping(name, field):
    return StructuringClassMapping(
        items=[
            StructuringItemMapping(
                field_class_to_id=BaseClassMapping(class_to_id={field: 0}),
                name=name,
            )
        ]
    )


def test_decoder_reads_the_canonical_structuring_mapping():
    decoder = StructuringDecoder.from_config(
        make_config(structuring_config={"num_fixed_slots": 1})
    )
    mapping = _schema_mapping("schema", "field")
    classes_mapping = SimpleNamespace(structuring_mapping=[mapping])

    assert decoder._build_field_class_maps(classes_mapping, 1) == [{0: "field"}]
    assert decoder._get_structuring_schema_names(
        [(classes_mapping, 0)],
        0,
    ) == ["schema"]
    assert (
        decoder._build_multi_level_contexts(
            classes_mapping,
            1,
        )[0]["mapping"]
        is mapping.items[0]
    )


def test_structuring_postprocessor_deduplicates_every_nested_record_list():
    records = [
        {"name": "Ada", "age": None},
        {"name": "Ada", "age": "36"},
        {"name": "Ada", "age": "37"},
        {"name": "ada", "age": "36"},
        {"name": "Ada", "tags": ["one"]},
        {"name": "Ada", "tags": ["one", "two"]},
    ]
    result = StructuringDecoder._postprocess_structuring({
        "schema": records,
        "nested": {
            "children": [
                {"value": "same", "meta": {"detail": None}},
                {"value": "same", "meta": {"detail": "full"}},
            ],
        },
    })

    assert result == {
        "schema": records[1:],
        "nested": {
            "children": [{
                "value": "same",
                "meta": {"detail": "full"},
            }],
        },
    }


def test_structuring_mapping_runs_dedup_as_an_optional_postprocessor():
    decoder = StructuringDecoder.from_config(
        make_config(structuring_config={"num_fixed_slots": 3})
    )

    def field(name, token):
        return {
            "field": name,
            "start": token,
            "end": token,
            "score": 0.9,
        }

    task_results = [[[
        [field("name", 0)],
        [field("name", 0), field("age", 1)],
        [field("name", 0), field("age", 2)],
    ]]]
    kwargs = {
        "valid_to_orig_idx": [0],
        "all_start_maps": [[0, 4, 7]],
        "all_end_maps": [[3, 6, 9]],
        "valid_texts": ["Ada 36 37"],
        "num_original": 1,
        "structures": {"schema_0": ["name", "age"]},
    }

    assert decoder.map_results(task_results, **kwargs) == [{
        "schema_0": [
            {"name": "Ada", "age": "36"},
            {"name": "Ada", "age": "37"},
        ],
    }]
    assert decoder.map_results(
        task_results,
        structuring_dedup=False,
        **kwargs,
    )[0]["schema_0"][0] == {"name": "Ada", "age": None}
