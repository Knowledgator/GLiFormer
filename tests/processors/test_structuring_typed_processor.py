"""Typed structuring fields in the structuring processor."""

from dataclasses import asdict

import pytest

from gliformer.config import StructuringHeadConfig
from gliformer.processing._structuring_hierarchy import (
    TYPED_ANCHORS_KEY,
    TYPED_UNRESOLVED_KEY,
    TYPED_VALUES_KEY,
)
from gliformer.processing.mappings import BatchClassesMapping
from gliformer.processing.schema import normalize_structuring_schemas
from gliformer.tasks.structuring.processor import StructuringProcessor
from tests.conftest import FakeWordsSplitter, make_config

TEXT = "Listing posted on March 3, 2024 : Sony headphones for $250 . Pickup within 2 weeks ."

TYPES = {
    "title": {"$type": "text", "$description": "item name"},
    "category": {
        "$type": "class",
        "$options": ["electronics", "furniture", "toys"],
        "$description": "product category",
    },
    "price": {"$type": "number", "$unit": "USD"},
    "posted": {"$type": "date"},
    "due": {"$type": "date"},
    "total": {"$type": "number"},
}
OUTPUT = {
    "title": "Sony headphones",
    "category": ["electronics"],
    "price": 250,
    "posted": "2024-03-03",
    "due": "2024-03-17",  # derived, not written in the text
    "total": 999,  # computed, not written in the text
}


def _processor(multi_level=True, typed=True, **overrides):
    config = make_config(
        structuring_config=asdict(
            StructuringHeadConfig(
                multi_level=multi_level,
                typed_fields=typed,
                typed_description_dropout=0.0,
                typed_shuffle_options=False,
                **overrides,
            )
        )
    )
    return StructuringProcessor(config, words_splitter=FakeWordsSplitter())


def _row(multi_level=True):
    if multi_level:
        return {
            "text": TEXT,
            "prompt": "Extract the listing.",
            "structuring": {"$root": dict(OUTPUT)},
            "structuring_types": {"$root": dict(TYPES)},
        }
    return {
        "text": TEXT,
        "structuring": {"listing": [dict(OUTPUT)]},
        "structuring_types": {"listing": dict(TYPES)},
    }


def _labels(processor, rows):
    mapping = processor.get_classes_mapping(rows)
    classes_mapping = BatchClassesMapping(cat_mapping=[], extraction_mapping=[], structuring_mapping=mapping)
    for row in rows:
        processor.resolve_spans(row)
    labels = processor.create_labels(rows, classes_mapping, max_seq_len=32)
    inputs = processor.prepare_typed_inputs(classes_mapping)
    return mapping, classes_mapping, labels, inputs


@pytest.mark.parametrize("multi_level", [True, False])
def test_class_values_move_aside_and_value_fields_become_spans(multi_level):
    processor = _processor(multi_level=multi_level)
    row = _row(multi_level)
    processor.get_classes_mapping([row])
    processor.resolve_spans(row)
    instance = next(iter(row["structuring"].values()))[0]
    assert instance[TYPED_VALUES_KEY] == {"category": ["electronics"]}
    # Values written in the text become their field's span; the gold value
    # is kept as the span's normalization target.
    assert instance["price"]["text"] == "250"
    assert instance["posted"]["text"] == "March 3, 2024"
    # Gold values are kept per field and span text, as written in the data.
    assert instance[TYPED_ANCHORS_KEY] == {
        "price": {"250": 250}, "posted": {"March 3, 2024": "2024-03-03"},
    }
    # Derived values have no span: they are neither targets nor negatives.
    assert "due" not in instance and "total" not in instance
    assert sorted(instance[TYPED_UNRESOLVED_KEY]) == ["due", "total"]


@pytest.mark.parametrize("multi_level", [True, False])
def test_typed_tensors(multi_level):
    processor = _processor(multi_level=multi_level)
    mapping, _classes, labels, inputs = _labels(processor, [_row(multi_level)])
    item = mapping[0].items[0]
    field_to_id = item.field_class_to_id.class_to_id
    assert set(item.field_types) == set(TYPES)

    # One class field with three options, one per [OPTION] in prompt order.
    assert inputs["structuring_typed_class_field_idx"].tolist() == [[field_to_id["category"]]]
    assert inputs["structuring_typed_option_index"].tolist() == [[[0, 1, 2]]]
    assert inputs["structuring_typed_option_count"].tolist() == [3]
    assert labels["structuring_typed_class_targets"][0, 0, 0].tolist() == [1.0, 0.0, 0.0]
    assert labels["structuring_typed_class_unit_mask"].all()
    assert not any("quantity" in key for key in (*labels, *inputs))

    # Derived values may be written differently; do not train "no span".
    bio = labels["structuring_typed_bio_field_mask"][0]
    assert not bio[field_to_id["total"]] and not bio[field_to_id["due"]]
    assert bio[field_to_id["price"]] and bio[field_to_id["category"]]

    # Grounded values are the value fields' span targets.
    span = labels["structuring_labels"][0, 0, :, field_to_id["posted"], 2]
    assert span.nonzero().flatten().tolist() == [3, 4, 5]


def test_prompt_renders_types_descriptions_and_options():
    processor = _processor()
    mapping, classes_mapping, _labels_, _inputs = _labels(processor, [_row()])
    prompt = processor.contribute_prompt(classes_mapping, 0)
    assert "[CHILD] category ( class , multiple ) : product category" in prompt
    assert prompt[prompt.index("[CHILD] category ( class , multiple ) : product category") + 1:][:3] == [
        "[OPTION] electronics", "[OPTION] furniture", "[OPTION] toys",
    ]
    assert "[CHILD] title : item name" in prompt
    assert "[CHILD] price ( number , USD )" in prompt
    assert "[CHILD] posted ( date )" in prompt


def test_inference_schema_recovers_types_from_descriptors():
    processor = _processor()
    structures = normalize_structuring_schemas({"$root": {
        "title": "str",
        "category": TYPES["category"],
        "posted": TYPES["posted"],
        "items": [{"name": "str", "kind": {"$type": "class", "$options": ["x", "y"]}}],
    }})
    item = {"tokenized_text": TEXT.split(), "reference_date": "2024-01-01"}
    processor.contribute_inference_input(item, structures=structures)
    mapping = processor.get_classes_mapping([item])
    field_types = mapping[0].items[0].field_types
    assert field_types["category"]["$type"] == "class"
    assert field_types["items.kind"]["$options"] == ["x", "y"]
    assert mapping[0].reference_date == "2024-01-01"
    inputs = processor.prepare_typed_inputs(BatchClassesMapping(cat_mapping=[], extraction_mapping=[], structuring_mapping=mapping))
    # Options are numbered across every class field of the row.
    assert inputs["structuring_typed_option_count"].tolist() == [5]
    assert inputs["structuring_typed_option_index"][0].tolist() == [[0, 1, 2], [3, 4, 0]]
    assert inputs["structuring_typed_option_mask"][0].tolist() == [
        [True, True, True], [True, True, False],
    ]
    assert field_types["posted"] == {"$type": "date"}


def test_typed_fields_disabled_keeps_span_behaviour():
    processor = _processor(typed=False)
    row = _row()
    mapping, _classes, labels, inputs = _labels(processor, [row])
    assert inputs == {}
    assert not any(key.startswith("structuring_typed") for key in labels)
    assert mapping[0].items[0].field_types == {}
    structures = normalize_structuring_schemas({"$root": {"posted": TYPES["posted"]}})
    item = {"tokenized_text": TEXT.split()}
    processor.contribute_inference_input(item, structures=structures)
    instance = item["structuring"][next(iter(item["structuring"]))][0]
    assert instance["posted"] == ""


def test_label_items_rebuilt_after_resolution_keep_gold_values():
    """The collator rebuilds label items from resolved rows and resolves again."""

    processor = _processor()
    row = _row()
    processor.get_classes_mapping([row])
    processor.resolve_spans(row)
    rebuilt = {
        "text": row["text"],
        "structuring": row["structuring"],
        processor.meta_key: row[processor.meta_key],
    }
    processor.resolve_spans(rebuilt)
    instance = next(iter(rebuilt["structuring"].values()))[0]
    assert instance[TYPED_VALUES_KEY]["category"] == ["electronics"]
    assert instance["price"]["text"] == "250"
    assert instance[TYPED_ANCHORS_KEY]["price"] == {"250": 250}


def test_list_items_are_grounded_on_source_spans():
    processor = _processor()
    row = {
        "text": "Sales on 2024-01-05 , February 5, 2024 and March 9, 2024 .",
        "structuring": {"$root": {"dates": ["2024-01-05", "2024-02-05", "2024-03-09"]}},
        "structuring_types": {"$root": {"dates": {"$type": "list", "$items": "date"}}},
    }
    processor.get_classes_mapping([row])
    processor.resolve_spans(row)
    instance = next(iter(row["structuring"].values()))[0]
    texts = [value["text"] for value in instance["dates"]]
    assert texts == ["2024-01-05", "February 5, 2024", "March 9, 2024"]
    assert TYPED_UNRESOLVED_KEY not in instance
