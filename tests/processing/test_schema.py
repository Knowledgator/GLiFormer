"""Schema-builder tests for flat, nested, and Pydantic structuring schemas."""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

import pytest

from glinext.processing.formatting import FieldType, StructuringOutputFormatter
from glinext.processing.schema import (
    GLiNExTSchema,
    _pydantic_to_field_types,
    build_structuring_output_formatter,
    normalize_structuring_schemas,
)
from glinext.tasks.structuring.processor import StructuringProcessor
from tests.conftest import FakeWordsSplitter, make_config

pydantic = pytest.importorskip("pydantic")
PYDANTIC_V2 = hasattr(pydantic.BaseModel, "model_fields")


class Address(pydantic.BaseModel):
    city: str
    postal_code: int | None = None


class LineItem(pydantic.BaseModel):
    sku: str
    quantity: int


class Order(pydantic.BaseModel):
    order_id: int
    address: Address
    items: list[LineItem]
    tags: list[int] = pydantic.Field(default_factory=list)
    optional_but_required: int | None
    scalar_union: int | str = "unknown"


class FlatModel(pydantic.BaseModel):
    name: str
    count: int = 0
    tags: list[str] = pydantic.Field(default_factory=list)


class FactoryModel(pydantic.BaseModel):
    value: int = pydantic.Field(default_factory=lambda: 7)


def test_flat_pydantic_schema_keeps_legacy_inference_shape():
    schema = GLiNExTSchema().add_structure("flat", FlatModel)

    assert schema.to_inference_kwargs() == {
        "structures": {
            "flat": {
                "fields": ["name", "count", "tags"],
                "required_fields": ["name"],
            }
        }
    }


def test_pydantic_defaults_and_factories_do_not_leak_undefined_sentinels():
    field_types = _pydantic_to_field_types(FlatModel)

    assert isinstance(field_types["name"], FieldType)
    assert field_types["name"].required is True
    assert isinstance(field_types["count"], FieldType)
    assert field_types["count"].default == 0
    assert isinstance(field_types["tags"], FieldType)
    assert field_types["tags"].default == []
    assert "undefined" not in repr(field_types).lower()

    formatter = GLiNExTSchema().add_structure("factory", FactoryModel).build_output_formatter()
    assert formatter is not None
    assert formatter.format({"factory": [{"value": "invalid"}]}) == {"factory": [{"value": 7}]}


def test_nested_pydantic_schema_emits_recursive_descriptor():
    descriptor = (
        GLiNExTSchema().add_structure("order", Order).to_inference_kwargs()["structures"]["order"]
    )

    assert descriptor["fields"] == {
        "order_id": "",
        "address": {"city": "", "postal_code": ""},
        "tags": [],
        "optional_but_required": "",
        "scalar_union": "",
    }
    expected_required = ["order_id", "address", "items"]
    if PYDANTIC_V2:
        # Pydantic v2 keeps Optional[T] required when it has no default;
        # Pydantic v1 implicitly gives that declaration a None default.
        expected_required.append("optional_but_required")
    expected_required.append("address.city")
    assert descriptor["required_fields"] == expected_required
    assert descriptor["children"] == {
        "items": {
            "fields": ["sku", "quantity"],
            "required_fields": ["sku", "quantity"],
        }
    }


def test_nested_pydantic_descriptor_is_a_multi_level_processor_contract():
    structures = GLiNExTSchema().add_structure("order", Order).to_inference_kwargs()["structures"]
    original = deepcopy(structures)
    processor = StructuringProcessor(
        make_config(
            default_ner_config=False,
            structuring_config={"structure_mode": "multi_level"},
        ),
        words_splitter=FakeWordsSplitter(),
    )
    item = {}

    # The public schema descriptor is passed through directly; callers do not
    # need a second multi-level-only schema API or an adapter.
    processor.contribute_inference_input(item, structures=structures)
    assert structures == original

    mapping = processor.get_classes_mapping([item])[0]
    assert mapping.multi_level is True
    assert list(mapping.items[0].field_class_to_id.class_to_id) == [
        "order_id",
        "address.city",
        "address.postal_code",
        "tags",
        "optional_but_required",
        "scalar_union",
        "items.sku",
        "items.quantity",
    ]
    hierarchy = mapping.items[0].hierarchy
    assert [node["path"] for node in hierarchy] == [[], ["items"]]
    assert hierarchy[1]["parent_path"] == []
    assert hierarchy[1]["parent_field_path"] == ["items"]


def test_nested_pydantic_formatter_recurses_through_objects_and_lists():
    formatter = GLiNExTSchema().add_structure("order", Order).build_output_formatter()
    assert formatter is not None

    result = formatter.format(
        {
            "order": [
                {
                    "order_id": "7",
                    "address": {"city": "Oslo", "postal_code": "1234"},
                    "items": [{"sku": "A-1", "quantity": "2"}],
                    "tags": ["10", "20"],
                    "optional_but_required": "9",
                    "scalar_union": "11",
                }
            ]
        }
    )

    assert result == {
        "order": [
            {
                "order_id": 7,
                "address": {"city": "Oslo", "postal_code": 1234},
                "items": [{"sku": "A-1", "quantity": 2}],
                "tags": [10, 20],
                "optional_but_required": 9,
                "scalar_union": 11,
            }
        ]
    }

    assert formatter.format({"order": [{"scalar_union": "not numeric"}]}) == {
        "order": [{"scalar_union": "not numeric"}]
    }


def test_descriptor_key_names_remain_literal_pydantic_fields_in_formatter():
    class LiteralFields(pydantic.BaseModel):
        fields: Address
        required_fields: int

    formatter = GLiNExTSchema().add_structure("literal", LiteralFields).build_output_formatter()
    assert formatter is not None
    assert formatter.format(
        {
            "literal": [
                {
                    "fields": {"city": "Oslo", "postal_code": "1234"},
                    "required_fields": "2",
                }
            ]
        }
    ) == {
        "literal": [
            {
                "fields": {"city": "Oslo", "postal_code": 1234},
                "required_fields": 2,
            }
        ]
    }


def test_descriptor_key_names_remain_valid_legacy_typed_mapping_fields():
    schema = GLiNExTSchema().add_structure(
        "literal",
        {"fields": {"inner": "int"}, "required_fields": "int"},
    )
    descriptor = schema.to_inference_kwargs()["structures"]["literal"]
    assert descriptor["fields"] == {
        "fields": {"inner": ""},
        "required_fields": "",
    }

    formatter = schema.build_output_formatter()
    assert formatter is not None
    assert formatter.format({"literal": [{"fields": {"inner": "2"}, "required_fields": "3"}]}) == {
        "literal": [{"fields": {"inner": 2}, "required_fields": 3}]
    }

    processor = StructuringProcessor(
        make_config(
            default_ner_config=False,
            structuring_config={"structure_mode": "multi_level"},
        ),
        words_splitter=FakeWordsSplitter(),
    )
    item = {}
    processor.contribute_inference_input(
        item,
        structures={"literal": descriptor},
    )
    mapping = processor.get_classes_mapping([item])[0]
    assert list(mapping.items[0].field_class_to_id.class_to_id) == [
        "fields.inner",
        "required_fields",
    ]


def test_fields_only_descriptor_remains_backward_compatible():
    schema = GLiNExTSchema().add_structure(
        "record",
        {"fields": {"count": "int"}},
    )

    assert schema.to_inference_kwargs() == {
        "structures": {
            "record": {
                "fields": ["count"],
                "required_fields": [],
            }
        }
    }
    formatter = schema.build_output_formatter()
    assert formatter is not None
    assert formatter.format({"record": [{"count": "4"}]}) == {
        "record": [{"count": 4}]
    }


def test_formatter_accepts_fields_only_descriptor_directly():
    formatter = StructuringOutputFormatter(
        {"record": {"fields": {"count": "int"}}}
    )

    assert formatter.format({"record": [{"count": "5"}]}) == {
        "record": [{"count": 5}]
    }


def test_formatter_accepts_recursive_processor_descriptor_directly():
    formatter = StructuringOutputFormatter(
        {
            "catalog": {
                "fields": {"seller": {"name": ""}},
                "children": {"items": ["sku", "quantity"]},
                "required_fields": ["seller.name"],
            }
        }
    )

    assert formatter.format(
        {
            "catalog": [
                {
                    "seller": {"name": "ACME"},
                    "items": [{"sku": "A-1", "quantity": "2"}],
                }
            ]
        }
    ) == {
        "catalog": [
            {
                "seller": {"name": "ACME"},
                "items": [{"sku": "A-1", "quantity": "2"}],
            }
        ]
    }


def test_nested_typed_mapping_uses_the_same_descriptor_shape():
    schema = GLiNExTSchema().add_structure(
        "order",
        {
            "order_id": "int",
            "seller": {"name": "str"},
            "items": [{"sku": "str", "quantity": "int"}],
            "scores": ["int"],
        },
    )

    descriptor = schema.to_inference_kwargs()["structures"]["order"]
    assert descriptor == {
        "fields": {
            "order_id": "",
            "seller": {"name": ""},
            "scores": [],
        },
        "children": {
            "items": {
                "fields": ["sku", "quantity"],
                "required_fields": [],
            }
        },
        "required_fields": [],
    }

    formatter = schema.build_output_formatter()
    assert formatter is not None
    assert formatter.format(
        {
            "order": [
                {
                    "order_id": "8",
                    "seller": {"name": "ACME"},
                    "items": [{"sku": "B-2", "quantity": "3"}],
                    "scores": ["4", "5"],
                }
            ]
        }
    ) == {
        "order": [
            {
                "order_id": 8,
                "seller": {"name": "ACME"},
                "items": [{"sku": "B-2", "quantity": 3}],
                "scores": [4, 5],
            }
        ]
    }


def test_pydantic_v1_compatibility_namespace_is_supported():
    v1 = pytest.importorskip("pydantic.v1")

    class V1Child(v1.BaseModel):
        value: int

    class V1Envelope(v1.BaseModel):
        children: list[V1Child]
        maybe: int | None
        tags: list[int] = v1.Field(default_factory=list)

    V1Envelope.update_forward_refs(V1Child=V1Child, Optional=Optional)
    schema = GLiNExTSchema().add_structure("envelope", V1Envelope)
    descriptor = schema.to_inference_kwargs()["structures"]["envelope"]

    # Pydantic v1 treats Optional[T] without an explicit default as optional.
    assert descriptor["required_fields"] == ["children"]
    assert descriptor["children"]["children"]["required_fields"] == ["value"]
    assert "undefined" not in repr(schema._structure_schemas).lower()

    formatter = schema.build_output_formatter()
    assert formatter is not None
    assert formatter.format(
        {
            "envelope": [
                {
                    "children": [{"value": "42"}],
                    "maybe": "5",
                    "tags": ["1", "2"],
                }
            ]
        }
    ) == {
        "envelope": [
            {
                "children": [{"value": 42}],
                "maybe": 5,
                "tags": [1, 2],
            }
        ]
    }


@pytest.mark.skipif(not hasattr(pydantic, "RootModel"), reason="Pydantic v2 only")
def test_pydantic_root_list_preserves_raw_shape_and_formats_items():
    class RootItems(pydantic.RootModel[list[LineItem]]):
        pass

    schema = GLiNExTSchema().add_structure("$root", RootItems)
    assert schema.to_inference_kwargs() == {"structures": {"$root": [{"sku": "", "quantity": ""}]}}

    formatter = schema.build_output_formatter()
    assert formatter is not None
    assert formatter.format([{"sku": "A-1", "quantity": "2"}]) == [{"sku": "A-1", "quantity": 2}]


def test_root_object_formatter_accepts_unwrapped_decoder_output():
    schema = GLiNExTSchema().add_structure("$root", Address)
    assert schema.to_inference_kwargs() == {
        "structures": {
            "$root": {
                "fields": ["city", "postal_code"],
                "required_fields": ["city"],
            }
        }
    }

    formatter = schema.build_output_formatter()
    assert formatter is not None
    assert formatter.format({"city": "Oslo", "postal_code": "1234"}) == {
        "city": "Oslo",
        "postal_code": 1234,
    }


def test_cyclic_pydantic_models_fail_with_a_clear_error():
    class RecursiveNode(pydantic.BaseModel):
        name: str
        children: list[RecursiveNode] = pydantic.Field(default_factory=list)

    rebuild = getattr(RecursiveNode, "model_rebuild", None)
    if callable(rebuild):
        rebuild()
    else:
        RecursiveNode.update_forward_refs(RecursiveNode=RecursiveNode)

    with pytest.raises(ValueError, match="Cyclic Pydantic structuring schemas"):
        GLiNExTSchema().add_structure("node", RecursiveNode)


def test_unified_template_supports_required_markers_and_portable_types():
    template = {
        "!name": "string",
        "journey_duration_days": "!integer",
        "cost_billion_usd": "number",
        "active": "boolean",
        "!!literal_bang": "string",
        "!mission_lifetime": {
            "expected_years": "integer",
            "maximum_years": "integer",
            "$required": ["expected_years"],
        },
        "agencies": ["string"],
        "instruments": [
            {
                "!name": "string",
                "abbreviation": "string",
            }
        ],
        "$required": ["active"],
    }

    schema = GLiNExTSchema().add_structure("mission", template)
    descriptor = schema.to_inference_kwargs()["structures"]["mission"]

    assert descriptor == {
        "fields": {
            "name": "",
            "journey_duration_days": "",
            "cost_billion_usd": "",
            "active": "",
            "!literal_bang": "",
            "mission_lifetime": {
                "expected_years": "",
                "maximum_years": "",
            },
            "agencies": [],
        },
        "children": {
            "instruments": {
                "fields": ["name", "abbreviation"],
                "required_fields": ["name"],
            }
        },
        "required_fields": [
            "name",
            "journey_duration_days",
            "mission_lifetime",
            "active",
            "mission_lifetime.expected_years",
        ],
    }
    assert schema.requires_multi_level is True

    formatter = schema.build_output_formatter()
    assert formatter is not None
    assert formatter.format(
        {
            "mission": [
                {
                    "name": "Voyager",
                    "journey_duration_days": "321",
                    "cost_billion_usd": "4.25",
                    "active": "yes",
                    "mission_lifetime": {
                        "expected_years": "5",
                        "maximum_years": "10",
                    },
                    "agencies": ["NASA", "ESA"],
                    "instruments": [{"name": "Camera", "abbreviation": "CAM"}],
                }
            ]
        }
    ) == {
        "mission": [
            {
                "name": "Voyager",
                "journey_duration_days": 321,
                "cost_billion_usd": 4.25,
                "active": True,
                "mission_lifetime": {
                    "expected_years": 5,
                    "maximum_years": 10,
                },
                "agencies": ["NASA", "ESA"],
                "instruments": [{"name": "Camera", "abbreviation": "CAM"}],
            }
        ]
    }


def test_advanced_template_field_options_apply_defaults_and_enums():
    schema = GLiNExTSchema().add_structure(
        "mission",
        {
            "status": {
                "$type": "string",
                "$enum": ["active", "retired"],
                "$default": "unknown",
                "$description": "Current mission status",
            },
            "mass": {"$type": "number", "$required": True},
            "scores": {"$type": "array", "$items": "integer"},
        },
    )

    assert schema.to_inference_kwargs() == {
        "structures": {
            "mission": {
                "fields": ["status", "mass", "scores"],
                "required_fields": ["mass"],
            }
        }
    }
    formatter = schema.build_output_formatter()
    assert formatter is not None
    assert formatter.format(
        {"mission": [{"status": "paused", "mass": "12.5", "scores": ["1", "2"]}]}
    ) == {
        "mission": [{"status": "unknown", "mass": 12.5, "scores": [1, 2]}]
    }


def test_same_flat_template_is_accepted_directly_by_flat_processor():
    processor = StructuringProcessor(
        make_config(default_ner_config=False, structuring_config={}),
        words_splitter=FakeWordsSplitter(),
    )
    item = {}

    processor.contribute_inference_input(
        item,
        structures={
            "mission": {
                "!name": "string",
                "cost": "number",
                "agencies": ["string"],
            }
        },
    )

    assert item["structuring_schema"]["mission"] == {
        "fields": ["name", "cost", "agencies"],
        "required_fields": ["name"],
    }
    mapping = processor.get_classes_mapping([item])[0]
    assert mapping.multi_level is False
    assert list(mapping.items[0].field_class_to_id.class_to_id) == [
        "name",
        "cost",
        "agencies",
    ]


def test_pydantic_model_is_accepted_directly_by_processor():
    processor = StructuringProcessor(
        make_config(default_ner_config=False, structuring_config={}),
        words_splitter=FakeWordsSplitter(),
    )
    item = {}

    processor.contribute_inference_input(
        item,
        structures={"flat": FlatModel},
    )

    assert item["structuring_schema"]["flat"] == {
        "fields": ["name", "count", "tags"],
        "required_fields": ["name"],
    }


def test_pydantic_output_validation_is_opt_in_and_returns_dicts():
    class ValidatedRecord(pydantic.BaseModel):
        name: str
        count: int = 2

    default_formatter = (
        GLiNExTSchema()
        .add_structure("record", ValidatedRecord)
        .build_output_formatter()
    )
    validating_formatter = (
        GLiNExTSchema()
        .add_structure("record", ValidatedRecord, validate_output=True)
        .build_output_formatter()
    )

    assert default_formatter is not None
    assert validating_formatter is not None
    raw = {"record": [{"name": "sample"}]}
    assert default_formatter.format(raw) == raw
    assert validating_formatter.format(raw) == {
        "record": [{"name": "sample", "count": 2}]
    }


def test_validate_output_rejects_non_pydantic_templates():
    with pytest.raises(TypeError, match="requires a Pydantic"):
        GLiNExTSchema().add_structure(
            "record",
            {"name": "string"},
            validate_output=True,
        )


def test_direct_template_normalizes_to_the_same_processor_wire_descriptor():
    template = {
        "!name": "string",
        "agencies": ["string"],
        "instruments": [{"!name": "string", "mass": "number"}],
    }

    assert normalize_structuring_schemas({"mission": template}) == (
        GLiNExTSchema()
        .add_structure("mission", template)
        .to_inference_kwargs()["structures"]
    )


def test_direct_pydantic_formatter_converts_and_optionally_validates():
    class DirectRecord(pydantic.BaseModel):
        count: int
        label: str = "default"

    formatter = build_structuring_output_formatter(
        {"record": DirectRecord},
        validate_output=True,
    )

    assert formatter is not None
    assert formatter.format({"record": [{"count": "3"}]}) == {
        "record": [{"count": 3, "label": "default"}]
    }


@pytest.mark.skipif(not hasattr(pydantic, "RootModel"), reason="Pydantic v2 only")
def test_named_pydantic_root_list_validates_the_whole_record_list():
    class RootItems(pydantic.RootModel[list[LineItem]]):
        pass

    formatter = (
        GLiNExTSchema()
        .add_structure("items", RootItems, validate_output=True)
        .build_output_formatter()
    )

    assert formatter is not None
    assert formatter.format(
        {"items": [{"sku": "A-1", "quantity": "2"}]}
    ) == {"items": [{"sku": "A-1", "quantity": 2}]}
