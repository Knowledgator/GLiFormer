import json

from glinex_eval.eval_glinext_text2json import (
    build_flat_fields,
    build_flat_schema,
    build_native_schema,
    convert_to_json_template,
    extract_flat,
    extract_native,
    flatten_json_by_path,
    normalize_template,
    prepare_eval_records,
)


def test_convert_to_json_template_preserves_nested_shape_and_types():
    target = {
        "name": "Alice",
        "age": 30,
        "score": 4.5,
        "active": True,
        "addresses": [{"city": "Rome", "zip": 10100}],
        "aliases": ["A", "Al"],
    }

    assert convert_to_json_template(target) == {
        "name": "string",
        "age": "integer",
        "score": "number",
        "active": "boolean",
        "addresses": [{"city": "string", "zip": "integer"}],
        "aliases": ["string"],
    }


def test_prepare_eval_records_accepts_output_or_extracted_json_strings():
    rows = [
        {
            "text": "Alice is 30.",
            "output": json.dumps({"name": "Alice", "age": 30}),
        },
        {
            "text": "Bob is active.",
            "extracted": {"name": "Bob", "active": True},
            "template": json.dumps({"name": "str", "active": "bool"}),
        },
        {"text": "Invalid", "output": "not JSON"},
    ]

    prepared = prepare_eval_records(rows, shuffle=False)

    assert prepared == [
        {
            "text": "Alice is 30.",
            "template": {"name": "string", "age": "integer"},
            "solution": {"name": "Alice", "age": 30},
        },
        {
            "text": "Bob is active.",
            "template": {"name": "string", "active": "boolean"},
            "solution": {"name": "Bob", "active": True},
        },
    ]


def test_native_schema_preserves_raw_root_shape():
    object_template = {"seller": {"name": "string"}}
    list_template = [{"name": "string"}]

    assert build_native_schema(object_template) == {"$root": object_template}
    assert build_native_schema(list_template) is list_template


def test_flat_schema_uses_path_qualified_list_fields_for_nested_records():
    template = {
        "catalog": [
            {
                "id": "string",
                "seller": {"name": "string"},
                "items": [{"sku": "string", "quantity": "integer"}],
            }
        ],
        "active": "boolean",
    }

    fields = build_flat_fields(template)

    assert fields == [
        {"path": "catalog.id", "name": "catalog.id", "is_list": True, "type": "string"},
        {
            "path": "catalog.seller.name",
            "name": "catalog.seller.name",
            "is_list": True,
            "type": "string",
        },
        {
            "path": "catalog.items.sku",
            "name": "catalog.items.sku",
            "is_list": True,
            "type": "string",
        },
        {
            "path": "catalog.items.quantity",
            "name": "catalog.items.quantity",
            "is_list": True,
            "type": "integer",
        },
        {"path": "active", "name": "active", "is_list": False, "type": "boolean"},
    ]
    assert build_flat_schema(fields) == {
        "record": {
            "catalog.id": ["string"],
            "catalog.seller.name": ["string"],
            "catalog.items.sku": ["string"],
            "catalog.items.quantity": ["integer"],
            "active": "boolean",
        }
    }


class _FakeModel:
    def __init__(self, prediction):
        self.prediction = prediction
        self.calls = []

    def structure(self, text, structures, **kwargs):
        self.calls.append((text, structures, kwargs))
        return self.prediction


def test_extract_native_calls_structure_and_builds_path_view():
    prediction = {"person": {"name": "Alice", "aliases": ["A", "Al"]}}
    model = _FakeModel(prediction)

    result, by_path = extract_native(
        model,
        "Alice, also A and Al",
        {"person": {"name": "string", "aliases": ["string"]}},
        threshold=0.4,
    )

    assert result == prediction
    assert by_path == {
        "person.name": ["Alice"],
        "person.aliases": ["A", "Al"],
    }
    assert model.calls[0][1] == {
        "$root": {"person": {"name": "string", "aliases": ["string"]}}
    }
    assert model.calls[0][2]["threshold"] == 0.4


def test_extract_flat_reconstructs_nested_paths():
    fields = build_flat_fields(
        {"person": {"name": "string", "aliases": ["string"]}}
    )
    model = _FakeModel(
        {"record": [{"person.name": "Alice", "person.aliases": ["A", "Al"]}]}
    )

    prediction, by_path = extract_flat(model, "Alice", fields, threshold=0.5)

    assert prediction == {"person": {"name": "Alice", "aliases": ["A", "Al"]}}
    assert by_path == {
        "person.name": ["Alice"],
        "person.aliases": ["A", "Al"],
    }


def test_flatten_json_by_path_merges_values_from_object_arrays():
    assert flatten_json_by_path(
        {"people": [{"name": "Alice"}, {"name": "Bob"}]}
    ) == {"people.name": ["Alice", "Bob"]}


def test_normalize_template_maps_common_aliases():
    assert normalize_template(
        {"name": "str", "count": "int", "score": "float", "tags": ["any"]}
    ) == {
        "name": "string",
        "count": "integer",
        "score": "number",
        "tags": ["string"],
    }


def test_normalize_template_preserves_descriptor_values_and_required_names():
    assert normalize_template(
        {
            "status": {"$type": "str", "$enum": ["active", "retired"]},
            "items": {"sku": "str", "$required": ["sku"]},
        }
    ) == {
        "status": {"$type": "string", "$enum": ["active", "retired"]},
        "items": {"sku": "string", "$required": ["sku"]},
    }
