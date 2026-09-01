import json

import pytest

from data_prod.generate_multilevel_array_text2json import (
    ArrayGenerationTask,
    GeminiSchemaFirstArrayGenerator,
    InvalidExampleError,
    parse_json_array,
    validate_array_instance,
    validate_array_schema,
)
from data_prod.generate_multilevel_text2json import DomainPath


def _item_schema():
    return {
        "type": "object",
        "properties": {
            "ticket_id": {"type": "string"},
            "requester": {"type": "string"},
            "issue": {"type": "string"},
            "assignment": {
                "type": "object",
                "properties": {
                    "team": {"type": "string"},
                    "workflow": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string"},
                        },
                        "required": ["status"],
                        "additionalProperties": False,
                    },
                },
                "required": ["team", "workflow"],
                "additionalProperties": False,
            },
        },
        "required": ["ticket_id", "requester", "issue", "assignment"],
        "additionalProperties": False,
    }


def _array_schema(record_count=2):
    return {
        "type": "array",
        "description": "Support tickets",
        "items": _item_schema(),
        "minItems": record_count,
        "maxItems": record_count,
    }


def _task(record_count=2):
    return ArrayGenerationTask(
        index=0,
        repeat=0,
        domain_path=DomainPath(
            domain_id="bus",
            domain_code="BUS",
            domain_name="Business",
            subdomain_id="bus.support",
            subdomain_name="Support",
            text_type_id="bus.support.queue",
            text_type_name="Support queue",
        ),
        language="English",
        facets={},
        schema_layers=3,
        record_count=record_count,
    )


def _records():
    return [
        {
            "ticket_id": "TKT-45678",
            "requester": "sarah.jones@company.com",
            "issue": "cannot access dashboard",
            "assignment": {
                "team": "Tech Support Team",
                "workflow": {"status": "in progress"},
            },
        },
        {
            "ticket_id": "TKT-45679",
            "requester": "mike.chen@client.com",
            "issue": "API timeout errors",
            "assignment": {
                "team": "Backend Team",
                "workflow": {"status": "investigating"},
            },
        },
    ]


def test_validate_array_schema_counts_item_object_layers():
    assert validate_array_schema(_array_schema(), expected_layers=3, expected_records=2) == 3


def test_validate_array_schema_rejects_object_root():
    with pytest.raises(InvalidExampleError, match="root must have type 'array'"):
        validate_array_schema(_item_schema(), expected_layers=3, expected_records=2)


def test_parse_json_array_requires_array_root():
    assert parse_json_array("```json\n[1, 2]\n```", "test") == [1, 2]
    with pytest.raises(InvalidExampleError, match="expected a root JSON array"):
        parse_json_array("{}", "test")


def test_validate_array_instance_requires_count_and_distinct_records():
    schema = _array_schema()
    validate_array_instance(_records(), schema, expected_records=2)
    with pytest.raises(InvalidExampleError, match="must be distinct"):
        validate_array_instance([_records()[0], _records()[0]], schema, expected_records=2)
    with pytest.raises(InvalidExampleError, match="expected 2 records"):
        validate_array_instance(_records()[:1], schema, expected_records=2)


def test_process_is_schema_first_and_emits_root_array(monkeypatch):
    generator = object.__new__(GeminiSchemaFirstArrayGenerator)
    calls = []
    text = (
        "TKT-45678 from sarah.jones@company.com: cannot access dashboard; "
        "Tech Support Team, in progress. TKT-45679 from mike.chen@client.com: "
        "API timeout errors; Backend Team, investigating."
    )

    def schema_call(task):
        calls.append("schema")
        return _array_schema()

    def text_call(task, schema):
        calls.append("text")
        return text

    def extraction_call(document, schema, record_count):
        calls.append("extraction")
        return _records()

    monkeypatch.setattr(generator, "generate_array_schema", schema_call)
    monkeypatch.setattr(generator, "generate_array_text", text_call)
    monkeypatch.setattr(generator, "extract_array", extraction_call)

    result = generator.process(_task())

    assert calls == ["schema", "text", "extraction"]
    assert result.accepted
    assert isinstance(result.example["structuring"], list)
    assert len(result.example["structuring"]) == 2
    assert result.schema_record["generation_order"] == ["schema", "text", "extraction"]


def test_process_skips_if_any_array_value_is_not_in_text(monkeypatch):
    generator = object.__new__(GeminiSchemaFirstArrayGenerator)
    monkeypatch.setattr(generator, "generate_array_schema", lambda task: _array_schema())
    monkeypatch.setattr(generator, "generate_array_text", lambda task, schema: "short text")
    monkeypatch.setattr(
        generator,
        "extract_array",
        lambda text, schema, record_count: _records(),
    )

    result = generator.process(_task())

    assert not result.accepted
    assert result.example is None
    assert result.rejection_reason.startswith("grounding:")


def test_schema_sidecar_is_json_serializable(monkeypatch):
    generator = object.__new__(GeminiSchemaFirstArrayGenerator)
    text = (
        "TKT-45678 from sarah.jones@company.com: cannot access dashboard; "
        "Tech Support Team, in progress. TKT-45679 from mike.chen@client.com: "
        "API timeout errors; Backend Team, investigating."
    )
    monkeypatch.setattr(generator, "generate_array_schema", lambda task: _array_schema())
    monkeypatch.setattr(generator, "generate_array_text", lambda task, schema: text)
    monkeypatch.setattr(
        generator,
        "extract_array",
        lambda document, schema, record_count: _records(),
    )

    result = generator.process(_task())

    assert json.loads(json.dumps(result.schema_record))["root_type"] == "array"
