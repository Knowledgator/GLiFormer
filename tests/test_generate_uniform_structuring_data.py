import json
from collections import Counter
from dataclasses import replace

import pytest

from data_prod.generate_multilevel_text2json import DomainPath, InvalidExampleError
from data_prod.generate_uniform_structuring_data import (
    PATTERNS,
    GeminiUniformPatternGenerator,
    UniformGenerationTask,
    build_uniform_tasks,
    expanded_array_counts,
    expanded_scalar_value_count,
    find_schema_key_leakage,
    find_ungrounded_values,
    validate_pattern_instance,
    validate_pattern_schema,
)


def _domain_path() -> DomainPath:
    return DomainPath(
        domain_id="bus",
        domain_code="BUS",
        domain_name="Business",
        subdomain_id="bus.operations",
        subdomain_name="Operations",
        text_type_id="bus.operations.report",
        text_type_name="Operations report",
    )


def _task(
    pattern: str,
    *,
    scalar_fields: int = 2,
    array_fields: int | None = None,
    items_per_array: int | None = None,
) -> UniformGenerationTask:
    if array_fields is None:
        array_fields = 0 if pattern in {"0", "1.5"} else 2
    if items_per_array is None:
        items_per_array = 0 if pattern == "0" else 2
    return UniformGenerationTask(
        index=0,
        repeat=0,
        domain_path=_domain_path(),
        language="English",
        facets={},
        pattern=pattern,
        scalar_fields=scalar_fields,
        array_fields=array_fields,
        items_per_array=items_per_array,
        max_total_values=1_000,
    )


def _object_schema(task: UniformGenerationTask, remaining_depth: int, prefix: str):
    properties = {
        f"{prefix}_field_{index}": {"type": "string"}
        for index in range(task.scalar_fields)
    }
    if remaining_depth:
        for index in range(task.array_fields):
            name = f"{prefix}_records_{index}"
            properties[name] = {
                "type": "array",
                "items": _object_schema(task, remaining_depth - 1, f"{prefix}_{index}"),
                "minItems": task.items_per_array,
                "maxItems": task.items_per_array,
            }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _schema(task: UniformGenerationTask):
    if task.root_type == "object":
        return _object_schema(task, task.array_depth, "root")
    return {
        "type": "array",
        "items": _object_schema(task, task.array_depth - 1, "root"),
        "minItems": task.items_per_array,
        "maxItems": task.items_per_array,
    }


def _instance(node, counter):
    if node["type"] == "string":
        value = f"value-{next(counter)}"
        return value
    if node["type"] == "array":
        return [_instance(node["items"], counter) for _ in range(node["minItems"])]
    return {
        key: _instance(child, counter)
        for key, child in node["properties"].items()
    }


def _counter():
    index = 0
    while True:
        yield index
        index += 1


def test_uniform_plan_balances_every_configured_axis():
    tasks = build_uniform_tasks(
        [_domain_path()],
        {},
        num_samples=6 * 24,
        patterns=PATTERNS,
        min_fields=1,
        max_fields=4,
        min_array_fields=1,
        max_array_fields=3,
        min_items=1,
        max_items=2,
        domain_filters=[],
        languages=["English"],
        seed=17,
        num_ontology_leaves=None,
        shuffle=True,
        max_total_values=1_000,
    )

    assert Counter(task.pattern for task in tasks) == {pattern: 24 for pattern in PATTERNS}
    for pattern in PATTERNS:
        pattern_tasks = [task for task in tasks if task.pattern == pattern]
        field_counts = Counter(task.scalar_fields for task in pattern_tasks)
        assert set(field_counts) == {1, 2, 3, 4}
        assert max(field_counts.values()) - min(field_counts.values()) <= 1

        if pattern not in {"0", "1.5"}:
            array_counts = Counter(task.array_fields for task in pattern_tasks)
            assert set(array_counts) == {1, 2, 3}
            assert max(array_counts.values()) - min(array_counts.values()) <= 1
        if pattern != "0":
            item_counts = Counter(task.items_per_array for task in pattern_tasks)
            assert set(item_counts) == {1, 2}

    assert {task.length_profile for task in tasks} == {
        "compact",
        "short",
        "medium",
        "long",
    }
    assert all(task.expanded_array_count <= 100 for task in tasks)
    assert all(task.expanded_array_items <= 100 for task in tasks)
    assert all(task.expanded_scalar_values <= 1_000 for task in tasks)


def test_expanded_array_counts_include_every_materialized_level():
    assert expanded_array_counts("3", array_fields=4, items_per_array=5) == (
        1_684,
        8_420,
    )
    assert expanded_array_counts("3", array_fields=4, items_per_array=1) == (84, 84)
    assert expanded_array_counts("2.5", array_fields=4, items_per_array=5) == (21, 105)
    assert expanded_scalar_value_count("3", 2, 2, 2) == 170


def test_deep_plan_reduces_items_but_keeps_three_levels():
    tasks = build_uniform_tasks(
        [_domain_path()],
        {},
        num_samples=80,
        patterns=["3"],
        min_fields=1,
        max_fields=8,
        min_array_fields=1,
        max_array_fields=4,
        min_items=1,
        max_items=5,
        domain_filters=[],
        languages=["English"],
        seed=42,
        num_ontology_leaves=None,
        shuffle=False,
        max_total_arrays=100,
        max_total_array_items=100,
        max_total_values=150,
    )

    assert all(task.array_depth == 3 for task in tasks)
    assert all(task.expanded_array_count <= 100 for task in tasks)
    assert all(task.expanded_array_items <= 100 for task in tasks)
    assert all(task.expanded_scalar_values <= 150 for task in tasks)
    assert max(task.items_per_array for task in tasks) < 5


@pytest.mark.parametrize("pattern", PATTERNS)
def test_all_pattern_schemas_and_instances_validate_and_ground(pattern):
    task = _task(pattern)
    schema = _schema(task)
    instance = _instance(schema, _counter())
    text = " | ".join(_all_strings(instance))

    validate_pattern_schema(schema, task)
    validate_pattern_instance(instance, schema)
    assert find_ungrounded_values(instance, text) == []


def _all_strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [leaf for item in value for leaf in _all_strings(item)]
    return [leaf for item in value.values() for leaf in _all_strings(item)]


def test_schema_validation_rejects_wrong_array_count():
    task = _task("2", array_fields=2)
    schema = _schema(task)
    del schema["properties"]["root_records_1"]
    schema["required"].remove("root_records_1")

    with pytest.raises(InvalidExampleError, match="must have 2 array fields"):
        validate_pattern_schema(schema, task)


def test_instance_validation_rejects_wrong_length_and_duplicate_records():
    task = _task("1.5", scalar_fields=1, items_per_array=2)
    schema = _schema(task)
    one = _instance(schema["items"], _counter())

    with pytest.raises(InvalidExampleError, match="must contain 2 items"):
        validate_pattern_instance([one], schema)
    with pytest.raises(InvalidExampleError, match="distinct records"):
        validate_pattern_instance([one, one], schema)
    with pytest.raises(InvalidExampleError, match="total array items exceeds"):
        validate_pattern_instance(
            [one, _instance(schema["items"], iter(range(100, 200)))],
            schema,
            max_total_array_items=1,
        )


def test_schema_key_leakage_only_flags_raw_snake_case_names():
    task = _task("0")
    schema = _schema(task)

    assert find_schema_key_leakage(
        schema,
        "Root field zero is stated naturally.",
    ) == []
    assert find_schema_key_leakage(
        schema,
        "The raw root_field_0 key leaked into the text.",
    ) == ["root_field_0"]


def test_text_generation_enforces_length_and_schema_key_leakage():
    task = replace(
        _task("0"),
        length_profile="compact",
        text_min_chars=350,
        text_max_chars=1_000,
    )
    schema = _schema(task)
    generator = object.__new__(GeminiUniformPatternGenerator)
    generator._text_max_output_tokens = 8_192

    generator._call = lambda prompt, config: "brief"
    with pytest.raises(InvalidExampleError, match="outside the tolerated"):
        generator.generate_pattern_text(task, schema)

    generator._call = lambda prompt, config: (
        "Natural document language without artificial labels. " * 7
        + "The raw root_field_0 key appears here."
    )
    with pytest.raises(InvalidExampleError, match="schema keys leaked"):
        generator.generate_pattern_text(task, schema)

    valid_text = "Natural compact document language with varied grounded facts. " * 7
    generator._call = lambda prompt, config: valid_text
    assert generator.generate_pattern_text(task, schema) == valid_text.strip()


def test_process_runs_schema_text_extraction_and_records_pattern(monkeypatch):
    task = _task("1")
    schema = _schema(task)
    instance = _instance(schema, _counter())
    text = " | ".join(_all_strings(instance))
    generator = object.__new__(GeminiUniformPatternGenerator)
    generator._example_attempts = 1
    calls = []

    def generate_schema(requested_task):
        calls.append("schema")
        return schema

    def generate_text(requested_task, requested_schema):
        calls.append("text")
        return text

    def extract(document, requested_task, requested_schema):
        calls.append("extraction")
        return instance

    monkeypatch.setattr(generator, "generate_pattern_schema", generate_schema)
    monkeypatch.setattr(generator, "generate_pattern_text", generate_text)
    monkeypatch.setattr(generator, "extract_pattern", extract)

    result = generator.process(task)

    assert calls == ["schema", "text", "extraction"]
    assert result.accepted
    assert result.example == {"text": text, "structuring": instance}
    assert result.schema_record["pattern"] == "1"
    assert json.loads(json.dumps(result.schema_record))["schema"] == schema


def test_process_retries_whole_example_after_ungrounded_value(monkeypatch):
    task = _task("0")
    schema = _schema(task)
    instance = _instance(schema, _counter())
    generator = object.__new__(GeminiUniformPatternGenerator)
    generator._example_attempts = 2
    texts = iter(["nothing is grounded", " | ".join(_all_strings(instance))])
    calls = Counter()

    def generate_schema(requested_task):
        calls["schema"] += 1
        return schema

    def generate_text(requested_task, requested_schema):
        calls["text"] += 1
        return next(texts)

    monkeypatch.setattr(generator, "generate_pattern_schema", generate_schema)
    monkeypatch.setattr(generator, "generate_pattern_text", generate_text)
    monkeypatch.setattr(
        generator,
        "extract_pattern",
        lambda document, requested_task, requested_schema: instance,
    )

    result = generator.process(task)

    assert result.accepted
    assert calls == {"schema": 2, "text": 2}
    assert result.schema_record["generation_attempt"] == 2
