import json

import pytest

from data_prod.generate_multilevel_text2json import (
    DomainPath,
    GeminiMultiLevelGenerator,
    GenerationTask,
    InvalidExampleError,
    build_tasks,
    find_ungrounded_values,
    load_domain_paths,
    parse_json_object,
    run_generation,
    validate_instance,
    validate_schema,
)


def _three_layer_schema():
    return {
        "type": "object",
        "properties": {
            "company": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "offices": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                            },
                            "required": ["city"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["name", "offices"],
                "additionalProperties": False,
            }
        },
        "required": ["company"],
        "additionalProperties": False,
    }


def _task():
    return GenerationTask(
        index=0,
        repeat=0,
        domain_path=DomainPath(
            domain_id="bus",
            domain_code="BUS",
            domain_name="Business",
            subdomain_id="bus.operations",
            subdomain_name="Operations",
            text_type_id="bus.operations.company_profile",
            text_type_name="Company profile",
        ),
        language="English",
        facets={},
        schema_layers=3,
    )


def test_validate_schema_counts_objects_and_not_arrays():
    assert validate_schema(_three_layer_schema()) == 3


@pytest.mark.parametrize(
    "mutation",
    [
        lambda schema: schema.update({"unexpected": True}),
        lambda schema: schema.update({"required": []}),
        lambda schema: schema.update({"additionalProperties": True}),
    ],
)
def test_validate_schema_rejects_corrupt_root(mutation):
    schema = _three_layer_schema()
    mutation(schema)
    with pytest.raises(InvalidExampleError, match="schema"):
        validate_schema(schema)


def test_validate_instance_and_exact_grounding():
    schema = _three_layer_schema()
    extraction = {
        "company": {
            "name": "Northwind Labs",
            "offices": [{"city": "Oslo"}, {"city": "Lima"}],
        }
    }
    text = "Northwind Labs opened offices in Oslo and Lima."

    validate_instance(extraction, schema)
    assert find_ungrounded_values(extraction, text) == []
    extraction["company"]["offices"][1]["city"] = "lima"
    assert find_ungrounded_values(extraction, text) == [("$.company.offices[1].city", "lima")]


def test_validate_instance_rejects_missing_key():
    extraction = {"company": {"name": "Northwind Labs", "offices": []}}
    with pytest.raises(InvalidExampleError, match="non-empty array"):
        validate_instance(extraction, _three_layer_schema())


def test_process_skips_entire_example_when_one_leaf_is_ungrounded(monkeypatch):
    generator = object.__new__(GeminiMultiLevelGenerator)
    monkeypatch.setattr(generator, "generate_text", lambda task: "Northwind Labs is in Oslo.")
    monkeypatch.setattr(
        generator, "generate_schema", lambda text, requested_layers: _three_layer_schema()
    )
    monkeypatch.setattr(
        generator,
        "extract",
        lambda text, schema: {
            "company": {
                "name": "Northwind Labs",
                "offices": [{"city": "Lima"}],
            }
        },
    )

    result = generator.process(_task())

    assert not result.accepted
    assert result.example is None
    assert result.rejection_reason.startswith("grounding:")


def test_run_generation_writes_glinext_shape_and_schema_sidecar(tmp_path):
    task = _task()

    class FakeGenerator:
        def process(self, requested_task):
            generator = object.__new__(GeminiMultiLevelGenerator)
            generator.generate_text = lambda task: "Northwind Labs is in Oslo."
            generator.generate_schema = lambda text, requested_layers: _three_layer_schema()
            generator.extract = lambda text, schema: {
                "company": {
                    "name": "Northwind Labs",
                    "offices": [{"city": "Oslo"}],
                }
            }
            return generator.process(requested_task)

    output_path = tmp_path / "examples.jsonl"
    schema_path = tmp_path / "schemas.jsonl"
    accepted, rejected = run_generation(
        [task],
        FakeGenerator(),
        output_path=output_path,
        schema_output_path=schema_path,
        append=False,
        workers=1,
        progress_every=1,
    )

    example = json.loads(output_path.read_text(encoding="utf-8"))
    schema_record = json.loads(schema_path.read_text(encoding="utf-8"))
    assert accepted == 1
    assert rejected == {}
    assert set(example) == {"text", "structuring"}
    assert schema_record["layer_count"] == 3


def test_parse_json_object_only_accepts_object():
    assert parse_json_object('```json\n{"ok": true}\n```', "test") == {"ok": True}
    with pytest.raises(InvalidExampleError, match="expected a JSON object"):
        parse_json_object("[]", "test")


def test_load_ontology_and_build_tasks(tmp_path):
    ontology = {
        "domains": [
            {
                "id": "med",
                "code": "MED",
                "name": "Medicine",
                "subdomains": [
                    {
                        "id": "med.notes",
                        "name": "Notes",
                        "text_types": [
                            {
                                "id": "med.notes.progress_note",
                                "name": "Progress note",
                            }
                        ],
                    }
                ],
            }
        ],
        "facets": {"register": ["clinical"]},
    }
    ontology_path = tmp_path / "ontology.json"
    ontology_path.write_text(json.dumps(ontology), encoding="utf-8")

    paths, facets = load_domain_paths(ontology_path)
    tasks = build_tasks(
        paths,
        facets,
        domain_filters=["MED"],
        examples_per_text_type=2,
        languages=["English"],
        min_schema_layers=2,
        max_schema_layers=2,
        seed=42,
        num_ontology_leaves=None,
        shuffle=False,
        max_examples=None,
    )

    assert len(tasks) == 2
    assert tasks[0].domain_path.text_type_id == "med.notes.progress_note"
    assert tasks[0].facets == {"register": "clinical"}
    assert tasks[0].schema_layers == 2


def test_build_tasks_randomly_samples_leaves_before_repeating():
    paths = [
        DomainPath(
            domain_id="gen",
            domain_code="GEN",
            domain_name="General",
            subdomain_id="gen.docs",
            subdomain_name="Documents",
            text_type_id=f"gen.docs.type_{index}",
            text_type_name=f"Type {index}",
        )
        for index in range(10)
    ]

    tasks = build_tasks(
        paths,
        {},
        domain_filters=[],
        examples_per_text_type=3,
        languages=["English"],
        min_schema_layers=2,
        max_schema_layers=2,
        seed=7,
        num_ontology_leaves=4,
        shuffle=False,
        max_examples=None,
    )

    selected_ids = [task.domain_path.text_type_id for task in tasks]
    assert len(tasks) == 12
    assert len(set(selected_ids)) == 4
    assert all(selected_ids.count(text_type_id) == 3 for text_type_id in set(selected_ids))


def test_build_tasks_rejects_oversized_leaf_sample():
    path = _task().domain_path
    with pytest.raises(ValueError, match="only 1 are available"):
        build_tasks(
            [path],
            {},
            domain_filters=[],
            examples_per_text_type=1,
            languages=["English"],
            min_schema_layers=2,
            max_schema_layers=2,
            seed=42,
            num_ontology_leaves=2,
            shuffle=False,
            max_examples=None,
        )
