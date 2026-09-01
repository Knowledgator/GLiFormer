import json

import pytest

from data_prod.generate_joint_relex import (
    AnnotationGraph,
    EntityType,
    GraphEntity,
    GraphRelation,
    JointGenerationResult,
    RelationType,
    TypeCatalog,
    ground_and_format_graph,
    run_generation,
    validate_graph,
    validate_type_catalog,
)
from data_prod.generate_multilevel_text2json import DomainPath, InvalidExampleError


def _catalog() -> TypeCatalog:
    return TypeCatalog(
        entity_types=(
            EntityType("Person", "A named person."),
            EntityType("Organization", "A named organization."),
        ),
        relation_types=(
            RelationType(
                "works for",
                "The person is employed by the organization.",
                ("Person",),
                ("Organization",),
            ),
        ),
    )


def _threshold_graph(entity_count: int) -> tuple[str, AnnotationGraph]:
    entities = [GraphEntity("Alice", "Person"), GraphEntity("Acme", "Organization")]
    entities.extend(GraphEntity(f"Person{index}", "Person") for index in range(entity_count - 3))
    entities.append(GraphEntity("Absent Ghost", "Person"))
    text = " ".join(entity.text for entity in entities[:-1])
    graph = AnnotationGraph(
        entities=tuple(entities),
        relations=(GraphRelation(0, "works for", 1),),
    )
    return text, graph


def _task():
    from data_prod.generate_joint_relex import JointGenerationTask

    return JointGenerationTask(
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
        entity_type_count=2,
        relation_type_count=1,
    )


def test_validate_type_catalog_canonicalizes_endpoint_names():
    payload = {
        "entity_types": [
            {"name": "Person", "description": "A person."},
            {"name": "Organization", "description": "An organization."},
        ],
        "relation_types": [
            {
                "name": "works for",
                "description": "Employment directed from person to organization.",
                "head_types": ["person"],
                "tail_types": ["ORGANIZATION"],
            }
        ],
    }

    catalog = validate_type_catalog(payload, expected_entity_types=2, expected_relation_types=1)

    assert catalog.relation_types[0].head_types == ("Person",)
    assert catalog.relation_types[0].tail_types == ("Organization",)


def test_validate_graph_checks_relation_endpoint_types():
    payload = {
        "entities": [
            {"text": "Acme", "label": "Organization"},
            {"text": "Alice", "label": "Person"},
        ],
        "relations": [{"head": 0, "label": "works for", "tail": 1}],
    }

    with pytest.raises(InvalidExampleError, match="head label"):
        validate_graph(payload, _catalog(), stage="validation")


def test_exactly_five_percent_hallucinated_entities_are_filtered():
    text, graph = _threshold_graph(20)

    grounded = ground_and_format_graph(
        text,
        graph,
        _catalog(),
        max_hallucinated_entity_ratio=0.05,
        min_type_coverage=0.6,
    )

    assert len(grounded.ner) == 19
    assert grounded.hallucinated_entities == ("Absent Ghost",)
    assert grounded.relations == ((0, "works for", 1),)
    assert all(text[item["start"] : item["end"]] == item["text"] for item in grounded.ner)


def test_more_than_five_percent_hallucinated_entities_rejects_example():
    text, graph = _threshold_graph(19)

    with pytest.raises(InvalidExampleError, match="5.26% > 5.00%"):
        ground_and_format_graph(
            text,
            graph,
            _catalog(),
            max_hallucinated_entity_ratio=0.05,
            min_type_coverage=0.6,
        )


def test_filtering_remaps_relation_indices():
    entities = [
        GraphEntity("Spare", "Person"),
        GraphEntity("Absent Ghost", "Organization"),
        GraphEntity("Alice", "Person"),
        GraphEntity("Acme", "Organization"),
    ]
    entities.extend(GraphEntity(f"Extra{index}", "Person") for index in range(16))
    text = " ".join(entity.text for index, entity in enumerate(entities) if index != 1)
    graph = AnnotationGraph(tuple(entities), (GraphRelation(2, "works for", 3),))

    grounded = ground_and_format_graph(
        text,
        graph,
        _catalog(),
        max_hallucinated_entity_ratio=0.05,
        min_type_coverage=0.6,
    )

    assert grounded.relations == ((1, "works for", 2),)
    assert grounded.ner[1]["text"] == "Alice"
    assert grounded.ner[2]["text"] == "Acme"


def test_run_generation_writes_joint_glinext_shape(tmp_path):
    task = _task()
    example = {
        "text": "Alice works for Acme.",
        "extraction": [
            {
                "name": None,
                "ner": [
                    {"text": "Alice", "start": 0, "end": 5, "label": "Person"},
                    {
                        "text": "Acme",
                        "start": 16,
                        "end": 20,
                        "label": "Organization",
                    },
                ],
                "relations": [[0, "works for", 1]],
            }
        ],
    }
    metadata = {"task_index": 0, "type_catalog": _catalog().as_dict()}

    class FakeGenerator:
        def process(self, requested_task):
            return JointGenerationResult(
                task=requested_task,
                example=example,
                metadata_record=metadata,
            )

    output_path = tmp_path / "joint.jsonl"
    metadata_path = tmp_path / "metadata.jsonl"
    accepted, rejected = run_generation(
        [task],
        FakeGenerator(),
        output_path=output_path,
        metadata_output_path=metadata_path,
        append=False,
        workers=1,
        progress_every=1,
    )

    assert accepted == 1
    assert rejected == {}
    assert json.loads(output_path.read_text(encoding="utf-8")) == example
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == metadata
