"""Public schema construction without a model instance."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from gliformer import BaseGLiFormer, GLiFormerSchema


def test_explicit_multitask_schema_passes_task_arguments_to_inference():
    schema = GLiFormerSchema(
        entities=["person", "organization"],
        classes=["business", "sports", "technology"],
        structures={"employee": ["name", "company"]},
    )
    captured = []

    def inference(texts, **kwargs):
        captured.append((texts, kwargs))
        return {"ner": [[]], "classification": [[]], "structuring": [{}]}

    model = SimpleNamespace(inference=inference)
    result = BaseGLiFormer.inference_from_schema(model, ["Alice joined Acme."], schema, threshold=0.7)

    assert captured == [(["Alice joined Acme."], {
        "entities": ["person", "organization"],
        "classes": ["business", "sports", "technology"],
        "structures": {"employee": {"fields": ["name", "company"], "required_fields": []}},
        "threshold": 0.7,
    })]
    assert result == {"ner": [[]], "classification": [[]], "structuring": [{}]}


def test_named_groups_and_joint_relations_are_independent_of_input_mutation():
    kwargs = {
        "entities": {"people": ["person"]},
        "classes": {"topic": ["business", "sports"]},
        "relations": {"employment": ["works_at"]},
        "joint_relations": {"staff": {"entities": ["person", "company"], "relations": ["works_at"]}},
    }
    expected = deepcopy(kwargs)
    schema = GLiFormerSchema(**kwargs)
    kwargs["entities"]["people"].append("organization")
    kwargs["joint_relations"]["staff"]["relations"].clear()

    assert schema.to_inference_kwargs() == expected
    assert GLiFormerSchema().to_inference_kwargs() == {}


def test_constructor_compiles_nested_types_and_formats_inference_output():
    schema = GLiFormerSchema(structures={
        "employee": {"name": "string", "age": "integer", "jobs": [{"years": "integer"}]},
    })
    assert schema.requires_multi_level
    wire = schema.to_inference_kwargs()
    model = SimpleNamespace(inference=lambda texts, **kwargs: {
        "structuring": [{"employee": [{"name": "Alice", "age": "30", "jobs": [{"years": "4"}]}]}],
    })

    result = BaseGLiFormer.inference_from_schema(model, ["Alice is 30."], schema)

    assert result["structuring"] == [{"employee": [{"name": "Alice", "age": 30, "jobs": [{"years": 4}]}]}]
    assert schema.to_inference_kwargs() == wire


def test_constructor_accepts_pydantic_models():
    from pydantic import BaseModel

    class Employee(BaseModel):
        name: str
        age: int

    schema = GLiFormerSchema(structures={"employee": Employee})
    formatter = schema.build_output_formatter()
    assert formatter.format({"employee": [{"name": "Alice", "age": "30"}]}) == {
        "employee": [{"name": "Alice", "age": 30}],
    }


@pytest.mark.parametrize("kwargs", [
    {"entities": "person"},
    {"classes": {"topic": "business"}},
    {"relations": [123]},
    {"joint_relations": {"staff": {"entities": ["person"]}}},
    {"structures": ["name"]},
])
def test_constructor_rejects_invalid_task_shapes(kwargs):
    with pytest.raises(TypeError):
        GLiFormerSchema(**kwargs)
