import pytest

from gliformer.serve.api import APIError
from gliformer.serve.api.labels import normalize_labels, render_label, validate_template


def test_five_label_forms():
    assert normalize_labels(None, "/l") is None
    assert [label.id for label in normalize_labels("person", "/l")] == ["person"]
    assert [label.text for label in normalize_labels(["a", "b"], "/l")] == ["a", "b"]
    described = normalize_labels({"person": "A human being's name"}, "/l")[0]
    assert (described.text, described.description) == ("person", "A human being's name")
    defined = normalize_labels(
        {"person": {"title": "Person", "description": "Names", "metadata": {"pii": True}}}, "/l"
    )[0]
    assert defined.text == "Person"
    assert defined.definition() == {"title": "Person", "description": "Names", "metadata": {"pii": True}}


def test_template_controls_description_rendering():
    labels = normalize_labels(
        {"a": {"title": "Alpha", "description": "First"}, "b": {"title": "Beta"}},
        "/l",
        template="{title}: {description}",
    )
    assert [label.text for label in labels] == ["Alpha: First", "Beta"]
    assert render_label("id", None, None, "{title}") == "id"


@pytest.mark.parametrize("template", ["{name}", "{description}", "{title"])
def test_bad_templates_are_rejected(template):
    with pytest.raises(ValueError):
        validate_template(template)


def test_duplicate_rendered_text_is_rejected():
    with pytest.raises(APIError) as error:
        normalize_labels({"a": {"title": "Same"}, "b": {"title": "Same"}}, "/tasks/t/labels")
    assert error.value.code == "INVALID_REQUEST"
    assert error.value.path == "/tasks/t/labels/b"


def test_examples_and_aliases_need_model_support():
    with pytest.raises(APIError) as error:
        normalize_labels({"a": {"aliases": ["x"]}}, "/l")
    assert error.value.code == "UNSUPPORTED_FIELD"
    assert normalize_labels({"a": {"aliases": ["x"]}}, "/l", supported_keys=frozenset({"aliases"}), extra_keys=frozenset({"aliases"}))


def test_unknown_definition_keys_and_bad_types_are_invalid():
    for spec in ({"a": {"colour": "red"}}, {"a": 3}, [1], 7, {"a": {"title": 3}}, {"": "x"}):
        with pytest.raises(APIError) as error:
            normalize_labels(spec, "/l")
        assert error.value.code == "INVALID_REQUEST"
