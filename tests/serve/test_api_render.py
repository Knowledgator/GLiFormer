import pytest

from gliformer.serve.api import APIError, render_input


def test_root_string_is_rendered_verbatim():
    rendered = render_input("Hi, I'm Tom Baker.")
    assert rendered.text == "Hi, I'm Tom Baker."
    location = rendered.locate(8, 17)
    assert (location.source, location.start, location.end, location.text) == ("", 8, 17, "Tom Baker")


def test_root_scalars_use_json_form():
    assert render_input(300).text == "300"
    assert render_input(True).text == "true"
    assert render_input(None).text == "null"


def test_markdown_example_from_contract():
    rendered = render_input({
        "ticket": "I was charged twice for order 8812.",
        "customer": {"name": "Tom Baker", "plan": "pro"},
        "orders": [{"id": "8812", "total": 300}, {"id": "8813", "total": 45}],
        "tags": ["billing", "urgent"],
    })
    assert rendered.text == (
        "# ticket\nI was charged twice for order 8812.\n\n"
        "# customer\n## name\nTom Baker\n## plan\npro\n\n"
        "# orders\n## 1\n### id\n8812\n### total\n300\n## 2\n### id\n8813\n### total\n45\n\n"
        "# tags\n- billing\n- urgent"
    )
    pointers = [leaf.pointer for leaf in rendered.leaves]
    assert pointers == [
        "/ticket", "/customer/name", "/customer/plan",
        "/orders/0/id", "/orders/0/total", "/orders/1/id", "/orders/1/total",
        "/tags/0", "/tags/1",
    ]
    start = rendered.text.index("Tom Baker")
    location = rendered.locate(start, start + 9)
    assert (location.source, location.start, location.end) == ("/customer/name", 0, 9)
    total = rendered.text.index("45")
    assert rendered.locate(total, total + 2).source == "/orders/1/total"


def test_headings_markers_and_cross_leaf_spans_have_no_location():
    rendered = render_input({"ticket": "Login broken", "tags": ["urgent"]})
    heading = rendered.text.index("ticket")
    assert rendered.locate(heading, heading + 6) is None
    marker = rendered.text.index("- ")
    assert rendered.locate(marker, marker + 3) is None
    assert rendered.locate(rendered.text.index("broken"), rendered.text.index("urgent") + 6) is None
    assert rendered.locate(3, 3) is None


def test_pointer_escaping_and_nested_depth_cap():
    deep = {"a": {"b": {"c": {"d": {"e": {"f": {"g": "leaf"}}}}}}}
    rendered = render_input({"x/y~z": "v", **deep})
    assert rendered.leaves[0].pointer == "/x~1y~0z"
    assert "####### " not in rendered.text
    assert "###### g\nleaf" in rendered.text


def test_empty_containers_render_heading_only_and_lists_of_lists_are_numbered():
    rendered = render_input({"empty": {}, "none": [], "grid": [[1, 2], [3]]})
    assert rendered.text == (
        "# empty\n\n# none\n\n# grid\n## 1\n- 1\n- 2\n## 2\n- 3"
    )
    assert [leaf.pointer for leaf in rendered.leaves] == ["/grid/0/0", "/grid/0/1", "/grid/1/0"]


def test_find_returns_every_occurrence_in_leaves():
    rendered = render_input({"a": "x y x", "b": "x"})
    assert [(loc.source, loc.start) for loc in rendered.find("x")] == [("/a", 0), ("/a", 4), ("/b", 0)]


def test_media_nodes_are_rejected_without_the_modality():
    with pytest.raises(APIError) as error:
        render_input({"photo": {"$type": "image", "data": "aGk=", "mime_type": "image/png"}})
    assert error.value.code == "UNSUPPORTED_MODALITY"
    assert error.value.path == "/input/photo"


def test_unknown_dollar_type_is_invalid():
    with pytest.raises(APIError) as error:
        render_input({"x": {"$type": "video"}})
    assert error.value.code == "INVALID_REQUEST"
    assert error.value.path == "/input/x/$type"


def test_other_dollar_keys_are_plain_data():
    rendered = render_input({"$price": 3})
    assert rendered.text == "# $price\n3"
