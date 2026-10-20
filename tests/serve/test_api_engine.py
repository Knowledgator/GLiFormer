import pytest

from gliformer.serve.api import APIError

from .fakes import FakeGLiFormer, body, make_server


def analyze(server, *args, **kwargs):
    return server.analyze(body(*args, **kwargs))["results"]


def error_of(server, request):
    with pytest.raises(APIError) as error:
        server.analyze(request)
    return error.value


# -- classification ---------------------------------------------------------------


def test_single_label_returns_top_label_without_threshold():
    server, model = make_server()
    result = analyze(server, "Refund me.", {
        "intent": {"type": "classify", "instruction": "Main intent?", "labels": {
            "refund": {"title": "Refund request"}, "tech": {"title": "Technical support"},
        }},
    })["intent"]
    assert result == {
        "type": "classify", "value": "refund",
        "scores": {"refund": 0.9, "tech": 0.2}, "threshold": None,
    }
    row = model.calls[0][0][0]
    assert row["classes"] == {"c0": ["Refund request", "Technical support"]}
    assert row["group_prompts"]["c0"] == {"name": None, "description": "Main intent?"}
    assert row["decoder_kwargs"]["task_controls"]["classification"] == {"threshold": -1.0, "multi_label": True}


def test_single_label_with_unmet_threshold_returns_null():
    server, _ = make_server()
    result = analyze(server, "x", {"t": {"type": "classify", "labels": ["a", "b"], "threshold": 0.95}})["t"]
    assert result["value"] is None
    assert result["threshold"] == 0.95


def test_multi_label_uses_deployment_default_and_may_be_empty():
    server, model = make_server()
    model.class_scores = [0.9, 0.6, 0.1]
    result = analyze(server, "x", {"t": {"type": "classify", "multi_label": True, "labels": ["a", "b", "c"]}})["t"]
    assert result["value"] == ["a", "b"]
    assert result["threshold"] == 0.5
    model.class_scores = [0.1]
    empty = analyze(server, "x", {"t": {"type": "classify", "multi_label": True, "labels": ["a"], "threshold": 0.3}})["t"]
    assert empty["value"] == [] and empty["threshold"] == 0.3


def test_boolean_is_single_label_over_true_false():
    server, model = make_server()
    result = analyze(server, "Refund me.", {"r": {"type": "boolean", "instruction": "Refund asked?"}})["r"]
    assert result == {
        "type": "boolean", "value": True, "truth_score": 0.9,
        "scores": {"true": 0.9, "false": 0.2}, "threshold": None,
    }
    labelled = {"type": "boolean", "labels": {"true": {"title": "refund requested"}}}
    analyze(server, "x", {"r": labelled})
    assert model.calls[-1][0][0]["classes"]["c0"] == ["refund requested", "false"]
    model.class_scores = [0.2, 0.7]
    assert analyze(server, "x", {"r": {"type": "boolean"}})["r"]["value"] is False


def test_ordinal_score_follows_order_and_labels():
    server, model = make_server()
    model.class_scores = [0.1, 0.3, 0.8]
    result = analyze(server, "x", {"u": {
        "type": "score", "labels": ["low", "medium", "high"], "order": ["low", "medium", "high"],
    }})["u"]
    assert result["value"] == "high"
    assert result["scores"] == {"low": 0.1, "medium": 0.3, "high": 0.8}


def test_label_definitions_are_returned_on_request():
    server, _ = make_server()
    results = server.analyze(body("x", {"t": {"type": "classify", "labels": {
        "a": {"title": "A", "metadata": {"id": 1}}, "b": "Bee",
    }}}, include_label_definitions=True))["results"]
    assert results["t"]["label_definitions"] == {
        "a": {"title": "A", "metadata": {"id": 1}},
        "b": {"description": "Bee"},
    }


# -- extraction ---------------------------------------------------------------------


def test_spans_report_leaf_source_and_drop_headings():
    server, _ = make_server()
    result = analyze(server, {"Ticket": "Login broken", "customer": {"name": "Tom Baker"}}, {
        "p": {"type": "span", "labels": {"person": {"title": "Person"}}},
    })["p"]
    # "Ticket" is a heading (no leaf); "Login" and "Tom Baker" are leaves.
    assert result["value"] == [
        {"label": "person", "text": "Login", "start": 0, "end": 5, "source": "/Ticket", "score": 0.9},
        {"label": "person", "text": "Tom Baker", "start": 0, "end": 9, "source": "/customer/name", "score": 0.9},
    ]
    assert result["threshold"] == 0.5


def test_span_tasks_are_routed_by_group_and_filtered_by_threshold():
    server, model = make_server()
    results = analyze(server, "Alice met Bob.", {
        "a": {"type": "span", "labels": ["person"]},
        "b": {"type": "span", "labels": ["name"], "threshold": 0.95},
        "open": {"type": "span", "instruction": "Who?"},
    })
    assert [span["text"] for span in results["a"]["value"]] == ["Alice", "Bob"]
    assert results["b"]["value"] == []
    assert "label" not in results["open"]["value"][0]
    row = model.calls[0][0][0]
    assert row["entities"] == {"e0": ["person"], "e1": ["name"], "e2": ["answer"]}
    assert row["threshold"] == 0.5


def test_non_flat_spans_and_relations_get_their_own_rows():
    server, model = make_server()
    results = analyze(server, "Alice founded Acme.", {
        "c": {"type": "classify", "labels": ["x", "y"]},
        "s": {"type": "span", "labels": ["person"]},
        "loose": {"type": "span", "labels": ["thing"], "flat": False},
        "r": {"type": "relation", "labels": {"founded": {"title": "founded"}}, "entity_labels": ["person", "organization"]},
    })
    rows = model.calls[0][0]
    assert [sorted(key for key in ("classes", "entities", "joint_relations") if key in row) for row in rows] == [
        ["classes", "entities"], ["entities"], ["joint_relations"],
    ]
    assert rows[1]["flat_ner"] is False
    relation = results["r"]["value"][0]
    assert relation["relation"] == "founded"
    assert relation["head"] == {"label": "person", "text": "Alice", "start": 0, "end": 5, "source": ""}
    assert relation["tail"]["text"] == "Acme"
    assert results["loose"]["value"][0]["label"] == "thing"


def test_relation_without_entity_labels_omits_entity_label():
    server, _ = make_server()
    value = analyze(server, "Alice founded Acme.", {"r": {"type": "relation", "labels": ["founded"]}})["r"]["value"]
    assert "label" not in value[0]["head"]


def test_long_input_runs_spans_over_windows_and_rejects_whole_input_tasks():
    server, model = make_server(max_model_len=30)
    names = [f"Nam{letter}" for letter in "abcdefghijklmnopqrst"]
    text = " ".join(f"{name} word word" for name in names)
    results = analyze(server, text, {"p": {"type": "span", "labels": ["person"]}})
    rows = model.calls[0][0]
    assert len(rows) > 1
    found = [span["text"] for span in results["p"]["value"]]
    assert found == names
    assert all(text[span["start"]:span["end"]] == span["text"] for span in results["p"]["value"])
    error = error_of(server, body(text, {"c": {"type": "classify", "labels": ["a"]}}))
    assert error.code == "INPUT_TOO_LONG" and error.status == 413


# -- json ---------------------------------------------------------------------------


def test_json_labels_map_titles_back_to_ids_and_parse_formats():
    server, model = make_server()
    model.struct_values = {
        "Invoice number": "INV-42",
        "Issue date": "October 1, 2026",
        "Total": "$1,234.50",
        "Due": "TBD",
        "Paid": "already paid",
    }
    text = "Invoice INV-42, issued October 1, 2026. Total $1,234.50. Due TBD. Already paid."
    result = server.analyze(body(text, {"inv": {"type": "json", "instruction": "Extract.", "labels": {
        "invoice_id": {"title": "Invoice number"},
        "issued": {"title": "Issue date", "type": "text", "format": "date"},
        "total": {"title": "Total", "type": "text", "format": "number"},
        "due": {"title": "Due", "type": "text", "format": "date"},
        "paid": {"title": "Paid", "type": "boolean"},
        "missing": {"title": "Missing"},
    }}}, include_evidence=True))["results"]["inv"]
    assert result["value"] == {
        "invoice_id": "INV-42", "issued": "2026-10-01", "total": 1234.5,
        "due": None, "paid": True, "missing": None,
    }
    assert result["unparsed"] == {"/due": {"text": "TBD", "start": 61, "end": 64, "source": ""}}
    assert result["evidence"]["/issued"] == [{"text": "October 1, 2026", "start": 23, "end": 38, "source": ""}]
    row = model.calls[0][0][0]
    assert row["structures"] == {"$root": {
        "Invoice number": "str", "Issue date": "str", "Total": "str",
        "Due": "str", "Paid": "bool", "Missing": "str",
    }}
    assert row["prompt"] == "Extract."
    assert "classes" not in row


def test_typed_checkpoint_gets_format_descriptors_and_predicted_fields():
    server, model = make_server(FakeGLiFormer(typed=True))
    model.struct_values = {"Total": 300.0, "Kind": "Flat", "Rooms": 2.0, "Tags": ["a", "Other"]}
    result = analyze(server, "x", {"j": {"type": "json", "labels": {
        "total": {"title": "Total", "type": "text", "format": "integer"},
        "kind": {"title": "Kind", "type": "options", "options": {"flat": {"title": "Flat"}, "house": {"title": "House"}}},
        "rooms": {"title": "Rooms", "type": "integer", "minimum": 0, "maximum": 10},
        "tags": {"title": "Tags", "type": "options", "options": ["a", "b"], "multi": True},
    }}})["j"]
    assert result["value"] == {"total": 300, "kind": "flat", "rooms": 2, "tags": ["a"]}
    structures = model.calls[0][0][0]["structures"]["$root"]
    assert structures["Total"] == {"$type": "text", "$format": "number", "$integer": True}
    assert structures["Kind"] == {"$type": "class", "$options": ["Flat", "House"], "$multi": False}
    assert structures["Rooms"] == {"$type": "number", "$integer": True, "$min": 0, "$max": 10}


def test_predicted_types_need_typed_heads():
    server, _ = make_server()
    error = error_of(server, body("x", {"j": {"type": "json", "labels": {"p": {"type": "estimate", "minimum": 1, "maximum": 2}}}}))
    assert error.code == "UNSUPPORTED_LABEL_TYPE"
    assert error.path == "/tasks/j/labels/p/type"
    assert error.details["supported"] == ["text", "boolean"]


@pytest.mark.parametrize("field, path", [
    ({"type": "number", "format": "date"}, "/tasks/j/labels/f/format"),
    ({"type": "text", "format": "weekday"}, "/tasks/j/labels/f/format"),
    ({"type": "text", "minimum": 3}, "/tasks/j/labels/f/minimum"),
    ({"type": "boolean", "format": "date"}, "/tasks/j/labels/f/format"),
    ({"type": "colour"}, "/tasks/j/labels/f/type"),
])
def test_invalid_json_field_properties(field, path):
    server, _ = make_server(FakeGLiFormer(typed=field.get("type") == "number"))
    error = error_of(server, body("x", {"j": {"type": "json", "labels": {"f": field}}}))
    assert error.code == "INVALID_REQUEST"
    assert error.path == path


def test_json_schema_maps_strings_numbers_and_records():
    server, model = make_server()
    model.struct_values = {
        "company": "Acme",
        "founded": "1999-02-01",
        "employees": [{"name": "Alice", "role": "engineer", "since": "2021"}, {"name": None, "role": None, "since": None}],
        "tags": ["billing"],
    }
    schema = {
        "type": "object",
        "properties": {
            "company": {"type": "string"},
            "founded": {"type": ["string", "null"], "format": "date"},
            "employees": {"type": "array", "items": {"$ref": "#/$defs/Employee"}},
            "tags": {"type": "array", "items": {"type": "string"}},
            "count": {"type": "integer"},
        },
        "$defs": {"Employee": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "role": {"type": "string", "enum": ["engineer", "designer"]},
                "since": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            },
            "required": ["name"],
        }},
    }
    server_typed, typed_model = make_server(FakeGLiFormer(typed=True))
    typed_model.struct_values = {"employees": [{"name": "Alice", "role": "engineer", "since": 2021.0}]}
    typed = analyze(server_typed, "x", {"o": {"type": "json", "schema": schema}})["o"]["value"]
    assert typed["employees"] == [{"name": "Alice", "role": "engineer", "since": 2021}]
    template = typed_model.calls[0][0][0]["structures"]["$root"]
    assert template["employees"] == [{
        "!name": "str",
        "role": {"$type": "class", "$options": ["engineer", "designer"], "$multi": False},
        "since": {"$type": "text", "$format": "number", "$integer": True},
    }]
    assert template["founded"] == {"$type": "text", "$format": "date"}
    # Without typed heads the enum needs the class head.
    error = error_of(server, body("x", {"o": {"type": "json", "schema": schema}}))
    assert error.code == "UNSUPPORTED_LABEL_TYPE"
    assert error.path == "/tasks/o/schema/$defs/Employee/properties/role/type" or error.path.endswith("/role/type")


def test_json_schema_without_enum_runs_on_plain_checkpoint():
    server, model = make_server()
    model.struct_values = {"company": "Acme", "employees": [{"name": "Alice"}], "count": "12 people"}
    value = analyze(server, "x", {"o": {"type": "json", "schema": {
        "type": "object",
        "properties": {
            "company": {"type": "string"},
            "employees": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}}}},
            "count": {"type": "integer"},
            "flag": {"type": "boolean"},
        },
    }}})["o"]["value"]
    assert value == {"company": "Acme", "employees": [{"name": "Alice"}], "count": 12, "flag": None}


def test_json_task_needs_exactly_one_declaration():
    server, _ = make_server()
    for task in ({"type": "json"}, {"type": "json", "labels": ["a"], "schema": {"type": "object"}}):
        assert error_of(server, body("x", {"j": task})).code == "INVALID_REQUEST"


def test_score_with_scale_needs_typed_heads_and_snaps_to_step():
    server, _ = make_server()
    task = {"type": "score", "instruction": "Urgency?", "scale": {"min": 1, "max": 5, "step": 1}}
    assert error_of(server, body("x", {"u": task})).code == "UNSUPPORTED_LABEL_TYPE"
    typed_server, model = make_server(FakeGLiFormer(typed=True))
    model.struct_values = {"score": 3.7}
    assert analyze(typed_server, "x", {"u": task})["u"] == {"type": "score", "value": 4, "scale": {"min": 1, "max": 5, "step": 1}}


# -- retrieval ----------------------------------------------------------------------


def test_embedding_and_rank():
    server, _ = make_server()
    results = analyze(server, "Which company introduced a smartphone?", {
        "v": {"type": "embedding"},
        "raw": {"type": "embedding", "normalize": False},
        "search": {"type": "rank", "top_k": 1, "candidates": [
            {"id": "fruit", "input": "Apples grow on fruit trees."},
            {"id": "phone", "input": "Apple develops the iPhone smartphone."},
        ]},
    })
    vector = results["v"]["value"]
    assert results["v"]["dimensions"] == 3 and results["v"]["normalized"] is True
    assert abs(sum(value * value for value in vector) - 1.0) < 1e-6
    assert results["raw"]["normalized"] is False
    assert [item["id"] for item in results["search"]["value"]] == ["phone"]
    assert results["search"]["value"][0]["rank"] == 1


def test_rank_rejects_duplicate_candidates():
    server, _ = make_server()
    task = {"type": "rank", "candidates": [{"id": "a", "input": "x"}, {"id": "a", "input": "y"}]}
    assert error_of(server, body("q", {"r": task})).path == "/tasks/r/candidates/1/id"


# -- validation -----------------------------------------------------------------------


@pytest.mark.parametrize("request_body, code, path", [
    ({"model": "other", "input": "x", "tasks": {"t": {"type": "span"}}}, "MODEL_NOT_FOUND", "/model"),
    ({"model": "test/gliformer", "tasks": {"t": {"type": "span"}}}, "INVALID_REQUEST", "/input"),
    ({"model": "test/gliformer", "input": "x", "tasks": {}}, "INVALID_REQUEST", "/tasks"),
    ({"model": "test/gliformer", "input": "x", "tasks": {"t": {"type": "span"}}, "extra": 1}, "INVALID_REQUEST", "/extra"),
    ({"model": "test/gliformer", "input": "x", "tasks": {"t": {"type": "translate"}}}, "INVALID_REQUEST", "/tasks/t/type"),
    ({"model": "test/gliformer", "input": "x", "tasks": {"t": {"type": "link"}}}, "UNSUPPORTED_TASK_TYPE", "/tasks/t/type"),
    ({"model": "test/gliformer", "input": "x", "tasks": {"t": {"type": "box", "labels": ["a"]}}}, "UNSUPPORTED_TASK_TYPE", "/tasks/t/type"),
    ({"model": "test/gliformer", "input": "x", "tasks": {"t": {"type": "span", "multi_label": True}}}, "INVALID_REQUEST", "/tasks/t/multi_label"),
    ({"model": "test/gliformer", "input": "x", "tasks": {"t": {"type": "classify", "labels": []}}}, "INVALID_REQUEST", "/tasks/t/labels"),
    ({"model": "test/gliformer", "input": "x", "tasks": {"t": {"type": "classify", "labels": ["a"], "threshold": 2}}}, "INVALID_REQUEST", "/tasks/t/threshold"),
    ({"model": "test/gliformer", "input": "x", "tasks": {"t": {"type": "boolean", "labels": ["maybe"]}}}, "INVALID_REQUEST", "/tasks/t/labels/maybe"),
    ({"model": "test/gliformer", "input": "x", "tasks": {"t": {"type": "score", "labels": ["a", "b"], "order": ["a"]}}}, "INVALID_REQUEST", "/tasks/t/order"),
    ({"model": "test/gliformer", "input": "x", "tasks": {"t": {"type": "relation"}}}, "INVALID_REQUEST", "/tasks/t/labels"),
    ({"model": "test/gliformer", "input": "x", "tasks": {"t": {"type": "embedding"}}, "options": {"verbose": True}}, "INVALID_REQUEST", "/options/verbose"),
    ({"model": "test/gliformer", "input": "x", "tasks": {"t": {"type": "embedding"}}, "options": {"reference_date": "soon"}}, "INVALID_REQUEST", "/options/reference_date"),
    ({"model": "test/gliformer", "input": "x", "tasks": {"t": {"type": "embedding"}}, "options": {"adapter_id": "custom"}}, "UNKNOWN_ADAPTER", "/options/adapter_id"),
])
def test_validation_errors_have_codes_and_paths(request_body, code, path):
    server, model = make_server()
    error = error_of(server, request_body)
    assert (error.code, error.path) == (code, path)
    assert model.calls == []


def test_validation_is_atomic():
    server, model = make_server()
    error_of(server, body("x", {
        "ok": {"type": "classify", "labels": ["a"]},
        "bad": {"type": "classify", "labels": ["a", "a"]},
    }))
    assert model.calls == []


def test_open_relex_head_rejects_entity_labels():
    server, _ = make_server(FakeGLiFormer(relation_head="open_relex"))
    error = error_of(server, body("x", {"r": {"type": "relation", "labels": ["a"], "entity_labels": ["b"]}}))
    assert error.code == "UNSUPPORTED_FIELD"


def test_task_limit_and_label_limit():
    server, _ = make_server(max_tasks=1, max_labels=2)
    assert error_of(server, body("x", {"a": {"type": "embedding"}, "b": {"type": "embedding"}})).path == "/tasks"
    assert error_of(server, body("x", {"a": {"type": "classify", "labels": ["1", "2", "3"]}})).path == "/tasks/a/labels"


def test_input_char_limit():
    server, _ = make_server(max_input_chars=5)
    error = error_of(server, body("too long", {"a": {"type": "span"}}))
    assert error.code == "INPUT_TOO_LONG"


def test_reference_date_and_usage_are_passed_through():
    server, model = make_server()
    response = server.analyze(body("x", {"j": {"type": "json", "labels": ["a"]}}, reference_date="2026-11-01", include_usage=True))
    assert model.calls[0][0][0]["reference_date"] == "2026-11-01"
    assert response["usage"]["rows"] == 1
    assert response["model"] == "test/gliformer"


# -- batching -------------------------------------------------------------------------


def test_requests_share_one_model_call_and_fail_independently():
    server, model = make_server()
    outcomes = server.analyze_many([
        body("Alice", {"p": {"type": "span", "labels": ["person"]}}),
        body("x", {"bad": {"type": "classify"}}),
        body("Bob", {"c": {"type": "classify", "labels": ["a"]}}),
    ])
    assert isinstance(outcomes[1], APIError)
    assert outcomes[0]["results"]["p"]["value"][0]["text"] == "Alice"
    assert outcomes[2]["results"]["c"]["value"] == "a"
    assert len(model.calls) == 1
    assert [row["text"] for row in model.calls[0][0]] == ["Alice", "Bob"]


def test_execution_failure_is_isolated_per_request(monkeypatch):
    server, model = make_server()
    original = model.inference_requests

    def flaky(requests, **kwargs):
        if any(row["text"] == "boom" for row in requests):
            raise ValueError("processor rejected the schema")
        return original(requests, **kwargs)

    monkeypatch.setattr(model, "inference_requests", flaky)
    outcomes = server.analyze_many([
        body("boom", {"c": {"type": "classify", "labels": ["a"]}}),
        body("fine", {"c": {"type": "classify", "labels": ["a"]}}),
    ])
    assert isinstance(outcomes[0], APIError) and outcomes[0].code == "INTERNAL"
    assert outcomes[1]["results"]["c"]["value"] == "a"


def test_batch_endpoint_keeps_ids_order_and_item_errors():
    server, _ = make_server()
    response = server.analyze_batch({
        "model": "test/gliformer",
        "options": {"include_usage": True},
        "items": [
            {"id": "t1", "input": "Alice", "tasks": {"p": {"type": "span", "labels": ["person"]}}},
            {"id": "t2", "input": "x", "tasks": {"p": {"type": "span", "labels": ["a", "a"]}}},
            {"input": "y", "tasks": {"c": {"type": "classify", "labels": ["a"]}}, "options": {"include_usage": False}},
        ],
    })
    items = response["items"]
    assert [item.get("id") for item in items] == ["t1", "t2", None]
    assert items[0]["results"]["p"]["value"][0]["text"] == "Alice" and "usage" in items[0]
    assert items[1]["error"]["path"] == "/items/1/tasks/p/labels/1"
    assert "usage" not in items[2]


def test_batch_envelope_errors_raise():
    server, _ = make_server(max_batch_items=1)
    with pytest.raises(APIError) as error:
        server.analyze_batch({"model": "test/gliformer", "items": [{}, {}]})
    assert error.value.path == "/items"
    with pytest.raises(APIError):
        server.analyze_batch({"model": "test/gliformer", "items": [{"id": "a"}, {"id": "a"}]})


def test_capabilities_reflect_heads():
    server, _ = make_server(model_aliases=["alias"], label_template="{title}: {description}")
    capabilities = server.capabilities()
    assert capabilities["task_types"] == [
        "classify", "boolean", "score", "span", "relation", "json", "embedding", "rank",
    ]
    assert capabilities["json_field_types"] == ["text", "boolean"]
    assert capabilities["label_template"] == "{title}: {description}"
    assert capabilities["limits"]["max_labels_per_task"] is None
    assert server.capabilities("alias")["model"] == "test/gliformer"
    typed, _ = make_server(FakeGLiFormer(typed=True, embedding=False))
    assert "estimate" in typed.capabilities()["json_field_types"]
    assert "rank" not in typed.capabilities()["task_types"]


def test_label_template_controls_rendered_labels():
    server, model = make_server(label_template="{title}: {description}")
    analyze(server, "x", {"c": {"type": "classify", "labels": {"a": {"title": "A", "description": "first"}, "b": "second"}}})
    assert model.calls[0][0][0]["classes"]["c0"] == ["A: first", "b: second"]


def test_render_task_names_option():
    server, model = make_server(render_task_names=True)
    analyze(server, "x", {"intent": {"type": "classify", "labels": ["a"]}})
    assert model.calls[0][0][0]["group_prompts"]["c0"]["name"] == "intent"
