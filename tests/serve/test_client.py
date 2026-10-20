import asyncio
import io
import json
import urllib.error

import pytest

from gliformer.serve.client import (
    AnalyzeResult,
    GLiFormerAPIError,
    GLiFormerClient,
    GLiFormerClientError,
)


class Recorder:
    """Stand-in for ``GLiFormerClient._request``."""

    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses or {}

    def __call__(self, method, suffix, payload=None):
        self.calls.append((method, suffix, payload))
        if suffix == "/models":
            return {"data": [{"id": "served/model", "aliases": []}]}
        response = self.responses.get(suffix)
        if callable(response):
            return response(payload)
        if response is not None:
            return response
        tasks = (payload or {}).get("tasks", {})
        return {
            "model": "served/model",
            "results": {name: {"type": task["type"], "value": f"{name}-value"} for name, task in tasks.items()},
        }


def make_client(monkeypatch, **responses):
    client = GLiFormerClient("http://server:8000/")
    recorder = Recorder(responses)
    monkeypatch.setattr(client, "_request", recorder)
    return client, recorder


def test_url_and_model_lookup(monkeypatch):
    client, recorder = make_client(monkeypatch)
    assert client.url == "http://server:8000/v1"
    assert client.model == "served/model"
    assert client.model == "served/model"
    assert [call[1] for call in recorder.calls] == ["/models"]
    assert GLiFormerClient(model="given").model == "given"


def test_analyze_builds_body_and_wraps_results(monkeypatch):
    client, recorder = make_client(monkeypatch)
    result = client.analyze(
        {"ticket": "Refund"},
        tasks={"intent": {"type": "classify", "labels": ["refund"]}},
        options={"include_usage": True},
    )
    method, suffix, payload = recorder.calls[-1]
    assert (method, suffix) == ("POST", "/analyze")
    assert payload == {
        "model": "served/model",
        "input": {"ticket": "Refund"},
        "tasks": {"intent": {"type": "classify", "labels": ["refund"]}},
        "options": {"include_usage": True},
    }
    assert isinstance(result, AnalyzeResult)
    assert result["intent"].value == "intent-value"
    assert result.model == "served/model"
    with pytest.raises(AttributeError):
        _ = result["intent"].missing


@pytest.mark.parametrize("method, args, kwargs, task", [
    ("classify", ("x", ["a", "b"]), {"multi_label": True}, {"type": "classify", "labels": ["a", "b"], "multi_label": True}),
    ("judge", ("x",), {"instruction": "Refund?"}, {"type": "boolean", "instruction": "Refund?"}),
    ("extract_spans", ("x",), {"labels": "person", "flat": False}, {"type": "span", "labels": "person", "flat": False}),
    ("extract_relations", ("x", ["founded"]), {"entity_labels": ["person"]}, {"type": "relation", "labels": ["founded"], "entity_labels": ["person"]}),
    ("extract_json", ("x",), {"labels": {"vendor": "Vendor"}}, {"type": "json", "labels": {"vendor": "Vendor"}}),
    ("rank", ("q", [{"id": "a", "input": "x"}]), {"top_k": 1}, {"type": "rank", "candidates": [{"id": "a", "input": "x"}], "top_k": 1}),
])
def test_wrappers_compile_to_one_task(monkeypatch, method, args, kwargs, task):
    client, recorder = make_client(monkeypatch)
    result = getattr(client, method)(*args, **kwargs)
    assert recorder.calls[-1][2]["tasks"] == {"result": task}
    assert result.value == "result-value"


def test_embed_returns_vector(monkeypatch):
    client, recorder = make_client(monkeypatch, **{"/analyze": {"results": {"result": {"type": "embedding", "value": [0.6, 0.8]}}}})
    assert client.embed("x", normalize=False) == [0.6, 0.8]
    assert recorder.calls[-1][2]["tasks"]["result"] == {"type": "embedding", "normalize": False}


def test_extract_json_accepts_pydantic_models(monkeypatch):
    pydantic = pytest.importorskip("pydantic")

    class Invoice(pydantic.BaseModel):
        vendor: str

    client, recorder = make_client(monkeypatch)
    client.extract_json("x", schema=Invoice)
    schema = recorder.calls[-1][2]["tasks"]["result"]["schema"]
    assert schema["properties"]["vendor"]["type"] == "string"


def test_batch_returns_results_and_item_errors(monkeypatch):
    client, recorder = make_client(monkeypatch, **{"/analyze/batch": {
        "model": "served/model",
        "items": [
            {"id": "a", "results": {"t": {"type": "classify", "value": "x"}}},
            {"id": "b", "error": {"code": "INVALID_REQUEST", "message": "bad", "path": "/items/1/tasks"}},
        ],
    }})
    outcomes = client.analyze_batch([{"id": "a", "input": "x", "tasks": {}}], options={"include_usage": True})
    assert recorder.calls[-1][2]["options"] == {"include_usage": True}
    assert outcomes[0].id == "a" and outcomes[0]["t"].value == "x"
    assert isinstance(outcomes[1], GLiFormerAPIError)
    assert (outcomes[1].id, outcomes[1].code, outcomes[1].path) == ("b", "INVALID_REQUEST", "/items/1/tasks")


def test_async_variants(monkeypatch):
    client, _ = make_client(monkeypatch)

    async def run():
        return await asyncio.gather(*(
            client.analyze_async("x", {"t": {"type": "span"}}) for _ in range(3)
        ))

    assert [result["t"].value for result in asyncio.run(run())] == ["t-value"] * 3


def test_capabilities_quotes_model_ids(monkeypatch):
    client, recorder = make_client(monkeypatch)
    client.capabilities("org/model name")
    assert recorder.calls[-1][1] == "/models/org/model%20name/capabilities"


def test_http_errors_become_api_errors(monkeypatch):
    client = GLiFormerClient(model="m")

    def fail(*_args, **_kwargs):
        payload = json.dumps({"error": {"code": "UNSUPPORTED_LABEL_TYPE", "message": "typed", "path": "/tasks/t"}}).encode()
        raise urllib.error.HTTPError("http://x", 422, "Unprocessable", {}, io.BytesIO(payload))

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(GLiFormerAPIError) as error:
        client.analyze("x", {"t": {"type": "json", "labels": ["a"]}})
    assert (error.value.status, error.value.code, error.value.path) == (422, "UNSUPPORTED_LABEL_TYPE", "/tasks/t")


def test_unreachable_server_is_a_client_error(monkeypatch):
    client = GLiFormerClient(model="m")

    def fail(*_args, **_kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(GLiFormerClientError) as error:
        client.health()
    assert not isinstance(error.value, GLiFormerAPIError)
