import asyncio

import pytest

from gliformer.serve.client import GLiFormerClient, PerText


def test_client_dispatches_lists_concurrently_and_preserves_adapter_ids(monkeypatch):
    client = GLiFormerClient(max_concurrency=2)
    payloads = []

    def post(payload):
        payloads.append(payload)
        return {"text": payload["text"], "adapter": payload.get("adapter_id")}

    monkeypatch.setattr(client, "_post", post)
    result = client.predict(
        ["one", "two"],
        entities=["item"],
        adapter_id=["legal", "finance"],
    )
    assert result == [
        {"text": "one", "adapter": "legal"},
        {"text": "two", "adapter": "finance"},
    ]
    assert {payload["adapter_id"] for payload in payloads} == {"legal", "finance"}


def test_client_rejects_wrong_number_of_adapter_ids():
    client = GLiFormerClient()
    with pytest.raises(ValueError, match="length 2"):
        client.predict(["one", "two"], entities=["item"], adapter_id=["legal"])


def test_client_supports_per_text_task_schemas(monkeypatch):
    client = GLiFormerClient(max_concurrency=2)
    monkeypatch.setattr(client, "_post", lambda payload: payload)
    results = client.predict(
        ["Alice", "Paris"],
        entities=[["person"], {"location": ["city", "country"]}],
    )
    assert [result["entities"] for result in results] == [
        ["person"],
        {"location": ["city", "country"]},
    ]


def test_client_supports_complex_per_text_tasks_without_schema_heuristics(monkeypatch):
    client = GLiFormerClient(max_concurrency=2)
    monkeypatch.setattr(client, "_post", lambda payload: payload)
    results = client.predict(
        ["Alice works at Acme", "Carol joined Globex"],
        per_text_tasks=[
            {
                "joint_relations": {
                    "employment": {
                        "entities": ["person", "organization"],
                        "relations": ["works_at"],
                    }
                }
            },
            {"structures": {"employee": ["name", "company"]}},
        ],
    )
    assert "joint_relations" in results[0]
    assert results[1]["structures"] == {"employee": ["name", "company"]}


def test_client_supports_explicit_per_text_complex_schemas(monkeypatch):
    client = GLiFormerClient(max_concurrency=2)
    monkeypatch.setattr(client, "_post", lambda payload: payload)
    results = client.predict(
        ["Alice works at Acme", "Paris is in France"],
        joint_relations=PerText([
            {
                "employment": {
                    "entities": ["person", "organization"],
                    "relations": ["works_at"],
                }
            },
            None,
        ]),
        structures=PerText([
            None,
            {"place": ["name", "country"]},
        ]),
    )

    assert results[0]["joint_relations"]["employment"]["relations"] == [
        "works_at"
    ]
    assert "structures" not in results[0]
    assert "joint_relations" not in results[1]
    assert results[1]["structures"] == {"place": ["name", "country"]}


def test_client_preserves_shared_list_structuring_schema(monkeypatch):
    client = GLiFormerClient(max_concurrency=2)
    monkeypatch.setattr(client, "_post", lambda payload: payload)
    schema = [{"name": "str"}, {"company": "str"}]

    results = client.predict(["one", "two"], structures=schema)

    assert results[0]["structures"] == schema
    assert results[1]["structures"] == schema


def test_client_predict_requests_preserves_fully_independent_payloads(monkeypatch):
    client = GLiFormerClient(max_concurrency=2)
    monkeypatch.setattr(client, "_post", lambda payload: payload)
    requests = [
        {"text": "one", "structures": [{"name": "str"}], "threshold": 0.2},
        {"text": "two", "entities": ["item"], "flat_ner": False},
    ]
    assert client.predict_requests(requests) == requests


def test_client_can_request_embeddings(monkeypatch):
    client = GLiFormerClient()
    monkeypatch.setattr(
        client,
        "_post",
        lambda payload, suffix="": {"embedding": [1.0]} if suffix else payload,
    )
    result = client.predict("hello", embed=True)
    assert result == [1.0]


def test_client_embed_text_uses_dedicated_endpoint(monkeypatch):
    client = GLiFormerClient()
    calls = []

    def post(payload, suffix=""):
        calls.append((payload, suffix))
        return {"embedding": [1.0, 2.0]}

    monkeypatch.setattr(client, "_post", post)
    assert client.embed_text("hello") == [1.0, 2.0]
    assert calls == [({"text": "hello"}, "/embeddings")]


def test_task_convenience_methods_unwrap_their_outputs(monkeypatch):
    client = GLiFormerClient()
    monkeypatch.setattr(
        client,
        "predict",
        lambda text, **kwargs: {"ner": [{"text": "Alice"}]},
    )
    assert client.predict_entities("Alice", ["person"]) == [{"text": "Alice"}]


def test_async_client_dispatches_all_requests(monkeypatch):
    client = GLiFormerClient(max_concurrency=2)
    monkeypatch.setattr(client, "_post", lambda payload: {"text": payload["text"]})

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)
    result = asyncio.run(client.predict_async(["one", "two"], entities=["item"]))
    assert result == [{"text": "one"}, {"text": "two"}]


def test_client_rejects_wrong_complex_per_text_count():
    client = GLiFormerClient()
    with pytest.raises(ValueError, match="per_text_tasks must have length 2"):
        client.predict(["one", "two"], per_text_tasks=[{"entities": ["item"]}])


def test_client_rejects_wrong_per_text_wrapper_count():
    client = GLiFormerClient()
    with pytest.raises(ValueError, match="PerText structures must have length 2"):
        client.predict(["one", "two"], structures=PerText([{"record": ["field"]}]))
