import asyncio
import signal
import sys
import types

import pytest
import torch
from fastapi.testclient import TestClient

import gliformer.serve.server as serve_module
from gliformer.serve import APIError, GLiFormerServeConfig, build_app
from gliformer.serve.http import _read_limited_body

from .fakes import FakeGLiFormer, body, make_server


def test_route_prefix_is_normalized():
    assert GLiFormerServeConfig(route_prefix="api").route_prefix == "/api"


def test_config_exports_model_loading_environment():
    config = GLiFormerServeConfig(enable_flashdeberta=True, tokenizer_threads=2)
    assert config.to_env_vars() == {
        "TOKENIZERS_PARALLELISM": "true",
        "USE_FLASHDEBERTA": "1",
    }


def test_precompiled_batch_sizes_do_not_exceed_max_batch_size():
    config = GLiFormerServeConfig(
        max_batch_size=4,
        precompiled_batch_sizes=[1, 2, 8, 16, 32],
    )
    assert config.precompiled_batch_sizes == [1, 2, 4]


def test_blocking_serve_shuts_down_owned_ray_runtime(monkeypatch):
    events = []
    handlers = {}

    class FakeServe:
        def start(self, **_kwargs):
            events.append("serve-start")

        def run(self, *_args, **_kwargs):
            return "handle"

        def shutdown(self):
            events.append("serve-shutdown")

    fake_serve = FakeServe()
    fake_ray = types.ModuleType("ray")
    fake_ray.serve = fake_serve
    fake_ray.initialized = False
    fake_ray.is_initialized = lambda: fake_ray.initialized

    def init(**_kwargs):
        fake_ray.initialized = True
        events.append("ray-init")

    def shutdown():
        fake_ray.initialized = False
        events.append("ray-shutdown")

    fake_ray.init = init
    fake_ray.shutdown = shutdown
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(serve_module, "build_deployment", lambda _config: "app")

    def install_handler(signum, handler):
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    monkeypatch.setattr(signal, "signal", install_handler)
    monkeypatch.setattr(
        "time.sleep",
        lambda _seconds: handlers[signal.SIGINT](signal.SIGINT, None),
    )

    handle = serve_module.serve(
        GLiFormerServeConfig(device="cpu", min_available_host_memory_gb=0),
        blocking=True,
    )

    assert handle == "handle"
    assert events == ["ray-init", "serve-start", "serve-shutdown", "ray-shutdown"]


def test_compile_warmup_uses_heterogeneous_serving_path():
    server, model = make_server(
        max_batch_size=3,
        precompiled_batch_sizes=[3],
        warmup_iterations=1,
    )
    model.config.joint_relex_config = None

    server._compile_and_warmup()

    assert model.compiled is True
    rows, kwargs = model.calls[0]
    assert [row["text"] for row in rows] == [
        "GLiFormer serving warmup sample 0.",
        "GLiFormer serving warmup sample 1.",
        "GLiFormer serving warmup sample 2.",
    ]
    assert kwargs["batch_size"] == 3
    assert kwargs["inference_collator"] == {"return_tokens": True}


def test_compile_warmup_supports_bounded_startup_backend(monkeypatch):
    server, model = make_server(
        max_batch_size=1,
        precompiled_batch_sizes=[1],
        warmup_iterations=1,
        compilation_backend="aot_eager",
    )
    model.config.ner_config = {}
    compiled = []

    def compile_model(module, **kwargs):
        compiled.append((module, kwargs))
        return module

    monkeypatch.setattr(torch, "compile", compile_model)
    server._compile_and_warmup()

    assert compiled == [
        (model.model, {"dynamic": True, "backend": "aot_eager"})
    ]


def test_config_rejects_unknown_compile_backend():
    with pytest.raises(ValueError, match="compilation_backend"):
        GLiFormerServeConfig(compilation_backend="unknown")


def test_sequence_packing_falls_back_for_unverified_backbone(caplog):
    server, _ = make_server(enable_sequence_packing=True)
    assert server.packing_config is None
    assert "unverified backbone" in caplog.text


def test_health_reports_loaded_server_state():
    server, _ = make_server()
    health = server.health()
    assert health["status"] == "ok"
    assert health["device"] == "cpu"
    assert health["polylora"] is False


def test_streaming_body_reader_stops_before_buffering_oversized_request():
    class Request:
        chunks_read = 0

        async def stream(self):
            for chunk in (b"1234", b"5678", b"must-not-be-read"):
                self.chunks_read += 1
                yield chunk

    request = Request()
    with pytest.raises(OverflowError, match="Request exceeds 7 bytes"):
        asyncio.run(_read_limited_body(request, 7))
    assert request.chunks_read == 2


def test_config_validates_thresholds_and_label_template():
    with pytest.raises(ValueError):
        GLiFormerServeConfig(default_thresholds={"span": 1.5})
    with pytest.raises(ValueError):
        GLiFormerServeConfig(label_template="{name}")
    assert GLiFormerServeConfig().route_prefix == "/v1"


def test_cuda_oom_retries_with_smaller_model_batch(monkeypatch):
    server, model = make_server(max_batch_size=2)
    batch_sizes = []
    original = model.inference_requests

    def inference_requests(requests, **kwargs):
        batch_sizes.append(kwargs["batch_size"])
        if kwargs["batch_size"] > 1:
            raise torch.cuda.OutOfMemoryError("synthetic")
        return original(requests, **kwargs)

    monkeypatch.setattr(model, "inference_requests", inference_requests)
    outcomes = server.analyze_many([
        body("Alice", {"p": {"type": "span", "labels": ["person"]}}),
        body("Bob", {"p": {"type": "span", "labels": ["person"]}}),
    ])
    assert batch_sizes == [2, 1]
    assert [item["results"]["p"]["value"][0]["text"] for item in outcomes] == ["Alice", "Bob"]


def test_rows_request_group_indices_and_reuse_the_collator():
    server, model = make_server()
    server.analyze(body("Alice", {"p": {"type": "span", "labels": ["person"]}}))
    _, kwargs = model.calls[0]
    assert kwargs["return_group_index"] is True
    assert kwargs["inference_collator"] == {"return_tokens": True}


def test_adapter_ids_reach_rows_and_embeddings(monkeypatch):
    server, model = make_server()
    monkeypatch.setattr(server, "ensure_adapter_loaded", lambda adapter_id: adapter_id)
    server.analyze(body("Alice", {
        "p": {"type": "span", "labels": ["person"]},
        "v": {"type": "embedding"},
    }, adapter_id="legal"))
    rows, _ = model.calls[0]
    assert rows[0]["adapter_id"] == "legal"
    assert model.calls[1][1]["adapter_ids"] == ["legal"]


# -- HTTP -------------------------------------------------------------------------


def http_client(**overrides):
    server, model = make_server(**overrides)

    async def analyze_one(request_body):
        return server.analyze_many([request_body])[0]

    app = build_app(lambda: server, analyze_one, server.config.max_request_bytes)
    return TestClient(app), server, model


def test_http_analyze_and_errors():
    client, _, _ = http_client(max_request_bytes=500)
    ok = client.post("/analyze", json=body("Alice", {"p": {"type": "span", "labels": ["person"]}}))
    assert ok.status_code == 200
    assert ok.json()["results"]["p"]["value"][0]["source"] == ""

    missing = client.post("/analyze", json={"model": "nope", "input": "x", "tasks": {"t": {"type": "span"}}})
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "MODEL_NOT_FOUND"

    invalid = client.post("/analyze", json=body("x", {"t": {"type": "classify"}}))
    assert invalid.status_code == 422
    assert invalid.json()["error"]["path"] == "/tasks/t/labels"

    broken = client.post("/analyze", content=b"{", headers={"content-type": "application/json"})
    assert broken.status_code == 422 and broken.json()["error"]["code"] == "INVALID_REQUEST"

    large = client.post("/analyze", json=body("x" * 600, {"t": {"type": "span"}}))
    assert large.status_code == 413 and large.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_http_batch_models_capabilities_health():
    client, _, _ = http_client()
    batch = client.post("/analyze/batch", json={
        "model": "test/gliformer",
        "items": [
            {"id": "a", "input": "Alice", "tasks": {"p": {"type": "span", "labels": ["person"]}}},
            "not an object",
        ],
    })
    assert batch.status_code == 200
    items = batch.json()["items"]
    assert items[0]["id"] == "a" and items[0]["results"]["p"]["value"]
    assert items[1]["error"]["path"] == "/items/1"

    bad_batch = client.post("/analyze/batch", json={"model": "test/gliformer", "items": []})
    assert bad_batch.status_code == 422

    assert client.get("/models").json() == {"data": [{"id": "test/gliformer", "aliases": []}]}
    capabilities = client.get("/models/test/gliformer/capabilities")
    assert capabilities.status_code == 200 and capabilities.json()["model"] == "test/gliformer"
    assert client.get("/models/other/capabilities").status_code == 404
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").status_code == 200
    assert client.get("/adapter-cache").json()["enabled"] is False


def test_unexpected_exceptions_become_internal_errors(monkeypatch):
    client, server, _ = http_client()
    monkeypatch.setattr(server, "models", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    client = TestClient(client.app, raise_server_exceptions=False)
    response = client.get("/models")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL"


def test_api_error_serialization():
    error = APIError("INPUT_TOO_LONG", "too long", "/input", {"tokens": 9})
    assert error.status == 413
    assert error.to_dict() == {"error": {
        "code": "INPUT_TOO_LONG", "message": "too long", "path": "/input", "details": {"tokens": 9},
    }}
    assert error.with_prefix("/items/3").path == "/items/3/input"
    with pytest.raises(ValueError):
        APIError("NOPE", "x")


def test_health_lists_task_types():
    server, _ = make_server(FakeGLiFormer(embedding=False))
    assert "embedding" not in server.health()["task_types"]
