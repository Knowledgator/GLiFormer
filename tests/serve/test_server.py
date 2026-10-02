import asyncio
import signal
import sys
import types
from types import SimpleNamespace

import pytest
import torch

import gliformer.serve.server as serve_module
from gliformer.serve import (
    GLiFormerFactory,
    GLiFormerServeConfig,
    GLiFormerServer,
    PerText,
)
from gliformer.serve.server import _read_limited_body


class FakeGLiFormer:
    def __init__(self):
        self.config = SimpleNamespace(model_variant="text")
        self.model = SimpleNamespace(
            token_rep_layer=SimpleNamespace(
                bert_layer=SimpleNamespace(model=SimpleNamespace())
            )
        )
        self.calls = []

    def eval(self):
        return self

    def compile(self):
        self.compiled = True

    def create_inference_collator(self, return_tokens=True):
        return {"return_tokens": return_tokens}

    def inference(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return {
            key: [{"text": text, "task": key} for text in texts]
            for key in ("entities", "classes", "relations", "joint_relations", "structures")
            if kwargs.get(key) is not None
        }

    def inference_requests(self, requests, **kwargs):
        self.calls.append(([item["text"] for item in requests], kwargs))
        task_names = {
            "entities": "entities",
            "classes": "classes",
            "relations": "relations",
            "joint_relations": "joint_relations",
            "structures": "structures",
        }
        return [
            {
                output: {"text": request["text"], "task": output}
                for field, output in task_names.items()
                if request.get(field) is not None
            }
            for request in requests
        ]

    def embed_text(self, texts, **kwargs):
        import torch

        self.calls.append((list(texts), {"embed_text": True, **kwargs}))
        return torch.tensor([[float(index), 1.0] for index, _ in enumerate(texts)])


def make_server(**overrides):
    model = FakeGLiFormer()
    config = GLiFormerServeConfig(device="cpu", dtype="float32", **overrides)
    return GLiFormerServer(config, model=model), model


def test_predict_routes_requested_tasks():
    server, model = make_server()
    result = server.predict(
        ["one", "two"],
        entities=["person"],
        classes=["positive", "negative"],
    )

    assert set(result) == {"entities", "classes"}
    assert model.calls[0][1]["adapter_ids"] is None
    assert model.calls[0][1]["inference_collator"] == {"return_tokens": True}


def test_unknown_adapter_is_rejected_when_polylora_is_disabled():
    server, _ = make_server()
    with pytest.raises(KeyError, match="Unknown LoRA adapter"):
        server.predict("text", entities=["person"], adapter_ids="tenant-a")


def test_different_tasks_and_schemas_share_one_physical_batch():
    server, model = make_server()
    result = server.predict_payloads(
        [
            {"text": "a", "entities": ["person"]},
            {"text": "b", "classes": ["positive"]},
            {"text": "c", "entities": ["organization"]},
        ]
    )

    assert [call[0] for call in model.calls] == [["a", "b", "c"]]
    assert [result[0]["entities"]["text"], result[2]["entities"]["text"]] == ["a", "c"]


def test_different_decode_controls_share_one_physical_batch():
    server, model = make_server()
    server.predict_payloads(
        [
            {"text": "a", "entities": ["person"], "threshold": 0.2},
            {
                "text": "b",
                "classes": ["positive"],
                "threshold": 0.9,
                "multi_label": True,
                "decoder_kwargs": {"custom_control": "second"},
            },
        ]
    )

    assert len(model.calls) == 1
    assert model.calls[0][0] == ["a", "b"]


def test_route_prefix_is_normalized():
    assert GLiFormerServeConfig(route_prefix="api").route_prefix == "/api"


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


def test_streaming_body_reader_preserves_valid_payload():
    class Request:
        async def stream(self):
            yield b'{"text":'
            yield b'"hello"}'

    assert asyncio.run(_read_limited_body(Request(), 64)) == b'{"text":"hello"}'


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
    model.config.ner_config = {}
    model.config.classification_config = {}
    model.config.open_relex_config = None
    model.config.joint_relex_config = None
    model.config.structuring_config = {"enabled": True}

    server._compile_and_warmup()

    assert model.compiled is True
    texts, kwargs = model.calls[0]
    assert texts == [
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


def test_observed_length_counts_nested_task_schemas():
    server, _ = make_server(calibration_min_seq_len=1, max_model_len=100)
    length = server.observed_seq_len(
        [{"text": "one two", "structures": {"person record": ["full name", "age"]}}]
    )
    assert length == 7


def test_health_reports_loaded_server_state():
    server, _ = make_server()
    health = server.health()
    assert health["status"] == "ok"
    assert health["device"] == "cpu"
    assert health["polylora"] is False


def test_metadata_reports_model_variant_limits_and_adapter_state():
    server, _ = make_server(max_batch_size=7)
    metadata = server.metadata()
    assert metadata["model_variant"] == "text"
    assert metadata["limits"]["max_batch_size"] == 7
    assert metadata["polylora"]["enabled"] is False
    assert metadata["sequence_packing"] == {
        "requested": False,
        "enabled": False,
        "max_length": None,
    }


def test_embedding_is_json_serializable_and_uses_batched_api():
    server, model = make_server()
    result = server.predict(["one", "two"], embed=True)
    assert result == {"embedding": [[0.0, 1.0], [1.0, 1.0]]}
    assert model.calls[0][1]["adapter_ids"] is None


def test_embedding_payloads_isolate_errors_and_preserve_order():
    server, _ = make_server()
    results = server.predict_embedding_payloads(
        [
            {"text": "one"},
            {"text": "bad", "entities": ["not-allowed"]},
            {"text": "two"},
        ],
        isolate_errors=True,
    )
    assert results[0] == {"embedding": [0.0, 1.0]}
    assert results[1]["status"] == 400
    assert results[2] == {"embedding": [1.0, 1.0]}


def test_embedding_cuda_oom_retries_with_smaller_minibatches(monkeypatch):
    server, model = make_server(max_batch_size=4)
    batch_sizes = []

    def embed_text(texts, **kwargs):
        batch_sizes.append(kwargs["batch_size"])
        if kwargs["batch_size"] > 1:
            raise torch.cuda.OutOfMemoryError("synthetic")
        return torch.ones(len(texts), 2)

    monkeypatch.setattr(model, "embed_text", embed_text)
    results = server.predict_embedding_payloads(
        [{"text": "one"}, {"text": "two"}]
    )
    assert batch_sizes == [2, 1]
    assert results == [
        {"embedding": [1.0, 1.0]},
        {"embedding": [1.0, 1.0]},
    ]


def test_payload_validation_rejects_unknown_fields_and_missing_tasks():
    server, _ = make_server()
    with pytest.raises(ValueError, match="Unsupported request fields"):
        server.validate_payload({"text": "hello", "entities": ["x"], "unsafe": True})
    with pytest.raises(ValueError, match="task schema"):
        server.validate_payload({"text": "hello"})
    with pytest.raises(ValueError, match="Unsupported request fields"):
        server.validate_payload({
            "text": "hello",
            "structures": {"record": ["field"]},
            "return_anchor_diagnostics": True,
        })


def test_batch_validation_errors_are_isolated_from_valid_requests():
    server, model = make_server()
    results = server.predict_payloads(
        [
            {"text": "bad"},
            {"text": "Alice", "entities": ["person"]},
            {"text": "unknown", "entities": ["person"], "adapter_id": "missing"},
        ],
        isolate_errors=True,
    )
    assert results[0]["status"] == 400
    assert results[1]["entities"]["text"] == "Alice"
    assert results[2]["status"] == 404
    assert model.calls[0][0] == ["Alice"]


def test_processor_error_falls_back_to_individual_requests(monkeypatch):
    server, model = make_server()
    original = model.inference_requests

    def inference_requests(requests, **kwargs):
        if any(request["text"] == "bad" for request in requests):
            if len(requests) > 1:
                raise ValueError("one schema is invalid")
            if requests[0]["text"] == "bad":
                raise ValueError("bad schema")
        return original(requests, **kwargs)

    monkeypatch.setattr(model, "inference_requests", inference_requests)
    results = server.predict_payloads(
        [
            {"text": "good", "entities": ["person"]},
            {"text": "bad", "entities": ["person"]},
        ],
        isolate_errors=True,
    )
    assert results[0]["entities"]["text"] == "good"
    assert results[1]["status"] == 400


def test_payload_validation_caps_simple_label_schemas():
    server, _ = make_server(max_labels=2)
    payload = server.validate_payload(
        {"text": "hello", "entities": ["a", "b", "c"], "classes": {"x": [], "y": [], "z": []}}
    )
    assert payload["entities"] == ["a", "b"]
    assert list(payload["classes"]) == ["x", "y"]


def test_factory_expands_per_text_schemas_and_adapters():
    tasks = GLiFormerFactory._tasks_for_items(
        {
            "entities": [["person"], ["location"]],
            "adapter_id": ["legal", "finance"],
            "threshold": 0.4,
        },
        2,
    )
    assert tasks == [
        {"entities": ["person"], "adapter_id": "legal", "threshold": 0.4},
        {"entities": ["location"], "adapter_id": "finance", "threshold": 0.4},
    ]


def test_factory_expands_complex_per_text_tasks():
    tasks = GLiFormerFactory._tasks_for_items(
        {
            "threshold": 0.4,
            "per_text_tasks": [
                {"joint_relations": {"employment": {"entities": ["person"]}}},
                {"structures": {"employee": ["name"]}},
            ],
        },
        2,
    )
    assert tasks == [
        {
            "threshold": 0.4,
            "joint_relations": {"employment": {"entities": ["person"]}},
        },
        {"threshold": 0.4, "structures": {"employee": ["name"]}},
    ]


def test_factory_expands_explicit_per_text_complex_schemas():
    tasks = GLiFormerFactory._tasks_for_items(
        {
            "joint_relations": PerText([{"employment": {}}, None]),
            "structures": PerText([None, {"employee": ["name"]}]),
        },
        2,
    )
    assert tasks == [
        {"joint_relations": {"employment": {}}, "structures": None},
        {"joint_relations": None, "structures": {"employee": ["name"]}},
    ]


def test_cuda_oom_retries_with_smaller_model_batch(monkeypatch):
    server, model = make_server(max_batch_size=2)
    batch_sizes = []

    def inference_requests(requests, **kwargs):
        batch_sizes.append(kwargs["batch_size"])
        if kwargs["batch_size"] > 1:
            raise torch.cuda.OutOfMemoryError("synthetic")
        return [{"ner": [{"text": request["text"]}]} for request in requests]

    monkeypatch.setattr(model, "inference_requests", inference_requests)
    result = server.predict_payloads(
        [
            {"text": "one", "entities": ["item"]},
            {"text": "two", "entities": ["item"]},
        ]
    )
    assert batch_sizes == [2, 1]
    assert [item["ner"][0]["text"] for item in result] == ["one", "two"]
