"""GLiFormer inference server and optional Ray Serve deployment."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import torch
from gliner import InferencePackingConfig

from gliformer import GLiFormer

from .api import AnalyzeEngine, APIError, capabilities_from_model
from .config import GLiFormerServeConfig
from .memory import GLiFormerMemoryEstimator

logger = logging.getLogger(__name__)


class GLiFormerServer:
    """Own one GLiFormer model and expose multi-task batched inference."""

    def __init__(self, config: GLiFormerServeConfig, model=None) -> None:
        self.config = config
        self._polylora_model = None
        self._adapter_id_re = re.compile(config.polylora_adapter_id_pattern)

        for key, value in config.to_env_vars().items():
            os.environ[key] = value
        if config.tokenizer_threads > 0:
            torch.set_num_threads(config.tokenizer_threads)
        torch.set_float32_matmul_precision("high")
        self.memory_estimator = GLiFormerMemoryEstimator(
            safety_factor=config.memory_overhead_factor,
            target_memory_fraction=config.target_memory_fraction,
            calibration_probe_batch_size=config.calibration_probe_batch_size,
        )
        if model is None and torch.cuda.is_available() and config.device.startswith("cuda"):
            self.memory_estimator.measure_cuda_context()

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        self.torch_dtype = dtype_map.get(config.dtype.lower(), torch.bfloat16)
        self.model = model or GLiFormer.from_pretrained(
            config.model,
            map_location=config.device,
            dtype=self.torch_dtype,
            max_length=config.max_model_len,
            max_width=config.max_span_width,
            quantize=config.quantization,
        )
        if config.enable_polylora:
            self._initialize_polylora()
        self.model.eval()
        create_collator = getattr(self.model, "create_inference_collator", None)
        self.collator = create_collator(return_tokens=True) if callable(create_collator) else None
        self.packing_config = self._create_packing_config()
        if model is None and torch.cuda.is_available() and config.device.startswith("cuda"):
            self.memory_estimator.measure_model_memory()
        if model is None and config.enable_compilation:
            self._compile_and_warmup()
            if torch.cuda.is_available() and config.device.startswith("cuda"):
                self.memory_estimator.measure_model_memory()
        if (
            model is None
            and config.enable_memory_calibration
            and torch.cuda.is_available()
            and config.device.startswith("cuda")
        ):
            self._calibrate_memory()
        processor = getattr(self.model, "data_processor", None)
        self._tokenizer = getattr(processor, "transformer_tokenizer", None)
        self.capabilities_info = capabilities_from_model(self.model, config)
        self.engine = AnalyzeEngine(
            self.capabilities_info,
            run_rows=self._run_rows,
            embed=self._embed,
            count_tokens=self._count_tokens,
            resolve_adapter=self._resolve_adapter,
            render_task_names=config.render_task_names,
        )

    def _create_packing_config(self) -> InferencePackingConfig | None:
        """Enable packing only for encoder paths with verified mask semantics."""
        if not self.config.enable_sequence_packing:
            return None
        if self.config.enable_polylora:
            logger.warning(
                "Sequence packing is disabled with PolyLoRA: packed streams cannot "
                "yet preserve independent per-row adapter routing"
            )
            return None
        try:
            backbone = self.model.model.token_rep_layer.bert_layer.model
        except AttributeError:
            logger.warning("Sequence packing is unavailable for this model variant")
            return None
        supported = {
            "DebertaModel", "DebertaV2Model", "LayoutDebertaModel",
            "ModernBertModel", "T5EncoderModel", "MT5EncoderModel", "T5Model",
        }
        backbone_name = backbone.__class__.__name__
        if backbone_name not in supported:
            logger.warning(
                "Sequence packing is disabled for unverified backbone %s",
                backbone_name,
            )
            return None
        logger.info(
            "Sequence packing enabled for %s (max stream length %d)",
            backbone_name,
            self.config.max_model_len,
        )
        return InferencePackingConfig(max_length=self.config.max_model_len)

    def _probe_tasks(self) -> dict[str, Any]:
        """Return a small combined schema covering every configured text head."""
        tasks: dict[str, Any] = {}
        for config_name, argument, value in (
            ("ner_config", "entities", ["entity"]),
            ("classification_config", "classes", ["class"]),
            ("open_relex_config", "relations", ["related_to"]),
            (
                "joint_relex_config",
                "joint_relations",
                {"relation": {"entities": ["entity"], "relations": ["related_to"]}},
            ),
            ("structuring_config", "structures", {"record": ["field"]}),
        ):
            if getattr(self.model.config, config_name, None) is not None:
                tasks[argument] = value
        if not tasks:
            raise RuntimeError("Serving warmup needs at least one supported text task head")
        return tasks

    def _probe_requests(self, size: int) -> list[dict[str, Any]]:
        templates = []
        for config_name, payload in (
            ("ner_config", {"entities": ["entity"]}),
            ("classification_config", {"classes": ["class"]}),
            ("open_relex_config", {"relations": ["related_to"]}),
            (
                "joint_relex_config",
                {
                    "joint_relations": {
                        "relation": {
                            "entities": ["entity"],
                            "relations": ["related_to"],
                        }
                    }
                },
            ),
            ("structuring_config", {"structures": {"record": ["field"]}}),
        ):
            if getattr(self.model.config, config_name, None) is not None:
                templates.append(payload)
        if not templates:
            raise RuntimeError("Serving warmup needs at least one supported text task head")
        return [
            {
                "text": f"GLiFormer serving warmup sample {index}.",
                **templates[index % len(templates)],
            }
            for index in range(size)
        ]

    def _compile_and_warmup(self) -> None:
        logger.info("Compiling GLiFormer and warming batch sizes %s", self.config.precompiled_batch_sizes)
        if self.config.compilation_backend == "inductor":
            compile_method = getattr(self.model, "compile", None)
            if callable(compile_method):
                compile_method()
        else:
            # Keep BaseGLiNER's Inductor default intact while allowing a
            # bounded-startup backend for complex multitask graphs.
            torch._dynamo.config.capture_scalar_outputs = True
            self.model.model = torch.compile(
                self.model.model,
                dynamic=True,
                backend=self.config.compilation_backend,
            )
        for size in self.config.precompiled_batch_sizes:
            for _ in range(self.config.warmup_iterations):
                self.model.inference_requests(
                    self._probe_requests(size),
                    batch_size=size,
                    inference_collator=self.collator,
                    **(
                        {"packing_config": self.packing_config}
                        if self.packing_config is not None else {}
                    ),
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _calibrate_memory(self) -> None:
        tasks = self._probe_tasks()

        def probe(texts: list[str]) -> object:
            return self.model.inference(
                texts,
                batch_size=len(texts),
                inference_collator=self.collator,
                packing_config=self.packing_config,
                **tasks,
            )

        self.memory_estimator.calibrate(
            probe,
            max_seq_len=self.config.max_model_len,
            min_seq_len=self.config.calibration_min_seq_len,
        )

    def _count_tokens(self, text: str) -> int:
        """Subword tokens of ``text`` as the model's encoder sees it."""
        tokenizer = self._tokenizer
        if tokenizer is None:
            return len(text.split())
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    @staticmethod
    def observed_seq_len_of(texts: list[str], minimum: int, maximum: int) -> int:
        observed = max([minimum, *(len(text.split()) for text in texts)])
        return min(observed, maximum)

    def observed_seq_len(self, texts: list[str]) -> int:
        return self.observed_seq_len_of(
            texts, self.config.calibration_min_seq_len, self.config.max_model_len,
        )

    def batch_size_fn(self, seq_len: int | None = None) -> int:
        if not torch.cuda.is_available() or not self.memory_estimator.per_sample_table:
            return self.config.max_batch_size
        return self.memory_estimator.batch_size_fn(
            seq_len or self.config.max_model_len,
            self.config.precompiled_batch_sizes,
        )

    def _get_polylora_target_model(self):
        try:
            return self.model.model.token_rep_layer.bert_layer.model
        except AttributeError as exc:
            variant = getattr(self.model.config, "model_variant", None)
            raise NotImplementedError(
                "PolyLoRA requires a GLiFormer text backbone; "
                f"model_variant={variant!r} does not expose one"
            ) from exc

    def _set_polylora_target_model(self, wrapped_model) -> None:
        self.model.model.token_rep_layer.bert_layer.model = wrapped_model

    def _initialize_polylora(self) -> None:
        try:
            from polylora import PolyLoraConfig, PolyLoraModel
        except ImportError as exc:
            raise ImportError(
                "enable_polylora=True requires the optional 'polylora' package"
            ) from exc

        target = self._get_polylora_target_model()
        if target.__class__.__module__.startswith("peft"):
            raise ValueError("PolyLoRA cannot wrap a backbone already wrapped by PEFT")
        poly_config = PolyLoraConfig(
            max_gpu_adapters=self.config.polylora_max_gpu_adapters,
            max_cpu_adapters=self.config.polylora_max_cpu_adapters,
            disk_cache_dir=self.config.polylora_disk_cache_dir,
            max_disk_adapters=self.config.polylora_max_disk_adapters,
            max_rank=self.config.polylora_max_rank,
            target_modules=self.config.polylora_adapter_weight_modules,
            base_adapter_id=self.config.polylora_base_adapter_id,
            enforce_right_padding=self.config.polylora_enforce_right_padding,
            use_triton_kernels=self.config.polylora_use_triton_kernels,
        )
        self._polylora_model = PolyLoraModel(target, poly_config)
        self._set_polylora_target_model(self._polylora_model)
        for adapter_id, adapter_path in self.config.polylora_adapters.items():
            self.load_adapter(adapter_id, adapter_path)

    def _validate_adapter_id(self, adapter_id: str) -> None:
        if not isinstance(adapter_id, str) or not self._adapter_id_re.fullmatch(adapter_id):
            raise ValueError("adapter_id must match polylora_adapter_id_pattern")
        if adapter_id == self.config.polylora_base_adapter_id:
            raise ValueError(f"{adapter_id!r} is reserved for base-only inference")

    def load_adapter(
        self,
        adapter_id: str,
        adapter_path: str,
        peft_adapter_name: str = "default",
    ) -> None:
        """Load a PEFT LoRA directory into the configured adapter stores."""
        if self._polylora_model is None:
            raise RuntimeError("PolyLoRA is not enabled")
        self._validate_adapter_id(adapter_id)
        self._polylora_model.load_adapter_from_disk(
            adapter_id, adapter_path, peft_adapter_name=peft_adapter_name
        )

    def ensure_adapter_loaded(self, adapter_id: str | None) -> str | None:
        if not self.config.enable_polylora:
            if adapter_id not in (None, self.config.polylora_base_adapter_id):
                raise KeyError(f"Unknown LoRA adapter id: {adapter_id}")
            return None
        if adapter_id in (None, self.config.polylora_base_adapter_id):
            return self.config.polylora_base_adapter_id
        self._validate_adapter_id(adapter_id)
        if self._polylora_model is not None and adapter_id in self._polylora_model.adapter_store:
            return adapter_id
        raise KeyError(f"Unknown LoRA adapter id: {adapter_id}")

    def adapter_cache_status(self, adapter_id: str | None = None) -> dict[str, Any]:
        if self._polylora_model is None:
            return {"enabled": False, "base_adapter_id": self.config.polylora_base_adapter_id}
        store = self._polylora_model.adapter_store
        disk_cache = getattr(store, "disk_cache", None)
        result: dict[str, Any] = {
            "enabled": True,
            "base_adapter_id": self.config.polylora_base_adapter_id,
            "loaded": sorted(store.adapters),
            "disk_cached": sorted(disk_cache.entries) if disk_cache is not None else [],
            "disk_cache_dir": str(disk_cache.cache_dir) if disk_cache is not None else None,
            "gpu_slots": list(self._polylora_model.adapter_cache.slot_to_adapter),
        }
        if adapter_id is not None:
            resolved = self.ensure_adapter_loaded(adapter_id)
            result.update(
                adapter_id=resolved,
                cached=(resolved == self.config.polylora_base_adapter_id)
                or (disk_cache is not None and resolved in disk_cache),
                cpu_resident=resolved in store.adapters,
                gpu_resident=resolved in self._polylora_model.adapter_cache.adapter_to_slot,
            )
        return result

    # -- analyze API --------------------------------------------------------

    def _run_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run planned model rows with OOM-safe adaptive batch sizes."""
        inference_requests = getattr(self.model, "inference_requests", None)
        if not callable(inference_requests):
            raise TypeError("Model does not implement heterogeneous inference_requests")
        safe_batch_size = min(
            len(rows),
            self.batch_size_fn(self.observed_seq_len([row["text"] for row in rows])),
        )
        while True:
            try:
                controls: dict[str, Any] = {
                    "batch_size": max(1, safe_batch_size),
                    "return_group_index": True,
                }
                if self.collator is not None:
                    controls["inference_collator"] = self.collator
                if self.packing_config is not None:
                    controls["packing_config"] = self.packing_config
                return inference_requests(rows, **controls)
            except torch.cuda.OutOfMemoryError:
                if safe_batch_size <= 1:
                    raise
                next_size = max(1, safe_batch_size // 2)
                logger.warning(
                    "CUDA OOM at model batch size %d; retrying at %d",
                    safe_batch_size,
                    next_size,
                )
                safe_batch_size = next_size
                torch.cuda.empty_cache()

    def _embed(self, texts: list[str], adapter_ids: list[str | None]) -> list[list[float]]:
        safe_batch_size = min(len(texts), self.batch_size_fn(self.observed_seq_len(texts)))
        resolved = adapter_ids if any(item is not None for item in adapter_ids) else None
        while True:
            try:
                return self.model.embed_text(
                    texts,
                    batch_size=max(1, safe_batch_size),
                    adapter_ids=resolved,
                ).detach().float().cpu().tolist()
            except torch.cuda.OutOfMemoryError:
                if safe_batch_size <= 1:
                    raise
                safe_batch_size = max(1, safe_batch_size // 2)
                logger.warning("CUDA OOM while embedding; retrying at %d", safe_batch_size)
                torch.cuda.empty_cache()

    def _resolve_adapter(self, adapter_id: str | None) -> str | None:
        return self.ensure_adapter_loaded(adapter_id)

    def analyze(self, body: Any) -> dict[str, Any]:
        """``POST /v1/analyze``. Raises ``APIError`` for a rejected request."""
        return self.engine.analyze(body)

    def analyze_many(self, bodies: list[Any]) -> list[dict[str, Any] | APIError]:
        """Independent analyze requests sharing model forwards."""
        return self.engine.analyze_many(bodies)

    def analyze_batch(self, body: Any) -> dict[str, Any]:
        """``POST /v1/analyze/batch``. Raises ``APIError`` for a bad envelope."""
        return self.engine.analyze_batch(body)

    def capabilities(self, model_id: str | None = None) -> dict[str, Any]:
        if model_id is not None and model_id not in self.capabilities_info.model_ids():
            raise APIError("MODEL_NOT_FOUND", f"Model {model_id!r} is not served here", "/model")
        return self.capabilities_info.to_dict()

    def models(self) -> dict[str, Any]:
        info = self.capabilities_info
        return {"data": [{"id": info.model, "aliases": list(info.aliases)}]}

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": self.config.model,
            "device": self.config.device,
            "cuda_available": torch.cuda.is_available(),
            "polylora": self.config.enable_polylora,
            "sequence_packing": self.packing_config is not None,
            "max_batch_size": self.batch_size_fn(),
            "task_types": list(self.capabilities_info.task_types),
        }


def build_app(get_server, analyze_one, max_request_bytes: int):
    """FastAPI app of the analyze API; see ``gliformer.serve.http.build_app``."""
    from .http import build_app as _build_app

    return _build_app(get_server, analyze_one, max_request_bytes)


def build_deployment(config: GLiFormerServeConfig):
    """Build a Ray Serve deployment without importing Ray for library users."""
    try:
        import ray.serve  # noqa: F401
    except ImportError as exc:
        raise ImportError("Serving requires `pip install 'gliformer[serve]'`") from exc
    from .http import build_ray_deployment

    return build_ray_deployment(config, GLiFormerServer)


def serve(config: GLiFormerServeConfig, blocking: bool = False):
    """Start a local or clustered Ray Serve application."""
    import ray
    from ray import serve as ray_serve

    _check_host_memory(config.min_available_host_memory_gb)
    owns_ray_runtime = not ray.is_initialized()
    if owns_ray_runtime:
        ray.init(address=config.ray_address, ignore_reinit_error=True)
    ray_serve.start(http_options={"host": "0.0.0.0", "port": config.http_port})
    handle = ray_serve.run(
        build_deployment(config), name="gliformer", route_prefix=config.route_prefix
    )
    logger.info("GLiFormer Serve is listening on port %d%s", config.http_port, config.route_prefix)
    if blocking:
        import signal
        import time

        stopping = False

        def stop(_signum, _frame):
            nonlocal stopping
            stopping = True

        previous_sigint = signal.signal(signal.SIGINT, stop)
        previous_sigterm = signal.signal(signal.SIGTERM, stop)
        try:
            while not stopping:
                time.sleep(1)
        finally:
            ray_serve.shutdown()
            if owns_ray_runtime and ray.is_initialized():
                ray.shutdown()
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)
    return handle


def _check_host_memory(minimum_gb: float) -> None:
    """Fail before Ray startup when Linux reports critically low host RAM."""
    if minimum_gb <= 0:
        return
    try:
        with open("/proc/meminfo", encoding="utf-8") as stream:
            values = {
                key.rstrip(":"): int(value.split()[0])
                for key, value in (line.split(maxsplit=1) for line in stream)
            }
        available_gb = values["MemAvailable"] / 1024**2
    except (OSError, KeyError, ValueError):
        return
    if available_gb < minimum_gb:
        raise RuntimeError(
            f"Refusing to start Ray Serve with only {available_gb:.2f} GiB host RAM "
            f"available; requires {minimum_gb:.2f} GiB. Set "
            "min_available_host_memory_gb=0 to override."
        )


def shutdown() -> None:
    from ray import serve as ray_serve

    ray_serve.shutdown()


class GLiFormerFactory:
    """Lifecycle-managed in-process facade over the Ray Serve deployment.

    ``analyze`` / ``analyze_batch`` take the same bodies as the HTTP routes
    and go through the deployment's dynamic batcher.
    """

    def __init__(self, model: str | None = None, *, config=None, **kwargs) -> None:
        if config is not None and (model is not None or kwargs):
            raise ValueError("Pass either config or model/kwargs, not both")
        if config is None:
            if model is None:
                raise ValueError("model or config is required")
            config = GLiFormerServeConfig(model=model, **kwargs)
        self.config = config
        self._handle = serve(config)
        self._closed = False

    @property
    def handle(self):
        return self._handle

    @property
    def model_id(self) -> str:
        return self.config.model_id or self.config.model

    def _body(self, input: Any, tasks: dict[str, Any], options: dict[str, Any] | None, model: str | None) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model or self.model_id, "input": input, "tasks": tasks}
        if options:
            body["options"] = options
        return body

    @staticmethod
    def _unwrap(response: dict[str, Any]) -> dict[str, Any]:
        if isinstance(response, dict) and "error" in response and "results" not in response and "items" not in response:
            error = response["error"]
            raise APIError(
                error.get("code", "INTERNAL"),
                error.get("message", ""),
                error.get("path"),
                error.get("details"),
            )
        return response

    def analyze(self, input: Any = None, tasks: dict[str, Any] | None = None, *, options=None, model=None, body=None):
        """Run one analyze request; raises ``APIError`` when it is rejected."""
        request = body if body is not None else self._body(input, tasks or {}, options, model)
        return self._unwrap(self._handle.handle_analyze.remote(request).result())

    def analyze_batch(self, items: list[dict[str, Any]], *, options=None, model=None):
        body: dict[str, Any] = {"model": model or self.model_id, "items": items}
        if options:
            body["options"] = options
        return self._unwrap(self._handle.handle_analyze_batch.remote(body).result())

    async def analyze_async(self, input: Any = None, tasks: dict[str, Any] | None = None, *, options=None, model=None, body=None):
        request = body if body is not None else self._body(input, tasks or {}, options, model)
        return self._unwrap(await self._handle.handle_analyze.remote(request))

    async def analyze_batch_async(self, items: list[dict[str, Any]], *, options=None, model=None):
        body: dict[str, Any] = {"model": model or self.model_id, "items": items}
        if options:
            body["options"] = options
        return self._unwrap(await self._handle.handle_analyze_batch.remote(body))

    def capabilities(self) -> dict[str, Any]:
        return self._handle.handle_capabilities.remote().result()

    def shutdown(self) -> None:
        if self._closed:
            return
        import ray

        shutdown()
        if ray.is_initialized():
            ray.shutdown()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.shutdown()
        return False

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
