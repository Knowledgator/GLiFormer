"""GLiFormer inference server and optional Ray Serve deployment."""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from typing import Any

import torch
from gliner import InferencePackingConfig

from gliformer import GLiFormer

from .client import PerText
from .config import GLiFormerServeConfig
from .memory import GLiFormerMemoryEstimator

logger = logging.getLogger(__name__)

HTTP_PAYLOAD_KEYS = {
    "text", "entities", "classes", "relations", "joint_relations", "structures",
    "threshold", "flat_ner", "multi_label", "adapter_id",
    "manual_structuring_count", "structuring_dedup", "objectness_threshold",
    "preserve_empty_records", "decoder_kwargs",
}
_BATCH_ERROR_KEY = "__gliformer_serve_error__"
_REQUEST_ERRORS = (KeyError, TypeError, ValueError, OverflowError)


def _batch_error(exc: Exception) -> dict[str, Any]:
    status = 500
    if isinstance(exc, OverflowError):
        status = 413
    elif isinstance(exc, KeyError):
        status = 404
    elif isinstance(exc, (TypeError, ValueError)):
        status = 400
    return {
        _BATCH_ERROR_KEY: True,
        "status": status,
        "error": str(exc),
        "error_type": type(exc).__name__,
    }


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

    @staticmethod
    def _schema_word_count(value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, str):
            return len(value.split())
        if isinstance(value, dict):
            return sum(
                GLiFormerServer._schema_word_count(key)
                + GLiFormerServer._schema_word_count(item)
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple)):
            return sum(GLiFormerServer._schema_word_count(item) for item in value)
        return 0

    def observed_seq_len(self, payloads: list[dict[str, Any]]) -> int:
        observed = self.config.calibration_min_seq_len
        for payload in payloads:
            words = len(payload.get("text", "").split())
            words += sum(
                self._schema_word_count(payload.get(key))
                for key in ("entities", "classes", "relations", "joint_relations", "structures")
            )
            observed = max(observed, words)
        return min(observed, self.config.max_model_len)

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

    def predict(
        self,
        texts: str | list[str],
        *,
        entities=None,
        classes=None,
        relations=None,
        joint_relations=None,
        structures=None,
        embed: bool = False,
        threshold: float | None = None,
        flat_ner: bool = True,
        multi_label: bool = False,
        batch_size: int | None = None,
        adapter_ids: str | list[str | None] | None = None,
        **kwargs,
    ) -> dict[str, list]:
        """Run any combination of GLiFormer text tasks."""
        normalized_texts = [texts] if isinstance(texts, str) else list(texts)
        if isinstance(adapter_ids, list):
            if len(adapter_ids) != len(normalized_texts):
                raise ValueError("adapter_ids length must match texts length")
            resolved = [self.ensure_adapter_loaded(item) for item in adapter_ids]
        else:
            adapter = self.ensure_adapter_loaded(adapter_ids)
            resolved = [adapter] * len(normalized_texts) if adapter is not None else None

        result: dict[str, list] = {}
        if any(item is not None for item in (
            entities, classes, relations, joint_relations, structures
        )):
            result.update(self.model.inference(
                normalized_texts,
                entities=entities,
                classes=classes,
                relations=relations,
                joint_relations=joint_relations,
                structures=structures,
                threshold=self.config.default_threshold if threshold is None else threshold,
                flat_ner=flat_ner,
                multi_label=multi_label,
                batch_size=batch_size or self.config.default_batch_size,
                adapter_ids=resolved,
                **({"inference_collator": self.collator} if self.collator is not None else {}),
                **({"packing_config": self.packing_config} if self.packing_config is not None else {}),
                **kwargs,
            ))
        if embed:
            embeddings = self.model.embed_text(
                normalized_texts,
                batch_size=batch_size or self.config.default_batch_size,
                adapter_ids=resolved,
            )
            result["embedding"] = embeddings.detach().cpu().tolist()
        return result

    def _filter_labels(self, value: Any) -> Any:
        """Apply the configured label cap without changing descriptions."""
        limit = self.config.max_labels
        if limit <= 0 or value is None:
            return value
        if isinstance(value, dict):
            return dict(list(value.items())[:limit])
        if isinstance(value, list):
            return value[:limit]
        return value

    def validate_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError("JSON body must be an object")
        unknown = sorted(set(payload) - HTTP_PAYLOAD_KEYS)
        if unknown:
            raise ValueError(f"Unsupported request fields: {', '.join(unknown)}")
        text = payload.get("text")
        if not isinstance(text, str):
            raise TypeError("JSON body must contain a string 'text'")
        if len(text) > self.config.max_text_chars:
            raise OverflowError(
                f"text exceeds max_text_chars={self.config.max_text_chars}"
            )
        if not any(payload.get(key) is not None for key in (
            "entities", "classes", "relations", "joint_relations", "structures"
        )):
            raise ValueError("At least one GLiFormer task schema is required")
        for key in ("flat_ner", "multi_label", "preserve_empty_records"):
            if key in payload and not isinstance(payload[key], bool):
                raise TypeError(f"{key} must be a boolean")
        if "threshold" in payload and not isinstance(payload["threshold"], (int, float)):
            raise TypeError("threshold must be numeric")
        validated = dict(payload)
        for key in ("entities", "classes", "relations"):
            validated[key] = self._filter_labels(validated.get(key))
        return validated

    @staticmethod
    def _request_group_key(payload: dict[str, Any]) -> str:
        """Group only by controls that alter the shared model forward."""
        control_keys = {
            "manual_structuring_count",
        }
        shared = {key: payload[key] for key in control_keys if key in payload}
        return json.dumps(shared, sort_keys=True, separators=(",", ":"), default=str)

    def _predict_payload_group(
        self,
        group: list[tuple[int, dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        first = group[0][1]
        texts = [item["text"] for _, item in group]
        safe_batch_size = min(
            len(texts),
            self.batch_size_fn(self.observed_seq_len([item for _, item in group])),
        )
        while True:
            try:
                controls = {
                    "threshold": self.config.default_threshold,
                    "flat_ner": True,
                    "multi_label": False,
                    "batch_size": safe_batch_size,
                }
                for key in ("manual_structuring_count",):
                    if key in first:
                        controls[key] = first[key]
                if self.collator is not None:
                    controls["inference_collator"] = self.collator
                if self.packing_config is not None:
                    controls["packing_config"] = self.packing_config
                inference_requests = getattr(self.model, "inference_requests", None)
                if not callable(inference_requests):
                    raise TypeError("Model does not implement heterogeneous inference_requests")
                return inference_requests([item for _, item in group], **controls)
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

    def predict_payloads(
        self,
        payloads: list[dict[str, Any]],
        *,
        isolate_errors: bool = False,
    ) -> list[dict[str, Any]]:
        """Batch compatible requests while optionally isolating item failures."""
        grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        results: list[dict[str, Any] | None] = [None] * len(payloads)
        for index, payload in enumerate(payloads):
            try:
                validated = self.validate_payload(payload)
                validated["adapter_id"] = self.ensure_adapter_loaded(
                    validated.get("adapter_id")
                )
                grouped[self._request_group_key(validated)].append((index, validated))
            except _REQUEST_ERRORS as exc:
                if not isolate_errors:
                    raise
                results[index] = _batch_error(exc)

        for group in grouped.values():
            try:
                batch_result = self._predict_payload_group(group)
            except _REQUEST_ERRORS as exc:
                if not isolate_errors or len(group) == 1:
                    if not isolate_errors:
                        raise
                    batch_result = [_batch_error(exc)]
                else:
                    # An invalid schema discovered by a processor must not
                    # poison otherwise valid requests in the same Ray batch.
                    batch_result = []
                    for item in group:
                        try:
                            batch_result.extend(self._predict_payload_group([item]))
                        except _REQUEST_ERRORS as item_exc:
                            batch_result.append(_batch_error(item_exc))
            for local_index, (original_index, _) in enumerate(group):
                results[original_index] = batch_result[local_index]
        return [item or {} for item in results]

    def predict_embedding_payloads(
        self,
        payloads: list[dict[str, Any]],
        *,
        isolate_errors: bool = False,
    ) -> list[dict[str, Any]]:
        """Validate and embed HTTP rows with adaptive OOM-safe minibatches."""
        results: list[dict[str, Any] | None] = [None] * len(payloads)
        valid: list[tuple[int, dict[str, Any]]] = []
        for index, payload in enumerate(payloads):
            try:
                if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
                    raise TypeError("JSON body must contain a string 'text'")
                if len(payload["text"]) > self.config.max_text_chars:
                    raise OverflowError(
                        f"text exceeds max_text_chars={self.config.max_text_chars}"
                    )
                unknown = set(payload) - {"text", "adapter_id"}
                if unknown:
                    raise ValueError(
                        f"Unsupported embedding fields: {', '.join(sorted(unknown))}"
                    )
                item = dict(payload)
                item["adapter_id"] = self.ensure_adapter_loaded(item.get("adapter_id"))
                valid.append((index, item))
            except _REQUEST_ERRORS as exc:
                if not isolate_errors:
                    raise
                results[index] = _batch_error(exc)

        if valid:
            texts = [item["text"] for _, item in valid]
            adapter_ids = [item.get("adapter_id") for _, item in valid]
            safe_batch_size = min(
                len(valid),
                self.batch_size_fn(self.observed_seq_len([item for _, item in valid])),
            )
            while True:
                try:
                    embeddings = self.model.embed_text(
                        texts,
                        batch_size=safe_batch_size,
                        adapter_ids=adapter_ids,
                    ).detach().cpu().tolist()
                    break
                except torch.cuda.OutOfMemoryError:
                    if safe_batch_size <= 1:
                        raise
                    next_size = max(1, safe_batch_size // 2)
                    logger.warning(
                        "CUDA OOM at embedding batch size %d; retrying at %d",
                        safe_batch_size,
                        next_size,
                    )
                    safe_batch_size = next_size
                    torch.cuda.empty_cache()
            for (original_index, _), embedding in zip(valid, embeddings, strict=True):
                results[original_index] = {"embedding": embedding}
        return [item or {} for item in results]

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": self.config.model,
            "device": self.config.device,
            "cuda_available": torch.cuda.is_available(),
            "polylora": self.config.enable_polylora,
            "embedding": getattr(self.model.config, "embedding_config", None) is not None,
            "sequence_packing": self.packing_config is not None,
            "max_batch_size": self.batch_size_fn(),
        }

    def metadata(self) -> dict[str, Any]:
        task_arguments = {
            "ner": "entities",
            "classification": "classes",
            "open_relex": "relations",
            "joint_relex": "joint_relations",
            "structuring": "structures",
            "embedding": None,
        }
        tasks = [
            name for name in task_arguments
            if getattr(self.model.config, f"{name}_config", None) is not None
        ]
        return {
            "model": self.config.model,
            "model_variant": getattr(self.model.config, "model_variant", None),
            "tasks": tasks,
            "task_arguments": {
                name: argument for name, argument in task_arguments.items() if name in tasks
            },
            "polylora": self.adapter_cache_status(),
            "sequence_packing": {
                "requested": self.config.enable_sequence_packing,
                "enabled": self.packing_config is not None,
                "max_length": (
                    self.packing_config.max_length
                    if self.packing_config is not None else None
                ),
            },
            "limits": {
                "max_batch_size": self.config.max_batch_size,
                "max_model_len": self.config.max_model_len,
                "max_text_chars": self.config.max_text_chars,
                "max_labels": self.config.max_labels,
            },
        }


async def _read_limited_body(request, max_bytes: int) -> bytes:
    """Read an ASGI request incrementally without buffering an oversized body."""
    body = bytearray()
    async for chunk in request.stream():
        if not chunk:
            continue
        if len(body) + len(chunk) > max_bytes:
            raise OverflowError(f"Request exceeds {max_bytes} bytes")
        body.extend(chunk)
    return bytes(body)


def build_deployment(config: GLiFormerServeConfig):
    """Build a Ray Serve deployment without importing Ray for library users."""
    try:
        from ray import serve
        from starlette.responses import JSONResponse
    except ImportError as exc:
        raise ImportError("Serving requires `pip install 'gliformer[serve]'`") from exc

    @serve.deployment(
        num_replicas=config.num_replicas,
        ray_actor_options={
            "num_gpus": config.num_gpus_per_replica,
            "num_cpus": config.num_cpus_per_replica,
        },
        max_ongoing_requests=config.max_ongoing_requests,
        max_queued_requests=config.queue_capacity,
    )
    class GLiFormerDeployment:
        def __init__(self, serve_config: GLiFormerServeConfig):
            self.server = GLiFormerServer(serve_config)
            self._infer.set_max_batch_size(self.server.batch_size_fn())

        @serve.batch(
            max_batch_size=config.max_batch_size,
            batch_wait_timeout_s=max(config.batch_wait_timeout_ms, 0.0) / 1000.0,
        )
        async def _infer(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
            self._infer.set_max_batch_size(
                self.server.batch_size_fn(self.server.observed_seq_len(payloads))
            )
            return self.server.predict_payloads(payloads, isolate_errors=True)

        @serve.batch(
            max_batch_size=config.max_batch_size,
            batch_wait_timeout_s=max(config.batch_wait_timeout_ms, 0.0) / 1000.0,
        )
        async def _embed(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
            self._embed.set_max_batch_size(
                self.server.batch_size_fn(self.server.observed_seq_len(payloads))
            )
            return self.server.predict_embedding_payloads(
                payloads,
                isolate_errors=True,
            )

        async def predict(self, text: str, adapter_id: str | None = None, **tasks):
            payload = {"text": text, **tasks}
            if adapter_id is not None:
                payload["adapter_id"] = adapter_id
            result = await self._infer(payload)
            if result.get(_BATCH_ERROR_KEY):
                error = result["error"]
                if result["status"] == 404:
                    raise KeyError(error)
                raise ValueError(error)
            return result

        async def __call__(self, request):
            path = request.url.path.rstrip("/")
            if request.method == "GET" and (
                path.endswith("/health") or path.endswith("/ready")
            ):
                return JSONResponse(self.server.health())
            if request.method == "GET" and path.endswith("/metadata"):
                return JSONResponse(self.server.metadata())
            if request.method == "GET" and request.url.path.rstrip("/").endswith("adapter-cache"):
                try:
                    return JSONResponse(
                        self.server.adapter_cache_status(request.query_params.get("adapter_id"))
                    )
                except (KeyError, ValueError) as exc:
                    return JSONResponse({"error": str(exc)}, status_code=404)
            if request.method != "POST":
                return JSONResponse({"error": "Use POST for inference"}, status_code=405)
            try:
                content_length = request.headers.get("content-length")
                if content_length and int(content_length) > config.max_request_bytes:
                    return JSONResponse(
                        {"error": f"Request exceeds {config.max_request_bytes} bytes"},
                        status_code=413,
                    )
                raw_body = await _read_limited_body(request, config.max_request_bytes)
                payload = json.loads(raw_body)
                if path.endswith("/embeddings"):
                    result = await self._embed(payload)
                else:
                    result = await self._infer(payload)
                if result.get(_BATCH_ERROR_KEY):
                    return JSONResponse(
                        {"error": result["error"]},
                        status_code=result["status"],
                    )
                return JSONResponse(result)
            except OverflowError as exc:
                return JSONResponse({"error": str(exc)}, status_code=413)
            except KeyError as exc:
                return JSONResponse({"error": str(exc)}, status_code=404)
            except (TypeError, ValueError) as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)

    return GLiFormerDeployment.bind(config)


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
    """Lifecycle-managed synchronous/async facade over Ray Serve.

    List inputs are dispatched concurrently as individual Ray requests so the
    deployment batcher, rather than the caller, controls physical batches.
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

    def predict(self, texts: str | list[str], **tasks):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        request_tasks = self._tasks_for_items(tasks, len(items))
        refs = [
            self._handle.predict.remote(text, **item_tasks)
            for text, item_tasks in zip(items, request_tasks, strict=True)
        ]
        values = [ref.result() for ref in refs]
        return values[0] if single else values

    def predict_requests(self, requests: list[dict[str, Any]]):
        request_tasks = self._validate_request_items(requests)
        refs = [
            self._handle.predict.remote(item.pop("text"), **item)
            for item in request_tasks
        ]
        return [ref.result() for ref in refs]

    async def predict_async(self, texts: str | list[str], **tasks):
        import asyncio

        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        request_tasks = self._tasks_for_items(tasks, len(items))
        values = list(await asyncio.gather(
            *(
                self._handle.predict.remote(text, **item_tasks)
                for text, item_tasks in zip(items, request_tasks, strict=True)
            )
        ))
        return values[0] if single else values

    async def predict_requests_async(self, requests: list[dict[str, Any]]):
        import asyncio

        request_tasks = self._validate_request_items(requests)
        return list(await asyncio.gather(*(
            self._handle.predict.remote(item.pop("text"), **item)
            for item in request_tasks
        )))

    @staticmethod
    def _validate_request_items(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        validated = []
        for index, request in enumerate(requests):
            if not isinstance(request, dict):
                raise TypeError(f"requests[{index}] must be an object")
            if not isinstance(request.get("text"), str):
                raise TypeError(f"requests[{index}] must contain a string 'text'")
            validated.append(dict(request))
        return validated

    @staticmethod
    def _tasks_for_items(tasks: dict[str, Any], count: int) -> list[dict[str, Any]]:
        tasks = dict(tasks)
        per_text_tasks = tasks.pop("per_text_tasks", None)
        expanded = [dict(tasks) for _ in range(count)]
        if per_text_tasks is not None:
            if len(per_text_tasks) != count:
                raise ValueError(f"per_text_tasks must have length {count}")
            if any(not isinstance(item, dict) for item in per_text_tasks):
                raise TypeError("Every per_text_tasks item must be an object")
            for index, item_tasks in enumerate(per_text_tasks):
                if "text" in item_tasks:
                    raise ValueError("per_text_tasks cannot override text")
                expanded[index].update(item_tasks)
        for key, value in tasks.items():
            if isinstance(value, PerText):
                for index, item_value in enumerate(value.expand(count, key)):
                    expanded[index][key] = item_value
        adapter_ids = tasks.get("adapter_id")
        if isinstance(adapter_ids, list):
            if len(adapter_ids) != count:
                raise ValueError(f"adapter_id list must have length {count}")
            for index, adapter_id in enumerate(adapter_ids):
                expanded[index]["adapter_id"] = adapter_id
        for key in ("entities", "classes", "relations"):
            value = tasks.get(key)
            if isinstance(value, PerText):
                continue
            if isinstance(value, list) and value and isinstance(value[0], (list, dict)):
                if len(value) != count:
                    raise ValueError(f"Per-text {key} must have length {count}")
                for index, item_value in enumerate(value):
                    expanded[index][key] = item_value
        return expanded

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
