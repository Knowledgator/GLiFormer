"""Configuration for GLiFormer inference serving."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GLiFormerServeConfig:
    """Model, HTTP, batching, and PolyLoRA serving options."""

    model: str = "knowledgator/gliformer-base-v1"
    # Public model ID reported by the API (defaults to ``model``) and extra
    # names accepted in the request's ``model`` field.
    model_id: str | None = None
    model_aliases: list[str] = field(default_factory=list)
    device: str = "cuda"
    dtype: str = "bfloat16"
    quantization: str | None = None
    max_model_len: int = 2048
    max_span_width: int = 12
    max_labels: int = -1
    max_input_chars: int = 1_000_000
    max_request_bytes: int = 10_000_000
    max_tasks: int = 32
    max_batch_items: int = 256
    # Cutoffs used when a task sets no ``threshold``. Single-label selections
    # (classify, boolean, ordinal score) apply no cutoff without one.
    default_thresholds: dict[str, float] = field(default_factory=lambda: {
        "classify": 0.5, "span": 0.5, "relation": 0.5, "json": 0.5,
    })
    # Text the model sees for a label: "{title}" or e.g. "{title}: {description}".
    label_template: str = "{title}"
    # Render task names as group names (off: tasks are anonymous groups).
    render_task_names: bool = False
    default_batch_size: int = 8
    tokenizer_threads: int = 4
    enable_compilation: bool = True
    compilation_backend: str = "inductor"
    enable_sequence_packing: bool = False
    enable_flashdeberta: bool = False
    precompiled_batch_sizes: list[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32]
    )
    warmup_iterations: int = 3
    enable_memory_calibration: bool = True
    target_memory_fraction: float = 0.8
    memory_overhead_factor: float = 1.3
    calibration_min_seq_len: int = 64
    calibration_probe_batch_size: int = 2

    num_replicas: int = 1
    num_gpus_per_replica: float = 1.0
    num_cpus_per_replica: float = 1.0
    max_batch_size: int = 32
    batch_wait_timeout_ms: float = 5.0
    max_ongoing_requests: int = 256
    queue_capacity: int = 4096
    route_prefix: str = "/v1"
    http_port: int = 8000
    ray_address: str | None = None
    min_available_host_memory_gb: float = 2.0

    enable_polylora: bool = False
    polylora_adapters: dict[str, str] = field(default_factory=dict)
    polylora_adapter_weight_modules: list[str] | None = None
    polylora_max_rank: int = 16
    polylora_max_gpu_adapters: int = 8
    polylora_max_cpu_adapters: int | None = 128
    polylora_disk_cache_dir: str | None = None
    polylora_max_disk_adapters: int | None = None
    polylora_base_adapter_id: str = "__base__"
    polylora_use_triton_kernels: bool = True
    polylora_enforce_right_padding: bool = True
    polylora_adapter_id_pattern: str = r"^[A-Za-z0-9_.-]{1,128}$"

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if self.default_batch_size < 1:
            raise ValueError("default_batch_size must be at least 1")
        if self.max_input_chars < 1 or self.max_request_bytes < 1:
            raise ValueError("request size limits must be positive")
        from .api.labels import validate_template

        validate_template(self.label_template)
        for name, value in self.default_thresholds.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"default_thresholds[{name!r}] must be in [0, 1]")
        self.default_thresholds = {
            name: float(value) for name, value in self.default_thresholds.items()
        }
        if not self.route_prefix.startswith("/"):
            self.route_prefix = "/" + self.route_prefix
        if not 0 < self.target_memory_fraction <= 1:
            raise ValueError("target_memory_fraction must be in (0, 1]")
        if self.compilation_backend not in {"inductor", "aot_eager", "eager"}:
            raise ValueError("compilation_backend must be inductor, aot_eager, or eager")
        self.precompiled_batch_sizes = sorted({
            size
            for size in (*self.precompiled_batch_sizes, self.max_batch_size)
            if 0 < size <= self.max_batch_size
        })

    def to_env_vars(self) -> dict[str, str]:
        """Environment settings that must be applied before model loading."""
        env: dict[str, str] = {}
        if self.tokenizer_threads > 0:
            env["TOKENIZERS_PARALLELISM"] = "true"
        if self.enable_flashdeberta:
            env["USE_FLASHDEBERTA"] = "1"
        return env
