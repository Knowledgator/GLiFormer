"""CLI entry point for GLiFormer Ray Serve.

Example:
    python -m gliformer.serve --model knowledgator/gliformer-base-v1
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def _parse_adapter(value: str) -> tuple[str, str]:
    parts = value.split("=", 1)
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError("adapter must be ID=PATH")
    return parts[0], parts[1]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start GLiFormer Ray Serve deployment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model",
        default="knowledgator/gliformer-base-v1",
        help="Hugging Face model id or local model directory",
    )
    model_group.add_argument(
        "--device", default="cuda", help="Model device (cuda, cuda:N, or cpu)"
    )
    model_group.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "fp16", "bfloat16", "bf16"],
        help="Model weight and compute dtype",
    )
    model_group.add_argument(
        "--quantization", choices=["int8"], help="Optional model quantization"
    )

    limits_group = parser.add_argument_group("Model Limits")
    limits_group.add_argument(
        "--max-model-len", type=int, default=2048, help="Maximum input sequence length"
    )
    limits_group.add_argument(
        "--max-span-width", type=int, default=12, help="Maximum extracted span width"
    )
    limits_group.add_argument(
        "--max-labels", type=int, default=-1, help="Maximum labels per simple task (-1 unlimited)"
    )
    limits_group.add_argument(
        "--max-text-chars", type=int, default=1_000_000, help="Maximum request text length"
    )
    limits_group.add_argument(
        "--max-request-bytes", type=int, default=2_000_000, help="Maximum HTTP body size"
    )

    threshold_group = parser.add_argument_group("Thresholds")
    threshold_group.add_argument(
        "--default-threshold", type=float, default=0.5, help="Default decoding threshold"
    )

    replica_group = parser.add_argument_group("Replica Configuration")
    replica_group.add_argument(
        "--num-replicas", type=int, default=1, help="Number of model replicas"
    )
    replica_group.add_argument(
        "--num-gpus-per-replica",
        type=float,
        default=None,
        help="GPU resources reserved by each replica; inferred from --device when omitted",
    )
    replica_group.add_argument(
        "--num-cpus-per-replica", type=float, default=1.0, help="CPU resources per replica"
    )

    batch_group = parser.add_argument_group("Batching Configuration")
    batch_group.add_argument(
        "--max-batch-size", type=int, default=32, help="Maximum Ray dynamic batch size"
    )
    batch_group.add_argument(
        "--default-batch-size",
        type=int,
        default=8,
        help="Internal fallback inference batch size",
    )
    batch_group.add_argument(
        "--batch-wait-timeout-ms",
        type=float,
        default=5.0,
        help="Maximum time Ray waits to accumulate a batch",
    )
    batch_group.add_argument(
        "--max-ongoing-requests",
        type=int,
        default=256,
        help="Maximum ongoing requests per replica",
    )
    batch_group.add_argument(
        "--queue-capacity", type=int, default=4096, help="Maximum queued HTTP requests"
    )
    batch_group.add_argument(
        "--precompiled-batch-sizes",
        type=_parse_int_list,
        default=[1, 2, 4, 8, 16, 32],
        help="Comma-separated batch sizes to warm up",
    )

    server_group = parser.add_argument_group("Server Configuration")
    server_group.add_argument(
        "--route-prefix", default="/gliformer", help="Ray Serve HTTP route prefix"
    )
    server_group.add_argument("--port", type=int, default=8000, help="HTTP listen port")
    server_group.add_argument(
        "--ray-address", default=None, help="Ray cluster address; local Ray when omitted"
    )

    performance_group = parser.add_argument_group("Performance Options")
    performance_group.add_argument(
        "--tokenizer-threads", type=int, default=4, help="PyTorch/tokenizer CPU threads"
    )
    performance_group.add_argument(
        "--no-compile", action="store_true", help="Disable torch.compile and shape warmup"
    )
    performance_group.add_argument(
        "--compile-backend",
        choices=["inductor", "aot_eager", "eager"],
        default="inductor",
        help="torch.compile backend; aot_eager has faster, safer startup",
    )
    performance_group.add_argument(
        "--enable-sequence-packing",
        action="store_true",
        help="Pack compatible short text requests with block-diagonal attention",
    )
    performance_group.add_argument(
        "--enable-flashdeberta",
        action="store_true",
        help="Enable FlashDeBERTa before loading a compatible backbone",
    )
    performance_group.add_argument(
        "--warmup-iterations", type=int, default=3, help="Warmup iterations per batch size"
    )

    memory_group = parser.add_argument_group("Memory Configuration")
    memory_group.add_argument(
        "--no-memory-calibration",
        action="store_true",
        help="Disable startup CUDA memory calibration",
    )
    memory_group.add_argument(
        "--target-memory-fraction",
        type=float,
        default=0.8,
        help="Fraction of free GPU memory available to inference batches",
    )
    memory_group.add_argument(
        "--memory-overhead-factor",
        type=float,
        default=1.3,
        help="Safety multiplier applied to calibrated per-sample memory",
    )
    memory_group.add_argument(
        "--calibration-min-seq-len",
        type=int,
        default=64,
        help="Shortest sequence used during startup calibration",
    )
    memory_group.add_argument(
        "--calibration-probe-batch-size",
        type=int,
        default=2,
        help="Probe batch size used during memory calibration",
    )
    memory_group.add_argument(
        "--min-available-host-memory-gb",
        type=float,
        default=2.0,
        help="Refuse startup below this free host RAM; zero disables the guard",
    )

    polylora_group = parser.add_argument_group("PolyLoRA Configuration")
    polylora_group.add_argument(
        "--enable-polylora", action="store_true", help="Enable per-request LoRA routing"
    )
    polylora_group.add_argument(
        "--polylora-adapter",
        action="append",
        type=_parse_adapter,
        default=[],
        metavar="ID=PATH",
        help="Preload an adapter; repeat for multiple adapters",
    )
    polylora_group.add_argument(
        "--polylora-adapter-weight-modules",
        type=_parse_str_list,
        default=None,
        help="Comma-separated target module names",
    )
    polylora_group.add_argument("--polylora-max-rank", type=int, default=16)
    polylora_group.add_argument("--polylora-max-gpu-adapters", type=int, default=8)
    polylora_group.add_argument("--polylora-max-cpu-adapters", type=int, default=128)
    polylora_group.add_argument("--polylora-disk-cache-dir", default=None)
    polylora_group.add_argument("--polylora-max-disk-adapters", type=int, default=None)
    polylora_group.add_argument("--polylora-base-adapter-id", default="__base__")
    polylora_group.add_argument(
        "--polylora-use-triton-kernels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use PolyLoRA Triton kernels",
    )

    args = parser.parse_args()
    num_gpus = args.num_gpus_per_replica
    if num_gpus is None:
        num_gpus = 0.0 if args.device.lower() == "cpu" else 1.0

    from .config import GLiFormerServeConfig
    from .server import serve

    config = GLiFormerServeConfig(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        quantization=args.quantization,
        max_model_len=args.max_model_len,
        max_span_width=args.max_span_width,
        max_labels=args.max_labels,
        max_text_chars=args.max_text_chars,
        max_request_bytes=args.max_request_bytes,
        default_threshold=args.default_threshold,
        num_replicas=args.num_replicas,
        num_gpus_per_replica=num_gpus,
        num_cpus_per_replica=args.num_cpus_per_replica,
        max_batch_size=args.max_batch_size,
        default_batch_size=args.default_batch_size,
        batch_wait_timeout_ms=args.batch_wait_timeout_ms,
        max_ongoing_requests=args.max_ongoing_requests,
        queue_capacity=args.queue_capacity,
        precompiled_batch_sizes=args.precompiled_batch_sizes,
        route_prefix=args.route_prefix,
        http_port=args.port,
        ray_address=args.ray_address,
        tokenizer_threads=args.tokenizer_threads,
        enable_compilation=not args.no_compile,
        compilation_backend=args.compile_backend,
        enable_sequence_packing=args.enable_sequence_packing,
        enable_flashdeberta=args.enable_flashdeberta,
        warmup_iterations=args.warmup_iterations,
        enable_memory_calibration=not args.no_memory_calibration,
        target_memory_fraction=args.target_memory_fraction,
        memory_overhead_factor=args.memory_overhead_factor,
        calibration_min_seq_len=args.calibration_min_seq_len,
        calibration_probe_batch_size=args.calibration_probe_batch_size,
        min_available_host_memory_gb=args.min_available_host_memory_gb,
        enable_polylora=args.enable_polylora,
        polylora_adapters=dict(args.polylora_adapter),
        polylora_adapter_weight_modules=args.polylora_adapter_weight_modules,
        polylora_max_rank=args.polylora_max_rank,
        polylora_max_gpu_adapters=args.polylora_max_gpu_adapters,
        polylora_max_cpu_adapters=args.polylora_max_cpu_adapters,
        polylora_disk_cache_dir=args.polylora_disk_cache_dir,
        polylora_max_disk_adapters=args.polylora_max_disk_adapters,
        polylora_base_adapter_id=args.polylora_base_adapter_id,
        polylora_use_triton_kernels=args.polylora_use_triton_kernels,
    )

    print("=" * 60)
    print("GLiFormer Ray Serve Configuration")
    print("=" * 60)
    print(f"Model: {config.model}")
    print(f"Device: {config.device}, dtype: {config.dtype}")
    print(f"Max batch size: {config.max_batch_size}")
    print(f"Precompiled batch sizes: {config.precompiled_batch_sizes}")
    print(f"Replicas: {config.num_replicas}")
    print(f"Endpoint: http://0.0.0.0:{config.http_port}{config.route_prefix}")
    print(f"Compilation: {'enabled' if config.enable_compilation else 'disabled'}")
    print(f"Sequence packing: {'enabled' if config.enable_sequence_packing else 'disabled'}")
    print(f"FlashDeBERTa: {'enabled' if config.enable_flashdeberta else 'disabled'}")
    print(f"Memory calibration: {'enabled' if config.enable_memory_calibration else 'disabled'}")
    print(f"PolyLoRA: {'enabled' if config.enable_polylora else 'disabled'}")
    print("=" * 60)

    serve(config, blocking=True)


if __name__ == "__main__":
    main()
