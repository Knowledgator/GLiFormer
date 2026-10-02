"""GPU-memory calibration and adaptive batch sizing for GLiFormer Serve."""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch

logger = logging.getLogger(__name__)


def calibration_lengths(max_length: int, minimum: int = 64) -> list[int]:
    lengths: list[int] = []
    value = max(1, minimum)
    while value < max_length:
        lengths.append(value)
        value *= 2
    lengths.append(max_length)
    return lengths


class GLiFormerMemoryEstimator:
    """Estimate a safe batch size from measured peak CUDA allocations."""

    def __init__(
        self,
        safety_factor: float = 1.3,
        target_memory_fraction: float = 0.8,
        calibration_probe_batch_size: int = 2,
    ) -> None:
        self.safety_factor = safety_factor
        self.target_memory_fraction = target_memory_fraction
        self.calibration_probe_batch_size = max(1, calibration_probe_batch_size)
        self.total_gpu_memory = 0
        self.cuda_context_bytes = 0
        self.model_memory_bytes = 0
        self.per_sample_table: dict[int, int] = {}

    def measure_cuda_context(self) -> None:
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        free, self.total_gpu_memory = torch.cuda.mem_get_info()
        self.cuda_context_bytes = self.total_gpu_memory - free

    def measure_model_memory(self) -> None:
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free, self.total_gpu_memory = torch.cuda.mem_get_info()
        self.model_memory_bytes = max(
            0, self.total_gpu_memory - free - self.cuda_context_bytes
        )

    def available_memory(self) -> int:
        available = (
            self.total_gpu_memory - self.cuda_context_bytes - self.model_memory_bytes
        )
        return max(0, int(available * self.target_memory_fraction))

    def calibrate(
        self,
        probe: Callable[[list[str]], object],
        max_seq_len: int,
        min_seq_len: int = 64,
    ) -> None:
        if not torch.cuda.is_available():
            return
        for seq_len in calibration_lengths(max_seq_len, min_seq_len):
            texts = ["word " * max(1, seq_len // 2)] * self.calibration_probe_batch_size
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()
            probe(texts)
            torch.cuda.synchronize()
            peak = max(1, torch.cuda.max_memory_allocated() - baseline)
            self.per_sample_table[seq_len] = max(
                1, peak // self.calibration_probe_batch_size
            )
            logger.info(
                "memory calibration seq_len=%d: %.1f MiB/sample",
                seq_len,
                self.per_sample_table[seq_len] / 1024**2,
            )

    def per_sample_at(self, seq_len: int) -> int:
        if not self.per_sample_table:
            raise RuntimeError("Memory estimator has not been calibrated")
        keys = sorted(self.per_sample_table)
        rounded = next((key for key in keys if key >= seq_len), keys[-1])
        return int(self.per_sample_table[rounded] * self.safety_factor)

    def batch_size_fn(self, seq_len: int, allowed_sizes: list[int]) -> int:
        if not allowed_sizes:
            return 1
        sizes = sorted(set(allowed_sizes))
        if not self.per_sample_table:
            return sizes[-1]
        if self.available_memory() <= 0:
            return sizes[0]
        per_sample = self.per_sample_at(seq_len)
        return next(
            (size for size in reversed(sizes) if per_sample * size <= self.available_memory()),
            sizes[0],
        )
