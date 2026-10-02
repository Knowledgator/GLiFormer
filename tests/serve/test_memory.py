import io

import pytest

from gliformer.serve.memory import GLiFormerMemoryEstimator, calibration_lengths
from gliformer.serve.server import _check_host_memory


def test_calibration_lengths_include_exact_maximum():
    assert calibration_lengths(300, 64) == [64, 128, 256, 300]


def test_memory_estimator_selects_largest_safe_configured_batch():
    estimator = GLiFormerMemoryEstimator(safety_factor=1.0, target_memory_fraction=1.0)
    estimator.total_gpu_memory = 1_000
    estimator.per_sample_table = {64: 100, 128: 250}
    assert estimator.batch_size_fn(64, [1, 2, 4, 8]) == 8
    assert estimator.batch_size_fn(128, [1, 2, 4, 8]) == 4


def test_memory_estimator_uses_smallest_batch_when_no_memory_is_available():
    estimator = GLiFormerMemoryEstimator(safety_factor=1.0, target_memory_fraction=1.0)
    estimator.total_gpu_memory = 100
    estimator.model_memory_bytes = 100
    estimator.per_sample_table = {64: 10}
    assert estimator.batch_size_fn(64, [1, 2, 4]) == 1


def test_host_memory_guard_fails_before_ray_start(monkeypatch):
    contents = "MemTotal: 1048576 kB\nMemAvailable: 524288 kB\n"
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.StringIO(contents))
    with pytest.raises(RuntimeError, match="Refusing to start Ray Serve"):
        _check_host_memory(1.0)
