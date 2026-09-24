"""Production serving API for GLiFormer.

Features include Ray dynamic batching, CUDA-memory-aware batch sizing,
compilation warmup, multi-task text inference, and per-row PolyLoRA routing.
"""

from .client import GLiFormerClient, GLiFormerClientError, PerText, get_client
from .config import GLiFormerServeConfig
from .memory import GLiFormerMemoryEstimator
from .server import (
    GLiFormerFactory,
    GLiFormerServer,
    build_deployment,
    serve,
    shutdown,
)

__all__ = [
    "GLiFormerClient",
    "GLiFormerClientError",
    "PerText",
    "get_client",
    "GLiFormerServeConfig",
    "GLiFormerMemoryEstimator",
    "GLiFormerFactory",
    "GLiFormerServer",
    "build_deployment",
    "serve",
    "shutdown",
]
