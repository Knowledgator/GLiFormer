"""Production serving for GLiFormer: the unified analyze API (``/v1``).

One input, many named tasks, one model (see ``NEW_API.md``). Features
include Ray dynamic batching across requests, CUDA-memory-aware batch sizing,
compilation warmup, and per-request PolyLoRA routing.
"""

from .api import AnalyzeEngine, APIError, Capabilities
from .client import (
    AnalyzeResult,
    GLiFormerAPIError,
    GLiFormerClient,
    GLiFormerClientError,
    TaskResult,
    get_client,
)
from .config import GLiFormerServeConfig
from .memory import GLiFormerMemoryEstimator
from .server import (
    GLiFormerFactory,
    GLiFormerServer,
    build_app,
    build_deployment,
    serve,
    shutdown,
)

__all__ = [
    "APIError",
    "AnalyzeEngine",
    "AnalyzeResult",
    "Capabilities",
    "GLiFormerAPIError",
    "GLiFormerClient",
    "GLiFormerClientError",
    "TaskResult",
    "get_client",
    "GLiFormerServeConfig",
    "GLiFormerMemoryEstimator",
    "GLiFormerFactory",
    "GLiFormerServer",
    "build_app",
    "build_deployment",
    "serve",
    "shutdown",
]
