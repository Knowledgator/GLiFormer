"""FastAPI routes of the analyze API (mounted under ``route_prefix``).

Kept apart from ``server.py`` so FastAPI is only imported when serving, and
so route annotations resolve without postponed evaluation.
"""

import asyncio
import json
import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .api import APIError

logger = logging.getLogger(__name__)


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


async def read_json_body(request, max_bytes: int) -> Any:
    """Parse a JSON request body, enforcing the byte limit (``PAYLOAD_TOO_LARGE``)."""
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > max_bytes:
        raise APIError("PAYLOAD_TOO_LARGE", f"Request body exceeds {max_bytes} bytes")
    try:
        raw_body = await _read_limited_body(request, max_bytes)
    except OverflowError as exc:
        raise APIError("PAYLOAD_TOO_LARGE", str(exc)) from exc
    try:
        return json.loads(raw_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise APIError("INVALID_REQUEST", f"Request body is not valid JSON: {exc}", "") from exc


def error_response(exc: APIError) -> JSONResponse:
    return JSONResponse(exc.to_dict(), status_code=exc.status)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def _api_error(_request, exc: APIError):
        return error_response(exc)

    @app.exception_handler(Exception)
    async def _internal_error(_request, exc: Exception):
        logger.exception("Unhandled analyze API error", exc_info=exc)
        return error_response(APIError("INTERNAL", f"{type(exc).__name__}: {exc}"))


# Route bodies shared by the standalone app and the Ray deployment. ``server``
# is a ``GLiFormerServer``; ``analyze_one`` runs one analyze body (through
# Ray's dynamic batcher when deployed) and returns a response or ``APIError``.


async def analyze_route(request: Request, analyze_one, max_request_bytes: int) -> JSONResponse:
    body = await read_json_body(request, max_request_bytes)
    outcome = await analyze_one(body)
    if isinstance(outcome, APIError):
        return error_response(outcome)
    return JSONResponse(outcome)


async def run_batch(server, analyze_one, body: Any) -> dict:
    items, model = server.engine.parse_batch(body)
    outcomes = await asyncio.gather(*(
        analyze_one({**item_body, "model": model}) if isinstance(item_body, dict)
        else analyze_one(item_body)
        for _, item_body in items
    ))
    return server.engine.batch_response(items, model, outcomes)


async def analyze_batch_route(request: Request, server, analyze_one, max_request_bytes: int) -> JSONResponse:
    body = await read_json_body(request, max_request_bytes)
    return JSONResponse(await run_batch(server, analyze_one, body))


def adapter_cache_response(server, adapter_id: str | None) -> JSONResponse:
    try:
        return JSONResponse(server.adapter_cache_status(adapter_id))
    except KeyError as exc:
        raise APIError("UNKNOWN_ADAPTER", str(exc).strip("'\""), "/adapter_id") from exc
    except ValueError as exc:
        raise APIError("INVALID_REQUEST", str(exc), "/adapter_id") from exc


def build_app(get_server, analyze_one, max_request_bytes: int) -> FastAPI:
    """Standalone FastAPI app of the analyze API (e.g. for tests or uvicorn).

    Mount it under the route prefix (``/v1``).
    """
    app = FastAPI(title="GLiFormer Analyze API", version="1.0")
    install_error_handlers(app)

    @app.post("/analyze")
    async def analyze(request: Request):
        return await analyze_route(request, analyze_one, max_request_bytes)

    @app.post("/analyze/batch")
    async def analyze_batch(request: Request):
        return await analyze_batch_route(request, get_server(), analyze_one, max_request_bytes)

    @app.get("/models")
    async def models():
        return JSONResponse(get_server().models())

    @app.get("/models/{model_id:path}/capabilities")
    async def capabilities(model_id: str):
        return JSONResponse(get_server().capabilities(model_id))

    @app.get("/health")
    async def health():
        return JSONResponse(get_server().health())

    @app.get("/ready")
    async def ready():
        return JSONResponse(get_server().health())

    @app.get("/adapter-cache")
    async def adapter_cache(adapter_id: str | None = None):
        return adapter_cache_response(get_server(), adapter_id)

    return app


def build_ray_deployment(config, server_cls):
    """Ray Serve deployment with the analyze routes as methods (one server per replica)."""
    from ray import serve

    app = FastAPI(title="GLiFormer Analyze API", version="1.0")
    install_error_handlers(app)
    max_request_bytes = config.max_request_bytes

    @serve.deployment(
        num_replicas=config.num_replicas,
        ray_actor_options={
            "num_gpus": config.num_gpus_per_replica,
            "num_cpus": config.num_cpus_per_replica,
        },
        max_ongoing_requests=config.max_ongoing_requests,
        max_queued_requests=config.queue_capacity,
    )
    @serve.ingress(app)
    class GLiFormerDeployment:
        def __init__(self, serve_config):
            self.server = server_cls(serve_config)
            self._analyze.set_max_batch_size(self.server.batch_size_fn())

        @serve.batch(
            max_batch_size=config.max_batch_size,
            batch_wait_timeout_s=max(config.batch_wait_timeout_ms, 0.0) / 1000.0,
        )
        async def _analyze(self, bodies: list) -> list:
            texts = [
                body["input"] for body in bodies
                if isinstance(body, dict) and isinstance(body.get("input"), str)
            ]
            self._analyze.set_max_batch_size(
                self.server.batch_size_fn(self.server.observed_seq_len(texts))
            )
            return await asyncio.to_thread(self.server.analyze_many, bodies)

        @app.post("/analyze")
        async def analyze(self, request: Request):
            return await analyze_route(request, self._analyze, max_request_bytes)

        @app.post("/analyze/batch")
        async def analyze_batch(self, request: Request):
            return await analyze_batch_route(request, self.server, self._analyze, max_request_bytes)

        @app.get("/models")
        async def models(self):
            return JSONResponse(self.server.models())

        @app.get("/models/{model_id:path}/capabilities")
        async def capabilities(self, model_id: str):
            return JSONResponse(self.server.capabilities(model_id))

        @app.get("/health")
        async def health(self):
            return JSONResponse(self.server.health())

        @app.get("/ready")
        async def ready(self):
            return JSONResponse(self.server.health())

        @app.get("/adapter-cache")
        async def adapter_cache(self, adapter_id: str | None = None):
            return adapter_cache_response(self.server, adapter_id)

        # Handle-level entry points used by ``GLiFormerFactory``.

        async def handle_analyze(self, body: Any) -> dict:
            outcome = await self._analyze(body)
            if isinstance(outcome, APIError):
                return {**outcome.to_dict(), "status": outcome.status}
            return outcome

        async def handle_analyze_batch(self, body: Any) -> dict:
            try:
                return await run_batch(self.server, self._analyze, body)
            except APIError as exc:
                return {**exc.to_dict(), "status": exc.status}

        async def handle_capabilities(self) -> dict:
            return self.server.capabilities()

    return GLiFormerDeployment.bind(config)
