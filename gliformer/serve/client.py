"""HTTP client for the GLiFormer analyze API (``/v1``)."""

from __future__ import annotations

import json
from typing import Any

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_ROUTE_PREFIX = "/v1"


class GLiFormerClientError(RuntimeError):
    """The server could not be reached or returned an unexpected response."""


class GLiFormerAPIError(GLiFormerClientError):
    """The server rejected a request with an API error object."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int | None = None,
        path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}" + (f" (at {path})" if path else ""))
        self.code = code
        self.message = message
        self.status = status
        self.path = path
        self.details = details

    @classmethod
    def from_payload(cls, payload: dict[str, Any], status: int | None = None) -> GLiFormerAPIError:
        error = payload.get("error") or {}
        return cls(
            error.get("code", "INTERNAL"),
            error.get("message", ""),
            status=status,
            path=error.get("path"),
            details=error.get("details"),
        )


class TaskResult(dict):
    """One task result; keys are also readable as attributes (``result.value``)."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class AnalyzeResult(dict):
    """Task results keyed by task name, plus ``model``, ``usage`` and ``raw``."""

    def __init__(self, response: dict[str, Any]) -> None:
        super().__init__({
            name: TaskResult(result)
            for name, result in (response.get("results") or {}).items()
        })
        self.model = response.get("model")
        self.usage = response.get("usage")
        self.raw = response


def _schema_from(schema: Any) -> Any:
    """Accept a JSON Schema dict or a Pydantic model class."""
    json_schema = getattr(schema, "model_json_schema", None)
    if callable(json_schema):
        return json_schema()
    legacy = getattr(schema, "schema", None)
    if isinstance(schema, type) and callable(legacy):
        return legacy()
    return schema


class GLiFormerClient:
    """Synchronous and asynchronous client for ``/v1/analyze``.

    Example:
        >>> client = GLiFormerClient("http://localhost:8000")
        >>> result = client.analyze(
        ...     "Hi, I'm Tom Baker. I need a refund.",
        ...     tasks={"pii": {"type": "span", "labels": ["person"]}},
        ... )
        >>> result["pii"].value
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        route_prefix: str = DEFAULT_ROUTE_PREFIX,
        *,
        model: str | None = None,
        timeout: float = 60.0,
        max_concurrency: int = 32,
    ) -> None:
        self.url = base_url.rstrip("/") + "/" + route_prefix.strip("/")
        self.timeout = timeout
        self.max_concurrency = max(1, max_concurrency)
        self._model = model

    # -- transport ------------------------------------------------------------

    def _request(self, method: str, suffix: str, payload: Any = None) -> Any:
        import urllib.error
        import urllib.request

        url = self.url + suffix
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
                raise GLiFormerAPIError.from_payload(parsed, exc.code) from exc
            raise GLiFormerClientError(f"GLiFormer Serve returned HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise GLiFormerClientError(f"Request to {url} failed: {exc}") from exc

    def _post(self, suffix: str, payload: Any) -> Any:
        return self._request("POST", suffix, payload)

    def _get(self, suffix: str) -> Any:
        return self._request("GET", suffix)

    # -- model -----------------------------------------------------------------

    @property
    def model(self) -> str:
        """The served model ID (looked up once when not given)."""
        if self._model is None:
            models = self.models().get("data") or []
            if not models:
                raise GLiFormerClientError("The server lists no models")
            self._model = models[0]["id"]
        return self._model

    def models(self) -> dict[str, Any]:
        return self._get("/models")

    def capabilities(self, model: str | None = None) -> dict[str, Any]:
        import urllib.parse

        model_id = urllib.parse.quote(model or self.model, safe="/")
        return self._get(f"/models/{model_id}/capabilities")

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def ready(self) -> dict[str, Any]:
        return self._get("/ready")

    def adapter_cache_status(self, adapter_id: str | None = None) -> dict[str, Any]:
        import urllib.parse

        suffix = "/adapter-cache"
        if adapter_id is not None:
            suffix += "?" + urllib.parse.urlencode({"adapter_id": adapter_id})
        return self._get(suffix)

    # -- analyze ------------------------------------------------------------------

    def body(
        self,
        input: Any,
        tasks: dict[str, Any],
        *,
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model or self.model, "input": input, "tasks": tasks}
        if options:
            body["options"] = options
        return body

    def analyze(
        self,
        input: Any,
        tasks: dict[str, Any],
        *,
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> AnalyzeResult:
        """One input, many named tasks. Raises ``GLiFormerAPIError`` when rejected."""
        return AnalyzeResult(self._post("/analyze", self.body(input, tasks, options=options, model=model)))

    def analyze_batch(
        self,
        items: list[dict[str, Any]],
        *,
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> list[AnalyzeResult | GLiFormerAPIError]:
        """Independent ``{id?, input, tasks, options?}`` items, in order.

        Each element is an ``AnalyzeResult`` (with ``.id``) or the item's
        ``GLiFormerAPIError``.
        """
        body: dict[str, Any] = {"model": model or self.model, "items": items}
        if options:
            body["options"] = options
        response = self._post("/analyze/batch", body)
        outcomes: list[AnalyzeResult | GLiFormerAPIError] = []
        for item in response.get("items", []):
            if "error" in item:
                outcome: Any = GLiFormerAPIError.from_payload(item)
            else:
                outcome = AnalyzeResult({"model": response.get("model"), **item})
            outcome.id = item.get("id")
            outcomes.append(outcome)
        return outcomes

    async def analyze_async(self, input: Any, tasks: dict[str, Any], **kwargs: Any) -> AnalyzeResult:
        import asyncio

        return await asyncio.to_thread(self.analyze, input, tasks, **kwargs)

    async def analyze_batch_async(self, items: list[dict[str, Any]], **kwargs: Any):
        import asyncio

        return await asyncio.to_thread(self.analyze_batch, items, **kwargs)

    # -- convenience wrappers ---------------------------------------------------------

    def _single(self, input: Any, task: dict[str, Any], options: dict[str, Any] | None) -> TaskResult:
        task = {key: value for key, value in task.items() if value is not None}
        return self.analyze(input, {"result": task}, options=options)["result"]

    def classify(
        self,
        input: Any,
        labels: Any,
        *,
        multi_label: bool = False,
        instruction: str | None = None,
        threshold: float | None = None,
        options: dict[str, Any] | None = None,
    ) -> TaskResult:
        return self._single(input, {
            "type": "classify", "labels": labels, "multi_label": multi_label,
            "instruction": instruction, "threshold": threshold,
        }, options)

    def judge(
        self,
        input: Any,
        instruction: str,
        *,
        labels: Any = None,
        threshold: float | None = None,
        options: dict[str, Any] | None = None,
    ) -> TaskResult:
        """A ``boolean`` decision."""
        return self._single(input, {
            "type": "boolean", "instruction": instruction, "labels": labels, "threshold": threshold,
        }, options)

    def extract_spans(
        self,
        input: Any,
        labels: Any = None,
        *,
        instruction: str | None = None,
        threshold: float | None = None,
        flat: bool | None = None,
        options: dict[str, Any] | None = None,
    ) -> TaskResult:
        return self._single(input, {
            "type": "span", "labels": labels, "instruction": instruction,
            "threshold": threshold, "flat": flat,
        }, options)

    def extract_relations(
        self,
        input: Any,
        labels: Any,
        *,
        entity_labels: Any = None,
        instruction: str | None = None,
        threshold: float | None = None,
        options: dict[str, Any] | None = None,
    ) -> TaskResult:
        return self._single(input, {
            "type": "relation", "labels": labels, "entity_labels": entity_labels,
            "instruction": instruction, "threshold": threshold,
        }, options)

    def extract_json(
        self,
        input: Any,
        labels: Any = None,
        *,
        schema: Any = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> TaskResult:
        """``labels`` (flat fields) or ``schema`` (JSON Schema or Pydantic model)."""
        return self._single(input, {
            "type": "json", "labels": labels,
            "schema": None if schema is None else _schema_from(schema),
            "instruction": instruction,
        }, options)

    def embed(
        self,
        input: Any,
        *,
        normalize: bool = True,
        options: dict[str, Any] | None = None,
    ) -> list[float]:
        return self._single(input, {"type": "embedding", "normalize": normalize}, options).value

    def rank(
        self,
        input: Any,
        candidates: list[dict[str, Any]],
        *,
        top_k: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> TaskResult:
        return self._single(input, {"type": "rank", "candidates": candidates, "top_k": top_k}, options)


def get_client(
    base_url: str = DEFAULT_BASE_URL,
    route_prefix: str = DEFAULT_ROUTE_PREFIX,
    timeout: float = 60.0,
    max_concurrency: int = 32,
) -> GLiFormerClient:
    """Construct a :class:`GLiFormerClient`."""
    return GLiFormerClient(
        base_url=base_url,
        route_prefix=route_prefix,
        timeout=timeout,
        max_concurrency=max_concurrency,
    )
