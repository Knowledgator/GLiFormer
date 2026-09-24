"""HTTP client for the GLiFormer Ray Serve deployment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_ROUTE_PREFIX = "/gliformer"


class GLiFormerClientError(RuntimeError):
    """Raised when GLiFormer Serve rejects a request or is unreachable."""


@dataclass(frozen=True)
class PerText:
    """Explicitly mark values that correspond one-to-one with input texts.

    This avoids guessing whether a nested list is one shared schema or a list
    of schemas. It is interpreted by the Python client/factory and is never
    serialized into an HTTP request.
    """

    values: list[Any] | tuple[Any, ...]

    def expand(self, count: int, name: str) -> list[Any]:
        values = list(self.values)
        if len(values) != count:
            raise ValueError(
                f"PerText {name} must have length {count}, got {len(values)}"
            )
        return values


class GLiFormerClient:
    """Synchronous and asynchronous client for all GLiFormer text tasks.

    Lists are dispatched as concurrent single-text requests, allowing Ray
    Serve's dynamic batcher to combine compatible schemas while preserving
    independent adapter ids.

    Example:
        >>> client = GLiFormerClient()
        >>> client.predict(
        ...     "Alice works at Acme",
        ...     entities=["person", "organization"],
        ... )
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        route_prefix: str = DEFAULT_ROUTE_PREFIX,
        timeout: float = 30.0,
        max_concurrency: int = 32,
    ) -> None:
        self.url = base_url.rstrip("/") + route_prefix
        self.timeout = timeout
        self.max_concurrency = max(1, max_concurrency)

    @staticmethod
    def _adapters_for_texts(
        adapter_id: str | list[str | None] | None,
        count: int,
    ) -> list[str | None]:
        if isinstance(adapter_id, list):
            if len(adapter_id) != count:
                raise ValueError(
                    f"Per-text adapter ids must have length {count}, got {len(adapter_id)}"
                )
            return adapter_id
        return [adapter_id] * count

    @staticmethod
    def _build_payload(
        text: str,
        adapter_id: str | None,
        tasks: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {"text": text, **tasks}
        if adapter_id is not None:
            payload["adapter_id"] = adapter_id
        return payload

    @staticmethod
    def _per_text_value(
        value: Any,
        count: int,
        name: str,
        *,
        infer_nested: bool = True,
    ) -> list[Any]:
        """Expand shared schemas or validate an explicit per-text schema list."""
        if isinstance(value, PerText):
            return value.expand(count, name)
        if (
            infer_nested
            and isinstance(value, list)
            and value
            and isinstance(value[0], (list, dict))
        ):
            if len(value) != count:
                raise ValueError(
                    f"Per-text {name} must have length {count}, got {len(value)}"
                )
            return value
        return [value] * count

    def _post(self, payload: dict[str, Any], suffix: str = "") -> dict[str, Any]:
        import json
        import urllib.error
        import urllib.request

        request = urllib.request.Request(
            self.url.rstrip("/") + suffix,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise GLiFormerClientError(
                f"GLiFormer Serve returned HTTP {exc.code}: {body}"
            ) from exc
        except Exception as exc:
            raise GLiFormerClientError(f"Request to {self.url} failed: {exc}") from exc

    def _get(self, suffix: str) -> dict[str, Any]:
        import json
        import urllib.error
        import urllib.request

        url = self.url.rstrip("/") + suffix
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise GLiFormerClientError(
                f"GLiFormer Serve returned HTTP {exc.code}: {body}"
            ) from exc
        except Exception as exc:
            raise GLiFormerClientError(f"Request to {url} failed: {exc}") from exc

    def _post_many(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not payloads:
            return []
        if len(payloads) == 1:
            return [self._post(payloads[0])]
        from concurrent.futures import ThreadPoolExecutor

        workers = min(self.max_concurrency, len(payloads))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self._post, payloads))

    def predict_requests(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Send fully independent task/schema payloads in input order."""
        payloads = []
        for index, request in enumerate(requests):
            if not isinstance(request, dict):
                raise TypeError(f"requests[{index}] must be an object")
            if not isinstance(request.get("text"), str):
                raise TypeError(f"requests[{index}] must contain a string 'text'")
            payloads.append(dict(request))
        return self._post_many(payloads)

    async def predict_requests_async(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        import asyncio

        return await asyncio.to_thread(self.predict_requests, requests)

    def predict(
        self,
        text: str | list[str],
        *,
        entities: Any = None,
        classes: Any = None,
        relations: Any = None,
        joint_relations: Any = None,
        structures: Any = None,
        embed: bool = False,
        threshold: float | None = None,
        flat_ner: bool = True,
        multi_label: bool = False,
        adapter_id: str | list[str | None] | None = None,
        per_text_tasks: list[dict[str, Any]] | None = None,
        **options: Any,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Run any combination of text tasks.

        A string returns one result dictionary. A list returns a result list in
        input order. ``adapter_id`` may be shared or contain one id per text.
        """
        if embed:
            if any(value is not None for value in (
                entities, classes, relations, joint_relations, structures
            )):
                raise ValueError("Embeddings use a separate request; call embed_text()")
            return self.embed_text(text, adapter_id=adapter_id)
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        if not texts:
            return []
        if per_text_tasks is not None:
            if single:
                raise ValueError("per_text_tasks is only valid for a list of texts")
            if len(per_text_tasks) != len(texts):
                raise ValueError(
                    f"per_text_tasks must have length {len(texts)}, got {len(per_text_tasks)}"
                )
            if any(not isinstance(item, dict) for item in per_text_tasks):
                raise TypeError("Every per_text_tasks item must be an object")
        tasks = {
            "flat_ner": flat_ner,
            "multi_label": multi_label,
            **options,
        }
        tasks = {key: value for key, value in tasks.items() if value is not None}
        if threshold is not None:
            tasks["threshold"] = threshold
        adapters = self._adapters_for_texts(adapter_id, len(texts))
        entity_values = self._per_text_value(entities, len(texts), "entities")
        class_values = self._per_text_value(classes, len(texts), "classes")
        relation_values = self._per_text_value(relations, len(texts), "relations")
        joint_relation_values = self._per_text_value(
            joint_relations,
            len(texts),
            "joint_relations",
            infer_nested=False,
        )
        structure_values = self._per_text_value(
            structures,
            len(texts),
            "structures",
            infer_nested=False,
        )
        payloads = [
            self._build_payload(
                item,
                adapter,
                {
                    **tasks,
                    **({"entities": item_entities} if item_entities is not None else {}),
                    **({"classes": item_classes} if item_classes is not None else {}),
                    **({"relations": item_relations} if item_relations is not None else {}),
                    **(
                        {"joint_relations": item_joint_relations}
                        if item_joint_relations is not None else {}
                    ),
                    **({"structures": item_structures} if item_structures is not None else {}),
                },
            )
            for (
                item,
                adapter,
                item_entities,
                item_classes,
                item_relations,
                item_joint_relations,
                item_structures,
            ) in zip(
                texts,
                adapters,
                entity_values,
                class_values,
                relation_values,
                joint_relation_values,
                structure_values,
                strict=True,
            )
        ]
        if per_text_tasks is not None:
            for payload, overrides in zip(payloads, per_text_tasks, strict=True):
                if "text" in overrides:
                    raise ValueError("per_text_tasks cannot override text")
                payload.update(overrides)
        results = self._post_many(payloads)
        return results[0] if single else results

    async def predict_async(
        self,
        text: str | list[str],
        **kwargs: Any,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Async facade preserving all validation and per-text routing."""
        import asyncio

        return await asyncio.to_thread(self.predict, text, **kwargs)

    def adapter_cache_status(self, adapter_id: str | None = None) -> dict[str, Any]:
        """Return PolyLoRA CPU, disk, and GPU residency information."""
        import urllib.parse

        suffix = "/adapter-cache"
        if adapter_id is not None:
            suffix += "?" + urllib.parse.urlencode({"adapter_id": adapter_id})
        return self._get(suffix)

    def is_adapter_cached(self, adapter_id: str) -> bool:
        return bool(self.adapter_cache_status(adapter_id).get("cached"))

    def health(self) -> dict[str, Any]:
        """Return model readiness and active serving configuration."""
        return self._get("/health")

    def metadata(self) -> dict[str, Any]:
        """Describe loaded heads, limits, model variant, and adapter state."""
        return self._get("/metadata")

    def predict_entities(self, text, entities, **kwargs):
        result = self.predict(text, entities=entities, **kwargs)
        if isinstance(text, str):
            return result.get("ner", [])
        return [item.get("ner", []) for item in result]

    def classify(self, text, classes, **kwargs):
        result = self.predict(text, classes=classes, **kwargs)
        if isinstance(text, str):
            return result.get("classification", [])
        return [item.get("classification", []) for item in result]

    def predict_joint_relations(self, text, joint_relations, **kwargs):
        result = self.predict(text, joint_relations=joint_relations, **kwargs)
        if isinstance(text, str):
            return result.get("joint_relex", [])
        return [item.get("joint_relex", []) for item in result]

    def structure(self, text, structures, **kwargs):
        result = self.predict(text, structures=structures, **kwargs)
        if isinstance(text, str):
            return result.get("structuring", {})
        return [item.get("structuring", {}) for item in result]

    def embed_text(
        self,
        text: str | list[str],
        *,
        adapter_id: str | list[str | None] | None = None,
    ):
        """Produce embeddings through the dedicated ``/embeddings`` route."""
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        adapters = self._adapters_for_texts(adapter_id, len(texts))
        payloads = [
            self._build_payload(value, adapter, {})
            for value, adapter in zip(texts, adapters, strict=True)
        ]
        if len(payloads) == 1:
            results = [self._post(payloads[0], "/embeddings")]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(self.max_concurrency, len(payloads))) as pool:
                results = list(pool.map(lambda payload: self._post(payload, "/embeddings"), payloads))
        embeddings = [result["embedding"] for result in results]
        return embeddings[0] if single else embeddings


def get_client(
    base_url: str = DEFAULT_BASE_URL,
    route_prefix: str = DEFAULT_ROUTE_PREFIX,
    timeout: float = 30.0,
    max_concurrency: int = 32,
) -> GLiFormerClient:
    """Construct a :class:`GLiFormerClient` with explicit connection limits."""
    return GLiFormerClient(
        base_url=base_url,
        route_prefix=route_prefix,
        timeout=timeout,
        max_concurrency=max_concurrency,
    )
