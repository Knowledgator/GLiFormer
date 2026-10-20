"""Error codes of the unified analyze API (NEW_API.md §13.5)."""

from __future__ import annotations

from typing import Any

ERROR_STATUS = {
    "INVALID_REQUEST": 422,
    "UNSUPPORTED_TASK_TYPE": 422,
    "UNSUPPORTED_MODALITY": 422,
    "UNSUPPORTED_LABEL_TYPE": 422,
    "UNSUPPORTED_FIELD": 422,
    "INPUT_TOO_LONG": 413,
    "PAYLOAD_TOO_LARGE": 413,
    "MODEL_NOT_FOUND": 404,
    "UNKNOWN_ADAPTER": 404,
    "INTERNAL": 500,
}


def escape_pointer_token(token: Any) -> str:
    """RFC 6901 escaping of one JSON Pointer reference token."""
    return str(token).replace("~", "~0").replace("/", "~1")


def pointer(*tokens: Any) -> str:
    """Join reference tokens into a JSON Pointer (``""`` for the root)."""
    return "".join("/" + escape_pointer_token(token) for token in tokens)


class APIError(Exception):
    """A request-level error with a stable code, HTTP status and JSON path."""

    def __init__(
        self,
        code: str,
        message: str,
        path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if code not in ERROR_STATUS:
            raise ValueError(f"Unknown API error code {code!r}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.path = path
        self.details = details

    @property
    def status(self) -> int:
        return ERROR_STATUS[self.code]

    def to_dict(self) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.path is not None:
            error["path"] = self.path
        if self.details:
            error["details"] = self.details
        return {"error": error}

    def with_prefix(self, prefix: str) -> APIError:
        """The same error with its path nested under ``prefix``."""
        return APIError(
            self.code,
            self.message,
            prefix + (self.path or ""),
            self.details,
        )

    def __repr__(self) -> str:
        return f"APIError({self.code!r}, {self.message!r}, path={self.path!r})"


def invalid(message: str, path: str | None = None, **details: Any) -> APIError:
    return APIError("INVALID_REQUEST", message, path, details or None)
