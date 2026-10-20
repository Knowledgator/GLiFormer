"""Render an analyze ``input`` into one Markdown text (NEW_API.md §3).

The renderer keeps the character range of every leaf, so a prediction
located in the rendered text can be reported as a JSON Pointer to the leaf
and offsets relative to the leaf's value.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from typing import Any

from .errors import APIError, invalid, pointer

MEDIA_TYPES = ("image", "audio", "layout")
MAX_HEADING_DEPTH = 6


@dataclass(frozen=True)
class Leaf:
    """One input value rendered verbatim at ``[start, end)`` of the text."""

    pointer: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Location:
    """A span mapped back to one input leaf."""

    source: str
    start: int
    end: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "source": self.source,
        }


@dataclass
class RenderedInput:
    text: str
    leaves: list[Leaf] = field(default_factory=list)
    media: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._starts = [leaf.start for leaf in self.leaves]

    def locate(self, start: int, end: int) -> Location | None:
        """Map a rendered ``[start, end)`` span to its leaf.

        Spans that are empty, touch headings or list markers, or cross two
        leaves have no location and return ``None``.
        """
        if start >= end:
            return None
        index = bisect.bisect_right(self._starts, start) - 1
        if index < 0:
            return None
        leaf = self.leaves[index]
        if start < leaf.start or end > leaf.end:
            return None
        local_start = start - leaf.start
        local_end = end - leaf.start
        return Location(
            source=leaf.pointer,
            start=local_start,
            end=local_end,
            text=leaf.text[local_start:local_end],
        )

    def find(self, needle: str) -> list[Location]:
        """Locations of every occurrence of ``needle`` inside one leaf."""
        if not needle:
            return []
        found = []
        for leaf in self.leaves:
            offset = leaf.text.find(needle)
            while offset >= 0:
                found.append(Location(
                    source=leaf.pointer,
                    start=offset,
                    end=offset + len(needle),
                    text=needle,
                ))
                offset = leaf.text.find(needle, offset + 1)
        return found


def _scalar_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _heading_text(key: str) -> str:
    return " ".join(str(key).splitlines()) if key else ""


class _Renderer:
    def __init__(self, supported_media: tuple[str, ...] = ()) -> None:
        self.parts: list[str] = []
        self.position = 0
        self.leaves: list[Leaf] = []
        self.media: list[tuple[str, dict[str, Any]]] = []
        self.supported_media = supported_media

    def emit(self, text: str, path: list[Any] | None = None) -> None:
        if path is not None:
            self.leaves.append(Leaf(
                pointer=pointer(*path),
                start=self.position,
                end=self.position + len(text),
                text=text,
            ))
        self.parts.append(text)
        self.position += len(text)

    def heading(self, depth: int, title: str) -> None:
        self.emit("#" * min(depth, MAX_HEADING_DEPTH) + " " + title)

    # -- validation ----------------------------------------------------

    def check_value(self, value: Any, path: list[Any]) -> None:
        if _is_scalar(value):
            if isinstance(value, float) and value != value:
                raise invalid("NaN is not valid JSON", pointer("input", *path))
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                self.check_value(item, [*path, index])
            return
        if isinstance(value, dict):
            if "$type" in value:
                self.check_media(value, path)
                return
            for key, item in value.items():
                if not isinstance(key, str):
                    raise invalid("Object keys must be strings", pointer("input", *path))
                self.check_value(item, [*path, key])
            return
        raise invalid(
            f"Unsupported input value of type {type(value).__name__}",
            pointer("input", *path),
        )

    def check_media(self, node: dict[str, Any], path: list[Any]) -> None:
        media_type = node.get("$type")
        where = pointer("input", *path)
        if media_type not in MEDIA_TYPES:
            raise invalid(
                f"Unknown $type {media_type!r}; '$type' is reserved for "
                f"media nodes ({', '.join(MEDIA_TYPES)})",
                where + "/$type",
            )
        if media_type not in self.supported_media:
            raise APIError(
                "UNSUPPORTED_MODALITY",
                f"Input modality '{media_type}' is not supported by this deployment",
                where,
                {"modality": media_type, "supported": ["text", *self.supported_media]},
            )
        if any(kind == media_type for kind, _ in self.media):
            raise invalid(
                f"At most one '{media_type}' node is allowed per input", where
            )
        self.media.append((media_type, node))

    # -- rendering -----------------------------------------------------

    def render_root(self, value: Any) -> None:
        if _is_scalar(value):
            self.emit(_scalar_text(value), [])
        elif isinstance(value, dict) and "$type" in value:
            pass  # media tokens carry it; no text
        elif isinstance(value, dict):
            self.render_object(value, 1, [], top=True)
        elif isinstance(value, list):
            self.render_list(value, 0, [], leading_newline=False)

    def render_object(self, obj: dict[str, Any], depth: int, path: list[Any], *, top: bool) -> None:
        for index, (key, value) in enumerate(obj.items()):
            if index:
                self.emit("\n\n" if top else "\n")
            self.heading(depth, _heading_text(key))
            self.render_body(value, depth, [*path, key])

    def render_body(self, value: Any, depth: int, path: list[Any]) -> None:
        """Content that follows a heading at ``depth``."""
        if _is_scalar(value):
            self.emit("\n")
            self.emit(_scalar_text(value), path)
        elif isinstance(value, dict) and "$type" in value:
            pass  # heading only
        elif isinstance(value, dict):
            if value:
                self.emit("\n")
                self.render_object(value, depth + 1, path, top=False)
        elif isinstance(value, list):
            self.render_list(value, depth, path, leading_newline=True)

    def render_list(self, items: list[Any], depth: int, path: list[Any], *, leading_newline: bool) -> None:
        if not items:
            return
        if all(_is_scalar(item) for item in items):
            for index, item in enumerate(items):
                if index or leading_newline:
                    self.emit("\n")
                self.emit("- ")
                self.emit(_scalar_text(item), [*path, index])
            return
        for index, item in enumerate(items):
            if index or leading_newline:
                self.emit("\n")
            self.heading(depth + 1, str(index + 1))
            self.render_body(item, depth + 1, [*path, index])


def render_input(value: Any, *, supported_media: tuple[str, ...] = ()) -> RenderedInput:
    """Validate ``value`` against the input grammar and render it."""
    renderer = _Renderer(supported_media)
    renderer.check_value(value, [])
    renderer.media.clear()
    renderer.render_root(value)
    return RenderedInput(
        text="".join(renderer.parts),
        leaves=renderer.leaves,
        media=renderer.media,
    )
