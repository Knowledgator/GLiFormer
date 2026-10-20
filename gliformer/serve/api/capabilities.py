"""What the loaded checkpoint can run (NEW_API.md §13.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TEXT_FORMATS = ("number", "integer", "duration", "date", "time", "datetime")
PREDICTED_FIELD_TYPES = (
    "options", "number", "integer", "duration", "date", "time", "datetime", "estimate",
)
BASE_FIELD_TYPES = ("text", "boolean")
ALL_TASK_TYPES = (
    "classify", "boolean", "score", "span", "relation", "json",
    "embedding", "rank", "box", "mask", "count", "time_span",
)
RESERVED_TASK_TYPES = ("text", "link")


@dataclass
class Capabilities:
    model: str
    aliases: list[str] = field(default_factory=list)
    modalities: list[str] = field(default_factory=lambda: ["text"])
    task_types: list[str] = field(default_factory=list)
    json_field_types: list[str] = field(default_factory=list)
    json_text_formats: list[str] = field(default_factory=list)
    typed_heads: bool = False
    relation_head: str | None = None          # "joint_relex" | "open_relex"
    label_template: str = "{title}"
    default_thresholds: dict[str, float] = field(default_factory=dict)
    limits: dict[str, int] = field(default_factory=dict)
    adapters_enabled: bool = False
    supported_label_keys: frozenset[str] = frozenset()

    def has_task(self, task_type: str) -> bool:
        return task_type in self.task_types

    def model_ids(self) -> list[str]:
        return [self.model, *self.aliases]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "aliases": list(self.aliases),
            "modalities": list(self.modalities),
            "task_types": list(self.task_types),
            "json_field_types": list(self.json_field_types),
            "json_text_formats": list(self.json_text_formats),
            "label_template": self.label_template,
            "default_thresholds": dict(self.default_thresholds),
            # Non-positive limits mean "unlimited" and are reported as null.
            "limits": {key: (value if value and value > 0 else None) for key, value in self.limits.items()},
            "adapters": {"enabled": self.adapters_enabled},
        }


def _head(model_config: Any, name: str) -> Any:
    return getattr(model_config, f"{name}_config", None)


def capabilities_from_model(model: Any, serve_config: Any) -> Capabilities:
    """Derive capabilities from the model's configured heads."""
    model_config = getattr(model, "config", None)
    task_types: list[str] = []
    if _head(model_config, "classification") is not None:
        task_types += ["classify", "boolean", "score"]
    if _head(model_config, "ner") is not None:
        task_types.append("span")
    relation_head = None
    if _head(model_config, "joint_relex") is not None:
        relation_head = "joint_relex"
    elif _head(model_config, "open_relex") is not None:
        relation_head = "open_relex"
    if relation_head is not None:
        task_types.append("relation")
    structuring = _head(model_config, "structuring")
    typed = bool(structuring is not None and getattr(structuring, "typed_fields", False))
    json_field_types: list[str] = []
    json_text_formats: list[str] = []
    if structuring is not None:
        task_types.append("json")
        json_field_types = list(BASE_FIELD_TYPES)
        json_text_formats = list(TEXT_FORMATS)
        if typed:
            json_field_types += list(PREDICTED_FIELD_TYPES)
    if _head(model_config, "embedding") is not None:
        task_types += ["embedding", "rank"]
    ordered = [name for name in ALL_TASK_TYPES if name in task_types]

    thresholds = {
        name: float(value)
        for name, value in serve_config.default_thresholds.items()
        if name in ordered
    }
    limits = {
        "max_input_tokens": int(serve_config.max_model_len),
        "max_input_chars": int(serve_config.max_input_chars),
        "max_input_bytes": int(serve_config.max_request_bytes),
        "max_tasks": int(serve_config.max_tasks),
        "max_labels_per_task": int(serve_config.max_labels),
        "max_batch_items": int(serve_config.max_batch_items),
    }
    return Capabilities(
        model=serve_config.model_id or serve_config.model,
        aliases=[
            alias for alias in [serve_config.model, *serve_config.model_aliases]
            if alias != (serve_config.model_id or serve_config.model)
        ],
        modalities=["text"],
        task_types=ordered,
        json_field_types=json_field_types,
        json_text_formats=json_text_formats,
        typed_heads=typed,
        relation_head=relation_head,
        label_template=serve_config.label_template,
        default_thresholds=thresholds,
        limits=limits,
        adapters_enabled=bool(serve_config.enable_polylora),
    )
