"""Validation of analyze requests into typed task specs (NEW_API.md §4, §13.4)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from .capabilities import ALL_TASK_TYPES, RESERVED_TASK_TYPES, Capabilities
from .errors import APIError, invalid, pointer
from .jsonfields import FieldNode, fields_from_labels, fields_from_schema
from .labels import Label, normalize_labels
from .render import RenderedInput, render_input

OPTION_KEYS = frozenset({
    "include_label_definitions",
    "include_usage",
    "include_evidence",
    "include_layout_text",
    "reference_date",
    "adapter_id",
})
BOOLEAN_OPTIONS = (
    "include_label_definitions", "include_usage", "include_evidence", "include_layout_text",
)
COMMON_TASK_FIELDS = frozenset({"type", "instruction"})
TASK_FIELDS: dict[str, frozenset[str]] = {
    "classify": frozenset({"threshold", "labels", "multi_label"}),
    "boolean": frozenset({"threshold", "labels"}),
    "score": frozenset({"threshold", "labels", "order", "scale"}),
    "span": frozenset({"threshold", "labels", "flat"}),
    "relation": frozenset({"threshold", "labels", "entity_labels", "flat"}),
    "json": frozenset({"labels", "schema"}),
    "embedding": frozenset({"normalize"}),
    "rank": frozenset({"candidates", "top_k"}),
    "box": frozenset({"threshold", "labels"}),
    "mask": frozenset({"threshold", "labels", "mode"}),
    "count": frozenset({"labels"}),
    "time_span": frozenset({"threshold", "labels", "mode"}),
}
MODES = {
    "mask": ("semantic", "instance", "panoptic"),
    "time_span": ("events", "diarization"),
}
REQUEST_KEYS = frozenset({"model", "input", "tasks", "options"})
BOOLEAN_IDS = ("true", "false")


@dataclass
class Candidate:
    id: str
    rendered: RenderedInput


@dataclass
class TaskSpec:
    name: str
    type: str
    path: str
    instruction: str | None = None
    threshold: float | None = None
    labels: list[Label] | None = None
    multi_label: bool = False
    order: list[str] | None = None
    scale: dict[str, float] | None = None
    entity_labels: list[Label] | None = None
    flat: bool = True
    fields: FieldNode | None = None
    candidates: list[Candidate] | None = None
    top_k: int | None = None
    normalize: bool = True

    @property
    def single_label(self) -> bool:
        return self.type in ("boolean", "score") or (self.type == "classify" and not self.multi_label)


@dataclass
class Options:
    include_label_definitions: bool = False
    include_usage: bool = False
    include_evidence: bool = False
    include_layout_text: bool = False
    reference_date: str | None = None
    adapter_id: str | None = None


@dataclass
class ParsedRequest:
    model: str
    rendered: RenderedInput
    tasks: list[TaskSpec]
    options: Options
    raw_input: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


# -- helpers --------------------------------------------------------------------


def _threshold(value: Any, where: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise invalid("'threshold' must be a number", where)
    if not 0.0 <= float(value) <= 1.0:
        raise invalid("'threshold' must be between 0 and 1", where)
    return float(value)


def _bool(value: Any, name: str, where: str) -> bool:
    if not isinstance(value, bool):
        raise invalid(f"'{name}' must be a boolean", where)
    return value


def parse_options(raw: Any, path: str = "/options") -> Options:
    if raw is None:
        return Options()
    if not isinstance(raw, dict):
        raise invalid("'options' must be an object", path)
    for key in raw:
        if key not in OPTION_KEYS:
            raise invalid(f"Unknown option '{key}'", f"{path}/{key}")
    options = Options()
    for key in BOOLEAN_OPTIONS:
        if key in raw:
            setattr(options, key, _bool(raw[key], key, f"{path}/{key}"))
    if raw.get("reference_date") is not None:
        value = raw["reference_date"]
        if not isinstance(value, str):
            raise invalid("'reference_date' must be an ISO 8601 string", f"{path}/reference_date")
        try:
            (datetime if "T" in value else date).fromisoformat(value)
        except ValueError as exc:
            raise invalid(
                f"'reference_date' is not ISO 8601: {exc}", f"{path}/reference_date"
            ) from exc
        options.reference_date = value
    if raw.get("adapter_id") is not None:
        if not isinstance(raw["adapter_id"], str):
            raise invalid("'adapter_id' must be a string", f"{path}/adapter_id")
        options.adapter_id = raw["adapter_id"]
    return options


def merge_options(batch: Any, item: Any) -> Any:
    """Item-level options override batch-level ones."""
    if batch is None:
        return item
    if item is None:
        return batch
    if not isinstance(batch, dict) or not isinstance(item, dict):
        return item
    return {**batch, **item}


# -- request ----------------------------------------------------------------------


class RequestParser:
    """Validate requests against the contract and the model's capabilities."""

    def __init__(self, capabilities: Capabilities) -> None:
        self.capabilities = capabilities

    def check_model(self, model: Any, path: str = "/model") -> str:
        if not isinstance(model, str) or not model:
            raise invalid("'model' must be a non-empty string", path)
        if model not in self.capabilities.model_ids():
            raise APIError(
                "MODEL_NOT_FOUND",
                f"Model {model!r} is not served here",
                path,
                {"available": self.capabilities.model_ids()},
            )
        return self.capabilities.model

    def render(self, value: Any, path: str = "/input") -> RenderedInput:
        try:
            rendered = render_input(
                value,
                supported_media=tuple(
                    modality for modality in self.capabilities.modalities if modality != "text"
                ),
            )
        except APIError as exc:
            if exc.path and exc.path.startswith("/input") and path != "/input":
                raise APIError(exc.code, exc.message, path + exc.path[len("/input"):], exc.details) from exc
            raise
        limit = self.capabilities.limits.get("max_input_chars", 0)
        if limit > 0 and len(rendered.text) > limit:
            raise APIError(
                "INPUT_TOO_LONG",
                f"Rendered input has {len(rendered.text)} characters; the limit is {limit}",
                path,
            )
        return rendered

    def parse(self, body: Any, *, check_model: bool = True) -> ParsedRequest:
        if not isinstance(body, dict):
            raise invalid("Expected a JSON object", "")
        for key in body:
            if key not in REQUEST_KEYS:
                raise invalid(f"Unknown request field '{key}'", pointer(key))
        model = self.check_model(body.get("model")) if check_model else self.capabilities.model
        if "input" not in body:
            raise invalid("'input' is required", "/input")
        options = parse_options(body.get("options"))
        tasks_raw = body.get("tasks")
        if not isinstance(tasks_raw, dict) or not tasks_raw:
            raise invalid("'tasks' must be a non-empty object", "/tasks")
        max_tasks = self.capabilities.limits.get("max_tasks", 0)
        if max_tasks > 0 and len(tasks_raw) > max_tasks:
            raise invalid(f"At most {max_tasks} tasks are allowed per request", "/tasks")
        rendered = self.render(body["input"])
        tasks = [
            self.parse_task(name, raw, pointer("tasks", name))
            for name, raw in tasks_raw.items()
        ]
        return ParsedRequest(
            model=model,
            rendered=rendered,
            tasks=tasks,
            options=options,
            raw_input=body["input"],
        )

    # -- tasks ----------------------------------------------------------------

    def parse_task(self, name: str, raw: Any, path: str) -> TaskSpec:
        if not name:
            raise invalid("Task names must be non-empty", path)
        if not isinstance(raw, dict):
            raise invalid("A task must be an object", path)
        task_type = raw.get("type")
        if task_type is None:
            raise invalid("Every task needs a 'type'", f"{path}/type")
        if task_type in RESERVED_TASK_TYPES:
            raise APIError(
                "UNSUPPORTED_TASK_TYPE",
                f"Task type '{task_type}' is reserved and not served by GLiFormer",
                f"{path}/type",
            )
        if task_type not in ALL_TASK_TYPES:
            raise invalid(
                f"Unknown task type {task_type!r}; expected one of {', '.join(ALL_TASK_TYPES)}",
                f"{path}/type",
            )
        allowed = COMMON_TASK_FIELDS | TASK_FIELDS[task_type]
        for key in raw:
            if key not in allowed:
                raise invalid(f"Field '{key}' is not valid for '{task_type}' tasks", f"{path}/{key}")
        if not self.capabilities.has_task(task_type):
            raise APIError(
                "UNSUPPORTED_TASK_TYPE",
                f"The model has no head for '{task_type}' tasks",
                f"{path}/type",
                {"model": self.capabilities.model, "supported": list(self.capabilities.task_types)},
            )
        task = TaskSpec(name=name, type=task_type, path=path)
        instruction = raw.get("instruction")
        if instruction is not None:
            if not isinstance(instruction, str):
                raise invalid("'instruction' must be a string", f"{path}/instruction")
            task.instruction = " ".join(instruction.split()) or None
        if raw.get("threshold") is not None:
            task.threshold = _threshold(raw["threshold"], f"{path}/threshold")
        if "mode" in raw and raw["mode"] not in MODES.get(task_type, ()):
            raise invalid(
                f"Unsupported mode {raw['mode']!r} for '{task_type}'", f"{path}/mode"
            )
        getattr(self, f"parse_{task_type}")(task, raw, path)
        if task.labels is not None:
            self.check_label_limit(task.labels, f"{path}/labels")
        return task

    def labels(self, spec: Any, where: str) -> list[Label] | None:
        return normalize_labels(
            spec,
            where,
            template=self.capabilities.label_template,
            supported_keys=self.capabilities.supported_label_keys,
        )

    def check_label_limit(self, labels: list[Label], where: str) -> None:
        limit = self.capabilities.limits.get("max_labels_per_task", 0)
        if limit > 0 and len(labels) > limit:
            raise invalid(f"At most {limit} labels are allowed per task", where)

    def parse_classify(self, task: TaskSpec, raw: dict[str, Any], path: str) -> None:
        if "multi_label" in raw:
            task.multi_label = _bool(raw["multi_label"], "multi_label", f"{path}/multi_label")
        task.labels = self.labels(raw.get("labels"), f"{path}/labels")
        if not task.labels:
            raise invalid("'classify' tasks need at least one label", f"{path}/labels")

    def parse_boolean(self, task: TaskSpec, raw: dict[str, Any], path: str) -> None:
        given = self.labels(raw.get("labels"), f"{path}/labels") or []
        by_id = {label.id: label for label in given}
        for label in given:
            if label.id not in BOOLEAN_IDS:
                raise invalid(
                    "'boolean' labels may only define 'true' and 'false'",
                    f"{path}/labels/{label.id}",
                )
        labels = [
            by_id.get(label_id) or Label(id=label_id, text=label_id)
            for label_id in BOOLEAN_IDS
        ]
        if labels[0].text == labels[1].text:
            raise invalid("'true' and 'false' render to the same text", f"{path}/labels")
        task.labels = labels

    def parse_score(self, task: TaskSpec, raw: dict[str, Any], path: str) -> None:
        has_scale = raw.get("scale") is not None
        has_labels = raw.get("labels") is not None or raw.get("order") is not None
        if has_scale == has_labels:
            raise invalid("A 'score' task needs exactly one of 'scale' or 'labels' + 'order'", path)
        if has_scale:
            scale = raw["scale"]
            if not isinstance(scale, dict) or set(scale) - {"min", "max", "step"}:
                raise invalid("'scale' must be {min, max, step?}", f"{path}/scale")
            for key in ("min", "max"):
                if not isinstance(scale.get(key), (int, float)) or isinstance(scale.get(key), bool):
                    raise invalid(f"'scale.{key}' must be a number", f"{path}/scale/{key}")
            if scale["min"] >= scale["max"]:
                raise invalid("'scale.min' must be below 'scale.max'", f"{path}/scale")
            step = scale.get("step")
            if step is not None and (not isinstance(step, (int, float)) or step <= 0):
                raise invalid("'scale.step' must be a positive number", f"{path}/scale/step")
            if raw.get("threshold") is not None:
                raise invalid("'threshold' is not valid with 'scale'", f"{path}/threshold")
            if not self.capabilities.typed_heads:
                raise APIError(
                    "UNSUPPORTED_LABEL_TYPE",
                    "'score' with 'scale' needs typed structuring heads.",
                    f"{path}/scale",
                    {"model": self.capabilities.model},
                )
            task.scale = {key: scale[key] for key in ("min", "max", "step") if key in scale}
            return
        task.labels = self.labels(raw.get("labels"), f"{path}/labels")
        if not task.labels or len(task.labels) < 2:
            raise invalid("An ordinal 'score' needs at least two labels", f"{path}/labels")
        order = raw.get("order")
        ids = [label.id for label in task.labels]
        if (
            not isinstance(order, list)
            or not all(isinstance(item, str) for item in order)
            or sorted(order) != sorted(ids)
            or len(set(order)) != len(order)
        ):
            raise invalid("'order' must list every label ID exactly once", f"{path}/order")
        task.order = list(order)

    def parse_span(self, task: TaskSpec, raw: dict[str, Any], path: str) -> None:
        if "flat" in raw:
            task.flat = _bool(raw["flat"], "flat", f"{path}/flat")
        task.labels = self.labels(raw.get("labels"), f"{path}/labels")
        if task.labels == []:
            raise invalid("'labels' must not be empty; use null for open extraction", f"{path}/labels")

    def parse_relation(self, task: TaskSpec, raw: dict[str, Any], path: str) -> None:
        if "flat" in raw:
            task.flat = _bool(raw["flat"], "flat", f"{path}/flat")
        task.labels = self.labels(raw.get("labels"), f"{path}/labels")
        if not task.labels:
            raise invalid("'relation' tasks need at least one relation label", f"{path}/labels")
        if raw.get("entity_labels") is not None:
            if self.capabilities.relation_head != "joint_relex":
                raise APIError(
                    "UNSUPPORTED_FIELD",
                    "This model's relation head does not use 'entity_labels'",
                    f"{path}/entity_labels",
                )
            task.entity_labels = self.labels(raw["entity_labels"], f"{path}/entity_labels")
            if not task.entity_labels:
                raise invalid("'entity_labels' must not be empty", f"{path}/entity_labels")

    def parse_json(self, task: TaskSpec, raw: dict[str, Any], path: str) -> None:
        has_labels = raw.get("labels") is not None
        has_schema = raw.get("schema") is not None
        if has_labels == has_schema:
            raise invalid("A 'json' task needs exactly one of 'labels' or 'schema'", path)
        if has_labels:
            task.fields, task.labels = fields_from_labels(
                raw["labels"], f"{path}/labels", self.capabilities,
            )
        else:
            task.fields = fields_from_schema(raw["schema"], f"{path}/schema", self.capabilities)

    def parse_embedding(self, task: TaskSpec, raw: dict[str, Any], path: str) -> None:
        if "normalize" in raw:
            task.normalize = _bool(raw["normalize"], "normalize", f"{path}/normalize")

    def parse_rank(self, task: TaskSpec, raw: dict[str, Any], path: str) -> None:
        candidates = raw.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise invalid("'rank' tasks need a non-empty 'candidates' list", f"{path}/candidates")
        parsed: list[Candidate] = []
        seen: set[str] = set()
        for index, candidate in enumerate(candidates):
            where = f"{path}/candidates/{index}"
            if not isinstance(candidate, dict) or set(candidate) - {"id", "input"} or "input" not in candidate:
                raise invalid("A candidate must be {id, input}", where)
            candidate_id = candidate.get("id")
            if not isinstance(candidate_id, str) or not candidate_id:
                raise invalid("Candidate 'id' must be a non-empty string", f"{where}/id")
            if candidate_id in seen:
                raise invalid(f"Duplicate candidate ID {candidate_id!r}", f"{where}/id")
            seen.add(candidate_id)
            parsed.append(Candidate(candidate_id, self.render(candidate["input"], f"{where}/input")))
        top_k = raw.get("top_k")
        if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1):
            raise invalid("'top_k' must be a positive integer", f"{path}/top_k")
        task.candidates = parsed
        task.top_k = top_k

    def _media_task(self, task: TaskSpec, raw: dict[str, Any], path: str) -> None:
        raise APIError(
            "UNSUPPORTED_MODALITY",
            f"'{task.type}' tasks need media input, which this deployment does not serve",
            f"{path}/type",
        )

    parse_box = parse_mask = parse_count = parse_time_span = _media_task
