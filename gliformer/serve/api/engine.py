"""Plan model rows for analyze requests, run them together, assemble results.

One request becomes a few model rows:

* the **main row** carries every ``classify`` / ``boolean`` / ordinal ``score``
  group and the ``span`` / ``relation`` groups (a second row holds tasks with
  ``flat: false``, since overlap control is per row);
* each ``json`` task (and ``score`` with ``scale``) gets its own row, because
  structuring in a shared row competes with the extraction groups for spans;
* inputs longer than the window run ``span`` / ``relation`` over overlapping
  window rows.

Rows of many requests run in one ``inference_requests`` call, so compatible
work shares model forwards.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .capabilities import Capabilities
from .errors import APIError, invalid
from .jsonfields import JsonShaper, model_template
from .labels import Label
from .render import RenderedInput
from .request import Options, ParsedRequest, RequestParser, TaskSpec, merge_options

logger = logging.getLogger(__name__)

ALL_SCORES = -1.0          # classification threshold that keeps every label
PROMPT_MARGIN = 8          # special tokens around prompt and text
OPEN_SPAN_LABEL = "answer"
OPEN_SPAN_LABEL_NO_INSTRUCTION = "entity"
DEFAULT_ENTITY_LABEL = "entity"
SCORE_FIELD = "score"


@dataclass
class Group:
    """One label group of a row and the task it belongs to."""

    key: str
    task: TaskSpec
    index: int                               # position in the model's output groups
    text_to_id: dict[str, str]
    entity_text_to_id: dict[str, str] | None = None


@dataclass
class Row:
    request: dict[str, Any]
    offset: int = 0                          # char offset of the row text in the rendered input
    classification: list[Group] = field(default_factory=list)
    extraction: list[Group] = field(default_factory=list)   # NER groups then relation groups
    json_task: TaskSpec | None = None
    output: dict[str, Any] | None = None


@dataclass
class Plan:
    parsed: ParsedRequest
    rows: list[Row] = field(default_factory=list)
    embed_texts: list[str] = field(default_factory=list)
    embeddings: list[list[float]] | None = None
    adapter_id: str | None = None
    started: float = field(default_factory=time.perf_counter)
    token_count: int = 0
    error: APIError | None = None


def _round(score: float) -> float:
    return round(float(score), 4)


def _greedy_flat(spans: list[dict[str, Any]], key=lambda span: (span["_start"], span["_end"])) -> list[dict[str, Any]]:
    """Keep the highest-scoring spans that do not overlap."""
    kept: list[dict[str, Any]] = []
    for span in sorted(spans, key=lambda item: -item["score"]):
        start, end = key(span)
        if all(end <= key(other)[0] or start >= key(other)[1] for other in kept):
            kept.append(span)
    return kept


class AnalyzeEngine:
    """Validate, plan, run, and assemble analyze requests."""

    def __init__(
        self,
        capabilities: Capabilities,
        *,
        run_rows: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
        embed: Callable[[list[str], list[str | None]], list[list[float]]] | None = None,
        count_tokens: Callable[[str], int] | None = None,
        resolve_adapter: Callable[[str | None], str | None] | None = None,
        render_task_names: bool = False,
        window_overlap: float = 0.25,
    ) -> None:
        self.capabilities = capabilities
        self.parser = RequestParser(capabilities)
        self.run_rows = run_rows
        self.embed = embed
        self.count_tokens = count_tokens or (lambda text: len(text.split()))
        self.resolve_adapter = resolve_adapter or (lambda adapter_id: adapter_id)
        self.render_task_names = render_task_names
        self.window_overlap = window_overlap

    # -- public ---------------------------------------------------------------

    def analyze(self, body: Any) -> dict[str, Any]:
        result = self.analyze_many([body])[0]
        if isinstance(result, APIError):
            raise result
        return result

    def analyze_many(self, bodies: Sequence[Any], *, check_model: bool = True) -> list[dict[str, Any] | APIError]:
        """Run independent requests together; each succeeds or fails alone."""
        outcomes: list[dict[str, Any] | APIError | None] = [None] * len(bodies)
        plans: list[tuple[int, Plan]] = []
        for index, body in enumerate(bodies):
            try:
                parsed = self.parser.parse(body, check_model=check_model)
                plans.append((index, self.plan(parsed)))
            except APIError as exc:
                outcomes[index] = exc
        if plans:
            self.execute([plan for _, plan in plans])
            for index, plan in plans:
                try:
                    outcomes[index] = plan.error or self.assemble(plan)
                except APIError as exc:
                    outcomes[index] = exc
                except Exception as exc:  # noqa: BLE001 - reported per request
                    logger.exception("Failed to assemble analyze results")
                    outcomes[index] = APIError("INTERNAL", f"Result assembly failed: {exc}")
        return outcomes  # type: ignore[return-value]

    def analyze_batch(self, body: Any) -> dict[str, Any]:
        """``/v1/analyze/batch``: many items, per-item errors, order kept."""
        items, model = self.parse_batch(body)
        bodies = [item_body for _, item_body in items]
        outcomes = self.analyze_many(bodies, check_model=False)
        return self.batch_response(items, model, outcomes)

    @staticmethod
    def batch_response(
        items: list[tuple[Any, Any]],
        model: str,
        outcomes: Sequence[dict[str, Any] | APIError],
    ) -> dict[str, Any]:
        """Assemble ``/v1/analyze/batch`` output from per-item outcomes."""
        response_items = []
        for (item_id, _), (index, outcome) in zip(items, enumerate(outcomes), strict=True):
            entry: dict[str, Any] = {}
            if item_id is not None:
                entry["id"] = item_id
            if isinstance(outcome, APIError):
                entry.update(outcome.with_prefix(f"/items/{index}").to_dict())
            else:
                entry["results"] = outcome["results"]
                if "usage" in outcome:
                    entry["usage"] = outcome["usage"]
            response_items.append(entry)
        return {"model": model, "items": response_items}

    def parse_batch(self, body: Any) -> tuple[list[tuple[Any, Any]], str]:
        if not isinstance(body, dict):
            raise invalid("The request body must be a JSON object", "")
        for key in body:
            if key not in ("model", "items", "options"):
                raise invalid(f"Unknown batch field '{key}'", f"/{key}")
        model = self.parser.check_model(body.get("model"))
        raw_items = body.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raise invalid("'items' must be a non-empty list", "/items")
        limit = self.capabilities.limits.get("max_batch_items", 0)
        if limit > 0 and len(raw_items) > limit:
            raise invalid(f"At most {limit} items are allowed per batch", "/items")
        batch_options = body.get("options")
        if batch_options is not None and not isinstance(batch_options, dict):
            raise invalid("'options' must be an object", "/options")
        items = []
        seen_ids: set[Any] = set()
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                items.append((None, item))
                continue
            item_id = item.get("id")
            if item_id is not None:
                if not isinstance(item_id, str):
                    raise invalid("Item 'id' must be a string", f"/items/{index}/id")
                if item_id in seen_ids:
                    raise invalid(f"Duplicate item ID {item_id!r}", f"/items/{index}/id")
                seen_ids.add(item_id)
            item_body = {key: value for key, value in item.items() if key != "id"}
            item_body["options"] = merge_options(batch_options, item.get("options"))
            if item_body["options"] is None:
                item_body.pop("options")
            items.append((item_id, item_body))
        return items, model

    # -- planning ---------------------------------------------------------------

    def threshold_for(self, task: TaskSpec) -> float:
        if task.threshold is not None:
            return task.threshold
        return float(self.capabilities.default_thresholds.get(task.type, 0.5))

    def reported_threshold(self, task: TaskSpec) -> float | None:
        """Threshold reported in results; ``None`` when no cutoff applies."""
        if task.single_label:
            return task.threshold
        return self.threshold_for(task)

    def group_prompt(self, task: TaskSpec) -> dict[str, Any]:
        return {
            "name": task.name if self.render_task_names else None,
            "description": task.instruction,
        }

    def prompt_tokens(self, texts: list[str | None], label_count: int, group_count: int) -> int:
        words = " ".join(text for text in texts if text)
        return self.count_tokens(words) + 2 * label_count + 3 * group_count if words else 0

    def plan(self, parsed: ParsedRequest) -> Plan:
        plan = Plan(parsed=parsed)
        options = parsed.options
        try:
            plan.adapter_id = self.resolve_adapter(options.adapter_id)
        except KeyError as exc:
            raise APIError(
                "UNKNOWN_ADAPTER", str(exc).strip("'\""), "/options/adapter_id"
            ) from exc
        except ValueError as exc:
            raise invalid(str(exc), "/options/adapter_id") from exc
        rendered = parsed.rendered
        text = rendered.text
        limit = int(self.capabilities.limits.get("max_input_tokens", 0) or 0)
        text_tokens = self.count_tokens(text)
        plan.token_count = text_tokens

        def fits(prompt_tokens: int, tokens: int = text_tokens) -> bool:
            return limit <= 0 or tokens + prompt_tokens + PROMPT_MARGIN <= limit

        def too_long(task: TaskSpec) -> APIError:
            return APIError(
                "INPUT_TOO_LONG",
                f"The rendered input ({text_tokens} tokens) exceeds the model window "
                f"({limit} tokens) for '{task.type}' task '{task.name}'",
                "/input",
                {"task": task.name, "tokens": text_tokens, "limit": limit},
            )

        classification = [task for task in parsed.tasks if task.type in ("classify", "boolean") or (task.type == "score" and task.scale is None)]
        extraction = [task for task in parsed.tasks if task.type in ("span", "relation")]
        structured = [task for task in parsed.tasks if task.type == "json" or (task.type == "score" and task.scale is not None)]
        embedding = [task for task in parsed.tasks if task.type in ("embedding", "rank")]

        def task_prompt(tasks: list[TaskSpec]) -> int:
            labels = [
                label.text
                for task in tasks
                for label in (*(task.labels or []), *(task.entity_labels or []))
            ]
            return self.prompt_tokens(
                [task.instruction for task in tasks] + labels, len(labels), len(tasks),
            )

        base_request = {
            "reference_date": options.reference_date,
            "adapter_id": plan.adapter_id,
        }
        cls_prompt = task_prompt(classification)
        for task in classification:
            if not fits(cls_prompt):
                raise too_long(task)

        # Row groups: flat spans share the main row with classification;
        # overlapping spans need their own row (overlap control is per row);
        # each relation task runs alone, since other label groups in the
        # prompt lower its entity scores.
        span_flat = [task for task in extraction if task.type == "span" and task.flat]
        span_loose = [task for task in extraction if task.type == "span" and not task.flat]
        relation_tasks = [task for task in extraction if task.type == "relation"]
        main_spans = span_flat if span_flat and fits(cls_prompt + task_prompt(span_flat)) else []
        if classification or main_spans:
            plan.rows.append(self.extraction_row(text, 0, classification, main_spans, base_request))
        groups = [] if main_spans or not span_flat else [span_flat]
        if span_loose:
            groups.append(span_loose)
        groups.extend([task] for task in relation_tasks)
        for tasks in groups:
            prompt = task_prompt(tasks)
            if fits(prompt):
                plan.rows.append(self.extraction_row(text, 0, [], tasks, base_request))
                continue
            if not fits(prompt, 1):
                raise too_long(tasks[0])
            for start, end in self.windows(text, limit - prompt - PROMPT_MARGIN):
                plan.rows.append(self.extraction_row(text[start:end], start, [], tasks, base_request))

        for task in structured:
            row = self.json_row(task, text, base_request)
            prompt = self.prompt_tokens([task.instruction, *self.field_names(task)], len(self.field_names(task)), 1)
            if not fits(prompt):
                raise too_long(task)
            plan.rows.append(row)

        for task in embedding:
            if not fits(0):
                raise too_long(task)
            if self.embed is None:
                raise APIError("UNSUPPORTED_TASK_TYPE", "Embeddings are not available", task.path + "/type")
            if text not in plan.embed_texts:
                plan.embed_texts.append(text)
            for candidate in task.candidates or []:
                if limit > 0 and self.count_tokens(candidate.rendered.text) + PROMPT_MARGIN > limit:
                    raise APIError(
                        "INPUT_TOO_LONG",
                        f"Candidate {candidate.id!r} exceeds the model window",
                        f"{task.path}/candidates",
                    )
                if candidate.rendered.text not in plan.embed_texts:
                    plan.embed_texts.append(candidate.rendered.text)
        return plan

    @staticmethod
    def field_names(task: TaskSpec) -> list[str]:
        names: list[str] = []

        def walk(node) -> None:
            if node is None:
                return
            if node.model_name:
                names.append(node.model_name)
            for child in node.children:
                walk(child)
            walk(node.item)

        walk(task.fields)
        return names or [SCORE_FIELD]

    def windows(self, text: str, budget: int) -> list[tuple[int, int]]:
        """Overlapping ``[start, end)`` character windows of at most ``budget`` tokens."""
        import re

        words = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        if not words:
            return [(0, len(text))]
        budget = max(budget, 1)
        costs = [max(1, self.count_tokens(text[start:end])) for start, end in words]
        result: list[tuple[int, int]] = []
        first = 0
        while first < len(words):
            total = 0
            last = first
            while last < len(words) and total + costs[last] <= budget:
                total += costs[last]
                last += 1
            last = max(last, first + 1)
            result.append((words[first][0], words[last - 1][1]))
            if last >= len(words):
                break
            step = max(1, int((last - first) * (1 - self.window_overlap)))
            first += step
        return result

    def extraction_row(
        self,
        text: str,
        offset: int,
        classification: list[TaskSpec],
        extraction: list[TaskSpec],
        base_request: dict[str, Any],
    ) -> Row:
        row = Row(request={"text": text, **base_request}, offset=offset)
        group_prompts: dict[str, dict[str, Any]] = {}
        classes: dict[str, list[str]] = {}
        for index, task in enumerate(classification):
            key = f"c{index}"
            labels = task.labels or []
            classes[key] = [label.text for label in labels]
            group_prompts[key] = self.group_prompt(task)
            row.classification.append(Group(key, task, index, {label.text: label.id for label in labels}))
        entities: dict[str, list[str]] = {}
        relations: dict[str, Any] = {}
        span_groups: list[Group] = []
        relation_groups: list[Group] = []
        for index, task in enumerate(extraction):
            key = f"e{index}"
            group_prompts[key] = self.group_prompt(task)
            if task.type == "span":
                labels = task.labels or [Label(
                    id="",
                    text=OPEN_SPAN_LABEL if task.instruction else OPEN_SPAN_LABEL_NO_INSTRUCTION,
                )]
                entities[key] = [label.text for label in labels]
                span_groups.append(Group(key, task, len(span_groups), {label.text: label.id for label in labels}))
            else:
                entity_labels = task.entity_labels or [Label(id="", text=DEFAULT_ENTITY_LABEL)]
                relations[key] = {
                    "entities": [label.text for label in entity_labels],
                    "relations": [label.text for label in task.labels or []],
                }
                relation_groups.append(Group(
                    key,
                    task,
                    len(relation_groups),
                    {label.text: label.id for label in task.labels or []},
                    {label.text: label.id for label in entity_labels},
                ))
        # Joint relation groups follow the entity groups in the model's
        # extraction list; open relation groups are numbered on their own.
        if self.capabilities.relation_head != "open_relex":
            for group in relation_groups:
                group.index += len(span_groups)
        row.extraction = span_groups + relation_groups
        thresholds = [self.threshold_for(task) for task in extraction]
        request = row.request
        if classes:
            request["classes"] = classes
        if entities:
            request["entities"] = entities
        if relations:
            if self.capabilities.relation_head == "open_relex":
                request["relations"] = {key: value["relations"] for key, value in relations.items()}
            else:
                request["joint_relations"] = relations
        request.update(
            group_prompts=group_prompts,
            threshold=min(thresholds) if thresholds else float(self.capabilities.default_thresholds.get("span", 0.5)),
            flat_ner=all(task.flat for task in extraction) if extraction else True,
            multi_label=False,
            decoder_kwargs={"task_controls": {
                "classification": {"threshold": ALL_SCORES, "multi_label": True},
            }},
        )
        return row

    def json_row(self, task: TaskSpec, text: str, base_request: dict[str, Any]) -> Row:
        if task.scale is not None:
            scale = task.scale
            descriptor = {"$type": "number", "$min": scale["min"], "$max": scale["max"]}
            if "step" in scale:
                descriptor["$step"] = scale["step"]
            structures = {"$root": {SCORE_FIELD: descriptor}}
        else:
            structures = model_template(
                task.fields,  # type: ignore[arg-type]
                typed=self.capabilities.typed_heads,
                show_description="{description}" in self.capabilities.label_template,
            )
        request = {
            "text": text,
            **base_request,
            "structures": structures,
            "threshold": float(self.capabilities.default_thresholds.get("json", 0.5)),
            "flat_ner": True,
            "multi_label": False,
        }
        if task.instruction:
            request["prompt"] = task.instruction
        return Row(request=request, json_task=task)

    # -- execution ------------------------------------------------------------------

    def execute(self, plans: list[Plan]) -> None:
        rows = [row for plan in plans for row in plan.rows]
        if rows:
            try:
                outputs = self.run_rows([row.request for row in rows])
            except Exception as exc:  # noqa: BLE001 - isolate per request below
                if len(plans) == 1:
                    plans[0].error = self.execution_error(exc)
                    return
                for plan in plans:
                    self.execute([plan])
                return
            for row, output in zip(rows, outputs, strict=True):
                row.output = output
        for plan in plans:
            if not plan.embed_texts or plan.error is not None:
                continue
            try:
                plan.embeddings = self.embed(  # type: ignore[misc]
                    plan.embed_texts, [plan.adapter_id] * len(plan.embed_texts)
                )
            except Exception as exc:  # noqa: BLE001
                plan.error = self.execution_error(exc)

    @staticmethod
    def execution_error(exc: Exception) -> APIError:
        if isinstance(exc, APIError):
            return exc
        logger.exception("Analyze execution failed", exc_info=exc)
        return APIError("INTERNAL", f"Model execution failed: {type(exc).__name__}: {exc}")

    # -- assembly -----------------------------------------------------------------------

    def assemble(self, plan: Plan) -> dict[str, Any]:
        parsed = plan.parsed
        results: dict[str, Any] = {}
        for task in parsed.tasks:
            results[task.name] = self.task_result(plan, task)
            if parsed.options.include_label_definitions:
                labels = task.labels if task.type != "boolean" else [
                    label for label in task.labels or [] if label.title or label.description or label.metadata
                ]
                if labels:
                    results[task.name]["label_definitions"] = {
                        label.id: label.definition() for label in labels
                    }
        response: dict[str, Any] = {"model": parsed.model, "results": results}
        if parsed.options.include_usage:
            response["usage"] = {
                "rows": len(plan.rows) + (1 if plan.embed_texts else 0),
                "tokens": plan.token_count,
                "latency_ms": round((time.perf_counter() - plan.started) * 1000.0, 1),
            }
        return response

    def task_result(self, plan: Plan, task: TaskSpec) -> dict[str, Any]:
        handler = {
            "classify": self.classification_result,
            "boolean": self.classification_result,
            "score": self.score_result,
            "span": self.span_result,
            "relation": self.relation_result,
            "json": self.json_result,
            "embedding": self.embedding_result,
            "rank": self.rank_result,
        }[task.type]
        return handler(plan, task)

    def find_group(self, plan: Plan, task: TaskSpec) -> tuple[Row, int, Group]:
        for row in plan.rows:
            for group in row.classification:
                if group.task is task:
                    return row, group.index, group
        raise APIError("INTERNAL", f"No model group was planned for task '{task.name}'")

    def label_scores(self, plan: Plan, task: TaskSpec) -> dict[str, float]:
        row, index, group = self.find_group(plan, task)
        groups = (row.output or {}).get("classification") or []
        predictions = groups[index] if index < len(groups) else []
        scores = {label.id: 0.0 for label in task.labels or []}
        for prediction in predictions or []:
            label_id = group.text_to_id.get(prediction.get("class_name"))
            if label_id is not None:
                scores[label_id] = _round(prediction.get("score", 0.0))
        return scores

    def classification_result(self, plan: Plan, task: TaskSpec) -> dict[str, Any]:
        scores = self.label_scores(plan, task)
        result: dict[str, Any] = {"type": task.type}
        if task.single_label:
            best = max(scores, key=lambda label_id: scores[label_id]) if scores else None
            value: Any = best
            if task.threshold is not None and best is not None and scores[best] < task.threshold:
                value = None
            if task.type == "boolean":
                value = None if value is None else value == "true"
            result["value"] = value
            if task.type == "boolean":
                result["truth_score"] = scores.get("true", 0.0)
        else:
            threshold = self.threshold_for(task)
            result["value"] = [label_id for label_id, score in scores.items() if score >= threshold]
        result["scores"] = scores
        result["threshold"] = self.reported_threshold(task)
        return result

    def score_result(self, plan: Plan, task: TaskSpec) -> dict[str, Any]:
        if task.scale is None:
            result = self.classification_result(plan, task)
            return result
        row = next(row for row in plan.rows if row.json_task is task)
        raw = (row.output or {}).get("structuring") or {}
        value = raw.get(SCORE_FIELD) if isinstance(raw, dict) else None
        if isinstance(value, list):
            value = next((item for item in value if item is not None), None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            scale = task.scale
            value = min(max(float(value), scale["min"]), scale["max"])
            step = scale.get("step")
            if step:
                value = scale["min"] + round((value - scale["min"]) / step) * step
            if float(value).is_integer():
                value = int(value)
        else:
            value = None
        return {"type": "score", "value": value, "scale": dict(task.scale)}

    def located(self, rendered: RenderedInput, row: Row, start: Any, end: Any) -> Any:
        if not isinstance(start, int) or not isinstance(end, int):
            return None
        return rendered.locate(start + row.offset, end + row.offset)

    def span_result(self, plan: Plan, task: TaskSpec) -> dict[str, Any]:
        rendered = plan.parsed.rendered
        threshold = self.threshold_for(task)
        merged: dict[tuple[int, int, str], dict[str, Any]] = {}
        for row in plan.rows:
            for group in row.extraction:
                if group.task is not task:
                    continue
                for entity in (row.output or {}).get("ner") or []:
                    if entity.get("group_index") != group.index:
                        continue
                    score = float(entity.get("score", 0.0))
                    if score < threshold:
                        continue
                    location = self.located(rendered, row, entity.get("start"), entity.get("end"))
                    if location is None:
                        continue
                    label_id = group.text_to_id.get(entity.get("label"))
                    if label_id is None:
                        continue
                    start = entity["start"] + row.offset
                    end = entity["end"] + row.offset
                    key = (start, end, label_id)
                    if key in merged and merged[key]["score"] >= score:
                        continue
                    span: dict[str, Any] = {}
                    if task.labels is not None:
                        span["label"] = label_id
                    span.update(location.to_dict())
                    span["score"] = _round(score)
                    span["_start"], span["_end"] = start, end
                    merged[key] = span
        spans = list(merged.values())
        windowed = sum(1 for row in plan.rows for group in row.extraction if group.task is task) > 1
        if task.flat and windowed:
            spans = _greedy_flat(spans)
        spans.sort(key=lambda span: (span["_start"], span["_end"]))
        for span in spans:
            del span["_start"], span["_end"]
        return {"type": "span", "value": spans, "threshold": threshold}

    def relation_result(self, plan: Plan, task: TaskSpec) -> dict[str, Any]:
        rendered = plan.parsed.rendered
        threshold = self.threshold_for(task)
        output_key = "open_relex" if self.capabilities.relation_head == "open_relex" else "joint_relex"
        merged: dict[tuple, dict[str, Any]] = {}
        for row in plan.rows:
            for group in row.extraction:
                if group.task is not task:
                    continue
                for triple in (row.output or {}).get(output_key) or []:
                    if triple.get("group_index") != group.index:
                        continue
                    score = float(triple.get("score", 0.0))
                    if score < threshold:
                        continue
                    relation_id = group.text_to_id.get(triple.get("relation"))
                    if relation_id is None:
                        continue
                    ends = []
                    for role in ("head", "tail"):
                        entity = triple.get(role) or {}
                        location = self.located(rendered, row, entity.get("start"), entity.get("end"))
                        if location is None:
                            break
                        item: dict[str, Any] = {}
                        if task.entity_labels is not None:
                            label_id = (group.entity_text_to_id or {}).get(entity.get("type"))
                            if label_id is not None:
                                item["label"] = label_id
                        item.update(location.to_dict())
                        ends.append((item, entity["start"] + row.offset, entity["end"] + row.offset))
                    if len(ends) != 2:
                        continue
                    key = (ends[0][1], ends[0][2], relation_id, ends[1][1], ends[1][2])
                    if key in merged and merged[key]["score"] >= score:
                        continue
                    merged[key] = {
                        "head": ends[0][0],
                        "relation": relation_id,
                        "tail": ends[1][0],
                        "score": _round(score),
                        "_order": (ends[0][1], ends[1][1]),
                    }
        relations = sorted(merged.values(), key=lambda item: item["_order"])
        for item in relations:
            del item["_order"]
        return {"type": "relation", "value": relations, "threshold": threshold}

    def json_result(self, plan: Plan, task: TaskSpec) -> dict[str, Any]:
        row = next(row for row in plan.rows if row.json_task is task)
        raw = (row.output or {}).get("structuring")
        shaper = JsonShaper(
            task.fields,  # type: ignore[arg-type]
            plan.parsed.rendered,
            want_evidence=plan.parsed.options.include_evidence,
        )
        shaped = shaper.shape(raw)
        result: dict[str, Any] = {"type": "json", "value": shaped.value}
        if plan.parsed.options.include_evidence:
            result["evidence"] = shaped.evidence
        if shaped.unparsed:
            result["unparsed"] = shaped.unparsed
        return result

    def vector(self, plan: Plan, text: str) -> list[float]:
        if plan.embeddings is None:
            raise APIError("INTERNAL", "Embeddings were not computed")
        return list(plan.embeddings[plan.embed_texts.index(text)])

    @staticmethod
    def normalized(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm > 0 else vector

    def embedding_result(self, plan: Plan, task: TaskSpec) -> dict[str, Any]:
        vector = self.vector(plan, plan.parsed.rendered.text)
        if task.normalize:
            vector = self.normalized(vector)
        return {
            "type": "embedding",
            "value": vector,
            "dimensions": len(vector),
            "normalized": task.normalize,
        }

    def rank_result(self, plan: Plan, task: TaskSpec) -> dict[str, Any]:
        query = self.normalized(self.vector(plan, plan.parsed.rendered.text))
        scored = []
        for candidate in task.candidates or []:
            vector = self.normalized(self.vector(plan, candidate.rendered.text))
            score = sum(a * b for a, b in zip(query, vector, strict=True))
            scored.append((candidate.id, score))
        scored.sort(key=lambda item: -item[1])
        if task.top_k is not None:
            scored = scored[:task.top_k]
        return {
            "type": "rank",
            "value": [
                {"id": candidate_id, "rank": index + 1, "score": _round(score)}
                for index, (candidate_id, score) in enumerate(scored)
            ],
        }


__all__ = ["AnalyzeEngine", "Options", "Plan", "Row"]
