"""Optional anchor alignment and nested formatting for structuring."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Any

import torch

from .structuring_types import (
    StructuringAnchorEntry,
    StructuringDiagnostics,
    parse_component_spec,
)

MULTI_LEVEL_RESULT_KEY = "__glinext_multi_level_result__"


def make_multi_level_group_result(
    nodes: list[dict],
    relation_scores: torch.Tensor | None,
    mapping,
    output_mode: str,
    *,
    presence_is_reliable: bool = False,
    preserve_empty_records: bool = False,
) -> dict:
    """Package decoded fields without compacting their original anchor axis."""

    return {
        MULTI_LEVEL_RESULT_KEY: True,
        "nodes": nodes,
        "relation_scores": (
            relation_scores.detach().float().cpu()
            if relation_scores is not None
            else None
        ),
        "mapping": mapping,
        "output_mode": output_mode,
        "presence_is_reliable": bool(presence_is_reliable),
        "preserve_empty_records": bool(preserve_empty_records),
    }


def is_multi_level_group_result(value: object) -> bool:
    return isinstance(value, dict) and bool(value.get(MULTI_LEVEL_RESULT_KEY))


def _set_path(target: dict, path: tuple[str, ...], value: object) -> None:
    if not path:
        return
    cursor = target
    for segment in path[:-1]:
        child = cursor.get(segment)
        if not isinstance(child, dict):
            child = {}
            cursor[segment] = child
        cursor = child
    cursor[path[-1]] = value


def _get_path(target: dict, path: tuple[str, ...]) -> object:
    cursor = target
    for segment in path:
        if not isinstance(cursor, dict) or segment not in cursor:
            return None
        cursor = cursor[segment]
    return cursor


def _field_value(field: dict, start_map, end_map, text: str) -> str:
    start = int(field.get("start", 0))
    end = int(field.get("end", start))
    if 0 <= start < len(start_map) and 0 <= end < len(end_map):
        return text[start_map[start]:end_map[end]]
    return str(field.get("text", ""))


def _shape_array(values: list[object], rank: int = 1) -> list:
    """Preserve declared array rank when exact inner grouping is unavailable."""

    shaped: list = list(values)
    for _ in range(max(1, int(rank)) - 1):
        shaped = [shaped]
    return shaped


def _has_json_content(value: object) -> bool:
    """Return whether a JSON value contains evidence beyond empty placeholders."""

    if value is None:
        return False
    if isinstance(value, dict):
        return any(_has_json_content(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_json_content(item) for item in value)
    return True


def _filter_empty_records(value: object) -> object:
    """Recursively remove null/empty-only records from nested JSON arrays."""

    if isinstance(value, dict):
        return {
            key: _filter_empty_records(item)
            for key, item in value.items()
        }
    if not isinstance(value, list):
        return value

    filtered = []
    for item in value:
        item = _filter_empty_records(item)
        if isinstance(item, dict) and not _has_json_content(item):
            continue
        filtered.append(item)
    return filtered


def _paths_conflict(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    """Return whether one JSON leaf path is a prefix of the other."""

    shared = min(len(left), len(right))
    return left != right and left[:shared] == right[:shared]


@dataclass(frozen=True)
class _MaterializedNodes:
    assigned: dict[int, tuple[str, ...]]
    objects: dict[int, dict]
    positions: dict[int, tuple[int, int]]
    value_entries: dict[int, dict]
    by_anchor: dict[int, dict]


class _AnchorAlignment:
    """Align related anchors and format qualified fields as nested JSON."""

    def __init__(self, relation_threshold: float = 0.5):
        self.relation_threshold = float(relation_threshold)

    @staticmethod
    def _hierarchy(mapping):
        hierarchy = list(getattr(mapping, "hierarchy", None) or [])
        by_path = {
            tuple(str(segment) for segment in node.get("path") or ()): node
            for node in hierarchy
        }
        field_meta = {}
        for path, node in by_path.items():
            for field in node.get("fields") or []:
                if isinstance(field, dict) and field.get("label"):
                    field_meta[str(field["label"])] = (
                        path,
                        tuple(str(segment) for segment in field.get("local_path") or ()),
                        dict(field.get("shape") or {}),
                    )
        return hierarchy, by_path, field_meta

    @staticmethod
    def _score(scores, parent_anchor: int, child_anchor: int) -> float:
        if scores is None:
            return 0.0
        if not (
            0 <= parent_anchor < scores.shape[0]
            and 0 <= child_anchor < scores.shape[1]
        ):
            return 0.0
        return float(scores[parent_anchor, child_anchor].item())

    def _expand_mixed_path_nodes(self, nodes, scores, by_path, field_meta):
        """Split a physical anchor that contains several hierarchy levels.

        Set-based heads occasionally place a complete subtree on one slot.  A
        physical slot is still useful evidence in that case, but treating it
        as exactly one JSON object discards every field outside the winning
        hierarchy path.  Turn each represented path into a logical node and
        add deterministic parent edges between adjacent paths from the same
        physical slot.
        """

        paths_by_source = {}
        for node_index, node in enumerate(nodes):
            paths_by_source[node_index] = {
                metadata[0]
                for field in node.get("fields") or []
                if (
                    metadata := field_meta.get(str(field.get("field")))
                ) is not None
            }
        mixed_sources = {
            node_index
            for node_index, paths in paths_by_source.items()
            if len(paths) > 1
        }
        if not mixed_sources:
            return nodes, scores, False

        # The same decoded span can leak onto more than one physical slot.
        # When a mixed slot is involved, retain its strongest assignment so
        # splitting does not manufacture duplicate roots from weak copies.
        duplicate_occurrences = defaultdict(list)
        for node_index, node in enumerate(nodes):
            for field_index, field in enumerate(node.get("fields") or []):
                if field_meta.get(str(field.get("field"))) is None:
                    continue
                key = (
                    str(field.get("field")),
                    int(field.get("start", 0)),
                    int(field.get("end", field.get("start", 0))),
                    str(field.get("text", "")),
                )
                duplicate_occurrences[key].append((
                    float(field.get("score", 0.0)),
                    node_index,
                    field_index,
                ))

        retained_fields = set()
        for occurrences in duplicate_occurrences.values():
            if len(occurrences) == 1 or not any(
                node_index in mixed_sources
                for _, node_index, _ in occurrences
            ):
                retained_fields.update(
                    (node_index, field_index)
                    for _, node_index, field_index in occurrences
                )
                continue
            _, node_index, field_index = max(
                occurrences,
                key=lambda occurrence: (
                    occurrence[0], -occurrence[1], -occurrence[2],
                ),
            )
            retained_fields.add((node_index, field_index))

        logical_nodes = []
        for node_index, node in enumerate(nodes):
            fields_by_path = defaultdict(list)
            unknown_fields = []
            original_fields = list(node.get("fields") or [])
            for field_index, field in enumerate(original_fields):
                metadata = field_meta.get(str(field.get("field")))
                if metadata is None:
                    unknown_fields.append(field)
                elif (node_index, field_index) in retained_fields:
                    fields_by_path[metadata[0]].append(field)

            if fields_by_path:
                ordered_paths = sorted(
                    fields_by_path, key=lambda path: (len(path), path),
                )
                for path_index, path in enumerate(ordered_paths):
                    path_fields = fields_by_path[path]
                    compacted_fields = []
                    fields_by_label = defaultdict(list)
                    for field in path_fields:
                        fields_by_label[str(field.get("field"))].append(field)
                    for label, occurrences in fields_by_label.items():
                        metadata = field_meta.get(label)
                        shape = metadata[2] if metadata is not None else {}
                        if shape.get("kind") == "array":
                            compacted_fields.extend(occurrences)
                        else:
                            compacted_fields.append(max(
                                occurrences,
                                key=lambda field: (
                                    float(field.get("score", 0.0)),
                                    -int(field.get("start", 10**12)),
                                ),
                            ))
                    logical_node = dict(node)
                    logical_node["source_anchor_index"] = int(
                        node["anchor_index"]
                    )
                    logical_node["path_hint"] = path
                    logical_node["fields"] = compacted_fields
                    if path_index == 0:
                        logical_node["fields"].extend(unknown_fields)
                    logical_nodes.append(logical_node)
            elif not original_fields or unknown_fields:
                logical_node = dict(node)
                logical_node["source_anchor_index"] = int(
                    node["anchor_index"]
                )
                logical_node["fields"] = unknown_fields
                logical_nodes.append(logical_node)

        for anchor_index, node in enumerate(logical_nodes):
            node["anchor_index"] = anchor_index

        size = len(logical_nodes)
        if scores is not None:
            logical_scores = torch.zeros(
                (size, size), dtype=scores.dtype, device=scores.device,
            )
        else:
            logical_scores = torch.zeros((size, size), dtype=torch.float32)

        for parent_index, parent in enumerate(logical_nodes):
            parent_source = int(parent["source_anchor_index"])
            parent_path = parent.get("path_hint")
            for child_index, child in enumerate(logical_nodes):
                if parent_index == child_index:
                    continue
                child_source = int(child["source_anchor_index"])
                if scores is not None and (
                    0 <= parent_source < scores.shape[0]
                    and 0 <= child_source < scores.shape[1]
                ):
                    logical_scores[parent_index, child_index] = scores[
                        parent_source, child_source
                    ]

                child_path = child.get("path_hint")
                child_meta = by_path.get(child_path, {})
                parent_path_raw = child_meta.get("parent_path")
                if (
                    parent_source == child_source
                    and parent_path is not None
                    and child_path is not None
                    and parent_path_raw is not None
                    and tuple(parent_path_raw) == parent_path
                    and self._earliest_position(parent)[0]
                    <= self._earliest_position(child)[0]
                ):
                    logical_scores[parent_index, child_index] = 1.0

        return logical_nodes, logical_scores, True

    def _assign_node_paths(self, nodes, scores, by_path, field_meta):
        assigned: dict[int, tuple[str, ...] | None] = {}
        node_by_anchor = {int(node["anchor_index"]): node for node in nodes}

        # Scalar fields are the strongest object-type evidence. Sum confidence
        # so an isolated noisy field does not override several consistent ones.
        for anchor, node in node_by_anchor.items():
            path_hint = node.get("path_hint")
            if path_hint is not None and tuple(path_hint) in by_path:
                assigned[anchor] = tuple(path_hint)
                continue
            evidence = defaultdict(float)
            for field in node.get("fields") or []:
                metadata = field_meta.get(str(field.get("field")))
                if metadata is not None:
                    evidence[metadata[0]] += float(field.get("score", 0.0))
            assigned[anchor] = (
                max(evidence, key=lambda path: (evidence[path], -len(path), path))
                if evidence
                else None
            )

        anchors = list(node_by_anchor)
        # Container-only nodes can be typed by a confident edge to/from a typed
        # neighbour. Hierarchy depth makes accepted edges acyclic by design.
        for _ in range(max(1, len(by_path))):
            changed = False
            for anchor in anchors:
                if assigned[anchor] is not None:
                    continue
                candidates = defaultdict(float)
                for other in anchors:
                    other_path = assigned[other]
                    if other_path is None:
                        continue
                    other_meta = by_path.get(other_path, {})
                    parent_path = other_meta.get("parent_path")
                    outgoing = self._score(scores, anchor, other)
                    if parent_path is not None and outgoing >= self.relation_threshold:
                        candidates[tuple(parent_path)] = max(
                            candidates[tuple(parent_path)], outgoing,
                        )

                    incoming = self._score(scores, other, anchor)
                    if incoming < self.relation_threshold:
                        continue
                    compatible_children = [
                        path
                        for path, metadata in by_path.items()
                        if tuple(metadata.get("parent_path") or ()) == other_path
                        and metadata.get("parent_path") is not None
                    ]
                    if len(compatible_children) == 1:
                        candidates[compatible_children[0]] = max(
                            candidates[compatible_children[0]], incoming,
                        )
                if candidates:
                    assigned[anchor] = max(
                        candidates,
                        key=lambda path: (candidates[path], -len(path), path),
                    )
                    changed = True
            if not changed:
                break

        # A node without an incoming predicted edge is a root record. Remaining
        # ambiguous connected nodes also fall back to root rather than inventing
        # a child key when several schema branches are possible.
        for anchor in anchors:
            if assigned[anchor] is None:
                assigned[anchor] = ()
        return assigned

    @staticmethod
    def _earliest_position(node: dict) -> tuple[int, int]:
        starts = [
            int(field.get("start", 10**12))
            for field in node.get("fields") or []
        ]
        return (min(starts, default=10**12), int(node["anchor_index"]))

    @staticmethod
    def _order_object_by_schema(
        value: dict,
        node_path: tuple[str, ...],
        hierarchy: list[dict],
        by_path: dict[tuple[str, ...], dict],
    ) -> dict:
        """Return a dictionary ordered exactly like the hierarchy prompt.

        The prompt renders scalar fields for an object first, followed by its
        child-object markers.  Reconstruction is evidence-driven and therefore
        builds keys in confidence/attachment order; normalize insertion order
        only after the graph is complete so JSON serialization is stable.
        """

        def new_tree_node():
            return {"segments": {}, "child_path": None}

        tree_cache = {}

        def add_path(tree, path, child_path=None):
            cursor = tree
            for segment in path:
                cursor = cursor["segments"].setdefault(
                    str(segment), new_tree_node(),
                )
            if child_path is not None:
                cursor["child_path"] = child_path

        def schema_tree(path):
            cached = tree_cache.get(path)
            if cached is not None:
                return cached
            tree = new_tree_node()
            tree_cache[path] = tree
            metadata = by_path.get(path, {})

            # This is also the order used by ``contribute_prompt``.
            for field in metadata.get("fields") or []:
                if not isinstance(field, dict):
                    continue
                add_path(
                    tree,
                    tuple(str(x) for x in field.get("local_path") or ()),
                )
            for child in hierarchy:
                if not isinstance(child, dict):
                    continue
                parent_path = child.get("parent_path")
                if (
                    parent_path is None
                    or tuple(str(x) for x in parent_path) != path
                ):
                    continue
                add_path(
                    tree,
                    tuple(
                        str(x)
                        for x in child.get("parent_field_path") or ()
                    ),
                    tuple(str(x) for x in child.get("path") or ()),
                )
            # Empty inline containers have no field token in the prompt. Keep
            # them deterministic after all prompt-visible paths.
            for container in metadata.get("containers") or []:
                if isinstance(container, dict):
                    add_path(
                        tree,
                        tuple(
                            str(x)
                            for x in container.get("local_path") or ()
                        ),
                    )
            return tree

        def order_hierarchy_value(child_value, child_path):
            if isinstance(child_value, dict):
                return order_mapping(child_value, schema_tree(child_path))
            if isinstance(child_value, list):
                return [
                    order_hierarchy_value(item, child_path)
                    for item in child_value
                ]
            return child_value

        def order_inline_value(child_value, tree):
            child_path = tree["child_path"]
            if child_path is not None:
                return order_hierarchy_value(child_value, child_path)
            if isinstance(child_value, dict):
                return order_mapping(child_value, tree)
            if isinstance(child_value, list) and tree["segments"]:
                return [
                    order_inline_value(item, tree)
                    for item in child_value
                ]
            return child_value

        def order_mapping(mapping, tree):
            ordered = {}
            for key, child_tree in tree["segments"].items():
                if key in mapping:
                    ordered[key] = order_inline_value(
                        mapping[key], child_tree,
                    )
            for key, child_value in mapping.items():
                if key not in ordered:
                    ordered[key] = child_value
            return ordered

        return order_mapping(value, schema_tree(node_path))

    def align_and_format(
        self,
        payload: dict,
        start_map: list[int],
        end_map: list[int],
        text: str,
        *,
        relation_threshold: float | None = None,
        diagnostics: dict | None = None,
    ) -> list[dict]:
        """Align and format one schema group's root records."""

        alignment = self
        if (
            relation_threshold is not None
            and float(relation_threshold) != self.relation_threshold
        ):
            # A call-time threshold is configuration, not mutable shared
            # state. This keeps shared alignment helpers re-entrant and avoids
            # cross-request leakage without changing the reconstruction path.
            alignment = copy(self)
            alignment.relation_threshold = float(relation_threshold)
        return alignment._align_and_format(
            payload,
            start_map,
            end_map,
            text,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _diagnostic_field(field: dict) -> dict:
        return {
            "field": str(field.get("field", "")),
            "text": str(field.get("text", "")),
            "score": round(float(field.get("score", 0.0)), 6),
        }

    def _initialize_diagnostics(
        self,
        raw_nodes: list[dict],
        physical_scores: torch.Tensor | None,
        diagnostics: dict | None,
    ) -> None:
        if diagnostics is None:
            return
        active_ids = sorted(
            {int(node["anchor_index"]) for node in raw_nodes}
        )
        diagnostics.clear()
        diagnostics.update({
            "relation_threshold": self.relation_threshold,
            "physical_slot_capacity": (
                int(physical_scores.shape[0])
                if physical_scores is not None
                else (max(active_ids, default=-1) + 1)
            ),
            "active_anchor_count": len(active_ids),
            "active_anchor_ids": active_ids,
            "active_anchors": [
                {
                    "anchor_id": int(node["anchor_index"]),
                    "presence_is_reliable": bool(
                        node.get("presence_is_reliable", False)
                    ),
                    "field_count": len(node.get("fields") or []),
                    "fields": [
                        self._diagnostic_field(field)
                        for field in node.get("fields") or []
                    ],
                }
                for node in raw_nodes
            ],
            "raw_relation_connections": [],
            "top_raw_relation_candidates": [],
            "logical_nodes": [],
            "connections": [],
            "merges": [],
        })
        raw_candidates = []
        if physical_scores is not None:
            for parent in active_ids:
                for child in active_ids:
                    if parent == child:
                        continue
                    score = self._score(physical_scores, parent, child)
                    edge = {
                        "parent_anchor_id": parent,
                        "child_anchor_id": child,
                        "score": round(score, 6),
                    }
                    raw_candidates.append(edge)
                    if score >= self.relation_threshold:
                        diagnostics["raw_relation_connections"].append(edge)
        raw_candidates.sort(
            key=lambda edge: (
                -edge["score"],
                edge["parent_anchor_id"],
                edge["child_anchor_id"],
            )
        )
        diagnostics["top_raw_relation_candidates"] = raw_candidates[:5]

    def _filter_active_nodes(
        self,
        payload: dict,
        raw_nodes: list[dict],
        scores: torch.Tensor | None,
        by_path: dict[tuple[str, ...], dict],
    ) -> tuple[list[dict], bool]:
        """Drop inert fixed slots while retaining connected containers."""

        raw_anchors = [int(node["anchor_index"]) for node in raw_nodes]
        connected: set[int] = set()
        if scores is not None and any(path for path in by_path):
            for parent in raw_anchors:
                for child in raw_anchors:
                    if (
                        parent != child
                        and self._score(scores, parent, child)
                        >= self.relation_threshold
                    ):
                        connected.update((parent, child))
        preserve_empty_records = bool(
            payload.get("preserve_empty_records", False)
        )
        nodes = [
            node
            for node in raw_nodes
            if (
                node.get("fields")
                or int(node["anchor_index"]) in connected
                or (
                    preserve_empty_records
                    and (
                        node.get("presence_is_reliable", False)
                        or payload.get("presence_is_reliable", False)
                    )
                )
            )
        ]
        return nodes, preserve_empty_records

    def _materialize_nodes(
        self,
        nodes: list[dict],
        scores: torch.Tensor | None,
        by_path: dict[tuple[str, ...], dict],
        field_meta: dict,
        start_map: list[int],
        end_map: list[int],
        text: str,
    ) -> _MaterializedNodes:
        """Assign schema paths and convert field evidence into JSON objects."""

        assigned = self._assign_node_paths(nodes, scores, by_path, field_meta)
        node_objects: dict[int, dict] = {}
        node_positions: dict[int, tuple[int, int]] = {}
        node_value_entries: dict[int, dict] = {}
        node_by_anchor = {
            int(node["anchor_index"]): node for node in nodes
        }

        for anchor, node in node_by_anchor.items():
            node_path = assigned[anchor]
            obj: dict = {}
            node_schema = by_path[node_path]
            containers = sorted(
                (
                    container
                    for container in node_schema.get("containers") or []
                    if isinstance(container, dict)
                ),
                key=lambda container: len(
                    container.get("local_path") or ()
                ),
            )
            for container in containers:
                local_path = tuple(
                    str(segment)
                    for segment in container.get("local_path") or ()
                )
                kind = container.get("kind")
                if kind == "object":
                    _set_path(obj, local_path, {})
                elif kind == "array":
                    _set_path(
                        obj,
                        local_path,
                        _shape_array(
                            [],
                            int(container.get("rank", 1)),
                        ),
                    )

            occurrences = defaultdict(list)
            for field in node.get("fields") or []:
                metadata = field_meta.get(str(field.get("field")))
                if metadata is None or metadata[0] != node_path:
                    continue
                value = _field_value(field, start_map, end_map, text)
                occurrences[metadata[1]].append((
                    int(field.get("start", 10**12)),
                    -float(field.get("score", 0.0)),
                    value,
                ))

            field_specs = {
                tuple(
                    str(segment)
                    for segment in field.get("local_path") or ()
                ): field
                for field in node_schema.get("fields") or []
                if isinstance(field, dict)
            }
            evidence_scores = {
                path: sum(-value[1] for value in values)
                for path, values in occurrences.items()
                if values
            }
            selected_paths: list[tuple[str, ...]] = []
            for path in sorted(
                evidence_scores,
                key=lambda candidate: (
                    -evidence_scores[candidate],
                    len(candidate),
                    candidate,
                ),
            ):
                if not any(
                    _paths_conflict(path, selected)
                    for selected in selected_paths
                ):
                    selected_paths.append(path)

            for local_path in selected_paths:
                field = field_specs[local_path]
                values = sorted(occurrences[local_path])
                shape = field.get("shape") or {}
                if shape.get("kind") == "array":
                    _set_path(
                        obj,
                        local_path,
                        _shape_array(
                            [value[2] for value in values],
                            int(shape.get("rank", 1)),
                        ),
                    )
                else:
                    best = min(values, key=lambda value: (value[1], value[0]))
                    _set_path(obj, local_path, best[2])

            for local_path, field in sorted(
                field_specs.items(),
                key=lambda item: (len(item[0]), item[0]),
            ):
                if local_path in selected_paths or any(
                    _paths_conflict(local_path, selected)
                    for selected in selected_paths
                ):
                    continue
                shape = field.get("shape") or {}
                _set_path(
                    obj,
                    local_path,
                    (
                        _shape_array([], int(shape.get("rank", 1)))
                        if shape.get("kind") == "array"
                        else None
                    ),
                )

            node_objects[anchor] = obj
            node_value_entries[anchor] = {
                path: [
                    (value[0], value[2]) for value in sorted(values)
                ]
                for path, values in occurrences.items()
                if path in selected_paths
            }
            node_positions[anchor] = self._earliest_position(node)

        return _MaterializedNodes(
            assigned=assigned,
            objects=node_objects,
            positions=node_positions,
            value_entries=node_value_entries,
            by_anchor=node_by_anchor,
        )

    def _select_parents(
        self,
        materialized: _MaterializedNodes,
        by_path: dict[tuple[str, ...], dict],
        scores: torch.Tensor | None,
        *,
        allow_text_fallback: bool,
    ) -> tuple[dict[int, int], dict[int, str]]:
        """Select one schema-compatible parent for every child node."""

        selected_parent: dict[int, int] = {}
        selection_reason: dict[int, str] = {}
        for child, child_path in materialized.assigned.items():
            expected_parent_raw = by_path[child_path].get("parent_path")
            if expected_parent_raw is None:
                continue
            expected_parent = tuple(expected_parent_raw)
            candidates = []
            for parent, parent_path in materialized.assigned.items():
                if parent == child or parent_path != expected_parent:
                    continue
                score = self._score(scores, parent, child)
                if score >= self.relation_threshold:
                    candidates.append((score, -parent, parent))
            if candidates:
                parent = max(candidates)[2]
                selected_parent[child] = parent
                parent_source = int(
                    materialized.by_anchor[parent].get(
                        "source_anchor_index", parent
                    )
                )
                child_source = int(
                    materialized.by_anchor[child].get(
                        "source_anchor_index", child
                    )
                )
                selection_reason[child] = (
                    "same_physical_anchor"
                    if parent_source == child_source
                    else "model_relation"
                )
                continue

            if allow_text_fallback:
                child_position = materialized.positions[child][0]
                preceding = []
                for parent, parent_path in materialized.assigned.items():
                    parent_position = materialized.positions[parent][0]
                    if (
                        parent != child
                        and parent_path == expected_parent
                        and parent_position < child_position
                        and parent_position < 10**12
                    ):
                        preceding.append((
                            parent_position,
                            self._score(scores, parent, child),
                            -parent,
                            parent,
                        ))
                if preceding:
                    selected_parent[child] = max(preceding)[3]
                    selection_reason[child] = "text_order_fallback"
        return selected_parent, selection_reason

    @staticmethod
    def _merge_sibling_fragments(
        materialized: _MaterializedNodes,
        selected_parent: dict[int, int],
        selection_reason: dict[int, str],
        *,
        enabled: bool,
    ) -> tuple[set[int], dict[int, int]]:
        """Merge close complementary fields representing the same child."""

        merged_anchors: set[int] = set()
        merged_into: dict[int, int] = {}
        if not enabled:
            return merged_anchors, merged_into

        siblings = defaultdict(list)
        for child, parent in selected_parent.items():
            child_path = materialized.assigned[child]
            if child_path:
                siblings[(parent, child_path)].append(child)

        for sibling_nodes in siblings.values():
            sibling_nodes.sort(
                key=lambda anchor: materialized.positions[anchor]
            )
            representative = None
            representative_paths: set[tuple[str, ...]] = set()
            representative_sources: set[int] = set()
            previous_position = None
            for child in sibling_nodes:
                evidence_paths = set(
                    materialized.value_entries.get(child, {})
                )
                source_anchor = int(
                    materialized.by_anchor[child].get(
                        "source_anchor_index", child
                    )
                )
                child_position = materialized.positions[child][0]
                conflicts = any(
                    left == right or _paths_conflict(left, right)
                    for left in representative_paths
                    for right in evidence_paths
                )
                can_merge = (
                    representative is not None
                    and bool(representative_paths)
                    and bool(evidence_paths)
                    and not conflicts
                    and source_anchor not in representative_sources
                    and previous_position is not None
                    and child_position - previous_position <= 12
                )
                if not can_merge:
                    representative = child
                    representative_paths = evidence_paths
                    representative_sources = {source_anchor}
                    previous_position = child_position
                    continue

                for local_path, values in materialized.value_entries[
                    child
                ].items():
                    _set_path(
                        materialized.objects[representative],
                        local_path,
                        deepcopy(
                            _get_path(materialized.objects[child], local_path)
                        ),
                    )
                    materialized.value_entries[representative][
                        local_path
                    ] = list(values)
                representative_paths.update(evidence_paths)
                representative_sources.add(source_anchor)
                previous_position = child_position
                merged_into[child] = representative
                merged_anchors.add(child)

        for child, parent in list(selected_parent.items()):
            original_parent = parent
            while parent in merged_into:
                parent = merged_into[parent]
            selected_parent[child] = parent
            if parent != original_parent:
                selection_reason[child] = (
                    f"{selection_reason.get(child, 'model_relation')}"
                    "_after_fragment_merge"
                )
        for merged_anchor in merged_anchors:
            selected_parent.pop(merged_anchor, None)
            selection_reason.pop(merged_anchor, None)
        return merged_anchors, merged_into

    @staticmethod
    def _attach_children(
        materialized: _MaterializedNodes,
        selected_parent: dict[int, int],
        by_path: dict[tuple[str, ...], dict],
    ) -> None:
        """Install selected child objects into their parent's array paths."""

        children_by_parent = defaultdict(list)
        for child, parent in selected_parent.items():
            children_by_parent[parent].append(child)
        for children in children_by_parent.values():
            children.sort(key=lambda child: materialized.positions[child])

        attachments = defaultdict(list)
        for parent, children in children_by_parent.items():
            for child in children:
                child_meta = by_path[materialized.assigned[child]]
                attach_path = tuple(
                    str(segment)
                    for segment in child_meta.get("parent_field_path") or ()
                )
                attachments[(parent, attach_path)].append((
                    materialized.positions[child][0],
                    materialized.objects[child],
                ))
        for (parent, attach_path), child_entries in attachments.items():
            values = list(
                materialized.value_entries.get(parent, {}).get(
                    attach_path, []
                )
            )
            values.extend(child_entries)
            values.sort(key=lambda entry: entry[0])
            container_rank = 1
            for container in by_path[
                materialized.assigned[parent]
            ].get("containers") or []:
                if (
                    isinstance(container, dict)
                    and tuple(
                        str(segment)
                        for segment in container.get("local_path") or ()
                    )
                    == attach_path
                    and container.get("kind") == "array"
                ):
                    container_rank = int(container.get("rank", 1))
                    break
            _set_path(
                materialized.objects[parent],
                attach_path,
                _shape_array(
                    [value for _, value in values],
                    container_rank,
                ),
            )

    def _collect_roots(
        self,
        payload: dict,
        nodes: list[dict],
        materialized: _MaterializedNodes,
        selected_parent: dict[int, int],
        merged_anchors: set[int],
        by_path: dict[tuple[str, ...], dict],
        hierarchy: list[dict],
        *,
        preserve_empty_records: bool,
    ) -> list[dict]:
        """Collect real roots and wrap disconnected descendants safely."""

        structural_anchors = {
            int(node["anchor_index"])
            for node in nodes
            if node.get("fields")
            and int(node["anchor_index"]) not in merged_anchors
        }
        if (
            preserve_empty_records
            and payload.get("presence_is_reliable", False)
        ):
            structural_anchors.update(
                int(node["anchor_index"]) for node in nodes
            )
        if preserve_empty_records:
            structural_anchors.update(
                int(node["anchor_index"])
                for node in nodes
                if node.get("presence_is_reliable", False)
            )
        structural_anchors.update(selected_parent)
        structural_anchors.update(selected_parent.values())

        root_anchors = [
            anchor
            for anchor, path in materialized.assigned.items()
            if (
                anchor in structural_anchors
                and path == ()
                and anchor not in selected_parent
            )
        ]
        root_anchors.sort(
            key=lambda anchor: materialized.positions[anchor]
        )
        roots = [
            materialized.objects[anchor] for anchor in root_anchors
        ]

        for anchor, path in materialized.assigned.items():
            if (
                anchor not in structural_anchors
                or anchor in merged_anchors
                or path == ()
                or anchor in selected_parent
            ):
                continue
            wrapped = deepcopy(materialized.objects[anchor])
            current_path = path
            while current_path:
                metadata = by_path[current_path]
                parent_obj: dict = {}
                attach_path = tuple(
                    str(segment)
                    for segment in metadata.get("parent_field_path") or ()
                )
                _set_path(parent_obj, attach_path, [wrapped])
                wrapped = parent_obj
                current_path = tuple(metadata.get("parent_path") or ())
            roots.append(wrapped)
        return [
            self._order_object_by_schema(root, (), hierarchy, by_path)
            for root in roots
        ]

    def _finalize_diagnostics(
        self,
        diagnostics: dict | None,
        materialized: _MaterializedNodes,
        selected_parent: dict[int, int],
        selection_reason: dict[int, str],
        merged_anchors: set[int],
        merged_into: dict[int, int],
        scores: torch.Tensor | None,
        physical_scores: torch.Tensor | None,
    ) -> None:
        if diagnostics is None:
            return
        diagnostics["logical_nodes"] = [
            {
                "logical_anchor_id": anchor,
                "source_anchor_id": int(
                    node.get("source_anchor_index", anchor)
                ),
                "schema_path": list(materialized.assigned[anchor]),
                "state": (
                    "merged" if anchor in merged_anchors else "active"
                ),
                "merged_into_logical_anchor_id": merged_into.get(anchor),
                "fields": [
                    self._diagnostic_field(field)
                    for field in node.get("fields") or []
                ],
            }
            for anchor, node in materialized.by_anchor.items()
        ]

        connections = []
        for child, parent in selected_parent.items():
            parent_source = int(
                materialized.by_anchor[parent].get(
                    "source_anchor_index", parent
                )
            )
            child_source = int(
                materialized.by_anchor[child].get(
                    "source_anchor_index", child
                )
            )
            raw_score = (
                None
                if parent_source == child_source
                else round(
                    self._score(
                        physical_scores,
                        parent_source,
                        child_source,
                    ),
                    6,
                )
            )
            connections.append({
                "parent_logical_anchor_id": parent,
                "child_logical_anchor_id": child,
                "parent_anchor_id": parent_source,
                "child_anchor_id": child_source,
                "parent_path": list(materialized.assigned[parent]),
                "child_path": list(materialized.assigned[child]),
                "selection_reason": selection_reason.get(
                    child, "model_relation"
                ),
                "raw_score": raw_score,
                "effective_score": round(
                    self._score(scores, parent, child), 6
                ),
            })
        diagnostics["connections"] = connections
        diagnostics["merges"] = [
            {
                "merged_logical_anchor_id": merged_anchor,
                "target_logical_anchor_id": target_anchor,
                "merged_source_anchor_id": int(
                    materialized.by_anchor[merged_anchor].get(
                        "source_anchor_index", merged_anchor
                    )
                ),
                "target_source_anchor_id": int(
                    materialized.by_anchor[target_anchor].get(
                        "source_anchor_index", target_anchor
                    )
                ),
            }
            for merged_anchor, target_anchor in merged_into.items()
        ]

    def _align_and_format(
        self,
        payload,
        start_map,
        end_map,
        text,
        *,
        diagnostics=None,
    ):
        mapping = payload.get("mapping")
        hierarchy, by_path, field_meta = self._hierarchy(mapping)

        scores = payload.get("relation_scores")
        if scores is not None and not torch.is_tensor(scores):
            scores = torch.as_tensor(scores)
        physical_scores = scores
        raw_nodes = list(payload.get("nodes") or [])
        self._initialize_diagnostics(
            raw_nodes,
            physical_scores,
            diagnostics,
        )

        if not by_path:
            return []

        nodes, preserve_empty_records = self._filter_active_nodes(
            payload,
            raw_nodes,
            scores,
            by_path,
        )
        if not nodes:
            return []

        nodes, scores, recovered_mixed_anchors = self._expand_mixed_path_nodes(
            nodes, scores, by_path, field_meta,
        )
        if not nodes:
            return []

        materialized = self._materialize_nodes(
            nodes,
            scores,
            by_path,
            field_meta,
            start_map,
            end_map,
            text,
        )
        selected_parent, selection_reason = self._select_parents(
            materialized,
            by_path,
            scores,
            allow_text_fallback=recovered_mixed_anchors,
        )
        merged_anchors, merged_into = self._merge_sibling_fragments(
            materialized,
            selected_parent,
            selection_reason,
            enabled=recovered_mixed_anchors,
        )

        self._attach_children(materialized, selected_parent, by_path)
        roots = self._collect_roots(
            payload,
            nodes,
            materialized,
            selected_parent,
            merged_anchors,
            by_path,
            hierarchy,
            preserve_empty_records=preserve_empty_records,
        )
        self._finalize_diagnostics(
            diagnostics,
            materialized,
            selected_parent,
            selection_reason,
            merged_anchors,
            merged_into,
            scores,
            physical_scores,
        )
        if preserve_empty_records:
            return roots
        return _filter_empty_records(roots)


@dataclass(frozen=True)
class ReconstructedStructuringGroup:
    """Uniform result returned after optional anchor alignment."""

    instances: list[dict]
    output_mode: str | None = None
    multi_level: bool = False
    diagnostics: StructuringDiagnostics | dict | None = None


_DECODER_MODE_ALIASES = {
    "base": "flat",
    "single_level": "flat",
    "singlelevel": "flat",
    "multilevel": "multi_level",
    "hierarchical": "multi_level",
    "hierarchy": "multi_level",
}


def _component_type(
    spec: object,
    default: str,
) -> tuple[str, dict[str, Any]]:
    return parse_component_spec(
        spec,
        default,
        component_name="structuring decoder component",
        aliases=_DECODER_MODE_ALIASES,
        instance_attributes=(
            "finalize_group",
            "reconstruct_group",
        ),
        allow_bool=True,
    )


def align_structuring_anchors(
    payload: dict,
    start_map: list[int],
    end_map: list[int],
    text: str,
    *,
    relation_threshold: float = 0.5,
    diagnostics: dict | None = None,
) -> list[dict]:
    """Apply the optional anchor-alignment and nested-formatting step."""

    return _AnchorAlignment(relation_threshold).align_and_format(
        payload,
        start_map,
        end_map,
        text,
        diagnostics=diagnostics,
    )


class StructuringDecoderComponent:
    """Common formatting path with optional anchor alignment."""

    def __init__(
        self,
        mode: str = "flat",
        *,
        relation_threshold: float = 0.5,
        anchor_relations_threshold: float | None = None,
    ):
        if mode not in {"flat", "multi_level"}:
            raise ValueError(f"Unknown structuring decoder mode {mode!r}")
        if anchor_relations_threshold is not None:
            relation_threshold = anchor_relations_threshold
        self.multi_level = mode == "multi_level"
        self.relation_threshold = float(relation_threshold)

    def finalize_group(
        self,
        entries: list[StructuringAnchorEntry],
        *,
        relation_scores=None,
        mapping=None,
        output_mode: str = "schemas",
        preserve_empty_records: bool = False,
    ) -> list[list[dict]] | dict:
        align_anchors = bool(
            getattr(mapping, "multi_level", self.multi_level)
        )
        if not align_anchors:
            return [entry["fields"] for entry in entries if entry["fields"]]
        return make_multi_level_group_result(
            entries,
            relation_scores,
            mapping,
            output_mode,
            preserve_empty_records=preserve_empty_records,
        )

    def reconstruct_group(
        self,
        payload: object,
        start_map: list[int],
        end_map: list[int],
        text: str,
        *,
        schema_fields: list[str],
        assemble_instance: Callable,
        fill_missing_fields: Callable,
        diagnostics: dict | None = None,
    ) -> ReconstructedStructuringGroup:
        if is_multi_level_group_result(payload):
            roots = align_structuring_anchors(
                payload,
                start_map,
                end_map,
                text,
                relation_threshold=self.relation_threshold,
                diagnostics=diagnostics,
            )
            return ReconstructedStructuringGroup(
                instances=roots,
                output_mode=str(payload.get("output_mode", "schemas")),
                multi_level=True,
                diagnostics=diagnostics,
            )

        raw_instances = payload if isinstance(payload, list) else [payload]
        instances: list[dict] = []
        for instance in raw_instances:
            if isinstance(instance, list):
                instance_dict = assemble_instance(
                    instance,
                    start_map,
                    end_map,
                    text,
                    schema_fields,
                )
                if instance_dict is not None:
                    instances.append(instance_dict)
            elif isinstance(instance, dict):
                instances.append(
                    fill_missing_fields(instance, schema_fields)
                )
        return ReconstructedStructuringGroup(instances=instances)


def resolve_structuring_decoder(
    spec: object = None,
    *,
    multi_level: bool = False,
    relation_threshold: float = 0.5,
):
    """Build shared formatting with optional anchor alignment enabled."""

    default = "multi_level" if multi_level else "flat"
    mode, params = _component_type(spec, default)
    if mode == "__instance__":
        return params["instance"]
    params.setdefault("relation_threshold", relation_threshold)
    return StructuringDecoderComponent(mode, **params)


__all__ = [
    "MULTI_LEVEL_RESULT_KEY",
    "ReconstructedStructuringGroup",
    "StructuringAnchorEntry",
    "StructuringDecoderComponent",
    "align_structuring_anchors",
    "is_multi_level_group_result",
    "make_multi_level_group_result",
    "resolve_structuring_decoder",
]
