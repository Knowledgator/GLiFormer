"""Inference graph reconstruction for opt-in multi-level structuring."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

import torch

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


def _paths_conflict(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    """Return whether one JSON leaf path is a prefix of the other."""

    shared = min(len(left), len(right))
    return left != right and left[:shared] == right[:shared]


class MultiLevelStructuringDecoder:
    """Turn qualified anchor fields plus directed edges back into nested JSON."""

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

    def reconstruct_group(
        self,
        payload: dict,
        start_map: list[int],
        end_map: list[int],
        text: str,
        *,
        relation_threshold: float | None = None,
        diagnostics: dict | None = None,
    ) -> list[dict]:
        """Reconstruct one schema group's root records."""

        old_threshold = self.relation_threshold
        if relation_threshold is not None:
            self.relation_threshold = float(relation_threshold)
        try:
            return self._reconstruct_group(
                payload,
                start_map,
                end_map,
                text,
                diagnostics=diagnostics,
            )
        finally:
            self.relation_threshold = old_threshold

    @staticmethod
    def _diagnostic_field(field: dict) -> dict:
        return {
            "field": str(field.get("field", "")),
            "text": str(field.get("text", "")),
            "score": round(float(field.get("score", 0.0)), 6),
        }

    def _reconstruct_group(
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
        raw_anchors = [int(node["anchor_index"]) for node in raw_nodes]

        if diagnostics is not None:
            active_ids = sorted(set(raw_anchors))
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
                        score = self._score(
                            physical_scores, parent, child,
                        )
                        edge = {
                            "parent_anchor_id": parent,
                            "child_anchor_id": child,
                            "score": round(score, 6),
                        }
                        raw_candidates.append(edge)
                        if score >= self.relation_threshold:
                            diagnostics[
                                "raw_relation_connections"
                            ].append(edge)
            raw_candidates.sort(
                key=lambda edge: (
                    -edge["score"],
                    edge["parent_anchor_id"],
                    edge["child_anchor_id"],
                )
            )
            diagnostics["top_raw_relation_candidates"] = raw_candidates[:5]

        if not by_path:
            return []

        # Fully empty fixed slots are not JSON objects. A container-only slot is
        # retained when it participates in a confident graph edge.
        connected = set()
        has_child_anchors = any(path for path in by_path)
        if scores is not None and has_child_anchors:
            for parent in raw_anchors:
                for child in raw_anchors:
                    if parent == child:
                        continue
                    if self._score(scores, parent, child) >= self.relation_threshold:
                        connected.update((parent, child))
        preserve_empty_records = bool(
            payload.get("preserve_empty_records", False)
        )
        nodes = [
            node for node in raw_nodes
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
        if not nodes:
            return []

        nodes, scores, recovered_mixed_anchors = self._expand_mixed_path_nodes(
            nodes, scores, by_path, field_meta,
        )
        if not nodes:
            return []

        assigned = self._assign_node_paths(nodes, scores, by_path, field_meta)
        node_objects = {}
        node_positions = {}
        node_value_entries = {}
        node_by_anchor = {int(node["anchor_index"]): node for node in nodes}

        for anchor, node in node_by_anchor.items():
            node_path = assigned[anchor]
            obj = {}
            node_schema = by_path[node_path]
            containers = sorted(
                (
                    container
                    for container in node_schema.get("containers") or []
                    if isinstance(container, dict)
                ),
                key=lambda container: len(container.get("local_path") or ()),
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
                        _shape_array([], int(container.get("rank", 1))),
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
                tuple(str(x) for x in field.get("local_path") or ()): field
                for field in node_schema.get("fields") or []
                if isinstance(field, dict)
            }
            evidence_scores = {
                path: sum(-value[1] for value in values)
                for path, values in occurrences.items()
                if values
            }
            selected_paths = []
            for path in sorted(
                evidence_scores,
                key=lambda candidate: (
                    -evidence_scores[candidate], len(candidate), candidate,
                ),
            ):
                if not any(
                    _paths_conflict(path, selected)
                    for selected in selected_paths
                ):
                    selected_paths.append(path)

            # Assemble evidence before schema defaults. This matters for union
            # paths such as ``x`` (scalar) versus ``x.value`` (object): a
            # missing branch must never overwrite the branch actually seen on
            # this anchor.
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
                    # A scalar schema field cannot become a list merely
                    # because several noisy candidates landed on one anchor.
                    # Prefer the highest-confidence occurrence, breaking ties
                    # by text order for deterministic output.
                    best = min(values, key=lambda value: (value[1], value[0]))
                    _set_path(obj, local_path, best[2])

            # Preserve predictable schema defaults only on branches that do
            # not conflict with observed evidence. Process shallow defaults
            # first so an object-shaped descendant wins over an ambiguous
            # missing scalar at the same prefix.
            for local_path, field in sorted(
                field_specs.items(), key=lambda item: (len(item[0]), item[0]),
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
                path: [(value[0], value[2]) for value in sorted(values)]
                for path, values in occurrences.items()
                if path in selected_paths
            }
            node_positions[anchor] = self._earliest_position(node)

        # Pick at most one compatible parent for every non-root node. Using the
        # hierarchy path as a guard rejects self-edges, cycles, and cross-branch
        # links even when their raw relation probability is high.
        selected_parent = {}
        selection_reason = {}
        for child, child_path in assigned.items():
            child_meta = by_path[child_path]
            expected_parent_raw = child_meta.get("parent_path")
            if expected_parent_raw is None:
                continue
            expected_parent = tuple(expected_parent_raw)
            candidates = []
            for parent, parent_path in assigned.items():
                if parent == child or parent_path != expected_parent:
                    continue
                score = self._score(scores, parent, child)
                if score >= self.relation_threshold:
                    candidates.append((score, -parent, parent))
            if candidates:
                selected_parent[child] = max(candidates)[2]
                parent = selected_parent[child]
                parent_source = int(
                    node_by_anchor[parent].get(
                        "source_anchor_index", parent,
                    )
                )
                child_source = int(
                    node_by_anchor[child].get(
                        "source_anchor_index", child,
                    )
                )
                selection_reason[child] = (
                    "same_physical_anchor"
                    if parent_source == child_source
                    else "model_relation"
                )
                continue

            # A collapsed subtree is strong evidence that this example uses
            # parent-before-child textual order even when the learned edge for
            # a separately anchored sibling is under-confident.  Use only the
            # nearest compatible preceding parent; ordinary disconnected
            # predictions retain the strict relation threshold above.
            if recovered_mixed_anchors:
                child_position = node_positions[child][0]
                preceding = []
                for parent, parent_path in assigned.items():
                    parent_position = node_positions[parent][0]
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

        # A collapsed prediction can also split complementary scalar fields
        # for one object across two physical slots (for example, a milestone
        # name on one slot and its due date on another). Once both fragments
        # resolve to the same parent, merge only close, non-conflicting sibling
        # fragments. Repeated fields still delimit separate JSON objects.
        merged_anchors = set()
        merged_into = {}
        if recovered_mixed_anchors:
            siblings = defaultdict(list)
            for child, parent in selected_parent.items():
                child_path = assigned[child]
                if child_path:
                    siblings[(parent, child_path)].append(child)

            for sibling_nodes in siblings.values():
                sibling_nodes.sort(key=lambda anchor: node_positions[anchor])
                representative = None
                representative_paths = set()
                representative_sources = set()
                previous_position = None
                for child in sibling_nodes:
                    evidence_paths = set(node_value_entries.get(child, {}))
                    source_anchor = int(
                        node_by_anchor[child].get(
                            "source_anchor_index", child,
                        )
                    )
                    child_position = node_positions[child][0]
                    conflicts = any(
                        left == right or _paths_conflict(left, right)
                        for left in representative_paths
                        for right in evidence_paths
                    )
                    can_merge = (
                        representative is not None
                        and representative_paths
                        and evidence_paths
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

                    for local_path, values in node_value_entries[child].items():
                        _set_path(
                            node_objects[representative],
                            local_path,
                            deepcopy(_get_path(node_objects[child], local_path)),
                        )
                        node_value_entries[representative][local_path] = list(
                            values
                        )
                    representative_paths.update(evidence_paths)
                    representative_sources.add(source_anchor)
                    previous_position = child_position
                    merged_into[child] = representative
                    merged_anchors.add(child)

            if merged_into:
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

        children_by_parent = defaultdict(list)
        for child, parent in selected_parent.items():
            children_by_parent[parent].append(child)
        for children in children_by_parent.values():
            children.sort(key=lambda child: node_positions[child])

        # Merge primitive and object members of mixed JSON arrays by their text
        # position. Child dictionaries are mutable references, so deeper
        # attachments remain visible after their parent has been installed.
        attachments = defaultdict(list)
        for parent, children in children_by_parent.items():
            for child in children:
                child_meta = by_path[assigned[child]]
                attach_path = tuple(
                    str(x) for x in child_meta.get("parent_field_path") or ()
                )
                attachments[(parent, attach_path)].append((
                    node_positions[child][0],
                    node_objects[child],
                ))
        for (parent, attach_path), child_entries in attachments.items():
            values = list(
                node_value_entries.get(parent, {}).get(attach_path, [])
            )
            values.extend(child_entries)
            values.sort(key=lambda entry: entry[0])
            container_rank = 1
            for container in by_path[assigned[parent]].get("containers") or []:
                if (
                    isinstance(container, dict)
                    and tuple(
                        str(segment)
                        for segment in container.get("local_path") or ()
                    ) == attach_path
                    and container.get("kind") == "array"
                ):
                    container_rank = int(container.get("rank", 1))
                    break
            _set_path(
                node_objects[parent],
                attach_path,
                _shape_array(
                    [value for _, value in values],
                    container_rank,
                ),
            )

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
            anchor for anchor, path in assigned.items()
            if (
                anchor in structural_anchors
                and path == ()
                and anchor not in selected_parent
            )
        ]
        root_anchors.sort(key=lambda anchor: node_positions[anchor])
        roots = [node_objects[anchor] for anchor in root_anchors]

        # Preserve disconnected child predictions by wrapping the missing
        # ancestors according to the schema instead of returning a malformed
        # object at the wrong level.
        for anchor, path in assigned.items():
            if (
                anchor not in structural_anchors
                or anchor in merged_anchors
                or path == ()
                or anchor in selected_parent
            ):
                continue
            wrapped = deepcopy(node_objects[anchor])
            current_path = path
            while current_path:
                metadata = by_path[current_path]
                parent_path = tuple(metadata.get("parent_path") or ())
                parent_obj = {}
                attach_path = tuple(
                    str(x) for x in metadata.get("parent_field_path") or ()
                )
                _set_path(parent_obj, attach_path, [wrapped])
                wrapped = parent_obj
                current_path = parent_path
            roots.append(wrapped)
        roots = [
            self._order_object_by_schema(root, (), hierarchy, by_path)
            for root in roots
        ]
        if diagnostics is not None:
            logical_nodes = []
            for anchor, node in node_by_anchor.items():
                source_anchor = int(
                    node.get("source_anchor_index", anchor)
                )
                logical_nodes.append({
                    "logical_anchor_id": anchor,
                    "source_anchor_id": source_anchor,
                    "schema_path": list(assigned[anchor]),
                    "state": (
                        "merged"
                        if anchor in merged_anchors
                        else "active"
                    ),
                    "merged_into_logical_anchor_id": merged_into.get(anchor),
                    "fields": [
                        self._diagnostic_field(field)
                        for field in node.get("fields") or []
                    ],
                })
            diagnostics["logical_nodes"] = logical_nodes

            connections = []
            for child, parent in selected_parent.items():
                parent_source = int(
                    node_by_anchor[parent].get(
                        "source_anchor_index", parent,
                    )
                )
                child_source = int(
                    node_by_anchor[child].get(
                        "source_anchor_index", child,
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
                    "parent_path": list(assigned[parent]),
                    "child_path": list(assigned[child]),
                    "selection_reason": selection_reason.get(
                        child, "model_relation",
                    ),
                    "raw_score": raw_score,
                    "effective_score": round(
                        self._score(scores, parent, child), 6,
                    ),
                })
            diagnostics["connections"] = connections
            diagnostics["merges"] = [
                {
                    "merged_logical_anchor_id": merged_anchor,
                    "target_logical_anchor_id": target_anchor,
                    "merged_source_anchor_id": int(
                        node_by_anchor[merged_anchor].get(
                            "source_anchor_index", merged_anchor,
                        )
                    ),
                    "target_source_anchor_id": int(
                        node_by_anchor[target_anchor].get(
                            "source_anchor_index", target_anchor,
                        )
                    ),
                }
                for merged_anchor, target_anchor in merged_into.items()
            ]
        return roots


__all__ = [
    "MULTI_LEVEL_RESULT_KEY",
    "MultiLevelStructuringDecoder",
    "is_multi_level_group_result",
    "make_multi_level_group_result",
]
