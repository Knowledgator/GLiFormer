"""Optional hierarchy normalization for shared structuring processing.

The existing structuring head consumes ``{schema: [flat_record, ...]}``.  This
module converts arbitrary JSON into that representation without changing the
head's field/span tensors:

* ordinary dictionaries are folded into the nearest record with dot-qualified
  field labels;
* every dictionary found in a record-valued list becomes another record anchor
  in the same schema group;
* a graph sidecar retains stable node ids, canonical key paths, and directed
  parent-to-child edges for relation supervision and inference reconstruction.

Keeping all nodes for one JSON tree in one schema group is intentional:
``RelationsRepLayer`` models the existing anchor axis directly as ``(BN,A,A)``.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from numbers import Integral
from typing import Any

from .structuring_types import (
    SchemaNode,
    parse_component_spec,
    schema_node_exemplar,
)

MULTI_LEVEL_META_KEY = "_glinext_structuring_multi_level"
SET_MULTI_LEVEL_META_KEY = "_glinext_set_structuring_multi_level"
MULTI_LEVEL_ROOT_KEY = "$root"
NODE_ID_KEY = "__glinext_multi_level_node_id__"
NODE_KEEP_KEY = "__glinext_multi_level_keep__"
INTERNAL_INSTANCE_KEYS = frozenset({NODE_ID_KEY, NODE_KEEP_KEY})

_ROOT_DATA_KEY = "__glinext_multi_level_root__"
_ROOT_SCHEMA_NAME = "root"


def is_internal_instance_key(key: object) -> bool:
    return key in INTERNAL_INSTANCE_KEYS


def _is_span_annotation(value: object) -> bool:
    """Return whether a dict is an already-resolved scalar span value."""

    # ``start``/``end`` are also common domain fields (date ranges, offsets,
    # time windows, and so on).  Treat a mapping as processor-internal span
    # data only when it has the complete resolved annotation signature:
    # textual value plus integer token boundaries.  This keeps ordinary JSON
    # such as ``{"start": 1938, "end": 1940}`` eligible for dot flattening.
    start = value.get("start") if isinstance(value, dict) else None
    end = value.get("end") if isinstance(value, dict) else None
    return (
        isinstance(value, dict)
        and isinstance(value.get("text"), str)
        and isinstance(start, Integral)
        and not isinstance(start, bool)
        and isinstance(end, Integral)
        and not isinstance(end, bool)
        and "start" in value
        and "end" in value
        and set(value).issubset({"text", "start", "end", "score"})
    )


def _is_schema_envelope(value: dict) -> bool:
    """Recognize the historical ``{schema: [record, ...]}`` envelope."""

    if not value:
        return True
    return all(
        isinstance(instances, list)
        and all(isinstance(instance, dict) for instance in instances)
        for instances in value.values()
    )


def _iter_object_children(value: object) -> Iterable[dict]:
    """Yield non-span dictionaries from possibly nested arrays."""

    if isinstance(value, list):
        for item in value:
            yield from _iter_object_children(item)
    elif isinstance(value, dict) and not _is_span_annotation(value):
        yield value


def _contains_object_children(value: object) -> bool:
    return next(iter(_iter_object_children(value)), None) is not None


def _iter_primitive_children(value: object) -> Iterable[object]:
    """Yield primitive members from arbitrarily nested arrays."""

    if isinstance(value, list):
        for item in value:
            yield from _iter_primitive_children(item)
    elif not isinstance(value, dict):
        yield value


def _array_rank(value: object) -> int:
    if not isinstance(value, list):
        return 0
    return 1 + max((_array_rank(item) for item in value), default=0)


def _path_label(path: tuple[str, ...]) -> str:
    return ".".join(path)


def _escaped_path_label(path: tuple[str, ...]) -> str:
    return ".".join(
        segment.replace("\\", "\\\\").replace(".", "\\.")
        for segment in path
    )


class _HierarchyNormalizer:
    """Optional hierarchy normalization used by structuring processing."""

    multi_level = True

    def __init__(
        self,
        child_token: str,
        end_token: str,
        *,
        data_key: str = "structuring",
        schema_key: str = "structuring_schema",
        meta_key: str = MULTI_LEVEL_META_KEY,
    ):
        self.child_token = child_token
        self.end_token = end_token
        self.data_key = str(data_key)
        self.schema_key = str(schema_key)
        self.meta_key = str(meta_key)

    @staticmethod
    def _new_group(data_key: str, name: str) -> dict:
        return {
            "data_key": data_key,
            "name": name,
            "instances": [],
            "hierarchy": [],
            "relations": [],
            "fields": [],
            "field_paths": {},
            "field_owners": {},
            "field_specs": {},
            "node_paths": {},
            "node_instances": {},
            "next_node_id": 0,
            "hierarchy_by_path": {},
        }

    @staticmethod
    def _ensure_hierarchy_node(
        group: dict,
        path: tuple[str, ...],
        parent_path: tuple[str, ...] | None,
        parent_field_path: tuple[str, ...],
    ) -> dict:
        existing = group["hierarchy_by_path"].get(path)
        if existing is not None:
            return existing
        node = {
            "path": list(path),
            "parent_path": list(parent_path) if parent_path is not None else None,
            "parent_field_path": list(parent_field_path),
            "fields": [],
            "containers": [],
        }
        group["hierarchy_by_path"][path] = node
        group["hierarchy"].append(node)
        return node

    @staticmethod
    def _field_label(
        group: dict,
        canonical_path: tuple[str, ...],
        node_path: tuple[str, ...],
    ) -> str:
        preferred = _path_label(canonical_path)
        existing = group["field_paths"].get(preferred)
        existing_owner = group["field_owners"].get(preferred)
        if existing is None or (
            tuple(existing) == canonical_path
            and tuple(existing_owner or ()) == node_path
        ):
            return preferred

        # A literal dotted key can collide with a genuinely nested path. Keep
        # the ordinary readable label for the first field and disambiguate only
        # the collision; canonical segments remain authoritative in metadata.
        escaped = _escaped_path_label(canonical_path)
        candidate = escaped
        suffix = 2
        while (
            candidate in group["field_paths"]
            and (
                tuple(group["field_paths"][candidate]) != canonical_path
                or tuple(group["field_owners"].get(candidate) or ()) != node_path
            )
        ):
            candidate = f"{escaped}#{suffix}"
            suffix += 1
        return candidate

    def _add_leaf(
        self,
        group: dict,
        hierarchy_node: dict,
        instance: dict,
        node_path: tuple[str, ...],
        local_path: tuple[str, ...],
        value: object,
        *,
        shape_value: object = None,
    ) -> None:
        canonical_path = node_path + local_path
        label = self._field_label(group, canonical_path, node_path)
        shape_source = value if shape_value is None else shape_value
        if shape_source is None:
            shape = {"kind": "null"}
        elif isinstance(shape_source, list):
            shape = {"kind": "array", "rank": _array_rank(shape_source)}
        else:
            shape = {"kind": "scalar"}
        if label not in group["field_paths"]:
            group["field_paths"][label] = list(canonical_path)
            group["field_owners"][label] = list(node_path)
            group["fields"].append(label)
            field_spec = {
                "label": label,
                "path": list(canonical_path),
                "local_path": list(local_path),
                "shape": shape,
            }
            hierarchy_node["fields"].append(field_spec)
            group["field_specs"][label] = field_spec
        else:
            field_spec = group["field_specs"].get(label)
            previous_shape = field_spec.get("shape", {}) if field_spec else {}
            previous_kind = previous_shape.get("kind")
            current_kind = shape["kind"]
            if field_spec is not None and previous_kind == "null":
                field_spec["shape"] = shape
            elif field_spec is not None and current_kind == "null":
                pass
            elif field_spec is not None and previous_kind == "mixed":
                pass
            elif field_spec is not None and previous_kind != current_kind:
                # A union schema can contain the same field as both scalar and
                # array.  Keep the historical occurrence-count behaviour for
                # that genuinely ambiguous case.
                field_spec["shape"] = {"kind": "mixed"}
            elif (
                field_spec is not None
                and current_kind == "array"
                and previous_kind == "array"
            ):
                field_spec["shape"]["rank"] = max(
                    int(previous_shape.get("rank", 1)),
                    int(shape.get("rank", 1)),
                )
        instance[label] = deepcopy(value)

    @staticmethod
    def _add_container(
        hierarchy_node: dict,
        local_path: tuple[str, ...],
        kind: str,
        *,
        rank: int = 1,
    ) -> None:
        """Record inline container shape even when it has no scalar leaves."""

        for container in hierarchy_node["containers"]:
            if tuple(container.get("local_path") or ()) != local_path:
                continue
            if container.get("kind") != kind:
                container["kind"] = "mixed"
                container.pop("rank", None)
            elif kind == "array":
                container["rank"] = max(
                    int(container.get("rank", 1)),
                    int(rank),
                )
            return
        container = {"local_path": list(local_path), "kind": kind}
        if kind == "array":
            container["rank"] = int(rank)
        hierarchy_node["containers"].append(container)

    def _walk_inline_mapping(
        self,
        mapping: dict,
        *,
        group: dict,
        hierarchy_node: dict,
        instance: dict,
        node_id: int,
        node_path: tuple[str, ...],
        prefix: tuple[str, ...],
    ) -> None:
        for raw_key, value in mapping.items():
            if is_internal_instance_key(raw_key):
                continue
            key = str(raw_key)
            local_path = prefix + (key,)

            if _is_span_annotation(value):
                self._add_leaf(
                    group, hierarchy_node, instance, node_path, local_path, value,
                )
                continue

            if isinstance(value, dict):
                self._add_container(hierarchy_node, local_path, "object")
                self._walk_inline_mapping(
                    value,
                    group=group,
                    hierarchy_node=hierarchy_node,
                    instance=instance,
                    node_id=node_id,
                    node_path=node_path,
                    prefix=local_path,
                )
                continue

            if isinstance(value, list) and _contains_object_children(value):
                self._add_container(
                    hierarchy_node,
                    local_path,
                    "array",
                    rank=_array_rank(value),
                )
                child_path = node_path + local_path
                self._ensure_hierarchy_node(
                    group,
                    child_path,
                    node_path,
                    local_path,
                )
                # Mixed scalar/object arrays are uncommon but valid JSON. Keep
                # their primitive values as a normal field while object values
                # receive anchors; reconstruction merges them under one key.
                primitive_values = [
                    deepcopy(item)
                    for item in _iter_primitive_children(value)
                ]
                if primitive_values:
                    self._add_leaf(
                        group,
                        hierarchy_node,
                        instance,
                        node_path,
                        local_path,
                        primitive_values,
                        shape_value=value,
                    )

                for child in _iter_object_children(value):
                    self._walk_object(
                        child,
                        group=group,
                        node_path=child_path,
                        parent_node_id=node_id,
                    )
                continue

            if isinstance(value, list):
                self._add_container(
                    hierarchy_node,
                    local_path,
                    "array",
                    rank=_array_rank(value),
                )

            self._add_leaf(
                group, hierarchy_node, instance, node_path, local_path, value,
            )

    def _walk_object(
        self,
        value: object,
        *,
        group: dict,
        node_path: tuple[str, ...],
        parent_node_id: int | None,
    ) -> int:
        if not isinstance(value, dict):
            value = {"value": value}

        node_id = int(group["next_node_id"])
        group["next_node_id"] = node_id + 1
        # Every JSON object is a structural target, including empty objects and
        # leaf objects whose scalar annotations cannot be grounded in text.
        # Keeping the anchor is what makes the relation graph lossless.
        instance = {NODE_ID_KEY: node_id, NODE_KEEP_KEY: True}
        group["instances"].append(instance)
        group["node_paths"][node_id] = list(node_path)
        group["node_instances"][node_id] = instance

        hierarchy_node = self._ensure_hierarchy_node(
            group,
            node_path,
            None if parent_node_id is None else tuple(
                group["node_paths"][parent_node_id]
            ),
            (),
        )
        if parent_node_id is not None:
            group["relations"].append([parent_node_id, node_id])
            # A parent with no primitive fields must survive span pruning so it
            # can still receive objectness and relation supervision.
            group["node_instances"][parent_node_id][NODE_KEEP_KEY] = True

        self._walk_inline_mapping(
            value,
            group=group,
            hierarchy_node=hierarchy_node,
            instance=instance,
            node_id=node_id,
            node_path=node_path,
            prefix=(),
        )
        return node_id

    @staticmethod
    def _public_group(group: dict) -> dict:
        return {
            "data_key": group["data_key"],
            "name": group["name"],
            "hierarchy": group["hierarchy"],
            "relations": group["relations"],
            "node_paths": {
                str(key): value for key, value in group["node_paths"].items()
            },
            "fields": group["fields"],
        }

    @staticmethod
    def _order_fields_for_prompt(group: dict) -> None:
        """Align class ids with the hierarchy prompt's field-token order."""

        by_path = {
            tuple(node.get("path") or ()): node
            for node in group["hierarchy"]
        }
        children: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
        for node in group["hierarchy"]:
            parent_path = node.get("parent_path")
            if parent_path is not None:
                children.setdefault(tuple(parent_path), []).append(
                    tuple(node.get("path") or ())
                )

        ordered = []
        seen = set()

        def visit(path: tuple[str, ...]) -> None:
            node = by_path.get(path)
            if node is None:
                return
            for field in node.get("fields") or []:
                label = field.get("label") if isinstance(field, dict) else field
                if label and label not in seen:
                    ordered.append(label)
                    seen.add(label)
            for child_path in children.get(path, []):
                visit(child_path)

        visit(())
        ordered.extend(label for label in group["fields"] if label not in seen)
        group["fields"] = ordered

    def normalize_item(
        self,
        item: dict,
        *,
        forced_output_mode: str | None = None,
    ) -> dict | None:
        """Normalize ``item['structuring']`` in-place, idempotently."""

        existing = item.get(self.meta_key)
        if isinstance(existing, dict):
            return existing

        raw = item.get(self.data_key)
        if not isinstance(raw, dict | list):
            return None

        raw = deepcopy(raw)
        if (
            isinstance(raw, dict)
            and set(raw) == {MULTI_LEVEL_ROOT_KEY}
        ):
            raw = raw[MULTI_LEVEL_ROOT_KEY]
            forced_output_mode = (
                "list" if isinstance(raw, list) else "object"
            )
        if forced_output_mode not in {None, "object", "list", "schemas"}:
            raise ValueError(
                "forced_output_mode must be object, list, schemas, or None"
            )
        if forced_output_mode == "object" and not isinstance(raw, dict):
            raise TypeError("A raw object root must be a dictionary")
        if forced_output_mode == "list" and not isinstance(raw, list):
            raise TypeError("A raw list root must be a list")
        if forced_output_mode == "schemas" and not isinstance(raw, dict):
            raise TypeError("Named structuring schemas must be a dictionary")
        existing_schema = deepcopy(item.get(self.schema_key) or {})
        groups: list[dict] = []

        if isinstance(raw, list) and forced_output_mode != "schemas":
            output_mode = "list"
            group = self._new_group(_ROOT_DATA_KEY, _ROOT_SCHEMA_NAME)
            groups.append(group)
            self._ensure_hierarchy_node(group, (), None, ())
            for value in raw:
                self._walk_object(
                    value,
                    group=group,
                    node_path=(),
                    parent_node_id=None,
                )
        elif forced_output_mode == "schemas" or (
            forced_output_mode is None and _is_schema_envelope(raw)
        ):
            output_mode = "schemas"
            for raw_name, values in raw.items():
                name = str(raw_name)
                group = self._new_group(name, name)
                groups.append(group)
                self._ensure_hierarchy_node(group, (), None, ())
                for value in values:
                    self._walk_object(
                        value,
                        group=group,
                        node_path=(),
                        parent_node_id=None,
                    )
        else:
            output_mode = "object"
            group = self._new_group(_ROOT_DATA_KEY, _ROOT_SCHEMA_NAME)
            groups.append(group)
            self._ensure_hierarchy_node(group, (), None, ())
            self._walk_object(
                raw,
                group=group,
                node_path=(),
                parent_node_id=None,
            )

        normalized = {}
        schema = {}
        public_groups = []
        for group in groups:
            self._order_fields_for_prompt(group)
            data_key = group["data_key"]
            normalized[data_key] = group["instances"]
            previous_spec = existing_schema.get(group["name"], {})
            required_fields = (
                list(previous_spec.get("required_fields") or [])
                if isinstance(previous_spec, dict)
                else []
            )
            schema[data_key] = {
                "fields": list(group["fields"]),
                "required_fields": required_fields,
                "hierarchy": deepcopy(group["hierarchy"]),
            }
            public_groups.append(self._public_group(group))

        meta = {
            "output_mode": output_mode,
            "groups": public_groups,
        }
        item[self.data_key] = normalized
        item[self.schema_key] = schema
        item[self.meta_key] = meta
        return meta

    @classmethod
    def _template_from_spec(cls, spec: object) -> tuple[dict, list[str]]:
        # Local import avoids a module cycle while keeping Pydantic/annotation
        # handling in the same compiler used by GLiNExTSchema.
        from .schema import compile_structuring_template

        node = compile_structuring_template(spec)
        if node.kind == "array":
            node = node.item or SchemaNode("object")
        template = schema_node_exemplar(node)
        if not isinstance(template, dict):
            raise TypeError("A structuring schema must describe an object")
        return template, list(node.required_fields)

    def contribute_inference_input(self, item: dict, structures: object) -> None:
        """Create a nested schema exemplar and run the shared normalizer."""

        if isinstance(structures, list):
            template, required = self._template_from_spec(structures)
            item[self.data_key] = [template]
            item[self.schema_key] = {
                _ROOT_SCHEMA_NAME: {"required_fields": required}
            }
            self.normalize_item(item)
            return

        if not isinstance(structures, dict):
            return

        if set(structures) == {MULTI_LEVEL_ROOT_KEY}:
            root_spec = structures[MULTI_LEVEL_ROOT_KEY]
            from .schema import compile_structuring_template

            root_node = compile_structuring_template(root_spec)
            if root_node.kind == "object":
                template = schema_node_exemplar(root_node)
                required = list(root_node.required_fields)
                item[self.data_key] = template
                item[self.schema_key] = {
                    _ROOT_SCHEMA_NAME: {"required_fields": required}
                }
                self.normalize_item(item, forced_output_mode="object")
                return
            if root_node.kind == "array":
                root_item = root_node.item or SchemaNode("object")
                template = schema_node_exemplar(root_item)
                required = list(root_item.required_fields)
                item[self.data_key] = [template]
                item[self.schema_key] = {
                    _ROOT_SCHEMA_NAME: {"required_fields": required}
                }
                self.normalize_item(item, forced_output_mode="list")
                return
            raise TypeError("$root structuring schema must describe an object or list")

        structuring = {}
        structuring_schema = {}
        for raw_name, spec in structures.items():
            name = str(raw_name)
            template, required = self._template_from_spec(spec)
            structuring[name] = [template]
            structuring_schema[name] = {"required_fields": required}
        item[self.data_key] = structuring
        item[self.schema_key] = structuring_schema
        self.normalize_item(item)

    def contribute_prompt(
        self,
        struct_item,
        *,
        parent_token: str,
        field_token: str,
        sep_token: str,
        use_labels_encoder: bool,
    ) -> list[str]:
        """Render a depth-first ``CHILD ... END`` hierarchy prompt."""

        prompt = [parent_token]
        field_map = struct_item.field_class_to_id
        if field_map.name:
            prompt.append(field_map.name)
        if field_map.description:
            prompt.append(field_map.description)

        hierarchy = list(getattr(struct_item, "hierarchy", None) or [])
        if not hierarchy:
            if not use_labels_encoder:
                prompt.extend(
                    f"{field_token} {field_name}"
                    for field_name in field_map.class_to_id
                )
            prompt.append(sep_token)
            return prompt

        by_path = {tuple(node.get("path") or ()): node for node in hierarchy}
        children: dict[tuple[str, ...], list[dict]] = {}
        for node in hierarchy:
            parent_path = node.get("parent_path")
            if parent_path is None:
                continue
            children.setdefault(tuple(parent_path), []).append(node)

        def render(node: dict) -> None:
            if not use_labels_encoder:
                for field in node.get("fields") or []:
                    label = field.get("label") if isinstance(field, dict) else field
                    if label:
                        prompt.append(f"{field_token} {label}")
            for child in children.get(tuple(node.get("path") or ()), []):
                child_name = ".".join(child.get("parent_field_path") or [])
                marker = self.child_token
                prompt.append(f"{marker} {child_name}" if child_name else marker)
                render(child)
                prompt.append(self.end_token)

        root = by_path.get(())
        if root is not None:
            render(root)
        prompt.append(sep_token)
        return prompt


_PROCESSOR_MODE_ALIASES = {
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
        component_name="structuring processor component",
        aliases=_PROCESSOR_MODE_ALIASES,
        instance_attributes=(
            "normalize_item",
            "contribute_inference_input",
            "contribute_prompt",
        ),
    )


class StructuringProcessorComponent:
    """Shared input processing with optional hierarchy normalization."""

    def __init__(
        self,
        mode: str = "flat",
        *,
        child_token: str,
        end_token: str,
        data_key: str = "structuring",
        schema_key: str = "structuring_schema",
        meta_key: str = MULTI_LEVEL_META_KEY,
    ):
        if mode not in {"flat", "multi_level"}:
            raise ValueError(f"Unknown structuring processor mode {mode!r}")
        self.multi_level = mode == "multi_level"
        self.data_key = str(data_key)
        self.schema_key = str(schema_key)
        self._hierarchy = (
            _HierarchyNormalizer(
                child_token=child_token,
                end_token=end_token,
                data_key=data_key,
                schema_key=schema_key,
                meta_key=meta_key,
            )
            if self.multi_level
            else None
        )

    def normalize_item(self, item: dict, **kwargs):
        if self._hierarchy is None:
            return None
        return self._hierarchy.normalize_item(item, **kwargs)

    def contribute_inference_input(
        self,
        item: dict,
        structures: object,
    ) -> None:
        if self._hierarchy is not None:
            self._hierarchy.contribute_inference_input(item, structures)
            return
        if isinstance(structures, list):
            raise ValueError(
                "Root-list or nested structuring schemas require "
                "multi_level=True or structure_mode='multi_level'"
            )
        if not isinstance(structures, dict):
            raise TypeError("structures must be a dictionary or list")

        data = {}
        schemas = {}
        for raw_name, spec in structures.items():
            schema_name = str(raw_name)
            from .schema import (
                compile_structuring_template,
                schema_node_requires_multi_level,
            )

            node = compile_structuring_template(spec)
            if schema_node_requires_multi_level(node):
                raise ValueError(
                    "This structuring template contains nested objects or "
                    "record arrays, but the loaded checkpoint is configured "
                    "for flat structuring. Nested schemas require "
                    "multi_level=True or structure_mode='multi_level' on a "
                    "compatible checkpoint."
                )
            if node.kind != "object":
                raise TypeError(
                    "A flat named structuring schema must describe an object"
                )
            fields = list(node.fields)
            data[schema_name] = (
                [dict.fromkeys(fields, "")] if fields else []
            )
            schema_spec = {
                "fields": fields,
                "required_fields": list(node.required_fields),
            }
            if node.description is not None:
                schema_spec["description"] = node.description
            schemas[schema_name] = schema_spec
        item[self.data_key] = data
        item[self.schema_key] = schemas

    def contribute_prompt(
        self,
        struct_item,
        *,
        parent_token: str,
        field_token: str,
        sep_token: str,
        use_labels_encoder: bool,
    ) -> list[str]:
        if self._hierarchy is not None:
            return self._hierarchy.contribute_prompt(
                struct_item,
                parent_token=parent_token,
                field_token=field_token,
                sep_token=sep_token,
                use_labels_encoder=use_labels_encoder,
            )
        prompt = [parent_token]
        field_map = struct_item.field_class_to_id
        if field_map.name:
            prompt.append(field_map.name)
        if field_map.description:
            prompt.append(field_map.description)
        if not use_labels_encoder:
            prompt.extend(
                f"{field_token} {field_name}"
                for field_name in field_map.class_to_id
            )
        prompt.append(sep_token)
        return prompt


def resolve_structuring_processor(
    spec: object = None,
    *,
    multi_level: bool = False,
    child_token: str,
    end_token: str,
    data_key: str = "structuring",
    schema_key: str = "structuring_schema",
    meta_key: str,
):
    """Build the common structuring processor with optional hierarchy hooks."""

    default = "multi_level" if multi_level else "flat"
    mode, params = _component_type(spec, default)
    if mode == "__instance__":
        return params["instance"]
    return StructuringProcessorComponent(
        mode,
        child_token=child_token,
        end_token=end_token,
        data_key=data_key,
        schema_key=schema_key,
        meta_key=meta_key,
        **params,
    )


__all__ = [
    "INTERNAL_INSTANCE_KEYS",
    "MULTI_LEVEL_META_KEY",
    "SET_MULTI_LEVEL_META_KEY",
    "MULTI_LEVEL_ROOT_KEY",
    "NODE_ID_KEY",
    "NODE_KEEP_KEY",
    "StructuringProcessorComponent",
    "is_internal_instance_key",
    "resolve_structuring_processor",
]
