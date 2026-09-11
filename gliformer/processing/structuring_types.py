"""Canonical internal contracts for structuring schemas and components.

Public APIs still accept the historical dictionary descriptors.  They are
compiled into :class:`SchemaNode` at the boundary so processing, formatting,
and decoding share one recursive representation.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypedDict


class _MissingType:
    def __repr__(self) -> str:
        return "MISSING"


MISSING = _MissingType()


class StructureMode(str, Enum):
    FLAT = "flat"
    MULTI_LEVEL = "multi_level"


class OutputMode(str, Enum):
    SCHEMAS = "schemas"
    OBJECT = "object"
    LIST = "list"


class NodeKind(str, Enum):
    SCALAR = "scalar"
    OBJECT = "object"
    ARRAY = "array"


Path = tuple[str, ...]


TYPE_NAME_ALIASES = {
    "str": "str",
    "string": "str",
    "int": "int",
    "integer": "int",
    "float": "float",
    "number": "float",
    "bool": "bool",
    "boolean": "bool",
    "date": "date",
    "datetime": "datetime",
    "list": "list",
    "array": "list",
}


def normalize_type_name(type_name: str) -> str:
    """Normalize portable JSON-style names and their Python aliases."""

    normalized = str(type_name).strip().lower()
    return TYPE_NAME_ALIASES.get(normalized, normalized)


class FieldType:
    """Conversion settings for a scalar structured field.

    ``MISSING`` distinguishes an omitted fallback from an explicit ``None``.
    This keeps legacy behaviour for ``FieldType("int")`` while allowing users
    to intentionally return ``None`` on conversion failure.
    """

    def __init__(
        self,
        type_name: str | Callable = "str",
        default: Any = MISSING,
        list_separator: str = r"\s*,\s*",
        list_item_type: str = "str",
        date_formats: list[str] | None = None,
        required: bool = False,
        default_factory: Callable[[], Any] | None = None,
        choices: list[Any] | tuple[Any, ...] | None = None,
        nullable: bool = False,
        description: str | None = None,
    ):
        self.type_name = (
            normalize_type_name(type_name)
            if isinstance(type_name, str)
            else type_name
        )
        self.default = default
        self.default_factory = default_factory
        self.list_separator = list_separator
        self.list_item_type = normalize_type_name(list_item_type)
        self.required = required
        self.choices = tuple(choices) if choices is not None else None
        self.nullable = bool(nullable)
        self.description = description
        self.date_formats = date_formats or [
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%d %b %Y",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ]


class UnionFieldType:
    """Scalar union whose alternatives are attempted in declaration order."""

    def __init__(self, alternatives: list[FieldType | str | Callable]):
        self.alternatives = alternatives


@dataclass
class SchemaNode:
    """Dependency-free recursive structuring schema representation."""

    kind: NodeKind | str
    type_spec: Any = None
    fields: dict[str, SchemaNode] = field(default_factory=dict)
    required_fields: list[str] = field(default_factory=list)
    item: SchemaNode | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        self.kind = NodeKind(self.kind)


@dataclass(frozen=True)
class RecursiveTypeSpec:
    """Marks an internal type tree whose dictionary keys are all literal."""

    value: Any


DESCRIPTOR_KEYS = frozenset(
    {"fields", "children", "required_fields", "description"}
)
FIELD_DESCRIPTOR_KEYS = frozenset(
    {
        "$type",
        "$items",
        "$required",
        "$default",
        "$enum",
        "$nullable",
        "$description",
    }
)
TEMPLATE_REQUIRED_KEY = "$required"


def is_structuring_descriptor(spec: object) -> bool:
    """Return whether *spec* uses the public processor descriptor shape."""

    if not isinstance(spec, Mapping) or not spec:
        return False
    if not set(spec).issubset(DESCRIPTOR_KEYS):
        return False
    if "fields" in spec and not isinstance(spec.get("fields"), list | dict):
        return False
    if "required_fields" in spec and not isinstance(
        spec.get("required_fields"), list
    ):
        return False
    if "children" in spec and not isinstance(spec.get("children"), dict):
        return False
    return any(
        key in spec for key in ("fields", "children", "required_fields")
    )


def _is_descriptor_candidate(spec: object) -> bool:
    return (
        isinstance(spec, Mapping)
        and bool(spec)
        and set(spec).issubset(DESCRIPTOR_KEYS)
        and any(
            key in spec
            for key in ("fields", "children", "required_fields")
        )
    )


ScalarNodeFactory = Callable[[Any], SchemaNode]
RequiredPredicate = Callable[[SchemaNode], bool]
ListNodeFactory = Callable[[list[Any]], SchemaNode]


def _is_field_descriptor(spec: Mapping[Any, Any]) -> bool:
    keys = set(spec)
    return bool(keys & {"$type", "$items", "$enum"})


def _infer_enum_type(values: list[Any]) -> str:
    if not values:
        raise ValueError("Structuring field '$enum' must not be empty")
    first = values[0]
    if isinstance(first, bool):
        return "bool"
    if isinstance(first, int):
        return "int"
    if isinstance(first, float):
        return "float"
    return "str"


def _configured_scalar_node(
    node: SchemaNode,
    spec: Mapping[str, Any],
) -> SchemaNode:
    if "$required" in spec and not isinstance(spec["$required"], bool):
        raise TypeError("Structuring field '$required' must be a boolean")
    if node.kind != NodeKind.SCALAR:
        unsupported = set(spec) & {
            "$default",
            "$enum",
            "$nullable",
        }
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise TypeError(
                f"Structuring field options {names} apply only to scalar fields"
            )
        node.description = (
            str(spec["$description"])
            if spec.get("$description") is not None
            else None
        )
        return node

    type_spec = node.type_spec
    if isinstance(type_spec, FieldType):
        field_type = copy.copy(type_spec)
    elif isinstance(type_spec, str) or callable(type_spec):
        field_type = FieldType(type_spec)
    else:
        field_type = FieldType("str")

    if "$required" in spec:
        field_type.required = spec["$required"]
    if "$default" in spec:
        field_type.default = spec["$default"]
    if "$enum" in spec:
        choices = spec["$enum"]
        if not isinstance(choices, list) or not choices:
            raise TypeError(
                "Structuring field '$enum' must be a non-empty list"
            )
        field_type.choices = tuple(choices)
    if "$nullable" in spec:
        nullable = spec["$nullable"]
        if not isinstance(nullable, bool):
            raise TypeError("Structuring field '$nullable' must be a boolean")
        field_type.nullable = nullable
    if "$description" in spec:
        description = spec["$description"]
        field_type.description = (
            str(description) if description is not None else None
        )
        node.description = field_type.description
    node.type_spec = field_type
    return node


def _schema_node_from_field_descriptor(
    spec: Mapping[str, Any],
    scalar_factory: ScalarNodeFactory,
    *,
    detect_descriptor: bool,
    is_required: RequiredPredicate | None,
    list_factory: ListNodeFactory | None,
) -> SchemaNode:
    unknown = set(spec) - FIELD_DESCRIPTOR_KEYS
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise TypeError(f"Unknown structuring field option(s): {names}")

    enum_values = spec.get("$enum")
    if enum_values is not None and (
        not isinstance(enum_values, list) or not enum_values
    ):
        raise TypeError("Structuring field '$enum' must be a non-empty list")

    if "$items" in spec:
        declared_type = normalize_type_name(str(spec.get("$type", "list")))
        if declared_type != "list":
            raise TypeError("A structuring field with '$items' must have type 'array' or 'list'")
        item_spec = spec["$items"]
        node = (
            list_factory([item_spec])
            if list_factory is not None
            else SchemaNode(
                NodeKind.ARRAY,
                item=schema_node_from_spec(
                    item_spec,
                    scalar_factory,
                    detect_descriptor=detect_descriptor,
                    is_required=is_required,
                    list_factory=list_factory,
                ),
            )
        )
        return _configured_scalar_node(node, spec)

    raw_type = spec.get("$type")
    if raw_type is None:
        raw_type = _infer_enum_type(enum_values)
    node = schema_node_from_spec(
        raw_type,
        scalar_factory,
        detect_descriptor=detect_descriptor,
        is_required=is_required,
        list_factory=list_factory,
    )
    return _configured_scalar_node(node, spec)


def _template_field_name(raw_name: Any) -> tuple[str, bool]:
    name = str(raw_name)
    if name.startswith("!!"):
        return name[1:], False
    if name.startswith("$$"):
        return name[1:], False
    if name.startswith("!"):
        if len(name) == 1:
            raise ValueError("A required structuring field name cannot be empty")
        return name[1:], True
    return name, False


def _required_scalar_value(spec: Any) -> tuple[Any, bool]:
    if not isinstance(spec, str):
        return spec, False
    if spec.startswith("!!"):
        return spec[1:], False
    if spec.startswith("!"):
        if len(spec) == 1:
            raise ValueError("A required structuring type cannot be empty")
        return spec[1:], True
    return spec, False


def _node_declares_required(node: SchemaNode) -> bool:
    return (
        node.kind == NodeKind.SCALAR
        and isinstance(node.type_spec, FieldType)
        and node.type_spec.required
    )


def schema_node_from_spec(
    spec: Any,
    scalar_factory: ScalarNodeFactory,
    *,
    detect_descriptor: bool = True,
    is_required: RequiredPredicate | None = None,
    list_factory: ListNodeFactory | None = None,
) -> SchemaNode:
    """Compile recursive mappings/lists/descriptors into ``SchemaNode``.

    Scalar interpretation stays with the caller because the schema builder
    understands Python/Pydantic annotations while the formatter only needs
    conversion descriptors.
    """

    if isinstance(spec, SchemaNode):
        return spec
    if isinstance(spec, Mapping):
        if detect_descriptor and _is_descriptor_candidate(spec):
            return schema_node_from_descriptor(
                spec,
                scalar_factory,
                is_required=is_required,
                list_factory=list_factory,
            )
        if _is_field_descriptor(spec):
            return _schema_node_from_field_descriptor(
                spec,
                scalar_factory,
                detect_descriptor=detect_descriptor,
                is_required=is_required,
                list_factory=list_factory,
            )

        explicit_required = spec.get(TEMPLATE_REQUIRED_KEY, [])
        if explicit_required is None:
            explicit_required = []
        if not isinstance(explicit_required, list) or not all(
            isinstance(name, str) for name in explicit_required
        ):
            raise TypeError(
                "Structuring template '$required' must be a list of field names"
            )

        node = SchemaNode(NodeKind.OBJECT)
        for raw_name, child_spec in spec.items():
            if raw_name == TEMPLATE_REQUIRED_KEY:
                continue
            name, key_required = _template_field_name(raw_name)
            child_spec, value_required = _required_scalar_value(child_spec)
            if name in node.fields:
                raise ValueError(
                    f"Duplicate structuring field {name!r} after marker escaping"
                )
            child = schema_node_from_spec(
                child_spec,
                scalar_factory,
                detect_descriptor=detect_descriptor,
                is_required=is_required,
                list_factory=list_factory,
            )
            node.fields[name] = child
            inferred_required = (
                is_required(child)
                if is_required is not None
                else _node_declares_required(child)
            )
            descriptor_required = (
                isinstance(child_spec, Mapping)
                and child_spec.get(TEMPLATE_REQUIRED_KEY) is True
            )
            if (
                key_required
                or value_required
                or descriptor_required
                or inferred_required
            ):
                node.required_fields.append(name)

        unknown_required = [
            path
            for path in explicit_required
            if path.split(".", 1)[0] not in node.fields
        ]
        if unknown_required:
            names = ", ".join(repr(name) for name in unknown_required)
            raise ValueError(
                f"Structuring template '$required' references unknown field(s): {names}"
            )
        node.required_fields = list(
            dict.fromkeys([*node.required_fields, *explicit_required])
        )
        return node
    if isinstance(spec, list):
        if list_factory is not None:
            return list_factory(spec)
        item_spec = spec[0] if spec else "str"
        return SchemaNode(
            NodeKind.ARRAY,
            item=schema_node_from_spec(
                item_spec,
                scalar_factory,
                detect_descriptor=detect_descriptor,
                is_required=is_required,
                list_factory=list_factory,
            ),
        )
    return scalar_factory(spec)


def schema_node_from_descriptor(
    spec: Mapping[str, Any],
    scalar_factory: ScalarNodeFactory,
    *,
    is_required: RequiredPredicate | None = None,
    list_factory: ListNodeFactory | None = None,
) -> SchemaNode:
    """Compile one validated legacy descriptor into ``SchemaNode``."""

    if not _is_descriptor_candidate(spec):
        raise TypeError("Invalid structuring schema descriptor")

    if "fields" in spec and not isinstance(spec.get("fields"), list | dict):
        raise TypeError(
            "Structuring descriptor 'fields' must be a list of strings or a dictionary"
        )
    if "required_fields" in spec and not isinstance(
        spec.get("required_fields"), list
    ):
        raise TypeError(
            "Structuring descriptor 'required_fields' must be a list of strings"
        )
    if "children" in spec and not isinstance(spec.get("children"), dict):
        raise TypeError(
            "Structuring descriptor 'children' must be a dictionary"
        )

    raw_fields = spec.get("fields") or []
    if isinstance(raw_fields, list):
        if not all(isinstance(name, str) for name in raw_fields):
            raise TypeError(
                "Structuring descriptor 'fields' must be a list of strings "
                "or a dictionary"
            )
        node = SchemaNode(
            NodeKind.OBJECT,
            fields={name: scalar_factory("str") for name in raw_fields},
        )
    else:
        node = schema_node_from_spec(
            raw_fields,
            scalar_factory,
            detect_descriptor=False,
            is_required=is_required,
            list_factory=list_factory,
        )

    raw_children = spec.get("children") or {}
    for raw_name, child_spec in raw_children.items():
        if isinstance(child_spec, list) and all(
            isinstance(name, str) for name in child_spec
        ):
            child = SchemaNode(
                NodeKind.OBJECT,
                fields={name: scalar_factory("str") for name in child_spec},
            )
        else:
            child = schema_node_from_spec(
                child_spec,
                scalar_factory,
                is_required=is_required,
                list_factory=list_factory,
            )
        node.fields[str(raw_name)] = (
            child
            if child.kind == NodeKind.ARRAY
            else SchemaNode(NodeKind.ARRAY, item=child)
        )

    required = spec.get("required_fields") or []
    if not all(isinstance(name, str) for name in required):
        raise TypeError(
            "Structuring descriptor 'required_fields' must be a list of strings"
        )
    node.required_fields = list(required)
    description = spec.get("description")
    node.description = str(description) if description is not None else None
    return node


def schema_node_exemplar(node: SchemaNode) -> Any:
    """Build the placeholder JSON shape consumed by hierarchy processing."""

    if node.kind == NodeKind.OBJECT:
        return {
            name: schema_node_exemplar(child)
            for name, child in node.fields.items()
        }
    if node.kind == NodeKind.ARRAY:
        return [
            schema_node_exemplar(
                node.item or SchemaNode(NodeKind.SCALAR, type_spec="str")
            )
        ]
    if isinstance(node.type_spec, FieldType) and node.type_spec.type_name == "list":
        return []
    return ""


def parse_component_spec(
    spec: object,
    default: str,
    *,
    component_name: str,
    aliases: Mapping[str, str] | None = None,
    instance_attributes: tuple[str, ...] = (),
    allow_bool: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Parse the common ``type``/``params`` component configuration shape."""

    if spec is None:
        return default, {}
    if allow_bool and isinstance(spec, bool):
        return ("multi_level" if spec else "flat"), {}
    if isinstance(spec, str):
        name, params = spec, {}
    elif isinstance(spec, Mapping):
        raw = dict(spec)
        name = raw.pop("type", raw.pop("name", default))
        configured = raw.pop("params", {})
        if not isinstance(configured, Mapping):
            raise TypeError(f"{component_name} params must be a mapping")
        params = {**configured, **raw}
    elif instance_attributes and all(
        hasattr(spec, attribute) for attribute in instance_attributes
    ):
        return "__instance__", {"instance": spec}
    else:
        suffix = ", bool" if allow_bool else ""
        raise TypeError(
            f"{component_name} must be a string, mapping{suffix}, or component instance"
        )
    normalized = str(name).lower().replace("-", "_")
    return (aliases or {}).get(normalized, normalized), params


class HierarchyNodeSpec(TypedDict, total=False):
    path: list[str]
    parent_path: list[str] | None
    parent_field_path: list[str]
    fields: list[dict[str, Any]]


class StructuringDiagnostics(TypedDict, total=False):
    active_anchor_count: int
    logical_anchor_count: int
    selected_connection_count: int
    raw_relation_connection_count: int


class StructuringAnchorEntry(TypedDict):
    anchor_index: int
    fields: list[dict[str, Any]]
    presence_is_reliable: bool


__all__ = [
    "DESCRIPTOR_KEYS",
    "FIELD_DESCRIPTOR_KEYS",
    "FieldType",
    "HierarchyNodeSpec",
    "MISSING",
    "NodeKind",
    "OutputMode",
    "Path",
    "RecursiveTypeSpec",
    "SchemaNode",
    "StructureMode",
    "StructuringAnchorEntry",
    "StructuringDiagnostics",
    "UnionFieldType",
    "is_structuring_descriptor",
    "normalize_type_name",
    "parse_component_spec",
    "schema_node_exemplar",
    "schema_node_from_descriptor",
    "schema_node_from_spec",
]
