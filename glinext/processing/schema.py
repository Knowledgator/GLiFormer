"""GLiNExTSchema — fluent builder for multi-task inference schemas."""

from __future__ import annotations

import collections.abc
import datetime
import enum
import types
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from decimal import Decimal
from typing import (
    Annotated,
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
)

from .formatting import (
    FieldType,
    StructuringOutputFormatter,
    _RecursiveTypeSpec,
    _UnionFieldType,
)

_MISSING = object()
_DESCRIPTOR_KEYS = {"fields", "children", "required_fields", "description"}
_LIST_ORIGINS = {
    list,
    tuple,
    set,
    frozenset,
    collections.abc.Sequence,
    collections.abc.MutableSequence,
}
_SCALAR_MAP: dict[type, str] = {
    str: "str",
    int: "int",
    float: "float",
    Decimal: "float",
    bool: "bool",
    datetime.date: "date",
    datetime.datetime: "datetime",
}
_SAFE_DEFAULT_FACTORIES = {list, dict, set, tuple, frozenset}


@dataclass
class _SchemaNode:
    """Recursive, dependency-free representation of a structuring schema."""

    kind: str
    type_spec: Any = None
    fields: dict[str, _SchemaNode] = dataclass_field(default_factory=dict)
    required_fields: list[str] = dataclass_field(default_factory=list)
    item: _SchemaNode | None = None
    description: str | None = None


@dataclass(frozen=True)
class _DefaultFactory:
    factory: Callable[[], Any]


def _is_pydantic_model_class(value: Any) -> bool:
    """Detect Pydantic v1/v2 models without importing Pydantic eagerly."""

    if not isinstance(value, type):
        return False
    return any(
        base.__name__ == "BaseModel" and base.__module__.split(".", 1)[0] == "pydantic"
        for base in value.__mro__
    )


def _is_pydantic_undefined(value: Any) -> bool:
    value_type = type(value)
    return value_type.__name__ in {
        "PydanticUndefinedType",
        "UndefinedType",
    } and value_type.__module__.split(".", 1)[0] in {"pydantic", "pydantic_core"}


def _pydantic_field_default(field_info: Any, required: bool) -> Any:
    """Return a safe field default, never a Pydantic undefined sentinel."""

    if required:
        return _MISSING

    factory = getattr(field_info, "default_factory", None)
    if factory is not None:
        # Avoid executing arbitrary user code merely to build an inference
        # schema.  The common immutable/container factories are safe and let
        # us preserve the historical conversion-fallback behaviour.
        if factory in _SAFE_DEFAULT_FACTORIES:
            return factory()
        return _DefaultFactory(factory)

    default = getattr(field_info, "default", _MISSING)
    if default is _MISSING or _is_pydantic_undefined(default) or default is None:
        return _MISSING
    return default


def _iter_pydantic_fields(model_cls: type) -> list[tuple[str, Any, bool, Any]]:
    """Read normalized field metadata from Pydantic v2 or ``pydantic.v1``."""

    v2_fields = getattr(model_cls, "model_fields", None)
    if isinstance(v2_fields, collections.abc.Mapping):
        result = []
        for name, field_info in v2_fields.items():
            is_required = getattr(field_info, "is_required", None)
            required = bool(is_required()) if callable(is_required) else False
            annotation = getattr(field_info, "annotation", Any)
            result.append(
                (
                    str(name),
                    annotation,
                    required,
                    _pydantic_field_default(field_info, required),
                )
            )
        return result

    v1_fields = getattr(model_cls, "__fields__", None)
    if isinstance(v1_fields, collections.abc.Mapping):
        result = []
        for name, field_info in v1_fields.items():
            required = bool(getattr(field_info, "required", False))
            annotation = getattr(field_info, "annotation", Any)
            if isinstance(annotation, str) or getattr(annotation, "__forward_arg__", None):
                annotation = getattr(field_info, "outer_type_", Any)
            result.append(
                (
                    str(name),
                    annotation,
                    required,
                    _pydantic_field_default(field_info, required),
                )
            )
        return result

    raise TypeError(f"{model_cls!r} is not a supported Pydantic model class")


def _make_scalar_type_spec(
    type_name: str | Callable,
    *,
    required: bool = False,
    default: Any = _MISSING,
) -> str | FieldType | Callable:
    if default is _MISSING and not required:
        return type_name
    kwargs: dict[str, Any] = {"type_name": type_name, "required": required}
    if isinstance(default, _DefaultFactory):
        kwargs["default_factory"] = default.factory
    elif default is not _MISSING:
        kwargs["default"] = default
    return FieldType(**kwargs)


def _scalar_type_name(annotation: Any) -> str:
    try:
        if annotation in _SCALAR_MAP:
            return _SCALAR_MAP[annotation]
    except TypeError:
        pass

    if isinstance(annotation, type):
        try:
            if issubclass(annotation, enum.Enum):
                values = [member.value for member in annotation]
                return _scalar_type_name(type(values[0])) if values else "str"
            if issubclass(annotation, bool):
                return "bool"
            if issubclass(annotation, datetime.datetime):
                return "datetime"
            if issubclass(annotation, datetime.date):
                return "date"
            if issubclass(annotation, int):
                return "int"
            if issubclass(annotation, float | Decimal):
                return "float"
            if issubclass(annotation, str):
                return "str"
        except TypeError:
            pass
    return "str"


def _is_supported_annotation_type(value: Any) -> bool:
    try:
        if value in _SCALAR_MAP or value in _LIST_ORIGINS:
            return True
    except TypeError:
        return False
    if not isinstance(value, type):
        return False
    try:
        return issubclass(
            value,
            str
            | int
            | float
            | bool
            | Decimal
            | datetime.date
            | datetime.datetime
            | enum.Enum,
        )
    except TypeError:
        return False


def _merge_union_nodes(nodes: list[_SchemaNode]) -> _SchemaNode:
    """Build the most informative schema that covers a union's variants."""

    if len(nodes) == 1:
        return nodes[0]

    object_nodes = [node for node in nodes if node.kind == "object"]
    if object_nodes:
        fields: dict[str, _SchemaNode] = {}
        for node in object_nodes:
            for name, child in node.fields.items():
                if name in fields:
                    fields[name] = _merge_union_nodes([fields[name], child])
                else:
                    fields[name] = child
        required = [
            name
            for name in object_nodes[0].required_fields
            if all(name in node.required_fields for node in object_nodes[1:])
        ]
        return _SchemaNode("object", fields=fields, required_fields=required)

    array_nodes = [node for node in nodes if node.kind == "array"]
    if array_nodes:
        items = [node.item for node in array_nodes if node.item is not None]
        return _SchemaNode(
            "array",
            item=_merge_union_nodes(items) if items else _SchemaNode("scalar", type_spec="str"),
        )

    alternatives: list[Any] = []
    for node in nodes:
        type_spec = node.type_spec
        if isinstance(type_spec, _UnionFieldType):
            alternatives.extend(type_spec.alternatives)
        else:
            alternatives.append(type_spec)
    return _SchemaNode("scalar", type_spec=_UnionFieldType(alternatives))


def _annotation_to_node(
    annotation: Any,
    *,
    required: bool = False,
    default: Any = _MISSING,
    model_stack: tuple[type, ...] = (),
) -> _SchemaNode:
    """Convert a Python/Pydantic annotation to a recursive schema node."""

    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]

    if _is_pydantic_model_class(annotation):
        return _pydantic_model_to_node(annotation, model_stack=model_stack)

    origin = get_origin(annotation)
    if origin in {Union, types.UnionType}:
        variants = [arg for arg in get_args(annotation) if arg is not type(None)]
        if not variants:
            return _SchemaNode(
                "scalar",
                type_spec=_make_scalar_type_spec("str", required=required, default=default),
            )
        return _merge_union_nodes(
            [
                _annotation_to_node(
                    variant,
                    required=required,
                    default=default,
                    model_stack=model_stack,
                )
                for variant in variants
            ]
        )

    if origin is Literal:
        values = get_args(annotation)
        annotation = type(values[0]) if values else str
        origin = get_origin(annotation)

    if annotation in _LIST_ORIGINS or origin in _LIST_ORIGINS:
        args = get_args(annotation)
        item_annotation = args[0] if args else str
        item_node = _annotation_to_node(
            item_annotation,
            model_stack=model_stack,
        )
        if item_node.kind == "scalar":
            item_spec = item_node.type_spec
            item_type = item_spec.type_name if isinstance(item_spec, FieldType) else item_spec
            if not isinstance(item_type, str):
                item_type = "str"
            kwargs: dict[str, Any] = {
                "type_name": "list",
                "list_item_type": item_type,
                "required": required,
            }
            if isinstance(default, _DefaultFactory):
                kwargs["default_factory"] = default.factory
            elif default is not _MISSING:
                kwargs["default"] = default
            return _SchemaNode("scalar", type_spec=FieldType(**kwargs))
        return _SchemaNode("array", item=item_node)

    type_name = _scalar_type_name(annotation)
    return _SchemaNode(
        "scalar",
        type_spec=_make_scalar_type_spec(type_name, required=required, default=default),
    )


def _pydantic_model_to_node(
    model_cls: type,
    *,
    model_stack: tuple[type, ...] = (),
) -> _SchemaNode:
    if model_cls in model_stack:
        chain = " -> ".join(cls.__name__ for cls in (*model_stack, model_cls))
        raise ValueError(
            "Cyclic Pydantic structuring schemas cannot be represented as a "
            f"finite extraction template: {chain}"
        )

    stack = (*model_stack, model_cls)
    fields = _iter_pydantic_fields(model_cls)
    is_v2_root = bool(getattr(model_cls, "__pydantic_root_model__", False))
    is_v1_root = len(fields) == 1 and fields[0][0] == "__root__"
    if is_v2_root or is_v1_root:
        _, annotation, required, default = fields[0]
        return _annotation_to_node(
            annotation,
            required=required,
            default=default,
            model_stack=stack,
        )

    node = _SchemaNode("object")
    for name, annotation, required, default in fields:
        node.fields[name] = _annotation_to_node(
            annotation,
            required=required,
            default=default,
            model_stack=stack,
        )
        if required:
            node.required_fields.append(name)
    return node


def _is_descriptor(spec: dict[str, Any]) -> bool:
    fields = spec.get("fields")
    required = spec.get("required_fields", [])
    children = spec.get("children", {})
    return (
        set(spec).issubset(_DESCRIPTOR_KEYS)
        and isinstance(fields, list | dict)
        and isinstance(required, list)
        and isinstance(children, dict)
    )


def _type_spec_required(type_spec: Any) -> bool:
    if isinstance(type_spec, FieldType):
        return type_spec.required
    if isinstance(type_spec, _UnionFieldType):
        return any(_type_spec_required(item) for item in type_spec.alternatives)
    return False


def _node_from_mapping(spec: dict[str, Any]) -> _SchemaNode:
    node = _SchemaNode("object")
    for raw_name, value_spec in spec.items():
        name = str(raw_name)
        child = _node_from_value_spec(value_spec)
        node.fields[name] = child
        if child.kind == "scalar" and _type_spec_required(child.type_spec):
            node.required_fields.append(name)
    return node


def _node_from_descriptor(spec: dict[str, Any]) -> _SchemaNode:
    raw_fields = spec.get("fields") or []
    if isinstance(raw_fields, list) and all(
        isinstance(field_name, str) for field_name in raw_fields
    ):
        node = _SchemaNode(
            "object",
            fields={
                field_name: _SchemaNode("scalar", type_spec="str") for field_name in raw_fields
            },
        )
    elif isinstance(raw_fields, dict):
        node = _node_from_mapping(raw_fields)
    else:
        raise TypeError("Structuring descriptor 'fields' must be a list of strings or a dictionary")

    raw_children = spec.get("children") or {}
    if not isinstance(raw_children, dict):
        raise TypeError("Structuring descriptor 'children' must be a dictionary")
    for raw_name, child_spec in raw_children.items():
        if isinstance(child_spec, list) and all(
            isinstance(field_name, str) for field_name in child_spec
        ):
            child_node = _SchemaNode(
                "object",
                fields={
                    field_name: _SchemaNode("scalar", type_spec="str") for field_name in child_spec
                },
            )
        else:
            child_node = _node_from_value_spec(child_spec)
        if child_node.kind == "array":
            node.fields[str(raw_name)] = child_node
        else:
            node.fields[str(raw_name)] = _SchemaNode("array", item=child_node)

    raw_required = spec.get("required_fields") or []
    if not isinstance(raw_required, list) or not all(
        isinstance(field_name, str) for field_name in raw_required
    ):
        raise TypeError("Structuring descriptor 'required_fields' must be a list of strings")
    node.required_fields = list(raw_required)
    node.description = spec.get("description")
    return node


def _node_from_value_spec(spec: Any) -> _SchemaNode:
    if _is_pydantic_model_class(spec):
        return _pydantic_model_to_node(spec)
    if isinstance(spec, FieldType):
        return _SchemaNode("scalar", type_spec=spec)
    if isinstance(spec, dict):
        return _node_from_descriptor(spec) if _is_descriptor(spec) else _node_from_mapping(spec)

    origin = get_origin(spec)
    if origin is not None or _is_supported_annotation_type(spec):
        return _annotation_to_node(spec)

    if isinstance(spec, list):
        if not spec:
            return _SchemaNode("scalar", type_spec=FieldType("list", list_item_type="str"))
        item_node = _node_from_value_spec(spec[0])
        if item_node.kind == "scalar":
            item_spec = item_node.type_spec
            item_type = item_spec.type_name if isinstance(item_spec, FieldType) else item_spec
            if not isinstance(item_type, str) or item_type == "":
                item_type = "str"
            return _SchemaNode(
                "scalar",
                type_spec=FieldType("list", list_item_type=item_type),
            )
        return _SchemaNode("array", item=item_node)

    if isinstance(spec, str):
        return _SchemaNode("scalar", type_spec=spec or "str")
    if callable(spec):
        return _SchemaNode("scalar", type_spec=spec)
    if spec is None:
        return _SchemaNode("scalar", type_spec="str")
    return _annotation_to_node(type(spec))


def _node_to_format_spec(node: _SchemaNode) -> Any:
    if node.kind == "scalar":
        return node.type_spec
    if node.kind == "array":
        return [_node_to_format_spec(node.item or _SchemaNode("scalar", type_spec="str"))]
    return {name: _node_to_format_spec(child) for name, child in node.fields.items()}


def _pydantic_to_field_types(model_cls: type) -> Any:
    """Convert a Pydantic v2 or ``pydantic.v1`` model to formatter types.

    Pydantic is detected lazily, so it remains an optional dependency.  Nested
    models use dictionaries, while ``list[BaseModel]`` fields use one-item
    lists.  Flat models retain the historical ``{field_name: type_spec}``
    result.
    """

    return _node_to_format_spec(_pydantic_model_to_node(model_cls))


def _field_path_exists(node: _SchemaNode, path: str) -> bool:
    if node.kind != "object":
        return False
    if path in node.fields:
        return True

    current = node
    segments = str(path).split(".")
    for index, segment in enumerate(segments):
        if current.kind != "object" or segment not in current.fields:
            return False
        current = current.fields[segment]
        if index == len(segments) - 1:
            return True
        if current.kind == "array":
            current = current.item or _SchemaNode("scalar", type_spec="str")
    return False


def _node_exemplar(node: _SchemaNode) -> Any:
    if node.kind == "object":
        return {name: _node_exemplar(child) for name, child in node.fields.items()}
    if node.kind == "array":
        return [_node_exemplar(node.item or _SchemaNode("scalar", type_spec="str"))]
    if isinstance(node.type_spec, FieldType) and node.type_spec.type_name == "list":
        return []
    return ""


def _descriptor_required_fields(node: _SchemaNode) -> list[str]:
    required = list(dict.fromkeys(node.required_fields))
    required_names = set(required)
    for name, child in node.fields.items():
        # Inline objects have no separate descriptor, so required descendants
        # are represented as dotted paths on their nearest descriptor.
        if child.kind != "object" or name not in required_names:
            continue
        for child_path in _descriptor_required_fields(child):
            dotted = f"{name}.{child_path}"
            if dotted not in required:
                required.append(dotted)
    return required


def _node_to_descriptor(
    node: _SchemaNode,
    *,
    description: str | None = None,
) -> dict[str, Any]:
    if node.kind != "object":
        raise TypeError("A named structuring schema must describe an object")

    children: dict[str, dict[str, Any]] = {}
    has_inline_objects = False
    for name, child in node.fields.items():
        if child.kind == "array" and child.item is not None and child.item.kind == "object":
            children[name] = _node_to_descriptor(child.item)
        elif child.kind in {"object", "array"}:
            has_inline_objects = True

    required = _descriptor_required_fields(node)
    if children or has_inline_objects:
        resolved_description = description if description is not None else node.description
        descriptor: dict[str, Any] = {
            "fields": {
                name: _node_exemplar(child)
                for name, child in node.fields.items()
                if name not in children
            },
            "children": children,
            "required_fields": required,
        }
        if resolved_description is not None:
            descriptor["description"] = resolved_description
        return descriptor

    # Preserve the exact flat descriptor accepted by earlier releases.
    return {
        "fields": list(node.fields),
        "required_fields": required,
    }


class GLiNExTSchema:
    """Builder for multi-task inference schemas.

    Usage::

        schema = GLiNExTSchema()
        schema.add_entities(["person", "org"], parent="general")
        schema.add_classes(["positive", "negative"])
        schema.add_relations(["works_at", "born_in"])
        schema.add_structure("person", ["name", "age", "occupation"])

        results = model.inference_from_schema(texts, schema)
    """

    def __init__(self):
        self._entity_groups: list[dict] = []
        self._class_groups: list[dict] = []
        self._relation_groups: list[dict] = []
        self._joint_groups: list[dict] = []
        self._structure_schemas: dict[str, dict] = {}

    # ── Entity groups (NER) ────────────────────────────────────────────

    def add_entities(
        self,
        entities: list[str],
        parent: str | None = None,
        description: str | None = None,
    ) -> GLiNExTSchema:
        """Add an NER entity group.

        Args:
            entities: Entity type labels (e.g. ["person", "org"]).
            parent: Optional parent group name.
            description: Optional description of the group.
        """
        self._entity_groups.append(
            {
                "name": parent,
                "description": description,
                "entities": list(entities),
            }
        )
        return self

    # ── Classification groups ──────────────────────────────────────────

    def add_classes(
        self,
        classes: list[str],
        parent: str | None = None,
        description: str | None = None,
    ) -> GLiNExTSchema:
        """Add a classification group.

        Args:
            classes: Class labels (e.g. ["positive", "negative"]).
            parent: Optional parent group name.
            description: Optional description of the group.
        """
        self._class_groups.append(
            {
                "name": parent,
                "description": description,
                "classes": list(classes),
            }
        )
        return self

    # ── Open relation extraction groups ────────────────────────────────

    def add_relations(
        self,
        relations: list[str],
        parent: str | None = None,
    ) -> GLiNExTSchema:
        """Add an open relation extraction group.

        Args:
            relations: Relation type labels (e.g. ["works_at", "born_in"]).
            parent: Optional parent group name.
        """
        self._relation_groups.append(
            {
                "name": parent,
                "relations": list(relations),
            }
        )
        return self

    # ── Joint NER + relation extraction groups ─────────────────────────

    def add_joint_entities_relations(
        self,
        entities: list[str],
        relations: list[str],
        parent: str | None = None,
        description: str | None = None,
    ) -> GLiNExTSchema:
        """Add a joint NER + relation extraction group.

        Args:
            entities: Entity type labels.
            relations: Relation type labels.
            parent: Optional parent group name.
            description: Optional description.
        """
        self._joint_groups.append(
            {
                "name": parent,
                "description": description,
                "entities": list(entities),
                "relations": list(relations),
            }
        )
        return self

    # ── Structuring schemas ────────────────────────────────────────────

    def add_structure(
        self,
        schema_name: str,
        fields: Any,
        description: str | None = None,
        required_fields: list[str] | None = None,
    ) -> GLiNExTSchema:
        """Add a structuring schema for JSON extraction.

        Args:
            schema_name: Name of the schema (e.g. "person").
            fields: One of:

                **Untyped list** — all values returned as str::

                    schema.add_structure("person", ["name", "age"])

                **Typed dict** — values converted to declared types::

                    schema.add_structure("person", {
                        "name": "str",
                        "age": "int",
                        "is_active": "bool",
                        "salary": "float",
                        "birth_date": "date",
                        "skills": FieldType("list", list_item_type="str"),
                    })

                    # Nested objects and lists of objects use the same shape
                    # as multi-level structuring input.
                    schema.add_structure("order", {
                        "id": "int",
                        "seller": {"name": "str"},
                        "items": [{"sku": "str", "quantity": "int"}],
                    })

                **Pydantic BaseModel** — fields and types extracted automatically::

                    class Person(BaseModel):
                        name: str
                        age: int
                        is_active: bool = False
                        skills: List[str] = []

                    schema.add_structure("person", Person)

                Both Pydantic v2 and ``pydantic.v1`` compatibility models are
                supported, including nested models and ``list[BaseModel]``.
                Pydantic stays optional and is imported only by user code.

                Supported type names: str, int, float, bool, list, date,
                datetime. For advanced control, use FieldType or a callable.
            description: Optional description of the schema.
            required_fields: Field names that must appear in every emitted
                instance. Acts as a post-processing filter: any instance
                where a required field is ``None`` is dropped from the
                output. Decoding itself is unaffected. If ``None``, the
                required set is auto-derived from ``FieldType.required`` or
                from Pydantic fields without a default.
        """
        if _is_pydantic_model_class(fields):
            node = _pydantic_model_to_node(fields)
            has_types = True
        elif isinstance(fields, dict):
            node = (
                _node_from_descriptor(fields)
                if _is_descriptor(fields)
                else _node_from_mapping(fields)
            )
            has_types = True
        elif isinstance(fields, list) and all(isinstance(field_name, str) for field_name in fields):
            # Historical untyped form.  Keep build_output_formatter() returning
            # None for it, even though the recursive node knows fields are str.
            node = _SchemaNode(
                "object",
                fields={
                    field_name: _SchemaNode("scalar", type_spec="str") for field_name in fields
                },
            )
            has_types = False
        elif isinstance(fields, list):
            if len(fields) != 1:
                raise TypeError("A nested structuring schema list must contain one object exemplar")
            item_node = _node_from_value_spec(fields[0])
            if item_node.kind != "object":
                raise TypeError("A nested structuring schema list must contain an object exemplar")
            node = _SchemaNode("array", item=item_node)
            has_types = True
        elif get_origin(fields) is not None:
            node = _annotation_to_node(fields)
            has_types = True
        else:
            raise TypeError(
                "Structuring fields must be a field-name list, typed mapping, "
                "nested exemplar, or Pydantic BaseModel class"
            )

        if node.kind == "scalar":
            raise TypeError("A structuring schema root must be an object or list of objects")

        # Named structuring groups are already lists of instances.  A root
        # model declared as list[Model] therefore describes the named instance
        # type, whereas "$root" preserves the raw list output shape.
        inference_node = node
        if schema_name != "$root" and node.kind == "array":
            if node.item is None or node.item.kind != "object":
                raise TypeError("A named structuring schema must contain object instances")
            inference_node = node.item

        requirement_node = (
            node.item
            if node.kind == "array" and node.item is not None and node.item.kind == "object"
            else inference_node
        )

        if required_fields is not None:
            # Explicit requirements override only this object level.  Nested
            # child descriptors retain their own Pydantic requirements.
            requirement_node.required_fields = [
                field_name
                for field_name in required_fields
                if _field_path_exists(requirement_node, field_name)
            ]

        resolved_description = description if description is not None else node.description
        formatter_node = node if schema_name == "$root" else inference_node
        field_types = _node_to_format_spec(formatter_node) if has_types else None

        self._structure_schemas[schema_name] = {
            "fields": list(inference_node.fields),
            "field_types": field_types,
            "description": resolved_description,
            "required_fields": list(requirement_node.required_fields),
            "node": node,
            "inference_node": inference_node,
        }
        return self

    # ── Conversion to inference kwargs ─────────────────────────────────

    def to_inference_kwargs(self) -> dict:
        """Convert accumulated schema into kwargs for GLiNExT.inference().

        Returns:
            Dict with keys: entities, classes, relations, joint_relations, structures.
            Only non-empty keys are included.
        """
        kwargs = {}

        if self._entity_groups:
            if len(self._entity_groups) == 1 and self._entity_groups[0]["name"] is None:
                kwargs["entities"] = self._entity_groups[0]["entities"]
            else:
                entities = {}
                for g in self._entity_groups:
                    key = g["name"] or f"group_{len(entities)}"
                    entities[key] = g["entities"]
                kwargs["entities"] = entities

        if self._class_groups:
            if len(self._class_groups) == 1 and self._class_groups[0]["name"] is None:
                kwargs["classes"] = self._class_groups[0]["classes"]
            else:
                classes = {}
                for g in self._class_groups:
                    key = g["name"] or f"group_{len(classes)}"
                    classes[key] = g["classes"]
                kwargs["classes"] = classes

        if self._relation_groups:
            if len(self._relation_groups) == 1 and self._relation_groups[0]["name"] is None:
                kwargs["relations"] = self._relation_groups[0]["relations"]
            else:
                relations = {}
                for g in self._relation_groups:
                    key = g["name"] or f"group_{len(relations)}"
                    relations[key] = g["relations"]
                kwargs["relations"] = relations

        if self._joint_groups:
            joint = {}
            for g in self._joint_groups:
                key = g["name"] or f"group_{len(joint)}"
                joint[key] = {
                    "entities": g["entities"],
                    "relations": g["relations"],
                }
            kwargs["joint_relations"] = joint

        if self._structure_schemas:
            structures = {}
            for name, schema in self._structure_schemas.items():
                node = schema["node"]
                if name == "$root" and node.kind == "array":
                    item = node.item or _SchemaNode("object")
                    structures[name] = [_node_exemplar(item)]
                    continue
                structures[name] = _node_to_descriptor(
                    schema["inference_node"],
                    description=schema.get("description"),
                )
            kwargs["structures"] = structures

        return kwargs

    def build_output_formatter(self) -> StructuringOutputFormatter | None:
        """Build a StructuringOutputFormatter from typed structure schemas.

        Returns:
            A formatter if any schema has field types defined, else None.
        """
        schema_types = {}
        for name, schema in self._structure_schemas.items():
            ft = schema.get("field_types")
            if ft is not None:
                schema_types[name] = _RecursiveTypeSpec(ft)
        if not schema_types:
            return None
        return StructuringOutputFormatter(schema_types)

    def __repr__(self) -> str:
        parts = []
        if self._entity_groups:
            parts.append(f"entities={len(self._entity_groups)} groups")
        if self._class_groups:
            parts.append(f"classes={len(self._class_groups)} groups")
        if self._relation_groups:
            parts.append(f"relations={len(self._relation_groups)} groups")
        if self._joint_groups:
            parts.append(f"joint={len(self._joint_groups)} groups")
        if self._structure_schemas:
            parts.append(f"structures={len(self._structure_schemas)} schemas")
        return f"GLiNExTSchema({', '.join(parts)})"
