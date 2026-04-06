"""GLiNExTSchema — fluent builder for multi-task inference schemas."""

from __future__ import annotations

import datetime
import sys
import types
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_args, get_origin

from .formatting import FieldType, StructuringOutputFormatter


def _pydantic_to_field_types(
    model_cls: Type,
) -> Dict[str, Union[str, FieldType]]:
    """Convert a pydantic BaseModel class to a ``{field_name: type_spec}`` dict.

    Supports common scalar types (str, int, float, bool, date, datetime) and
    ``list``/``List[T]`` with typed items.  ``Optional[T]`` is unwrapped to
    ``T`` with ``default=None``.
    """
    _SCALAR_MAP: Dict[type, str] = {
        str: "str",
        int: "int",
        float: "float",
        bool: "bool",
        datetime.date: "date",
        datetime.datetime: "datetime",
    }

    field_types: Dict[str, Union[str, FieldType]] = {}

    for name, field_info in model_cls.model_fields.items():
        annotation = field_info.annotation
        default = field_info.default

        # Unwrap Optional[T] (Union[T, None] or T | None)
        origin = get_origin(annotation)
        if origin is Union or (sys.version_info >= (3, 10) and isinstance(annotation, types.UnionType)):
            args = [a for a in get_args(annotation) if a is not type(None)]
            if len(args) == 1:
                annotation = args[0]
                origin = get_origin(annotation)

        # List[T] or list[T]
        if origin is list:
            item_args = get_args(annotation)
            item_type = "str"
            if item_args:
                item_type = _SCALAR_MAP.get(item_args[0], "str")
            field_types[name] = FieldType(
                "list",
                list_item_type=item_type,
                default=[] if default is None else default,
            )
            continue

        # Scalar types
        if annotation in _SCALAR_MAP:
            ft_kwargs: Dict[str, Any] = {"type_name": _SCALAR_MAP[annotation]}
            if default is not None:
                ft_kwargs["default"] = default
            # Use plain string for simple cases (no custom default)
            if "default" not in ft_kwargs:
                field_types[name] = _SCALAR_MAP[annotation]
            else:
                field_types[name] = FieldType(**ft_kwargs)
            continue

        # Fallback — treat as str
        field_types[name] = "str"

    return field_types


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
        self._entity_groups: List[dict] = []
        self._class_groups: List[dict] = []
        self._relation_groups: List[dict] = []
        self._joint_groups: List[dict] = []
        self._structure_schemas: Dict[str, dict] = {}

    # ── Entity groups (NER) ────────────────────────────────────────────

    def add_entities(
        self,
        entities: List[str],
        parent: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "GLiNExTSchema":
        """Add an NER entity group.

        Args:
            entities: Entity type labels (e.g. ["person", "org"]).
            parent: Optional parent group name.
            description: Optional description of the group.
        """
        self._entity_groups.append({
            "name": parent,
            "description": description,
            "entities": list(entities),
        })
        return self

    # ── Classification groups ──────────────────────────────────────────

    def add_classes(
        self,
        classes: List[str],
        parent: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "GLiNExTSchema":
        """Add a classification group.

        Args:
            classes: Class labels (e.g. ["positive", "negative"]).
            parent: Optional parent group name.
            description: Optional description of the group.
        """
        self._class_groups.append({
            "name": parent,
            "description": description,
            "classes": list(classes),
        })
        return self

    # ── Open relation extraction groups ────────────────────────────────

    def add_relations(
        self,
        relations: List[str],
        parent: Optional[str] = None,
    ) -> "GLiNExTSchema":
        """Add an open relation extraction group.

        Args:
            relations: Relation type labels (e.g. ["works_at", "born_in"]).
            parent: Optional parent group name.
        """
        self._relation_groups.append({
            "name": parent,
            "relations": list(relations),
        })
        return self

    # ── Joint NER + relation extraction groups ─────────────────────────

    def add_joint_entities_relations(
        self,
        entities: List[str],
        relations: List[str],
        parent: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "GLiNExTSchema":
        """Add a joint NER + relation extraction group.

        Args:
            entities: Entity type labels.
            relations: Relation type labels.
            parent: Optional parent group name.
            description: Optional description.
        """
        self._joint_groups.append({
            "name": parent,
            "description": description,
            "entities": list(entities),
            "relations": list(relations),
        })
        return self

    # ── Structuring schemas ────────────────────────────────────────────

    def add_structure(
        self,
        schema_name: str,
        fields: Union[List[str], Dict[str, Union[str, FieldType, Callable]], Type],
        description: Optional[str] = None,
    ) -> "GLiNExTSchema":
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

                **Pydantic BaseModel** — fields and types extracted automatically::

                    class Person(BaseModel):
                        name: str
                        age: int
                        is_active: bool = False
                        skills: List[str] = []

                    schema.add_structure("person", Person)

                Supported type names: str, int, float, bool, list, date,
                datetime. For advanced control, use FieldType or a callable.
            description: Optional description of the schema.
        """
        # Pydantic BaseModel class
        if isinstance(fields, type) and hasattr(fields, "model_fields"):
            field_types = _pydantic_to_field_types(fields)
            field_names = list(field_types.keys())
        elif isinstance(fields, dict):
            field_names = list(fields.keys())
            field_types = dict(fields)
        else:
            field_names = list(fields)
            field_types = None

        self._structure_schemas[schema_name] = {
            "fields": field_names,
            "field_types": field_types,
            "description": description,
        }
        return self

    # ── Conversion to inference kwargs ─────────────────────────────────

    def to_inference_kwargs(self) -> Dict:
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
            kwargs["structures"] = {
                name: schema["fields"]
                for name, schema in self._structure_schemas.items()
            }

        return kwargs

    def build_output_formatter(self) -> Optional[StructuringOutputFormatter]:
        """Build a StructuringOutputFormatter from typed structure schemas.

        Returns:
            A formatter if any schema has field types defined, else None.
        """
        schema_types = {}
        for name, schema in self._structure_schemas.items():
            ft = schema.get("field_types")
            if ft is not None:
                schema_types[name] = ft
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
