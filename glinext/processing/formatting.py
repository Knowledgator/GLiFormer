"""Structuring output formatter — converts raw text spans to typed values.

Since structuring schemas can specify fields of various types (str, int, float,
bool, list, date, etc.), the raw text extracted by the decoder needs nuanced
conversion into proper Python objects.
"""

import re
from collections.abc import Callable
from datetime import date, datetime
from typing import Any, Optional


class FieldType:
    """Descriptor for a structured field's expected type and conversion rules.

    Args:
        type_name: One of the supported type names (str, int, float, bool,
            list, date, datetime) or a callable for custom conversion.
        default: Default value when conversion fails. If None, the raw text
            is kept as-is on failure.
        list_separator: Separator pattern for list-typed fields (default: comma).
        list_item_type: Type name for individual items within a list field.
        date_formats: Date/datetime format strings to try in order.
        required: Whether the field must always be present in the output.
            Acts as a post-processing filter: any instance where this field
            comes back as ``None`` is dropped from the output. Decoding is
            unaffected.
        default_factory: Zero-argument factory evaluated lazily when
            conversion fails. Used by Pydantic default factories.
    """

    def __init__(
        self,
        type_name: str | Callable = "str",
        default: Any = None,
        list_separator: str = r"\s*,\s*",
        list_item_type: str = "str",
        date_formats: list[str] | None = None,
        required: bool = False,
        default_factory: Callable[[], Any] | None = None,
    ):
        self.type_name = type_name
        self.default = default
        self.default_factory = default_factory
        self.list_separator = list_separator
        self.list_item_type = list_item_type
        self.required = required
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


class _UnionFieldType:
    """Internal descriptor that tries scalar union variants in order."""

    def __init__(self, alternatives: list[FieldType | str | Callable]):
        self.alternatives = alternatives


class _FormatNode:
    """Internal recursive representation of a structuring type schema."""

    def __init__(
        self,
        kind: str,
        *,
        field_type: FieldType | _UnionFieldType | None = None,
        fields: dict[str, "_FormatNode"] | None = None,
        item: Optional["_FormatNode"] = None,
    ):
        self.kind = kind
        self.field_type = field_type
        self.fields = fields or {}
        self.item = item


class _RecursiveTypeSpec:
    """Marks an internal type tree so descriptor-like field names stay literal."""

    def __init__(self, value: Any):
        self.value = value


# ── Built-in converters ──────────────────────────────────────────────────

_TRUE_STRINGS = frozenset({"true", "yes", "1", "on", "y", "t"})
_FALSE_STRINGS = frozenset({"false", "no", "0", "off", "n", "f"})


def _convert_bool(text: str) -> bool | None:
    """Convert text to bool, return None if ambiguous."""
    t = text.strip().lower()
    if t in _TRUE_STRINGS:
        return True
    if t in _FALSE_STRINGS:
        return False
    return None


def _convert_int(text: str) -> int | None:
    """Convert text to int, handling common formatting (commas, spaces)."""
    cleaned = text.strip().replace(",", "").replace(" ", "")
    try:
        return int(cleaned)
    except ValueError:
        # Try float→int for values like "42.0"
        try:
            f = float(cleaned)
            if f == int(f):
                return int(f)
        except ValueError:
            pass
    return None


def _convert_float(text: str) -> float | None:
    """Convert text to float, handling common formatting."""
    cleaned = text.strip().replace(",", "").replace(" ", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _convert_date(text: str, formats: list[str]) -> date | None:
    """Try multiple date formats, return first successful parse."""
    t = text.strip()
    for fmt in formats:
        try:
            return datetime.strptime(t, fmt).date()
        except ValueError:
            continue
    return None


def _convert_datetime(text: str, formats: list[str]) -> datetime | None:
    """Try multiple datetime formats, return first successful parse."""
    t = text.strip()
    for fmt in formats:
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            continue
    return None


class StructuringOutputFormatter:
    """Converts raw structuring text outputs into typed Python values.

    Usage::

        formatter = StructuringOutputFormatter({
            "person": {
                "name": "str",
                "age": "int",
                "is_active": "bool",
                "skills": FieldType("list", list_item_type="str"),
                "birth_date": "date",
                "salary": "float",
            }
        })

        # raw_output is the dict returned by GLiNExT structuring inference
        typed_output = formatter.format(raw_output)

    Supported type names: str, int, float, bool, list, date, datetime.
    For advanced control, pass a FieldType instance or a callable.
    """

    def __init__(
        self,
        schema_types: dict[str, Any],
        strict: bool = False,
    ):
        """
        Args:
            schema_types: ``{schema_name: {field_name: type_spec}}``.
                Type spec can be a string name, a FieldType, or a callable.
                Nested dictionaries and one-item lists describe nested objects
                and lists of objects, respectively.  A ``"$root"`` schema can
                be used to format an unwrapped root object or list.
            strict: If True, raise ValueError on conversion failure instead
                of falling back to raw text.
        """
        self.strict = strict
        self._schema_nodes: dict[str, _FormatNode] = {}
        self._field_types: dict[str, dict[str, FieldType | _UnionFieldType]] = {}
        for schema_name, schema_spec in schema_types.items():
            node = self._normalize_node(schema_spec)
            self._schema_nodes[schema_name] = node

            # Keep the historical flat map available for callers which use
            # this implementation detail.  Nested nodes live in
            # ``_schema_nodes`` and are intentionally omitted here.
            self._field_types[schema_name] = (
                {
                    field_name: field_node.field_type
                    for field_name, field_node in node.fields.items()
                    if field_node.kind == "scalar" and field_node.field_type is not None
                }
                if node.kind == "object"
                else {}
            )

    @staticmethod
    def _normalize_type(
        type_spec: str | FieldType | _UnionFieldType | Callable,
    ) -> FieldType | _UnionFieldType:
        """Normalize a type specification into a FieldType."""
        if isinstance(type_spec, FieldType | _UnionFieldType):
            return type_spec
        if callable(type_spec) and not isinstance(type_spec, str):
            return FieldType(type_name=type_spec)
        if isinstance(type_spec, str):
            return FieldType(type_name=type_spec)
        raise TypeError(f"Unsupported type spec: {type_spec!r}")

    @staticmethod
    def _is_descriptor(spec: dict[str, Any]) -> bool:
        """Return whether *spec* is a processor-style schema descriptor."""

        descriptor_keys = {"fields", "children", "required_fields", "description"}
        fields = spec.get("fields")
        required = spec.get("required_fields", [])
        children = spec.get("children", {})
        return (
            set(spec).issubset(descriptor_keys)
            and isinstance(fields, list | dict)
            and isinstance(required, list)
            and isinstance(children, dict)
        )

    @classmethod
    def _normalize_node(cls, spec: Any) -> _FormatNode:
        """Normalize a legacy or recursive schema specification."""

        if isinstance(spec, _FormatNode):
            return spec
        if isinstance(spec, _RecursiveTypeSpec):
            return cls._normalize_type_tree(spec.value)

        if isinstance(spec, dict):
            if cls._is_descriptor(spec):
                fields_spec = spec.get("fields") or []
                if isinstance(fields_spec, list):
                    fields = {
                        str(field_name): _FormatNode("scalar", field_type=FieldType("str"))
                        for field_name in fields_spec
                        if isinstance(field_name, str)
                    }
                else:
                    fields = {
                        str(field_name): cls._normalize_node(type_spec)
                        for field_name, type_spec in fields_spec.items()
                    }

                children = spec.get("children") or {}
                if isinstance(children, dict):
                    for child_name, child_spec in children.items():
                        if isinstance(child_spec, list) and all(
                            isinstance(field_name, str) for field_name in child_spec
                        ):
                            child_node = _FormatNode(
                                "object",
                                fields={
                                    field_name: _FormatNode("scalar", field_type=FieldType("str"))
                                    for field_name in child_spec
                                },
                            )
                        else:
                            child_node = cls._normalize_node(child_spec)
                        if child_node.kind != "array":
                            child_node = _FormatNode("array", item=child_node)
                        fields[str(child_name)] = child_node
                return _FormatNode("object", fields=fields)

            return _FormatNode(
                "object",
                fields={
                    str(field_name): cls._normalize_node(type_spec)
                    for field_name, type_spec in spec.items()
                },
            )

        if isinstance(spec, list):
            item_spec = spec[0] if spec else "str"
            return _FormatNode("array", item=cls._normalize_node(item_spec))

        # Empty-string values are the canonical inference-schema placeholder.
        # They carry string semantics rather than naming an unsupported type.
        if spec == "":
            spec = "str"
        return _FormatNode("scalar", field_type=cls._normalize_type(spec))

    @classmethod
    def _normalize_type_tree(cls, spec: Any) -> _FormatNode:
        """Normalize an internal recursive tree without descriptor ambiguity."""

        if isinstance(spec, dict):
            return _FormatNode(
                "object",
                fields={
                    str(field_name): cls._normalize_type_tree(type_spec)
                    for field_name, type_spec in spec.items()
                },
            )
        if isinstance(spec, list):
            item_spec = spec[0] if spec else "str"
            return _FormatNode("array", item=cls._normalize_type_tree(item_spec))
        if spec == "":
            spec = "str"
        return _FormatNode("scalar", field_type=cls._normalize_type(spec))

    def format(
        self,
        raw_output: Any,
    ) -> Any:
        """Format a single text's structuring output.

        Args:
            raw_output: ``{schema_name: [instance_dict, ...]}``, as returned
                by ``GLiNExT.structure()`` or the ``"structuring"`` key of
                ``GLiNExT.inference()``.

        Returns:
            Same structure with field values converted to their declared types.
        """
        root_node = self._schema_nodes.get("$root")
        if root_node is not None:
            # Hierarchy formatting returns root objects/lists unwrapped.
            # Also accept an explicit wrapper for callers working directly
            # with inference schemas.
            if isinstance(raw_output, dict) and set(raw_output) == {"$root"}:
                return {"$root": self._format_node(raw_output["$root"], root_node, "$root", "")}
            return self._format_node(raw_output, root_node, "$root", "")

        if not isinstance(raw_output, dict):
            return raw_output

        result = {}
        for schema_name, instances in raw_output.items():
            node = self._schema_nodes.get(schema_name)
            if node is None:
                # No type info for this schema — pass through unchanged
                result[schema_name] = instances
                continue
            if node.kind == "object" and isinstance(instances, list):
                result[schema_name] = [
                    self._format_node(instance, node, schema_name, "") for instance in instances
                ]
            else:
                result[schema_name] = self._format_node(instances, node, schema_name, "")
        return result

    def format_batch(
        self,
        batch_output: list[Any],
    ) -> list[Any]:
        """Format a batch of structuring outputs (one per text)."""
        return [self.format(item) for item in batch_output]

    def _format_node(
        self,
        value: Any,
        node: _FormatNode,
        schema_name: str,
        field_path: str,
    ) -> Any:
        """Recursively format *value* according to *node*."""

        if value is None:
            return None

        if node.kind == "object":
            if isinstance(value, dict):
                return self._format_object(value, node, schema_name, field_path)
            # Be permissive when a decoder returns multiple candidates for an
            # inline object.  Shape is preserved while each object is typed.
            if isinstance(value, list):
                return [
                    self._format_node(item, node, schema_name, field_path)
                    if isinstance(item, dict)
                    else item
                    for item in value
                ]
            return value

        if node.kind == "array":
            item_node = node.item or _FormatNode("scalar", field_type=FieldType("str"))
            if isinstance(value, list):
                return [
                    self._format_node(
                        item,
                        item_node,
                        schema_name,
                        f"{field_path}[]" if field_path else "[]",
                    )
                    for item in value
                ]
            if isinstance(value, dict):
                return self._format_node(value, item_node, schema_name, field_path)
            return value

        ft = node.field_type or FieldType("str")
        field_name = field_path or "$root"
        return self._convert_value(value, ft, schema_name, field_name)

    def _format_object(
        self,
        instance: dict[str, Any],
        node: _FormatNode,
        schema_name: str,
        parent_path: str,
    ) -> dict[str, Any]:
        """Recursively convert fields in a single object."""

        formatted = {}
        for field_name, value in instance.items():
            field_node = node.fields.get(field_name)
            if field_node is None:
                formatted[field_name] = value
                continue
            path = f"{parent_path}.{field_name}" if parent_path else field_name
            formatted[field_name] = self._format_node(value, field_node, schema_name, path)
        return formatted

    def _format_instance(
        self,
        instance: dict[str, Any],
        field_types: dict[str, FieldType],
        schema_name: str,
    ) -> dict[str, Any]:
        """Convert all field values in a single instance dict."""
        node = _FormatNode(
            "object",
            fields={
                field_name: _FormatNode("scalar", field_type=field_type)
                for field_name, field_type in field_types.items()
            },
        )
        return self._format_object(instance, node, schema_name, "")

    def _convert_value(
        self,
        value: Any,
        ft: FieldType | _UnionFieldType,
        schema_name: str,
        field_name: str,
    ) -> Any:
        """Convert a single value according to its FieldType."""
        if isinstance(ft, _UnionFieldType):
            return self._convert_union_value(value, ft, schema_name, field_name)

        if ft.type_name == "list":
            return self._convert_list_value(value, ft)

        # GLiNER2 treats non-list fields as scalar: when multiple spans were
        # decoded for a scalar field, keep the highest-ranked/first value.
        if isinstance(value, list):
            if not value:
                return self._fallback_value(ft)
            return self._convert_single(value[0], ft, schema_name, field_name)

        return self._convert_single(value, ft, schema_name, field_name)

    def _convert_union_value(
        self,
        value: Any,
        union_type: _UnionFieldType,
        schema_name: str,
        field_name: str,
    ) -> Any:
        """Try each scalar union alternative without applying defaults early."""

        if isinstance(value, list):
            if not value:
                first = self._normalize_type(union_type.alternatives[0])
                return self._fallback_value(first) if isinstance(first, FieldType) else None
            value = value[0]
        raw_text = self._extract_text(value)
        alternatives = [self._normalize_type(item) for item in union_type.alternatives]
        for alternative in alternatives:
            if isinstance(alternative, _UnionFieldType):
                continue
            type_name = alternative.type_name
            if callable(type_name) and not isinstance(type_name, str):
                try:
                    return type_name(raw_text)
                except Exception:
                    continue
            converted = self._apply_builtin(raw_text, type_name, alternative)
            if converted is not None or type_name == "str":
                return raw_text if type_name == "str" else converted

        if self.strict:
            names = [getattr(item, "type_name", item) for item in alternatives]
            raise ValueError(
                f"Cannot convert field '{field_name}' in schema '{schema_name}' "
                f"to any of {names!r}: {raw_text!r}"
            )
        first = alternatives[0]
        return self._fallback_value(first, raw_text) if isinstance(first, FieldType) else raw_text

    def _extract_text(self, value: Any) -> str:
        """Extract raw text from a decoded field value."""
        if isinstance(value, dict) and "text" in value:
            return str(value["text"])
        return str(value) if value is not None else ""

    def _convert_list_value(self, value: Any, ft: FieldType) -> list:
        """Convert a field declared as list, preserving multi-span values."""
        raw_values = value if isinstance(value, list) else [value]
        item_ft = FieldType(type_name=ft.list_item_type)
        result = []

        for raw_value in raw_values:
            raw_text = self._extract_text(raw_value)
            for part in re.split(ft.list_separator, raw_text):
                part = part.strip()
                if not part:
                    continue
                if ft.list_item_type == "str":
                    result.append(part)
                else:
                    converted = self._apply_builtin(part, ft.list_item_type, item_ft)
                    result.append(converted if converted is not None else part)

        return result

    def _convert_single(
        self,
        value: Any,
        ft: FieldType,
        schema_name: str,
        field_name: str,
    ) -> Any:
        """Convert a single (non-list) value."""
        raw_text = self._extract_text(value)

        type_name = ft.type_name

        # Callable converter
        if callable(type_name) and not isinstance(type_name, str):
            try:
                return type_name(raw_text)
            except Exception:
                return self._handle_failure(raw_text, ft, schema_name, field_name)

        # Built-in converters
        converted = self._apply_builtin(raw_text, type_name, ft)
        if converted is not None:
            return converted

        # str type always succeeds
        if type_name == "str":
            return raw_text

        return self._handle_failure(raw_text, ft, schema_name, field_name)

    def _apply_builtin(self, text: str, type_name: str, ft: FieldType) -> Any:
        """Apply a built-in type converter. Returns None on failure."""
        if type_name == "str":
            return text
        if type_name == "int":
            return _convert_int(text)
        if type_name == "float":
            return _convert_float(text)
        if type_name == "bool":
            return _convert_bool(text)
        if type_name == "date":
            return _convert_date(text, ft.date_formats)
        if type_name == "datetime":
            return _convert_datetime(text, ft.date_formats)
        if type_name == "list":
            return self._convert_list(text, ft)
        return None

    def _convert_list(self, text: str, ft: FieldType) -> list:
        """Split text into a list and optionally convert items."""
        parts = re.split(ft.list_separator, text)
        parts = [p.strip() for p in parts if p.strip()]
        if ft.list_item_type == "str":
            return parts
        item_ft = FieldType(type_name=ft.list_item_type)
        result = []
        for part in parts:
            converted = self._apply_builtin(part, ft.list_item_type, item_ft)
            result.append(converted if converted is not None else part)
        return result

    def _handle_failure(
        self,
        raw_text: str,
        ft: FieldType,
        schema_name: str,
        field_name: str,
    ) -> Any:
        """Handle conversion failure: strict raises, otherwise use default or raw text."""
        if self.strict:
            raise ValueError(
                f"Cannot convert field '{field_name}' in schema '{schema_name}' "
                f"to {ft.type_name!r}: {raw_text!r}"
            )
        return self._fallback_value(ft, raw_text)

    @staticmethod
    def _fallback_value(ft: FieldType, raw_value: Any = None) -> Any:
        if ft.default_factory is not None:
            try:
                return ft.default_factory()
            except Exception:
                return raw_value
        if ft.default is not None:
            return ft.default
        return raw_value
