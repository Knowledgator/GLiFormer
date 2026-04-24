"""Structuring output formatter — converts raw text spans to typed values.

Since structuring schemas can specify fields of various types (str, int, float,
bool, list, date, etc.), the raw text extracted by the decoder needs nuanced
conversion into proper Python objects.
"""

import re
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Union


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
    """

    def __init__(
        self,
        type_name: Union[str, Callable] = "str",
        default: Any = None,
        list_separator: str = r"\s*,\s*",
        list_item_type: str = "str",
        date_formats: Optional[List[str]] = None,
    ):
        self.type_name = type_name
        self.default = default
        self.list_separator = list_separator
        self.list_item_type = list_item_type
        self.date_formats = date_formats or [
            "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
            "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y",
            "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
        ]


# ── Built-in converters ──────────────────────────────────────────────────

_TRUE_STRINGS = frozenset({"true", "yes", "1", "on", "y", "t"})
_FALSE_STRINGS = frozenset({"false", "no", "0", "off", "n", "f"})


def _convert_bool(text: str) -> Optional[bool]:
    """Convert text to bool, return None if ambiguous."""
    t = text.strip().lower()
    if t in _TRUE_STRINGS:
        return True
    if t in _FALSE_STRINGS:
        return False
    return None


def _convert_int(text: str) -> Optional[int]:
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


def _convert_float(text: str) -> Optional[float]:
    """Convert text to float, handling common formatting."""
    cleaned = text.strip().replace(",", "").replace(" ", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _convert_date(text: str, formats: List[str]) -> Optional[date]:
    """Try multiple date formats, return first successful parse."""
    t = text.strip()
    for fmt in formats:
        try:
            return datetime.strptime(t, fmt).date()
        except ValueError:
            continue
    return None


def _convert_datetime(text: str, formats: List[str]) -> Optional[datetime]:
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
        schema_types: Dict[str, Dict[str, Union[str, FieldType, Callable]]],
        strict: bool = False,
    ):
        """
        Args:
            schema_types: ``{schema_name: {field_name: type_spec}}``.
                Type spec can be a string name, a FieldType, or a callable.
            strict: If True, raise ValueError on conversion failure instead
                of falling back to raw text.
        """
        self.strict = strict
        self._field_types: Dict[str, Dict[str, FieldType]] = {}
        for schema_name, fields in schema_types.items():
            self._field_types[schema_name] = {}
            for field_name, type_spec in fields.items():
                self._field_types[schema_name][field_name] = self._normalize_type(type_spec)

    @staticmethod
    def _normalize_type(type_spec: Union[str, FieldType, Callable]) -> FieldType:
        """Normalize a type specification into a FieldType."""
        if isinstance(type_spec, FieldType):
            return type_spec
        if callable(type_spec) and not isinstance(type_spec, str):
            return FieldType(type_name=type_spec)
        if isinstance(type_spec, str):
            return FieldType(type_name=type_spec)
        raise TypeError(f"Unsupported type spec: {type_spec!r}")

    def format(
        self,
        raw_output: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Format a single text's structuring output.

        Args:
            raw_output: ``{schema_name: [instance_dict, ...]}``, as returned
                by ``GLiNExT.structure()`` or the ``"structuring"`` key of
                ``GLiNExT.inference()``.

        Returns:
            Same structure with field values converted to their declared types.
        """
        result = {}
        for schema_name, instances in raw_output.items():
            field_types = self._field_types.get(schema_name)
            if field_types is None:
                # No type info for this schema — pass through unchanged
                result[schema_name] = instances
                continue
            result[schema_name] = [
                self._format_instance(instance, field_types, schema_name)
                for instance in instances
            ]
        return result

    def format_batch(
        self,
        batch_output: List[Dict[str, List[Dict[str, Any]]]],
    ) -> List[Dict[str, List[Dict[str, Any]]]]:
        """Format a batch of structuring outputs (one per text)."""
        return [self.format(item) for item in batch_output]

    def _format_instance(
        self,
        instance: Dict[str, Any],
        field_types: Dict[str, FieldType],
        schema_name: str,
    ) -> Dict[str, Any]:
        """Convert all field values in a single instance dict."""
        formatted = {}
        for field_name, value in instance.items():
            ft = field_types.get(field_name)
            if ft is None:
                formatted[field_name] = value
                continue
            formatted[field_name] = self._convert_value(
                value, ft, schema_name, field_name,
            )
        return formatted

    def _convert_value(
        self,
        value: Any,
        ft: FieldType,
        schema_name: str,
        field_name: str,
    ) -> Any:
        """Convert a single value according to its FieldType."""
        if ft.type_name == "list":
            return self._convert_list_value(value, ft)

        # GLiNER2 treats non-list fields as scalar: when multiple spans were
        # decoded for a scalar field, keep the highest-ranked/first value.
        if isinstance(value, list):
            if not value:
                return ft.default if ft.default is not None else None
            return self._convert_single(value[0], ft, schema_name, field_name)

        return self._convert_single(value, ft, schema_name, field_name)

    def _extract_text(self, value: Any) -> str:
        """Extract raw text from a decoded field value."""
        if isinstance(value, dict) and "text" in value:
            return str(value["text"])
        return str(value) if value is not None else ""

    def _convert_list_value(self, value: Any, ft: FieldType) -> List:
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

    def _convert_list(self, text: str, ft: FieldType) -> List:
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
        self, raw_text: str, ft: FieldType, schema_name: str, field_name: str,
    ) -> Any:
        """Handle conversion failure: strict raises, otherwise use default or raw text."""
        if self.strict:
            raise ValueError(
                f"Cannot convert field '{field_name}' in schema '{schema_name}' "
                f"to {ft.type_name!r}: {raw_text!r}"
            )
        if ft.default is not None:
            return ft.default
        return raw_text
