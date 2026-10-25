"""``json`` task fields: validation, model templates, and output shaping.

A ``json`` task declares fields either as ``labels`` (flat fields with
JSON field properties, NEW_API.md §10.2) or as a JSON Schema (§10.4). Both
compile to one ``FieldNode`` tree, which builds the structuring template the
model runs and turns the model's output back into the declared shape.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from ...processing.typed_fields import TypedFieldSpec
from .capabilities import Capabilities
from .errors import APIError, invalid
from .labels import Label, normalize_labels
from .render import Location, RenderedInput

FIELD_TYPES = (
    "text", "boolean", "options", "number", "integer", "duration",
    "date", "time", "datetime",
)
TEXT_FORMATS = ("number", "integer", "duration", "date", "time", "datetime")
NUMERIC_FORMATS = ("number", "integer", "duration")
DATE_FORMATS = ("date", "datetime")

TYPE_PROPS: dict[str, frozenset[str]] = {
    "text": frozenset({"format", "decimals", "unit", "dayfirst"}),
    "boolean": frozenset(),
    "options": frozenset({"options", "multi"}),
    "number": frozenset({"decimals", "unit"}),
    "integer": frozenset({"unit"}),
    "duration": frozenset({"decimals", "unit"}),
    "date": frozenset({"dayfirst"}),
    "time": frozenset(),
    "datetime": frozenset({"dayfirst"}),
}
FIELD_PROPS = frozenset({"type"}).union(*TYPE_PROPS.values())
# Value fields read their value from the normalization head.
VALUE_FIELD_TYPES = ("number", "integer", "duration", "date", "time", "datetime")


@dataclass
class FieldNode:
    """One declared field or container of a ``json`` task."""

    kind: str                                   # "scalar" | "object" | "array"
    key: str | None = None                      # output key
    model_name: str | None = None               # field name shown to the model
    path: str = ""                              # request path, for errors
    type: str = "text"                          # scalar field type
    props: dict[str, Any] = field(default_factory=dict)
    description: str | None = None
    options: list[Label] | None = None
    children: list[FieldNode] = field(default_factory=list)
    item: FieldNode | None = None
    required: bool = False

    @property
    def multi(self) -> bool:
        return self.type == "options" and bool(self.props.get("multi", False))

    def empty_value(self) -> Any:
        if self.kind == "array" or self.multi:
            return []
        return None


# -- validation ---------------------------------------------------------------


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_field_props(
    node: FieldNode,
    raw: dict[str, Any],
    capabilities: Capabilities,
    template: str,
) -> None:
    """Check the JSON field properties of one scalar field."""
    field_type = raw.get("type", "text")
    where = node.path
    if field_type not in FIELD_TYPES:
        raise invalid(
            f"Unknown field type {field_type!r}; expected one of {', '.join(FIELD_TYPES)}",
            f"{where}/type",
        )
    if field_type not in capabilities.json_field_types:
        raise APIError(
            "UNSUPPORTED_LABEL_TYPE",
            f"Field type '{field_type}' needs typed structuring heads.",
            f"{where}/type",
            {"model": capabilities.model, "supported": list(capabilities.json_field_types)},
        )
    allowed = TYPE_PROPS[field_type]
    for key in raw:
        if key == "type" or key not in FIELD_PROPS:
            continue
        if key not in allowed:
            if key == "format":
                message = "'format' is only valid on fields of type 'text'"
            else:
                message = f"'{key}' is not valid on fields of type '{field_type}'"
            raise invalid(message, f"{where}/{key}")
    node.type = field_type
    props = {key: raw[key] for key in raw if key in allowed}

    fmt = props.get("format")
    if fmt is not None:
        if fmt not in TEXT_FORMATS:
            raise invalid(
                f"Unknown format {fmt!r}; expected one of {', '.join(TEXT_FORMATS)}",
                f"{where}/format",
            )
        if fmt not in capabilities.json_text_formats:
            raise APIError(
                "UNSUPPORTED_LABEL_TYPE",
                f"Text format '{fmt}' is not supported by this model.",
                f"{where}/format",
            )
    if field_type == "text":
        if ("decimals" in props or "unit" in props) and fmt not in NUMERIC_FORMATS:
            raise invalid(
                "'decimals' and 'unit' need a numeric 'format' on text fields", where
            )
        if "dayfirst" in props and fmt not in DATE_FORMATS:
            raise invalid("'dayfirst' needs format 'date' or 'datetime'", f"{where}/dayfirst")
    if "decimals" in props and not (
        isinstance(props["decimals"], int) and not isinstance(props["decimals"], bool)
        and props["decimals"] >= 0
    ):
        raise invalid("'decimals' must be a non-negative integer", f"{where}/decimals")
    if "unit" in props and not isinstance(props["unit"], str):
        raise invalid("'unit' must be a string", f"{where}/unit")
    if "dayfirst" in props and not isinstance(props["dayfirst"], bool):
        raise invalid("'dayfirst' must be a boolean", f"{where}/dayfirst")
    if "multi" in props and not isinstance(props["multi"], bool):
        raise invalid("'multi' must be a boolean", f"{where}/multi")
    if field_type == "options":
        if "options" not in props:
            raise invalid("Fields of type 'options' need 'options'", where)
        options = normalize_labels(
            props["options"], f"{where}/options", template=template,
        )
        if not options:
            raise invalid("'options' must list at least one choice", f"{where}/options")
        node.options = options
        props["options"] = [option.id for option in options]
    node.props = props
    try:
        TypedFieldSpec.from_value(scalar_descriptor(node, typed=True, show_description=False))
    except (TypeError, ValueError) as exc:
        raise invalid(str(exc), where) from exc


# -- labels form ----------------------------------------------------------------


def fields_from_labels(
    spec: Any,
    path: str,
    capabilities: Capabilities,
) -> tuple[FieldNode, list[Label]]:
    labels = normalize_labels(
        spec, path,
        template="{title}",
        extra_keys=FIELD_PROPS,
        supported_keys=capabilities.supported_label_keys,
    )
    if not labels:
        raise invalid("A json task needs at least one field", path)
    root = FieldNode(kind="object", path=path)
    for label in labels:
        node = FieldNode(
            kind="scalar",
            key=label.id,
            model_name=label.text,
            path=f"{path}/{_escape(label.id)}",
            description=label.description,
        )
        validate_field_props(node, label.props, capabilities, capabilities.label_template)
        root.children.append(node)
    return root, labels


# -- JSON Schema form -----------------------------------------------------------


def fields_from_schema(schema: Any, path: str, capabilities: Capabilities) -> FieldNode:
    if not isinstance(schema, dict):
        raise invalid("'schema' must be a JSON Schema object", path)
    resolver = _SchemaResolver(schema, path)
    node = resolver.node(schema, path, key=None, depth=0)
    if node.kind == "scalar":
        raise invalid("The root of a json schema must be an object or an array", path)
    _validate_tree(node, capabilities)
    return node


class _SchemaResolver:
    def __init__(self, root: dict[str, Any], path: str) -> None:
        self.root = root
        self.path = path

    def resolve(self, node: dict[str, Any], where: str, seen: tuple[str, ...] = ()) -> dict[str, Any]:
        ref = node.get("$ref")
        if ref is None:
            return node
        if not isinstance(ref, str) or not ref.startswith("#/"):
            raise invalid("Only local '$ref' values ('#/...') are supported", f"{where}/$ref")
        if ref in seen:
            raise invalid("Recursive '$ref' schemas are not supported", f"{where}/$ref")
        target: Any = self.root
        for token in ref[2:].split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            if not isinstance(target, dict) or token not in target:
                raise invalid(f"Unresolvable '$ref' {ref!r}", f"{where}/$ref")
            target = target[token]
        if not isinstance(target, dict):
            raise invalid(f"'$ref' {ref!r} does not point to a schema", f"{where}/$ref")
        merged = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
        return self.resolve(merged, where, (*seen, ref))

    def unwrap_nullable(self, node: dict[str, Any], where: str) -> dict[str, Any]:
        for keyword in ("anyOf", "oneOf"):
            options = node.get(keyword)
            if not isinstance(options, list):
                continue
            non_null = [
                option for option in options
                if not (isinstance(option, dict) and option.get("type") == "null")
            ]
            if len(non_null) != 1 or not isinstance(non_null[0], dict):
                raise invalid(
                    f"'{keyword}' is only supported for an optional (nullable) value",
                    f"{where}/{keyword}",
                )
            rest = {k: v for k, v in node.items() if k != keyword}
            return {**self.resolve(non_null[0], where), **rest}
        return node

    @staticmethod
    def schema_type(node: dict[str, Any], where: str) -> str | None:
        value = node.get("type")
        if isinstance(value, list):
            types = [item for item in value if item != "null"]
            if len(types) != 1:
                raise invalid("Union types other than T|null are not supported", f"{where}/type")
            return types[0]
        return value

    def node(self, schema: dict[str, Any], where: str, *, key: str | None, depth: int) -> FieldNode:
        if depth > 16:
            raise invalid("JSON Schema nesting is too deep", where)
        schema = self.unwrap_nullable(self.resolve(schema, where), where)
        schema_type = self.schema_type(schema, where)
        model_name = key
        description = schema.get("description") if isinstance(schema.get("description"), str) else None
        if schema_type == "object" or (schema_type is None and "properties" in schema):
            properties = schema.get("properties") or {}
            if not isinstance(properties, dict) or not properties:
                raise invalid("Object schemas need 'properties'", where)
            required = schema.get("required") or []
            node = FieldNode(kind="object", key=key, model_name=model_name, path=where, description=description)
            for child_key, child_schema in properties.items():
                if not isinstance(child_schema, dict):
                    raise invalid("Property schemas must be objects", f"{where}/properties/{_escape(child_key)}")
                child = self.node(
                    child_schema,
                    f"{where}/properties/{_escape(child_key)}",
                    key=child_key,
                    depth=depth + 1,
                )
                child.required = child_key in required
                node.children.append(child)
            return node
        if schema_type == "array":
            items = schema.get("items")
            if not isinstance(items, dict):
                raise invalid("Array schemas need an 'items' schema", where)
            items_resolved = self.unwrap_nullable(self.resolve(items, f"{where}/items"), f"{where}/items")
            if "enum" in items_resolved:
                return self.scalar(
                    {**items_resolved, "x-multi": True}, f"{where}/items", key, model_name, description,
                )
            item = self.node(items, f"{where}/items", key=None, depth=depth + 1)
            return FieldNode(
                kind="array", key=key, model_name=model_name, path=where,
                item=item, description=description,
            )
        return self.scalar(schema, where, key, model_name, description)

    def scalar(
        self,
        schema: dict[str, Any],
        where: str,
        key: str | None,
        model_name: str | None,
        description: str | None,
    ) -> FieldNode:
        schema_type = self.schema_type(schema, where)
        raw: dict[str, Any] = {}
        field_type = schema.get("x-field-type")
        if "enum" in schema:
            values = schema["enum"]
            if not isinstance(values, list) or not values:
                raise invalid("'enum' must be a non-empty list", f"{where}/enum")
            raw["type"] = "options"
            raw["options"] = [str(value) for value in values if value is not None]
            if schema.get("x-multi"):
                raw["multi"] = True
        elif field_type is not None:
            raw["type"] = field_type
            if field_type == "text" and "x-format" in schema:
                raw["format"] = schema["x-format"]
        elif "x-format" in schema:
            raw["type"] = "text"
            raw["format"] = schema["x-format"]
        elif schema_type == "string" or schema_type is None:
            raw["type"] = "text"
            fmt = schema.get("format")
            if fmt in ("date", "time", "date-time"):
                raw["format"] = {"date-time": "datetime"}.get(fmt, fmt)
        elif schema_type in ("number", "integer"):
            raw["type"] = "text"
            raw["format"] = schema_type
        elif schema_type == "boolean":
            raw["type"] = "boolean"
        else:
            raise invalid(f"Unsupported JSON Schema type {schema_type!r}", f"{where}/type")
        # Range constraints (minimum, maximum, multipleOf) are not used: values
        # are read from the text, not predicted inside a range.
        numeric = raw["type"] in ("number", "integer", "duration") or (
            raw["type"] == "text" and raw.get("format") in NUMERIC_FORMATS
        )
        if numeric and "x-unit" in schema:
            raw["unit"] = schema["x-unit"]
        return FieldNode(
            kind="scalar", key=key, model_name=model_name, path=where,
            description=description, props=raw,
        )


def _validate_tree(node: FieldNode, capabilities: Capabilities) -> None:
    if node.kind == "scalar":
        raw = node.props
        node.props = {}
        validate_field_props(node, raw, capabilities, capabilities.label_template)
        return
    if node.kind == "array":
        _validate_tree(node.item, capabilities)  # type: ignore[arg-type]
        return
    seen: dict[str, str] = {}
    for child in node.children:
        name = child.model_name or ""
        if name in seen:
            raise invalid(f"Fields {seen[name]!r} and {child.key!r} render to the same name", child.path)
        seen[name] = child.key or ""
        _validate_tree(child, capabilities)


# -- model template ---------------------------------------------------------------


def value_type(node: FieldNode) -> str | None:
    """Value type of a scalar field: number/integer/duration/date/time/datetime.

    ``text`` fields with a ``format`` are value fields too.
    """

    if node.type == "text":
        return node.props.get("format")
    return node.type if node.type in VALUE_FIELD_TYPES else None


def scalar_descriptor(node: FieldNode, *, typed: bool, show_description: bool) -> Any:
    props = node.props
    description = node.description if show_description else None
    kind = value_type(node)
    if kind is not None:
        if not typed and description is None:
            return "str"
        descriptor: dict[str, Any] = {"$type": "number" if kind == "integer" else kind}
        if kind == "integer":
            descriptor["$integer"] = True
        for key in ("unit", "decimals", "dayfirst"):
            if key in props:
                descriptor[f"${key}"] = props[key]
        if not typed:
            descriptor = {"$type": "text"}
    elif node.type == "text":
        if not typed or description is None:
            return "str"
        descriptor = {"$type": "text"}
    elif node.type == "boolean":
        if not typed or description is None:
            return "bool"
        descriptor = {"$type": "bool"}
    else:
        descriptor = {
            "$type": "class",
            "$options": [option.text for option in node.options or []],
            "$multi": bool(props.get("multi", False)),
        }
    if description is not None:
        descriptor["$description"] = description
    return descriptor


def model_template(root: FieldNode, *, typed: bool, show_description: bool) -> dict[str, Any]:
    """The ``structures`` argument the model runs for this task."""

    def build(node: FieldNode, record_item: bool) -> Any:
        if node.kind == "scalar":
            return scalar_descriptor(node, typed=typed, show_description=show_description)
        if node.kind == "array":
            item = node.item
            if item.kind == "object":  # type: ignore[union-attr]
                return [build(item, True)]  # type: ignore[arg-type]
            if item.kind == "scalar" and item.type == "text" and "format" not in item.props:  # type: ignore[union-attr]
                return ["str"]
            if not typed:
                raise APIError(
                    "UNSUPPORTED_LABEL_TYPE",
                    "Lists of typed values need typed structuring heads.",
                    node.path,
                )
            item_type = item.props.get("format") or item.type  # type: ignore[union-attr]
            return {"$type": "list", "$items": "number" if item_type == "integer" else item_type}
        result: dict[str, Any] = {}
        for child in node.children:
            name = child.model_name or child.key or ""
            if record_item and child.required:
                name = "!" + name
            result[name] = build(child, False)
        return result

    return {"$root": build(root, False)}


# -- output -------------------------------------------------------------------------

# Normalized date/time values: "2024-05", "2024-05-18", "14:30", "2024-05-18T14:30:00".
_ISO_VALUE = re.compile(
    r"-?\d{4}(-\d{2}(-\d{2})?)?(T\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?"
    r"|--\d{2}-\d{2}|\d{2}(:\d{2}(:\d{2}(\.\d+)?)?)?"
)

_NEGATIVE = re.compile(
    r"^\s*(no|false|n|0|none|never|nope|negative)\b|\b(not|never|unpaid|without|"
    r"declined|denied|refused|cancelled|canceled|failed)\b|n't\b",
    re.IGNORECASE,
)


def span_to_bool(text: str) -> bool:
    """A boolean field's span: negated phrasing is ``False``, anything else ``True``."""
    return not bool(_NEGATIVE.search(text))


def _clean_numbers(value: Any, decimals: int | None = None) -> Any:
    """Integral floats become ints, recursively."""
    if isinstance(value, float):
        return _clean_number(value, decimals=decimals)
    if isinstance(value, dict):
        return {key: _clean_numbers(item, decimals) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_numbers(item, decimals) for item in value]
    return value


def _clean_number(value: float, integer: bool = False, decimals: int | None = None) -> int | float:
    if integer:
        return int(round(value))
    if decimals is not None:
        value = round(value, decimals)
    if isinstance(value, float) and value.is_integer() and abs(value) < 2**53:
        return int(value)
    return value


@dataclass
class JsonOutput:
    value: Any
    evidence: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    unparsed: dict[str, dict[str, Any]] = field(default_factory=dict)


class JsonShaper:
    """Turn the model's structuring output into the declared JSON value."""

    def __init__(self, root: FieldNode, rendered: RenderedInput, *, want_evidence: bool) -> None:
        self.root = root
        self.rendered = rendered
        self.want_evidence = want_evidence
        self.evidence: dict[str, list[dict[str, Any]]] = {}
        self.unparsed: dict[str, dict[str, Any]] = {}

    def shape(self, raw: Any) -> JsonOutput:
        value = self.node_value(self.root, raw, "")
        return JsonOutput(value=value, evidence=self.evidence, unparsed=self.unparsed)

    def node_value(self, node: FieldNode, raw: Any, where: str) -> Any:
        if node.kind == "object":
            if isinstance(raw, list):
                raw = next((item for item in raw if isinstance(item, dict)), None)
            raw = raw if isinstance(raw, dict) else {}
            result = {}
            for child in node.children:
                name = child.model_name or child.key or ""
                child_where = f"{where}/{_escape(child.key or '')}"
                result[child.key] = self.node_value(child, raw.get(name), child_where)
            return result
        if node.kind == "array":
            if raw is None:
                return []
            items = raw if isinstance(raw, list) else [raw]
            values = []
            for item in items:
                value = self.node_value(node.item, item, f"{where}/{len(values)}")  # type: ignore[arg-type]
                if value is None or value == {} or (
                    isinstance(value, dict) and all(v in (None, []) for v in value.values())
                ):
                    continue
                values.append(value)
            return values
        return self.scalar_value(node, raw, where)

    def scalar_value(self, node: FieldNode, raw: Any, where: str) -> Any:
        if node.multi:
            items = raw if isinstance(raw, list) else ([] if raw is None else [raw])
            return self.options_ids(node, items)
        if isinstance(raw, list):
            raw = next((item for item in raw if item not in (None, "")), None)
        if raw is None or raw == "":
            return None
        if value_type(node) is not None:
            return self.value_field(node, raw, where)
        if node.type == "text":
            text = str(raw)
            self.add_evidence(where, text)
            return text
        if node.type == "boolean":
            if isinstance(raw, bool):
                return raw
            text = str(raw)
            self.add_evidence(where, text)
            return span_to_bool(text)
        ids = self.options_ids(node, [raw])
        return ids[0] if ids else None

    def options_ids(self, node: FieldNode, items: list[Any]) -> list[str]:
        by_text = {option.text: option.id for option in node.options or []}
        by_folded = {option.text.casefold(): option.id for option in node.options or []}
        by_id = {option.id: option.id for option in node.options or []}
        ids: list[str] = []
        for item in items:
            text = str(item)
            label_id = by_text.get(text) or by_id.get(text) or by_folded.get(text.casefold())
            if label_id is not None and label_id not in ids:
                ids.append(label_id)
        return ids

    def value_field(self, node: FieldNode, raw: Any, where: str) -> Any:
        """Value of a number/date field, as normalized by the model.

        Span text is not parsed here: a field the model returned only as text
        is reported under ``unparsed`` with its location and has no value.
        """

        kind = value_type(node)
        span_text = None
        if isinstance(raw, dict) and "value" in raw:
            span_text = raw.get("text")
            raw = raw["value"]
        value = None
        if kind in NUMERIC_FORMATS:
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                value = _clean_number(
                    float(raw), integer=kind == "integer", decimals=node.props.get("decimals"),
                )
        elif isinstance(raw, str) and _ISO_VALUE.fullmatch(raw.strip()):
            value = raw.strip()
        if value is not None:
            if isinstance(span_text, str) and span_text:
                self.add_evidence(where, span_text)
            return value
        text = str(raw)
        location = self.first_location(text)
        self.unparsed[where] = location.to_dict() if location is not None else {"text": text}
        return None

    def first_location(self, text: str) -> Location | None:
        found = self.rendered.find(text)
        return found[0] if found else None

    def add_evidence(self, where: str, text: str, location: Location | None = None) -> None:
        if not self.want_evidence:
            return
        location = location or self.first_location(text)
        if location is not None:
            self.evidence.setdefault(where, []).append(location.to_dict())



def _escape(token: str) -> str:
    return str(token).replace("~", "~0").replace("/", "~1")
