"""Tests for StructuringOutputFormatter."""

from datetime import date, datetime

import pytest

from glinext.processing.formatting import (
    FieldType,
    StructuringOutputFormatter,
    _convert_bool,
    _convert_float,
    _convert_int,
)

# ── Low-level converter tests ────────────────────────────────────────────

class TestConvertBool:
    def test_true_variants(self):
        for s in ["true", "True", "YES", "1", "on", "y", "T"]:
            assert _convert_bool(s) is True

    def test_false_variants(self):
        for s in ["false", "False", "NO", "0", "off", "n", "F"]:
            assert _convert_bool(s) is False

    def test_ambiguous_returns_none(self):
        assert _convert_bool("maybe") is None
        assert _convert_bool("") is None


class TestConvertInt:
    def test_basic(self):
        assert _convert_int("42") == 42
        assert _convert_int("-7") == -7

    def test_with_commas(self):
        assert _convert_int("1,000") == 1000

    def test_float_like(self):
        assert _convert_int("42.0") == 42

    def test_invalid(self):
        assert _convert_int("abc") is None
        assert _convert_int("42.5") is None


class TestConvertFloat:
    def test_basic(self):
        assert _convert_float("3.14") == pytest.approx(3.14)
        assert _convert_float("-0.5") == pytest.approx(-0.5)

    def test_with_commas(self):
        assert _convert_float("1,000.5") == pytest.approx(1000.5)

    def test_invalid(self):
        assert _convert_float("abc") is None


# ── Formatter tests ──────────────────────────────────────────────────────

class TestStructuringOutputFormatter:
    def test_str_passthrough(self):
        fmt = StructuringOutputFormatter({"person": {"name": "str"}})
        result = fmt.format({"person": [{"name": "John"}]})
        assert result == {"person": [{"name": "John"}]}

    def test_int_conversion(self):
        fmt = StructuringOutputFormatter({"person": {"age": "int"}})
        result = fmt.format({"person": [{"age": "30"}]})
        assert result == {"person": [{"age": 30}]}

    def test_float_conversion(self):
        fmt = StructuringOutputFormatter({"product": {"price": "float"}})
        result = fmt.format({"product": [{"price": "9.99"}]})
        assert result["product"][0]["price"] == pytest.approx(9.99)

    def test_bool_conversion(self):
        fmt = StructuringOutputFormatter({"person": {"active": "bool"}})
        result = fmt.format({"person": [{"active": "yes"}]})
        assert result == {"person": [{"active": True}]}

    def test_date_conversion(self):
        fmt = StructuringOutputFormatter({"event": {"date": "date"}})
        result = fmt.format({"event": [{"date": "2024-03-15"}]})
        assert result == {"event": [{"date": date(2024, 3, 15)}]}

    def test_datetime_conversion(self):
        fmt = StructuringOutputFormatter({"event": {"ts": "datetime"}})
        result = fmt.format({"event": [{"ts": "2024-03-15 10:30:00"}]})
        assert result == {"event": [{"ts": datetime(2024, 3, 15, 10, 30)}]}

    def test_list_conversion(self):
        fmt = StructuringOutputFormatter({
            "person": {"skills": FieldType("list", list_item_type="str")}
        })
        result = fmt.format({"person": [{"skills": "python, java, rust"}]})
        assert result == {"person": [{"skills": ["python", "java", "rust"]}]}

    def test_list_with_typed_items(self):
        fmt = StructuringOutputFormatter({
            "data": {"scores": FieldType("list", list_item_type="int")}
        })
        result = fmt.format({"data": [{"scores": "10, 20, 30"}]})
        assert result == {"data": [{"scores": [10, 20, 30]}]}

    def test_custom_callable(self):
        fmt = StructuringOutputFormatter({
            "data": {"upper": str.upper}
        })
        result = fmt.format({"data": [{"upper": "hello"}]})
        assert result == {"data": [{"upper": "HELLO"}]}

    def test_mixed_types(self):
        fmt = StructuringOutputFormatter({
            "person": {
                "name": "str",
                "age": "int",
                "salary": "float",
                "active": "bool",
            }
        })
        result = fmt.format({"person": [{
            "name": "Alice",
            "age": "28",
            "salary": "75000.50",
            "active": "true",
        }]})
        assert result == {"person": [{
            "name": "Alice",
            "age": 28,
            "salary": pytest.approx(75000.50),
            "active": True,
        }]}

    def test_unknown_schema_passes_through(self):
        fmt = StructuringOutputFormatter({"person": {"name": "str"}})
        result = fmt.format({"other": [{"x": "y"}]})
        assert result == {"other": [{"x": "y"}]}

    def test_unknown_field_passes_through(self):
        fmt = StructuringOutputFormatter({"person": {"name": "str"}})
        result = fmt.format({"person": [{"name": "Alice", "extra": "val"}]})
        assert result == {"person": [{"name": "Alice", "extra": "val"}]}

    def test_conversion_failure_keeps_raw(self):
        fmt = StructuringOutputFormatter({"data": {"num": "int"}})
        result = fmt.format({"data": [{"num": "not a number"}]})
        assert result == {"data": [{"num": "not a number"}]}

    def test_conversion_failure_strict_raises(self):
        fmt = StructuringOutputFormatter({"data": {"num": "int"}}, strict=True)
        with pytest.raises(ValueError, match="Cannot convert"):
            fmt.format({"data": [{"num": "not a number"}]})

    def test_conversion_failure_with_default(self):
        fmt = StructuringOutputFormatter({
            "data": {"num": FieldType("int", default=0)}
        })
        result = fmt.format({"data": [{"num": "bad"}]})
        assert result == {"data": [{"num": 0}]}

    def test_explicit_none_default_differs_from_omitted_default(self):
        fmt = StructuringOutputFormatter(
            {
                "data": {
                    "raw": FieldType("int"),
                    "nullable": FieldType("int", default=None),
                }
            }
        )

        result = fmt.format(
            {"data": [{"raw": "invalid", "nullable": "invalid"}]}
        )
        assert result == {
            "data": [{"raw": "invalid", "nullable": None}]
        }

    def test_format_batch(self):
        fmt = StructuringOutputFormatter({"person": {"age": "int"}})
        batch = [
            {"person": [{"age": "25"}]},
            {"person": [{"age": "30"}]},
        ]
        result = fmt.format_batch(batch)
        assert result == [
            {"person": [{"age": 25}]},
            {"person": [{"age": 30}]},
        ]

    def test_multi_value_field(self):
        """When decoder returns a list for a field (multiple spans)."""
        fmt = StructuringOutputFormatter({"person": {"name": "str", "age": "int"}})
        result = fmt.format({"person": [{"name": "Alice", "age": ["25", "26"]}]})
        assert result == {"person": [{"name": "Alice", "age": 25}]}

    def test_list_field_preserves_multiple_decoded_spans(self):
        fmt = StructuringOutputFormatter({
            "person": {"skills": FieldType("list", list_item_type="str")}
        })
        result = fmt.format({"person": [{"skills": ["python", "java"]}]})
        assert result == {"person": [{"skills": ["python", "java"]}]}

    def test_scalar_str_field_collapses_multiple_decoded_spans(self):
        fmt = StructuringOutputFormatter({"person": {"name": "str"}})
        result = fmt.format({"person": [{"name": ["Alice", "Bob"]}]})
        assert result == {"person": [{"name": "Alice"}]}

    def test_dict_value_with_text(self):
        """Values that are still in {text, start, end} format."""
        fmt = StructuringOutputFormatter({"person": {"age": "int"}})
        result = fmt.format({"person": [{"age": {"text": "30", "start": 5, "end": 7}}]})
        assert result == {"person": [{"age": 30}]}

    def test_date_multiple_formats(self):
        fmt = StructuringOutputFormatter({"event": {"d": "date"}})
        for text, expected in [
            ("March 15, 2024", date(2024, 3, 15)),
            ("15/03/2024", date(2024, 3, 15)),
            ("03/15/2024", date(2024, 3, 15)),
        ]:
            result = fmt.format({"event": [{"d": text}]})
            assert result["event"][0]["d"] == expected, f"Failed for {text}"
