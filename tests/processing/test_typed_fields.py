"""Typed structuring field specs, value parsing, and source spans."""

import pytest

from gliformer.processing.typed_fields import (
    TypedFieldSpec,
    find_anchor,
    is_typed_descriptor,
    lookup_group_types,
    spec_from_json_schema,
    specs_from_json_schema,
)

TEXT = (
    "Signed on March 3, 2024; payment due 30 days later at 5:30 pm. "
    "Price $1,250.50 (approx 1.2 million users)."
)


class TestSpec:
    def test_aliases_and_kinds(self):
        # Number and date fields are span fields with a value type.
        number = TypedFieldSpec.from_value("int")
        assert number.kind == "span" and number.value_type == "number"
        assert TypedFieldSpec.from_value({"$type": "int"}).integer
        assert TypedFieldSpec.from_value({"$type": "date-time"}).type == "datetime"
        assert TypedFieldSpec.from_value({"$options": ["a", "b"]}).kind == "class"
        text = TypedFieldSpec.from_value("str")
        assert text.kind == "span" and text.value_type is None
        dates = TypedFieldSpec.from_value({"type": "list", "items": "date"})
        assert dates.kind == "span" and dates.value_type == "date" and dates.temporal

    def test_round_trip_descriptor(self):
        spec = TypedFieldSpec.from_value({
            "$type": "number", "$unit": "USD", "$decimals": 2, "$description": "price",
        })
        again = TypedFieldSpec.from_value(spec.to_dict())
        assert again == spec
        assert is_typed_descriptor(spec.to_dict())

    @pytest.mark.parametrize(
        "spec",
        [
            {"$type": "class"},
            {"$type": "class", "$options": ["a", "a"]},
            {"$type": "number", "$decimals": 20},
        ],
    )
    def test_invalid_specs(self, spec):
        with pytest.raises(ValueError):
            TypedFieldSpec.from_value(spec)

    def test_text_with_format_is_a_value_field(self):
        assert TypedFieldSpec.from_value({"$type": "text", "$format": "date"}) == TypedFieldSpec(type="date")
        integer = TypedFieldSpec.from_value({"$type": "text", "$format": "integer", "$unit": "USD"})
        assert integer.type == "number" and integer.integer and integer.unit == "USD"
        with pytest.raises(ValueError):
            TypedFieldSpec.from_value({"$type": "class", "$options": ["a"], "$format": "date"})

    def test_prompt_suffix(self):
        assert TypedFieldSpec(type="text").prompt_suffix() == ""
        assert TypedFieldSpec(type="class", options=("a",), multi=False).prompt_suffix() == (
            "( class , single )"
        )
        number = TypedFieldSpec(type="number", unit="USD", integer=True)
        assert number.prompt_suffix() == "( number , USD , integer )"
        assert TypedFieldSpec(type="date").prompt_suffix() == "( date )"


class TestParsing:
    @pytest.mark.parametrize(
        "text",
        ["March 3, 2024", "3 March 2024", "March 3rd, 2024", "2024-03-03", "March 3 , 2024"],
    )
    def test_date_parsing(self, text):
        spec = TypedFieldSpec(type="date")
        assert spec.parse_text(text) == spec.encode("2024-03-03")

    def test_number_time_duration_parsing(self):
        assert TypedFieldSpec(type="number").parse_text("$ 1 , 250 . 50") == 1250.5
        assert TypedFieldSpec(type="number").parse_text("1.2 million") == 1.2e6
        assert TypedFieldSpec(type="time").parse_text("5:30 pm") == 17.5 * 3600
        assert TypedFieldSpec(type="duration", unit="day").parse_text("2 weeks") == 14

    def test_text_fields_do_not_parse(self):
        assert TypedFieldSpec(type="text").parse_text("42") is None
        assert TypedFieldSpec(type="text").encode(42) is None


class TestSourceSpans:
    def test_exact_spans(self):
        date = TypedFieldSpec(type="date")
        anchor = find_anchor(TEXT, date, date.encode("2024-03-03"))
        assert TEXT[anchor["start"]:anchor["end"]] == "March 3, 2024"
        number = TypedFieldSpec(type="number")
        assert find_anchor(TEXT, number, 1250.5)["text"] == "1,250.50"
        time = TypedFieldSpec(type="time")
        assert find_anchor(TEXT, time, time.encode("17:30"))["text"] == "5:30 pm"
        duration = TypedFieldSpec(type="duration", unit="day")
        assert find_anchor(TEXT, duration, 30)["text"] == "30 days"

    def test_derived_values_have_no_span(self):
        date = TypedFieldSpec(type="date")
        assert find_anchor(TEXT, date, date.encode("2024-04-02")) is None
        assert find_anchor(TEXT, TypedFieldSpec(type="number"), 1251.0) is None

    def test_evidence_is_preferred(self):
        text = "Paid $40 on Monday. Refund of $40 issued later."
        number = TypedFieldSpec(type="number")
        anchor = find_anchor(text, number, 40, snippets=["Refund of $40"])
        assert anchor["start"] == text.index("40 issued")


class TestJsonSchema:
    def test_generator_schema_properties(self):
        schema = {
            "type": "object",
            "properties": {
                "segment": {
                    "type": "array", "items": {"type": "string", "enum": ["a", "b", "c"]},
                    "x-field-type": "class", "description": "segment",
                },
                "units": {"type": ["integer", "null"], "x-field-type": "number"},
                "audit": {"type": ["string", "null"], "format": "date", "x-field-type": "date"},
            },
        }
        specs = specs_from_json_schema(schema)
        assert specs["segment"].options == ("a", "b", "c")
        assert specs["units"].integer
        assert specs["audit"].type == "date"
        assert spec_from_json_schema({"type": "string"}).type == "text"

    def test_group_lookup(self):
        types = {"$root": {"a": {"$type": "date"}}}
        assert lookup_group_types(types, names=("root", "$root")) == {"a": {"$type": "date"}}
        assert lookup_group_types({"a": {"$type": "date"}}, names=("x",)) == {
            "a": {"$type": "date"}
        }
        assert lookup_group_types(None, names=("x",)) == {}


class TestRealWorldTemporalSpans:
    """Spans as the decoder sees them: word tokens re-joined with spaces."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("September 30 , 2024", "2024-09-30"),
            ("Sept 9, 2024", "2024-09-09"),
            ("14 - OCT - 2024", "2024-10-14"),
            ("18 januari 2023", "2023-01-18"),
            ("3. März 2024", "2024-03-03"),
            ("Monday, 3 March 2024", "2024-03-03"),
            ("2024-10-28T19", "2024-10-28"),
        ],
    )
    def test_dates(self, text, expected):
        spec = TypedFieldSpec(type="date")
        assert spec.parse_text(text) == spec.encode(expected)

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("11 : 20 AM", "11:20"),
            ("10 : 11 : 45", "10:11:45"),
            ("14 : 30 EST", "14:30"),
            ("16 : 10 PM", "16:10"),
            ("2025-02-14T03 : 11 : 00 . 102Z", "03:11"),
        ],
    )
    def test_times(self, text, expected):
        spec = TypedFieldSpec(type="time")
        assert int(spec.parse_text(text)) == int(spec.encode(expected))

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("2024-10-22 09 : 14 : 02", "2024-10-22T09:14:02"),
            ("14 January 2025 06 : 15", "2025-01-14T06:15:00"),
            ("11 : 42 : 10 EST on November 3 , 2024", "2024-11-03T11:42:10"),
            ("2024-11-03T11:42:10-05:00", "2024-11-03T11:42:10"),
        ],
    )
    def test_datetimes_keep_wall_clock(self, text, expected):
        spec = TypedFieldSpec(type="datetime")
        assert int(spec.parse_text(text)) == int(spec.encode(expected))

    def test_context_supplies_missing_information(self):
        context = "Rebooted on November 3, 2024. Failure logged at 11:42:10 EST; doors at 3:00 PM."
        datetime_spec = TypedFieldSpec(type="datetime")
        start = context.index("11:42")
        assert datetime_spec.parse_in_context(
            "11 : 42 : 10 EST", context, start
        ) == datetime_spec.encode("2024-11-03T11:42:10")
        time_spec = TypedFieldSpec(type="time")
        assert time_spec.parse_in_context(
            "3 : 00", context, context.index("3:00")
        ) == time_spec.encode("15:00")

    def test_day_first_detection_and_option(self):
        spec = TypedFieldSpec(type="date")
        context = "Received 25/03/2024, shipped 05/11/2024."
        assert spec.parse_in_context("05/11/2024", context) == spec.encode("2024-11-05")
        assert spec.parse_text("05/11/2024") == spec.encode("2024-05-11")
        day_first = TypedFieldSpec.from_value({"$type": "date", "$dayfirst": True})
        assert day_first.parse_text("05/11/2024") == day_first.encode("2024-11-05")
        # Training accepts whichever order reproduces the gold value.
        anchor = find_anchor("Meeting 05/11/2024.", spec, spec.encode("2024-11-05"))
        assert anchor["x"] == spec.encode("2024-11-05")
