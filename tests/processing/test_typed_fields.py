"""Typed structuring field specs, value codecs, anchors, and targets."""

import math
import random

import pytest
import torch

from gliformer.processing.typed_fields import (
    ORIGIN_ABSOLUTE,
    ORIGIN_ANCHOR,
    ORIGIN_REFERENCE,
    TypedFieldSpec,
    build_quantity_target,
    decode_quantity,
    find_anchor,
    is_typed_descriptor,
    lookup_group_types,
    quantile_from_histogram,
    spec_from_json_schema,
    specs_from_json_schema,
    stat_level,
    target_cdf,
)
from gliformer.tasks.structuring.typed import target_cdf as torch_target_cdf

TEXT = (
    "Signed on March 3, 2024; payment due 30 days later at 5:30 pm. "
    "Price $1,250.50 (approx 1.2 million users)."
)


class TestSpec:
    def test_aliases_and_kinds(self):
        assert TypedFieldSpec.from_value("int").kind == "quantity"
        assert TypedFieldSpec.from_value({"$type": "int"}).integer
        assert TypedFieldSpec.from_value({"$type": "date-time"}).type == "datetime"
        assert TypedFieldSpec.from_value({"$options": ["a", "b"]}).kind == "class"
        assert TypedFieldSpec.from_value("str").kind == "span"
        assert TypedFieldSpec.from_value({"type": "list", "items": "date"}).kind == "span"

    def test_default_statistics(self):
        assert TypedFieldSpec(type="date").stats == ("value",)
        estimate = TypedFieldSpec(type="estimate", minimum=1, maximum=10)
        assert estimate.stats == ("value", "median", "p10", "p90")
        quantiles = TypedFieldSpec.from_value({"$type": "number", "$quantiles": [0.05, 0.95]})
        assert quantiles.stats == ("value", "p5", "p95")

    def test_round_trip_descriptor(self):
        spec = TypedFieldSpec.from_value({
            "$type": "forecast", "$min": 0, "$max": 400, "$horizon": 3,
            "$interval": "month", "$description": "units",
        })
        again = TypedFieldSpec.from_value(spec.to_dict())
        assert again == spec
        assert is_typed_descriptor(spec.to_dict())

    @pytest.mark.parametrize(
        "spec",
        [
            {"$type": "class"},
            {"$type": "class", "$options": ["a", "a"]},
            {"$type": "estimate", "$min": 0, "$max": 10, "$scale": "log"},
            {"$type": "estimate", "$min": 5, "$max": 1},
            {"$type": "forecast", "$horizon": 100},
            {"$type": "number", "$stats": ["p150"]},
        ],
    )
    def test_invalid_specs(self, spec):
        with pytest.raises(ValueError):
            TypedFieldSpec.from_value(spec)

    def test_prompt_suffix(self):
        assert TypedFieldSpec(type="text").prompt_suffix() == ""
        assert TypedFieldSpec(type="class", options=("a",), multi=False).prompt_suffix() == (
            "( class , single )"
        )
        estimate = TypedFieldSpec(
            type="estimate", minimum=1, maximum=5000, scale="log", unit="USD",
        )
        assert estimate.prompt_suffix() == "( estimate , USD , 1 - 5000 , log )"

    def test_stat_levels(self):
        assert stat_level("value") is None
        assert stat_level("median") == 0.5
        assert stat_level("p10") == pytest.approx(0.1)


class TestCodecs:
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

    def test_decode_rounding(self):
        assert TypedFieldSpec(type="number", integer=True).decode(41.7) == 42
        assert TypedFieldSpec(type="number", decimals=2).decode(math.pi) == 3.14
        assert TypedFieldSpec(type="number").decode(250.0000000023) == 250.0
        bounded = TypedFieldSpec(type="estimate", minimum=0, maximum=100, step=5)
        assert bounded.decode(52.4) == 50.0
        assert bounded.decode(130) == 100.0
        assert TypedFieldSpec(type="time").decode(17.5 * 3600) == "17:30"

    @pytest.mark.parametrize(
        "spec,value,origin",
        [
            (TypedFieldSpec(type="date"), "2024-03-17", None),
            (TypedFieldSpec(type="datetime"), "2024-03-17T08:15:00", None),
            (TypedFieldSpec(type="number"), 1234.5, 1000.0),
            (TypedFieldSpec(type="estimate", minimum=1, maximum=5000, scale="log"), 950, None),
            (TypedFieldSpec(type="estimate", minimum=0, maximum=10), 3.25, None),
        ],
    )
    def test_unit_round_trip(self, spec, value, origin):
        if origin is None and not spec.bounded:
            origin = spec.default_origin()
        x = spec.encode(value)
        assert spec.from_unit(spec.to_unit(x, origin), origin) == pytest.approx(x, rel=1e-9)

    def test_resolve_origin(self):
        date = TypedFieldSpec(type="date")
        assert date.resolve_origin(19785.0, "2024-01-01") == (ORIGIN_ANCHOR, 19785.0)
        mode, origin = date.resolve_origin(None, "2024-01-01")
        assert mode == ORIGIN_REFERENCE and origin == date.encode("2024-01-01")
        assert date.resolve_origin(None, None)[0] == ORIGIN_ABSOLUTE
        bounded = TypedFieldSpec(type="estimate", minimum=0, maximum=1)
        assert bounded.resolve_origin(5.0, "2024-01-01") == (ORIGIN_ABSOLUTE, None)


class TestAnchors:
    def test_exact_anchors(self):
        date = TypedFieldSpec(type="date")
        anchor = find_anchor(TEXT, date, date.encode("2024-03-03"))
        assert TEXT[anchor["start"]:anchor["end"]] == "March 3, 2024"
        number = TypedFieldSpec(type="number")
        assert find_anchor(TEXT, number, 1250.5)["text"] == "1,250.50"
        time = TypedFieldSpec(type="time")
        assert find_anchor(TEXT, time, time.encode("17:30"))["text"] == "5:30 pm"
        duration = TypedFieldSpec(type="duration", unit="day")
        assert find_anchor(TEXT, duration, 30)["text"] == "30 days"

    def test_offset_anchor_for_derived_dates(self):
        date = TypedFieldSpec(type="date")
        anchor = find_anchor(TEXT, date, date.encode("2024-04-02"))
        assert anchor["text"] == "March 3, 2024"
        assert anchor["x"] == date.encode("2024-03-03")
        assert find_anchor(TEXT, date, date.encode("2024-04-02"), allow_offset=False) is None
        # Numbers never use offset anchors: a sum is not "a span plus a delta".
        assert find_anchor(TEXT, TypedFieldSpec(type="number"), 1251.0) is None

    def test_evidence_is_preferred(self):
        text = "Paid $40 on Monday. Refund of $40 issued later."
        number = TypedFieldSpec(type="number")
        anchor = find_anchor(text, number, 40, snippets=["Refund of $40"])
        assert anchor["start"] == text.index("40 issued")


class TestTargets:
    def test_estimate_target(self):
        spec = TypedFieldSpec(type="estimate", minimum=1, maximum=5000, scale="log")
        target = build_quantity_target(
            {"value": 950, "median": 950, "p10": 700, "p90": 1300},
            spec,
            origin_mode=ORIGIN_ABSOLUTE,
            origin=None,
        )
        assert target.present
        assert sorted(level for level, _ in target.points[0]) == [0.1, 0.5, 0.9]
        assert target.point_u[0] == pytest.approx(spec.to_unit(950, None))

    def test_forecast_and_null_and_invalid(self):
        spec = TypedFieldSpec(type="forecast", minimum=0, maximum=100, horizon=2)
        target = build_quantity_target(
            {"value": [1, 2], "p10": [0, 1], "p90": [2, 3]},
            spec, origin_mode=0, origin=None,
        )
        assert len(target.points) == 2 and len(target.point_u) == 2
        assert build_quantity_target(None, spec, origin_mode=0, origin=None).present is False
        assert build_quantity_target({"value": [1]}, spec, origin_mode=0, origin=None) is None

    def test_point_value_supervises_cdf_as_point_mass(self):
        spec = TypedFieldSpec(type="date")
        origin = spec.encode("2024-03-03")
        target = build_quantity_target(
            "2024-03-17", spec, origin_mode=ORIGIN_ANCHOR, origin=origin,
        )
        cdf = target_cdf(target.points[0], 16)
        assert set(cdf) <= {0.0, 1.0}

    def test_torch_cdf_matches_reference(self):
        rng = random.Random(0)
        for _ in range(100):
            levels = rng.sample([0.1, 0.25, 0.5, 0.75, 0.9], rng.randint(1, 4))
            points = [(level, rng.random()) for level in levels]
            lv = torch.zeros(5)
            vv = torch.zeros(5)
            mask = torch.zeros(5, dtype=torch.bool)
            for index, (level, u) in enumerate(points):
                lv[index], vv[index], mask[index] = level, u, True
            cdf, valid = torch_target_cdf(lv[None], vv[None], mask[None], 32)
            assert valid.all()
            assert torch.allclose(cdf[0], torch.tensor(target_cdf(points, 32)), atol=1e-4)


class TestDecoding:
    def test_histogram_statistics(self):
        spec = TypedFieldSpec(type="estimate", minimum=0, maximum=100)
        probs = [0.0] * 100
        probs[40:60] = [0.05] * 20
        median = quantile_from_histogram(probs, 0.5)
        assert median == pytest.approx(0.5)
        decoded = decode_quantity(spec, [probs], [[0.5] * 100], origin=None)
        assert set(decoded) == {"value", "median", "p10", "p90"}
        assert decoded["p10"] <= decoded["median"] <= decoded["p90"]

    def test_point_date_relative_to_anchor(self):
        spec = TypedFieldSpec(type="date")
        origin = spec.encode("2024-03-03")
        u = spec.to_unit(origin + 14, origin)
        bins = 256
        probs = [0.0] * bins
        index = int(u * bins)
        probs[index] = 1.0
        offsets = [0.5] * bins
        offsets[index] = u * bins - index
        assert decode_quantity(spec, [probs], [offsets], origin=origin) == "2024-03-17"


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
                "price": {
                    "type": ["object", "null"], "x-field-type": "estimate", "x-scale": "log",
                    "x-unit": "USD",
                    "properties": {"value": {"type": "number", "minimum": 1, "maximum": 10, "x-step": 0.01}},
                },
                "sales": {
                    "type": ["object", "null"], "x-field-type": "forecast", "x-scale": "linear",
                    "x-horizon": 3, "x-interval": "month",
                    "properties": {"value": {"type": "array", "items": {
                        "type": "number", "minimum": 0, "maximum": 400,
                    }}},
                },
            },
        }
        specs = specs_from_json_schema(schema)
        assert specs["segment"].options == ("a", "b", "c")
        assert specs["units"].integer
        assert specs["audit"].type == "date"
        assert specs["price"].bounded and specs["price"].scale == "log"
        assert specs["sales"].horizon == 3 and specs["sales"].maximum == 400
        assert spec_from_json_schema({"type": "string"}).type == "text"

    def test_group_lookup(self):
        types = {"$root": {"a": {"$type": "date"}}}
        assert lookup_group_types(types, names=("root", "$root")) == {"a": {"$type": "date"}}
        assert lookup_group_types({"a": {"$type": "date"}}, names=("x",)) == {
            "a": {"$type": "date"}
        }
        assert lookup_group_types(None, names=("x",)) == {}
