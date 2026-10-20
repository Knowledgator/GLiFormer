"""Regression targets, mentions, and processor tensors for typed fields."""

from dataclasses import asdict

import pytest
import torch

from gliformer.config import StructuringHeadConfig
from gliformer.processing import typed_regression as tr
from gliformer.processing.mappings import BatchClassesMapping
from gliformer.processing.typed_fields import TypedFieldSpec
from gliformer.tasks.structuring.processor import StructuringProcessor
from tests.conftest import FakeWordsSplitter, make_config


class TestNumberDecomposition:
    @pytest.mark.parametrize("value", [0.42, 7.0, 2220.0, 9.98e6, 7.4e11, -350.0])
    def test_round_trip_through_exponent_and_digits(self, value):
        spec = TypedFieldSpec(type="number")
        target = tr.number_target(value, spec)
        probs = [0.0] * tr.NUM_EXP_CLASSES
        probs[max(range(len(target.exponent_soft)), key=target.exponent_soft.__getitem__)] = 1.0
        decoded, _z, negative = tr.decode_number_median(
            probs, target.digits, target.z, 1.0 if value < 0 else 0.0, spec,
        )
        assert decoded == pytest.approx(value, rel=1e-6)
        assert negative == (value < 0)

    def test_zero_and_decade_edges(self):
        spec = TypedFieldSpec(type="number")
        zero = tr.number_target(0.0, spec)
        assert zero.zero and zero.exponent_soft[0] == 1.0
        edge = tr.number_target(9.98e6, spec)
        k6, k7 = tr.exponent_index(6), tr.exponent_index(7)
        assert edge.exponent_soft[k6] == pytest.approx(0.8)
        assert edge.exponent_soft[k7] == pytest.approx(0.2)
        # The argmax says 10^7, but k=6 lies closest to the direct regression.
        probs = [0.0] * tr.NUM_EXP_CLASSES
        probs[k7] = 1.0
        decoded, _, _ = tr.decode_number_median(probs, edge.digits, edge.z, 0.0, spec)
        assert decoded == pytest.approx(9.98e6, rel=1e-6)

    def test_spaces(self):
        bounded = TypedFieldSpec(type="estimate", minimum=1, maximum=5000, scale="log")
        assert tr.number_space(bounded) == "bounded"
        assert 0.0 <= tr.number_to_space(230, bounded) <= 1.0
        signed = TypedFieldSpec.from_value({"$type": "number", "$signed": True})
        assert tr.number_from_space(tr.number_to_space(-4.0, signed), signed) == pytest.approx(-4.0)
        millions = TypedFieldSpec.from_value({"$type": "number", "$unit_scale": 1e6})
        assert tr.number_to_space(4.3e9, millions) == pytest.approx(3.6334685, rel=1e-6)

    def test_interpolate_levels(self):
        values = [float(i) for i in range(len(tr.QUANTILE_LEVELS))]
        assert tr.interpolate_level(0.5, values) == tr.MEDIAN_INDEX
        assert tr.interpolate_level(0.2, values) == pytest.approx(1.0 + (0.1 / 0.15))


class TestTemporal:
    def test_components(self):
        target = tr.temporal_target("2024-11-03T11:42:10-05:00", tr.FAMILY_DATETIME)
        assert (target.year, target.month, target.day, target.hour, target.minute, target.second) == (
            2024, 11, 3, 11, 42, 10,
        )
        assert tr.temporal_target("4:30 pm", tr.FAMILY_TIME).hour == 16

    def test_compose_clamps_impossible_days(self):
        assert tr.compose_temporal(tr.FAMILY_DATE, year=2023, month=2, day=31) == "2023-02-28"
        assert tr.compose_temporal(tr.FAMILY_TIME, hour=9, minute=5) == "09:05"


class TestMentions:
    def test_numbers_years_and_glued_years(self):
        words = (
            "Invoice of 14 February 2024 : total $ 2 , 220 . 00 ; FY2023 revenue 4.3 billion "
            "at 10 : 30"
        ).split()
        mentions = tr.find_mentions(words)
        found = [(words[m.start:m.end + 1], m.value, m.is_year) for m in mentions]
        assert (["14", "February", "2024"], 2024.0, True) in found
        assert (["2", ",", "220", ".", "00"], 2220.0, False) in found
        assert (["FY2023"], 2023.0, True) in found
        assert (["4.3", "billion"], 4.3e9, False) in found
        # Clock times and the day inside a date are not numbers.
        assert all(value not in (10.0, 30.0, 14.0) for _, value, _ in found)


TYPES = {
    "price": {"$type": "number"},
    "posted": {"$type": "date"},
    "due": {"$type": "date", "$stats": ["value", "p10", "p90"]},
    "pickup": {"$type": "time"},
    "fair": {"$type": "estimate", "$min": 1, "$max": 5000, "$scale": "log"},
    "posted_text": {"$type": "text", "$format": "date"},
}
TEXT = "Posted on March 3, 2024 for $250. Pickup at 5:30 pm. Built in 2019."
GOLD = {
    "price": 250,
    "posted": "2024-03-03",
    "due": {"value": "2024-03-17", "p10": "2024-03-10", "p90": "2024-03-24"},
    "pickup": "17:30",
    "fair": {"value": 230, "median": 230, "p10": 180, "p90": 300},
    "posted_text": "2024-03-03",
}


def _processor():
    config = make_config(structuring_config=asdict(StructuringHeadConfig(
        multi_level=True, typed_fields=True, typed_quantity_head="regression",
        typed_description_dropout=0.0, typed_shuffle_options=False,
    )))
    return StructuringProcessor(config, words_splitter=FakeWordsSplitter())


def test_processor_regression_tensors():
    processor = _processor()
    row = {
        "text": TEXT,
        "structuring": {"$root": dict(GOLD)},
        "structuring_types": {"$root": TYPES},
        "reference_date": "2024-01-01",
    }
    mapping = processor.get_classes_mapping([row])
    classes = BatchClassesMapping(cat_mapping=[], extraction_mapping=[], structuring_mapping=mapping)
    processor.resolve_spans(row)
    labels = processor.create_labels([row], classes, max_seq_len=64)
    words = TEXT.split()
    inputs = processor.prepare_typed_inputs(classes, texts=[words])

    # The extraction field is grounded on its verbatim span.
    instance = next(iter(row["structuring"].values()))[0]
    assert instance["posted_text"]["text"] == "March 3, 2024"

    family = inputs["structuring_typed_quantity_family"][0].tolist()
    ids = inputs["structuring_typed_quantity_field_idx"][0].tolist()
    field_to_id = mapping[0].items[0].field_class_to_id.class_to_id
    slot = {label: ids.index(field_to_id[label]) for label in ("price", "posted", "due", "pickup", "fair")}
    assert family[slot["price"]] == tr.FAMILY_NUMBER and family[slot["posted"]] == tr.FAMILY_DATE
    assert family[slot["pickup"]] == tr.FAMILY_TIME
    assert inputs["structuring_typed_quantity_space"][0, slot["fair"]] == 1  # bounded
    mention_features = inputs["structuring_typed_mention_features"][0]
    is_year = mention_features[:, 3].bool() & inputs["structuring_typed_mention_mask"][0]
    years = mention_features[is_year, 4] * 50 + 2000
    assert sorted(round(year) for year in years.tolist()) == [2019, 2024]

    z = labels["structuring_typed_number_z"][0, 0]
    assert z[slot["price"], 0].item() == pytest.approx(2.39794, rel=1e-4)
    month = labels["structuring_typed_temporal_month"][0, 0]
    day = labels["structuring_typed_temporal_day"][0, 0]
    assert month[slot["posted"]] == 2 and day[slot["posted"]] == 2  # zero-based March 3
    assert labels["structuring_typed_temporal_hour"][0, 0, slot["pickup"]] == 17
    assert labels["structuring_typed_temporal_month"][0, 0, slot["pickup"]] == -1
    assert labels["structuring_typed_temporal_era"][0, 0, slot["posted"]] == tr.ERA_CE
    assert labels["structuring_typed_temporal_year_digits"][0, 0, slot["posted"]].tolist() == [2, 0, 2, 4]
    assert labels["structuring_typed_temporal_year_digits"][0, 0, slot["pickup"]].tolist() == [-1] * 4
    quantiles = labels["structuring_typed_temporal_quantiles"][0, 0, slot["due"]]
    mask = labels["structuring_typed_temporal_quantiles_mask"][0, 0, slot["due"]]
    assert quantiles[mask].tolist() == [-7.0, 7.0]
    fair_quantiles = labels["structuring_typed_number_quantiles_mask"][0, 0, slot["fair"], 0]
    assert fair_quantiles.sum() == 3


def test_commas_join_digit_groups_but_not_dates():
    words = "Date : October 14 , 2024 ; paid $ 12 , 500 , then 7 , 8".split()
    found = {(tuple(words[m.start:m.end + 1]), m.value, m.is_year) for m in tr.find_mentions(words)}
    assert (("October", "14", ",", "2024"), 2024.0, True) in found
    assert (("12", ",", "500"), 12500.0, False) in found
    assert (("7",), 7.0, False) in found and (("8",), 8.0, False) in found


class TestYearEncoding:
    @pytest.mark.parametrize("year", [2024, 1, 0, -43, 1066, 9999])
    def test_round_trip(self, year):
        era, digits = tr.year_to_era_digits(year)
        assert tr.era_digits_to_year(era, digits) == year

    def test_bce_dates(self):
        assert tr.year_to_era_digits(-43) == (tr.ERA_BCE, [0, 0, 4, 4])
        target = tr.temporal_target("-0043-03-15", tr.FAMILY_DATE)
        assert (target.year, target.month, target.day) == (-43, 3, 15)
        assert tr.shift_temporal("-0043-03-15", tr.FAMILY_DATE, 30) == "-0043-04-14"

    def test_calendar_matches_python(self):
        from datetime import date

        for ordinal in range(1, 3_000_000, 9973):
            day = date.fromordinal(ordinal)
            days = (day - date(1970, 1, 1)).days
            assert tr.days_from_civil(day.year, day.month, day.day) == days
            assert tr.civil_from_days(days) == (day.year, day.month, day.day)
