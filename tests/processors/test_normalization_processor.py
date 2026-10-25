"""Normalization targets built from structuring gold values and the parser."""

from dataclasses import asdict

import pytest
import torch

from gliformer.config import StructuringHeadConfig
from gliformer.processing.mappings import BatchClassesMapping
from gliformer.tasks.normalization.processor import NormalizationProcessor
from gliformer.tasks.normalization.values import FAMILY_DATE, FAMILY_NONE, IGNORE, NULL
from gliformer.tasks.structuring.processor import StructuringProcessor
from tests.conftest import FakeWordsSplitter, make_config

TEXT = "Listing posted on May 2024 : Sony headphones for 250 dollars , built in 2019 ."
TYPES = {
    "title": {"$type": "text"},
    "price": {"$type": "number"},
    "posted": {"$type": "date"},
    "total": {"$type": "number"},
}
OUTPUT = {"title": "Sony headphones", "price": 250, "posted": "2024-05", "total": 999}


def _processors(parser=False, multi_level=True, **normalization):
    config = make_config(
        structuring_config=asdict(StructuringHeadConfig(
            multi_level=multi_level, typed_fields=True,
            typed_description_dropout=0.0, typed_shuffle_options=False,
        )),
        normalization_config={"parser": {"enabled": parser}, **normalization},
    )
    structuring = StructuringProcessor(config, words_splitter=FakeWordsSplitter())
    normalization_processor = NormalizationProcessor(config)
    normalization_processor.structuring_processor = structuring
    return structuring, normalization_processor


def _labels(parser=False, multi_level=True, **normalization):
    structuring, normalization_processor = _processors(parser, multi_level, **normalization)
    name = "$root" if multi_level else "listing"
    row = {
        "text": TEXT,
        "structuring": {name: dict(OUTPUT) if multi_level else [dict(OUTPUT)]},
        "structuring_types": {name: TYPES},
    }
    mapping = structuring.get_classes_mapping([row])
    classes = BatchClassesMapping(cat_mapping=[], extraction_mapping=[], structuring_mapping=mapping)
    structuring.create_labels([row], classes, max_seq_len=32)
    return normalization_processor.create_labels([row], classes, max_seq_len=32), TEXT.split()


def _by_text(labels, words):
    result = {}
    for slot, (start, end) in enumerate(labels["normalization_span_idx"][0].tolist()):
        if labels["normalization_span_mask"][0, slot]:
            result[" ".join(words[start:end + 1])] = slot
    return result


@pytest.mark.parametrize("multi_level", [True, False])
def test_gold_value_fields_become_labelled_spans(multi_level):
    labels, words = _labels(multi_level=multi_level)
    slots = _by_text(labels, words)
    # Text fields and derived values (the total) have no normalization span.
    assert set(slots) == {"May 2024", "250"}
    price, posted = slots["250"], slots["May 2024"]
    assert labels["normalization_number_value"][0, price].item() == 250.0
    assert labels["normalization_number_mask"][0].tolist()[price] is True
    assert labels["normalization_number_mask"][0].tolist()[posted] is False
    assert labels["normalization_temporal_family"][0, posted].item() == FAMILY_DATE
    assert labels["normalization_year"][0, posted].item() == 2024
    assert labels["normalization_month"][0, posted].item() == 5
    assert labels["normalization_day"][0, posted].item() == NULL
    assert labels["normalization_hour"][0, posted].item() == IGNORE
    assert labels["normalization_temporal_family"][0, price].item() == FAMILY_NONE
    assert labels["normalization_number_value"].dtype == torch.float64


def test_parser_adds_silver_spans_and_gold_wins():
    labels, words = _labels(parser=True)
    slots = _by_text(labels, words)
    assert {"May 2024", "250", "2019"} <= set(slots)
    year = slots["2019"]
    assert labels["normalization_number_value"][0, year].item() == 2019.0
    assert labels["normalization_year"][0, year].item() == 2019
    assert labels["normalization_month"][0, year].item() == NULL


def test_limits_drop_spans():
    labels, words = _labels(parser=True, max_spans=1)
    assert labels["normalization_span_idx"].shape[1] == 1
    structuring, normalization_processor = _processors(parser=True)
    row = {"text": TEXT, "structuring": {"$root": {}}, "structuring_types": {"$root": TYPES}}
    targets = normalization_processor.item_targets(row, limit=10)
    assert all(target.end < 10 for target in targets)


def test_no_labelled_spans_gives_no_tensors():
    structuring, normalization_processor = _processors()
    row = {"text": "No values here .", "structuring": {"$root": {"title": "values"}},
           "structuring_types": {"$root": TYPES}}
    mapping = structuring.get_classes_mapping([row])
    classes = BatchClassesMapping(cat_mapping=[], extraction_mapping=[], structuring_mapping=mapping)
    structuring.create_labels([row], classes, max_seq_len=8)
    assert normalization_processor.create_labels([row], classes) is None


def test_temporal_gold_values_are_grounded_by_components():
    from gliformer.processing.typed_fields import TypedFieldSpec

    ground = StructuringProcessor._ground_value
    text = "Paid May 3, 2024 and again in May 2024 ; shipped 05/11/2024 ."
    span, gold = ground(text, TypedFieldSpec(type="date"), "2024-05", None)
    assert span["text"] == "May 2024" and gold == "2024-05"
    assert ground(text, TypedFieldSpec(type="date"), "2024-05-03", None)[0]["text"] == "May 3, 2024"
    # Day-first order is tried when the month-first reading does not match.
    assert ground(text, TypedFieldSpec(type="date"), "2024-11-05", None)[0]["text"] == "05/11/2024"
    assert ground(text, TypedFieldSpec(type="date"), "2024-06", None) is None
