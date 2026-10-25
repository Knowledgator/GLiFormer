"""Scoring of typed fields in gliformer_eval.eval_typed."""

import pytest

from gliformer.processing.typed_fields import TypedFieldSpec
from gliformer_eval.eval_typed import diagnostics, score_field


def _score(raw, gold, pred):
    return score_field(TypedFieldSpec.from_value(raw), gold, pred)


def test_dates_match_exact_day():
    assert _score({"$type": "date"}, "2024-05-18", "2024-05-18")[2] == 1.0
    assert _score({"$type": "date"}, "2024-05-18", "May 18, 2024")[2] == 1.0
    assert _score({"$type": "date"}, "2024-05-18", "2024-05-19")[2] == 0.0


def test_numbers_within_one_percent_and_magnitudes():
    mode, kind, score, extras = _score({"$type": "number"}, 100.0, 100.5)
    assert (mode, kind, score) == ("value", "number", 1.0)
    assert extras["magnitude"] == "1e2" and extras["rel_error"] == pytest.approx(0.005)
    assert _score({"$type": "number"}, 100.0, "$1.2 million")[2] == 0.0
    results = {("value", "number"): {"scores": [1.0], "rel_error": [0.005], "abs_error": [],
                                     "magnitude": ["1e2"]}}
    assert "1e2" in diagnostics(results)


def test_class_text_and_list_scoring():
    assert _score({"$type": "class", "$options": ["a", "b"]}, ["a"], ["a", "b"])[2] == pytest.approx(2 / 3)
    assert _score({"$type": "text"}, "Sony", "sony.")[2] == 1.0
    assert _score({"$type": "list"}, ["x", "y"], ["x"])[0] == "extraction"
