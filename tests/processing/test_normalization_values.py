"""Parser and value helpers of the normalization task."""

import pytest

from gliformer.tasks.normalization.parser import NormalizationParser, gold_parts
from gliformer.tasks.normalization.values import (
    FAMILY_DATE,
    FAMILY_DATETIME,
    FAMILY_TIME,
    IGNORE,
    NULL,
    TemporalParts,
    component_codes,
    compose_partial,
    year_class,
)

WORDS = (
    "Signed on March 3 , 2024 ; revenue rose to $ 1 , 250.50 ( 4.3 billion ) in 2021 , "
    "at 5:30 pm , opened May 2024 , one hundred and five units , 2024-05-18T14:30"
).split()


def _found(parser):
    return {" ".join(WORDS[s.start:s.end + 1]): s for s in parser.find(WORDS)}


def test_parser_reads_numbers_dates_and_times():
    found = _found(NormalizationParser(number_words=True))
    assert found["March 3 , 2024"].parts == TemporalParts(2024, 3, 3)
    assert found["1 , 250.50"].number == 1250.5
    assert found["4.3 billion"].number == 4.3e9
    assert found["5:30 pm"].parts == TemporalParts(hour=17, minute=30)
    assert found["May 2024"].parts == TemporalParts(2024, 5, None)
    assert found["one hundred and five"].number == 105
    assert found["2024-05-18T14:30"].family == FAMILY_DATETIME
    # A year written alone is both a number and a year-only date.
    year = found["2021"]
    assert year.number == 2021 and year.parts == TemporalParts(2021) and year.family == FAMILY_DATE
    # Numbers inside dates and times are not separate spans.
    assert "3" not in found and "30" not in found


def test_parser_switches():
    assert all(s.number is None for s in NormalizationParser(numbers=False).find(WORDS))
    found = _found(NormalizationParser(dates=False, times=False))
    assert all(s.parts is None for s in found.values())
    assert "one hundred and five" not in _found(NormalizationParser())
    assert len(NormalizationParser(max_mentions=2).find(WORDS)) == 2


@pytest.mark.parametrize("value,family,parts", [
    ("2024-05", FAMILY_DATE, TemporalParts(2024, 5)),
    ("May 2024", FAMILY_DATE, TemporalParts(2024, 5)),
    ("March 3, 2024", FAMILY_DATE, TemporalParts(2024, 3, 3)),
    ("--05-18", FAMILY_DATE, TemporalParts(None, 5, 18)),
    ("5:30 pm", FAMILY_TIME, TemporalParts(hour=17, minute=30)),
    ("14:30:05", FAMILY_TIME, TemporalParts(hour=14, minute=30, second=5)),
    ("2024-05-18T14:30", FAMILY_DATETIME, TemporalParts(2024, 5, 18, 14, 30)),
])
def test_gold_values_keep_written_precision(value, family, parts):
    assert gold_parts(value, family) == parts


def test_component_codes_and_composition():
    codes = component_codes(TemporalParts(2024, 5), FAMILY_DATE)
    assert codes == {"year": 2024, "month": 5, "day": NULL,
                     "hour": IGNORE, "minute": IGNORE, "second": IGNORE}
    time_codes = component_codes(TemporalParts(hour=9, minute=5), FAMILY_TIME)
    assert time_codes["year"] == IGNORE and time_codes["second"] == NULL
    assert component_codes(TemporalParts(-43, 3, 15), FAMILY_DATE)["year"] == IGNORE
    assert compose_partial(FAMILY_DATE, TemporalParts(2024, 5)) == "2024-05"
    assert compose_partial(FAMILY_DATE, TemporalParts(None, 5, 18)) == "--05-18"
    assert compose_partial(FAMILY_DATETIME, TemporalParts(2024, 5, 18, 9, 5)) == "2024-05-18T09:05"
    assert compose_partial(FAMILY_TIME, TemporalParts(hour=9)) == "09"
    assert year_class(1850, 1800, 2199) == 50
    assert year_class(1700, 1800, 2199) == 400
    assert year_class(None, 1800, 2199) == 401


def test_parser_skips_mentions_with_two_readings():
    words = "sold 1,286 units , 5 m long , on 05/11/2024 and 25/11/2024 , for 1,286.50 USD".split()
    found = {" ".join(words[s.start:s.end + 1]): s for s in NormalizationParser().find(words)}
    assert "1,286" not in found and "5 m" not in found and "5" not in found
    assert "05/11/2024" not in found and "11" not in found
    assert found["25/11/2024"].parts == TemporalParts(2024, 11, 25)
    assert found["1,286.50"].number == 1286.5


@pytest.mark.parametrize("text,value", [
    ("fifteen and a half", 15.5),
    ("two and three quarters", 2.75),
    ("a hundred and five", 105),
])
def test_number_words_with_fractions(text, value):
    from gliformer.processing.typed_fields import TypedFieldSpec

    assert TypedFieldSpec(type="number").parse_text(text) == value


def test_gold_numbers_ground_only_on_spans_that_convert_exactly():
    from gliformer.processing.structuring_processor import StructuringProcessor
    from gliformer.processing.typed_fields import TypedFieldSpec

    ground = StructuringProcessor._ground_value
    number = TypedFieldSpec(type="number")
    text = "Net cash flow: minus eighty-two point nine five billion dollars; budget 29 Tausend; cost (944,000)."
    assert ground(text, number, -82.95e9, None)[0]["text"] == "minus eighty-two point nine five billion"
    assert ground(text, number, 29000, None)[0]["text"] == "29 Tausend"
    assert ground(text, number, -944000, None)[0]["text"] == "(944,000)"
    # "944,000" alone converts to a positive number: not a target for -944000
    # in a text without the parentheses.
    assert ground("cost 944,000 below plan", number, -944000, None) is None
