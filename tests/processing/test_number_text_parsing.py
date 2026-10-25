"""Copied number spans: formats, words, fractions, signs, and cut-off spans."""

import pytest

from gliformer.processing.typed_fields import TypedFieldSpec, parse_text_value


@pytest.mark.parametrize(
    "text, value",
    [
        # English digits keep working.
        ("1,286 employees", 1286),
        ("1,234.56", 1234.56),
        ("1.382", 1.382),
        ("$5.4 million", 5.4e6),
        ("-5", -5),
        ("Patient 2951", 2951),
        # European grouping and decimal comma.
        ("1.382.216,45 EUR", 1382216.45),
        ("6.043,4 mg/dl", 6043.4),
        ("998,9", 998.9),
        ("0,25 %", 0.25),
        ("8 . 693 , 0 millones", 8693.0),  # word-joined span
        ("285 , 8 mg", 285.8),
        # Scientific notation.
        ("2.3E-6", 2.3e-6),
        ("4.6 × 10^-4", 4.6e-4),
        # Fractions.
        ("1½ tablets", 1.5),
        ("½", 0.5),
        ("3 1/2", 3.5),
        ("3/4", 0.75),
        ("1 and a quarter", 1.25),
        # Words.
        ("eight hundred and seventy-three", 873),
        ("five thousand thirty-three", 5033),
        ("hundred and sixteen", 116),
        ("a hundred", 100),
        ("two and three quarters", 2.75),
        ("three quarters", 0.75),
        ("six and one quarter people", 6.25),
        ("minus twenty point five", -20.5),
        ("zero point zero seven seven dollars", 0.077),
        ("one point five million", 1.5e6),
        # Accounting negatives.
        ("(395,327)", -395327),
        ("( 395 , 327 )", -395327),
    ],
)
def test_parse_number_text(text, value):
    assert parse_text_value(text, "number") == pytest.approx(value, rel=1e-12)


CONTEXT = (
    "Revenue 3,227,830 | Operating expenses (335,075) | Net income (loss) (395,327) "
    "in (2023), growth (5%), temp -12.5 C, range 10-15"
)


@pytest.mark.parametrize(
    "context, span, value",
    [
        (CONTEXT, "(395,327", -395327),  # the span kept one parenthesis
        (CONTEXT, "395,327)", -395327),
        (CONTEXT, "335,075", -335075),
        (CONTEXT, "3,227,830", 3227830),
        (CONTEXT, "2023", 2023),  # a year in parentheses is not negative
        (CONTEXT, "5%", 5),  # "(5%)" is an aside
        (CONTEXT, "12.5", -12.5),
        (CONTEXT, "15", 15),  # a range, not a minus
        ("concentration was 4.6 × 10^-4 mol/L", "4.6", 4.6e-4),
        ("Take 1½ tablets", "1", 1.5),
        ("Take 1 and a quarter tablets", "1", 1.25),
        ("alcanzaron 8.693,0 millones", "8.693", 8693.0),
        ("was 1,286 people", "1", 1286),
        ("revenue of 5 million", "5", 5),  # multiplier words are not pulled in
        ("on 3/4/2024", "3", 3),
    ],
)
def test_parse_in_context_extends_and_signs_spans(context, span, value):
    spec = TypedFieldSpec(type="number")
    assert spec.parse_in_context(span, context, context.find(span)) == pytest.approx(value)
