"""Regression quantity head: losses, gradients, inference outputs, decoding."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from gliformer.config import StructuringHeadConfig
from gliformer.processing import typed_regression as tr
from gliformer.processing.mappings import BaseClassMapping, StructuringItemMapping
from gliformer.processing.structuring_decoder import StructuringDecoder
from gliformer.tasks.structuring.typed_regression import (
    TypedRegressionModule,
    days_from_civil,
    quantile_offsets,
)
from tests.heads.conftest import D, make_config


def test_quantile_offsets_never_cross():
    offsets = quantile_offsets(torch.randn(4, tr.NUM_QUANTILE_OFFSETS) * 3)
    assert torch.all(offsets[:, 1:] >= offsets[:, :-1])
    assert torch.all(offsets[:, tr.MEDIAN_INDEX] == 0)


def test_days_from_civil_matches_python():
    from datetime import date

    years = torch.tensor([1970, 2000, 2024, 2024, 1999])
    months = torch.tensor([1, 2, 2, 3, 12])
    days = torch.tensor([1, 29, 29, 1, 31])
    expected = [(date(y, m, d) - date(1970, 1, 1)).days for y, m, d in zip(years.tolist(), months.tolist(), days.tolist())]
    assert days_from_civil(years, months, days).tolist() == expected


def _batch_targets(groups=1, records=1, fields=2, steps=1, mentions=3):
    exp_soft = torch.zeros(groups, records, fields, steps, tr.NUM_EXP_CLASSES)
    exp_soft[..., tr.exponent_index(2)] = 1.0
    return {
        "space": torch.zeros(groups, fields, dtype=torch.long),
        "exp_soft": exp_soft,
        "digits": torch.full((groups, records, fields, steps), 0.39794),
        "negative": torch.zeros(groups, records, fields, steps),
        "zero": torch.zeros(groups, records, fields, steps, dtype=torch.bool),
        "z": torch.full((groups, records, fields, steps), 2.39794),
        "quantiles": torch.zeros(groups, records, fields, steps, 7),
        "quantiles_mask": torch.zeros(groups, records, fields, steps, 7, dtype=torch.bool),
    }


def test_module_losses_backpropagate():
    torch.manual_seed(0)
    module = TypedRegressionModule(D, 0.0, max_horizon=4)
    words = torch.randn(1, 10, D)
    word_mask = torch.ones(1, 10, dtype=torch.bool)
    spans = torch.tensor([[[2, 3], [6, 6], [9, 12]]])  # last one is truncated
    values = torch.tensor([[[0, 3.3, 0, 1, 0.48], [0, 2.4, 0, 0, 0], [0, 1.0, 0, 0, 0]]])
    _memory, visible = module.mention_memory(
        words, word_mask, spans, values, torch.ones(1, 3, dtype=torch.bool),
    )
    assert visible.tolist() == [[True, True, False]]
    vectors = torch.randn(1, 1, 2, D, requires_grad=True)
    outputs = module.forward_heads(vectors, 1)
    unit = torch.tensor([[[True, False]]])
    step = torch.ones(1, 1, 2, 1, dtype=torch.bool)
    median, quantile = module.number_losses(outputs["number"], _batch_targets(), unit, step)
    temporal_targets = {
        "year_mask": torch.tensor([[[False, True]]]),
        "era": torch.tensor([[[-1, 0]]]),
        "year_digits": torch.tensor([[[[-1] * 4, [2, 0, 2, 4]]]]),
        "month": torch.tensor([[[-1, 2]]]),
        "day": torch.tensor([[[-1, 2]]]),
        "hour": torch.tensor([[[-1, -1]]]),
        "minute": torch.tensor([[[-1, -1]]]),
        "second": torch.tensor([[[-1, -1]]]),
        "quantiles": torch.tensor([[[[0.0] * 7, [0, -7.0, 0, 0, 0, 7.0, 0]]]]),
        "quantiles_mask": torch.tensor([[[[False] * 7, [False, True, False, False, False, True, False]]]]),
        "quantile_scale": torch.ones(1, 2),
    }
    component, temporal_quantile = module.temporal_losses(
        outputs["temporal"], temporal_targets, torch.tensor([[[False, True]]]),
    )
    total = median + quantile + component + temporal_quantile
    assert torch.isfinite(total)
    total.backward()
    assert vectors.grad is not None and torch.isfinite(vectors.grad).all()
    assert outputs["temporal"]["year_digit_logits"].shape == (1, 1, 2, 4, 10)


def _decoder():
    return StructuringDecoder(make_config(structuring_config=asdict(StructuringHeadConfig(
        typed_fields=True, typed_quantity_head="regression",
        typed_conformal={"date": 2.0},
    ))))


def _outputs(fields=2, mentions=1):
    exp_probs = torch.zeros(1, 1, fields, 1, tr.NUM_EXP_CLASSES)
    exp_probs[..., tr.exponent_index(9)] = 1.0
    month = torch.zeros(1, 1, fields, 12)
    month[..., 2] = 1.0
    day = torch.zeros(1, 1, fields, 31)
    day[..., 30] = 0.6  # day 31 does not exist in April
    day[..., 29] = 0.4
    month[..., 2] = 0.0
    month[..., 3] = 1.0  # April
    return {
        "family": torch.tensor([[tr.FAMILY_NUMBER, tr.FAMILY_DATE]]),
        "space": torch.zeros(1, fields, dtype=torch.long),
        "scale": torch.ones(1, fields),
        "field_mask": torch.ones(1, fields, dtype=torch.bool),
        "presence_logit": torch.full((1, 1, fields), 5.0),
        "number_negative_prob": torch.zeros(1, 1, fields, 1),
        "number_exp_probs": exp_probs,
        "number_digits": torch.full((1, 1, fields, 1), 0.633468),
        "number_z": torch.full((1, 1, fields, 1), 9.633468),
        "number_bounded_median": torch.zeros(1, 1, fields, 1),
        "number_offsets": torch.tensor([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]).expand(1, 1, fields, 1, 7),
        "temporal_offsets": torch.tensor([-6.0, -4.0, -1.0, 0.0, 1.0, 4.0, 6.0]).expand(1, 1, fields, 7),
        "era_probs": torch.tensor([0.9, 0.1]).expand(1, 1, fields, 2),
        "year_digit_probs": torch.stack([
            torch.nn.functional.one_hot(torch.tensor(digit), 10).float() for digit in (2, 0, 2, 4)
        ]).expand(1, 1, fields, 4, 10),
        "month_probs": month,
        "day_probs": day,
        "hour_probs": torch.zeros(1, 1, fields, 24),
        "minute_probs": torch.zeros(1, 1, fields, 60),
        "second_probs": torch.zeros(1, 1, fields, 60),
    }


@pytest.mark.parametrize("details", [False, True])
def test_decoder_reads_regression_outputs(details):
    decoder = _decoder()
    item = StructuringItemMapping(
        field_class_to_id=BaseClassMapping(class_to_id={"revenue": 0, "delivery": 1}),
        field_types={
            "revenue": {"$type": "number", "$stats": ["value", "p10", "p90"]},
            "delivery": {"$type": "date", "$stats": ["value", "p10", "p90"]},
        },
    )
    entries = [{"anchor_index": 0, "fields": [], "presence_is_reliable": True}]
    output = SimpleNamespace(structuring_typed_regression=_outputs())
    entries = decoder._attach_typed_fields(
        entries, output, group_idx=0, text_idx=0, struct_item=item, reference_date=None,
        texts=None, span_idx=None, objectness_logits=None, details=details,
    )
    values = {field["field"]: field["value"] for field in entries[0]["fields"]}
    revenue = values["revenue"]["value"] if details else values["revenue"]
    delivery = values["delivery"]["value"] if details else values["delivery"]
    assert revenue["value"] == pytest.approx(4.3e9, rel=1e-4)
    assert revenue["p10"] < revenue["value"] < revenue["p90"]
    # Year digits 2-0-2-4 (CE); April has no 31st, so the 30th wins.
    assert delivery["value"] == "2024-04-30"
    # Offsets of +/-4 days widened by the conformal 2 days, rounded outwards.
    assert delivery["p10"] == "2024-04-24" and delivery["p90"] == "2024-05-06"
    if details:
        assert set(values["revenue"]) == {"value", "p_null", "confidence", "guess"}


def test_bce_years_decode_to_expanded_iso():
    decoder = _decoder()
    outputs = _outputs()
    outputs["era_probs"] = torch.tensor([0.1, 0.9]).expand(1, 1, 2, 2)
    outputs["year_digit_probs"] = torch.stack([
        torch.nn.functional.one_hot(torch.tensor(digit), 10).float() for digit in (0, 0, 4, 4)
    ]).expand(1, 1, 2, 4, 10)
    month = torch.zeros(1, 1, 2, 12)
    month[..., 2] = 1.0
    day = torch.zeros(1, 1, 2, 31)
    day[..., 14] = 1.0
    outputs["month_probs"], outputs["day_probs"] = month, day
    item = StructuringItemMapping(
        field_class_to_id=BaseClassMapping(class_to_id={"revenue": 0, "event": 1}),
        field_types={"revenue": {"$type": "number"}, "event": {"$type": "date"}},
    )
    entries = [{"anchor_index": 0, "fields": [], "presence_is_reliable": True}]
    entries = decoder._attach_typed_fields(
        entries, SimpleNamespace(structuring_typed_regression=outputs), group_idx=0,
        text_idx=0, struct_item=item, reference_date=None, texts=None, span_idx=None,
        objectness_logits=None,
    )
    values = {field["field"]: field["value"] for field in entries[0]["fields"]}
    assert values["event"] == "-0043-03-15"  # the Ides of March, 44 BCE
