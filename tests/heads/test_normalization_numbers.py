"""Number heads of the normalization head."""

from types import SimpleNamespace

import pytest
import torch

from gliformer.config import NormalizationNumberConfig
from gliformer.tasks.normalization.decoder import NormalizationDecoder
from gliformer.tasks.normalization.numbers import (
    SIGN_NEGATIVE,
    SIGN_POSITIVE,
    SIGN_ZERO,
    build_number_heads,
    scientific_targets,
    sign_states,
)
from tests.heads.test_normalization_head import _config, _head, _labels, _shared

D = 64
VALUES = [0.0, 1.0, -2.5, 250.0, 1250.5, 4.3e9, -0.0042, 2019.0, 7.9e11, 12345.0, 3.14159, 900.0]


def _data():
    torch.manual_seed(0)
    vectors = torch.randn(1, len(VALUES), D)
    values = torch.tensor([VALUES], dtype=torch.float64)
    return vectors, values, torch.ones(1, len(VALUES), dtype=torch.bool)


def test_sign_states_and_scientific_targets():
    values = torch.tensor([-3.0, 0.0, 99999.6, 1e20, 0.000123456], dtype=torch.float64)
    assert sign_states(values).tolist() == [SIGN_NEGATIVE, SIGN_ZERO, SIGN_POSITIVE, SIGN_POSITIVE, SIGN_POSITIVE]
    mask = torch.tensor([True, True, True, True, False])
    exp_class, digits = scientific_targets(values, mask, (-10, 15), 5)
    assert exp_class.tolist() == [10, -100, 15, -100, -100]  # 1e20 is out of range
    assert digits[0].tolist() == [3, 0, 0, 0, 0]
    # 99999.6 rounds to 1.0000 x 10^5 (the carry moves into the exponent).
    assert digits[2].tolist() == [1, 0, 0, 0, 0]
    assert digits[1].tolist() == [-100] * 5


def _fit(kind, steps=400, **settings):
    vectors, values, mask = _data()
    cfg = NormalizationNumberConfig(heads=[kind], **({kind: settings} if settings else {}))
    head = build_number_heads(D, 0.0, cfg)[kind]
    optimizer = torch.optim.Adam(head.parameters(), lr=3e-3)
    first = None
    for _ in range(steps):
        loss, _stats = head.loss(head(vectors), values, mask)
        first = loss.item() if first is None else first
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    head.eval()
    with torch.no_grad():
        decoded, confidence = head.decode(head(vectors))
    return head, first, loss.item(), decoded[0], confidence[0], values[0]


def test_sci_digits_head_learns_exact_digits():
    _head_, first, last, decoded, confidence, values = _fit("sci_digits", apply_coefficient=False)
    assert last < first
    assert decoded.dtype == torch.float64 and confidence.shape == values.shape
    # Exact up to the 5 significant digits the head writes (3.14159 -> 3.1416).
    rounded = torch.tensor([float(f"{value:.5g}") for value in values.tolist()], dtype=torch.float64)
    exact = (decoded - rounded).abs() <= 1e-9 * rounded.abs().clamp(min=1.0)
    assert exact.float().mean() == 1.0, decoded.tolist()


def test_coefficient_scales_the_digits():
    head, *_rest = _fit("sci_digits", steps=50)
    assert head.use_coefficient and head.apply_coefficient
    features = torch.randn(3, head.digit_in.in_features)
    exponent = torch.tensor([10, 11, 12])
    digits = torch.tensor([[2, 5, 0, 0, 0]] * 3)
    coefficient = head._coefficient(features, exponent, digits)
    assert ((coefficient >= 10 ** -0.5) & (coefficient <= 10 ** 0.5)).all()


@pytest.mark.parametrize("kind,settings", [
    ("regression", {}),
    ("regression", {"transform": "log10"}),
    ("bins", {}),
    ("bins", {"transform": "log10", "range": [-6.0, 12.0], "count": 512}),
])
def test_regression_and_bins_heads_train_and_decode(kind, settings):
    _head_, first, last, decoded, confidence, values = _fit(kind, steps=800, **settings)
    assert last < first
    relative = (decoded - values).abs() / values.abs().clamp(min=1.0)
    # Most values land close; bins are limited by their resolution.
    assert (relative < 0.05).float().mean() >= 0.6, decoded.tolist()
    assert torch.isfinite(confidence).all()


def test_l1_regression_loss_decreases():
    _head_, first, last, *_rest = _fit("regression", steps=200, loss="l1")
    assert last < first


def test_heads_stay_fp32_and_bin_edges_float64():
    cfg = NormalizationNumberConfig(heads=["bins", "sci_digits", "regression"])
    heads = build_number_heads(D, 0.0, cfg).to(torch.bfloat16)
    assert all(p.dtype == torch.float32 for p in heads.parameters())
    assert heads["bins"].edges.dtype == torch.float64
    state = heads["bins"].state_dict()
    state["edges"] = state["edges"] + 1.0
    with pytest.raises(ValueError, match="bin edges differ"):
        heads["bins"].load_state_dict(state)


def test_normalization_head_trains_numbers_and_decodes_in_eval():
    head = _head(number={"heads": ["sci_digits", "bins"]}).train()
    out = head(_shared(), {}, **_labels())
    assert "number_sci_digits_loss" in out.extra and "number_bins_loss" in out.extra
    out.loss.backward()
    assert all(p.grad is not None for p in head.number_heads.parameters())
    assert "number" not in out.extra  # not decoded while training
    head.eval()
    with torch.no_grad():
        out = head(_shared(), {}, **_labels())
    assert set(out.extra["number"]) == {"sci_digits", "bins"}
    assert out.extra["number"]["bins"].shape == (2, 2)


def test_no_number_heads():
    head = _head(number={"heads": []}).eval()
    with torch.no_grad():
        out = head(_shared(), {}, **_labels())
    assert len(head.number_heads) == 0 and "number" not in out.extra


def test_decoder_reports_the_decode_with_head_and_all_heads():
    decoder = NormalizationDecoder(_config(number={"heads": ["bins", "sci_digits"], "decode_with": "sci_digits"}))
    output = SimpleNamespace(
        normalization_span_idx=torch.tensor([[[0, 1]]]),
        normalization_span_mask=torch.tensor([[True]]),
        normalization_temporal=None,
        normalization_number={
            "bins": torch.tensor([[247.2]], dtype=torch.float64),
            "sci_digits": torch.tensor([[250.0]], dtype=torch.float64),
        },
    )
    values = decoder.span_values(output, 0, 0, 1)
    assert values == {"number": 250, "numbers": {"bins": 247.2, "sci_digits": 250}}


@pytest.mark.parametrize("bad", [
    {"heads": ["histogram"]},
    {"heads": ["bins"], "decode_with": "regression"},
    {"bins": {"range": [5.0, 1.0]}},
    {"sci_digits": {"digits": 0}},
    {"regression": {"loss": "mse"}},
])
def test_invalid_number_config(bad):
    with pytest.raises(ValueError):
        NormalizationNumberConfig(**bad)


def _value_output():
    from tests.heads.test_normalization_head import _decoder_output

    output = _decoder_output(year_class=224, month=4, day=31)  # span (2, 4): "2024-05"
    output.normalization_number = {"bins": torch.tensor([[1234.567]], dtype=torch.float64)}
    return output


def test_field_value_follows_the_declared_type():
    from gliformer.processing.typed_fields import TypedFieldSpec

    decoder = NormalizationDecoder(_config())
    output = _value_output()
    value = decoder.field_value
    assert value(output, 0, 2, 4, TypedFieldSpec(type="number")) == 1234.567
    assert value(output, 0, 2, 4, TypedFieldSpec(type="number", integer=True)) == 1235
    assert value(output, 0, 2, 4, TypedFieldSpec(type="number", decimals=1)) == 1234.6
    assert value(output, 0, 2, 4, TypedFieldSpec(type="date")) == "2024-05"
    assert value(output, 0, 2, 4, TypedFieldSpec(type="list", items="date")) == "2024-05"
    # A full value is forced when partial values are not allowed.
    assert value(output, 0, 2, 4, TypedFieldSpec(type="date", partial=False)).count("-") == 2
    assert value(output, 0, 2, 4, TypedFieldSpec(type="text")) is None
    assert value(output, 0, 0, 4, TypedFieldSpec(type="number")) is None  # span not read


def test_structuring_decoder_takes_head_values_only_for_declared_types():
    from dataclasses import asdict

    from gliformer.config import StructuringHeadConfig
    from gliformer.processing.mappings import BaseClassMapping, StructuringItemMapping
    from gliformer.processing.structuring_decoder import StructuringDecoder
    from tests.heads.conftest import make_config

    config = make_config(
        structuring_config=asdict(StructuringHeadConfig(typed_fields=True)),
        normalization_config={},
    )
    decoder = StructuringDecoder(config)
    item = StructuringItemMapping(
        field_class_to_id=BaseClassMapping(class_to_id={"total": 0, "issued": 1, "note": 2, "count": 3}),
        field_types={"total": {"$type": "number"}, "issued": {"$type": "date"}},
    )
    entries = [{"anchor_index": 0, "fields": [
        {"field": "total", "start": 2, "end": 4, "text": "1,234.567"},
        {"field": "issued", "start": 2, "end": 4, "text": "May 2024"},
        {"field": "note", "start": 2, "end": 4, "text": "May 2024"},  # no declared type
        {"field": "count", "start": 5, "end": 5, "text": "12"},
    ]}]
    entries = decoder._attach_normalized_values(entries, _value_output(), text_idx=0, struct_item=item)
    fields = {field["field"]: field for field in entries[0]["fields"]}
    assert fields["total"]["value"] == 1234.567
    assert fields["issued"]["value"] == "2024-05"
    assert "value" not in fields["note"] and "value" not in fields["count"]


def test_value_spans_option_returns_value_with_original_text():
    from dataclasses import asdict

    from gliformer.config import StructuringHeadConfig
    from gliformer.processing._structuring_alignment import field_value_with_span
    from gliformer.processing.mappings import BaseClassMapping, StructuringItemMapping
    from gliformer.processing.structuring_decoder import StructuringDecoder
    from tests.heads.conftest import make_config

    decoder = StructuringDecoder(make_config(
        structuring_config=asdict(StructuringHeadConfig(typed_fields=True)), normalization_config={},
    ))
    item = StructuringItemMapping(
        field_class_to_id=BaseClassMapping(class_to_id={"total": 0}),
        field_types={"total": {"$type": "number"}},
    )
    entries = [{"anchor_index": 0, "fields": [{"field": "total", "start": 2, "end": 4, "text": "1 , 234.567"}]}]
    entries = decoder._attach_normalized_values(
        entries, _value_output(), text_idx=0, struct_item=item, value_spans=True,
    )
    field = entries[0]["fields"][0]
    assert field["value"] == {"value": 1234.567, "text": "1 , 234.567"}
    text = "Total is 1,234.567 USD"
    mapped = field_value_with_span(field, [0, 6, 9, 10, 12], [5, 8, 10, 11, 18], text)
    assert mapped == {"value": 1234.567, "text": "1,234.567", "start": 9, "end": 18}
