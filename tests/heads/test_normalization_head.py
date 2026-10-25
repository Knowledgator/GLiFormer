"""Normalization head: span encoding, calendar losses, span collection, decoding."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from gliformer.config import StructuringHeadConfig
from gliformer.tasks import SharedRepresentations, TaskHeadOutput
from gliformer.tasks.normalization.decoder import NormalizationDecoder
from gliformer.tasks.normalization.model import (
    NormalizationHead,
    component_targets,
    year_digit_targets,
    year_targets,
)
from gliformer.tasks.normalization.values import IGNORE, NULL
from tests.heads.conftest import D, make_config

B, W, T = 2, 10, 14


def _config(**normalization):
    return make_config(
        structuring_config=asdict(StructuringHeadConfig(num_fixed_slots=2)),
        normalization_config=normalization,
    )


def _head(**normalization):
    return NormalizationHead.from_config(_config(**normalization))


def _shared():
    return SharedRepresentations(
        token_embeds=torch.randn(B, T, D),
        input_ids=torch.ones(B, T, dtype=torch.long),
        attention_mask=torch.ones(B, T, dtype=torch.long),
        words_embedding=torch.randn(B, W, D),
        mask=torch.ones(B, W, dtype=torch.long),
        prompts_embedding=torch.randn(B, 3, D),
        prompts_embedding_mask=torch.ones(B, 3, dtype=torch.long),
    )


def _words_mask():
    # Subwords 2..13 map to words 1..10 (1-based); prompt tokens are 0.
    mask = torch.zeros(B, T, dtype=torch.long)
    mask[:, 2:12] = torch.arange(1, 11)
    mask[:, 12:] = 10
    return mask


def _labels():
    def codes(values):
        return torch.tensor(values, dtype=torch.long)

    return {
        "normalization_span_idx": torch.tensor([[[1, 3], [5, 5]], [[0, 1], [0, 0]]]),
        "normalization_span_mask": torch.tensor([[True, True], [True, False]]),
        "normalization_number_value": torch.tensor([[0.0, 250.0], [0.0, 0.0]], dtype=torch.float64),
        "normalization_number_mask": torch.tensor([[False, True], [False, False]]),
        "normalization_temporal_family": torch.tensor([[1, -1], [2, -1]]),
        "normalization_year": codes([[2024, IGNORE], [IGNORE, IGNORE]]),
        "normalization_month": codes([[5, IGNORE], [IGNORE, IGNORE]]),
        "normalization_day": codes([[NULL, IGNORE], [IGNORE, IGNORE]]),
        "normalization_hour": codes([[IGNORE, IGNORE], [17, IGNORE]]),
        "normalization_minute": codes([[IGNORE, IGNORE], [30, IGNORE]]),
        "normalization_second": codes([[IGNORE, IGNORE], [NULL, IGNORE]]),
    }


@pytest.mark.parametrize("encoder", ["pooled", "token_attention"])
def test_supervised_temporal_loss_backpropagates(encoder):
    head = _head(span_encoder=encoder).train()
    out = head(_shared(), {}, **_labels(), span_token_words=_words_mask())
    assert torch.isfinite(out.loss) and torch.isfinite(out.extra["temporal_loss"])
    out.loss.backward()
    grads = [p.grad for p in head.temporal.parameters()]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
    assert out.extra["temporal"]["year"].shape == (B, 2, 400 + 2)


def test_token_attention_needs_subword_word_ids():
    head = _head(span_encoder="token_attention")
    with pytest.raises(ValueError, match="token_attention"):
        head(_shared(), {}, **_labels())


def test_batch_without_targets_keeps_parameters_attached():
    head = _head().train()
    labels = _labels()
    for name in ("year", "month", "day", "hour", "minute", "second"):
        labels[f"normalization_{name}"].fill_(IGNORE)
    labels["normalization_number_mask"].fill_(False)
    out = head(_shared(), {}, **labels)
    assert out.loss is not None and out.loss.item() == 0.0
    out.loss.backward()
    assert all(p.grad is not None for p in head.parameters())


def test_target_codes_map_to_classes():
    codes = torch.tensor([5, NULL, IGNORE, 13])
    assert component_targets(codes, 13, 1).tolist() == [4, 12, IGNORE, IGNORE]
    years = torch.tensor([1850, 1700, NULL, IGNORE])
    assert year_targets(years, 1800, 2199).tolist() == [50, 400, 401, IGNORE]
    digits = year_digit_targets(torch.tensor([2024, NULL]))
    assert digits.tolist() == [[2, 0, 2, 4], [IGNORE] * 4]


def test_inference_collects_unique_spans_from_ner_and_structuring():
    head = _head(max_span_width=3).eval()
    structuring = TaskHeadOutput(extra={
        "entity_spans": torch.tensor([[[1, 2], [4, 4]], [[0, 0], [2, 9]]]),
        "entity_mask": torch.tensor([[True, True], [True, True]]),
        "batch_origin": torch.tensor([1, 0]),  # flat groups map to text rows
    })
    ner_logits = torch.full((1, W, 1, 3), -9.0)
    ner_logits[0, 4, 0] = torch.tensor([9.0, 9.0, 9.0])  # span (4, 4) of row 0
    ner = TaskHeadOutput(logits=ner_logits, extra={"batch_origin": torch.tensor([0])})
    with torch.no_grad():
        out = head(_shared(), {"structuring": structuring, "ner": ner}, threshold=0.5)
    spans = [
        [tuple(span) for span, keep in zip(idx.tolist(), mask.tolist()) if keep]
        for idx, mask in zip(out.extra["span_idx"], out.extra["span_mask"])
    ]
    # Row 0: (0, 0) from structuring, (4, 4) from NER; (2, 9) is too wide.
    assert spans == [[(0, 0), (4, 4)], [(1, 2), (4, 4)]]
    assert "temporal" in out.extra and "loss" not in out.extra


def _decoder_output(year_class, month, day, hour=None, minute=None):
    """Output with one span whose logits pick the given classes."""

    def one_hot(classes, index):
        logits = torch.full((1, 1, classes), -20.0)
        logits[0, 0, index] = 20.0
        return logits

    temporal = {
        "year": one_hot(402, year_class),
        "month": one_hot(13, month),
        "day": one_hot(32, day),
        "hour": one_hot(25, 24 if hour is None else hour),
        "minute": one_hot(61, 60 if minute is None else minute),
        "second": one_hot(61, 60),
        "year_digits": torch.zeros(1, 1, 4, 10),
    }
    return SimpleNamespace(
        normalization_span_idx=torch.tensor([[[2, 4]]]),
        normalization_span_mask=torch.tensor([[True]]),
        normalization_temporal=temporal,
    )


def test_decoder_reads_partial_and_full_values():
    decoder = NormalizationDecoder(_config())
    partial = _decoder_output(year_class=224, month=4, day=31)  # 2024, May, no day
    # Every family is decoded; callers read the one their field asks for.
    values = decoder.span_values(partial, 0, 2, 4)
    assert values["date"] == "2024-05" and values["datetime"] == "2024-05"
    clock = _decoder_output(year_class=401, month=12, day=31, hour=17, minute=30)
    values = decoder.decode(clock)[0][0]
    assert values["time"] == "17:30" and values["start"] == 2
    assert decoder.span_values(clock, 0, 1, 4) is None
    mapped = decoder.map_results(
        decoder.decode(clock), [0], [[0, 5, 10, 15, 20]], [[4, 9, 14, 19, 24]],
        ["aaaa bbbb cccc dddd eeee"], 1,
    )
    assert mapped[0][0]["text"] == "cccc dddd eeee" and mapped[0][0]["start"] == 10


def test_out_of_range_years_come_from_the_digit_head():
    decoder = NormalizationDecoder(_config())
    output = _decoder_output(year_class=400, month=2, day=0)  # out of range, March, 1st
    output.normalization_temporal["year_digits"][0, 0] = torch.nn.functional.one_hot(
        torch.tensor([1, 6, 5, 0]), 10,
    ).float() * 20
    assert decoder.span_values(output, 0, 2, 4)["date"] == "1650-03-01"


@pytest.mark.parametrize("bad", [
    {"span_encoder": "lstm"},
    {"span_sources": ["vision"]},
    {"temporal": {"year_range": None, "year_digits": False}},
    {"max_spans": 0},
])
def test_invalid_normalization_config(bad):
    with pytest.raises(ValueError):
        _config(**bad)


def test_normalization_needs_a_span_head():
    with pytest.raises(ValueError, match="needs a structuring_config or ner_config"):
        make_config(ner_config=None, default_ner_config=False, normalization_config={})
