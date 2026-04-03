"""Tests for Count decoder."""

import pytest
import torch
from dataclasses import dataclass
from typing import Optional

from glinext.tasks.count.decoder import CountDecoder
from glinext.config import CountHeadConfig
from tests.conftest import make_config


@dataclass
class FakeModelOutput:
    count_logits: Optional[torch.Tensor] = None
    count_batch_origin: Optional[torch.Tensor] = None
    batch_size: Optional[int] = None


def _make_decoder(mode="regression"):
    from dataclasses import asdict
    config = make_config(count_config=asdict(CountHeadConfig(mode=mode)))
    return CountDecoder.from_config(config)


class TestCountDecoder:
    def test_empty(self):
        decoder = _make_decoder()
        out = FakeModelOutput()
        assert decoder.decode(out) == []

    def test_regression_mode(self):
        decoder = _make_decoder("regression")
        logits = torch.tensor([[2.7], [0.3], [-0.5]])
        out = FakeModelOutput(count_logits=logits)
        result = decoder.decode(out)
        assert result == [3, 0, 0]  # round then clamp min=0

    def test_classification_mode(self):
        decoder = _make_decoder("classification")
        logits = torch.tensor([[0.1, 5.0, 0.2], [3.0, 0.1, 0.1]])
        out = FakeModelOutput(count_logits=logits)
        result = decoder.decode(out)
        assert result == [1, 0]  # argmax

    def test_with_batch_origin(self):
        decoder = _make_decoder("regression")
        logits = torch.tensor([[1.0], [2.0], [3.0]])
        out = FakeModelOutput(
            count_logits=logits,
            count_batch_origin=torch.tensor([0, 0, 1]),
            batch_size=2,
        )
        result = decoder.decode(out)
        assert len(result) == 2
        assert len(result[0]) == 2  # groups 0,1 for batch item 0
        assert len(result[1]) == 1

    def test_negative_clamped_to_zero(self):
        decoder = _make_decoder("regression")
        logits = torch.tensor([[-5.0]])
        out = FakeModelOutput(count_logits=logits)
        result = decoder.decode(out)
        assert result == [0]