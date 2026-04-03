"""Tests for Classification decoder."""

import pytest
import torch
from dataclasses import dataclass
from typing import Optional

from glinext.tasks.classification.decoder import ClassificationDecoder
from tests.conftest import make_config


@dataclass
class FakeModelOutput:
    cat_logits: Optional[torch.Tensor] = None
    cat_batch_origin: Optional[torch.Tensor] = None
    batch_size: Optional[int] = None


@pytest.fixture
def decoder():
    config = make_config()
    return ClassificationDecoder.from_config(config)


class TestClassificationDecoder:
    def test_empty_logits(self, decoder):
        out = FakeModelOutput()
        assert decoder.decode(out) == []

    def test_single_prediction(self, decoder):
        # 1 group, 3 classes; class 0 strong, class 1 weak, class 2 strong
        logits = torch.tensor([[5.0, -5.0, 3.0]])
        out = FakeModelOutput(cat_logits=logits)
        result = decoder.decode(out)
        assert len(result) == 1
        ids = {p["class_id"] for p in result[0]}
        assert 0 in ids
        assert 2 in ids
        assert 1 not in ids

    def test_all_below_threshold(self, decoder):
        logits = torch.full((1, 3), -10.0)
        out = FakeModelOutput(cat_logits=logits)
        result = decoder.decode(out)
        assert result[0] == []

    def test_threshold_override(self, decoder):
        logits = torch.tensor([[1.0, 0.5]])  # sigmoid≈0.73, 0.62
        out = FakeModelOutput(cat_logits=logits)
        result = decoder.decode(out, threshold=0.7)
        ids = {p["class_id"] for p in result[0]}
        assert 0 in ids
        assert 1 not in ids

    def test_multiple_groups(self, decoder):
        logits = torch.tensor([[5.0, -5.0], [-5.0, 5.0]])
        out = FakeModelOutput(cat_logits=logits)
        result = decoder.decode(out)
        assert len(result) == 2
        assert result[0][0]["class_id"] == 0
        assert result[1][0]["class_id"] == 1

    def test_with_batch_origin(self, decoder):
        logits = torch.tensor([[5.0], [-5.0], [5.0]])
        out = FakeModelOutput(
            cat_logits=logits,
            cat_batch_origin=torch.tensor([0, 1, 1]),
            batch_size=2,
        )
        result = decoder.decode(out)
        assert len(result) == 2
        # batch 0: 1 group with 1 prediction
        assert len(result[0]) == 1
        assert len(result[0][0]) == 1
        # batch 1: 2 groups
        assert len(result[1]) == 2

    def test_score_is_probability(self, decoder):
        logits = torch.tensor([[2.0]])
        out = FakeModelOutput(cat_logits=logits)
        result = decoder.decode(out)
        score = result[0][0]["score"]
        assert 0.0 < score < 1.0
        assert abs(score - torch.sigmoid(torch.tensor(2.0)).item()) < 1e-5