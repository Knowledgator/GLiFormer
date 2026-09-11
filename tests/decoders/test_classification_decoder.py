"""Tests for Classification decoder."""

import pytest
import torch
from dataclasses import dataclass
from typing import Optional

from gliformer.tasks.classification.decoder import ClassificationDecoder
from tests.conftest import make_config


@dataclass
class FakeModelOutput:
    cat_logits: Optional[torch.Tensor] = None
    cat_batch_origin: Optional[torch.Tensor] = None
    batch_size: Optional[int] = None

    def __post_init__(self):
        if self.cat_batch_origin is None and self.batch_size is None and self.cat_logits is not None:
            BN = self.cat_logits.shape[0]
            self.cat_batch_origin = torch.arange(BN)
            self.batch_size = BN


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
        assert len(result) == 1  # 1 batch item
        assert len(result[0]) == 1  # 1 group
        names = {p["class_name"] for p in result[0][0]}
        assert "0" in names
        assert "2" in names
        assert "1" not in names

    def test_all_below_threshold(self, decoder):
        logits = torch.full((1, 3), -10.0)
        out = FakeModelOutput(cat_logits=logits)
        result = decoder.decode(out)
        assert result[0][0] == []

    def test_threshold_override(self, decoder):
        logits = torch.tensor([[1.0, 0.5]])  # sigmoid≈0.73, 0.62
        out = FakeModelOutput(cat_logits=logits)
        result = decoder.decode(out, threshold=0.7)
        names = {p["class_name"] for p in result[0][0]}
        assert "0" in names
        assert "1" not in names

    def test_multiple_groups(self, decoder):
        logits = torch.tensor([[5.0, -5.0], [-5.0, 5.0]])
        out = FakeModelOutput(cat_logits=logits)
        result = decoder.decode(out)
        assert len(result) == 2  # 2 batch items (BN=B=2)
        assert result[0][0][0]["class_name"] == "0"
        assert result[1][0][0]["class_name"] == "1"

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
        score = result[0][0][0]["score"]
        assert 0.0 < score < 1.0
        assert abs(score - torch.sigmoid(torch.tensor(2.0)).item()) < 1e-5


class TestClassificationDecoderSingleLabel:
    def test_single_label_picks_best(self, decoder):
        logits = torch.tensor([[2.0, 5.0, 1.0]])
        out = FakeModelOutput(cat_logits=logits)
        result = decoder.decode(out, multi_label=False)
        assert len(result) == 1  # 1 batch item
        assert len(result[0]) == 1  # 1 group
        assert len(result[0][0]) == 1  # 1 prediction
        assert result[0][0][0]["class_name"] == "1"

    def test_single_label_all_below_threshold(self, decoder):
        logits = torch.full((1, 3), -10.0)
        out = FakeModelOutput(cat_logits=logits)
        result = decoder.decode(out, multi_label=False)
        assert result[0][0] == []

    def test_single_label_vs_multi_label(self, decoder):
        logits = torch.tensor([[3.0, 4.0]])
        out = FakeModelOutput(cat_logits=logits)
        multi = decoder.decode(out, multi_label=True)
        single = decoder.decode(out, multi_label=False)
        assert len(multi[0][0]) == 2
        assert len(single[0][0]) == 1
        assert single[0][0][0]["class_name"] == "1"

    def test_single_label_multiple_groups(self, decoder):
        logits = torch.tensor([[5.0, 3.0], [1.0, 6.0]])
        out = FakeModelOutput(cat_logits=logits)
        result = decoder.decode(out, multi_label=False)
        assert len(result) == 2  # 2 batch items (BN=B=2)
        assert result[0][0][0]["class_name"] == "0"
        assert result[1][0][0]["class_name"] == "1"
