"""Tests for Embedding decoder."""

import pytest
import torch
from dataclasses import dataclass
from typing import Optional

from gliformer.tasks.embedding.decoder import EmbeddingDecoder
from tests.conftest import make_config


@dataclass
class FakeModelOutput:
    embedding_logits: Optional[torch.Tensor] = None


@pytest.fixture
def decoder():
    config = make_config()
    return EmbeddingDecoder.from_config(config)


class TestEmbeddingDecoder:
    def test_empty(self, decoder):
        out = FakeModelOutput()
        assert decoder.decode(out) == []

    def test_basic(self, decoder):
        logits = torch.tensor([0.9, 0.1, 0.5])
        out = FakeModelOutput(embedding_logits=logits)
        result = decoder.decode(out)
        assert len(result) == 3
        assert abs(result[0] - 0.9) < 1e-5
        assert abs(result[1] - 0.1) < 1e-5

    def test_nested_list(self, decoder):
        logits = torch.tensor([[0.8], [0.2]])
        out = FakeModelOutput(embedding_logits=logits)
        result = decoder.decode(out)
        assert len(result) == 2
        assert isinstance(result[0], list)