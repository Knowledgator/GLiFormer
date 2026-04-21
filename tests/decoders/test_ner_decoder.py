"""Tests for NER decoder."""

import pytest
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from glinext.tasks.ner.decoder import NERDecoder
from glinext.tasks.span_decoder import Span
from tests.conftest import make_config


@dataclass
class FakeModelOutput:
    """Minimal model output for NER decoder tests."""
    ner_logits: Optional[torch.Tensor] = None
    span_logits: Optional[torch.Tensor] = None
    span_idx: Optional[torch.Tensor] = None
    span_mask: Optional[torch.Tensor] = None
    ner_batch_origin: Optional[torch.Tensor] = None
    batch_size: Optional[int] = None

    def __post_init__(self):
        # Auto-fill batch_origin/batch_size for BN=B (1:1) when not provided
        if self.ner_batch_origin is None and self.batch_size is None:
            logits = self.ner_logits if self.ner_logits is not None else self.span_logits
            if logits is not None:
                BN = logits.shape[0]
                self.ner_batch_origin = torch.arange(BN)
                self.batch_size = BN


@pytest.fixture
def decoder():
    config = make_config()
    return NERDecoder.from_config(config)


def _bio_logits(BN, L, C, spans):
    """Build (BN, L, C, 3) logits with strong signals at given span positions.

    spans: list of (bn, start, end, class_idx) — 0-indexed class.
    """
    logits = torch.full((BN, L, C, 3), -10.0)
    for bn, start, end, cls in spans:
        logits[bn, start, cls, 0] = 5.0
        logits[bn, end, cls, 1] = 5.0
        for t in range(start, end + 1):
            logits[bn, t, cls, 2] = 5.0
    return logits


class TestNERDecoderTokenLevel:
    def test_empty_logits(self, decoder):
        out = FakeModelOutput()
        assert decoder.decode(out) == []

    def test_single_entity(self, decoder):
        logits = _bio_logits(1, 5, 2, [(0, 0, 0, 0)])
        out = FakeModelOutput(ner_logits=logits)
        id_to_classes = {0: "person", 1: "location"}
        result = decoder.decode(out, classes_mapping=id_to_classes)
        # result[batch_item][group][span]
        assert len(result) == 1
        assert len(result[0]) == 1  # 1 group
        assert len(result[0][0]) == 1  # 1 span in group
        assert result[0][0][0].entity_type == "person"

    def test_multiple_entities(self, decoder):
        logits = _bio_logits(1, 6, 2, [(0, 0, 0, 0), (0, 3, 4, 1)])
        out = FakeModelOutput(ner_logits=logits)
        id_to_classes = {0: "person", 1: "location"}
        result = decoder.decode(out, classes_mapping=id_to_classes)
        assert len(result[0][0]) == 2  # 2 spans in group 0

    def test_batch_multiple_groups(self, decoder):
        logits = _bio_logits(2, 5, 1, [(0, 0, 0, 0), (1, 2, 3, 0)])
        out = FakeModelOutput(ner_logits=logits)
        id_to_classes = {0: "entity"}
        result = decoder.decode(out, classes_mapping=id_to_classes)
        # BN=B=2, 1:1 mapping: 2 batch items, 1 group each
        assert len(result) == 2
        assert len(result[0]) == 1  # 1 group for batch item 0
        assert len(result[1]) == 1  # 1 group for batch item 1

    def test_with_batch_origin(self, decoder):
        logits = _bio_logits(3, 5, 1, [(0, 0, 0, 0), (1, 1, 1, 0), (2, 2, 2, 0)])
        batch_origin = torch.tensor([0, 0, 1])
        out = FakeModelOutput(
            ner_logits=logits,
            ner_batch_origin=batch_origin,
            batch_size=2,
        )
        result = decoder.decode(out, classes_mapping={0: "A"})
        # 2 batch items: batch 0 has groups 0,1; batch 1 has group 2
        assert len(result) == 2
        assert len(result[0]) == 2  # two groups for batch item 0
        assert len(result[1]) == 1

    def test_threshold_override(self, decoder):
        logits = torch.full((1, 5, 1, 3), 0.0)  # sigmoid(0)=0.5 — borderline
        logits[0, 0, 0, :] = 0.5  # sigmoid≈0.62
        out = FakeModelOutput(ner_logits=logits)
        # High threshold should filter
        result = decoder.decode(out, classes_mapping={0: "A"}, threshold=0.99)
        assert len(result[0][0]) == 0


class TestNERDecoderSpanLevel:
    def test_span_level_preferred(self, decoder):
        B, S, C = 1, 2, 1
        span_logits = torch.full((B, S, C), -10.0)
        span_logits[0, 0, 0] = 5.0
        span_idx = torch.tensor([[[0, 1], [3, 4]]])
        span_mask = torch.ones(B, S, dtype=torch.bool)

        # Also provide ner_logits (should be ignored when span_logits present)
        ner_logits = torch.full((B, 5, C, 3), -10.0)

        out = FakeModelOutput(
            ner_logits=ner_logits,
            span_logits=span_logits,
            span_idx=span_idx,
            span_mask=span_mask,
        )
        result = decoder.decode(out, classes_mapping={0: "person"})
        assert len(result) == 1  # 1 batch item
        assert len(result[0]) == 1  # 1 group
        assert len(result[0][0]) == 1  # 1 span
        assert result[0][0][0].entity_type == "person"

    def test_span_level_with_batch_origin(self, decoder):
        B, S, C = 2, 2, 1
        span_logits = torch.full((B, S, C), -10.0)
        span_logits[0, 0, 0] = 5.0
        span_logits[1, 1, 0] = 5.0
        span_idx = torch.tensor([[[0, 1], [3, 4]], [[0, 1], [2, 3]]])
        span_mask = torch.ones(B, S, dtype=torch.bool)

        out = FakeModelOutput(
            span_logits=span_logits,
            span_idx=span_idx,
            span_mask=span_mask,
            ner_batch_origin=torch.tensor([0, 0]),
            batch_size=1,
        )
        result = decoder.decode(out, classes_mapping={0: "A"})
        assert len(result) == 1  # 1 batch item
        assert len(result[0]) == 2  # 2 groups