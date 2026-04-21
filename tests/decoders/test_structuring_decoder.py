"""Tests for Structuring decoder."""

import pytest
import torch
from dataclasses import dataclass
from typing import Optional

from glinext.tasks.structuring.decoder import StructuringDecoder
from glinext.processing.mappings import (
    BaseClassMapping, StructuringItemMapping, StructuringClassMapping,
    BatchClassesMapping, CatClassMapping, ExtractionClassMapping, OpenRelexClassMapping,
)
from tests.conftest import make_config


@dataclass
class FakeModelOutput:
    structuring_logits: Optional[torch.Tensor] = None
    structuring_batch_origin: Optional[torch.Tensor] = None
    structuring_anchor_mask: Optional[torch.Tensor] = None
    structuring_span_logits: Optional[torch.Tensor] = None
    structuring_span_idx: Optional[torch.Tensor] = None
    structuring_span_mask: Optional[torch.Tensor] = None
    batch_size: Optional[int] = None

    def __post_init__(self):
        if self.structuring_batch_origin is None and self.batch_size is None:
            logits = self.structuring_logits if self.structuring_logits is not None else self.structuring_span_logits
            if logits is not None:
                BN = logits.shape[0]
                self.structuring_batch_origin = torch.arange(BN)
                self.batch_size = BN


@pytest.fixture
def decoder():
    config = make_config()
    return StructuringDecoder.from_config(config)


def _make_field_mapping(fields, batch_size=1):
    # class_to_id uses 0-indexed values; get_reverse_mapping produces {0: name, 1: age, ...}
    field_map = BaseClassMapping(class_to_id={f: i for i, f in enumerate(fields)})
    item = StructuringItemMapping(field_class_to_id=field_map, name="schema")
    return BatchClassesMapping(
        cat_mapping=[CatClassMapping(cat_class_to_id=[]) for _ in range(batch_size)],
        extraction_mapping=[ExtractionClassMapping() for _ in range(batch_size)],
        structuring_mapping=[StructuringClassMapping(items=[item]) for _ in range(batch_size)],
        open_relex_mapping=[OpenRelexClassMapping() for _ in range(batch_size)],
    )


def _bio_structuring_logits(BN, X, L, C, spans):
    """Build (BN, X, L, C, 3) logits.

    spans: list of (bn, x, start, end, cls).
    """
    logits = torch.full((BN, X, L, C, 3), -10.0)
    for bn, x, start, end, cls in spans:
        logits[bn, x, start, cls, 0] = 5.0
        logits[bn, x, end, cls, 1] = 5.0
        for t in range(start, end + 1):
            logits[bn, x, t, cls, 2] = 5.0
    return logits


class TestStructuringTokenLevel:
    def test_empty(self, decoder):
        out = FakeModelOutput()
        assert decoder.decode(out) == []

    def test_single_instance_single_field(self, decoder):
        # 1 group, 1 instance, 5 tokens, 2 fields
        logits = _bio_structuring_logits(1, 1, 5, 2, [(0, 0, 0, 0, 0)])
        out = FakeModelOutput(structuring_logits=logits)
        mapping = _make_field_mapping(["name", "age"])
        result = decoder.decode(out, classes_mapping=mapping)
        assert len(result) == 1  # 1 batch item
        assert len(result[0]) == 1  # 1 group
        assert len(result[0][0]) == 1  # 1 instance
        assert result[0][0][0][0]["field"] == "name"

    def test_multiple_instances(self, decoder):
        logits = _bio_structuring_logits(1, 2, 5, 1, [
            (0, 0, 0, 0, 0),  # instance 0
            (0, 1, 3, 4, 0),  # instance 1
        ])
        anchor_mask = torch.ones(1, 2, dtype=torch.bool)
        out = FakeModelOutput(structuring_logits=logits, structuring_anchor_mask=anchor_mask)
        mapping = _make_field_mapping(["value"])
        result = decoder.decode(out, classes_mapping=mapping)
        assert len(result[0][0]) == 2  # 2 instances in group 0

    def test_anchor_mask(self, decoder):
        logits = _bio_structuring_logits(1, 2, 5, 1, [
            (0, 0, 0, 0, 0),
            (0, 1, 2, 2, 0),
        ])
        anchor_mask = torch.tensor([[True, False]])
        out = FakeModelOutput(structuring_logits=logits, structuring_anchor_mask=anchor_mask)
        mapping = _make_field_mapping(["f"])
        result = decoder.decode(out, classes_mapping=mapping)
        assert len(result[0][0]) == 1  # only instance 0

    def test_with_texts(self, decoder):
        logits = _bio_structuring_logits(1, 1, 5, 1, [(0, 0, 0, 0, 0)])
        out = FakeModelOutput(structuring_logits=logits)
        mapping = _make_field_mapping(["name"])
        result = decoder.decode(
            out, classes_mapping=mapping, texts=[["John", "lives", "in", "New", "York"]],
        )
        assert result[0][0][0][0]["text"] == "John"

    def test_no_spans_returns_empty_instances(self, decoder):
        logits = torch.full((1, 1, 5, 2, 3), -10.0)
        out = FakeModelOutput(structuring_logits=logits)
        result = decoder.decode(out)
        assert result[0][0] == []


class TestStructuringSpanLevel:
    def test_span_level_basic(self, decoder):
        BN, X, S, C = 1, 1, 3, 2
        span_logits = torch.full((BN, X, S, C), -10.0)
        span_logits[0, 0, 0, 0] = 5.0  # span 0, field 0
        span_logits[0, 0, 2, 1] = 5.0  # span 2, field 1

        span_idx = torch.tensor([[[0, 0], [1, 2], [3, 4]]])
        span_mask = torch.ones(BN, S, dtype=torch.bool)
        anchor_mask = torch.ones(BN, X, dtype=torch.bool)

        out = FakeModelOutput(
            structuring_span_logits=span_logits,
            structuring_span_idx=span_idx,
            structuring_span_mask=span_mask,
            structuring_anchor_mask=anchor_mask,
        )
        mapping = _make_field_mapping(["name", "location"])
        result = decoder.decode(out, classes_mapping=mapping)
        assert len(result) == 1  # 1 batch item
        assert len(result[0]) == 1  # 1 group
        assert len(result[0][0]) == 1  # 1 instance
        fields = {f["field"] for f in result[0][0][0]}
        assert fields == {"name", "location"}

    def test_span_level_masked(self, decoder):
        BN, X, S, C = 1, 1, 3, 1
        span_logits = torch.full((BN, X, S, C), 5.0)
        span_idx = torch.tensor([[[0, 0], [1, 1], [2, 2]]])
        span_mask = torch.tensor([[True, False, True]])
        anchor_mask = torch.ones(BN, X, dtype=torch.bool)

        out = FakeModelOutput(
            structuring_span_logits=span_logits,
            structuring_span_idx=span_idx,
            structuring_span_mask=span_mask,
            structuring_anchor_mask=anchor_mask,
        )
        mapping = _make_field_mapping(["f"])
        result = decoder.decode(out, classes_mapping=mapping)
        # Only 2 valid spans (0 and 2), but greedy search may reduce overlapping
        instance_fields = result[0][0][0]
        starts = {f["start"] for f in instance_fields}
        assert 1 not in starts

    def test_with_batch_origin(self, decoder):
        BN, X, S, C = 2, 1, 2, 1
        span_logits = torch.full((BN, X, S, C), -10.0)
        span_logits[0, 0, 0, 0] = 5.0
        span_logits[1, 0, 1, 0] = 5.0

        span_idx = torch.tensor([[[0, 0], [1, 1]], [[0, 0], [2, 3]]])
        span_mask = torch.ones(BN, S, dtype=torch.bool)
        anchor_mask = torch.ones(BN, X, dtype=torch.bool)

        out = FakeModelOutput(
            structuring_span_logits=span_logits,
            structuring_span_idx=span_idx,
            structuring_span_mask=span_mask,
            structuring_anchor_mask=anchor_mask,
            structuring_batch_origin=torch.tensor([0, 1]),
            batch_size=2,
        )
        mapping = _make_field_mapping(["f"], batch_size=BN)
        result = decoder.decode(out, classes_mapping=mapping)
        assert len(result) == 2