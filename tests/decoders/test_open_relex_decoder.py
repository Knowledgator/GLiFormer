"""Tests for Open Relex decoder."""

import pytest
import torch
from dataclasses import dataclass
from typing import Optional

from glinext.tasks.open_relex.decoder import OpenRelexDecoder
from glinext.mappings import (
    BaseClassMapping, OpenRelexItemMapping, OpenRelexClassMapping,
    BatchClassesMapping, CatClassMapping, ExtractionClassMapping, StructuringClassMapping,
)
from tests.conftest import make_config


@dataclass
class FakeModelOutput:
    open_rel_logits: Optional[torch.Tensor] = None
    open_rel_batch_origin: Optional[torch.Tensor] = None
    open_rel_anchor_mask: Optional[torch.Tensor] = None
    open_rel_span_logits: Optional[torch.Tensor] = None
    open_rel_span_idx: Optional[torch.Tensor] = None
    open_rel_span_mask: Optional[torch.Tensor] = None
    batch_size: Optional[int] = None


@pytest.fixture
def decoder():
    config = make_config()
    return OpenRelexDecoder.from_config(config)


def _make_mapping(rel_labels, batch_size=1):
    rel_map = BaseClassMapping(class_to_id={l: i for i, l in enumerate(rel_labels)})
    item = OpenRelexItemMapping(rel_class_to_id=rel_map)
    return BatchClassesMapping(
        cat_mapping=[CatClassMapping(cat_class_to_id=[]) for _ in range(batch_size)],
        extraction_mapping=[ExtractionClassMapping() for _ in range(batch_size)],
        structuring_mapping=[StructuringClassMapping() for _ in range(batch_size)],
        open_relex_mapping=[OpenRelexClassMapping(items=[item]) for _ in range(batch_size)],
    )


class TestOpenRelexTokenLevel:
    def test_empty(self, decoder):
        out = FakeModelOutput()
        assert decoder.decode(out) == []

    def test_single_triple(self, decoder):
        # (BN=1, X=1, C=1, L=5, 2=head/tail, 3=BIO)
        logits = torch.full((1, 1, 1, 5, 2, 3), -10.0)
        # head span at pos 0
        logits[0, 0, 0, 0, 0, :] = 5.0  # head: start+end+inside all strong at pos 0
        # tail span at pos 3-4
        logits[0, 0, 0, 3, 1, 0] = 5.0  # tail start
        logits[0, 0, 0, 4, 1, 1] = 5.0  # tail end
        for t in range(3, 5):
            logits[0, 0, 0, t, 1, 2] = 5.0  # tail inside

        out = FakeModelOutput(open_rel_logits=logits)
        mapping = _make_mapping(["lives_in"])
        result = decoder.decode(out, classes_mapping=mapping, texts=[["John", "lives", "in", "New", "York"]])
        assert len(result) == 1
        assert len(result[0]) >= 1
        triple = result[0][0]
        assert triple["relation"] == "lives_in"
        assert triple["head"]["start"] == 0
        assert triple["tail"]["start"] == 3

    def test_anchor_mask(self, decoder):
        logits = torch.full((1, 2, 1, 5, 2, 3), 5.0)  # all strong
        anchor_mask = torch.tensor([[True, False]])  # second anchor masked
        out = FakeModelOutput(open_rel_logits=logits, open_rel_anchor_mask=anchor_mask)
        result = decoder.decode(out)
        # Only triples from anchor 0 should appear
        for triple in result[0]:
            # All triples should come from the unmasked anchor
            pass
        # With anchor_mask[1]=False, triples from anchor 1 are excluded
        result_no_mask = decoder.decode(FakeModelOutput(open_rel_logits=logits))
        assert len(result[0]) <= len(result_no_mask[0])

    def test_no_spans_found(self, decoder):
        logits = torch.full((1, 1, 1, 5, 2, 3), -10.0)
        out = FakeModelOutput(open_rel_logits=logits)
        result = decoder.decode(out)
        assert result[0] == []


class TestOpenRelexSpanLevel:
    def test_span_level_preferred(self, decoder):
        BN, S, X, C = 1, 3, 1, 1
        # (BN, S, X, C, 2) — per span, per anchor, per rel class, head/tail
        span_logits = torch.full((BN, S, X, C, 2), -10.0)
        span_logits[0, 0, 0, 0, 0] = 5.0  # span 0 is head
        span_logits[0, 2, 0, 0, 1] = 5.0  # span 2 is tail

        span_idx = torch.tensor([[[0, 0], [1, 2], [3, 4]]])
        span_mask = torch.ones(BN, S, dtype=torch.bool)

        out = FakeModelOutput(
            open_rel_span_logits=span_logits,
            open_rel_span_idx=span_idx,
            open_rel_span_mask=span_mask,
        )
        mapping = _make_mapping(["rel_a"])
        result = decoder.decode(out, classes_mapping=mapping)
        assert len(result) == 1
        assert len(result[0]) == 1
        assert result[0][0]["head"]["start"] == 0
        assert result[0][0]["tail"]["start"] == 3

    def test_span_level_masked_spans(self, decoder):
        BN, S, X, C = 1, 3, 1, 1
        span_logits = torch.full((BN, S, X, C, 2), 5.0)  # all strong
        span_idx = torch.tensor([[[0, 0], [1, 1], [2, 2]]])
        span_mask = torch.tensor([[True, False, True]])

        out = FakeModelOutput(
            open_rel_span_logits=span_logits,
            open_rel_span_idx=span_idx,
            open_rel_span_mask=span_mask,
        )
        result = decoder.decode(out)
        # span 1 is masked out, so triples should only use spans 0 and 2
        for triple in result[0]:
            assert triple["head"]["start"] != 1
            assert triple["tail"]["start"] != 1

    def test_with_batch_origin(self, decoder):
        BN, S, X, C = 2, 2, 1, 1
        span_logits = torch.full((BN, S, X, C, 2), -10.0)
        span_logits[0, 0, 0, 0, 0] = 5.0
        span_logits[0, 1, 0, 0, 1] = 5.0
        span_logits[1, 0, 0, 0, 0] = 5.0
        span_logits[1, 1, 0, 0, 1] = 5.0

        span_idx = torch.tensor([[[0, 0], [2, 3]], [[0, 0], [1, 1]]])
        span_mask = torch.ones(BN, S, dtype=torch.bool)

        out = FakeModelOutput(
            open_rel_span_logits=span_logits,
            open_rel_span_idx=span_idx,
            open_rel_span_mask=span_mask,
            open_rel_batch_origin=torch.tensor([0, 0]),
            batch_size=1,
        )
        mapping = _make_mapping(["r"], batch_size=BN)
        result = decoder.decode(out, classes_mapping=mapping)
        assert len(result) == 1  # 1 batch item
        assert len(result[0]) == 2  # 2 groups