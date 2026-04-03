"""Tests for Joint Relex decoder."""

import pytest
import torch
from dataclasses import dataclass
from typing import Optional

from glinext.tasks.joint_relex.decoder import JointRelexDecoder
from glinext.tasks.span_decoder import Span
from glinext.mappings import (
    BaseClassMapping, ExtractionItemMapping, ExtractionClassMapping,
    BatchClassesMapping, CatClassMapping, StructuringClassMapping, OpenRelexClassMapping,
)
from tests.conftest import make_config


@dataclass
class FakeModelOutput:
    ner_logits: Optional[torch.Tensor] = None
    span_logits: Optional[torch.Tensor] = None
    span_idx: Optional[torch.Tensor] = None
    span_mask: Optional[torch.Tensor] = None
    ner_batch_origin: Optional[torch.Tensor] = None
    joint_rel_logits: Optional[torch.Tensor] = None
    joint_rel_idx: Optional[torch.Tensor] = None
    joint_rel_mask: Optional[torch.Tensor] = None
    joint_rel_batch_origin: Optional[torch.Tensor] = None
    batch_size: Optional[int] = None


def _bio_logits(BN, L, C, spans):
    logits = torch.full((BN, L, C, 3), -10.0)
    for bn, start, end, cls in spans:
        logits[bn, start, cls, 0] = 5.0
        logits[bn, end, cls, 1] = 5.0
        for t in range(start, end + 1):
            logits[bn, t, cls, 2] = 5.0
    return logits


@pytest.fixture
def decoder():
    config = make_config()
    return JointRelexDecoder.from_config(config)


def _make_classes_mapping():
    ner_map = BaseClassMapping(class_to_id={"person": 0, "location": 1}, name="entities")
    rel_map = BaseClassMapping(class_to_id={"lives_in": 0}, name="entities")
    item = ExtractionItemMapping(ner_class_to_id=ner_map, rel_class_to_id=rel_map)
    ext = ExtractionClassMapping(items=[item])
    return BatchClassesMapping(
        cat_mapping=[CatClassMapping(cat_class_to_id=[])],
        extraction_mapping=[ext],
        structuring_mapping=[StructuringClassMapping()],
        open_relex_mapping=[OpenRelexClassMapping()],
    )


class TestJointRelexDecoder:
    def test_empty(self, decoder):
        out = FakeModelOutput()
        assert decoder.decode(out) == []

    def test_single_relation(self, decoder):
        # 1 group, 5 tokens, 2 NER classes: entity 0 at (0,0), entity 1 at (3,4)
        ner_logits = _bio_logits(1, 5, 2, [(0, 0, 0, 0), (0, 3, 4, 1)])

        # 1 entity pair, 1 relation class, strong positive
        rel_logits = torch.tensor([[[5.0]]])  # (1, 1, 1)
        rel_idx = torch.tensor([[[0, 1]]])    # (1, 1, 2) — pair: entity 0 → entity 1
        rel_mask = torch.ones(1, 1, dtype=torch.bool)

        out = FakeModelOutput(
            ner_logits=ner_logits,
            joint_rel_logits=rel_logits,
            joint_rel_idx=rel_idx,
            joint_rel_mask=rel_mask,
        )

        ner_id_to_classes = {1: "person", 2: "location"}
        result = decoder.decode(
            out,
            classes_mapping=ner_id_to_classes,
            texts=[["John", "lives", "in", "New", "York"]],
        )
        assert len(result) == 1
        assert len(result[0]) == 1
        triple = result[0][0]
        assert triple["head"]["start"] == 0
        assert triple["tail"]["start"] == 3
        assert triple["score"] > 0.5

    def test_no_relations_below_threshold(self, decoder):
        ner_logits = _bio_logits(1, 5, 1, [(0, 0, 0, 0)])
        rel_logits = torch.tensor([[[-10.0]]])
        rel_idx = torch.tensor([[[0, 0]]])
        rel_mask = torch.ones(1, 1, dtype=torch.bool)

        out = FakeModelOutput(
            ner_logits=ner_logits,
            joint_rel_logits=rel_logits,
            joint_rel_idx=rel_idx,
            joint_rel_mask=rel_mask,
        )
        result = decoder.decode(out, classes_mapping={1: "A"})
        assert result[0] == []

    def test_masked_pairs_skipped(self, decoder):
        ner_logits = _bio_logits(1, 5, 1, [(0, 0, 0, 0), (0, 2, 2, 0)])
        rel_logits = torch.tensor([[[5.0], [5.0]]])  # 2 pairs, both strong
        rel_idx = torch.tensor([[[0, 1], [1, 0]]])
        rel_mask = torch.tensor([[True, False]])  # second pair masked

        out = FakeModelOutput(
            ner_logits=ner_logits,
            joint_rel_logits=rel_logits,
            joint_rel_idx=rel_idx,
            joint_rel_mask=rel_mask,
        )
        result = decoder.decode(out, classes_mapping={1: "A"})
        assert len(result[0]) == 1

    def test_entity_out_of_range_fallback(self, decoder):
        ner_logits = _bio_logits(1, 5, 1, [(0, 0, 0, 0)])  # only 1 entity
        rel_logits = torch.tensor([[[5.0]]])
        rel_idx = torch.tensor([[[0, 5]]])  # entity 5 doesn't exist
        rel_mask = torch.ones(1, 1, dtype=torch.bool)

        out = FakeModelOutput(
            ner_logits=ner_logits,
            joint_rel_logits=rel_logits,
            joint_rel_idx=rel_idx,
            joint_rel_mask=rel_mask,
        )
        result = decoder.decode(out, classes_mapping={1: "A"})
        assert len(result[0]) == 1
        assert result[0][0]["tail"]["start"] == -1  # fallback

    def test_with_batch_classes_mapping(self, decoder):
        ner_logits = _bio_logits(1, 5, 2, [(0, 0, 0, 0), (0, 3, 4, 1)])
        rel_logits = torch.tensor([[[5.0]]])
        rel_idx = torch.tensor([[[0, 1]]])
        rel_mask = torch.ones(1, 1, dtype=torch.bool)

        out = FakeModelOutput(
            ner_logits=ner_logits,
            joint_rel_logits=rel_logits,
            joint_rel_idx=rel_idx,
            joint_rel_mask=rel_mask,
        )
        mapping = _make_classes_mapping()
        result = decoder.decode(out, classes_mapping=mapping)
        assert len(result) == 1
        triple = result[0][0]
        assert triple["relation"] == "lives_in"