"""Tests for Joint Relex decoder."""

import pytest
import torch
from dataclasses import dataclass
from typing import Optional

from gliformer.tasks.joint_relex.decoder import JointRelexDecoder
from gliformer.tasks.span_decoder import Span
from gliformer.processing.mappings import (
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
    joint_rel_entity_spans: Optional[torch.Tensor] = None
    joint_rel_entity_class_idx: Optional[torch.Tensor] = None
    batch_size: Optional[int] = None

    def __post_init__(self):
        if self.batch_size is None:
            logits = self.ner_logits if self.ner_logits is not None else self.span_logits
            if logits is not None:
                BN = logits.shape[0]
                if self.ner_batch_origin is None:
                    self.ner_batch_origin = torch.arange(BN)
                if self.joint_rel_batch_origin is None and self.joint_rel_logits is not None:
                    self.joint_rel_batch_origin = torch.arange(BN)
                self.batch_size = BN


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

        ner_id_to_classes = {0: "person", 1: "location"}
        result = decoder.decode(
            out,
            classes_mapping=ner_id_to_classes,
            texts=[["John", "lives", "in", "New", "York"]],
        )
        assert len(result) == 1  # 1 batch item
        assert len(result[0]) == 1  # 1 group
        assert len(result[0][0]) == 1  # 1 triple
        triple = result[0][0][0]
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
        result = decoder.decode(out, classes_mapping={0: "A"})
        assert result[0][0] == []

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
        result = decoder.decode(out, classes_mapping={0: "A"})
        assert len(result[0][0]) == 1

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
        result = decoder.decode(out, classes_mapping={0: "A"})
        assert len(result[0][0]) == 1
        assert result[0][0][0]["tail"]["start"] == -1  # fallback

    def test_negative_entity_index_does_not_select_last_entity(self, decoder):
        ner_logits = _bio_logits(1, 3, 1, [(0, 0, 0, 0), (0, 2, 2, 0)])
        out = FakeModelOutput(
            ner_logits=ner_logits,
            joint_rel_logits=torch.tensor([[[5.0]]]),
            joint_rel_idx=torch.tensor([[[-1, 0]]]),
            joint_rel_mask=torch.ones(1, 1, dtype=torch.bool),
        )

        result = decoder.decode(out, classes_mapping={0: "A"})

        assert result[0][0][0]["head"]["start"] == -1

    def test_equal_boundaries_are_mapped_by_entity_class(self, decoder):
        # Multi-label decoding orders the stronger location first. The model
        # entity axis deliberately orders person first, so boundary-only
        # matching would collapse both endpoints onto location.
        ner_logits = _bio_logits(
            1, 2, 2,
            [(0, 0, 0, 0), (0, 0, 0, 1)],
        )
        ner_logits[0, 0, 1] = 7.0
        out = FakeModelOutput(
            ner_logits=ner_logits,
            joint_rel_logits=torch.tensor([[[5.0]]]),
            joint_rel_idx=torch.tensor([[[0, 1]]]),
            joint_rel_mask=torch.ones(1, 1, dtype=torch.bool),
            joint_rel_entity_spans=torch.tensor([[[0, 0], [0, 0]]]),
            joint_rel_entity_class_idx=torch.tensor([[0, 1]]),
        )

        result = decoder.decode(
            out,
            classes_mapping=_make_classes_mapping(),
            multi_label=True,
        )

        triple = result[0][0][0]
        assert triple["head"]["type"] == "person"
        assert triple["tail"]["type"] == "location"
        assert triple["head"]["entity_idx"] != triple["tail"]["entity_idx"]

    def test_group_text_resolution_uses_batch_origin(self, decoder):
        ner_logits = _bio_logits(
            2, 3, 1,
            [(0, 0, 0, 0), (0, 2, 2, 0), (1, 0, 0, 0), (1, 2, 2, 0)],
        )
        out = FakeModelOutput(
            ner_logits=ner_logits,
            ner_batch_origin=torch.tensor([0, 0]),
            joint_rel_logits=torch.tensor([[[-10.0]], [[5.0]]]),
            joint_rel_idx=torch.tensor([[[0, 1]], [[0, 1]]]),
            joint_rel_mask=torch.ones(2, 1, dtype=torch.bool),
            joint_rel_batch_origin=torch.tensor([0, 0]),
            batch_size=1,
        )

        result = decoder.decode(
            out,
            classes_mapping=[{0: "entity"}, {0: "entity"}],
            texts=[["first", "gap", "second"]],
        )

        triple = result[0][1][0]
        assert triple["head"]["text"] == "first"
        assert triple["tail"]["text"] == "second"

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
        assert len(result) == 1  # 1 batch item
        assert len(result[0]) == 1  # 1 group
        triple = result[0][0][0]
        assert triple["relation"] == "lives_in"

    def test_map_results_converts_triple_spans_to_character_offsets(
        self, decoder,
    ):
        task_results = [[[{
            "head": {"start": 0, "end": 0, "text": "John"},
            "tail": {"start": 1, "end": 1, "text": "London"},
            "relation": "lives_in",
            "score": 0.9,
        }]]]

        result = decoder.map_results(
            task_results,
            valid_to_orig_idx=[0],
            all_start_maps=[[0, 5]],
            all_end_maps=[[4, 11]],
            valid_texts=["John London"],
            num_original=1,
        )

        assert result[0][0]["head"] == {
            "start": 0, "end": 4, "text": "John",
        }
        assert result[0][0]["tail"] == {
            "start": 5, "end": 11, "text": "London",
        }
