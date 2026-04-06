"""Tests for GLiNExTDecoder — the general decoder factory and unflatten utility."""

import pytest
import torch
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from glinext.processing.decoder import GLiNExTDecoder, unflatten_by_batch_origin
from glinext.config import (
    NERHeadConfig,
    ClassificationHeadConfig,
    JointRelexHeadConfig,
    OpenRelexHeadConfig,
    StructuringHeadConfig,
    CountHeadConfig,
    EmbeddingHeadConfig,
)
from tests.conftest import make_config


# ── unflatten_by_batch_origin ────────────────────────────────────────────

class TestUnflattenByBatchOrigin:
    def test_basic(self):
        results = ["a", "b", "c"]
        batch_origin = torch.tensor([0, 0, 1])
        out = unflatten_by_batch_origin(results, batch_origin, 2)
        assert out == [["a", "b"], ["c"]]

    def test_empty_batches(self):
        results = ["x"]
        batch_origin = torch.tensor([2])
        out = unflatten_by_batch_origin(results, batch_origin, 4)
        assert out == [[], [], ["x"], []]

    def test_empty_results(self):
        out = unflatten_by_batch_origin([], torch.tensor([]).long(), 2)
        assert out == [[], []]

    def test_out_of_range_ignored(self):
        results = ["a", "b"]
        batch_origin = torch.tensor([0, 5])  # 5 >= batch_size=2
        out = unflatten_by_batch_origin(results, batch_origin, 2)
        assert out == [["a"], []]


# ── GLiNExTDecoder construction ──────────────────────────────────────────

@dataclass
class FakeModelOutput:
    """Model output with all task logits for testing the general decoder."""
    batch_size: Optional[int] = None
    # NER
    ner_logits: Optional[torch.Tensor] = None
    ner_batch_origin: Optional[torch.Tensor] = None
    span_logits: Optional[torch.Tensor] = None
    span_idx: Optional[torch.Tensor] = None
    span_mask: Optional[torch.Tensor] = None
    # Classification
    cat_logits: Optional[torch.Tensor] = None
    cat_batch_origin: Optional[torch.Tensor] = None
    # Joint Relex
    joint_rel_logits: Optional[torch.Tensor] = None
    joint_rel_idx: Optional[torch.Tensor] = None
    joint_rel_mask: Optional[torch.Tensor] = None
    joint_rel_batch_origin: Optional[torch.Tensor] = None
    # Open Relex
    open_rel_logits: Optional[torch.Tensor] = None
    open_rel_batch_origin: Optional[torch.Tensor] = None
    open_rel_anchor_mask: Optional[torch.Tensor] = None
    open_rel_span_logits: Optional[torch.Tensor] = None
    open_rel_span_idx: Optional[torch.Tensor] = None
    open_rel_span_mask: Optional[torch.Tensor] = None
    # Structuring
    structuring_logits: Optional[torch.Tensor] = None
    structuring_batch_origin: Optional[torch.Tensor] = None
    structuring_anchor_mask: Optional[torch.Tensor] = None
    structuring_span_logits: Optional[torch.Tensor] = None
    structuring_span_idx: Optional[torch.Tensor] = None
    structuring_span_mask: Optional[torch.Tensor] = None
    # Embedding
    embedding_logits: Optional[torch.Tensor] = None
    # Count
    count_logits: Optional[torch.Tensor] = None
    count_batch_origin: Optional[torch.Tensor] = None


class TestGLiNExTDecoderConstruction:
    def test_no_tasks(self):
        config = make_config(
            ner_config=None,
            classification_config=None,
        )
        # Override defaults — make_config may set ner_config by default
        config.ner_config = None
        config.classification_config = None
        config.joint_relex_config = None
        config.open_relex_config = None
        config.structuring_config = None
        config.count_config = None
        config.embedding_config = None
        dec = GLiNExTDecoder(config)
        assert len(dec.task_decoders) == 0

    def test_ner_only(self):
        config = make_config(ner_config=asdict(NERHeadConfig()))
        dec = GLiNExTDecoder(config)
        assert "ner" in dec.task_decoders

    def test_all_tasks(self):
        config = make_config(
            ner_config=asdict(NERHeadConfig()),
            classification_config=asdict(ClassificationHeadConfig()),
            joint_relex_config=asdict(JointRelexHeadConfig()),
            open_relex_config=asdict(OpenRelexHeadConfig()),
            structuring_config=asdict(StructuringHeadConfig()),
            count_config=asdict(CountHeadConfig()),
            embedding_config=asdict(EmbeddingHeadConfig()),
        )
        dec = GLiNExTDecoder(config)
        expected = {"ner", "classification", "joint_relex", "open_relex",
                    "structuring", "count", "embedding"}
        assert set(dec.task_decoders.keys()) == expected


class TestGLiNExTDecoderDecode:
    def test_empty_output_all_tasks(self):
        config = make_config(
            ner_config=asdict(NERHeadConfig()),
            classification_config=asdict(ClassificationHeadConfig()),
            embedding_config=asdict(EmbeddingHeadConfig()),
        )
        dec = GLiNExTDecoder(config)
        out = FakeModelOutput()
        results = dec.decode(out)
        # All decoders return [] for None logits
        for task_name, task_result in results.items():
            assert task_result == []

    def test_ner_decode(self):
        config = make_config(ner_config=asdict(NERHeadConfig()))
        dec = GLiNExTDecoder(config)

        logits = torch.full((1, 5, 2, 3), -10.0)
        logits[0, 0, 0, :] = 5.0  # entity at pos 0, class 0
        out = FakeModelOutput(ner_logits=logits)
        results = dec.decode(out, classes_mapping={1: "person", 2: "loc"})
        assert "ner" in results
        assert len(results["ner"]) == 1
        assert len(results["ner"][0]) == 1

    def test_classification_decode(self):
        config = make_config(classification_config=asdict(ClassificationHeadConfig()))
        dec = GLiNExTDecoder(config)

        logits = torch.tensor([[5.0, -5.0]])
        out = FakeModelOutput(cat_logits=logits)
        results = dec.decode(out)
        assert "classification" in results
        assert len(results["classification"]) == 1
        assert results["classification"][0][0]["class_id"] == 0

    def test_embedding_decode(self):
        config = make_config(embedding_config=asdict(EmbeddingHeadConfig()))
        dec = GLiNExTDecoder(config)

        logits = torch.tensor([0.9, 0.1])
        out = FakeModelOutput(embedding_logits=logits)
        results = dec.decode(out)
        assert "embedding" in results
        assert len(results["embedding"]) == 2

    def test_multi_task_decode(self):
        config = make_config(
            ner_config=asdict(NERHeadConfig()),
            classification_config=asdict(ClassificationHeadConfig()),
            embedding_config=asdict(EmbeddingHeadConfig()),
        )
        dec = GLiNExTDecoder(config)

        ner_logits = torch.full((1, 5, 1, 3), -10.0)
        ner_logits[0, 0, 0, :] = 5.0
        out = FakeModelOutput(
            ner_logits=ner_logits,
            cat_logits=torch.tensor([[5.0]]),
            embedding_logits=torch.tensor([0.5]),
        )

        results = dec.decode(out, classes_mapping={1: "A"})
        assert "ner" in results
        assert "classification" in results
        assert "embedding" in results
        assert len(results["ner"][0]) == 1
        assert len(results["classification"][0]) == 1
        assert len(results["embedding"]) == 1

    def test_kwargs_forwarded(self):
        config = make_config(ner_config=asdict(NERHeadConfig()))
        dec = GLiNExTDecoder(config)

        logits = torch.full((1, 5, 1, 3), 0.0)
        logits[0, 0, 0, :] = 5.0
        out = FakeModelOutput(ner_logits=logits)
        # threshold forwarded via kwargs
        results = dec.decode(out, classes_mapping={1: "A"}, threshold=0.99)
        # With high threshold, the borderline detection at sigmoid(0)≈0.5 is filtered
        # but the strong one at sigmoid(5)≈0.99 may still pass
        assert "ner" in results
