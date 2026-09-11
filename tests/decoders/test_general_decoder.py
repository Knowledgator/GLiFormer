"""Tests for GLiFormerDecoder — the general decoder factory and unflatten utility."""

import torch
from dataclasses import asdict, dataclass
from typing import Optional

from gliformer.processing.decoder import GLiFormerDecoder, unflatten_by_batch_origin
from gliformer.config import (
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


# ── GLiFormerDecoder construction ──────────────────────────────────────────

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

    def __post_init__(self):
        """Auto-fill batch_origin/batch_size for BN=B (1:1) when not provided."""
        _task_logits = {
            "ner": (self.ner_logits if self.ner_logits is not None else self.span_logits, "ner_batch_origin"),
            "cat": (self.cat_logits, "cat_batch_origin"),
            "count": (self.count_logits, "count_batch_origin"),
            "open_rel": (self.open_rel_logits if self.open_rel_logits is not None else self.open_rel_span_logits, "open_rel_batch_origin"),
            "structuring": (self.structuring_logits if self.structuring_logits is not None else self.structuring_span_logits, "structuring_batch_origin"),
        }
        for logits, origin_field in _task_logits.values():
            if logits is not None and getattr(self, origin_field) is None:
                BN = logits.shape[0]
                setattr(self, origin_field, torch.arange(BN))
                if self.batch_size is None:
                    self.batch_size = BN


class TestGLiFormerDecoderConstruction:
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
        dec = GLiFormerDecoder(config)
        assert len(dec.task_decoders) == 0

    def test_ner_only(self):
        config = make_config(ner_config=asdict(NERHeadConfig()))
        dec = GLiFormerDecoder(config)
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
        dec = GLiFormerDecoder(config)
        expected = {"ner", "classification", "joint_relex", "open_relex",
                    "structuring", "count", "embedding"}
        assert set(dec.task_decoders.keys()) == expected


class TestGLiFormerDecoderDecode:
    def test_empty_output_all_tasks(self):
        config = make_config(
            ner_config=asdict(NERHeadConfig()),
            classification_config=asdict(ClassificationHeadConfig()),
            embedding_config=asdict(EmbeddingHeadConfig()),
        )
        dec = GLiFormerDecoder(config)
        out = FakeModelOutput()
        results = dec.decode(out)
        # All decoders return [] for None logits
        for task_name, task_result in results.items():
            assert task_result == []

    def test_ner_decode(self):
        config = make_config(ner_config=asdict(NERHeadConfig()))
        dec = GLiFormerDecoder(config)

        logits = torch.full((1, 5, 2, 3), -10.0)
        logits[0, 0, 0, :] = 5.0  # entity at pos 0, class 0
        out = FakeModelOutput(ner_logits=logits)
        results = dec.decode(out, classes_mapping={0: "person", 1: "loc"})
        assert "ner" in results
        assert len(results["ner"]) == 1  # 1 batch item
        assert len(results["ner"][0]) == 1  # 1 group
        assert len(results["ner"][0][0]) == 1  # 1 span

    def test_classification_decode(self):
        config = make_config(classification_config=asdict(ClassificationHeadConfig()))
        dec = GLiFormerDecoder(config)

        logits = torch.tensor([[5.0, -5.0]])
        out = FakeModelOutput(cat_logits=logits)
        results = dec.decode(out)
        assert "classification" in results
        assert len(results["classification"]) == 1  # 1 batch item
        assert len(results["classification"][0]) == 1  # 1 group
        assert results["classification"][0][0][0]["class_name"] == "0"

    def test_embedding_decode(self):
        config = make_config(embedding_config=asdict(EmbeddingHeadConfig()))
        dec = GLiFormerDecoder(config)

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
        dec = GLiFormerDecoder(config)

        ner_logits = torch.full((1, 5, 1, 3), -10.0)
        ner_logits[0, 0, 0, :] = 5.0
        out = FakeModelOutput(
            ner_logits=ner_logits,
            cat_logits=torch.tensor([[5.0]]),
            embedding_logits=torch.tensor([0.5]),
        )

        results = dec.decode(out, classes_mapping={0: "A"})
        assert "ner" in results
        assert "classification" in results
        assert "embedding" in results
        assert len(results["ner"][0]) == 1  # 1 group for batch item 0
        assert len(results["ner"][0][0]) == 1  # 1 span in group 0
        assert len(results["classification"][0]) == 1  # 1 group
        assert len(results["classification"][0][0]) == 1  # 1 prediction
        assert len(results["embedding"]) == 1

    def test_kwargs_forwarded(self):
        config = make_config(ner_config=asdict(NERHeadConfig()))
        dec = GLiFormerDecoder(config)

        logits = torch.full((1, 5, 1, 3), 0.0)
        logits[0, 0, 0, :] = 5.0
        out = FakeModelOutput(ner_logits=logits)
        # threshold forwarded via kwargs
        results = dec.decode(out, classes_mapping={0: "A"}, threshold=0.99)
        # With high threshold, the borderline detection at sigmoid(0)≈0.5 is filtered
        # but the strong one at sigmoid(5)≈0.99 may still pass
        assert "ner" in results


def test_map_results_routes_opt_in_anchor_diagnostics_separately():
    task_name = "structuring"
    config = make_config(
        ner_config=None,
        structuring_config=None,
    )
    decoder = GLiFormerDecoder(config)

    class ProbeDecoder:
        def map_results(
            self,
            task_results,
            *,
            anchor_diagnostics_output=None,
            **kwargs,
        ):
            assert task_results == ["decoded"]
            assert anchor_diagnostics_output is not None
            anchor_diagnostics_output.append({
                "summary": {"activated_anchor_count": 3},
                "groups": [],
            })
            return ["mapped"]

    decoder.task_decoders = {task_name: ProbeDecoder()}

    result = decoder.map_results(
        {task_name: ["decoded"]},
        return_anchor_diagnostics=True,
    )

    assert result == {
        task_name: ["mapped"],
        f"{task_name}_anchor_diagnostics": [{
            "summary": {"activated_anchor_count": 3},
            "groups": [],
        }],
    }
