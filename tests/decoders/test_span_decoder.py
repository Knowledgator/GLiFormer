"""Tests for SpanDecoder base class — BIO decoding, span-level decoding, greedy search."""

import pytest
import torch

from glinext.tasks.span_decoder import Span, SpanDecoder


@pytest.fixture
def decoder():
    """SpanDecoder with a minimal config stub."""
    class _Cfg:
        pass
    # SpanDecoder.decode() raises NotImplementedError — that's fine, we test helpers
    d = SpanDecoder.__new__(SpanDecoder)
    d.config = _Cfg()
    d.threshold = 0.5
    return d


# ── greedy_search ────────────────────────────────────────────────────────

class TestGreedySearch:
    def test_empty(self, decoder):
        assert decoder.greedy_search([]) == []

    def test_no_overlap(self, decoder):
        spans = [
            Span(0, 1, "A", 0.9),
            Span(3, 4, "B", 0.8),
        ]
        result = decoder.greedy_search(spans, flat_ner=True)
        assert len(result) == 2
        # sorted by start
        assert result[0].start == 0
        assert result[1].start == 3

    def test_overlapping_flat_keeps_higher_score(self, decoder):
        spans = [
            Span(0, 2, "A", 0.7),
            Span(1, 3, "B", 0.9),
        ]
        result = decoder.greedy_search(spans, flat_ner=True)
        assert len(result) == 1
        assert result[0].entity_type == "B"

    def test_overlapping_nested_allows_containment(self, decoder):
        outer = Span(0, 4, "A", 0.9)
        inner = Span(1, 3, "B", 0.8)
        result = decoder.greedy_search([outer, inner], flat_ner=False)
        assert len(result) == 2

    def test_same_span_different_labels_no_multi(self, decoder):
        spans = [
            Span(0, 2, "A", 0.9),
            Span(0, 2, "B", 0.8),
        ]
        result = decoder.greedy_search(spans, flat_ner=True, multi_label=False)
        assert len(result) == 1
        assert result[0].entity_type == "A"

    def test_same_span_different_labels_multi(self, decoder):
        spans = [
            Span(0, 2, "A", 0.9),
            Span(0, 2, "B", 0.8),
        ]
        result = decoder.greedy_search(spans, flat_ner=True, multi_label=True)
        assert len(result) == 2


# ── decode_bio_spans ─────────────────────────────────────────────────────

class TestDecodeBioSpans:
    def _make_logits(self, L, C, spans):
        """Build (L, C, 3) logits with strong signals at given spans.

        spans: list of (start, end, class_idx) — 0-indexed class.
        """
        # Fill with very negative values so sigmoid → ~0
        logits = torch.full((L, C, 3), -10.0)
        for start, end, cls in spans:
            logits[start, cls, 0] = 5.0   # start
            logits[end, cls, 1] = 5.0     # end
            for t in range(start, end + 1):
                logits[t, cls, 2] = 5.0   # inside
        return logits

    def test_single_span(self, decoder):
        id_to_classes = {0: "person", 1: "location"}
        logits = self._make_logits(6, 2, [(0, 0, 0)])  # class 0 → id 1 → "person"
        spans = decoder.decode_bio_spans(logits, id_to_classes, threshold=0.5)
        assert len(spans) == 1
        assert spans[0].entity_type == "person"
        assert spans[0].start == 0
        assert spans[0].end == 0

    def test_two_spans_different_classes(self, decoder):
        id_to_classes = {0: "person", 1: "location"}
        logits = self._make_logits(6, 2, [(0, 0, 0), (3, 4, 1)])
        spans = decoder.decode_bio_spans(logits, id_to_classes, threshold=0.5)
        assert len(spans) == 2
        types = {s.entity_type for s in spans}
        assert types == {"person", "location"}

    def test_no_spans_below_threshold(self, decoder):
        logits = torch.full((5, 2, 3), -10.0)
        spans = decoder.decode_bio_spans(logits, {0: "A"}, threshold=0.5)
        assert spans == []

    def test_multi_token_span(self, decoder):
        id_to_classes = {0: "location"}
        logits = self._make_logits(6, 1, [(2, 4, 0)])
        spans = decoder.decode_bio_spans(logits, id_to_classes, threshold=0.5)
        assert len(spans) == 1
        assert spans[0].start == 2
        assert spans[0].end == 4


# ── decode_bio_spans_batch ───────────────────────────────────────────────

class TestDecodeBioSpansBatch:
    def test_batch(self, decoder):
        L, C = 5, 2
        id_to_classes = {0: "A", 1: "B"}
        logits = torch.full((3, L, C, 3), -10.0)
        # sample 0: span at (0,0) class 0
        logits[0, 0, 0, 0] = 5.0
        logits[0, 0, 0, 1] = 5.0
        logits[0, 0, 0, 2] = 5.0
        # sample 1: no spans
        # sample 2: span at (1,2) class 1
        logits[2, 1, 1, 0] = 5.0
        logits[2, 2, 1, 1] = 5.0
        for t in range(1, 3):
            logits[2, t, 1, 2] = 5.0

        result = decoder.decode_bio_spans_batch(logits, id_to_classes, 3, threshold=0.5)
        assert len(result) == 3
        assert len(result[0]) == 1
        assert result[0][0].entity_type == "A"
        assert len(result[1]) == 0
        assert len(result[2]) == 1
        assert result[2][0].entity_type == "B"

    def test_per_sample_id_to_classes(self, decoder):
        L, C = 4, 1
        logits = torch.full((2, L, C, 3), -10.0)
        logits[0, 0, 0, :] = 5.0
        logits[1, 1, 0, :] = 5.0
        per_sample = [{0: "X"}, {0: "Y"}]
        result = decoder.decode_bio_spans_batch(logits, per_sample, 2, threshold=0.5)
        assert result[0][0].entity_type == "X"
        assert result[1][0].entity_type == "Y"


# ── decode_span_level ────────────────────────────────────────────────────

class TestDecodeSpanLevel:
    def test_basic(self, decoder):
        B, S, C = 1, 3, 2
        span_logits = torch.full((B, S, C), -10.0)
        span_logits[0, 0, 0] = 5.0  # span 0, class 0
        span_logits[0, 2, 1] = 5.0  # span 2, class 1
        span_idx = torch.tensor([[[0, 1], [2, 3], [4, 5]]])
        span_mask = torch.ones(B, S, dtype=torch.bool)

        id_to_classes = {0: "person", 1: "location"}
        result = decoder.decode_span_level(
            span_logits, span_idx, span_mask, id_to_classes, threshold=0.5,
        )
        assert len(result) == 1
        assert len(result[0]) == 2
        types = {s.entity_type for s in result[0]}
        assert types == {"person", "location"}

    def test_masked_spans_ignored(self, decoder):
        B, S, C = 1, 3, 1
        span_logits = torch.full((B, S, C), 5.0)
        span_idx = torch.tensor([[[0, 1], [2, 3], [4, 5]]])
        span_mask = torch.tensor([[True, False, True]])

        result = decoder.decode_span_level(
            span_logits, span_idx, span_mask, {0: "A"}, threshold=0.5,
        )
        assert len(result[0]) == 2

    def test_below_threshold(self, decoder):
        B, S, C = 1, 2, 1
        span_logits = torch.full((B, S, C), -10.0)
        span_idx = torch.zeros(B, S, 2, dtype=torch.long)
        span_mask = torch.ones(B, S, dtype=torch.bool)

        result = decoder.decode_span_level(
            span_logits, span_idx, span_mask, {0: "A"}, threshold=0.5,
        )
        assert result[0] == []


# ── resolve_span_text ────────────────────────────────────────────────────

class TestResolveSpanText:
    def test_basic(self):
        texts = [["John", "lives", "in", "New", "York"]]
        assert SpanDecoder.resolve_span_text(texts, 0, 3, 4) == "New York"

    def test_single_token(self):
        texts = [["John", "lives"]]
        assert SpanDecoder.resolve_span_text(texts, 0, 0, 0) == "John"

    def test_none_texts(self):
        assert SpanDecoder.resolve_span_text(None, 0, 0, 0) == ""

    def test_out_of_range_batch(self):
        assert SpanDecoder.resolve_span_text([["a"]], 5, 0, 0) == ""