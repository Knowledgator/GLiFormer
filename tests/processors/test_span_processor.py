"""Tests for SpanProcessor base class."""

import pytest
import torch

from glinext.tasks.span_processor import SpanProcessor
from tests.conftest import make_config, FakeWordsSplitter


class ConcreteSpanProcessor(SpanProcessor):
    """Minimal concrete subclass for testing abstract SpanProcessor."""

    def get_classes_mapping(self, batch_list, **kwargs):
        return None

    def create_labels(self, batch_list, classes_mapping, **kwargs):
        return None


@pytest.fixture
def span_proc():
    config = make_config()
    return ConcreteSpanProcessor(config, words_splitter=FakeWordsSplitter())


class TestBuildTokenCharMaps:
    def test_basic(self):
        tokens = [("John", 0, 4), ("lives", 5, 10), ("in", 11, 13)]
        s2t, e2t = SpanProcessor._build_token_char_maps(tokens)
        assert s2t == {0: 0, 5: 1, 11: 2}
        assert e2t == {4: 0, 10: 1, 13: 2}

    def test_empty(self):
        s2t, e2t = SpanProcessor._build_token_char_maps([])
        assert s2t == {}
        assert e2t == {}


class TestResolveTextSpan:
    def test_found(self):
        text = "John lives in New York"
        tokens = [("John", 0, 4), ("lives", 5, 10), ("in", 11, 13),
                  ("New", 14, 17), ("York", 18, 22)]
        assert SpanProcessor._resolve_text_span(text, tokens, "John") == (0, 0)
        assert SpanProcessor._resolve_text_span(text, tokens, "lives") == (1, 1)

    def test_multi_word(self):
        text = "John lives in New York"
        tokens = [("John", 0, 4), ("lives", 5, 10), ("in", 11, 13),
                  ("New", 14, 17), ("York", 18, 22)]
        assert SpanProcessor._resolve_text_span(text, tokens, "New York") == (3, 4)

    def test_not_found(self):
        text = "John lives in New York"
        tokens = [("John", 0, 4), ("lives", 5, 10)]
        assert SpanProcessor._resolve_text_span(text, tokens, "London") == (-1, -1)

    def test_case_insensitive(self):
        text = "John lives in New York"
        tokens = [("John", 0, 4), ("lives", 5, 10), ("in", 11, 13),
                  ("New", 14, 17), ("York", 18, 22)]
        assert SpanProcessor._resolve_text_span(text, tokens, "john") == (0, 0)

    def test_character_offsets_trim_formatting_whitespace(self):
        text = "Intro   Burton argued.  "
        tokens = [
            ("Intro", 0, 5),
            ("Burton", 8, 14),
            ("argued", 15, 21),
            (".", 21, 22),
        ]
        mention = "  Burton argued.  "

        assert SpanProcessor._char_span_to_token_range(
            text,
            tokens,
            6,
            24,
            mention,
        ) == (1, 3)


class TestResolveEntitySpans:
    def test_text_based(self):
        text = "John lives in New York"
        tokens = [("John", 0, 4), ("lives", 5, 10), ("in", 11, 13),
                  ("New", 14, 17), ("York", 18, 22)]
        ner = [["John", "person"], ["New York", "location"]]
        result = SpanProcessor._resolve_entity_spans(text, tokens, ner)
        assert len(result) == 2
        assert result[0] == [0, 0, "person"]
        assert result[1] == [3, 4, "location"]

    def test_pre_resolved(self):
        text = "John lives in New York"
        tokens = [("John", 0, 4)]
        ner = [[0, 0, "person"], [3, 4, "location"]]
        result = SpanProcessor._resolve_entity_spans(text, tokens, ner)
        assert result == [[0, 0, "person"], [3, 4, "location"]]

    def test_empty(self):
        assert SpanProcessor._resolve_entity_spans("text", [], []) == []
        assert SpanProcessor._resolve_entity_spans("text", [], None) == []


class TestTokenizeText:
    def test_basic(self, span_proc):
        item = {"text": "hello world"}
        tokens_with_spans, tokens = span_proc._tokenize_text(item)
        assert tokens == ["hello", "world"]
        assert len(tokens_with_spans) == 2
        assert "tokenized_text" in item
        assert item["tokenized_text"] == ["hello", "world"]

    def test_caching(self, span_proc):
        item = {"text": "hello world", "tokenized_text": ["cached"]}
        tokens_with_spans, tokens = span_proc._tokenize_text(item)
        # Should still tokenize, but not overwrite existing cache
        assert tokens == ["hello", "world"]
        assert item["tokenized_text"] == ["cached"]

    def test_empty_text(self, span_proc):
        item = {"text": ""}
        result = span_proc._tokenize_text(item)
        assert result == (None, None)

    def test_no_text(self, span_proc):
        item = {}
        result = span_proc._tokenize_text(item)
        assert result == (None, None)


class TestGenerateNegativeSpans:
    def test_generates_non_overlapping(self, span_proc):
        positive = {(0, 0), (3, 4)}
        negatives = span_proc._generate_negative_spans(positive, num_tokens=10, num_negatives=5)
        for start, end in negatives:
            assert (start, end) not in positive
            # Check no overlap with positive spans
            for ps, pe in positive:
                assert end < ps or start > pe

    def test_respects_count(self, span_proc):
        positive = set()
        negatives = span_proc._generate_negative_spans(positive, num_tokens=20, num_negatives=3)
        assert len(negatives) == 3

    def test_no_duplicates(self, span_proc):
        positive = set()
        negatives = span_proc._generate_negative_spans(positive, num_tokens=100, num_negatives=10)
        assert len(negatives) == len(set(negatives))


class TestCollectSpanCandidates:
    def test_basic(self, span_proc):
        positives = [(0, 0), (3, 4)]
        span_idx, span_mask, pos_set = span_proc._collect_span_candidates(positives, 10, neg_ratio=0.0)
        assert span_idx.shape == (2, 2)
        assert span_mask.shape == (2,)
        assert span_mask.all()
        assert pos_set == {(0, 0), (3, 4)}

    def test_with_negatives(self, span_proc):
        positives = [(0, 0), (3, 4)]
        span_idx, span_mask, pos_set = span_proc._collect_span_candidates(positives, 10, neg_ratio=1.0)
        # Should have 2 positives + ~2 negatives
        assert span_idx.shape[0] >= 2

    def test_empty(self, span_proc):
        span_idx, span_mask, pos_set = span_proc._collect_span_candidates([], 10, neg_ratio=1.0)
        assert span_idx is None
        assert span_mask is None
        assert pos_set == set()

    def test_zero_tokens(self, span_proc):
        span_idx, span_mask, pos_set = span_proc._collect_span_candidates([(0, 0)], 0, neg_ratio=1.0)
        assert span_idx is None
