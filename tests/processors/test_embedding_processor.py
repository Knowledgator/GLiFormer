"""Tests for embedding processor."""

import pytest
import torch

from glinext.tasks.embedding.processor import EmbeddingProcessor
from tests.conftest import make_config


@pytest.fixture
def emb_proc():
    config = make_config(embedding_config={})
    return EmbeddingProcessor(config)


class TestGetClassesMapping:
    def test_returns_none(self, emb_proc):
        assert emb_proc.get_classes_mapping([{"text": "hello"}]) is None


class TestCreateLabels:
    def test_basic(self, emb_proc, embedding_item):
        result = emb_proc.create_labels([embedding_item], None)
        assert result is not None
        assert "embedding_pair_idx" in result
        assert "embedding_labels" in result

        pair_idx = result["embedding_pair_idx"]
        labels = result["embedding_labels"]
        assert pair_idx.shape == (2, 2)
        assert labels.shape == (2,)

        # Pair indices: first pair (0,1), second pair (2,3)
        assert pair_idx[0].tolist() == [0, 1]
        assert pair_idx[1].tolist() == [2, 3]
        assert labels[0].item() == pytest.approx(0.9)
        assert labels[1].item() == pytest.approx(0.3)

    def test_empty(self, emb_proc):
        result = emb_proc.create_labels([{"text": "hello"}], None)
        assert result is None

    def test_multi_item_batch(self, emb_proc):
        batch = [
            {"embedding": [["a", "b", 0.5]]},
            {"embedding": [["c", "d", 0.8]]},
        ]
        result = emb_proc.create_labels(batch, None)
        assert result["embedding_pair_idx"].shape == (2, 2)
        # First item's pair: (0,1), second item: offset by 2 -> (2,3)
        assert result["embedding_pair_idx"][0].tolist() == [0, 1]
        assert result["embedding_pair_idx"][1].tolist() == [2, 3]

    def test_item_without_embeddings_increments_offset(self, emb_proc):
        batch = [
            {"text": "no embedding"},  # no embedding, offset += 1
            {"embedding": [["a", "b", 0.7]]},
        ]
        result = emb_proc.create_labels(batch, None)
        assert result is not None
        assert result["embedding_pair_idx"][0].tolist() == [1, 2]