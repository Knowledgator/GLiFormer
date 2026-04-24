"""Shared fixtures for processor tests."""

import pytest
from unittest.mock import MagicMock
from dataclasses import asdict

from glinext.config import (
    GLiNextConfig,
    NERHeadConfig,
    ClassificationHeadConfig,
    JointRelexHeadConfig,
    OpenRelexHeadConfig,
    StructuringHeadConfig,
    CountHeadConfig,
    EmbeddingHeadConfig,
)
from glinext.processing.mappings import (
    BaseClassMapping,
    CatClassMapping,
    ExtractionItemMapping,
    ExtractionClassMapping,
    StructuringItemMapping,
    StructuringClassMapping,
    OpenRelexItemMapping,
    OpenRelexClassMapping,
    BatchClassesMapping,
)


# ── Fake words splitter ────────────────────────────────────────────────

class FakeWordsSplitter:
    """Whitespace tokenizer that yields (token, char_start, char_end) tuples."""

    def __call__(self, text):
        tokens = []
        i = 0
        for word in text.split():
            start = text.index(word, i)
            end = start + len(word)
            tokens.append((word, start, end))
            i = end
        return tokens


@pytest.fixture
def words_splitter():
    return FakeWordsSplitter()


# ── Minimal config ─────────────────────────────────────────────────────

def make_config(**overrides):
    """Build a GLiNextConfig with sensible test defaults."""
    defaults = dict(
        ent_token="[ENT]",
        sep_token="[SEP]",
        seq_token="[SEQ]",
        cat_token="[CAT]",
        rel_token="[REL]",
        parent_token="[P]",
        child_token="[CHILD]",
        max_len=512,
        words_splitter_type="whitespace",
    )
    defaults.update(overrides)
    return GLiNextConfig(**defaults)


@pytest.fixture
def base_config():
    """Config with NER enabled (default)."""
    return make_config()


@pytest.fixture
def all_tasks_config():
    """Config with all tasks enabled."""
    return make_config(
        ner_config=asdict(NERHeadConfig()),
        classification_config=asdict(ClassificationHeadConfig()),
        joint_relex_config=asdict(JointRelexHeadConfig()),
        open_relex_config=asdict(OpenRelexHeadConfig()),
        structuring_config=asdict(StructuringHeadConfig()),
        count_config=asdict(CountHeadConfig()),
        embedding_config=asdict(EmbeddingHeadConfig()),
    )


# ── Sample data builders ──────────────────────────────────────────────

@pytest.fixture
def ner_item():
    """Single NER data item with pre-resolved token spans."""
    return {
        "text": "John lives in New York",
        "extraction": [
            {
                "name": "entities",
                "ner": [
                    [0, 0, "person"],     # "John"
                    [3, 4, "location"],   # "New York"
                ],
            }
        ],
    }


@pytest.fixture
def ner_item_text_spans():
    """NER data item with text-based spans (not yet resolved)."""
    return {
        "text": "John lives in New York",
        "extraction": [
            {
                "name": "entities",
                "ner": [
                    ["John", "person"],
                    ["New York", "location"],
                ],
            }
        ],
    }


@pytest.fixture
def classification_item():
    return {
        "text": "This is a positive review",
        "classification": [
            {
                "name": "sentiment",
                "all_labels": ["positive", "negative", "neutral"],
                "true_labels": ["positive"],
            }
        ],
    }


@pytest.fixture
def joint_relex_item():
    """Joint NER + relation extraction item.

    Format: [head_id, rel_type, tail_id].
    """
    return {
        "text": "John lives in New York",
        "extraction": [
            {
                "name": "entities",
                "ner": [
                    [0, 0, "person"],
                    [3, 4, "location"],
                ],
                "relations": [
                    [0, "lives_in", 1],
                ],
            }
        ],
    }


@pytest.fixture
def open_relex_item():
    """Open relation extraction item with text mentions."""
    return {
        "text": "John lives in New York",
        "open_relex": [
            {
                "name": "relations",
                "relations": [
                    {
                        "relation": "lives_in",
                        "head": {"text": "John", "start": 0, "end": 0},
                        "tail": {"text": "New York", "start": 3, "end": 4},
                    },
                ],
            }
        ],
    }


@pytest.fixture
def structuring_item():
    """Structuring item with resolved spans."""
    return {
        "text": "John lives in New York",
        "structuring": {
            "person": [
                {
                    "name": {"text": "John", "start": 0, "end": 0},
                    "location": {"text": "New York", "start": 3, "end": 4},
                }
            ]
        },
    }


@pytest.fixture
def embedding_item():
    return {
        "text": "",
        "embedding": [
            ["hello world", "hi there", 0.9],
            ["cat", "dog", 0.3],
        ],
    }


@pytest.fixture
def multi_task_item():
    """Item with data for all tasks."""
    return {
        "text": "John lives in New York",
        "classification": [
            {
                "name": "topic",
                "all_labels": ["geography", "biography"],
                "true_labels": ["geography"],
            }
        ],
        "extraction": [
            {
                "name": "entities",
                "ner": [
                    [0, 0, "person"],
                    [3, 4, "location"],
                ],
                "relations": [
                    [0, "lives_in", 1],
                ],
            }
        ],
        "open_relex": [
            {
                "name": "relations",
                "relations": [
                    {
                        "relation": "lives_in",
                        "head": {"text": "John", "start": 0, "end": 0},
                        "tail": {"text": "New York", "start": 3, "end": 4},
                    },
                ],
            }
        ],
        "structuring": {
            "person": [
                {
                    "name": {"text": "John", "start": 0, "end": 0},
                    "location": {"text": "New York", "start": 3, "end": 4},
                }
            ]
        },
        "embedding": [
            ["hello", "world", 0.8],
        ],
    }


# ── Mapping builders ──────────────────────────────────────────────────

def make_extraction_mapping(ner_labels, rel_labels=None, name=None):
    ner_map = BaseClassMapping(
        class_to_id={l: i for i, l in enumerate(ner_labels)},
        name=name,
    )
    rel_map = None
    if rel_labels:
        rel_map = BaseClassMapping(
            class_to_id={l: i for i, l in enumerate(rel_labels)},
            name=name,
        )
    return ExtractionItemMapping(ner_class_to_id=ner_map, rel_class_to_id=rel_map)


def make_batch_classes_mapping(
    cat_mappings=None,
    extraction_mappings=None,
    structuring_mappings=None,
    open_relex_mappings=None,
    batch_size=1,
):
    if cat_mappings is None:
        cat_mappings = [CatClassMapping(cat_class_to_id=[]) for _ in range(batch_size)]
    if extraction_mappings is None:
        extraction_mappings = [ExtractionClassMapping() for _ in range(batch_size)]
    if structuring_mappings is None:
        structuring_mappings = [StructuringClassMapping() for _ in range(batch_size)]
    if open_relex_mappings is None:
        open_relex_mappings = [OpenRelexClassMapping() for _ in range(batch_size)]
    return BatchClassesMapping(
        cat_mapping=cat_mappings,
        extraction_mapping=extraction_mappings,
        structuring_mapping=structuring_mappings,
        open_relex_mapping=open_relex_mappings,
    )