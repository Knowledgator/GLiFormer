"""Shared fixtures for task head tests."""

import pytest
import torch
from dataclasses import asdict

from gliformer.config import (
    GLiFormerConfig,
    NERHeadConfig,
    ClassificationHeadConfig,
    JointRelexHeadConfig,
    OpenRelexHeadConfig,
    StructuringHeadConfig,
    CountHeadConfig,
    EmbeddingHeadConfig,
)
from gliformer.tasks import SharedRepresentations, TaskFlatInputs


# ── Dimensions ───────────────────────────────────────────────────────────

D = 64   # hidden size
B = 2    # batch size
W = 10   # word/sequence length
C = 3    # number of classes / prompt embeddings


def make_config(**overrides):
    """Build a GLiFormerConfig with test defaults (small hidden_size)."""
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
        hidden_size=D,
        encoder_config={"hidden_size": D, "model_type": "deberta-v2"},
        projector_hidden_act="gelu",
    )
    defaults.update(overrides)
    return GLiFormerConfig(**defaults)


@pytest.fixture
def shared():
    """SharedRepresentations with random tensors at standard test dimensions."""
    return SharedRepresentations(
        token_embeds=torch.randn(B, W, D),
        input_ids=torch.ones(B, W, dtype=torch.long),
        attention_mask=torch.ones(B, W, dtype=torch.long),
        words_embedding=torch.randn(B, W, D),
        mask=torch.ones(B, W, dtype=torch.long),
        prompts_embedding=torch.randn(B, C, D),
        prompts_embedding_mask=torch.ones(B, C, dtype=torch.long),
    )


@pytest.fixture
def flat_inputs():
    """TaskFlatInputs at BN=B (one group per batch item)."""
    BN = B
    return TaskFlatInputs(
        words_embedding=torch.randn(BN, W, D),
        mask=torch.ones(BN, W, dtype=torch.long),
        parent_embedding=torch.randn(BN, D),
        child_embedding=torch.randn(BN, C, D),
        child_mask=torch.ones(BN, C, dtype=torch.long),
        batch_origin=torch.arange(BN),
    )
