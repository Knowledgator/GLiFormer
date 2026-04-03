"""Shared layers — re-exports from submodules for backward compatibility."""

from .mlp import create_mlp, FeaturesProjector
from .rnn import LstmSeq2SeqEncoder
from .pair_rep import PairRepLayer, PromptRelationExtractor
from .anchored_scorer import AnchoredSpanScorer
from .attention import SelfAttentionBlock, CrossAttentionBlock, Fuser, LayerwiseAttention
from .rotary import RotaryEmbedding, rotate_half, apply_rotary_pos_emb
from .groups import RotaryGroupLSTM, QueryGroupLSTM, QueryGroupTransformer, AnchorCrossAttentionLayer
from .anchor_layer import (
    AnchorLayer, ParentAnchorLayer, FixedAnchorLayer,
    RotaryAnchorLayer, QueryLSTMAnchorLayer, QueryTransformerAnchorLayer,
)
from .anchor_modeling import (
    AnchorModeling, LinearAnchorModeling, LSTMAnchorModeling, MLPAnchorModeling,
)
from .pooling import Pooling, MeanPooling, CLSPooling, MaxPooling, WeightedPooling

__all__ = [
    "create_mlp", "FeaturesProjector",
    "LstmSeq2SeqEncoder",
    "PairRepLayer", "PromptRelationExtractor",
    "AnchoredSpanScorer",
    "SelfAttentionBlock", "CrossAttentionBlock", "Fuser", "LayerwiseAttention",
    "RotaryEmbedding", "rotate_half", "apply_rotary_pos_emb",
    "RotaryGroupLSTM", "QueryGroupLSTM", "QueryGroupTransformer", "AnchorCrossAttentionLayer",
    "AnchorLayer", "ParentAnchorLayer", "FixedAnchorLayer",
    "RotaryAnchorLayer", "QueryLSTMAnchorLayer", "QueryTransformerAnchorLayer",
    "AnchorModeling", "LinearAnchorModeling", "LSTMAnchorModeling", "MLPAnchorModeling",
    "Pooling", "MeanPooling", "CLSPooling", "MaxPooling", "WeightedPooling",
]
