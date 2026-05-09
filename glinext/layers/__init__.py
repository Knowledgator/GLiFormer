"""Shared layers — re-exports from submodules for backward compatibility."""

from .mlp import create_mlp, FeaturesProjector
from .rnn import RnnSeq2SeqEncoder
from .pair_rep import PairRepLayer, PromptRelationExtractor
from .anchored_scorer import AnchoredSpanScorer
from .attention import SelfAttentionBlock, CrossAttentionBlock, Fuser, LayerwiseAttention, CrossModalTokenFusion
from .rotary import RotaryEmbedding, rotate_half, apply_rotary_pos_emb
from .groups import RotaryGroupRNN, QueryGroupRNN, QueryGroupTransformer, AnchorCrossAttentionLayer
from .anchor_layer import (
    AnchorLayer, ParentAnchorLayer, FeatureAnchorLayer, FixedAnchorLayer,
    FixedRNNAnchorLayer, FixedTransformerAnchorLayer,
    RotaryAnchorLayer, QueryRNNAnchorLayer, QueryTransformerAnchorLayer,
)
from .anchor_modeling import (
    AnchorModeling, LinearAnchorModeling, RNNAnchorModeling, MLPAnchorModeling,
    TransformerAnchorModeling,
)
from .pooling import Pooling, MeanPooling, CLSPooling, MaxPooling, WeightedPooling

__all__ = [
    "create_mlp", "FeaturesProjector",
    "RnnSeq2SeqEncoder",
    "PairRepLayer", "PromptRelationExtractor",
    "AnchoredSpanScorer",
    "CrossModalTokenFusion",
    "SelfAttentionBlock", "CrossAttentionBlock", "Fuser", "LayerwiseAttention",
    "RotaryEmbedding", "rotate_half", "apply_rotary_pos_emb",
    "RotaryGroupRNN", "QueryGroupRNN", "QueryGroupTransformer", "AnchorCrossAttentionLayer",
    "AnchorLayer", "ParentAnchorLayer", "FeatureAnchorLayer", "FixedAnchorLayer",
    "FixedRNNAnchorLayer", "FixedTransformerAnchorLayer", 
    "RotaryAnchorLayer", "QueryRNNAnchorLayer", "QueryTransformerAnchorLayer",
    "AnchorModeling", "LinearAnchorModeling", "RNNAnchorModeling", "MLPAnchorModeling", "TransformerAnchorModeling",
    "Pooling", "MeanPooling", "CLSPooling", "MaxPooling", "WeightedPooling",
]
