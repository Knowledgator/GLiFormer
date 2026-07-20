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
    AnchorModeling, IdentityAnchorModeling, LinearAnchorModeling,
    RNNAnchorModeling, MLPAnchorModeling,
    TransformerAnchorModeling,
)
from .pooling import Pooling, MeanPooling, CLSPooling, MaxPooling, WeightedPooling
from .position import (
    LearnedGrid2DPositionEmbedding,
    PositionEmbedding,
    NoPositionEmbedding,
    Sine1DPositionEmbedding,
    Linear1DPositionEmbedding,
    MLP1DPositionEmbedding,
    Sine2DPositionEmbedding,
    Linear2DPositionEmbedding,
    MLP2DPositionEmbedding,
    LearnedIndexPositionEmbedding,
    covering_grid_2d,
    normalized_grid_1d,
    normalized_grid_2d,
)

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
    "AnchorModeling", "IdentityAnchorModeling", "LinearAnchorModeling",
    "RNNAnchorModeling", "MLPAnchorModeling", "TransformerAnchorModeling",
    "Pooling", "MeanPooling", "CLSPooling", "MaxPooling", "WeightedPooling",
    "PositionEmbedding", "NoPositionEmbedding", "Sine2DPositionEmbedding",
    "Sine1DPositionEmbedding", "Linear1DPositionEmbedding",
    "MLP1DPositionEmbedding",
    "Linear2DPositionEmbedding", "MLP2DPositionEmbedding",
    "LearnedIndexPositionEmbedding", "LearnedGrid2DPositionEmbedding",
    "covering_grid_2d", "normalized_grid_1d", "normalized_grid_2d",
]
