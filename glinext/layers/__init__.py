"""Shared layers — re-exports from submodules for backward compatibility."""

from .anchor_layer import (
    AnchorLayer,
    FeatureAnchorLayer,
    FixedAnchorLayer,
    FixedRNNAnchorLayer,
    FixedTransformerAnchorLayer,
    ParentAnchorLayer,
    PositionBucketAnchorLayer,
    QueryRNNAnchorLayer,
    QueryTransformerAnchorLayer,
    RotaryAnchorLayer,
    TopKDensityDistinctAnchorLayer,
    TopKDistinctAnchorLayer,
    TopKNormAnchorLayer,
    TopKParentAnchorLayer,
)
from .anchor_modeling import (
    AnchorModeling,
    IdentityAnchorModeling,
    LinearAnchorModeling,
    MLPAnchorModeling,
    RNNAnchorModeling,
    TransformerAnchorModeling,
)
from .anchor_normalization import (
    AnchorNormalizer,
    CenterRMSAnchorNormalization,
    L2AnchorNormalization,
    LayerNormAnchorNormalization,
    NoAnchorNormalization,
    RMSNormAnchorNormalization,
)
from .anchor_relations import AnchorPairRelationsLayer, AnchorPairRelationsOutput
from .anchored_scorer import AnchoredSpanScorer
from .attention import (
    CrossAttentionBlock,
    CrossModalTokenFusion,
    Fuser,
    LayerwiseAttention,
    SelfAttentionBlock,
)
from .attention_bias import (
    AttentionBias,
    CausalAttentionBias,
    CompositeAttentionBias,
    GaussianDistanceAttentionBias,
    LocalWindowAttentionBias,
    NoAttentionBias,
    RelativeMLPAttentionBias,
)
from .groups import (
    AnchorCrossAttentionLayer,
    PostNormAnchorRefinementBlock,
    PreNormAnchorRefinementBlock,
    QueryGroupRNN,
    QueryGroupTransformer,
    RotaryGroupRNN,
)
from .mlp import FeaturesProjector, create_mlp
from .pair_rep import PairRepLayer, PromptRelationExtractor
from .pooling import CLSPooling, MaxPooling, MeanPooling, Pooling, WeightedPooling
from .position import (
    AnchorRefinementPositionEmbeddings,
    FixedSinusoidal1DPositionEmbedding,
    Fourier1DPositionEmbedding,
    LearnedGrid2DPositionEmbedding,
    LearnedIndexPositionEmbedding,
    Linear1DPositionEmbedding,
    Linear2DPositionEmbedding,
    MLP1DPositionEmbedding,
    MLP2DPositionEmbedding,
    NoPositionEmbedding,
    PositionEmbedding,
    RefinementPositionEncoding,
    Sine1DPositionEmbedding,
    SineBBox2DPositionEmbedding,
    Sine2DPositionEmbedding,
    covering_grid_2d,
    masked_normalized_grid_1d,
    normalized_grid_1d,
    normalized_grid_2d,
)
from .rnn import RnnSeq2SeqEncoder
from .rotary import RotaryEmbedding, apply_rotary_pos_emb, rotate_half

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
    "PositionBucketAnchorLayer", "TopKNormAnchorLayer", "TopKDistinctAnchorLayer",
    "TopKParentAnchorLayer", "TopKDensityDistinctAnchorLayer",
    "FixedRNNAnchorLayer", "FixedTransformerAnchorLayer", 
    "RotaryAnchorLayer", "QueryRNNAnchorLayer", "QueryTransformerAnchorLayer",
    "AnchorModeling", "IdentityAnchorModeling", "LinearAnchorModeling",
    "RNNAnchorModeling", "MLPAnchorModeling", "TransformerAnchorModeling",
    "AnchorNormalizer", "NoAnchorNormalization",
    "LayerNormAnchorNormalization", "RMSNormAnchorNormalization",
    "L2AnchorNormalization", "CenterRMSAnchorNormalization",
    "AnchorPairRelationsLayer", "AnchorPairRelationsOutput",
    "AttentionBias", "NoAttentionBias", "GaussianDistanceAttentionBias",
    "LocalWindowAttentionBias", "CausalAttentionBias",
    "RelativeMLPAttentionBias", "CompositeAttentionBias",
    "PreNormAnchorRefinementBlock", "PostNormAnchorRefinementBlock",
    "Pooling", "MeanPooling", "CLSPooling", "MaxPooling", "WeightedPooling",
    "PositionEmbedding", "NoPositionEmbedding", "Sine2DPositionEmbedding",
    "SineBBox2DPositionEmbedding", "RefinementPositionEncoding",
    "AnchorRefinementPositionEmbeddings",
    "FixedSinusoidal1DPositionEmbedding", "Fourier1DPositionEmbedding",
    "Sine1DPositionEmbedding", "Linear1DPositionEmbedding",
    "MLP1DPositionEmbedding",
    "Linear2DPositionEmbedding", "MLP2DPositionEmbedding",
    "LearnedIndexPositionEmbedding", "LearnedGrid2DPositionEmbedding",
    "covering_grid_2d", "normalized_grid_1d", "normalized_grid_2d",
    "masked_normalized_grid_1d",
]
