import dataclasses
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

from transformers.models.auto import CONFIG_MAPPING

from gliner.config import BaseGLiNERConfig

from . import backbones as _layout_backbones  # noqa: F401 - registers custom AutoConfig entries
from .backbones import get_backbone, normalize_backbone_type


@dataclass
class TaskHeadConfig:
    """Loss controls shared by every configurable task head."""

    loss_coef: float = 1.0
    # None means inherit the global focal loss value supplied by training args.
    focal_loss_alpha: Optional[float] = None
    focal_loss_gamma: Optional[float] = None
    focal_loss_prob_margin: Optional[float] = None

    def __post_init__(self):
        if not math.isfinite(float(self.loss_coef)) or self.loss_coef < 0:
            raise ValueError("loss_coef must be finite and non-negative")
        for name in (
            "focal_loss_alpha",
            "focal_loss_gamma",
            "focal_loss_prob_margin",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite when configured")
        if self.focal_loss_alpha is not None and self.focal_loss_alpha > 1:
            raise ValueError("positive focal_loss_alpha must be at most 1")


@dataclass
class AnchorHeadConfig(TaskHeadConfig):
    """Configuration for heads that acquire and model semantic anchors."""

    anchor_mode: str = "parent"
    anchor_modeling: str = "linear"
    feature_anchor_mlp: bool = False
    feature_anchor_mlp_hidden_multiplier: int = 1
    anchor_refine_layers: int = 0
    anchor_refine_heads: int = 8
    # Post-norm preserves historical checkpoints. Set-prediction decoders can
    # opt into pre-norm plus LayerScale so a common attention response cannot
    # erase the distinct learned query residual at the first layer.
    anchor_refine_norm: str = "post_norm"
    anchor_refine_layer_scale_init: Optional[float] = None
    anchor_context_gate_init: float = 0.1
    anchor_context_gate_trainable: bool = True
    parent_token_index: int = -1
    embed_parent_token: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.anchor_refine_norm not in {"post_norm", "pre_norm"}:
            raise ValueError(
                "anchor_refine_norm must be 'post_norm' or 'pre_norm'"
            )
        if self.anchor_refine_layer_scale_init is not None:
            layer_scale = float(self.anchor_refine_layer_scale_init)
            if not math.isfinite(layer_scale) or layer_scale < 0:
                raise ValueError(
                    "anchor_refine_layer_scale_init must be finite and non-negative"
                )
        if not math.isfinite(float(self.anchor_context_gate_init)):
            raise ValueError("anchor_context_gate_init must be finite")


@dataclass
class BaseHeadConfig(AnchorHeadConfig):
    """Span-capable anchor head config retained as the text-head base."""

    represent_spans: bool = False
    neg_spans_ratio: float = 1.0
    span_loss_coef: float = 1.0


@dataclass
class NERHeadConfig(BaseHeadConfig):
    pass


@dataclass
class ClassificationHeadConfig(AnchorHeadConfig):
    cat_token_index: int = -1
    embed_cat_token: bool = True
    pooling_type: str = "mean"  # "mean", "cls", "max"
    scorer_type: str = "dot"  # "dot", "weighted-dot", "mlp", "hopfield"


@dataclass
class MediaClassificationHeadConfig(AnchorHeadConfig):
    """Shared configuration for pooled vision and audio classification."""

    pooling_type: str = "mean"
    scorer_type: str = "dot"


# Public modality-specific names remain aliases for source/checkpoint
# compatibility; there is only one implementation and one field hierarchy.
ImageClassificationHeadConfig = MediaClassificationHeadConfig
AudioClassificationHeadConfig = MediaClassificationHeadConfig


@dataclass
class SetPredictionHeadConfig(AnchorHeadConfig):
    """Shared configuration for slot-based media set-prediction heads."""

    anchor_mode: str = "fixed"
    num_fixed_slots: int = 100
    max_count: int = 100
    anchor_num_heads: int = 4
    anchor_num_layers: int = 2
    # Independent sigmoid classes need absolute logits with a stable scale;
    # unlike softmax, a shared large offset does not cancel at inference.
    scorer_type: str = "scaled-dot"
    class_loss_coef: float = 1.0
    objectness_loss_coef: float = 1.0
    objectness_positive_weight: float = 1.0
    objectness_negative_weight: float = 1.0
    # Objectness has a very different positive/negative population from
    # matched-slot classification, so it needs an independently configurable
    # focal policy. ``None`` inherits the task/global value.
    objectness_focal_loss_alpha: Optional[float] = 0.25
    objectness_focal_loss_gamma: Optional[float] = None
    objectness_focal_loss_prob_margin: Optional[float] = None
    # Set-prediction geometry is decoded from the refined representation. Keep
    # memory positions in that value stream by default; keys-only positions
    # tell attention where to look but discard that location before decoding.
    memory_position_in_values: bool = True
    multi_label: bool = True
    class_probability: str = "sigmoid"
    matcher_class_cost: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.num_fixed_slots <= 0:
            raise ValueError("num_fixed_slots must be positive")
        if self.max_count <= 0:
            raise ValueError("max_count must be positive")
        if self.anchor_num_heads <= 0 or self.anchor_num_layers <= 0:
            raise ValueError("anchor_num_heads and anchor_num_layers must be positive")
        if self.class_probability not in {"sigmoid", "softmax"}:
            raise ValueError("class_probability must be 'sigmoid' or 'softmax'")
        if self.multi_label and self.class_probability != "sigmoid":
            raise ValueError("multi-label set prediction requires class_probability='sigmoid'")
        if self.objectness_positive_weight < 0 or self.objectness_negative_weight < 0:
            raise ValueError("objectness positive/negative weights must be non-negative")
        for name in (
            "objectness_focal_loss_alpha",
            "objectness_focal_loss_gamma",
            "objectness_focal_loss_prob_margin",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite when configured")
        if (
            self.objectness_focal_loss_alpha is not None
            and self.objectness_focal_loss_alpha > 1
        ):
            raise ValueError(
                "positive objectness_focal_loss_alpha must be at most 1"
            )
        for name in (
            "class_loss_coef",
            "objectness_loss_coef",
            "matcher_class_cost",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass
class ObjectDetectionHeadConfig(SetPredictionHeadConfig):
    PREDICTION_ARCHITECTURE_FIELDS: ClassVar[tuple[str, ...]] = (
        "anchor_mode",
        "anchor_modeling",
        "feature_anchor_mlp",
        "feature_anchor_mlp_hidden_multiplier",
        "anchor_refine_layers",
        "anchor_refine_heads",
        "anchor_refine_norm",
        "anchor_refine_layer_scale_init",
        "anchor_context_gate_init",
        "anchor_context_gate_trainable",
        "parent_token_index",
        "embed_parent_token",
        "num_fixed_slots",
        "max_count",
        "anchor_num_heads",
        "anchor_num_layers",
        "scorer_type",
        "memory_position_embedding_type",
        "query_position_embedding_type",
        "memory_position_embedding_kwargs",
        "query_position_embedding_kwargs",
        "memory_position_in_values",
        "iterative_box_refinement",
        "bbox_refinement_detach",
        "bbox_head_zero_init",
        "spatial_attention_bias_type",
        "spatial_attention_sigma",
        "spatial_attention_bias_weight",
        "reference_box_mode",
        "reference_box_initialization",
        "reference_box_initial_size",
        "reference_box_grid_margin",
    )

    anchor_refine_layers: int = 2
    # Explicit, independently configurable spatial-position strategies.
    # memory: image-patch key positions; query: object-slot positions.
    memory_position_embedding_type: str = "sine2d"
    query_position_embedding_type: str = "sine2d"
    memory_position_embedding_kwargs: Optional[dict[str, Any]] = None
    query_position_embedding_kwargs: Optional[dict[str, Any]] = None
    reference_box_mode: str = "learned"
    # Grid preserves historical checkpoints. Random learned references avoid
    # exposing a deterministic spatial codebook to new set-prediction runs.
    reference_box_initialization: str = "grid"
    reference_box_initial_size: float = 0.1
    reference_box_grid_margin: float = 0.1
    # Modern DAB-style decoding options. Defaults retain historical one-shot
    # checkpoint behavior; the scratch vision configuration enables them.
    iterative_box_refinement: bool = False
    bbox_refinement_detach: bool = True
    bbox_head_zero_init: bool = False
    auxiliary_detection_loss_coef: float = 0.0
    spatial_attention_bias_type: str = "none"
    spatial_attention_sigma: float = 0.2
    spatial_attention_bias_weight: float = 1.0
    bbox_loss_coef: float = 5.0
    bbox_l1_format: str = "cxcywh"
    iou_loss_coef: float = 2.0
    matcher_bbox_cost: float = 5.0
    matcher_giou_cost: float = 2.0

    def __post_init__(self):
        super().__post_init__()
        from .layers.position import PositionEmbedding, covering_grid_2d

        self.memory_position_embedding_kwargs = dict(
            self.memory_position_embedding_kwargs or {}
        )
        self.query_position_embedding_kwargs = dict(
            self.query_position_embedding_kwargs or {}
        )
        if not 0.0 < self.reference_box_initial_size < 1.0:
            raise ValueError("reference_box_initial_size must be between 0 and 1")
        if not 0.0 <= self.reference_box_grid_margin < 0.5:
            raise ValueError("reference_box_grid_margin must be in [0, 0.5)")
        memory_strategy = PositionEmbedding.strategy_class(
            self.memory_position_embedding_type
        )
        query_strategy = PositionEmbedding.strategy_class(
            self.query_position_embedding_type
        )
        if memory_strategy.coordinate_dimensions not in {None, 2}:
            raise ValueError("Detection memory positions must be two-dimensional")
        if query_strategy.coordinate_dimensions not in {None, 2, 4}:
            raise ValueError(
                "Detection query positions must encode 2D centers or 2D boxes"
            )
        if query_strategy.requires_num_embeddings:
            self.query_position_embedding_kwargs.setdefault(
                "num_embeddings",
                self.num_fixed_slots,
            )
        if query_strategy.requires_grid_size:
            self.query_position_embedding_kwargs.setdefault(
                "grid_size",
                covering_grid_2d(self.num_fixed_slots),
            )
        if memory_strategy.requires_grid_size and "grid_size" not in (
            self.memory_position_embedding_kwargs
        ):
            raise ValueError(
                "memory learned-grid positions require an explicit base grid_size"
            )
        PositionEmbedding.validate_kwargs(
            self.memory_position_embedding_type,
            self.memory_position_embedding_kwargs,
        )
        PositionEmbedding.validate_kwargs(
            self.query_position_embedding_type,
            self.query_position_embedding_kwargs,
        )
        if memory_strategy.requires_num_embeddings:
            raise ValueError(
                "memory_position_embedding_type must support variable-size image grids"
            )
        if self.reference_box_mode not in {"none", "learned"}:
            raise ValueError("reference_box_mode must be 'none' or 'learned'")
        if self.reference_box_initialization not in {"grid", "random"}:
            raise ValueError(
                "reference_box_initialization must be 'grid' or 'random'"
            )
        if self.iterative_box_refinement and self.reference_box_mode != "learned":
            raise ValueError(
                "iterative_box_refinement requires reference_box_mode='learned'"
            )
        if self.iterative_box_refinement and self.anchor_refine_layers <= 0:
            raise ValueError(
                "iterative_box_refinement requires anchor_refine_layers > 0"
            )
        if self.reference_box_mode == "learned" and self.anchor_mode not in {
            "fixed",
            "fixed_rnn",
            "fixed_transformer",
        }:
            raise ValueError(
                "reference_box_mode='learned' requires a fixed anchor mode"
            )
        if query_strategy.requires_coordinates and self.reference_box_mode == "none":
            raise ValueError(
                "Coordinate-based query positions require reference_box_mode='learned'"
            )
        if query_strategy.requires_num_embeddings and self.anchor_mode not in {
            "fixed",
            "fixed_rnn",
            "fixed_transformer",
        }:
            raise ValueError(
                "Learned index query positions require a fixed anchor mode"
            )
        coefficients = (
            self.bbox_loss_coef,
            self.iou_loss_coef,
            self.matcher_bbox_cost,
            self.matcher_giou_cost,
            self.auxiliary_detection_loss_coef,
        )
        if any(coefficient < 0 for coefficient in coefficients):
            raise ValueError("detection loss and matcher coefficients must be non-negative")
        if not any(
            coefficient > 0
            for coefficient in (
                self.matcher_class_cost,
                self.matcher_bbox_cost,
                self.matcher_giou_cost,
            )
        ):
            raise ValueError("at least one matcher cost must be positive")
        if self.bbox_l1_format not in {"cxcywh", "xyxy"}:
            raise ValueError("bbox_l1_format must be 'cxcywh' or 'xyxy'")
        if self.spatial_attention_bias_type not in {"none", "gaussian"}:
            raise ValueError(
                "spatial_attention_bias_type must be 'none' or 'gaussian'"
            )
        if (
            self.spatial_attention_bias_type != "none"
            and self.anchor_refine_layers <= 0
        ):
            raise ValueError(
                "spatial attention bias requires anchor_refine_layers > 0"
            )
        if (
            self.spatial_attention_bias_type != "none"
            and self.reference_box_mode != "learned"
        ):
            raise ValueError(
                "spatial attention bias requires reference_box_mode='learned'"
            )
        if not math.isfinite(float(self.spatial_attention_sigma)) or (
            self.spatial_attention_sigma <= 0
        ):
            raise ValueError("spatial_attention_sigma must be finite and positive")
        if not math.isfinite(float(self.spatial_attention_bias_weight)) or (
            self.spatial_attention_bias_weight < 0
        ):
            raise ValueError(
                "spatial_attention_bias_weight must be finite and non-negative"
            )
        if self.auxiliary_detection_loss_coef > 0 and (
            not self.iterative_box_refinement or self.anchor_refine_layers < 2
        ):
            raise ValueError(
                "auxiliary_detection_loss_coef requires iterative refinement "
                "with at least two decoder layers"
            )

    def prediction_architecture_signature(self) -> tuple[tuple[str, Any], ...]:
        """Return only fields that construct or parameterize prediction modules."""

        mode = "rotary" if self.anchor_mode == "rnn" else self.anchor_mode
        fields = [
            "anchor_modeling",
            "parent_token_index",
            "embed_parent_token",
            "scorer_type",
            "anchor_refine_layers",
            "anchor_context_gate_init",
            "anchor_context_gate_trainable",
            "reference_box_mode",
            "iterative_box_refinement",
            "bbox_refinement_detach",
            "bbox_head_zero_init",
            "spatial_attention_bias_type",
            "spatial_attention_sigma",
            "spatial_attention_bias_weight",
        ]
        if mode == "features":
            fields.extend(
                (
                    "feature_anchor_mlp",
                    "feature_anchor_mlp_hidden_multiplier",
                )
            )
        if mode in {"fixed", "fixed_rnn", "fixed_transformer"}:
            fields.append("num_fixed_slots")
        if mode in {"rotary", "query_rnn", "query_transformer"}:
            fields.append("max_count")
        if mode in {"fixed_transformer", "query_transformer"}:
            fields.extend(("anchor_num_heads", "anchor_num_layers"))
        if self.anchor_refine_layers > 0:
            fields.extend(
                (
                    "anchor_refine_heads",
                    "anchor_refine_norm",
                    "anchor_refine_layer_scale_init",
                    "memory_position_embedding_type",
                    "query_position_embedding_type",
                    "memory_position_embedding_kwargs",
                    "query_position_embedding_kwargs",
                    "memory_position_in_values",
                )
            )
        if self.reference_box_mode == "learned":
            fields.extend(
                (
                    "reference_box_initial_size",
                    "reference_box_grid_margin",
                    "reference_box_initialization",
                )
            )
        return (("anchor_mode", mode),) + tuple(
            (name, getattr(self, name)) for name in fields
        )


@dataclass
class SegmentationHeadConfig(ObjectDetectionHeadConfig):
    num_prototypes: int = 32
    mask_size: int = 128
    mask_loss_coef: float = 1.0
    reuse_detection_head: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.num_prototypes <= 0:
            raise ValueError("num_prototypes must be positive")
        if self.mask_size <= 0:
            raise ValueError("mask_size must be positive")
        if self.mask_loss_coef < 0:
            raise ValueError("mask_loss_coef must be non-negative")


@dataclass
class AudioSegmentationHeadConfig(SetPredictionHeadConfig):
    anchor_refine_layers: int = 2
    memory_position_embedding_type: str = "sine1d"
    query_position_embedding_type: str = "sine1d"
    memory_position_embedding_kwargs: Optional[dict[str, Any]] = None
    query_position_embedding_kwargs: Optional[dict[str, Any]] = None
    segment_loss_coef: float = 5.0
    matcher_segment_cost: float = 5.0
    num_prototypes: int = 32
    mask_size: int = 256
    mask_loss_coef: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        from .layers.position import PositionEmbedding

        self.memory_position_embedding_kwargs = dict(
            self.memory_position_embedding_kwargs or {}
        )
        self.query_position_embedding_kwargs = dict(
            self.query_position_embedding_kwargs or {}
        )
        memory_strategy = PositionEmbedding.strategy_class(
            self.memory_position_embedding_type
        )
        query_strategy = PositionEmbedding.strategy_class(
            self.query_position_embedding_type
        )
        for role, strategy in (
            ("memory", memory_strategy),
            ("query", query_strategy),
        ):
            if strategy.coordinate_dimensions not in {None, 1}:
                raise ValueError(
                    f"Audio segmentation {role} positions must be one-dimensional"
                )
            if strategy.requires_grid_size:
                raise ValueError(
                    f"Audio segmentation {role} positions cannot use a 2D grid"
                )
        if memory_strategy.requires_num_embeddings:
            raise ValueError(
                "Audio memory positions must support variable sequence lengths"
            )
        if query_strategy.requires_num_embeddings:
            self.query_position_embedding_kwargs.setdefault(
                "num_embeddings",
                self.num_fixed_slots,
            )
            if self.anchor_mode not in {
                "fixed",
                "fixed_rnn",
                "fixed_transformer",
            }:
                raise ValueError(
                    "Learned index query positions require a fixed anchor mode"
                )
        PositionEmbedding.validate_kwargs(
            self.memory_position_embedding_type,
            self.memory_position_embedding_kwargs,
        )
        PositionEmbedding.validate_kwargs(
            self.query_position_embedding_type,
            self.query_position_embedding_kwargs,
        )
        if self.num_prototypes <= 0:
            raise ValueError("num_prototypes must be positive")
        if self.mask_size <= 0:
            raise ValueError("mask_size must be positive")
        coefficients = (
            self.segment_loss_coef,
            self.matcher_segment_cost,
            self.mask_loss_coef,
        )
        if any(coefficient < 0 for coefficient in coefficients):
            raise ValueError(
                "audio segmentation loss and matcher coefficients must be non-negative"
            )
        if self.matcher_class_cost == 0 and self.matcher_segment_cost == 0:
            raise ValueError("at least one audio segmentation matcher cost must be positive")


@dataclass
class JointRelexHeadConfig(BaseHeadConfig):
    """Config for joint NER + relation extraction (GLiNER-relex style).

    Inherits NER scoring from NERHead and adds adjacency-based
    entity pair scoring against [REL] type embeddings.
    """
    layer_type: str = "none"                # "dot", "weighted-dot", "mlp"
    pair_rep_type: str = "concat_proj"     # pair representation type
    triples_layer: Optional[str] = None    # optional triples scoring layer
    embed_rel_token: bool = True
    rel_token_index: int = -1
    adjacency_loss_coef: float = 1.0


@dataclass
class OpenRelexHeadConfig(BaseHeadConfig):
    """Config for anchor-based relation extraction (GLiNER2 style).

    Standalone head — no NER dependency. Uses configurable anchor layers
    to extract head/tail spans directly per (anchor, rel_type) pair.
    """
    anchor_mode: str = "fixed"          # "fixed", "features", "rotary", "query_rnn", "query_transformer"
    num_fixed_slots: int = 10
    max_count: int = 20
    anchor_num_heads: int = 4
    anchor_num_layers: int = 2
    rel_token_index: int = -1
    embed_rel_token: bool = True


@dataclass
class StructuringHeadConfig(BaseHeadConfig):
    anchor_mode: str = "rnn"  # "rnn", "features", "query_rnn", "query_transformer", "fixed"
    anchor_num_heads: int = 4
    anchor_num_layers: int = 2
    max_count: int = 20
    num_fixed_slots: int = 10  # number of learnable anchor slots (for anchor_mode="fixed")
    child_token_index: int = -1
    embed_child_token: bool = True
    # When True, BIO/span loss is computed against the optimal Hungarian
    # assignment of predicted anchors → gold instances per sample, removing
    # the spurious order signal of the data. When False, falls back to the
    # original positional loss.
    use_anchor_matching: bool = True
    # Loss reduction over the masked BIO/span tensor. "sum" matches GLiNER
    # behaviour; "mean" normalises by the number of valid elements which
    # decouples gradient magnitude from anchor count and sequence length.
    bio_loss_reduction: str = "sum"  # "sum" | "mean"
    # GLiNER-style negative sampling on BIO/span losses. ``negatives`` is the
    # keep-rate for negative-labelled positions; ``masking`` selects the
    # granularity at which negatives are sampled.
    #   "none"   — no sampling (keep all negatives)
    #   "global" — Bernoulli per-element over labels==0
    #   "label"  — drop negatives only for (anchor, field) cells with no
    #              positives anywhere in the sequence
    #   "span"   — drop negatives only for (anchor, token) positions with no
    #              positives across fields
    #   "anchor" — drop negatives only for anchor slots with no positives
    #              (pure-negative anchors). Matches the structuring use-case
    #              where most unmatched fixed slots are pure negatives.
    negatives: float = 1.0
    masking: str = "none"
    # Anchor-objectness head: per-anchor sigmoid score "is this slot used?"
    # supervised against the Hungarian-matched mask. At inference, anchors
    # with sigmoid(logit) < threshold are filtered out before BIO decoding.
    anchor_objectness: bool = False
    anchor_objectness_loss_coef: float = 1.0
    anchor_objectness_threshold: float = 0.5
    # Diagnostic logging: when True, the head emits per-batch positive vs
    # negative loss totals split between matched and unmatched anchors. Used
    # to diagnose the imbalance behind "many fields return None".
    log_loss_stats: bool = False
    log_loss_stats_every: int = 50  # log cadence in optimiser steps


@dataclass
class CountHeadConfig:
    mode: str = "regression"
    max_count: int = 20
    loss_coef: float = 1.0


@dataclass
class EmbeddingHeadConfig:
    loss_coef: float = 1.0
    pooling_type: str = "mean"  # "mean", "cls", "max", "weighted"
    similarity_fn: str = "cosine"  # "cosine", "dot", "l2"
    loss_fn: str = "mse"  # "mse", "contrastive", "triplet"
    projection_dim: Optional[int] = None


class GLiNextConfig(BaseGLiNERConfig):
    model_type = "glinext"
    expected_model_variant = None
    TASK_CONFIG_ATTRS = {
        "ner": "ner_config",
        "classification": "classification_config",
        "image_classification": "image_classification_config",
        "audio_classification": "audio_classification_config",
        "object_detection": "object_detection_config",
        "segmentation": "segmentation_config",
        "audio_segmentation": "audio_segmentation_config",
        "joint_relex": "joint_relex_config",
        "open_relex": "open_relex_config",
        "structuring": "structuring_config",
        "count": "count_config",
        "embedding": "embedding_config",
    }

    @staticmethod
    def _migrate_non_span_config(config: dict) -> None:
        """Drop fields serialized by the former overly broad head base class."""

        for obsolete in (
            "represent_spans",
            "neg_spans_ratio",
            "span_loss_coef",
            "obj_token_index",
            "embed_obj_token",
        ):
            config.pop(obsolete, None)

    @staticmethod
    def _migrate_detection_config(
        config: dict,
        *,
        segmentation: bool = False,
    ) -> None:
        """Translate legacy detector switches and discard obsolete target policy."""
        GLiNextConfig._migrate_non_span_config(config)
        has_explicit_memory_value_policy = "memory_position_in_values" in config
        modern_position_schema = any(
            name in config
            for name in (
                "memory_position_embedding_type",
                "query_position_embedding_type",
                "memory_position_embedding_kwargs",
                "query_position_embedding_kwargs",
            )
        )
        legacy_detection_schema = any(
            name in config
            for name in (
                "reference_points",
                "dense_coord_features",
                "bbox_prior_grid",
                "slot_pos_emb_std",
                "pos_emb_scale",
            )
        )
        if segmentation and legacy_detection_schema:
            # Older segmentation checkpoints own an independently trained
            # detector branch. Do not silently replace it with detection-head
            # weights merely because sharing became the modern default.
            config.setdefault("reuse_detection_head", False)
        if legacy_detection_schema:
            # Preserve pre-hierarchy detector inference: old checkpoints scored
            # raw label embeddings with mutually exclusive softmax classes. Old
            # files commonly serialized ``anchor_modeling: linear`` even though
            # that setting had no live consumer, so it must be replaced rather
            # than treated as an explicit modern choice.
            config["anchor_modeling"] = "identity"
            config.setdefault("scorer_type", "dot")
            config["multi_label"] = False
            config["class_probability"] = "softmax"
        if not has_explicit_memory_value_policy:
            if legacy_detection_schema:
                # The original detector added dense coordinates before using
                # image tokens as attention values.
                config["memory_position_in_values"] = True
            elif modern_position_schema:
                # The first registry-backed implementation used positions in
                # keys only. Preserve those checkpoint equations on load.
                config["memory_position_in_values"] = False
        legacy_reference = config.pop("reference_points", None)
        legacy_dense = config.pop("dense_coord_features", None)
        legacy_prior = config.pop("bbox_prior_grid", None)

        aliases = {
            "pos_emb_scale": "position_embedding_scale",
            "slot_pos_emb_std": "position_embedding_init_std",
            "default_box_size": "reference_box_initial_size",
            "bbox_prior_margin": "reference_box_grid_margin",
        }
        for old_name, new_name in aliases.items():
            value = config.pop(old_name, None)
            if value is not None:
                config.setdefault(new_name, value)

        # Older detector configs shared one set of constructor options between
        # memory and query positions.  Keep them loadable while moving to two
        # independent dictionaries that can configure arbitrary registry types.
        shared_position_kwargs = {}
        for config_name, strategy_name in (
            ("position_embedding_scale", "scale"),
            ("position_embedding_temperature", "temperature"),
            ("position_embedding_init_std", "init_std"),
        ):
            value = config.pop(config_name, None)
            if value is not None:
                shared_position_kwargs[strategy_name] = value
        for kwargs_name in (
            "memory_position_embedding_kwargs",
            "query_position_embedding_kwargs",
        ):
            strategy_kwargs = dict(config.get(kwargs_name) or {})
            position_type_name = (
                "memory_position_embedding_type"
                if kwargs_name.startswith("memory")
                else "query_position_embedding_type"
            )
            default_position_type = "sine2d"
            from .layers.position import PositionEmbedding

            strategy = PositionEmbedding.strategy_class(
                config.get(position_type_name, default_position_type)
            )
            for name, value in shared_position_kwargs.items():
                if name in strategy.config_fields:
                    strategy_kwargs.setdefault(name, value)
            if strategy_kwargs:
                config[kwargs_name] = strategy_kwargs

        if legacy_reference is not None:
            config.setdefault(
                "reference_box_mode",
                "learned" if legacy_reference or legacy_prior else "none",
            )
            config.setdefault(
                "query_position_embedding_type",
                "sine2d" if legacy_reference else "learned",
            )
        elif legacy_prior:
            config.setdefault("reference_box_mode", "learned")

        if legacy_dense is not None:
            if legacy_reference:
                memory_type = "sine2d"
            elif legacy_dense:
                memory_type = "linear2d"
            else:
                memory_type = "none"
            config.setdefault("memory_position_embedding_type", memory_type)

        for obsolete in (
            "min_bbox_side_pixels",
            "bbox_dedup_iou_threshold",
            "object_selection_strategy",
            "class_conditioned_bbox",
            "drop_cls_token_for_dense",
            "matcher_class_probability",
        ):
            config.pop(obsolete, None)

    def __init__(
        self,
        # Per-task sub-configs (None = disabled, dict or dataclass = enabled)
        ner_config: Optional[dict] = None,
        default_ner_config: Optional[bool] = None,
        classification_config: Optional[dict] = None,
        image_classification_config: Optional[dict] = None,
        audio_classification_config: Optional[dict] = None,
        object_detection_config: Optional[dict] = None,
        segmentation_config: Optional[dict] = None,
        audio_segmentation_config: Optional[dict] = None,
        relations_config: Optional[dict] = None,  # backward compat alias for joint_relex_config
        joint_relex_config: Optional[dict] = None,
        open_relex_config: Optional[dict] = None,
        structuring_config: Optional[dict] = None,
        count_config: Optional[dict] = None,
        embedding_config: Optional[dict] = None,
        # Shared layers across tasks (None = each task creates its own)
        shared_anchor_modeling: Optional[str] = None,  # "linear", "rnn", "mlp" — shared AnchorModeling layer
        shared_anchor_refine_layers: int = 0,  # shared AnchorCrossAttentionLayer (0 = disabled)
        shared_anchor_refine_heads: int = 8,
        # Labels encoder (bi-encoder style)
        labels_encoder: Optional[str] = None,
        labels_encoder_config: Optional[dict] = None,
        # Model variant / multimodal fusion
        backbone_type: str = "auto",
        model_variant: str = "text",
        multimodal_fusion: str = "uni-encoder",
        media_parent_embedding_source: str = "fixed",
        use_layout: bool = False,
        layout_image_tokens: bool = True,
        max_page_embeddings: int = 1024,
        # Optional multimodal encoders
        vision_model_name: Optional[str] = None,
        vision_encoder_type: Optional[str] = None,
        vision_encoder_config: Optional[dict] = None,
        vision_in_channels: int = 3,
        vision_patch_size: int = 16,
        vision_position_embedding_type: Optional[str] = None,
        vision_position_embedding_kwargs: Optional[dict[str, Any]] = None,
        vision_position_embeddings: Optional[bool] = None,
        vision_feature_stride: Optional[Any] = None,
        vision_feature_spatial_shape: Optional[Any] = None,
        vision_feature_prefix_tokens: Optional[int] = None,
        audio_model_name: Optional[str] = None,
        audio_encoder_type: Optional[str] = None,
        audio_encoder_config: Optional[dict] = None,
        audio_in_channels: int = 1,
        audio_num_layers: int = 3,
        audio_stride: int = 4,
        audio_freq_stride: int = 2,
        audio_time_stride: Optional[int] = None,
        audio_input_format: str = "freq_first",
        audio_processor_type: str = "custom",
        audio_processor_name: Optional[str] = None,
        audio_sampling_rate: Optional[int] = None,
        audio_do_resample: bool = True,
        audio_processor_output_format: str = "raw",
        audio_do_normalize: bool = False,
        audio_n_fft: int = 400,
        audio_hop_length: Optional[int] = None,
        audio_win_length: Optional[int] = None,
        audio_n_mels: int = 80,
        audio_power: float = 2.0,
        omni_modalities: Optional[list[str]] = None,
        # Special tokens
        seq_token: str = "[SEQ]",
        cat_token: str = "[CAT]",
        rel_token: str = "[REL]",
        parent_token: str = "[PARENT]",
        child_token: str = "[CHILD]",
        obj_token: str = "[OBJ]",
        # Per-task parent tokens
        per_task_parents: Optional[bool] = None,  # True = distinct per-task tokens; False = shared [PARENT]; None = auto-detect
        ner_parent_token: Optional[str] = None,
        cat_parent_token: Optional[str] = None,
        open_rel_parent_token: Optional[str] = None,
        struct_parent_token: Optional[str] = None,
        # Parent token index (resolved during model init, like class_token_index)
        parent_token_index: int = -1,
        embed_parent_token: bool = True,
        # ── Backward compat: flat params auto-migrated to sub-configs ──
        # Layer selection
        relations_layer: Optional[str] = None,
        classifier_layer: Optional[str] = None,
        groups_layer: Optional[str] = None,  # backward compat alias for anchor_mode
        count_layer: Optional[str] = None,
        # Relations flat params
        rel_mode: str = "adjacency",
        pair_rep_type: str = "concat_proj",
        triples_layer: Optional[str] = None,
        embed_rel_token: bool = True,
        rel_token_index: int = -1,
        # Classification flat params
        cat_token_index: int = -1,
        embed_cat_token: bool = True,
        # Loss coefficients (flat)
        ner_loss_coef: float = 1.0,
        cat_loss_coef: float = 1.0,
        image_classification_loss_coef: float = 1.0,
        audio_classification_loss_coef: float = 1.0,
        object_detection_loss_coef: float = 1.0,
        segmentation_loss_coef: float = 1.0,
        audio_segmentation_loss_coef: float = 1.0,
        rel_loss_coef: float = 1.0,
        adjacency_loss_coef: float = 1.0,
        count_loss_coef: float = 1.0,
        groups_loss_coef: float = 1.0,
        embedding_loss_coef: float = 1.0,
        structuring_loss_coef: float = 1.0,
        # Span representation (flat)
        represent_spans: bool = False,
        neg_spans_ratio: float = 1.0,
        span_loss_coef: float = 1.0,
        # Count (flat)
        count_mode: str = "regression",
        max_count: int = 20,
        # Groups (flat)
        anchor_num_heads: int = 4,
        anchor_num_layers: int = 2,
        # Structuring (flat)
        child_token_index: int = -1,
        embed_child_token: bool = True,
        obj_token_index: int = -1,
        embed_obj_token: bool = True,
        image_size: int = 224,
        vision_processor_type: str = "custom",
        vision_processor_name: Optional[str] = None,
        vision_resize_size: Optional[Any] = None,
        vision_center_crop_size: Optional[Any] = None,
        vision_interpolation: str = "bilinear",
        vision_do_rescale: bool = True,
        vision_do_normalize: bool = False,
        vision_image_mean: Optional[list[float]] = None,
        vision_image_std: Optional[list[float]] = None,
        **kwargs,
    ):
        deprecated_processor_fields = {
            "image_processor_name",
            "audio_feature_type",
            "audio_log_mel",
        }
        # Accepted and discarded only so older checkpoints remain loadable. These
        # fields never controlled a vision module and are no longer serialized.
        kwargs.pop("vision_num_layers", None)
        kwargs.pop("vision_stride", None)
        deprecated_present = sorted(deprecated_processor_fields.intersection(kwargs))
        if deprecated_present:
            raise ValueError(
                "Unsupported processor config field(s): "
                f"{', '.join(deprecated_present)}. Use canonical processor fields only."
            )

        allowed_processor_types = {"custom", "auto"}
        if vision_processor_type not in allowed_processor_types:
            raise ValueError(
                "vision_processor_type must be one of "
                f"{sorted(allowed_processor_types)}, got {vision_processor_type!r}."
            )
        if audio_processor_type not in allowed_processor_types:
            raise ValueError(
                "audio_processor_type must be one of "
                f"{sorted(allowed_processor_types)}, got {audio_processor_type!r}."
            )
        allowed_audio_formats = {
            "raw",
            "spectrogram",
            "mel_spectrogram",
            "log_mel_spectrogram",
        }
        if audio_processor_output_format not in allowed_audio_formats:
            raise ValueError(
                "audio_processor_output_format must be one of "
                f"{sorted(allowed_audio_formats)}, got {audio_processor_output_format!r}."
            )

        super().__init__(**kwargs)

        # ── Migrate flat params to sub-configs if sub-configs not provided ──

        allowed_model_variants = {"text", "vision", "audio", "omni", "layout"}
        if model_variant not in allowed_model_variants:
            raise ValueError(
                "model_variant must be one of "
                f"{sorted(allowed_model_variants)}, got {model_variant!r}."
            )
        media_parent_embedding_source = str(media_parent_embedding_source).lower().replace("-", "_")
        media_parent_embedding_source = {
            "avg": "mean",
            "average": "mean",
        }.get(media_parent_embedding_source, media_parent_embedding_source)
        allowed_media_parent_sources = {"fixed", "first", "mean", "sum"}
        if media_parent_embedding_source not in allowed_media_parent_sources:
            raise ValueError(
                "media_parent_embedding_source must be one of "
                f"{sorted(allowed_media_parent_sources)}, got {media_parent_embedding_source!r}."
            )
        expected_model_variant = getattr(self, "expected_model_variant", None)
        if expected_model_variant is not None and model_variant != expected_model_variant:
            raise ValueError(
                f"{self.__class__.__name__} requires model_variant={expected_model_variant!r}, "
                f"got {model_variant!r}."
            )

        if default_ner_config is None:
            default_ner_config = model_variant not in {"vision", "audio"}

        # NER: on by default for legacy/text/omni configs, but single-media
        # configs avoid carrying unused text heads.
        if ner_config is None and default_ner_config:
            ner_config = {}
        if isinstance(ner_config, dict):
            ner_config = dict(ner_config)
            ner_config.pop("scorer_type", None)  # backward compat: scorer_type removed
            ner_config.setdefault("loss_coef", ner_loss_coef)
            ner_config.setdefault("represent_spans", represent_spans)
            ner_config.setdefault("neg_spans_ratio", neg_spans_ratio)
            ner_config.setdefault("span_loss_coef", span_loss_coef)
            self.ner_config = NERHeadConfig(**ner_config)
        else:
            self.ner_config = ner_config

        # Classification
        if classification_config is None and classifier_layer is not None:
            classification_config = {
                "cat_token_index": cat_token_index,
                "embed_cat_token": embed_cat_token,
                "loss_coef": cat_loss_coef,
            }
        if isinstance(classification_config, dict):
            classification_config = dict(classification_config)
            classification_config.pop("layer_type", None)  # backward compat: layer_type removed
            self._migrate_non_span_config(classification_config)
            self.classification_config = ClassificationHeadConfig(**classification_config)
        else:
            self.classification_config = classification_config

        if isinstance(image_classification_config, dict):
            image_classification_config = dict(image_classification_config)
            self._migrate_non_span_config(image_classification_config)
            image_classification_config.setdefault("loss_coef", image_classification_loss_coef)
            self.image_classification_config = ImageClassificationHeadConfig(**image_classification_config)
        else:
            self.image_classification_config = image_classification_config

        if isinstance(audio_classification_config, dict):
            audio_classification_config = dict(audio_classification_config)
            self._migrate_non_span_config(audio_classification_config)
            audio_classification_config.setdefault("loss_coef", audio_classification_loss_coef)
            self.audio_classification_config = AudioClassificationHeadConfig(**audio_classification_config)
        else:
            self.audio_classification_config = audio_classification_config

        if isinstance(object_detection_config, dict):
            object_detection_config = dict(object_detection_config)
            self._migrate_detection_config(object_detection_config)
            object_detection_config.setdefault("loss_coef", object_detection_loss_coef)
            self.object_detection_config = ObjectDetectionHeadConfig(**object_detection_config)
        else:
            self.object_detection_config = object_detection_config

        if segmentation_config is not None:
            if dataclasses.is_dataclass(segmentation_config) and not isinstance(
                segmentation_config,
                type,
            ):
                segmentation_config = dataclasses.asdict(segmentation_config)
            elif isinstance(segmentation_config, dict):
                segmentation_config = dict(segmentation_config)
            else:
                raise TypeError(
                    "segmentation_config must be a dict, a dataclass instance, or None"
                )
            self._migrate_detection_config(
                segmentation_config,
                segmentation=True,
            )
            if (
                segmentation_config.get("reuse_detection_head", True)
                and self.object_detection_config is not None
            ):
                # Reuse is an explicit architecture policy, not a side effect
                # of whether configuration arrived as a dict or dataclass.
                # Set reuse_detection_head=False to own a separate detector.
                # Loss, matching, focal, and mask policy remain independent.
                for name in ObjectDetectionHeadConfig.PREDICTION_ARCHITECTURE_FIELDS:
                    segmentation_config[name] = getattr(
                        self.object_detection_config,
                        name,
                    )
            segmentation_config.setdefault("loss_coef", segmentation_loss_coef)
            self.segmentation_config = SegmentationHeadConfig(**segmentation_config)
        else:
            self.segmentation_config = None

        if isinstance(audio_segmentation_config, dict):
            audio_segmentation_config = dict(audio_segmentation_config)
            self._migrate_non_span_config(audio_segmentation_config)
            audio_segmentation_config.setdefault("loss_coef", audio_segmentation_loss_coef)
            self.audio_segmentation_config = AudioSegmentationHeadConfig(**audio_segmentation_config)
        else:
            self.audio_segmentation_config = audio_segmentation_config

        # Joint Relex (backward compat: relations_config → joint_relex_config)
        if joint_relex_config is None and relations_config is not None:
            joint_relex_config = relations_config
        if joint_relex_config is None and relations_layer is not None:
            joint_relex_config = {
                "layer_type": relations_layer,
                "pair_rep_type": pair_rep_type,
                "triples_layer": triples_layer,
                "embed_rel_token": embed_rel_token,
                "rel_token_index": rel_token_index,
                "loss_coef": rel_loss_coef,
                "adjacency_loss_coef": adjacency_loss_coef,
            }
        if isinstance(joint_relex_config, dict):
            self.joint_relex_config = JointRelexHeadConfig(**joint_relex_config)
        else:
            self.joint_relex_config = joint_relex_config
        # Backward compat alias
        self.relations_config = self.joint_relex_config

        # Open Relex
        if isinstance(open_relex_config, dict):
            self.open_relex_config = OpenRelexHeadConfig(**open_relex_config)
        else:
            self.open_relex_config = open_relex_config

        # Structuring
        if structuring_config is None and groups_layer is not None:
            structuring_config = {
                "anchor_mode": groups_layer,
                "anchor_num_heads": anchor_num_heads,
                "anchor_num_layers": anchor_num_layers,
                "max_count": max_count,
                "child_token_index": child_token_index,
                "embed_child_token": embed_child_token,
                "loss_coef": structuring_loss_coef,
            }
        if isinstance(structuring_config, dict):
            self.structuring_config = StructuringHeadConfig(**structuring_config)
        else:
            self.structuring_config = structuring_config

        # Count
        if count_config is None and count_layer is not None:
            count_config = {
                "mode": count_mode,
                "max_count": max_count,
                "loss_coef": count_loss_coef,
            }
        if isinstance(count_config, dict):
            self.count_config = CountHeadConfig(**count_config)
        else:
            self.count_config = count_config

        # Embedding
        if isinstance(embedding_config, dict):
            embedding_config.setdefault("loss_coef", embedding_loss_coef)
            self.embedding_config = EmbeddingHeadConfig(**embedding_config)
        else:
            self.embedding_config = embedding_config

        # Labels encoder config
        if isinstance(labels_encoder_config, dict):
            labels_encoder_config = dict(labels_encoder_config)
            labels_encoder_config["model_type"] = labels_encoder_config.get("model_type", "deberta-v2")
            labels_encoder_config = CONFIG_MAPPING[labels_encoder_config["model_type"]](**labels_encoder_config)
        self.labels_encoder = labels_encoder
        self.labels_encoder_config = labels_encoder_config

        self.backbone_type = normalize_backbone_type(backbone_type)
        if self.backbone_type != "auto":
            get_backbone(self.backbone_type)

        vision_tasks_enabled = (
            self.image_classification_config is not None
            or self.object_detection_config is not None
            or self.segmentation_config is not None
        )
        dense_vision_tasks_enabled = (
            self.object_detection_config is not None
            or self.segmentation_config is not None
        )
        if dense_vision_tasks_enabled and vision_center_crop_size is not None:
            raise ValueError(
                "Detection and segmentation do not support center-cropped image "
                "processing without annotation transform metadata"
            )
        if dense_vision_tasks_enabled and vision_processor_type == "auto":
            raise ValueError(
                "Detection and segmentation require vision_processor_type='custom' "
                "until external processor geometry metadata is available"
            )
        if vision_tasks_enabled and vision_model_name is None and vision_encoder_type is None:
            vision_encoder_type = "patch"
        audio_tasks_enabled = (
            self.audio_classification_config is not None
            or self.audio_segmentation_config is not None
        )
        if audio_tasks_enabled and audio_model_name is None and audio_encoder_type is None:
            audio_encoder_type = "conv"

        self.model_variant = model_variant
        self.multimodal_fusion = multimodal_fusion
        self.media_parent_embedding_source = media_parent_embedding_source
        self.use_layout = use_layout
        self.layout_image_tokens = layout_image_tokens
        self.max_page_embeddings = max_page_embeddings

        # Multimodal encoder config
        if isinstance(vision_encoder_config, dict):
            vision_encoder_config = dict(vision_encoder_config)
            if "model_type" not in vision_encoder_config:
                raise ValueError("vision_encoder_config requires a model_type")
            vision_encoder_config = CONFIG_MAPPING[vision_encoder_config["model_type"]](**vision_encoder_config)
        if isinstance(audio_encoder_config, dict):
            audio_encoder_config = dict(audio_encoder_config)
            if "model_type" not in audio_encoder_config:
                raise ValueError("audio_encoder_config requires a model_type")
            audio_encoder_config = CONFIG_MAPPING[audio_encoder_config["model_type"]](**audio_encoder_config)
        self.vision_model_name = vision_model_name
        self.vision_encoder_type = vision_encoder_type
        self.vision_encoder_config = vision_encoder_config
        self.vision_in_channels = vision_in_channels
        self.vision_patch_size = vision_patch_size
        local_vision_encoder = (
            vision_encoder_type in {"patch", "path", "cnn"}
            or (vision_encoder_type is None and vision_model_name is None)
        )
        if vision_position_embedding_type is None:
            vision_position_embedding_type = (
                "learned_grid2d"
                if local_vision_encoder and vision_position_embeddings is not False
                else "none"
            )
        elif (
            not local_vision_encoder
            and str(vision_position_embedding_type).lower().replace("-", "_")
            != "none"
        ):
            raise ValueError(
                "vision_position_embedding_type configures only the local patch "
                "encoder; AutoModel backbones own their positional embeddings"
            )
        elif (
            vision_position_embeddings is not None
            and bool(vision_position_embeddings)
            != (str(vision_position_embedding_type).lower() != "none")
        ):
            raise ValueError(
                "vision_position_embeddings conflicts with "
                "vision_position_embedding_type"
            )
        from .layers.position import PositionEmbedding

        vision_position_embedding_type = (
            str(vision_position_embedding_type).lower().replace("-", "_")
        )
        vision_position_strategy = PositionEmbedding.strategy_class(
            vision_position_embedding_type
        )
        vision_position_embedding_kwargs = dict(
            vision_position_embedding_kwargs or {}
        )
        if vision_position_strategy.requires_num_embeddings:
            image_hw = (
                tuple(image_size)
                if isinstance(image_size, (list, tuple))
                else (image_size, image_size)
            )
            patch_hw = (
                tuple(vision_patch_size)
                if isinstance(vision_patch_size, (list, tuple))
                else (vision_patch_size, vision_patch_size)
            )
            vision_position_embedding_kwargs.setdefault(
                "num_embeddings",
                (int(image_hw[0]) // int(patch_hw[0]))
                * (int(image_hw[1]) // int(patch_hw[1])),
            )
        PositionEmbedding.validate_kwargs(
            vision_position_embedding_type,
            vision_position_embedding_kwargs,
        )
        self.vision_position_embedding_type = vision_position_embedding_type
        self.vision_position_embedding_kwargs = vision_position_embedding_kwargs
        if vision_feature_stride is not None and vision_feature_spatial_shape is not None:
            raise ValueError(
                "Configure only one of vision_feature_stride and "
                "vision_feature_spatial_shape"
            )
        for name, value in (
            ("vision_feature_stride", vision_feature_stride),
            ("vision_feature_spatial_shape", vision_feature_spatial_shape),
        ):
            if value is None:
                continue
            values = value if isinstance(value, (list, tuple)) else (value, value)
            if len(values) != 2 or any(int(item) <= 0 for item in values):
                raise ValueError(f"{name} must be a positive int or pair")
        if vision_feature_prefix_tokens is not None and vision_feature_prefix_tokens < 0:
            raise ValueError("vision_feature_prefix_tokens must be non-negative")
        self.vision_feature_stride = vision_feature_stride
        self.vision_feature_spatial_shape = vision_feature_spatial_shape
        self.vision_feature_prefix_tokens = vision_feature_prefix_tokens
        self.audio_model_name = audio_model_name
        self.audio_encoder_type = audio_encoder_type
        self.audio_encoder_config = audio_encoder_config
        self.audio_in_channels = audio_in_channels
        self.audio_num_layers = audio_num_layers
        self.audio_stride = audio_stride
        self.audio_freq_stride = audio_freq_stride
        self.audio_time_stride = audio_time_stride
        self.audio_input_format = audio_input_format
        self.audio_processor_type = audio_processor_type
        self.audio_processor_name = audio_processor_name
        self.audio_sampling_rate = audio_sampling_rate
        self.audio_do_resample = audio_do_resample
        self.audio_processor_output_format = audio_processor_output_format
        self.audio_do_normalize = audio_do_normalize
        self.audio_n_fft = audio_n_fft
        self.audio_hop_length = audio_hop_length
        self.audio_win_length = audio_win_length
        self.audio_n_mels = audio_n_mels
        self.audio_power = audio_power
        self.omni_modalities = tuple(omni_modalities) if omni_modalities is not None else None

        # Projector
        self.projector_hidden_act = kwargs.pop("projector_hidden_act", "gelu")

        # Special tokens
        self.seq_token = seq_token
        self.cat_token = cat_token
        self.rel_token = rel_token
        self.parent_token = parent_token
        self.parent_token_index = parent_token_index
        self.embed_parent_token = embed_parent_token
        self.child_token = child_token
        self.obj_token = obj_token
        self.obj_token_index = obj_token_index
        self.embed_obj_token = embed_obj_token
        self.image_size = image_size
        self.vision_processor_type = vision_processor_type
        self.vision_processor_name = vision_processor_name
        self.vision_resize_size = vision_resize_size
        self.vision_center_crop_size = vision_center_crop_size
        self.vision_interpolation = vision_interpolation
        self.vision_do_rescale = vision_do_rescale
        self.vision_do_normalize = vision_do_normalize
        self.vision_image_mean = vision_image_mean
        self.vision_image_std = vision_image_std
        # Per-task parent tokens
        if per_task_parents is True:
            # Distinct parent tokens per task (use explicit overrides or defaults)
            self.ner_parent_token = ner_parent_token or "[ENT_P]"
            self.cat_parent_token = cat_parent_token or "[CAT_P]"
            self.open_rel_parent_token = open_rel_parent_token or "[REL_P]"
            self.struct_parent_token = struct_parent_token or "[STRUCT_P]"
        else:
            # Shared parent token (or explicit per-task overrides for backward compat)
            self.ner_parent_token = ner_parent_token or parent_token
            self.cat_parent_token = cat_parent_token or parent_token
            self.open_rel_parent_token = open_rel_parent_token or parent_token
            self.struct_parent_token = struct_parent_token or parent_token
        # Resolve auto-detect: if None, infer from whether tokens are actually distinct
        if per_task_parents is None:
            tokens = {self.ner_parent_token, self.cat_parent_token,
                      self.open_rel_parent_token, self.struct_parent_token}
            self.per_task_parents = len(tokens) > 1
        else:
            self.per_task_parents = per_task_parents

        # ── Backward compat: keep flat attributes for code that reads them ──
        self.relations_layer = relations_layer or (self.joint_relex_config.layer_type if self.joint_relex_config else None)
        self.classifier_layer = classifier_layer
        self.groups_layer = groups_layer or (self.structuring_config.anchor_mode if self.structuring_config else None)
        self.count_layer = count_layer or ("regression" if self.count_config else None)

        self.rel_mode = rel_mode
        self.pair_rep_type = self.joint_relex_config.pair_rep_type if self.joint_relex_config else pair_rep_type
        self.triples_layer = self.joint_relex_config.triples_layer if self.joint_relex_config else triples_layer
        self.embed_rel_token = self.joint_relex_config.embed_rel_token if self.joint_relex_config else embed_rel_token
        self.rel_token_index = self.joint_relex_config.rel_token_index if self.joint_relex_config else rel_token_index

        self.cat_token_index = self.classification_config.cat_token_index if self.classification_config else cat_token_index
        self.embed_cat_token = self.classification_config.embed_cat_token if self.classification_config else embed_cat_token
        self.represent_spans = self.ner_config.represent_spans if self.ner_config else represent_spans
        self.neg_spans_ratio = self.ner_config.neg_spans_ratio if self.ner_config else neg_spans_ratio
        self.span_loss_coef = self.ner_config.span_loss_coef if self.ner_config else span_loss_coef

        self.anchor_num_heads = self.structuring_config.anchor_num_heads if self.structuring_config else anchor_num_heads
        self.anchor_num_layers = self.structuring_config.anchor_num_layers if self.structuring_config else anchor_num_layers
        self.child_token_index = self.structuring_config.child_token_index if self.structuring_config else child_token_index
        self.embed_child_token = self.structuring_config.embed_child_token if self.structuring_config else embed_child_token

        self.count_mode = self.count_config.mode if self.count_config else count_mode
        self.max_count = max_count

        self.ner_loss_coef = self.ner_config.loss_coef if self.ner_config else ner_loss_coef
        self.cat_loss_coef = self.classification_config.loss_coef if self.classification_config else cat_loss_coef
        self.image_classification_loss_coef = (
            self.image_classification_config.loss_coef
            if self.image_classification_config else image_classification_loss_coef
        )
        self.audio_classification_loss_coef = (
            self.audio_classification_config.loss_coef
            if self.audio_classification_config else audio_classification_loss_coef
        )
        self.object_detection_loss_coef = (
            self.object_detection_config.loss_coef
            if self.object_detection_config else object_detection_loss_coef
        )
        self.segmentation_loss_coef = (
            self.segmentation_config.loss_coef
            if self.segmentation_config else segmentation_loss_coef
        )
        self.audio_segmentation_loss_coef = (
            self.audio_segmentation_config.loss_coef
            if self.audio_segmentation_config else audio_segmentation_loss_coef
        )
        self.rel_loss_coef = self.joint_relex_config.loss_coef if self.joint_relex_config else rel_loss_coef
        self.adjacency_loss_coef = self.joint_relex_config.adjacency_loss_coef if self.joint_relex_config else adjacency_loss_coef
        self.count_loss_coef = self.count_config.loss_coef if self.count_config else count_loss_coef
        self.groups_loss_coef = groups_loss_coef
        self.embedding_loss_coef = self.embedding_config.loss_coef if self.embedding_config else embedding_loss_coef
        self.structuring_loss_coef = self.structuring_config.loss_coef if self.structuring_config else structuring_loss_coef

        # Shared layers config
        self.shared_anchor_modeling = shared_anchor_modeling
        self.shared_anchor_refine_layers = shared_anchor_refine_layers
        self.shared_anchor_refine_heads = shared_anchor_refine_heads

    @property
    def uses_per_task_parents(self) -> bool:
        """True when per-task parent tokens are distinct from each other."""
        return self.per_task_parents

    def get_task_config(self, task_name: str):
        """Return the task sub-config for a canonical task name."""
        attr_name = self.TASK_CONFIG_ATTRS.get(task_name)
        if attr_name is None:
            return None
        return getattr(self, attr_name, None)

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        for key, value in output.items():
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                output[key] = dataclasses.asdict(value)
        output["model_type"] = self.model_type
        return output


_TEXT_CONFIG_FIELDS = (
    "ner_config",
    "classification_config",
    "joint_relex_config",
    "open_relex_config",
    "structuring_config",
    "count_config",
    "embedding_config",
)
_VISION_CONFIG_FIELDS = (
    "image_classification_config",
    "object_detection_config",
    "segmentation_config",
)
_AUDIO_CONFIG_FIELDS = (
    "audio_classification_config",
    "audio_segmentation_config",
)

_BASE_SERIALIZED_FIELDS = frozenset(
    {
        # Base GLiNER/encoder fields.
        "model_type",
        "model_name",
        "name",
        "max_width",
        "hidden_size",
        "dropout",
        "fine_tune",
        "subtoken_pooling",
        "span_mode",
        "post_fusion_schema",
        "num_post_fusion_layers",
        "vocab_size",
        "max_neg_type_ratio",
        "max_types",
        "max_len",
        "words_splitter_type",
        "num_rnn_layers",
        "fuse_layers",
        "embed_ent_token",
        "class_token_index",
        "encoder_config",
        "ent_token",
        "sep_token",
        "_attn_implementation",
        "token_loss_coef",
        "span_loss_coef",
        "represent_spans",
        "neg_spans_ratio",
        # Shared GLiNExT fields.
        "shared_anchor_modeling",
        "shared_anchor_refine_layers",
        "shared_anchor_refine_heads",
        "labels_encoder",
        "labels_encoder_config",
        "backbone_type",
        "model_variant",
        "multimodal_fusion",
        "media_parent_embedding_source",
        "use_layout",
        "seq_token",
        "cat_token",
        "rel_token",
        "parent_token",
        "child_token",
        "obj_token",
        "per_task_parents",
        "ner_parent_token",
        "cat_parent_token",
        "open_rel_parent_token",
        "struct_parent_token",
        "parent_token_index",
        "embed_parent_token",
        "projector_hidden_act",
        # Token ids commonly supplied by PretrainedConfig.
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "transformers_version",
    }
)

_TEXT_SERIALIZED_FIELDS = frozenset(
    {
        "ner_config",
        "classification_config",
        "joint_relex_config",
        "relations_config",
        "open_relex_config",
        "structuring_config",
        "count_config",
        "embedding_config",
        "relations_layer",
        "classifier_layer",
        "groups_layer",
        "count_layer",
        "rel_mode",
        "pair_rep_type",
        "triples_layer",
        "embed_rel_token",
        "rel_token_index",
        "cat_token_index",
        "embed_cat_token",
        "ner_loss_coef",
        "cat_loss_coef",
        "rel_loss_coef",
        "adjacency_loss_coef",
        "count_loss_coef",
        "groups_loss_coef",
        "embedding_loss_coef",
        "structuring_loss_coef",
        "count_mode",
        "max_count",
        "anchor_num_heads",
        "anchor_num_layers",
        "child_token_index",
        "embed_child_token",
    }
)

_VISION_SERIALIZED_FIELDS = frozenset(
    {
        "image_classification_config",
        "object_detection_config",
        "segmentation_config",
        "vision_model_name",
        "vision_encoder_type",
        "vision_encoder_config",
        "vision_in_channels",
        "vision_patch_size",
        "vision_position_embedding_type",
        "vision_position_embedding_kwargs",
        "vision_feature_stride",
        "vision_feature_spatial_shape",
        "vision_feature_prefix_tokens",
        "obj_token_index",
        "embed_obj_token",
        "image_size",
        "vision_processor_type",
        "vision_processor_name",
        "vision_resize_size",
        "vision_center_crop_size",
        "vision_interpolation",
        "vision_do_rescale",
        "vision_do_normalize",
        "vision_image_mean",
        "vision_image_std",
        "image_classification_loss_coef",
        "object_detection_loss_coef",
        "segmentation_loss_coef",
    }
)

_AUDIO_SERIALIZED_FIELDS = frozenset(
    {
        "audio_classification_config",
        "audio_segmentation_config",
        "audio_model_name",
        "audio_encoder_type",
        "audio_encoder_config",
        "audio_in_channels",
        "audio_num_layers",
        "audio_stride",
        "audio_freq_stride",
        "audio_time_stride",
        "audio_input_format",
        "audio_processor_type",
        "audio_processor_name",
        "audio_sampling_rate",
        "audio_do_resample",
        "audio_processor_output_format",
        "audio_do_normalize",
        "audio_n_fft",
        "audio_hop_length",
        "audio_win_length",
        "audio_n_mels",
        "audio_power",
        "obj_token_index",
        "embed_obj_token",
        "audio_classification_loss_coef",
        "audio_segmentation_loss_coef",
    }
)

_LAYOUT_SERIALIZED_FIELDS = _TEXT_SERIALIZED_FIELDS | frozenset(
    {
        "image_size",
        "vision_processor_type",
        "vision_processor_name",
        "vision_resize_size",
        "vision_center_crop_size",
        "vision_interpolation",
        "vision_do_rescale",
        "vision_do_normalize",
        "vision_image_mean",
        "vision_image_std",
        "layout_image_tokens",
        "max_page_embeddings",
    }
)

_OMNI_SERIALIZED_FIELDS = (
    _TEXT_SERIALIZED_FIELDS
    | _VISION_SERIALIZED_FIELDS
    | _AUDIO_SERIALIZED_FIELDS
    | frozenset(
        {
            "omni_modalities",
            "layout_image_tokens",
            "max_page_embeddings",
        }
    )
)


_MEDIA_UNUSED_SERIALIZED_FIELDS = frozenset(
    {
        "represent_spans",
        "neg_spans_ratio",
        "span_loss_coef",
        "token_loss_coef",
    }
)


def _serialize_only(
    config: GLiNextConfig,
    field_names: frozenset[str],
    *,
    excluded: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    output = GLiNextConfig.to_dict(config)
    allowed = (_BASE_SERIALIZED_FIELDS | field_names) - excluded
    return {key: value for key, value in output.items() if key in allowed}


def _non_null_config_names(config: GLiNextConfig, names: tuple[str, ...]) -> list[str]:
    return [name for name in names if getattr(config, name, None) is not None]


class GLiNextTextConfig(GLiNextConfig):
    """Text-only GLiNExT config.

    This keeps the legacy text defaults, including enabling NER when no explicit
    ``ner_config`` is provided, but rejects vision/audio task configs.
    """

    model_type = "glinext-text"
    expected_model_variant = "text"

    def __init__(self, *args, model_variant: str = "text", **kwargs):
        kwargs.setdefault("default_ner_config", True)
        super().__init__(*args, model_variant=model_variant, **kwargs)
        disallowed = _non_null_config_names(self, _VISION_CONFIG_FIELDS + _AUDIO_CONFIG_FIELDS)
        if disallowed:
            raise ValueError(
                f"{self.__class__.__name__} supports text tasks only; "
                f"received configs: {', '.join(disallowed)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return _serialize_only(self, _TEXT_SERIALIZED_FIELDS)


class GLiNextLayoutConfig(GLiNextTextConfig):
    """Text config with layout coordinates enabled."""

    model_type = "glinext-layout"
    expected_model_variant = "layout"

    def __init__(
        self,
        *args,
        model_variant: str = "layout",
        use_layout: bool = True,
        **kwargs,
    ):
        super().__init__(*args, model_variant=model_variant, use_layout=use_layout, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return _serialize_only(self, _LAYOUT_SERIALIZED_FIELDS)


class GLiNextVisionConfig(GLiNextConfig):
    """Vision-only bi-encoder GLiNExT config."""

    model_type = "glinext-vision"
    expected_model_variant = "vision"

    def __init__(self, *args, model_variant: str = "vision", **kwargs):
        kwargs.setdefault("default_ner_config", False)
        super().__init__(*args, model_variant=model_variant, **kwargs)
        disallowed = _non_null_config_names(self, _TEXT_CONFIG_FIELDS + _AUDIO_CONFIG_FIELDS)
        if disallowed:
            raise ValueError(
                f"{self.__class__.__name__} supports vision tasks only; "
                f"received configs: {', '.join(disallowed)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return _serialize_only(
            self,
            _VISION_SERIALIZED_FIELDS,
            excluded=_MEDIA_UNUSED_SERIALIZED_FIELDS,
        )


class GLiNextAudioConfig(GLiNextConfig):
    """Audio-only bi-encoder GLiNExT config."""

    model_type = "glinext-audio"
    expected_model_variant = "audio"

    def __init__(self, *args, model_variant: str = "audio", **kwargs):
        kwargs.setdefault("default_ner_config", False)
        super().__init__(*args, model_variant=model_variant, **kwargs)
        disallowed = _non_null_config_names(self, _TEXT_CONFIG_FIELDS + _VISION_CONFIG_FIELDS)
        if disallowed:
            raise ValueError(
                f"{self.__class__.__name__} supports audio tasks only; "
                f"received configs: {', '.join(disallowed)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return _serialize_only(
            self,
            _AUDIO_SERIALIZED_FIELDS,
            excluded=_MEDIA_UNUSED_SERIALIZED_FIELDS,
        )


class GLiNextOmniConfig(GLiNextConfig):
    """Omni config that can combine text, vision, audio, and layout settings."""

    model_type = "glinext-omni"
    expected_model_variant = "omni"

    def __init__(self, *args, model_variant: str = "omni", **kwargs):
        kwargs.setdefault("default_ner_config", True)
        super().__init__(*args, model_variant=model_variant, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return _serialize_only(self, _OMNI_SERIALIZED_FIELDS)


GLINEXT_MODEL_TYPE_TO_CONFIG_CLASS = {
    GLiNextTextConfig.model_type: GLiNextTextConfig,
    GLiNextLayoutConfig.model_type: GLiNextLayoutConfig,
    GLiNextVisionConfig.model_type: GLiNextVisionConfig,
    GLiNextAudioConfig.model_type: GLiNextAudioConfig,
    GLiNextOmniConfig.model_type: GLiNextOmniConfig,
}

GLINEXT_MODEL_TYPE_TO_VARIANT = {
    GLiNextTextConfig.model_type: "text",
    GLiNextLayoutConfig.model_type: "layout",
    GLiNextVisionConfig.model_type: "vision",
    GLiNextAudioConfig.model_type: "audio",
    GLiNextOmniConfig.model_type: "omni",
}

GLINEXT_VARIANT_TO_CONFIG_CLASS = {
    "text": GLiNextTextConfig,
    "layout": GLiNextLayoutConfig,
    "vision": GLiNextVisionConfig,
    "audio": GLiNextAudioConfig,
    "omni": GLiNextOmniConfig,
}


def resolve_glinext_config_class(config_dict: dict[str, Any]):
    model_type = config_dict.get("model_type")
    if model_type in GLINEXT_MODEL_TYPE_TO_CONFIG_CLASS:
        return GLINEXT_MODEL_TYPE_TO_CONFIG_CLASS[model_type]
    if model_type not in {None, GLiNextConfig.model_type}:
        raise ValueError(
            "model_type must be one of "
            f"{sorted(GLINEXT_MODEL_TYPE_TO_CONFIG_CLASS)}, got {model_type!r}."
        )

    variant = config_dict.get("model_variant") or "text"
    try:
        return GLINEXT_VARIANT_TO_CONFIG_CLASS[variant]
    except KeyError as exc:
        raise ValueError(
            "model_variant must be one of "
            f"{sorted(GLINEXT_VARIANT_TO_CONFIG_CLASS)}, got {variant!r}."
        ) from exc
