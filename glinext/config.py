import dataclasses
from dataclasses import dataclass
from typing import Any, Optional

from transformers.models.auto import CONFIG_MAPPING

from gliner.config import BaseGLiNERConfig


@dataclass
class BaseHeadConfig:
    """Base config for all task heads that use the anchor paradigm."""
    loss_coef: float = 1.0
    anchor_mode: str = "parent"
    anchor_modeling: str = "linear"
    anchor_refine_layers: int = 0
    anchor_refine_heads: int = 8
    represent_spans: bool = False
    neg_spans_ratio: float = 1.0
    span_loss_coef: float = 1.0
    parent_token_index: int = -1
    embed_parent_token: bool = True


@dataclass
class NERHeadConfig(BaseHeadConfig):
    pass


@dataclass
class ClassificationHeadConfig(BaseHeadConfig):
    cat_token_index: int = -1
    embed_cat_token: bool = True
    pooling_type: str = "mean"  # "mean", "cls", "max"
    scorer_type: str = "dot"  # "dot", "weighted-dot", "mlp", "hopfield"


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
    anchor_mode: str = "fixed"          # "fixed", "rotary", "query_lstm", "query_transformer"
    num_fixed_slots: int = 10
    max_count: int = 20
    anchor_num_heads: int = 4
    anchor_num_layers: int = 2
    rel_token_index: int = -1
    embed_rel_token: bool = True


@dataclass
class StructuringHeadConfig(BaseHeadConfig):
    anchor_mode: str = "lstm"  # "lstm", "query_lstm", "query_transformer", "fixed"
    anchor_num_heads: int = 4
    anchor_num_layers: int = 2
    max_count: int = 20
    num_fixed_slots: int = 10  # number of learnable anchor slots (for anchor_mode="fixed")
    child_token_index: int = -1
    embed_child_token: bool = True


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


class GLiNextConfig(BaseGLiNERConfig):
    model_type = "glinext"

    def __init__(
        self,
        # Per-task sub-configs (None = disabled, dict or dataclass = enabled)
        ner_config: Optional[dict] = None,
        classification_config: Optional[dict] = None,
        relations_config: Optional[dict] = None,  # backward compat alias for joint_relex_config
        joint_relex_config: Optional[dict] = None,
        open_relex_config: Optional[dict] = None,
        structuring_config: Optional[dict] = None,
        count_config: Optional[dict] = None,
        embedding_config: Optional[dict] = None,
        # Shared layers across tasks (None = each task creates its own)
        shared_anchor_modeling: Optional[str] = None,  # "linear", "lstm", "mlp" — shared AnchorModeling layer
        shared_anchor_refine_layers: int = 0,  # shared AnchorCrossAttentionLayer (0 = disabled)
        shared_anchor_refine_heads: int = 8,
        # Labels encoder (bi-encoder style)
        labels_encoder: Optional[str] = None,
        labels_encoder_config: Optional[dict] = None,
        # Special tokens
        seq_token: str = "[SEQ]",
        cat_token: str = "[CAT]",
        rel_token: str = "[REL]",
        parent_token: str = "[PARENT]",
        child_token: str = "[CHILD]",
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
        **kwargs,
    ):
        super().__init__(**kwargs)

        # ── Migrate flat params to sub-configs if sub-configs not provided ──

        # NER: always on by default (use flat params if no sub-config)
        if ner_config is None:
            ner_config = {}
        if isinstance(ner_config, dict):
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
            classification_config.pop("layer_type", None)  # backward compat: layer_type removed
            self.classification_config = ClassificationHeadConfig(**classification_config)
        else:
            self.classification_config = classification_config

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
            labels_encoder_config["model_type"] = labels_encoder_config.get("model_type", "deberta-v2")
            labels_encoder_config = CONFIG_MAPPING[labels_encoder_config["model_type"]](**labels_encoder_config)
        self.labels_encoder = labels_encoder
        self.labels_encoder_config = labels_encoder_config

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

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        for key, value in output.items():
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                output[key] = dataclasses.asdict(value)
        return output
