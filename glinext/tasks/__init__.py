"""Task head abstractions for GLiNExT modular architecture."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import torch
from torch import nn
from transformers.utils import ModelOutput

from ..layers import (
    AnchorCrossAttentionLayer,
    AnchorLayer,
    AnchorModeling,
    AnchorNormalizer,
    AttentionBias,
    RefinementPositionEncoding,
)


@dataclass
class SharedRepresentations:
    """Shared encoder outputs passed to all task heads."""
    token_embeds: torch.Tensor           # (B_enc, S, D) raw encoder output
    input_ids: torch.Tensor              # (B_enc, S)
    attention_mask: torch.Tensor         # (B_enc, S)
    words_embedding: torch.Tensor        # (B, W, D) word-level embeddings
    mask: torch.Tensor                   # (B, W) valid word mask
    prompts_embedding: torch.Tensor      # (B, C, D) prompt embeddings
    prompts_embedding_mask: torch.Tensor # (B, C) prompt mask
    vision_embedding: Optional[torch.Tensor] = None  # (B, V, D) visual token embeddings
    vision_mask: Optional[torch.Tensor] = None        # (B, V) valid visual token mask
    audio_embedding: Optional[torch.Tensor] = None    # (B, A, D) audio token embeddings
    audio_mask: Optional[torch.Tensor] = None         # (B, A) valid audio token mask
    image_sizes: Optional[torch.Tensor] = None        # (B, 2) original image sizes as (height, width)


@dataclass
class TaskFlatInputs:
    """Per-task flattened representations with BN indexing.

    BN = total number of subtask groups across the batch for a specific task.
    E.g., if batch item 0 has 3 NER schemas and item 1 has 1, BN=4 for NER.
    """
    words_embedding: torch.Tensor     # (BN, W, D) backward-compatible text/feature embeddings
    mask: torch.Tensor                # (BN, W) backward-compatible text/feature mask
    parent_embedding: torch.Tensor    # (BN, D) parent embedding per group
    child_embedding: torch.Tensor     # (BN, max_C, D) child embeddings per group
    child_mask: torch.Tensor          # (BN, max_C) child mask per group
    batch_origin: torch.Tensor        # (BN,) maps flat idx to original batch idx
    feature_embedding: Optional[torch.Tensor] = None  # (BN, L, D) task input features
    feature_mask: Optional[torch.Tensor] = None        # (BN, L) task input mask
    feature_spatial_shape: Optional[torch.Tensor] = None  # (BN, 2) dense height/width
    feature_prefix_tokens: Optional[torch.Tensor] = None  # (BN,) non-spatial token count


@dataclass
class TaskHeadOutput(ModelOutput):
    """Output from a single task head."""
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    extra: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.extra is None:
            self.extra = {}


class StructuringHeadExtra(TypedDict, total=False):
    """Canonical extras emitted by the anchor-conditioned structuring head."""

    groups_output: torch.Tensor
    anchor_mask: torch.Tensor
    objectness_logits: torch.Tensor | None
    span_logits: torch.Tensor | None
    span_idx: torch.Tensor | None
    span_mask: torch.Tensor | None
    loss_stats: dict[str, Any] | None
    anchor_relation_scores: torch.Tensor | None
    anchor_relation_loss: torch.Tensor | None
    anchor_matches: list[list[tuple[int, int]]] | None


class SetStructuringHeadExtra(StructuringHeadExtra, total=False):
    """Canonical entity-first structuring tensors plus compatibility aliases."""

    entity_logits: torch.Tensor
    entity_spans: torch.Tensor
    entity_mask: torch.Tensor
    entity_representations: torch.Tensor
    entity_field_logits: torch.Tensor
    membership_logits: torch.Tensor
    entity_assignment_logits: torch.Tensor
    entity_anchor_logits: torch.Tensor
    assignment_logits: torch.Tensor
    structuring_logits: torch.Tensor
    entity_loss: torch.Tensor | None
    assignment_loss: torch.Tensor | None
    objectness_loss: torch.Tensor | None


@dataclass
class StructuringTaskHeadOutput(TaskHeadOutput):
    extra: StructuringHeadExtra | None = None


@dataclass
class SetStructuringTaskHeadOutput(TaskHeadOutput):
    extra: SetStructuringHeadExtra | None = None


class TaskHead(ABC, nn.Module):
    """Abstract base class for task heads."""
    name: str = ""
    dependencies: List[str] = []
    loss_coef: float = 1.0

    @staticmethod
    def _build_anchor_layer(task_cfg, hidden_size, dropout):
        anchor_spec = getattr(task_cfg, "anchor_layer", None)
        if anchor_spec is None:
            anchor_spec = getattr(task_cfg, "anchor_mode", "parent")
        return AnchorLayer.from_config(
            anchor_spec,
            hidden_size,
            max_count=getattr(task_cfg, "max_count", 20),
            num_slots=getattr(task_cfg, "num_fixed_slots", 10),
            num_heads=getattr(task_cfg, "anchor_num_heads", 4),
            num_layers=getattr(task_cfg, "anchor_num_layers", 2),
            dropout=dropout,
            feature_mlp=getattr(task_cfg, "feature_anchor_mlp", False),
            feature_mlp_hidden_multiplier=getattr(
                task_cfg,
                "feature_anchor_mlp_hidden_multiplier",
                1,
            ),
            context_gate_init=getattr(
                task_cfg,
                "anchor_context_gate_init",
                0.1,
            ),
            context_gate_trainable=getattr(
                task_cfg,
                "anchor_context_gate_trainable",
                True,
            ),
            position_bucket_normalization=getattr(
                task_cfg,
                "position_bucket_normalization",
                "none",
            ),
        )

    @staticmethod
    def _build_anchor_normalizer(task_cfg, hidden_size):
        return AnchorNormalizer.from_config(
            getattr(task_cfg, "anchor_normalization", "none"),
            hidden_size,
        )

    @staticmethod
    def _build_anchor_refinement(
        task_cfg,
        hidden_size,
        dropout,
        shared_layers,
    ):
        if "anchor_refine" in shared_layers:
            return shared_layers["anchor_refine"]
        refinement_spec = getattr(task_cfg, "anchor_refinement", None)
        if refinement_spec is None:
            refinement_spec = {
                "type": "cross_attention",
                "params": {
                    "num_heads": getattr(task_cfg, "anchor_refine_heads", 8),
                    "num_layers": getattr(task_cfg, "anchor_refine_layers", 0),
                    "dropout": dropout,
                    "norm_style": getattr(
                        task_cfg,
                        "anchor_refine_norm",
                        "post_norm",
                    ),
                    "layer_scale_init": getattr(
                        task_cfg,
                        "anchor_refine_layer_scale_init",
                        None,
                    ),
                },
            }
        return AnchorCrossAttentionLayer.from_config(
            refinement_spec,
            hidden_size,
            dropout=dropout,
        )

    @staticmethod
    def _position_component_spec(task_cfg, role):
        component = getattr(task_cfg, f"anchor_{role}_position", None)
        if component is not None:
            return component
        return {
            "type": getattr(
                task_cfg,
                f"{role}_position_embedding_type",
                "none",
            ),
            "params": dict(
                getattr(
                    task_cfg,
                    f"{role}_position_embedding_kwargs",
                    None,
                )
                or {}
            ),
        }

    @classmethod
    def _build_refinement_positions(cls, task_cfg, hidden_size):
        memory_usage = getattr(
            task_cfg,
            "anchor_memory_position_usage",
            None,
        )
        if memory_usage is None:
            memory_usage = (
                "keys_and_values"
                if getattr(task_cfg, "memory_position_in_values", False)
                else "keys_only"
            )
        capacity_resolver = getattr(
            task_cfg,
            "effective_anchor_num_slots",
            None,
        )
        num_query_embeddings = (
            capacity_resolver()
            if capacity_resolver is not None
            else getattr(task_cfg, "num_fixed_slots", None)
        )
        return RefinementPositionEncoding.from_config(
            {
                "memory": cls._position_component_spec(task_cfg, "memory"),
                "query": cls._position_component_spec(task_cfg, "query"),
                "memory_usage": memory_usage,
            },
            hidden_size,
            num_query_embeddings=num_query_embeddings,
        )

    @staticmethod
    def _legacy_cross_attention_bias_spec(task_cfg):
        bucket_bias = getattr(
            task_cfg,
            "position_bucket_attention_bias_type",
            "none",
        )
        if bucket_bias != "none":
            return {
                "type": "gaussian_distance",
                "params": {
                    "sigma": getattr(
                        task_cfg,
                        "position_bucket_attention_sigma",
                        0.5,
                    ),
                    "weight": getattr(
                        task_cfg,
                        "position_bucket_attention_bias_weight",
                        1.0,
                    ),
                    "units": "query_steps",
                },
            }
        spatial_bias = getattr(
            task_cfg,
            "spatial_attention_bias_type",
            "none",
        )
        if spatial_bias != "none":
            return {
                "type": "gaussian_distance",
                "params": {
                    "sigma": getattr(
                        task_cfg,
                        "spatial_attention_sigma",
                        0.2,
                    ),
                    "weight": getattr(
                        task_cfg,
                        "spatial_attention_bias_weight",
                        1.0,
                    ),
                    "units": "normalized",
                    "query_dimensions": [0, 1],
                    "key_dimensions": [0, 1],
                },
            }
        return None

    @classmethod
    def _build_refinement_biases(cls, task_cfg, num_heads):
        self_bias_spec = getattr(
            task_cfg,
            "anchor_self_attention_bias",
            None,
        )
        cross_bias_spec = getattr(
            task_cfg,
            "anchor_cross_attention_bias",
            None,
        )
        if cross_bias_spec is None:
            cross_bias_spec = cls._legacy_cross_attention_bias_spec(task_cfg)
        return (
            AttentionBias.from_config(self_bias_spec, num_heads=num_heads),
            AttentionBias.from_config(cross_bias_spec, num_heads=num_heads),
        )

    def _init_anchor_components(
        self,
        task_cfg,
        config,
        hidden_size,
        dropout,
        shared_layers,
    ):
        """Initialize independent anchor components owned by this task head.

        Span representations are added only for configs in the span-capable
        text hierarchy; media and classification heads do not receive dead span
        attributes.
        """
        self.anchor_layer = self._build_anchor_layer(
            task_cfg,
            hidden_size,
            dropout,
        )
        self.anchor_normalizer = self._build_anchor_normalizer(
            task_cfg,
            hidden_size,
        )

        if "anchor_modeling" in shared_layers:
            self.anchor_modeling = shared_layers["anchor_modeling"]
        else:
            self.anchor_modeling = AnchorModeling.from_config(
                getattr(task_cfg, "anchor_modeling", "linear"),
                hidden_size, dropout=dropout,
            )

        refinement = self._build_anchor_refinement(
            task_cfg,
            hidden_size,
            dropout,
            shared_layers,
        )
        if refinement is not None:
            self.anchor_refine = refinement

        self.anchor_refine_positions = None
        self.anchor_self_attention_bias = None
        self.anchor_cross_attention_bias = None
        if hasattr(self, "anchor_refine"):
            self.anchor_refine_positions = self._build_refinement_positions(
                task_cfg,
                hidden_size,
            )
            (
                self.anchor_self_attention_bias,
                self.anchor_cross_attention_bias,
            ) = self._build_refinement_biases(
                task_cfg,
                num_heads=self.anchor_refine.num_heads,
            )

        if hasattr(task_cfg, "represent_spans"):
            self.represent_spans = bool(task_cfg.represent_spans)
            self.span_loss_coef = float(task_cfg.span_loss_coef)
        if getattr(self, "represent_spans", False):
            from gliner.modeling.span_rep import SpanRepLayer
            self.span_rep_layer = SpanRepLayer(
                span_mode="token_level",
                hidden_size=hidden_size,
                max_width=getattr(config, "max_width", 12),
                dropout=dropout,
            )

    # Compatibility for external task heads that still call the old helper.
    def _init_anchor_pipeline(self, *args, **kwargs):
        return self._init_anchor_components(*args, **kwargs)

    def _generate_anchors(
        self,
        context_embedding: torch.Tensor,
        feature_embeddings: torch.Tensor | None = None,
        *,
        count: torch.Tensor | None = None,
        threshold: float = 0.5,
        feature_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchors, anchor_mask = self.anchor_layer(
            context_embedding,
            feature_embeddings,
            count=count,
            threshold=threshold,
            feature_mask=feature_mask,
        )
        anchors = self.anchor_normalizer(
            anchors,
            anchor_mask,
            source_embeddings=feature_embeddings,
            source_mask=feature_mask,
        )
        return anchors, anchor_mask

    def _refine_anchors(
        self,
        anchors: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_mask: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
        query_coordinates: torch.Tensor | None = None,
        memory_coordinates: torch.Tensor | None = None,
        memory_positions: torch.Tensor | None = None,
        query_spatial_shape: tuple[int, int] | None = None,
        memory_spatial_shape: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if not hasattr(self, "anchor_refine"):
            return anchors
        if not isinstance(self.anchor_refine, AnchorCrossAttentionLayer):
            return self.anchor_refine(
                anchors,
                memory,
                token_mask=memory_mask,
                query_mask=anchor_mask,
            )
        return self.anchor_refine(
            anchors,
            memory,
            token_mask=memory_mask,
            query_mask=anchor_mask,
            query_coordinates=query_coordinates,
            memory_coordinates=memory_coordinates,
            memory_pos_emb=memory_positions,
            query_spatial_shape=query_spatial_shape,
            memory_spatial_shape=memory_spatial_shape,
            position_encoding=self.anchor_refine_positions,
            memory_position_in_values=(
                self.anchor_refine_positions.memory_position_in_values
                if self.anchor_refine_positions is not None
                else False
            ),
            self_attention_bias_module=self.anchor_self_attention_bias,
            cross_attention_bias_module=self.anchor_cross_attention_bias,
        )

    @property
    def memory_position_embedding(self):
        positions = getattr(self, "anchor_refine_positions", None)
        return None if positions is None else positions.memory_embedding

    @property
    def query_position_embedding(self):
        positions = getattr(self, "anchor_refine_positions", None)
        return None if positions is None else positions.query_embedding

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Migrate modality-owned position modules into the shared object."""

        current_keys = set(self.state_dict())
        for old_prefix, new_prefix in (
            (
                "memory_position_embedding.",
                "anchor_refine_positions.memory_embedding.",
            ),
            (
                "query_position_embedding.",
                "anchor_refine_positions.query_embedding.",
            ),
        ):
            for old_key in tuple(state_dict):
                qualified_old_prefix = f"{prefix}{old_prefix}"
                if not old_key.startswith(qualified_old_prefix):
                    continue
                suffix = old_key[len(qualified_old_prefix) :]
                new_name = f"{new_prefix}{suffix}"
                new_key = f"{prefix}{new_name}"
                if new_name in current_keys and new_key not in state_dict:
                    state_dict[new_key] = state_dict[old_key]
                state_dict.pop(old_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _model_anchors(
        self,
        anchors: torch.Tensor,
        children: torch.Tensor,
        *,
        anchor_mask: torch.Tensor | None = None,
        child_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.anchor_modeling(
            anchors,
            children,
            anchor_mask=anchor_mask,
            child_mask=child_mask,
        )

    def _reduce_fused_anchors(self, fused: torch.Tensor, anchor_mask: torch.Tensor) -> torch.Tensor:
        if fused.dim() != 4:
            return fused
        if fused.shape[1] == 1:
            return fused.squeeze(1)

        anchor_weights = anchor_mask.float()
        fused = (fused * anchor_weights[:, :, None, None]).sum(dim=1)
        return fused / anchor_weights.sum(dim=1).clamp(min=1)[:, None, None]

    @abstractmethod
    def forward(
        self,
        shared: SharedRepresentations,
        dependency_outputs: Dict[str, TaskHeadOutput],
        **batch,
    ) -> TaskHeadOutput:
        ...

    @classmethod
    def from_config(cls, config, **kwargs) -> Optional["TaskHead"]:
        """Construct from GLiNextConfig, returning None if this head is disabled."""
        ...


class TaskDecoder(ABC):
    """Abstract base class for task-specific decoders."""

    def __init__(self, config):
        self.config = config

    @classmethod
    def from_config(cls, config, **kwargs) -> "TaskDecoder":
        """Construct decoder from config."""
        return cls(config, **kwargs)

    @abstractmethod
    def decode(self, model_output, classes_mapping=None, **kwargs):
        """Decode model output into structured predictions."""
        ...

    def map_results(
        self,
        task_results: list,
        valid_to_orig_idx: List[int],
        all_start_maps: List[List[int]],
        all_end_maps: List[List[int]],
        valid_texts: List[str],
        num_original: int,
        **kwargs,
    ) -> List:
        """Map decoded task results back to original input order."""
        output = [[] for _ in range(num_original)]
        for valid_i, result in enumerate(task_results):
            orig_i = valid_to_orig_idx[valid_i]
            output[orig_i] = result
        return output


class TaskProcessor(ABC):
    """Abstract base class for task-specific data processors."""

    def __init__(self, config, tokenizer=None, words_splitter=None, **kwargs):
        self.config = config

    @staticmethod
    def _normalize_label_groups(labels) -> Dict[str, List[str]]:
        """Normalize ``List[str]`` or ``Dict[str, List[str]]`` to dict form."""
        if labels is None:
            return {}
        if isinstance(labels, list):
            return {None: list(dict.fromkeys(labels))}
        if isinstance(labels, dict):
            return {k: list(dict.fromkeys(v)) for k, v in labels.items()}
        raise TypeError(f"Expected list or dict for labels, got {type(labels)}")

    @abstractmethod
    def get_classes_mapping(self, batch_list, **kwargs):
        """Build task-specific class mappings from raw data."""
        ...

    @abstractmethod
    def create_labels(self, batch_list, classes_mapping, **kwargs) -> Optional[Dict[str, torch.Tensor]]:
        """Create task-specific label tensors."""
        ...

    def contribute_prompt(self, classes_mapping, batch_idx, use_labels_encoder=False) -> List[str]:
        """Return prompt tokens for this task in a single batch item."""
        return []

    def get_augmentable_label_groups(self, batch_list, classes_mapping):
        """Return label-bearing groups exposed to training-time augmentation.

        Mapping-less tasks inherit the empty implementation. Concrete task
        processors opt in by returning ``AugmentableLabelGroup`` descriptors;
        keeping this hook non-abstract preserves compatibility with external
        task processors.
        """
        return []

    def resolve_spans(self, item):
        """Resolve text spans to token indices (task-specific)."""
        pass

    def contribute_inference_input(self, item: Dict[str, Any], **kwargs):
        """Add task-specific inference stubs to one collator input item."""
        return None

    def empty_inference_result(self, num_texts: int, **kwargs) -> Optional[Dict[str, List]]:
        """Return this task's empty inference result when requested."""
        return None

    def prepare_label_encoder_inputs(self, classes_mapping, labels_tokenizer) -> Optional[Dict]:
        """Tokenize label strings for the labels encoder."""
        return None


@dataclass(frozen=True)
class TaskDefinition:
    """Registry entry for a concrete GLiNExT task."""

    name: str
    head_module: str
    head_class_name: str
    modality: str

    def load_head_class(self):
        module = import_module(self.head_module)
        return getattr(module, self.head_class_name)


class TaskRegistry:
    """Ordered task registry used by model orchestration."""

    def __init__(self, definitions: List[TaskDefinition]):
        self._definitions = tuple(definitions)
        self._by_name = {definition.name: definition for definition in self._definitions}

    def __iter__(self):
        return iter(self._definitions)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def get(self, name: str) -> Optional[TaskDefinition]:
        return self._by_name.get(name)

    @property
    def execution_order(self) -> Tuple[str, ...]:
        return tuple(definition.name for definition in self._definitions)

    def task_names_for_modality(self, modality: str) -> Tuple[str, ...]:
        return tuple(
            definition.name
            for definition in self._definitions
            if definition.modality == modality
        )

    @property
    def text_tasks(self) -> Tuple[str, ...]:
        return self.task_names_for_modality("text")

    @property
    def vision_tasks(self) -> Tuple[str, ...]:
        return self.task_names_for_modality("vision")

    @property
    def audio_tasks(self) -> Tuple[str, ...]:
        return self.task_names_for_modality("audio")

    def head_classes(self):
        for definition in self._definitions:
            yield definition.load_head_class()


TASK_REGISTRY = TaskRegistry(
    [
        TaskDefinition("ner", "glinext.tasks.ner.model", "NERHead", "text"),
        TaskDefinition("classification", "glinext.tasks.classification.model", "ClassificationHead", "text"),
        TaskDefinition("count", "glinext.tasks.count.model", "CountHead", "text"),
        TaskDefinition("joint_relex", "glinext.tasks.joint_relex.model", "JointRelexHead", "text"),
        TaskDefinition("open_relex", "glinext.tasks.open_relex.model", "OpenRelexHead", "text"),
        TaskDefinition(
            "set_open_relex",
            "glinext.tasks.set_open_relex.model",
            "SetOpenRelexHead",
            "text",
        ),
        TaskDefinition("structuring", "glinext.tasks.structuring.model", "StructuringHead", "text"),
        TaskDefinition(
            "set_structuring",
            "glinext.tasks.set_structuring.model",
            "SetStructuringHead",
            "text",
        ),
        TaskDefinition("image_classification", "glinext.tasks.vision.model", "ImageClassificationHead", "vision"),
        TaskDefinition("object_detection", "glinext.tasks.vision.model", "ObjectDetectionHead", "vision"),
        TaskDefinition("segmentation", "glinext.tasks.vision.model", "SegmentationHead", "vision"),
        TaskDefinition("audio_classification", "glinext.tasks.audio.model", "AudioClassificationHead", "audio"),
        TaskDefinition("audio_segmentation", "glinext.tasks.audio.model", "AudioSegmentationHead", "audio"),
        TaskDefinition("embedding", "glinext.tasks.embedding.model", "EmbeddingHead", "text"),
    ]
)
