"""Task head abstractions for GLiNExT modular architecture."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers.utils import ModelOutput

from ..layers import AnchorLayer, AnchorModeling, AnchorCrossAttentionLayer


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


class TaskHead(ABC, nn.Module):
    """Abstract base class for task heads."""
    name: str = ""
    dependencies: List[str] = []
    loss_coef: float = 1.0

    def _init_anchor_pipeline(self, task_cfg, config, hidden_size, dropout, shared_layers):
        """Initialize the shared anchor pipeline: anchor layer, modeling, refinement, span rep.

        Sets: self.anchor_layer, self.anchor_modeling, self.represent_spans,
              self.span_loss_coef, and optionally self.anchor_refine, self.span_rep_layer.
        """
        anchor_mode = getattr(task_cfg, "anchor_mode", "parent")
        if anchor_mode == "rnn":
            anchor_mode = "rotary"

        self.anchor_layer = AnchorLayer.from_config(
            anchor_mode, hidden_size,
            max_count=getattr(task_cfg, "max_count", 20),
            num_slots=getattr(task_cfg, "num_fixed_slots", 10),
            num_heads=getattr(task_cfg, "anchor_num_heads", 4),
            num_layers=getattr(task_cfg, "anchor_num_layers", 2),
            dropout=dropout,
            feature_mlp=getattr(task_cfg, "feature_anchor_mlp", False),
            feature_mlp_hidden_multiplier=getattr(task_cfg, "feature_anchor_mlp_hidden_multiplier", 1),
        )

        if "anchor_modeling" in shared_layers:
            self.anchor_modeling = shared_layers["anchor_modeling"]
        else:
            self.anchor_modeling = AnchorModeling.from_config(
                getattr(task_cfg, "anchor_modeling", "linear"),
                hidden_size, dropout=dropout,
            )

        if "anchor_refine" in shared_layers:
            self.anchor_refine = shared_layers["anchor_refine"]
        else:
            refine_layers = getattr(task_cfg, "anchor_refine_layers", 0)
            if refine_layers > 0:
                refine_heads = getattr(task_cfg, "anchor_refine_heads", 8)
                self.anchor_refine = AnchorCrossAttentionLayer(
                    hidden_size, num_heads=refine_heads,
                    num_layers=refine_layers, dropout=dropout,
                )

        self.represent_spans = getattr(task_cfg, "represent_spans", False)
        self.span_loss_coef = getattr(task_cfg, "span_loss_coef", 1.0)
        if self.represent_spans:
            from gliner.modeling.span_rep import SpanRepLayer
            self.span_rep_layer = SpanRepLayer(
                span_mode="token_level",
                hidden_size=hidden_size,
                max_width=getattr(config, "max_width", 12),
                dropout=dropout,
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
        TaskDefinition("structuring", "glinext.tasks.structuring.model", "StructuringHead", "text"),
        TaskDefinition("image_classification", "glinext.tasks.vision.model", "ImageClassificationHead", "vision"),
        TaskDefinition("object_detection", "glinext.tasks.vision.model", "ObjectDetectionHead", "vision"),
        TaskDefinition("segmentation", "glinext.tasks.vision.model", "SegmentationHead", "vision"),
        TaskDefinition("audio_classification", "glinext.tasks.audio.model", "AudioClassificationHead", "audio"),
        TaskDefinition("audio_segmentation", "glinext.tasks.audio.model", "AudioSegmentationHead", "audio"),
        TaskDefinition("embedding", "glinext.tasks.embedding.model", "EmbeddingHead", "text"),
    ]
)
