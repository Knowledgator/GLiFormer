"""Task head abstractions for GLiNExT modular architecture."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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


@dataclass
class TaskFlatInputs:
    """Per-task flattened representations with BN indexing.

    BN = total number of subtask groups across the batch for a specific task.
    E.g., if batch item 0 has 3 NER schemas and item 1 has 1, BN=4 for NER.
    """
    words_embedding: torch.Tensor     # (BN, W, D) word embeddings repeated per group
    mask: torch.Tensor                # (BN, W) word mask repeated per group
    parent_embedding: torch.Tensor    # (BN, D) parent embedding per group
    child_embedding: torch.Tensor     # (BN, max_C, D) child embeddings per group
    child_mask: torch.Tensor          # (BN, max_C) child mask per group
    batch_origin: torch.Tensor        # (BN,) maps flat idx to original batch idx


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
        if anchor_mode == "lstm":
            anchor_mode = "rotary"

        self.anchor_layer = AnchorLayer.from_config(
            anchor_mode, hidden_size,
            max_count=getattr(task_cfg, "max_count", 20),
            num_slots=getattr(task_cfg, "num_fixed_slots", 10),
            num_heads=getattr(task_cfg, "anchor_num_heads", 4),
            num_layers=getattr(task_cfg, "anchor_num_layers", 2),
            dropout=dropout,
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


class TaskProcessor(ABC):
    """Abstract base class for task-specific data processors."""

    def __init__(self, config, tokenizer=None, words_splitter=None, **kwargs):
        self.config = config

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

    def prepare_label_encoder_inputs(self, classes_mapping, labels_tokenizer) -> Optional[Dict]:
        """Tokenize label strings for the labels encoder."""
        return None
