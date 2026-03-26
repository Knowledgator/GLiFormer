"""Task head abstractions for GLiNExT modular architecture."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from transformers.utils import ModelOutput


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
class TaskHeadOutput(ModelOutput):
    """Output from a single task head."""
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class TaskHead(ABC, nn.Module):
    """Abstract base class for task heads."""
    name: str = ""
    dependencies: List[str] = []
    loss_coef: float = 1.0

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
