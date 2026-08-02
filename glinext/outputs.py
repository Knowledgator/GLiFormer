"""Model output containers grouped by modality."""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers.utils import ModelOutput


@dataclass
class GLiNExTBaseOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    batch_size: Optional[int] = None


@dataclass
class GLiNExTTextOutput(GLiNExTBaseOutput):
    ner_logits: Optional[torch.FloatTensor] = None
    ner_batch_origin: Optional[torch.LongTensor] = None
    span_logits: Optional[torch.FloatTensor] = None
    span_idx: Optional[torch.LongTensor] = None
    span_mask: Optional[torch.Tensor] = None

    cat_logits: Optional[torch.FloatTensor] = None
    cat_batch_origin: Optional[torch.LongTensor] = None

    joint_rel_logits: Optional[torch.FloatTensor] = None
    joint_rel_batch_origin: Optional[torch.LongTensor] = None
    joint_rel_idx: Optional[torch.LongTensor] = None
    joint_rel_mask: Optional[torch.Tensor] = None
    joint_rel_entity_spans: Optional[torch.LongTensor] = None

    open_rel_logits: Optional[torch.FloatTensor] = None
    open_rel_batch_origin: Optional[torch.LongTensor] = None
    open_rel_anchor_mask: Optional[torch.Tensor] = None
    open_rel_span_logits: Optional[torch.FloatTensor] = None
    open_rel_span_idx: Optional[torch.LongTensor] = None
    open_rel_span_mask: Optional[torch.Tensor] = None

    count_logits: Optional[torch.FloatTensor] = None
    count_batch_origin: Optional[torch.LongTensor] = None

    groups_output: Optional[torch.FloatTensor] = None
    groups_mask: Optional[torch.Tensor] = None

    structuring_logits: Optional[torch.FloatTensor] = None
    structuring_batch_origin: Optional[torch.LongTensor] = None
    structuring_anchor_mask: Optional[torch.Tensor] = None
    structuring_objectness_logits: Optional[torch.FloatTensor] = None
    structuring_anchor_relation_scores: Optional[torch.FloatTensor] = None
    structuring_span_logits: Optional[torch.FloatTensor] = None
    structuring_span_idx: Optional[torch.LongTensor] = None
    structuring_span_mask: Optional[torch.Tensor] = None

    # Composite set structuring: classical NER token logits, NER-derived field
    # logits for selected spans, and second-stage anchor-membership logits are
    # deliberately separate.
    set_structuring_entity_logits: Optional[torch.FloatTensor] = None
    set_structuring_field_logits: Optional[torch.FloatTensor] = None
    set_structuring_logits: Optional[torch.FloatTensor] = None
    # Trainer-friendly transpose of membership logits: variable entity count
    # occupies dimension 1, which Hugging Face evaluation can pad.
    set_structuring_assignment_logits: Optional[torch.FloatTensor] = None
    set_structuring_batch_origin: Optional[torch.LongTensor] = None
    set_structuring_anchor_mask: Optional[torch.Tensor] = None
    set_structuring_objectness_logits: Optional[torch.FloatTensor] = None
    set_structuring_anchor_relation_scores: Optional[torch.FloatTensor] = None
    set_structuring_span_idx: Optional[torch.LongTensor] = None
    set_structuring_span_mask: Optional[torch.Tensor] = None

    embedding_logits: Optional[torch.FloatTensor] = None

    words_embedding: Optional[torch.FloatTensor] = None
    mask: Optional[torch.LongTensor] = None
    prompts_embedding: Optional[torch.FloatTensor] = None
    prompts_embedding_mask: Optional[torch.LongTensor] = None


@dataclass
class GLiNExTLayoutOutput(GLiNExTTextOutput):
    """Text output produced by the document-layout variant."""


@dataclass
class GLiNExTVisionOutput(GLiNExTBaseOutput):
    image_classification_logits: Optional[torch.FloatTensor] = None
    image_classification_batch_origin: Optional[torch.LongTensor] = None

    object_detection_logits: Optional[torch.FloatTensor] = None
    object_detection_batch_origin: Optional[torch.LongTensor] = None
    object_detection_boxes: Optional[torch.FloatTensor] = None
    object_detection_objectness_logits: Optional[torch.FloatTensor] = None
    object_detection_anchor_mask: Optional[torch.Tensor] = None

    segmentation_logits: Optional[torch.FloatTensor] = None
    segmentation_batch_origin: Optional[torch.LongTensor] = None
    segmentation_boxes: Optional[torch.FloatTensor] = None
    segmentation_objectness_logits: Optional[torch.FloatTensor] = None
    segmentation_anchor_mask: Optional[torch.Tensor] = None
    segmentation_mask_logits: Optional[torch.FloatTensor] = None
    segmentation_mask_validity: Optional[torch.Tensor] = None
    segmentation_prototypes: Optional[torch.FloatTensor] = None
    segmentation_coefficients: Optional[torch.FloatTensor] = None

    vision_embedding: Optional[torch.FloatTensor] = None
    vision_mask: Optional[torch.LongTensor] = None
    words_embedding: Optional[torch.FloatTensor] = None
    mask: Optional[torch.LongTensor] = None


@dataclass
class GLiNExTAudioOutput(GLiNExTBaseOutput):
    audio_classification_logits: Optional[torch.FloatTensor] = None
    audio_classification_batch_origin: Optional[torch.LongTensor] = None

    audio_segmentation_logits: Optional[torch.FloatTensor] = None
    audio_segmentation_batch_origin: Optional[torch.LongTensor] = None
    audio_segmentation_segments: Optional[torch.FloatTensor] = None
    audio_segmentation_objectness_logits: Optional[torch.FloatTensor] = None
    audio_segmentation_anchor_mask: Optional[torch.Tensor] = None
    audio_segmentation_mask_logits: Optional[torch.FloatTensor] = None
    audio_segmentation_prototypes: Optional[torch.FloatTensor] = None
    audio_segmentation_coefficients: Optional[torch.FloatTensor] = None

    audio_embedding: Optional[torch.FloatTensor] = None
    audio_mask: Optional[torch.LongTensor] = None
    words_embedding: Optional[torch.FloatTensor] = None
    mask: Optional[torch.LongTensor] = None


@dataclass
class GLiNExTOmniOutput(GLiNExTAudioOutput, GLiNExTVisionOutput, GLiNExTTextOutput):
    """Union output for omni models.

    Multiple inheritance keeps the flat ``ModelOutput`` attribute API while
    making each modality's field ownership explicit.
    """


@dataclass
class GLiNExTOutput(GLiNExTOmniOutput):
    """Backward-compatible union output used by older callers."""
