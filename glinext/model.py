"""GLiNExT: unified multi-task model — thin orchestrator over modular task heads."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
from torch import nn
from transformers.utils import ModelOutput

from gliner.modeling.base import BaseModel
from gliner.modeling.layers import CrossFuser, LstmSeq2SeqEncoder
from gliner.modeling.utils import (
    extract_word_embeddings,
    extract_prompt_features,
    extract_prompt_features_and_word_embeddings,
)

from .config import GLiNextConfig
from .tasks import SharedRepresentations, TaskFlatInputs, TaskHeadOutput
from .layers import AnchorModeling, AnchorCrossAttentionLayer
from .tasks.ner.model import NERHead
from .tasks.classification.model import ClassificationHead
from .tasks.joint_relex.model import JointRelexHead
from .tasks.open_relex.model import OpenRelexHead
from .tasks.count.model import CountHead
from .tasks.structuring.model import StructuringHead
from .tasks.embedding.model import EmbeddingHead
from .tasks.vision.model import ImageClassificationHead, ObjectDetectionHead, SegmentationHead
from .tasks.audio.model import AudioClassificationHead, AudioSegmentationHead
from .encoders.audio import AudioEncoder
from .encoders.omni import (
    OmniEncoderOutput,
    TextAudioOmniBiEncoder,
    TextAudioOmniEncoder,
    TextVisionOmniBiEncoder,
    TextVisionOmniEncoder,
    TriOmniBiEncoder,
    TriOmniEncoder,
)
from .encoders.text import TextBiEncoder, TextEncoder
from .encoders.vision import VisionEncoder

@dataclass
class GLiNExTOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    # Original batch size (B) for unflattening BN-indexed outputs
    batch_size: Optional[int] = None
    # NER
    ner_logits: Optional[torch.FloatTensor] = None
    ner_batch_origin: Optional[torch.LongTensor] = None
    span_logits: Optional[torch.FloatTensor] = None
    span_idx: Optional[torch.LongTensor] = None
    span_mask: Optional[torch.Tensor] = None
    # Classification
    cat_logits: Optional[torch.FloatTensor] = None
    cat_batch_origin: Optional[torch.LongTensor] = None
    # Joint Relex (NER + relation extraction)
    joint_rel_logits: Optional[torch.FloatTensor] = None
    joint_rel_batch_origin: Optional[torch.LongTensor] = None
    joint_rel_idx: Optional[torch.LongTensor] = None
    joint_rel_mask: Optional[torch.Tensor] = None
    joint_rel_entity_spans: Optional[torch.LongTensor] = None
    # Open Relex (anchor-based relation extraction)
    open_rel_logits: Optional[torch.FloatTensor] = None
    open_rel_batch_origin: Optional[torch.LongTensor] = None
    open_rel_anchor_mask: Optional[torch.Tensor] = None
    open_rel_span_logits: Optional[torch.FloatTensor] = None
    open_rel_span_idx: Optional[torch.LongTensor] = None
    open_rel_span_mask: Optional[torch.Tensor] = None
    # Count
    count_logits: Optional[torch.FloatTensor] = None
    count_batch_origin: Optional[torch.LongTensor] = None
    # Groups / Structuring
    groups_output: Optional[torch.FloatTensor] = None
    groups_mask: Optional[torch.Tensor] = None
    # Structuring (anchor-based span extraction)
    structuring_logits: Optional[torch.FloatTensor] = None
    structuring_batch_origin: Optional[torch.LongTensor] = None
    structuring_anchor_mask: Optional[torch.Tensor] = None
    structuring_objectness_logits: Optional[torch.FloatTensor] = None
    structuring_span_logits: Optional[torch.FloatTensor] = None
    structuring_span_idx: Optional[torch.LongTensor] = None
    structuring_span_mask: Optional[torch.Tensor] = None
    # Embedding similarity
    embedding_logits: Optional[torch.FloatTensor] = None
    # Vision tasks
    image_classification_logits: Optional[torch.FloatTensor] = None
    image_classification_batch_origin: Optional[torch.LongTensor] = None
    audio_classification_logits: Optional[torch.FloatTensor] = None
    audio_classification_batch_origin: Optional[torch.LongTensor] = None
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
    segmentation_prototypes: Optional[torch.FloatTensor] = None
    segmentation_coefficients: Optional[torch.FloatTensor] = None
    audio_segmentation_logits: Optional[torch.FloatTensor] = None
    audio_segmentation_batch_origin: Optional[torch.LongTensor] = None
    audio_segmentation_segments: Optional[torch.FloatTensor] = None
    audio_segmentation_objectness_logits: Optional[torch.FloatTensor] = None
    audio_segmentation_anchor_mask: Optional[torch.Tensor] = None
    audio_segmentation_mask_logits: Optional[torch.FloatTensor] = None
    audio_segmentation_prototypes: Optional[torch.FloatTensor] = None
    audio_segmentation_coefficients: Optional[torch.FloatTensor] = None
    # Embeddings (for downstream use)
    words_embedding: Optional[torch.FloatTensor] = None
    mask: Optional[torch.LongTensor] = None
    prompts_embedding: Optional[torch.FloatTensor] = None
    prompts_embedding_mask: Optional[torch.LongTensor] = None


# Fixed execution order respecting dependencies
_EXECUTION_ORDER = ["ner", "classification", "count", "joint_relex", "open_relex",
                    "structuring", "image_classification", "object_detection",
                    "segmentation", "audio_classification", "audio_segmentation",
                    "embedding"]

# Head classes in registration order
_HEAD_CLASSES = [NERHead, ClassificationHead, CountHead, JointRelexHead,
                 OpenRelexHead, StructuringHead, ImageClassificationHead,
                 ObjectDetectionHead, SegmentationHead, AudioClassificationHead,
                 AudioSegmentationHead, EmbeddingHead]


class CrossModalTokenFusion(nn.Module):
    """Fuse non-text modality tokens into text word embeddings without changing word length."""

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        text_embeddings: torch.Tensor,
        modality_embeddings: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if modality_embeddings is None or modality_embeddings.shape[1] == 0:
            return text_embeddings
        key_padding_mask = None
        if modality_mask is not None:
            key_padding_mask = ~modality_mask.bool()
        attended, _ = self.attention(
            text_embeddings,
            modality_embeddings,
            modality_embeddings,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.norm(text_embeddings + self.dropout(attended))


def _normalize_model_variant(value: Optional[str]) -> str:
    return (value or "text-only").replace("_", "-").lower()


def _normalize_fusion(value: Optional[str]) -> str:
    return (value or "uni-encoder").replace("_", "-").lower()


def _has_labels_encoder(module: nn.Module) -> bool:
    return hasattr(module, "encode_labels")


def _extract_sequence_embeddings(output) -> torch.Tensor:
    if isinstance(output, OmniEncoderOutput):
        if output.text_embeddings is not None:
            return output.text_embeddings
        return output.embeddings
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    return output


class BaseGLiNextModel(BaseModel):
    """Unified multi-task model composing optional task heads.

    Each task is a standalone TaskHead subclass. The model registers enabled heads
    via nn.ModuleDict and executes them in dependency order during forward().
    """

    def __init__(
        self,
        config: GLiNextConfig,
        from_pretrained: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        super().__init__(config, from_pretrained, cache_dir)

        # ── Shared encoder ──────────────────────────────────────────────
        self.token_rep_layer = self._init_token_rep_layer(config, from_pretrained, cache_dir)

        if config.num_rnn_layers > 0:
            self.rnn = LstmSeq2SeqEncoder(config, num_layers=config.num_rnn_layers)

        if config.post_fusion_schema:
            self.cross_fuser = CrossFuser(
                config.hidden_size,
                config.hidden_size,
                num_heads=self.token_rep_layer.bert_layer.model.config.num_attention_heads,
                num_layers=config.num_post_fusion_layers,
                dropout=config.dropout,
                schema=config.post_fusion_schema,
            )

        # ── Shared layers (optional) ───────────────────────────────────
        shared_layers = {}
        if config.shared_anchor_modeling is not None:
            self.shared_anchor_modeling = AnchorModeling.from_config(
                config.shared_anchor_modeling, config.hidden_size, dropout=config.dropout,
            )
            shared_layers["anchor_modeling"] = self.shared_anchor_modeling

        if config.shared_anchor_refine_layers > 0:
            self.shared_anchor_refine = AnchorCrossAttentionLayer(
                config.hidden_size,
                num_heads=config.shared_anchor_refine_heads,
                num_layers=config.shared_anchor_refine_layers,
                dropout=config.dropout,
            )
            shared_layers["anchor_refine"] = self.shared_anchor_refine

        # ── Register task heads ─────────────────────────────────────────
        self.heads = nn.ModuleDict()
        for HeadClass in _HEAD_CLASSES:
            head = HeadClass.from_config(
                config,
                from_pretrained=from_pretrained,
                cache_dir=cache_dir,
                shared_layers=shared_layers,
            )
            if head is not None:
                self.heads[head.name] = head

    def _init_token_rep_layer(self, config, from_pretrained, cache_dir):
        if config.labels_encoder is not None:
            return TextBiEncoder(config, from_pretrained, cache_dir=cache_dir)
        return TextEncoder(config, from_pretrained, cache_dir=cache_dir)

    def _encode_label_type(
        self,
        label_input_ids: Optional[torch.Tensor],
        label_attention_mask: Optional[torch.Tensor],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if label_input_ids is None or not _has_labels_encoder(self.token_rep_layer):
            return None
        labels_embeds = self.token_rep_layer.encode_labels(label_input_ids, label_attention_mask)
        return labels_embeds.unsqueeze(0).expand(batch_size, -1, -1)

    def _predict_structuring_counts_from_count_head(
        self,
        count_logits: Optional[torch.Tensor],
        flat_inputs_map: Dict[str, TaskFlatInputs],
    ) -> Optional[torch.Tensor]:
        """Project count-head outputs onto structuring groups during inference.

        The count head is built over a concatenated flat order of:
        classification groups, extraction groups, then structuring groups.
        Structuring counts therefore live in the final contiguous block.
        """
        if count_logits is None or "count" not in flat_inputs_map or "structuring" not in flat_inputs_map:
            return None

        struct_bn = flat_inputs_map["structuring"].batch_origin.shape[0]
        if struct_bn == 0:
            return None

        count_cfg = self.config.count_config
        if count_cfg and count_cfg.mode == "classification":
            predicted = count_logits.argmax(dim=-1)
            print(f"[DEBUG count_head] mode=classification logits.shape={tuple(count_logits.shape)} "
                  f"top3_argmax_per_row={count_logits.topk(min(3, count_logits.shape[-1]), dim=-1).indices.tolist()} "
                  f"top3_probs_per_row={count_logits.softmax(dim=-1).topk(min(3, count_logits.shape[-1]), dim=-1).values.tolist()}")
        else:
            predicted = count_logits.squeeze(-1).round().long()
            print(f"[DEBUG count_head] mode=regression raw={count_logits.squeeze(-1).tolist()}")

        predicted = predicted.clamp(min=0)
        print(f"[DEBUG count_head] all_predicted={predicted.tolist()} (struct_bn={struct_bn})")
        if predicted.shape[0] < struct_bn:
            return None

        struct_predicted = predicted[-struct_bn:]
        print(f"[DEBUG count_head] structuring_count={struct_predicted.tolist()}")
        return struct_predicted

    def _task_config_for_loss(self, task_name: str):
        if task_name == "ner":
            return self.config.ner_config
        if task_name == "classification":
            return self.config.classification_config
        if task_name == "image_classification":
            return self.config.image_classification_config
        if task_name == "audio_classification":
            return self.config.audio_classification_config
        if task_name == "object_detection":
            return self.config.object_detection_config
        if task_name == "segmentation":
            return self.config.segmentation_config
        if task_name == "audio_segmentation":
            return self.config.audio_segmentation_config
        if task_name == "joint_relex":
            return self.config.joint_relex_config
        if task_name == "open_relex":
            return self.config.open_relex_config
        if task_name == "structuring":
            return self.config.structuring_config
        return None

    @staticmethod
    def _runtime_loss_value(runtime_kwargs: dict, focal_name: str, short_name: str):
        if focal_name in runtime_kwargs and runtime_kwargs[focal_name] is not None:
            return runtime_kwargs[focal_name]
        if short_name in runtime_kwargs and runtime_kwargs[short_name] is not None:
            return runtime_kwargs[short_name]
        return None

    def _resolve_task_focal_loss_kwargs(self, task_name: str, runtime_kwargs: dict) -> dict:
        task_cfg = self._task_config_for_loss(task_name)
        resolved = {}
        for cfg_name, loss_name in (
            ("focal_loss_alpha", "alpha"),
            ("focal_loss_gamma", "gamma"),
            ("focal_loss_prob_margin", "prob_margin"),
        ):
            value = getattr(task_cfg, cfg_name, None) if task_cfg is not None else None
            if value is None:
                value = self._runtime_loss_value(runtime_kwargs, cfg_name, loss_name)
            if value is not None:
                resolved[loss_name] = value
        return resolved

    def _make_task_loss_fn(self, task_name: str, runtime_kwargs: dict):
        focal_kwargs = self._resolve_task_focal_loss_kwargs(task_name, runtime_kwargs)
        if not focal_kwargs:
            return self._loss

        def task_loss_fn(logits, labels, **call_kwargs):
            merged_kwargs = dict(focal_kwargs)
            merged_kwargs.update(
                {key: value for key, value in call_kwargs.items() if value is not None}
            )
            return self._loss(logits, labels, **merged_kwargs)

        return task_loss_fn

    def _encode_all_labels_batched(
        self,
        batch_size: int,
        cat_labels_input_ids: Optional[torch.Tensor] = None,
        cat_labels_attention_mask: Optional[torch.Tensor] = None,
        rel_labels_input_ids: Optional[torch.Tensor] = None,
        rel_labels_attention_mask: Optional[torch.Tensor] = None,
        child_labels_input_ids: Optional[torch.Tensor] = None,
        child_labels_attention_mask: Optional[torch.Tensor] = None,
        open_rel_labels_input_ids: Optional[torch.Tensor] = None,
        open_rel_labels_attention_mask: Optional[torch.Tensor] = None,
    ):
        """Batch all task label inputs into a single BiEncoder pass, then split results."""
        if not _has_labels_encoder(self.token_rep_layer):
            return None, None, None, None

        # Collect all non-None label inputs with their sizes
        parts = []
        sizes = []
        for ids, mask in [
            (cat_labels_input_ids, cat_labels_attention_mask),
            (rel_labels_input_ids, rel_labels_attention_mask),
            (child_labels_input_ids, child_labels_attention_mask),
            (open_rel_labels_input_ids, open_rel_labels_attention_mask),
        ]:
            if ids is not None:
                parts.append((ids, mask))
                sizes.append(ids.shape[0])
            else:
                parts.append(None)
                sizes.append(0)

        total = sum(sizes)
        if total == 0:
            return None, None, None, None

        # Concatenate all label inputs along batch dim
        all_ids = []
        all_masks = []
        for part in parts:
            if part is not None:
                all_ids.append(part[0])
                all_masks.append(part[1])

        if not all_ids:
            return None, None, None, None

        # Pad to same seq length and concatenate
        max_len = max(ids.shape[1] for ids in all_ids)
        padded_ids = []
        padded_masks = []
        for ids, m in zip(all_ids, all_masks):
            if ids.shape[1] < max_len:
                pad_size = max_len - ids.shape[1]
                ids = torch.nn.functional.pad(ids, (0, pad_size), value=0)
                m = torch.nn.functional.pad(m, (0, pad_size), value=0)
            padded_ids.append(ids)
            padded_masks.append(m)

        batched_ids = torch.cat(padded_ids, dim=0)
        batched_masks = torch.cat(padded_masks, dim=0)

        # Single encoder pass
        all_embeds = self.token_rep_layer.encode_labels(batched_ids, batched_masks)

        # Split back and expand to batch size
        results = []
        offset = 0
        for size in sizes:
            if size > 0:
                embeds = all_embeds[offset:offset + size]
                embeds = embeds.unsqueeze(0).expand(batch_size, -1, -1)
                results.append(embeds)
                offset += size
            else:
                results.append(None)

        return results[0], results[1], results[2], results[3]

    def _encode_vision_tokens(
        self,
        pixel_values: Optional[torch.Tensor],
        vision_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if pixel_values is None:
            return None, None
        encoder = getattr(self, "vision_encoder", None)
        if encoder is None:
            feature_encoders = getattr(self.token_rep_layer, "feature_encoders", None)
            if feature_encoders is not None and "vision" in feature_encoders:
                encoder = feature_encoders["vision"]
        if encoder is None:
            return None, None
        vision_kwargs = {
            key.removeprefix("vision_"): value
            for key, value in kwargs.items()
            if key.startswith("vision_")
        }
        vision_tokens = encoder(pixel_values, **vision_kwargs)
        if vision_attention_mask is not None and vision_attention_mask.shape[-1] == vision_tokens.shape[1]:
            vision_mask = vision_attention_mask.to(device=vision_tokens.device)
        else:
            vision_mask = torch.ones(vision_tokens.shape[:2], dtype=torch.long, device=vision_tokens.device)
        return vision_tokens, vision_mask

    def _encode_audio_tokens(
        self,
        input_values: Optional[torch.Tensor],
        audio_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if input_values is None:
            return None, None
        encoder = getattr(self, "audio_encoder", None)
        if encoder is None:
            feature_encoders = getattr(self.token_rep_layer, "feature_encoders", None)
            if feature_encoders is not None and "audio" in feature_encoders:
                encoder = feature_encoders["audio"]
        if encoder is None:
            return None, None
        audio_kwargs = {
            key.removeprefix("audio_"): value
            for key, value in kwargs.items()
            if key.startswith("audio_")
        }
        audio_tokens = encoder(input_values, attention_mask=audio_attention_mask, **audio_kwargs)
        if audio_attention_mask is not None and audio_attention_mask.shape[-1] == audio_tokens.shape[1]:
            audio_mask = audio_attention_mask.to(device=audio_tokens.device)
        else:
            audio_mask = torch.ones(audio_tokens.shape[:2], dtype=torch.long, device=audio_tokens.device)
        return audio_tokens, audio_mask

    # ── Prompt order for parent offset computation ────────────────
    # prepare_inputs adds groups in this order: text tasks, then vision tasks.
    _PROMPT_TASK_ORDER = [
        "classification", "ner", "open_relex", "structuring",
        "image_classification", "object_detection", "segmentation",
        "audio_classification", "audio_segmentation",
    ]

    def _parent_offset_for_item(self, classes_mapping, task_name, batch_idx):
        """Compute the parent token offset for `task_name` within batch item `batch_idx`.

        Parents appear in the same prompt order used by the processor.
        Returns the index of the first parent for this task within the item's parent list.
        """
        offset = 0
        task_group_counts = {
            "classification": lambda i: len(classes_mapping.cat_mapping[i].cat_class_to_id),
            "ner": lambda i: len(classes_mapping.extraction_mapping[i].items),
            "joint_relex": lambda i: len(classes_mapping.extraction_mapping[i].items),
            "open_relex": lambda i: len(classes_mapping.open_relex_mapping[i].items)
                if hasattr(classes_mapping, 'open_relex_mapping') and i < len(classes_mapping.open_relex_mapping) else 0,
            "structuring": lambda i: len(classes_mapping.structuring_mapping[i].items)
                if hasattr(classes_mapping, 'structuring_mapping') and i < len(classes_mapping.structuring_mapping) else 0,
            "image_classification": lambda i: len(classes_mapping.image_classification_mapping[i].items)
                if hasattr(classes_mapping, 'image_classification_mapping') and i < len(classes_mapping.image_classification_mapping) else 0,
            "audio_classification": lambda i: len(classes_mapping.audio_classification_mapping[i].items)
                if hasattr(classes_mapping, 'audio_classification_mapping') and i < len(classes_mapping.audio_classification_mapping) else 0,
            "object_detection": lambda i: len(classes_mapping.object_detection_mapping[i].items)
                if hasattr(classes_mapping, 'object_detection_mapping') and i < len(classes_mapping.object_detection_mapping) else 0,
            "segmentation": lambda i: len(classes_mapping.segmentation_mapping[i].items)
                if hasattr(classes_mapping, 'segmentation_mapping') and i < len(classes_mapping.segmentation_mapping) else 0,
            "audio_segmentation": lambda i: len(classes_mapping.audio_segmentation_mapping[i].items)
                if hasattr(classes_mapping, 'audio_segmentation_mapping') and i < len(classes_mapping.audio_segmentation_mapping) else 0,
        }
        # Map joint_relex to same prompt position as ner
        effective_task = "ner" if task_name == "joint_relex" else task_name
        for t in self._PROMPT_TASK_ORDER:
            if t == effective_task:
                break
            offset += task_group_counts.get(t, lambda i: 0)(batch_idx)
        return offset

    def _child_sizes_for_task(self, classes_mapping, task_name, batch_idx, group_idx):
        """Number of child tokens for a specific group."""
        if task_name in ("ner", "joint_relex"):
            return len(classes_mapping.extraction_mapping[batch_idx].items[group_idx].ner_class_to_id.class_to_id)
        elif task_name == "classification":
            return len(classes_mapping.cat_mapping[batch_idx].cat_class_to_id[group_idx].class_to_id)
        elif task_name == "structuring":
            return len(classes_mapping.structuring_mapping[batch_idx].items[group_idx].field_class_to_id.class_to_id)
        elif task_name == "open_relex":
            return len(classes_mapping.open_relex_mapping[batch_idx].items[group_idx].rel_class_to_id.class_to_id)
        elif task_name == "image_classification":
            return len(classes_mapping.image_classification_mapping[batch_idx].items[group_idx].class_to_id.class_to_id)
        elif task_name == "audio_classification":
            return len(classes_mapping.audio_classification_mapping[batch_idx].items[group_idx].class_to_id.class_to_id)
        elif task_name == "object_detection":
            return len(classes_mapping.object_detection_mapping[batch_idx].items[group_idx].class_to_id.class_to_id)
        elif task_name == "segmentation":
            return len(classes_mapping.segmentation_mapping[batch_idx].items[group_idx].class_to_id.class_to_id)
        elif task_name == "audio_segmentation":
            return len(classes_mapping.audio_segmentation_mapping[batch_idx].items[group_idx].class_to_id.class_to_id)
        return 0

    def _get_flat_iter(self, classes_mapping, task_name):
        """Return the appropriate flat iterator for a task."""
        iters = {
            "ner": classes_mapping.flat_extraction_iter,
            "joint_relex": classes_mapping.flat_extraction_iter,
            "classification": classes_mapping.flat_cat_iter,
            "structuring": classes_mapping.flat_structuring_iter,
            "open_relex": classes_mapping.flat_open_relex_iter,
            "image_classification": classes_mapping.flat_image_classification_iter,
            "audio_classification": classes_mapping.flat_audio_classification_iter,
            "object_detection": classes_mapping.flat_object_detection_iter,
            "segmentation": classes_mapping.flat_segmentation_iter,
            "audio_segmentation": classes_mapping.flat_audio_segmentation_iter,
        }
        return iters[task_name]

    def _build_flat_rel_prompts(
        self,
        rel_prompts_batch: torch.Tensor,
        rel_prompts_batch_mask: torch.Tensor,
        classes_mapping,
        embed_dim: int,
        device,
        dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Split batch-level [REL] prompts into per-extraction-group tensors.

        The prompt for each batch item concatenates [REL] tokens across its
        extraction groups in group order, so we slice by per-group rel-class
        counts to recover (BN, max_C_rel, D) aligned with the joint_relex
        flat_inputs batch axis.
        """
        slices: List[Tuple[int, int, int]] = []
        item_rel_offset: dict = {}
        for _, batch_idx, group_idx, ext_mapping in classes_mapping.flat_extraction_iter():
            rel_map = ext_mapping.rel_class_to_id
            n_rel = len(rel_map.class_to_id) if rel_map is not None else 0
            start = item_rel_offset.get(batch_idx, 0)
            slices.append((batch_idx, start, start + n_rel))
            item_rel_offset[batch_idx] = start + n_rel

        BN = len(slices)
        if BN == 0:
            return None, None

        max_C = max((ce - cs for _, cs, ce in slices), default=0)
        flat = torch.zeros(BN, max_C, embed_dim, device=device, dtype=dtype)
        flat_mask = torch.zeros(BN, max_C, device=device, dtype=rel_prompts_batch_mask.dtype)
        for idx, (bi, cs, ce) in enumerate(slices):
            n = ce - cs
            if n > 0 and ce <= rel_prompts_batch.shape[1]:
                flat[idx, :n] = rel_prompts_batch[bi, cs:ce]
                flat_mask[idx, :n] = rel_prompts_batch_mask[bi, cs:ce]
        return flat, flat_mask

    def _build_flat_rel_prompts_from_label_embeds(
        self,
        rel_label_embeds: torch.Tensor,
        label_group_sizes: Optional[torch.Tensor],
        classes_mapping,
        embed_dim: int,
        device,
        dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Split labels-encoder rel embeds into per-group tensors via label_group_sizes."""
        if label_group_sizes is None:
            return None, None
        BN = int(label_group_sizes.shape[0])
        if BN == 0:
            return None, None

        batch_origins: List[int] = []
        for _, batch_idx, _, _ in classes_mapping.flat_extraction_iter():
            batch_origins.append(batch_idx)

        cumsum = torch.cumsum(label_group_sizes, 0)
        max_C = int(label_group_sizes.max().item()) if BN > 0 else 0
        flat = torch.zeros(BN, max_C, embed_dim, device=device, dtype=dtype)
        flat_mask = torch.zeros(BN, max_C, device=device, dtype=torch.long)
        for g in range(BN):
            c_start = 0 if g == 0 else int(cumsum[g - 1].item())
            c_end = int(cumsum[g].item())
            n = c_end - c_start
            if n > 0 and c_end <= rel_label_embeds.shape[1]:
                flat[g, :n] = rel_label_embeds[batch_origins[g], c_start:c_end]
                flat_mask[g, :n] = 1
        return flat, flat_mask

    def _build_flat_inputs(
        self,
        parent_embeds: torch.Tensor,
        parent_mask: torch.Tensor,
        child_embeds: torch.Tensor,
        child_mask: Optional[torch.Tensor],
        words_embedding: torch.Tensor,
        word_mask: torch.Tensor,
        classes_mapping,
        task_name: str,
        label_group_sizes: Optional[torch.Tensor] = None,
        per_task_parents: bool = False,
    ) -> Optional[TaskFlatInputs]:
        """Build TaskFlatInputs for a specific task by flattening B-indexed tensors to BN.

        Args:
            parent_embeds: (B, max_P, D) parent token embeddings
            parent_mask: (B, max_P) parent mask
            child_embeds: (B, max_C, D) task-specific child embeddings
            child_mask: (B, max_C) or None
            words_embedding: (B, W, D)
            word_mask: (B, W)
            classes_mapping: BatchClassesMapping
            task_name: "ner", "classification", "structuring", "open_relex", "joint_relex"
            label_group_sizes: (BN,) for labels encoder path — number of children per flat group
            per_task_parents: when True, parent_embeds contains only this task's parents
                (no offset needed); when False, uses shared parent tensor with offset computation
        """
        device = words_embedding.device
        D = words_embedding.shape[-1]

        flat_iter = self._get_flat_iter(classes_mapping, task_name)

        # Collect group descriptors
        batch_origins: List[int] = []
        parent_positions: List[Tuple[int, int]] = []  # (batch_idx, parent_pos)
        child_slices: List[Tuple[int, int, int]] = []  # (batch_idx, start, end)

        # Track per-item child offset for prompt-based splitting
        item_child_offset: dict = {}

        for flat_idx, batch_idx, group_idx, _ in flat_iter():
            batch_origins.append(batch_idx)

            # Parent position within the parent tensor
            if per_task_parents:
                # Per-task parents: positions are task-local (no offset)
                parent_positions.append((batch_idx, group_idx))
            else:
                # Shared parents: offset by preceding tasks in prompt order
                p_offset = self._parent_offset_for_item(classes_mapping, task_name, batch_idx)
                parent_positions.append((batch_idx, p_offset + group_idx))

            if label_group_sizes is None:
                # Prompt path: children within batch item, accumulated by group
                if batch_idx not in item_child_offset:
                    item_child_offset[batch_idx] = 0
                c_start = item_child_offset[batch_idx]
                c_size = self._child_sizes_for_task(classes_mapping, task_name, batch_idx, group_idx)
                child_slices.append((batch_idx, c_start, c_start + c_size))
                item_child_offset[batch_idx] = c_start + c_size

        BN = len(batch_origins)
        if BN == 0:
            return None

        batch_origin = torch.tensor(batch_origins, dtype=torch.long, device=device)

        # Gather words (BN, W, D) and mask (BN, W)
        flat_words = words_embedding[batch_origin]
        flat_word_mask = word_mask[batch_origin]

        # Gather parent embeddings (BN, D)
        flat_parent = torch.zeros(BN, D, device=device, dtype=parent_embeds.dtype)
        for idx, (bi, pp) in enumerate(parent_positions):
            if pp < parent_embeds.shape[1]:
                flat_parent[idx] = parent_embeds[bi, pp]

        # Gather and pad child embeddings (BN, max_C_per_group, D)
        if label_group_sizes is not None:
            # Labels encoder path: children indexed flat across all groups
            cumsum = torch.cumsum(label_group_sizes, 0)
            max_C = int(label_group_sizes.max().item()) if BN > 0 else 0
            flat_children = torch.zeros(BN, max_C, D, device=device, dtype=child_embeds.dtype)
            flat_child_mask = torch.zeros(BN, max_C, device=device, dtype=torch.float)
            for g in range(BN):
                c_start = 0 if g == 0 else int(cumsum[g - 1].item())
                c_end = int(cumsum[g].item())
                n = c_end - c_start
                if n > 0 and c_end <= child_embeds.shape[1]:
                    flat_children[g, :n] = child_embeds[batch_origins[g], c_start:c_end]
                    flat_child_mask[g, :n] = 1.0
        else:
            # Prompt path: children per batch item, split by group sizes
            max_C = max((ce - cs for _, cs, ce in child_slices), default=0)
            flat_children = torch.zeros(BN, max_C, D, device=device, dtype=child_embeds.dtype)
            flat_child_mask = torch.zeros(BN, max_C, device=device, dtype=torch.float)
            for idx, (bi, cs, ce) in enumerate(child_slices):
                n = ce - cs
                if n > 0 and ce <= child_embeds.shape[1]:
                    flat_children[idx, :n] = child_embeds[bi, cs:ce]
                    if child_mask is not None:
                        flat_child_mask[idx, :n] = child_mask[bi, cs:ce].float()
                    else:
                        flat_child_mask[idx, :n] = 1.0

        return TaskFlatInputs(
            words_embedding=flat_words,
            mask=flat_word_mask,
            parent_embedding=flat_parent,
            child_embedding=flat_children,
            child_mask=flat_child_mask,
            batch_origin=batch_origin,
        )

    def get_representations(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text_lengths: torch.Tensor,
        words_mask: torch.Tensor,
        labels_input_ids: Optional[torch.Tensor] = None,
        labels_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run encoder and extract prompt + word embeddings."""
        encoder_kwargs = {k: kwargs[k] for k in ("packing_config", "pair_attention_mask") if k in kwargs}

        if _has_labels_encoder(self.token_rep_layer) and labels_input_ids is not None:
            token_embeds, labels_embeds = self.token_rep_layer(
                input_ids, attention_mask, labels_input_ids, labels_attention_mask, **encoder_kwargs,
            )
            batch_size, _, embed_dim = token_embeds.shape
            max_text_length = text_lengths.max()
            words_embedding, mask = extract_word_embeddings(
                token_embeds, words_mask, attention_mask,
                batch_size, max_text_length, embed_dim, text_lengths,
            )
            labels_embeds = labels_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            labels_mask = torch.ones(labels_embeds.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device)
            if hasattr(self, "cross_fuser"):
                labels_embeds, words_embedding = self.cross_fuser(labels_embeds, words_embedding, labels_mask, mask)
            if hasattr(self, "rnn"):
                words_embedding = self.rnn(words_embedding, mask)
            return token_embeds, labels_embeds, labels_mask, words_embedding, mask
        else:
            token_embeds = self.token_rep_layer(input_ids, attention_mask, **encoder_kwargs)
            prompts_embedding, prompts_embedding_mask, words_embedding, mask = (
                extract_prompt_features_and_word_embeddings(
                    self.config.class_token_index, token_embeds, input_ids, attention_mask,
                    text_lengths, words_mask, self.config.embed_ent_token,
                )
            )
            if hasattr(self, "rnn"):
                words_embedding = self.rnn(words_embedding, mask)
            return token_embeds, prompts_embedding, prompts_embedding_mask, words_embedding, mask

    def encode_embedding_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return _extract_sequence_embeddings(
            self.token_rep_layer(input_ids, attention_mask, **kwargs)
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        words_mask: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
        # NER
        ner_labels: Optional[torch.Tensor] = None,
        span_idx: Optional[torch.Tensor] = None,
        span_mask: Optional[torch.Tensor] = None,
        span_labels: Optional[torch.Tensor] = None,
        # Classification
        cat_labels: Optional[torch.Tensor] = None,
        # Vision tasks
        image_classification_labels: Optional[torch.Tensor] = None,
        audio_classification_labels: Optional[torch.Tensor] = None,
        object_detection_class_labels: Optional[torch.Tensor] = None,
        object_detection_bbox_labels: Optional[torch.Tensor] = None,
        object_detection_object_mask: Optional[torch.Tensor] = None,
        segmentation_class_labels: Optional[torch.Tensor] = None,
        segmentation_bbox_labels: Optional[torch.Tensor] = None,
        segmentation_object_mask: Optional[torch.Tensor] = None,
        segmentation_mask_labels: Optional[torch.Tensor] = None,
        audio_segmentation_class_labels: Optional[torch.Tensor] = None,
        audio_segmentation_segment_labels: Optional[torch.Tensor] = None,
        audio_segmentation_object_mask: Optional[torch.Tensor] = None,
        audio_segmentation_mask_labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        vision_attention_mask: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        # Joint Relex
        rel_labels: Optional[torch.Tensor] = None,
        rel_pair_mask: Optional[torch.Tensor] = None,
        rel_span_idx: Optional[torch.Tensor] = None,
        rel_span_mask: Optional[torch.Tensor] = None,
        # Open Relex
        open_rel_labels: Optional[torch.Tensor] = None,
        open_rel_count: Optional[torch.Tensor] = None,
        # Count
        count_targets: Optional[torch.Tensor] = None,
        # Groups / Structuring
        count_val: Optional[torch.Tensor] = None,
        structuring_labels: Optional[torch.Tensor] = None,
        structuring_count: Optional[torch.Tensor] = None,
        # Embedding similarity
        embedding_labels: Optional[torch.Tensor] = None,
        embedding_pair_idx: Optional[torch.Tensor] = None,
        embedding_input_ids: Optional[torch.Tensor] = None,
        embedding_attention_mask: Optional[torch.Tensor] = None,
        # Labels encoder (bi-encoder) — NER labels
        labels_input_ids: Optional[torch.Tensor] = None,
        labels_attention_mask: Optional[torch.Tensor] = None,
        # Labels encoder — classification labels
        cat_labels_input_ids: Optional[torch.Tensor] = None,
        cat_labels_attention_mask: Optional[torch.Tensor] = None,
        # Labels encoder — relation labels
        rel_labels_input_ids: Optional[torch.Tensor] = None,
        rel_labels_attention_mask: Optional[torch.Tensor] = None,
        # Labels encoder — structuring/child labels
        child_labels_input_ids: Optional[torch.Tensor] = None,
        child_labels_attention_mask: Optional[torch.Tensor] = None,
        # Labels encoder — open relex labels
        open_rel_labels_input_ids: Optional[torch.Tensor] = None,
        open_rel_labels_attention_mask: Optional[torch.Tensor] = None,
        # Misc
        threshold: float = 0.5,
        adjacency_threshold: float = 0.5,
        # Debug / manual override — when set, ignores the count-head prediction
        # and uses this constant value as the structuring anchor count.
        manual_structuring_count: Optional[int] = None,
        **kwargs,
    ) -> GLiNExTOutput:

        classes_mapping = kwargs.get("classes_mapping")
        if pixel_values is None:
            pixel_values = kwargs.get("pixel_values")
        if vision_attention_mask is None:
            vision_attention_mask = kwargs.get("vision_attention_mask")
        if input_values is None:
            input_values = kwargs.get("input_values")
        if audio_attention_mask is None:
            audio_attention_mask = kwargs.get("audio_attention_mask")

        # ── 1. Encode ────────────────────────────────────────────────────
        token_embeds, prompts_embedding, prompts_embedding_mask, words_embedding, mask = (
            self.get_representations(
                input_ids, attention_mask, text_lengths, words_mask,
                labels_input_ids=labels_input_ids,
                labels_attention_mask=labels_attention_mask,
                pixel_values=pixel_values,
                vision_attention_mask=vision_attention_mask,
                **{
                    k: kwargs[k]
                    for k in kwargs
                    if (
                        k in (
                            "packing_config", "pair_attention_mask",
                            "pixel_values", "vision_attention_mask",
                            "input_values", "audio_attention_mask",
                            "word_bboxes", "bbox",
                        )
                        or k.startswith(("vision_", "audio_", "text_"))
                    )
                },
            )
        )

        vision_embedding, vision_mask = self._encode_vision_tokens(
            pixel_values,
            vision_attention_mask=vision_attention_mask,
            **kwargs,
        )
        audio_embedding, audio_mask = self._encode_audio_tokens(
            input_values,
            audio_attention_mask=audio_attention_mask,
            **kwargs,
        )

        shared = SharedRepresentations(
            token_embeds=token_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            words_embedding=words_embedding,
            mask=mask,
            prompts_embedding=prompts_embedding,
            prompts_embedding_mask=prompts_embedding_mask,
            vision_embedding=vision_embedding,
            vision_mask=vision_mask,
            audio_embedding=audio_embedding,
            audio_mask=audio_mask,
            image_sizes=image_sizes,
        )

        batch_size = words_embedding.shape[0]
        embed_dim = words_embedding.shape[-1]
        total_loss = torch.tensor(0.0, device=words_embedding.device)

        # ── 1b. Encode task-specific labels via labels encoder ──────────
        cat_label_embeds, rel_label_embeds, child_label_embeds, open_rel_label_embeds = (
            self._encode_all_labels_batched(
                batch_size,
                cat_labels_input_ids, cat_labels_attention_mask,
                rel_labels_input_ids, rel_labels_attention_mask,
                child_labels_input_ids, child_labels_attention_mask,
                open_rel_labels_input_ids, open_rel_labels_attention_mask,
            )
        )

        # ── 1c. Extract parent embeddings ───────────────────────────────
        parent_embeds = None
        parent_mask_t = None
        per_task_parent_embeds: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        if classes_mapping is not None:
            if self.config.uses_per_task_parents:
                # Per-task parent tokens: extract each task's parents independently
                _task_parent_cfgs = {
                    "ner": self.config.ner_config,
                    "classification": self.config.classification_config,
                    "open_relex": self.config.open_relex_config,
                    "structuring": self.config.structuring_config,
                    "image_classification": self.config.image_classification_config,
                    "object_detection": self.config.object_detection_config,
                    "segmentation": self.config.segmentation_config,
                    "audio_classification": self.config.audio_classification_config,
                    "audio_segmentation": self.config.audio_segmentation_config,
                }
                for t_name, t_cfg in _task_parent_cfgs.items():
                    if t_cfg is not None and getattr(t_cfg, 'parent_token_index', -1) > 0:
                        p_e, p_m = extract_prompt_features(
                            t_cfg.parent_token_index, token_embeds, input_ids, attention_mask,
                            batch_size, embed_dim, getattr(t_cfg, 'embed_parent_token', True),
                        )
                        per_task_parent_embeds[t_name] = (p_e, p_m)
                # Joint relex shares NER's parent token
                if "ner" in per_task_parent_embeds:
                    per_task_parent_embeds["joint_relex"] = per_task_parent_embeds["ner"]
            elif self.config.parent_token_index > 0:
                # Legacy: single shared parent token for all tasks
                parent_embeds, parent_mask_t = extract_prompt_features(
                    self.config.parent_token_index, token_embeds, input_ids, attention_mask,
                    batch_size, embed_dim, self.config.embed_parent_token,
                )

        # ── 1d. Extract task-specific child embeddings (prompt path) ────
        # NER children = prompts_embedding (already extracted via class_token_index)
        ner_child_embeds = prompts_embedding
        ner_child_mask = prompts_embedding_mask

        # Classification children (if active and not using labels encoder)
        cat_child_embeds, cat_child_mask = None, None
        if "classification" in self.heads and cat_label_embeds is None:
            cat_cfg = self.config.classification_config
            cat_child_embeds, cat_child_mask = extract_prompt_features(
                cat_cfg.cat_token_index, token_embeds, input_ids, attention_mask,
                batch_size, embed_dim, cat_cfg.embed_cat_token,
            )
        elif cat_label_embeds is not None:
            cat_child_embeds = cat_label_embeds
            cat_child_mask = torch.ones(
                cat_label_embeds.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device,
            )

        # Joint relex: per-group rel prompts (split per extraction group)
        joint_rel_flat_prompts, joint_rel_flat_mask = None, None
        if "joint_relex" in self.heads and rel_label_embeds is None and classes_mapping is not None:
            jr_cfg = self.config.joint_relex_config
            rel_prompts_batch, rel_prompts_batch_mask = extract_prompt_features(
                jr_cfg.rel_token_index, token_embeds, input_ids, attention_mask,
                batch_size, embed_dim, jr_cfg.embed_rel_token,
            )
            joint_rel_flat_prompts, joint_rel_flat_mask = self._build_flat_rel_prompts(
                rel_prompts_batch, rel_prompts_batch_mask, classes_mapping, embed_dim,
                device=words_embedding.device, dtype=rel_prompts_batch.dtype,
            )
        elif "joint_relex" in self.heads and rel_label_embeds is not None and classes_mapping is not None:
            joint_rel_flat_prompts, joint_rel_flat_mask = self._build_flat_rel_prompts_from_label_embeds(
                rel_label_embeds, kwargs.get("rel_labels_group_size"), classes_mapping,
                embed_dim, device=words_embedding.device, dtype=rel_label_embeds.dtype,
            )

        # Open relex children
        open_rel_child_embeds, open_rel_child_mask = None, None
        if "open_relex" in self.heads and open_rel_label_embeds is None:
            or_cfg = self.config.open_relex_config
            open_rel_child_embeds, open_rel_child_mask = extract_prompt_features(
                or_cfg.rel_token_index, token_embeds, input_ids, attention_mask,
                batch_size, embed_dim, or_cfg.embed_rel_token,
            )
        elif open_rel_label_embeds is not None:
            open_rel_child_embeds = open_rel_label_embeds
            open_rel_child_mask = torch.ones(
                open_rel_label_embeds.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device,
            )

        # Structuring children
        struct_child_embeds, struct_child_mask = None, None
        if "structuring" in self.heads and child_label_embeds is None:
            s_cfg = self.config.structuring_config
            struct_child_embeds, struct_child_mask = extract_prompt_features(
                s_cfg.child_token_index, token_embeds, input_ids, attention_mask,
                batch_size, embed_dim, s_cfg.embed_child_token,
            )
        elif child_label_embeds is not None:
            struct_child_embeds = child_label_embeds
            struct_child_mask = torch.ones(
                child_label_embeds.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device,
            )

        # Vision/audio task children ([OBJ] label prompts)
        obj_child_embeds, obj_child_mask = None, None
        media_tasks = (
            "image_classification", "object_detection", "segmentation",
            "audio_classification", "audio_segmentation",
        )
        if any(name in self.heads for name in media_tasks):
            obj_child_embeds, obj_child_mask = extract_prompt_features(
                self.config.obj_token_index, token_embeds, input_ids, attention_mask,
                batch_size, embed_dim, self.config.embed_obj_token,
            )

        media_child_prompts: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {}
        if obj_child_embeds is not None and classes_mapping is not None:
            offsets = [0 for _ in range(batch_size)]
            for task_name in media_tasks:
                if task_name not in self.heads:
                    continue
                counts = []
                mapping_list = getattr(classes_mapping, f"{task_name}_mapping", [])
                for bi in range(batch_size):
                    if bi < len(mapping_list):
                        counts.append(sum(len(item.class_to_id.class_to_id) for item in mapping_list[bi].items))
                    else:
                        counts.append(0)
                max_count = max(counts, default=0)
                if max_count == 0:
                    media_child_prompts[task_name] = (None, None)
                    continue
                task_embeds = torch.zeros(
                    batch_size, max_count, embed_dim,
                    device=obj_child_embeds.device, dtype=obj_child_embeds.dtype,
                )
                task_mask = torch.zeros(
                    batch_size, max_count,
                    device=obj_child_embeds.device, dtype=obj_child_mask.dtype,
                )
                for bi, count in enumerate(counts):
                    start = offsets[bi]
                    end = start + count
                    if count > 0 and end <= obj_child_embeds.shape[1]:
                        task_embeds[bi, :count] = obj_child_embeds[bi, start:end]
                        task_mask[bi, :count] = obj_child_mask[bi, start:end]
                    offsets[bi] = end
                media_child_prompts[task_name] = (task_embeds, task_mask)

        # ── 1e. Build TaskFlatInputs per task ───────────────────────────
        flat_inputs_map = {}
        _use_per_task = bool(per_task_parent_embeds)
        _has_parents = parent_embeds is not None or _use_per_task

        if classes_mapping is not None and _has_parents:
            label_group_sizes_map = {
                "ner": kwargs.get("ner_labels_group_size"),
                "classification": kwargs.get("cat_labels_group_size"),
                "structuring": kwargs.get("child_labels_group_size"),
                "open_relex": kwargs.get("open_rel_labels_group_size"),
                "image_classification": kwargs.get("image_classification_labels_group_size"),
                "audio_classification": kwargs.get("audio_classification_labels_group_size"),
                "object_detection": kwargs.get("object_detection_labels_group_size"),
                "segmentation": kwargs.get("segmentation_labels_group_size"),
                "audio_segmentation": kwargs.get("audio_segmentation_labels_group_size"),
            }

            task_child_map = {
                "ner": (ner_child_embeds, ner_child_mask),
                "joint_relex": (ner_child_embeds, ner_child_mask),
                "classification": (cat_child_embeds, cat_child_mask),
                "structuring": (struct_child_embeds, struct_child_mask),
                "open_relex": (open_rel_child_embeds, open_rel_child_mask),
                "image_classification": media_child_prompts.get("image_classification", (obj_child_embeds, obj_child_mask)),
                "audio_classification": media_child_prompts.get("audio_classification", (obj_child_embeds, obj_child_mask)),
                "object_detection": media_child_prompts.get("object_detection", (obj_child_embeds, obj_child_mask)),
                "segmentation": media_child_prompts.get("segmentation", (obj_child_embeds, obj_child_mask)),
                "audio_segmentation": media_child_prompts.get("audio_segmentation", (obj_child_embeds, obj_child_mask)),
            }

            for task_name in (
                "ner", "joint_relex", "classification", "structuring", "open_relex",
                "image_classification", "object_detection", "segmentation",
                "audio_classification", "audio_segmentation",
            ):
                if task_name not in self.heads:
                    continue
                child_e, child_m = task_child_map.get(task_name, (None, None))
                if child_e is None:
                    continue

                # Select parent embeddings: per-task or shared
                if _use_per_task:
                    if task_name not in per_task_parent_embeds:
                        continue
                    t_parent_e, t_parent_m = per_task_parent_embeds[task_name]
                else:
                    t_parent_e, t_parent_m = parent_embeds, parent_mask_t

                lgs = label_group_sizes_map.get(
                    "ner" if task_name == "joint_relex" else task_name
                )
                fi = self._build_flat_inputs(
                    t_parent_e, t_parent_m,
                    child_e, child_m,
                    audio_embedding if task_name in ("audio_classification", "audio_segmentation") and audio_embedding is not None
                    else vision_embedding if task_name in ("image_classification", "object_detection", "segmentation") and vision_embedding is not None
                    else words_embedding,
                    audio_mask if task_name in ("audio_classification", "audio_segmentation") and audio_mask is not None
                    else vision_mask if task_name in ("image_classification", "object_detection", "segmentation") and vision_mask is not None
                    else mask,
                    classes_mapping, task_name,
                    label_group_sizes=lgs,
                    per_task_parents=_use_per_task,
                )
                if fi is not None:
                    flat_inputs_map[task_name] = fi

            # Count uses ALL parents (cat + ner + struct), so build a combined flat input
            if "count" in self.heads:
                count_batch_origins = []
                count_parent_embeds_list = []

                _count_tasks = [
                    ("classification", classes_mapping.flat_cat_iter),
                    ("ner", classes_mapping.flat_extraction_iter),
                    ("structuring", classes_mapping.flat_structuring_iter),
                ]

                for c_task, c_iter in _count_tasks:
                    if _use_per_task:
                        t_pe = per_task_parent_embeds.get(c_task)
                        if t_pe is None:
                            continue
                        t_parent_e, _ = t_pe
                        for _, bi, gi, _ in c_iter():
                            count_batch_origins.append(bi)
                            if gi < t_parent_e.shape[1]:
                                count_parent_embeds_list.append(t_parent_e[bi, gi])
                            else:
                                count_parent_embeds_list.append(
                                    torch.zeros(embed_dim, device=words_embedding.device, dtype=t_parent_e.dtype)
                                )
                    else:
                        for _, bi, gi, _ in c_iter():
                            count_batch_origins.append(bi)
                            p_off = self._parent_offset_for_item(classes_mapping, c_task, bi)
                            pp = p_off + gi
                            if pp < parent_embeds.shape[1]:
                                count_parent_embeds_list.append(parent_embeds[bi, pp])
                            else:
                                count_parent_embeds_list.append(
                                    torch.zeros(embed_dim, device=words_embedding.device, dtype=parent_embeds.dtype)
                                )

                BN_count = len(count_batch_origins)
                if BN_count > 0:
                    bo = torch.tensor(count_batch_origins, dtype=torch.long, device=words_embedding.device)
                    flat_parent = torch.stack(count_parent_embeds_list)
                    flat_inputs_map["count"] = TaskFlatInputs(
                        words_embedding=words_embedding[bo],
                        mask=mask[bo],
                        parent_embedding=flat_parent,
                        child_embedding=torch.zeros(BN_count, 0, embed_dim, device=words_embedding.device),
                        child_mask=torch.zeros(BN_count, 0, device=words_embedding.device),
                        batch_origin=bo,
                    )

        # ── 1e. Encode embedding pair texts (separate batch) ───────────
        embedding_encodings = None
        embedding_encoding_mask = None
        if embedding_input_ids is not None and embedding_pair_idx is not None:
            emb_token_embeds = self.encode_embedding_tokens(
                embedding_input_ids, embedding_attention_mask,
            )
            embedding_encodings = emb_token_embeds
            embedding_encoding_mask = embedding_attention_mask

        # Collect all batch kwargs for heads
        batch_kwargs = dict(
            ner_labels=ner_labels, span_idx=span_idx, span_mask=span_mask,
            span_labels=span_labels, cat_labels=cat_labels, rel_labels=rel_labels,
            image_classification_labels=image_classification_labels,
            audio_classification_labels=audio_classification_labels,
            object_detection_class_labels=object_detection_class_labels,
            object_detection_bbox_labels=object_detection_bbox_labels,
            object_detection_object_mask=object_detection_object_mask,
            segmentation_class_labels=segmentation_class_labels,
            segmentation_bbox_labels=segmentation_bbox_labels,
            segmentation_object_mask=segmentation_object_mask,
            segmentation_mask_labels=segmentation_mask_labels,
            audio_segmentation_class_labels=audio_segmentation_class_labels,
            audio_segmentation_segment_labels=audio_segmentation_segment_labels,
            audio_segmentation_object_mask=audio_segmentation_object_mask,
            audio_segmentation_mask_labels=audio_segmentation_mask_labels,
            rel_pair_mask=rel_pair_mask,
            rel_span_idx=rel_span_idx, rel_span_mask=rel_span_mask,
            open_rel_labels=open_rel_labels, open_rel_count=open_rel_count,
            count_targets=count_targets,
            count_val=count_val, structuring_labels=structuring_labels,
            structuring_count=structuring_count, embedding_labels=embedding_labels,
            embedding_pair_idx=embedding_pair_idx,
            embedding_encodings=embedding_encodings,
            embedding_encoding_mask=embedding_encoding_mask,
            threshold=threshold, adjacency_threshold=adjacency_threshold,
        )

        # ── 2. Execute heads in order ───────────────────────────────────
        head_outputs = {}
        for name in _EXECUTION_ORDER:
            if name not in self.heads:
                continue

            # Skip anchor-based heads that have no flat_inputs — they require
            # flattened parent/child embeddings to operate.
            if name in (
                "ner", "classification", "count", "joint_relex", "open_relex",
                "structuring", "image_classification", "object_detection", "segmentation",
                "audio_classification", "audio_segmentation",
            ):
                if name not in flat_inputs_map:
                    continue

            head = self.heads[name]

            dep_outputs = {d: head_outputs[d] for d in head.dependencies if d in head_outputs}

            extra_kwargs = {}

            # Pass flat_inputs if available
            if name in flat_inputs_map:
                extra_kwargs["flat_inputs"] = flat_inputs_map[name]

            if name == "structuring" and structuring_count is None:
                if manual_structuring_count is not None and "structuring" in flat_inputs_map:
                    struct_bn = flat_inputs_map["structuring"].batch_origin.shape[0]
                    forced = flat_inputs_map["structuring"].parent_embedding.new_full(
                        (struct_bn,), int(manual_structuring_count), dtype=torch.long,
                    )
                    extra_kwargs["structuring_count"] = forced
                    print(f"[DEBUG model.forward] manual_structuring_count={int(manual_structuring_count)} "
                          f"applied to {struct_bn} groups (overrides count head)")
                else:
                    predicted_structuring_count = self._predict_structuring_counts_from_count_head(
                        head_outputs.get("count", TaskHeadOutput()).logits,
                        flat_inputs_map,
                    )
                    if predicted_structuring_count is not None:
                        extra_kwargs["structuring_count"] = predicted_structuring_count

            # Pass task-specific label embeds for heads that use them directly
            if name == "joint_relex" and rel_label_embeds is not None:
                extra_kwargs["rel_label_embeds"] = rel_label_embeds
            if name == "joint_relex":
                extra_kwargs["flat_rel_prompts"] = joint_rel_flat_prompts
                extra_kwargs["flat_rel_prompts_mask"] = joint_rel_flat_mask

            # Pass loss function for heads that use focal loss.
            if name in (
                "ner", "classification", "joint_relex", "open_relex", "structuring",
                "image_classification", "object_detection", "segmentation",
                "audio_classification", "audio_segmentation",
            ):
                extra_kwargs["base_loss_fn"] = self._make_task_loss_fn(name, kwargs)

            call_kwargs = dict(batch_kwargs)
            call_kwargs.update(extra_kwargs)
            output = head(shared, dependency_outputs=dep_outputs, **call_kwargs)
            head_outputs[name] = output

            if output.loss is not None:
                total_loss = total_loss + head.loss_coef * output.loss

        # ── 3. Collect outputs ──────────────────────────────────────────
        has_any_loss = any(ho.loss is not None for ho in head_outputs.values())
        final_loss = total_loss if has_any_loss else None

        ner_out = head_outputs.get("ner", TaskHeadOutput())
        cat_out = head_outputs.get("classification", TaskHeadOutput())
        joint_rel_out = head_outputs.get("joint_relex", TaskHeadOutput())
        open_rel_out = head_outputs.get("open_relex", TaskHeadOutput())
        count_out = head_outputs.get("count", TaskHeadOutput())
        struct_out = head_outputs.get("structuring", TaskHeadOutput())
        image_cls_out = head_outputs.get("image_classification", TaskHeadOutput())
        audio_cls_out = head_outputs.get("audio_classification", TaskHeadOutput())
        det_out = head_outputs.get("object_detection", TaskHeadOutput())
        seg_out = head_outputs.get("segmentation", TaskHeadOutput())
        audio_seg_out = head_outputs.get("audio_segmentation", TaskHeadOutput())
        emb_out = head_outputs.get("embedding", TaskHeadOutput())

        # Prefer the standalone NER head for exported NER outputs.
        # Fall back to joint_relex only when NER is not active.
        effective_ner_logits = ner_out.logits if ner_out.logits is not None else joint_rel_out.logits
        effective_ner_extra = ner_out.extra if ner_out.logits is not None else joint_rel_out.extra
        effective_ner_origin = (
            flat_inputs_map["ner"].batch_origin if "ner" in flat_inputs_map
            else flat_inputs_map["joint_relex"].batch_origin if "joint_relex" in flat_inputs_map
            else flat_inputs_map.get("ner", TaskFlatInputs(
                words_embedding=words_embedding, mask=mask,
                parent_embedding=torch.empty(0), child_embedding=torch.empty(0),
                child_mask=torch.empty(0), batch_origin=torch.arange(batch_size, device=words_embedding.device),
            )).batch_origin
        )

        return GLiNExTOutput(
            loss=final_loss,
            batch_size=batch_size,
            ner_logits=effective_ner_logits,
            ner_batch_origin=effective_ner_origin,
            span_logits=effective_ner_extra.get("span_logits"),
            span_idx=effective_ner_extra.get("span_idx"),
            span_mask=effective_ner_extra.get("span_mask"),
            cat_logits=cat_out.logits,
            cat_batch_origin=flat_inputs_map["classification"].batch_origin if "classification" in flat_inputs_map else None,
            joint_rel_logits=joint_rel_out.extra.get("rel_logits"),
            joint_rel_batch_origin=flat_inputs_map["joint_relex"].batch_origin if "joint_relex" in flat_inputs_map else None,
            joint_rel_idx=joint_rel_out.extra.get("rel_idx"),
            joint_rel_mask=joint_rel_out.extra.get("rel_mask"),
            joint_rel_entity_spans=joint_rel_out.extra.get("rel_entity_spans"),
            open_rel_logits=open_rel_out.logits,
            open_rel_batch_origin=flat_inputs_map["open_relex"].batch_origin if "open_relex" in flat_inputs_map else None,
            open_rel_anchor_mask=open_rel_out.extra.get("anchor_mask"),
            open_rel_span_logits=open_rel_out.extra.get("span_logits"),
            open_rel_span_idx=open_rel_out.extra.get("span_idx"),
            open_rel_span_mask=open_rel_out.extra.get("span_mask"),
            count_logits=count_out.logits,
            count_batch_origin=flat_inputs_map["count"].batch_origin if "count" in flat_inputs_map else None,
            groups_output=struct_out.extra.get("groups_output"),
            groups_mask=struct_out.extra.get("anchor_mask"),
            structuring_logits=struct_out.logits,
            structuring_batch_origin=flat_inputs_map["structuring"].batch_origin if "structuring" in flat_inputs_map else None,
            structuring_anchor_mask=struct_out.extra.get("anchor_mask"),
            structuring_objectness_logits=struct_out.extra.get("objectness_logits"),
            structuring_span_logits=struct_out.extra.get("span_logits"),
            structuring_span_idx=struct_out.extra.get("span_idx"),
            structuring_span_mask=struct_out.extra.get("span_mask"),
            embedding_logits=emb_out.logits,
            image_classification_logits=image_cls_out.logits,
            image_classification_batch_origin=flat_inputs_map["image_classification"].batch_origin if "image_classification" in flat_inputs_map else None,
            audio_classification_logits=audio_cls_out.logits,
            audio_classification_batch_origin=flat_inputs_map["audio_classification"].batch_origin if "audio_classification" in flat_inputs_map else None,
            object_detection_logits=det_out.logits,
            object_detection_batch_origin=flat_inputs_map["object_detection"].batch_origin if "object_detection" in flat_inputs_map else None,
            object_detection_boxes=det_out.extra.get("bbox_preds"),
            object_detection_objectness_logits=det_out.extra.get("objectness_logits"),
            object_detection_anchor_mask=det_out.extra.get("anchor_mask"),
            segmentation_logits=seg_out.logits,
            segmentation_batch_origin=flat_inputs_map["segmentation"].batch_origin if "segmentation" in flat_inputs_map else None,
            segmentation_boxes=seg_out.extra.get("bbox_preds"),
            segmentation_objectness_logits=seg_out.extra.get("objectness_logits"),
            segmentation_anchor_mask=seg_out.extra.get("anchor_mask"),
            segmentation_mask_logits=seg_out.extra.get("mask_logits"),
            segmentation_prototypes=seg_out.extra.get("prototypes"),
            segmentation_coefficients=seg_out.extra.get("coefficients"),
            audio_segmentation_logits=audio_seg_out.logits,
            audio_segmentation_batch_origin=flat_inputs_map["audio_segmentation"].batch_origin if "audio_segmentation" in flat_inputs_map else None,
            audio_segmentation_segments=audio_seg_out.extra.get("segment_preds"),
            audio_segmentation_objectness_logits=audio_seg_out.extra.get("objectness_logits"),
            audio_segmentation_anchor_mask=audio_seg_out.extra.get("anchor_mask"),
            audio_segmentation_mask_logits=audio_seg_out.extra.get("mask_logits"),
            audio_segmentation_prototypes=audio_seg_out.extra.get("prototypes"),
            audio_segmentation_coefficients=audio_seg_out.extra.get("coefficients"),
            words_embedding=words_embedding,
            mask=mask,
            prompts_embedding=prompts_embedding,
            prompts_embedding_mask=prompts_embedding_mask,
        )

    def loss(self, *args, **kwargs):
        """Compute loss via forward pass."""
        output = self.forward(*args, **kwargs)
        return output.loss


class GLiNExTTextModel(BaseGLiNextModel):
    """Text-only GLiNExT model."""

    pass


class GLiNExTTextVisionBiEncoderModel(BaseGLiNextModel):
    """Text + vision model with separate encoders and cross-modal token fusion."""

    def __init__(self, config, from_pretrained=False, cache_dir=None):
        super().__init__(config, from_pretrained=from_pretrained, cache_dir=cache_dir)
        self.vision_encoder = VisionEncoder(
            config,
            from_pretrained=from_pretrained,
            cache_dir=cache_dir,
        )
        self.vision_fusion = CrossModalTokenFusion(
            config.hidden_size,
            num_heads=self._fusion_num_heads(),
            dropout=config.dropout,
        )

    def _fusion_num_heads(self) -> int:
        model_config = getattr(getattr(self.token_rep_layer, "bert_layer", None), "model", None)
        model_config = getattr(model_config, "config", None)
        return int(getattr(model_config, "num_attention_heads", 8))

    @staticmethod
    def _mask_for_tokens(tokens: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None and mask.shape[-1] == tokens.shape[1]:
            return mask.to(device=tokens.device)
        return torch.ones(tokens.shape[:2], dtype=torch.long, device=tokens.device)

    def get_representations(self, *args, **kwargs):
        pixel_values = kwargs.pop("pixel_values", None)
        vision_attention_mask = kwargs.pop("vision_attention_mask", None)
        output = super().get_representations(*args, **kwargs)
        if pixel_values is None:
            return output

        token_embeds, prompts_embedding, prompts_embedding_mask, words_embedding, mask = output
        vision_kwargs = {
            key.removeprefix("vision_"): value
            for key, value in kwargs.items()
            if key.startswith("vision_")
        }
        vision_tokens = self.vision_encoder(pixel_values, **vision_kwargs)
        vision_mask = self._mask_for_tokens(vision_tokens, vision_attention_mask)
        words_embedding = self.vision_fusion(words_embedding, vision_tokens, vision_mask)
        return token_embeds, prompts_embedding, prompts_embedding_mask, words_embedding, mask


class GLiNExTTextVisionUniEncoderModel(BaseGLiNextModel):
    """Text + vision model with early fusion inside the text transformer."""

    omni_encoder_cls = TextVisionOmniEncoder
    omni_bi_encoder_cls = TextVisionOmniBiEncoder

    def _init_token_rep_layer(self, config, from_pretrained, cache_dir):
        if config.labels_encoder is not None:
            return self.omni_bi_encoder_cls(
                config,
                from_pretrained=from_pretrained,
                cache_dir=cache_dir,
            )
        return self.omni_encoder_cls(
            config,
            from_pretrained=from_pretrained,
            cache_dir=cache_dir,
        )

    @staticmethod
    def _omni_kwargs(kwargs: dict) -> dict:
        allowed = {
            "packing_config", "pair_attention_mask", "pixel_values",
            "vision_attention_mask", "input_values", "audio_attention_mask",
            "bbox", "word_bboxes",
        }
        result = {k: kwargs[k] for k in allowed if k in kwargs}
        result.update({k: v for k, v in kwargs.items() if k.startswith(("vision_", "audio_", "text_"))})
        return result

    def get_representations(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text_lengths: torch.Tensor,
        words_mask: torch.Tensor,
        labels_input_ids: Optional[torch.Tensor] = None,
        labels_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        omni_kwargs = self._omni_kwargs(kwargs)
        if _has_labels_encoder(self.token_rep_layer) and labels_input_ids is not None:
            output = self.token_rep_layer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels_input_ids=labels_input_ids,
                labels_attention_mask=labels_attention_mask,
                return_dict=True,
                **omni_kwargs,
            )
            token_embeds = output.text_embeddings
            labels_embeds = output.labels_embeddings
            batch_size, _, embed_dim = token_embeds.shape
            max_text_length = text_lengths.max()
            words_embedding, mask = extract_word_embeddings(
                token_embeds, words_mask, attention_mask,
                batch_size, max_text_length, embed_dim, text_lengths,
            )
            labels_embeds = labels_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            labels_mask = torch.ones(labels_embeds.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device)
            if hasattr(self, "cross_fuser"):
                labels_embeds, words_embedding = self.cross_fuser(labels_embeds, words_embedding, labels_mask, mask)
            if hasattr(self, "rnn"):
                words_embedding = self.rnn(words_embedding, mask)
            return token_embeds, labels_embeds, labels_mask, words_embedding, mask

        output = self.token_rep_layer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **omni_kwargs,
        )
        token_embeds = output.text_embeddings
        prompts_embedding, prompts_embedding_mask, words_embedding, mask = (
            extract_prompt_features_and_word_embeddings(
                self.config.class_token_index, token_embeds, input_ids, attention_mask,
                text_lengths, words_mask, self.config.embed_ent_token,
            )
        )
        if hasattr(self, "rnn"):
            words_embedding = self.rnn(words_embedding, mask)
        return token_embeds, prompts_embedding, prompts_embedding_mask, words_embedding, mask

    def encode_embedding_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        output = self.token_rep_layer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **self._omni_kwargs(kwargs),
        )
        return output.text_embeddings


class GLiNExTTextVisionLayoutModel(GLiNExTTextVisionUniEncoderModel):
    """Text + vision + layout variant.

    Layout coordinates can be supplied as ``word_bboxes`` or ``bbox`` with shape
    ``(batch, words, 4)``. They are projected into the word embedding space.
    """

    def __init__(self, config, from_pretrained=False, cache_dir=None):
        super().__init__(config, from_pretrained=from_pretrained, cache_dir=cache_dir)
        self.layout_projection = nn.Sequential(
            nn.Linear(4, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
        )

    def get_representations(self, *args, **kwargs):
        word_bboxes = kwargs.pop("word_bboxes", None)
        if word_bboxes is None:
            word_bboxes = kwargs.pop("bbox", None)
        output = super().get_representations(*args, **kwargs)
        if word_bboxes is None:
            return output
        token_embeds, prompts_embedding, prompts_embedding_mask, words_embedding, mask = output
        if word_bboxes.shape[1] != words_embedding.shape[1]:
            word_bboxes = word_bboxes[:, :words_embedding.shape[1]]
            if word_bboxes.shape[1] < words_embedding.shape[1]:
                pad = words_embedding.shape[1] - word_bboxes.shape[1]
                word_bboxes = torch.nn.functional.pad(word_bboxes, (0, 0, 0, pad))
        layout_embedding = self.layout_projection(word_bboxes.to(device=words_embedding.device, dtype=words_embedding.dtype))
        words_embedding = words_embedding + layout_embedding
        return token_embeds, prompts_embedding, prompts_embedding_mask, words_embedding, mask


class GLiNExTTextAudioModel(GLiNExTTextVisionBiEncoderModel):
    """Text + audio model with separate encoders and cross-modal token fusion."""

    def __init__(self, config, from_pretrained=False, cache_dir=None):
        BaseGLiNextModel.__init__(self, config, from_pretrained=from_pretrained, cache_dir=cache_dir)
        self.audio_encoder = AudioEncoder(
            config,
            from_pretrained=from_pretrained,
            cache_dir=cache_dir,
        )
        self.audio_fusion = CrossModalTokenFusion(
            config.hidden_size,
            num_heads=self._fusion_num_heads(),
            dropout=config.dropout,
        )

    def get_representations(self, *args, **kwargs):
        input_values = kwargs.pop("input_values", None)
        audio_attention_mask = kwargs.pop("audio_attention_mask", None)
        output = BaseGLiNextModel.get_representations(self, *args, **kwargs)
        if input_values is None:
            return output

        token_embeds, prompts_embedding, prompts_embedding_mask, words_embedding, mask = output
        audio_kwargs = {
            key.removeprefix("audio_"): value
            for key, value in kwargs.items()
            if key.startswith("audio_")
        }
        audio_tokens = self.audio_encoder(
            input_values,
            attention_mask=audio_attention_mask,
            **audio_kwargs,
        )
        audio_mask = self._mask_for_tokens(audio_tokens, audio_attention_mask)
        words_embedding = self.audio_fusion(words_embedding, audio_tokens, audio_mask)
        return token_embeds, prompts_embedding, prompts_embedding_mask, words_embedding, mask


class GLiNExTOmniModel(GLiNExTTextVisionUniEncoderModel):
    """Text + vision + audio model with early fusion inside the text transformer."""

    omni_encoder_cls = TriOmniEncoder
    omni_bi_encoder_cls = TriOmniBiEncoder


class GLiNExTTextAudioUniEncoderModel(GLiNExTTextVisionUniEncoderModel):
    """Text + audio model with early fusion inside the text transformer."""

    omni_encoder_cls = TextAudioOmniEncoder
    omni_bi_encoder_cls = TextAudioOmniBiEncoder


def resolve_glinext_model_class(config) -> type[BaseGLiNextModel]:
    variant = _normalize_model_variant(getattr(config, "model_variant", None))
    fusion = _normalize_fusion(getattr(config, "multimodal_fusion", None))

    if variant in {"text", "text-only", "textonly"}:
        if getattr(config, "use_layout", False):
            return GLiNExTTextVisionLayoutModel
        has_vision = bool(getattr(config, "vision_model_name", None) or getattr(config, "vision_encoder_type", None))
        has_audio = bool(getattr(config, "audio_model_name", None) or getattr(config, "audio_encoder_type", None))
        if has_vision and has_audio:
            return GLiNExTOmniModel
        if has_vision:
            return GLiNExTTextVisionBiEncoderModel if fusion == "bi-encoder" else GLiNExTTextVisionUniEncoderModel
        if has_audio:
            return GLiNExTTextAudioModel if fusion == "bi-encoder" else GLiNExTTextAudioUniEncoderModel
        return GLiNExTTextModel

    if variant in {"text-vision", "vision"}:
        return GLiNExTTextVisionBiEncoderModel if fusion == "bi-encoder" else GLiNExTTextVisionUniEncoderModel
    if variant in {"text-vision-layout", "vision-layout", "layout"}:
        return GLiNExTTextVisionLayoutModel
    if variant in {"text-audio", "audio"}:
        return GLiNExTTextAudioModel if fusion == "bi-encoder" else GLiNExTTextAudioUniEncoderModel
    if variant in {"omni", "text-vision-audio"}:
        return GLiNExTOmniModel
    raise ValueError(f"Unknown GLiNExT model_variant: {getattr(config, 'model_variant', None)!r}")


class GLiNExTModel(BaseGLiNextModel):
    """Backward-compatible text model class.

    The user-facing :class:`glinext.glinext.GLiNExT` class selects concrete
    multimodal variants via :func:`resolve_glinext_model_class`.
    """

    pass
