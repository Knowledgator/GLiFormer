"""GLiNExT: unified multi-task model — thin orchestrator over modular task heads."""

from dataclasses import fields
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
from torch import nn

from gliner.modeling.base import BaseModel
from gliner.modeling.layers import CrossFuser, LstmSeq2SeqEncoder
from gliner.modeling.utils import (
    extract_word_embeddings,
    extract_prompt_features,
    extract_prompt_features_and_word_embeddings,
)

from .config import GLiNextConfig
from .outputs import (
    GLiNExTAudioOutput,
    GLiNExTLayoutOutput,
    GLiNExTOmniOutput,
    GLiNExTOutput,
    GLiNExTTextOutput,
    GLiNExTVisionOutput,
)
from .tasks import SharedRepresentations, TASK_REGISTRY, TaskFlatInputs, TaskHeadOutput
from .layers import AnchorModeling, AnchorCrossAttentionLayer
from .encoders.audio import AudioBiEncoder
from .encoders.omni import (
    LayoutBiEncoder,
    LayoutEncoder,
    OmniEncoderOutput,
    TriOmniBiEncoder,
    TriOmniEncoder,
)
from .encoders.text import TextBiEncoder, TextEncoder
from .encoders.vision import VisionBiEncoder


def _normalize_model_variant(value: Optional[str]) -> str:
    return value or "text"


_VARIANT_MODALITIES = {
    "omni": ("text", "vision", "audio"),
}


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


def _filtered_output(output_cls, **kwargs):
    allowed = {field.name for field in fields(output_cls)}
    return output_cls(**{key: value for key, value in kwargs.items() if key in allowed})


def _cache_forward_modality(module: nn.Module, name: str, tokens, mask) -> None:
    cache = getattr(module, "_forward_modality_cache", None)
    if cache is not None and tokens is not None:
        cache[name] = (tokens, mask)


def _apply_modality_input_mask(mask: torch.Tensor, input_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if input_mask is None:
        return mask
    input_mask = input_mask.to(device=mask.device, dtype=mask.dtype)
    if input_mask.dim() == 1:
        input_mask = input_mask[:, None]
    return mask * input_mask


class BaseGLiNextModel(BaseModel):
    """Unified multi-task model composing optional task heads.

    Each task is a standalone TaskHead subclass. The model registers enabled heads
    via nn.ModuleDict and executes them in dependency order during forward().
    """

    enabled_task_names: Optional[Tuple[str, ...]] = None

    def __init__(
        self,
        config: GLiNextConfig,
        from_pretrained: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        super().__init__(config, from_pretrained, cache_dir)

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

        self.heads = nn.ModuleDict()
        enabled_task_names = getattr(self, "enabled_task_names", None)
        enabled_task_names = set(enabled_task_names) if enabled_task_names is not None else None
        for HeadClass in TASK_REGISTRY.head_classes():
            head = HeadClass.from_config(
                config,
                from_pretrained=from_pretrained,
                cache_dir=cache_dir,
                shared_layers=shared_layers,
            )
            if head is not None:
                if enabled_task_names is not None and head.name not in enabled_task_names:
                    continue
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

    @staticmethod
    def _runtime_loss_value(runtime_kwargs: dict, focal_name: str, short_name: str):
        if focal_name in runtime_kwargs and runtime_kwargs[focal_name] is not None:
            return runtime_kwargs[focal_name]
        if short_name in runtime_kwargs and runtime_kwargs[short_name] is not None:
            return runtime_kwargs[short_name]
        return None

    def _resolve_task_focal_loss_kwargs(self, task_name: str, runtime_kwargs: dict) -> dict:
        task_cfg = self.config.get_task_config(task_name)
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
        cache = getattr(self, "_forward_modality_cache", None)
        if cache is not None and "vision" in cache:
            return cache["vision"]
        if pixel_values is None:
            return None, None
        encoder = getattr(self, "vision_encoder", None)
        if encoder is None:
            feature_encoders = getattr(self.token_rep_layer, "feature_encoders", None)
            if feature_encoders is not None and "vision" in feature_encoders:
                encoder = feature_encoders["vision"]
        if encoder is None:
            return None, None
        vision_tokens = encoder(pixel_values)
        if vision_attention_mask is not None and vision_attention_mask.shape[-1] == vision_tokens.shape[1]:
            vision_mask = vision_attention_mask.to(device=vision_tokens.device)
        else:
            vision_mask = torch.ones(vision_tokens.shape[:2], dtype=torch.long, device=vision_tokens.device)
        vision_mask = _apply_modality_input_mask(vision_mask, kwargs.get("vision_input_mask"))
        return vision_tokens, vision_mask

    def _encode_audio_tokens(
        self,
        audio_values: Optional[torch.Tensor],
        audio_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        cache = getattr(self, "_forward_modality_cache", None)
        if cache is not None and "audio" in cache:
            return cache["audio"]
        if audio_values is None:
            return None, None
        encoder = getattr(self, "audio_encoder", None)
        if encoder is None:
            feature_encoders = getattr(self.token_rep_layer, "feature_encoders", None)
            if feature_encoders is not None and "audio" in feature_encoders:
                encoder = feature_encoders["audio"]
        if encoder is None:
            return None, None
        audio_tokens = encoder(audio_values, attention_mask=audio_attention_mask)
        if audio_attention_mask is not None and audio_attention_mask.shape[-1] == audio_tokens.shape[1]:
            audio_mask = audio_attention_mask.to(device=audio_tokens.device)
        else:
            audio_mask = torch.ones(audio_tokens.shape[:2], dtype=torch.long, device=audio_tokens.device)
        audio_mask = _apply_modality_input_mask(audio_mask, kwargs.get("audio_input_mask"))
        return audio_tokens, audio_mask

    def _build_flat_rel_prompts(
        self,
        rel_prompts: torch.Tensor,
        classes_mapping,
        embed_dim: int,
        device,
        dtype,
        rel_prompts_mask: Optional[torch.Tensor] = None,
        label_group_sizes: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Split relation prompts/label embeds into per-extraction-group tensors.

        Prompt-token inputs store relation labels per batch item, concatenated
        across that item's extraction groups. Labels-encoder inputs store
        relation labels globally across flat groups, with ``label_group_sizes``
        preserving the group boundaries.
        """
        if label_group_sizes is None and rel_prompts_mask is None:
            return None, None

        slices: List[Tuple[int, int, int]] = []
        if label_group_sizes is not None:
            cumsum = torch.cumsum(label_group_sizes, 0)
            for flat_idx, batch_idx, _, _ in classes_mapping.flat_extraction_iter():
                c_start = 0 if flat_idx == 0 else int(cumsum[flat_idx - 1].item())
                c_end = int(cumsum[flat_idx].item())
                slices.append((batch_idx, c_start, c_end))
        else:
            item_rel_offset: dict = {}
            for _, batch_idx, _, ext_mapping in classes_mapping.flat_extraction_iter():
                rel_map = ext_mapping.rel_class_to_id
                n_rel = len(rel_map.class_to_id) if rel_map is not None else 0
                start = item_rel_offset.get(batch_idx, 0)
                slices.append((batch_idx, start, start + n_rel))
                item_rel_offset[batch_idx] = start + n_rel

        BN = len(slices)
        if BN == 0:
            return None, None

        max_C = max((ce - cs for _, cs, ce in slices), default=0)
        mask_dtype = rel_prompts_mask.dtype if rel_prompts_mask is not None else torch.long
        flat = torch.zeros(BN, max_C, embed_dim, device=device, dtype=dtype)
        flat_mask = torch.zeros(BN, max_C, device=device, dtype=mask_dtype)
        for idx, (bi, cs, ce) in enumerate(slices):
            n = ce - cs
            if n > 0 and ce <= rel_prompts.shape[1]:
                flat[idx, :n] = rel_prompts[bi, cs:ce]
                if rel_prompts_mask is not None:
                    flat_mask[idx, :n] = rel_prompts_mask[bi, cs:ce]
                else:
                    flat_mask[idx, :n] = 1
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

        flat_iter = classes_mapping.flat_iter(task_name)

        # Collect group descriptors
        batch_origins: List[int] = []
        parent_positions: List[Tuple[int, int]] = []  # (batch_idx, parent_pos)
        child_slices: List[Tuple[int, int, int]] = []  # (batch_idx, start, end)

        # Track per-item child offset for prompt-based splitting
        item_child_offset: dict = {}

        for flat_idx, batch_idx, group_idx, _ in flat_iter:
            batch_origins.append(batch_idx)

            # Parent position within the parent tensor
            if per_task_parents:
                # Per-task parents: positions are task-local (no offset)
                parent_positions.append((batch_idx, group_idx))
            else:
                # Shared parents: offset by preceding tasks in prompt order
                p_offset = classes_mapping.parent_offset_for_item(task_name, batch_idx)
                parent_positions.append((batch_idx, p_offset + group_idx))

            if label_group_sizes is None:
                # Prompt path: children within batch item, accumulated by group
                if batch_idx not in item_child_offset:
                    item_child_offset[batch_idx] = 0
                c_start = item_child_offset[batch_idx]
                c_size = classes_mapping.child_size(task_name, batch_idx, group_idx)
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
            feature_embedding=flat_words,
            feature_mask=flat_word_mask,
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

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement its own forward pass."
        )

    def loss(self, *args, **kwargs):
        """Compute loss via forward pass."""
        output = self.forward(*args, **kwargs)
        return output.loss


class _GLiNExTJointForwardModel(BaseGLiNextModel):
    """Shared text/joint task orchestration for concrete GLiNExT models."""

    output_cls = GLiNExTOutput
    _media_task_names = (
        "image_classification", "object_detection", "segmentation",
        "audio_classification", "audio_segmentation",
    )
    _flat_input_task_names = (
        "ner", "joint_relex", "classification", "structuring", "open_relex",
        "image_classification", "object_detection", "segmentation",
        "audio_classification", "audio_segmentation",
    )
    _flat_required_task_names = _flat_input_task_names + ("count",)
    _focal_loss_task_names = (
        "ner", "classification", "joint_relex", "open_relex", "structuring",
        "image_classification", "object_detection", "segmentation",
        "audio_classification", "audio_segmentation",
    )

    @staticmethod
    def _resolve_media_argument(value, kwargs: dict, key: str):
        return kwargs.get(key) if value is None else value

    def _encode_forward_representations(
        self,
        *,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        words_mask: Optional[torch.Tensor],
        text_lengths: Optional[torch.Tensor],
        labels_input_ids: Optional[torch.Tensor],
        labels_attention_mask: Optional[torch.Tensor],
        pixel_values: Optional[torch.Tensor],
        vision_attention_mask: Optional[torch.Tensor],
        audio_values: Optional[torch.Tensor],
        audio_attention_mask: Optional[torch.Tensor],
        include_media: bool,
        kwargs: dict,
    ) -> dict:
        representation_keys = {
            "packing_config",
            "pair_attention_mask",
            "token_type_ids",
            "position_ids",
            "head_mask",
            "output_attentions",
            "output_hidden_states",
            "return_dict",
            "bbox",
            "pixel_values",
            "audio_values",
            "vision_attention_mask",
            "audio_attention_mask",
            "vision_input_mask",
            "audio_input_mask",
        }
        direct_media_keys = {
            "pixel_values",
            "vision_attention_mask",
            "audio_values",
            "audio_attention_mask",
        }
        representation_kwargs = {
            key: kwargs[key]
            for key in kwargs
            if (
                key not in direct_media_keys
                and key in representation_keys
            )
        }
        if include_media:
            representation_kwargs.update(
                {
                    "pixel_values": pixel_values,
                    "vision_attention_mask": vision_attention_mask,
                    "audio_values": audio_values,
                    "audio_attention_mask": audio_attention_mask,
                    "vision_input_mask": kwargs.get("vision_input_mask"),
                    "audio_input_mask": kwargs.get("audio_input_mask"),
                }
            )

        token_embeds, prompts_embedding, prompts_embedding_mask, words_embedding, mask = (
            self.get_representations(
                input_ids,
                attention_mask,
                text_lengths,
                words_mask,
                labels_input_ids=labels_input_ids,
                labels_attention_mask=labels_attention_mask,
                **representation_kwargs,
            )
        )

        if include_media:
            vision_embedding, vision_mask = self._encode_vision_tokens(
                pixel_values,
                vision_attention_mask=vision_attention_mask,
                **kwargs,
            )
            audio_embedding, audio_mask = self._encode_audio_tokens(
                audio_values,
                audio_attention_mask=audio_attention_mask,
                **kwargs,
            )
        else:
            vision_embedding, vision_mask = None, None
            audio_embedding, audio_mask = None, None

        return {
            "token_embeds": token_embeds,
            "prompts_embedding": prompts_embedding,
            "prompts_embedding_mask": prompts_embedding_mask,
            "words_embedding": words_embedding,
            "mask": mask,
            "vision_embedding": vision_embedding,
            "vision_mask": vision_mask,
            "audio_embedding": audio_embedding,
            "audio_mask": audio_mask,
        }

    def _build_shared_representations(
        self,
        *,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        image_sizes: Optional[torch.Tensor],
        include_media: bool,
        representations: dict,
    ) -> SharedRepresentations:
        return SharedRepresentations(
            token_embeds=representations["token_embeds"],
            input_ids=input_ids,
            attention_mask=attention_mask,
            words_embedding=representations["words_embedding"],
            mask=representations["mask"],
            prompts_embedding=representations["prompts_embedding"],
            prompts_embedding_mask=representations["prompts_embedding_mask"],
            vision_embedding=representations["vision_embedding"],
            vision_mask=representations["vision_mask"],
            audio_embedding=representations["audio_embedding"],
            audio_mask=representations["audio_mask"],
            image_sizes=image_sizes if include_media else None,
        )

    def _build_forward_flat_inputs(
        self,
        *,
        classes_mapping,
        token_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        words_embedding: torch.Tensor,
        mask: torch.Tensor,
        prompts_embedding: torch.Tensor,
        prompts_embedding_mask: torch.Tensor,
        batch_size: int,
        embed_dim: int,
        cat_label_embeds: Optional[torch.Tensor],
        rel_label_embeds: Optional[torch.Tensor],
        child_label_embeds: Optional[torch.Tensor],
        open_rel_label_embeds: Optional[torch.Tensor],
        vision_embedding: Optional[torch.Tensor],
        vision_mask: Optional[torch.Tensor],
        audio_embedding: Optional[torch.Tensor],
        audio_mask: Optional[torch.Tensor],
        include_media: bool,
        kwargs: dict,
    ) -> Tuple[Dict[str, TaskFlatInputs], Optional[torch.Tensor], Optional[torch.Tensor]]:
        parent_embeds = None
        parent_mask_t = None
        per_task_parent_embeds: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        media_tasks = self._media_task_names if include_media else ()

        if classes_mapping is not None:
            if self.config.uses_per_task_parents:
                task_parent_cfgs = {
                    "ner": self.config.ner_config,
                    "classification": self.config.classification_config,
                    "open_relex": self.config.open_relex_config,
                    "structuring": self.config.structuring_config,
                }
                if include_media:
                    task_parent_cfgs.update(
                        {
                            "image_classification": self.config.image_classification_config,
                            "object_detection": self.config.object_detection_config,
                            "segmentation": self.config.segmentation_config,
                            "audio_classification": self.config.audio_classification_config,
                            "audio_segmentation": self.config.audio_segmentation_config,
                        }
                    )
                for task_name, task_cfg in task_parent_cfgs.items():
                    if task_cfg is not None and getattr(task_cfg, "parent_token_index", -1) > 0:
                        parent_e, parent_m = extract_prompt_features(
                            task_cfg.parent_token_index,
                            token_embeds,
                            input_ids,
                            attention_mask,
                            batch_size,
                            embed_dim,
                            getattr(task_cfg, "embed_parent_token", True),
                        )
                        per_task_parent_embeds[task_name] = (parent_e, parent_m)
                if "ner" in per_task_parent_embeds:
                    per_task_parent_embeds["joint_relex"] = per_task_parent_embeds["ner"]
            elif self.config.parent_token_index > 0:
                parent_embeds, parent_mask_t = extract_prompt_features(
                    self.config.parent_token_index,
                    token_embeds,
                    input_ids,
                    attention_mask,
                    batch_size,
                    embed_dim,
                    self.config.embed_parent_token,
                )

        ner_child_embeds = prompts_embedding
        ner_child_mask = prompts_embedding_mask

        cat_child_embeds, cat_child_mask = None, None
        if "classification" in self.heads and cat_label_embeds is None:
            cat_cfg = self.config.classification_config
            cat_child_embeds, cat_child_mask = extract_prompt_features(
                cat_cfg.cat_token_index,
                token_embeds,
                input_ids,
                attention_mask,
                batch_size,
                embed_dim,
                cat_cfg.embed_cat_token,
            )
        elif cat_label_embeds is not None:
            cat_child_embeds = cat_label_embeds
            cat_child_mask = torch.ones(
                cat_label_embeds.shape[:-1],
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        joint_rel_flat_prompts, joint_rel_flat_mask = None, None
        if "joint_relex" in self.heads and rel_label_embeds is None and classes_mapping is not None:
            jr_cfg = self.config.joint_relex_config
            rel_prompts_batch, rel_prompts_batch_mask = extract_prompt_features(
                jr_cfg.rel_token_index,
                token_embeds,
                input_ids,
                attention_mask,
                batch_size,
                embed_dim,
                jr_cfg.embed_rel_token,
            )
            joint_rel_flat_prompts, joint_rel_flat_mask = self._build_flat_rel_prompts(
                rel_prompts_batch,
                classes_mapping,
                embed_dim,
                device=words_embedding.device,
                dtype=rel_prompts_batch.dtype,
                rel_prompts_mask=rel_prompts_batch_mask,
            )
        elif "joint_relex" in self.heads and rel_label_embeds is not None and classes_mapping is not None:
            joint_rel_flat_prompts, joint_rel_flat_mask = self._build_flat_rel_prompts(
                rel_label_embeds,
                classes_mapping,
                embed_dim,
                device=words_embedding.device,
                dtype=rel_label_embeds.dtype,
                label_group_sizes=kwargs.get("rel_labels_group_size"),
            )

        open_rel_child_embeds, open_rel_child_mask = None, None
        if "open_relex" in self.heads and open_rel_label_embeds is None:
            or_cfg = self.config.open_relex_config
            open_rel_child_embeds, open_rel_child_mask = extract_prompt_features(
                or_cfg.rel_token_index,
                token_embeds,
                input_ids,
                attention_mask,
                batch_size,
                embed_dim,
                or_cfg.embed_rel_token,
            )
        elif open_rel_label_embeds is not None:
            open_rel_child_embeds = open_rel_label_embeds
            open_rel_child_mask = torch.ones(
                open_rel_label_embeds.shape[:-1],
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        struct_child_embeds, struct_child_mask = None, None
        if "structuring" in self.heads and child_label_embeds is None:
            s_cfg = self.config.structuring_config
            struct_child_embeds, struct_child_mask = extract_prompt_features(
                s_cfg.child_token_index,
                token_embeds,
                input_ids,
                attention_mask,
                batch_size,
                embed_dim,
                s_cfg.embed_child_token,
            )
        elif child_label_embeds is not None:
            struct_child_embeds = child_label_embeds
            struct_child_mask = torch.ones(
                child_label_embeds.shape[:-1],
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        obj_child_embeds, obj_child_mask = None, None
        if media_tasks and any(name in self.heads for name in media_tasks):
            obj_child_embeds, obj_child_mask = extract_prompt_features(
                self.config.obj_token_index,
                token_embeds,
                input_ids,
                attention_mask,
                batch_size,
                embed_dim,
                self.config.embed_obj_token,
            )

        media_child_prompts: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {}
        if obj_child_embeds is not None and classes_mapping is not None:
            offsets = [0 for _ in range(batch_size)]
            for task_name in media_tasks:
                if task_name not in self.heads:
                    continue
                counts = []
                mapping_list = getattr(classes_mapping, f"{task_name}_mapping", [])
                for batch_idx in range(batch_size):
                    if batch_idx < len(mapping_list):
                        counts.append(
                            sum(len(item.class_to_id.class_to_id) for item in mapping_list[batch_idx].items)
                        )
                    else:
                        counts.append(0)
                max_count = max(counts, default=0)
                if max_count == 0:
                    media_child_prompts[task_name] = (None, None)
                    continue
                task_embeds = torch.zeros(
                    batch_size,
                    max_count,
                    embed_dim,
                    device=obj_child_embeds.device,
                    dtype=obj_child_embeds.dtype,
                )
                task_mask = torch.zeros(
                    batch_size,
                    max_count,
                    device=obj_child_embeds.device,
                    dtype=obj_child_mask.dtype,
                )
                for batch_idx, count in enumerate(counts):
                    start = offsets[batch_idx]
                    end = start + count
                    if count > 0 and end <= obj_child_embeds.shape[1]:
                        task_embeds[batch_idx, :count] = obj_child_embeds[batch_idx, start:end]
                        task_mask[batch_idx, :count] = obj_child_mask[batch_idx, start:end]
                    offsets[batch_idx] = end
                media_child_prompts[task_name] = (task_embeds, task_mask)

        flat_inputs_map: Dict[str, TaskFlatInputs] = {}
        use_per_task = bool(per_task_parent_embeds)
        has_parents = parent_embeds is not None or use_per_task

        if classes_mapping is not None and has_parents:
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

            task_names = (
                "ner", "joint_relex", "classification", "structuring", "open_relex",
                *media_tasks,
            )
            for task_name in task_names:
                if task_name not in self.heads:
                    continue
                child_e, child_m = task_child_map.get(task_name, (None, None))
                if child_e is None:
                    continue

                if use_per_task:
                    if task_name not in per_task_parent_embeds:
                        continue
                    task_parent_e, task_parent_m = per_task_parent_embeds[task_name]
                else:
                    task_parent_e, task_parent_m = parent_embeds, parent_mask_t

                label_group_sizes = label_group_sizes_map.get(
                    "ner" if task_name == "joint_relex" else task_name
                )
                flat_words, flat_mask = self._features_for_task(
                    task_name=task_name,
                    words_embedding=words_embedding,
                    word_mask=mask,
                    vision_embedding=vision_embedding,
                    vision_mask=vision_mask,
                    audio_embedding=audio_embedding,
                    audio_mask=audio_mask,
                )
                flat_inputs = self._build_flat_inputs(
                    task_parent_e,
                    task_parent_m,
                    child_e,
                    child_m,
                    flat_words,
                    flat_mask,
                    classes_mapping,
                    task_name,
                    label_group_sizes=label_group_sizes,
                    per_task_parents=use_per_task,
                )
                if flat_inputs is not None:
                    flat_inputs_map[task_name] = flat_inputs

            if "count" in self.heads:
                self._build_count_flat_inputs(
                    flat_inputs_map,
                    classes_mapping=classes_mapping,
                    words_embedding=words_embedding,
                    mask=mask,
                    embed_dim=embed_dim,
                    parent_embeds=parent_embeds,
                    per_task_parent_embeds=per_task_parent_embeds,
                    use_per_task=use_per_task,
                )

        return flat_inputs_map, joint_rel_flat_prompts, joint_rel_flat_mask

    @staticmethod
    def _features_for_task(
        *,
        task_name: str,
        words_embedding: torch.Tensor,
        word_mask: torch.Tensor,
        vision_embedding: Optional[torch.Tensor],
        vision_mask: Optional[torch.Tensor],
        audio_embedding: Optional[torch.Tensor],
        audio_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if task_name in ("image_classification", "object_detection", "segmentation"):
            if vision_embedding is None or vision_mask is None:
                raise ValueError(
                    f"{task_name} requires vision embeddings. Provide pixel_values "
                    "for rows with active vision task groups."
                )
            return vision_embedding, vision_mask

        if task_name in ("audio_classification", "audio_segmentation"):
            if audio_embedding is None or audio_mask is None:
                raise ValueError(
                    f"{task_name} requires audio embeddings. Provide audio_values, "
                    "audio, or audio feature tensors for rows with active audio task groups."
                )
            return audio_embedding, audio_mask

        return words_embedding, word_mask

    def _build_count_flat_inputs(
        self,
        flat_inputs_map: Dict[str, TaskFlatInputs],
        *,
        classes_mapping,
        words_embedding: torch.Tensor,
        mask: torch.Tensor,
        embed_dim: int,
        parent_embeds: Optional[torch.Tensor],
        per_task_parent_embeds: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        use_per_task: bool,
    ) -> None:
        count_batch_origins = []
        count_parent_embeds_list = []
        count_tasks = [
            ("classification", classes_mapping.flat_cat_iter),
            ("ner", classes_mapping.flat_extraction_iter),
            ("structuring", classes_mapping.flat_structuring_iter),
        ]

        for task_name, flat_iter in count_tasks:
            if use_per_task:
                task_parent_pair = per_task_parent_embeds.get(task_name)
                if task_parent_pair is None:
                    continue
                task_parent_e, _ = task_parent_pair
                for _, batch_idx, group_idx, _ in flat_iter():
                    count_batch_origins.append(batch_idx)
                    if group_idx < task_parent_e.shape[1]:
                        count_parent_embeds_list.append(task_parent_e[batch_idx, group_idx])
                    else:
                        count_parent_embeds_list.append(
                            torch.zeros(embed_dim, device=words_embedding.device, dtype=task_parent_e.dtype)
                        )
            else:
                for _, batch_idx, group_idx, _ in flat_iter():
                    count_batch_origins.append(batch_idx)
                    parent_offset = classes_mapping.parent_offset_for_item(task_name, batch_idx)
                    parent_pos = parent_offset + group_idx
                    if parent_embeds is not None and parent_pos < parent_embeds.shape[1]:
                        count_parent_embeds_list.append(parent_embeds[batch_idx, parent_pos])
                    elif parent_embeds is not None:
                        count_parent_embeds_list.append(
                            torch.zeros(embed_dim, device=words_embedding.device, dtype=parent_embeds.dtype)
                        )

        count_size = len(count_batch_origins)
        if count_size == 0:
            return
        batch_origin = torch.tensor(count_batch_origins, dtype=torch.long, device=words_embedding.device)
        flat_inputs_map["count"] = TaskFlatInputs(
            words_embedding=words_embedding[batch_origin],
            mask=mask[batch_origin],
            parent_embedding=torch.stack(count_parent_embeds_list),
            child_embedding=torch.zeros(count_size, 0, embed_dim, device=words_embedding.device),
            child_mask=torch.zeros(count_size, 0, device=words_embedding.device),
            batch_origin=batch_origin,
            feature_embedding=words_embedding[batch_origin],
            feature_mask=mask[batch_origin],
        )

    def _encode_embedding_pair_inputs(
        self,
        embedding_input_ids: Optional[torch.Tensor],
        embedding_attention_mask: Optional[torch.Tensor],
        embedding_pair_idx: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if embedding_input_ids is None or embedding_pair_idx is None:
            return None, None
        return (
            self.encode_embedding_tokens(embedding_input_ids, embedding_attention_mask),
            embedding_attention_mask,
        )

    def _execute_forward_heads(
        self,
        *,
        shared: SharedRepresentations,
        flat_inputs_map: Dict[str, TaskFlatInputs],
        batch_kwargs: dict,
        rel_label_embeds: Optional[torch.Tensor],
        joint_rel_flat_prompts: Optional[torch.Tensor],
        joint_rel_flat_mask: Optional[torch.Tensor],
        structuring_count: Optional[torch.Tensor],
        manual_structuring_count: Optional[int],
        runtime_kwargs: dict,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, TaskHeadOutput]]:
        total_loss = torch.tensor(0.0, device=shared.words_embedding.device)
        head_outputs: Dict[str, TaskHeadOutput] = {}

        for name in TASK_REGISTRY.execution_order:
            if name not in self.heads:
                continue
            if name in self._flat_required_task_names and name not in flat_inputs_map:
                continue

            head = self.heads[name]
            dep_outputs = {dep: head_outputs[dep] for dep in head.dependencies if dep in head_outputs}
            extra_kwargs = {}

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

            if name == "joint_relex" and rel_label_embeds is not None:
                extra_kwargs["rel_label_embeds"] = rel_label_embeds
            if name == "joint_relex":
                extra_kwargs["flat_rel_prompts"] = joint_rel_flat_prompts
                extra_kwargs["flat_rel_prompts_mask"] = joint_rel_flat_mask

            if name in self._focal_loss_task_names:
                extra_kwargs["base_loss_fn"] = self._make_task_loss_fn(name, runtime_kwargs)

            call_kwargs = dict(batch_kwargs)
            call_kwargs.update(extra_kwargs)
            output = head(shared, dependency_outputs=dep_outputs, **call_kwargs)
            head_outputs[name] = output

            if output.loss is not None:
                total_loss = total_loss + head.loss_coef * output.loss

        final_loss = total_loss if any(output.loss is not None for output in head_outputs.values()) else None
        return final_loss, head_outputs

    def _collect_forward_output(
        self,
        *,
        final_loss: Optional[torch.Tensor],
        head_outputs: Dict[str, TaskHeadOutput],
        flat_inputs_map: Dict[str, TaskFlatInputs],
        batch_size: int,
        words_embedding: torch.Tensor,
        mask: torch.Tensor,
        prompts_embedding: torch.Tensor,
        prompts_embedding_mask: torch.Tensor,
        vision_embedding: Optional[torch.Tensor],
        vision_mask: Optional[torch.Tensor],
        audio_embedding: Optional[torch.Tensor],
        audio_mask: Optional[torch.Tensor],
    ) -> GLiNExTOutput:
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

        return _filtered_output(
            self.output_cls,
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
            vision_embedding=vision_embedding,
            vision_mask=vision_mask,
            audio_embedding=audio_embedding,
            audio_mask=audio_mask,
            words_embedding=words_embedding,
            mask=mask,
            prompts_embedding=prompts_embedding,
            prompts_embedding_mask=prompts_embedding_mask,
        )

    def _forward_task_heads(
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
        audio_values: Optional[torch.Tensor] = None,
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
        include_media: bool = True,
        **kwargs,
    ) -> GLiNExTOutput:

        self._forward_modality_cache = {}
        classes_mapping = kwargs.get("classes_mapping")
        pixel_values = self._resolve_media_argument(pixel_values, kwargs, "pixel_values")
        vision_attention_mask = self._resolve_media_argument(
            vision_attention_mask, kwargs, "vision_attention_mask"
        )
        audio_values = self._resolve_media_argument(audio_values, kwargs, "audio_values")
        audio_attention_mask = self._resolve_media_argument(
            audio_attention_mask, kwargs, "audio_attention_mask"
        )

        # ── 1. Encode ────────────────────────────────────────────────────
        representations = self._encode_forward_representations(
            input_ids=input_ids,
            attention_mask=attention_mask,
            words_mask=words_mask,
            text_lengths=text_lengths,
            labels_input_ids=labels_input_ids,
            labels_attention_mask=labels_attention_mask,
            pixel_values=pixel_values,
            vision_attention_mask=vision_attention_mask,
            audio_values=audio_values,
            audio_attention_mask=audio_attention_mask,
            include_media=include_media,
            kwargs=kwargs,
        )
        token_embeds = representations["token_embeds"]
        prompts_embedding = representations["prompts_embedding"]
        prompts_embedding_mask = representations["prompts_embedding_mask"]
        words_embedding = representations["words_embedding"]
        mask = representations["mask"]
        vision_embedding = representations["vision_embedding"]
        vision_mask = representations["vision_mask"]
        audio_embedding = representations["audio_embedding"]
        audio_mask = representations["audio_mask"]

        shared = self._build_shared_representations(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_sizes=image_sizes,
            include_media=include_media,
            representations=representations,
        )

        batch_size = words_embedding.shape[0]
        embed_dim = words_embedding.shape[-1]

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

        # ── 1c-e. Build TaskFlatInputs per task ─────────────────────────
        flat_inputs_map, joint_rel_flat_prompts, joint_rel_flat_mask = (
            self._build_forward_flat_inputs(
                classes_mapping=classes_mapping,
                token_embeds=token_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask,
                words_embedding=words_embedding,
                mask=mask,
                prompts_embedding=prompts_embedding,
                prompts_embedding_mask=prompts_embedding_mask,
                batch_size=batch_size,
                embed_dim=embed_dim,
                cat_label_embeds=cat_label_embeds,
                rel_label_embeds=rel_label_embeds,
                child_label_embeds=child_label_embeds,
                open_rel_label_embeds=open_rel_label_embeds,
                vision_embedding=vision_embedding,
                vision_mask=vision_mask,
                audio_embedding=audio_embedding,
                audio_mask=audio_mask,
                include_media=include_media,
                kwargs=kwargs,
            )
        )

        # ── 1f. Encode embedding pair texts (separate batch) ───────────
        embedding_encodings, embedding_encoding_mask = self._encode_embedding_pair_inputs(
            embedding_input_ids,
            embedding_attention_mask,
            embedding_pair_idx,
        )

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

        # ── 2-3. Execute heads and collect outputs ─────────────────────
        final_loss, head_outputs = self._execute_forward_heads(
            shared=shared,
            flat_inputs_map=flat_inputs_map,
            batch_kwargs=batch_kwargs,
            rel_label_embeds=rel_label_embeds,
            joint_rel_flat_prompts=joint_rel_flat_prompts,
            joint_rel_flat_mask=joint_rel_flat_mask,
            structuring_count=structuring_count,
            manual_structuring_count=manual_structuring_count,
            runtime_kwargs=kwargs,
        )
        return self._collect_forward_output(
            final_loss=final_loss,
            head_outputs=head_outputs,
            flat_inputs_map=flat_inputs_map,
            batch_size=batch_size,
            words_embedding=words_embedding,
            mask=mask,
            prompts_embedding=prompts_embedding,
            prompts_embedding_mask=prompts_embedding_mask,
            vision_embedding=vision_embedding,
            vision_mask=vision_mask,
            audio_embedding=audio_embedding,
            audio_mask=audio_mask,
        )

    def _forward_omni_task_heads(self, *args, **kwargs) -> GLiNExTOutput:
        return self._forward_task_heads(*args, include_media=True, **kwargs)

    def _forward_text_task_heads(
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
        manual_structuring_count: Optional[int] = None,
        **kwargs,
    ) -> GLiNExTTextOutput:
        classes_mapping = kwargs.get("classes_mapping")
        representation_keys = {
            "packing_config",
            "pair_attention_mask",
            "bbox",
            "page_token_ids",
            "pixel_values",
            "vision_attention_mask",
            "image_batch_idx",
            "image_page_ids",
        }
        representation_kwargs = {
            key: kwargs[key]
            for key in kwargs
            if key in representation_keys
        }
        token_embeds, prompts_embedding, prompts_embedding_mask, words_embedding, mask = (
            self.get_representations(
                input_ids,
                attention_mask,
                text_lengths,
                words_mask,
                labels_input_ids=labels_input_ids,
                labels_attention_mask=labels_attention_mask,
                **representation_kwargs,
            )
        )

        shared = SharedRepresentations(
            token_embeds=token_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            words_embedding=words_embedding,
            mask=mask,
            prompts_embedding=prompts_embedding,
            prompts_embedding_mask=prompts_embedding_mask,
            vision_embedding=None,
            vision_mask=None,
            audio_embedding=None,
            audio_mask=None,
            image_sizes=None,
        )

        batch_size = words_embedding.shape[0]
        embed_dim = words_embedding.shape[-1]
        cat_label_embeds, rel_label_embeds, child_label_embeds, open_rel_label_embeds = (
            self._encode_all_labels_batched(
                batch_size,
                cat_labels_input_ids, cat_labels_attention_mask,
                rel_labels_input_ids, rel_labels_attention_mask,
                child_labels_input_ids, child_labels_attention_mask,
                open_rel_labels_input_ids, open_rel_labels_attention_mask,
            )
        )
        flat_inputs_map, joint_rel_flat_prompts, joint_rel_flat_mask = (
            self._build_forward_flat_inputs(
                classes_mapping=classes_mapping,
                token_embeds=token_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask,
                words_embedding=words_embedding,
                mask=mask,
                prompts_embedding=prompts_embedding,
                prompts_embedding_mask=prompts_embedding_mask,
                batch_size=batch_size,
                embed_dim=embed_dim,
                cat_label_embeds=cat_label_embeds,
                rel_label_embeds=rel_label_embeds,
                child_label_embeds=child_label_embeds,
                open_rel_label_embeds=open_rel_label_embeds,
                vision_embedding=None,
                vision_mask=None,
                audio_embedding=None,
                audio_mask=None,
                include_media=False,
                kwargs=kwargs,
            )
        )
        embedding_encodings, embedding_encoding_mask = self._encode_embedding_pair_inputs(
            embedding_input_ids,
            embedding_attention_mask,
            embedding_pair_idx,
        )
        batch_kwargs = dict(
            ner_labels=ner_labels, span_idx=span_idx, span_mask=span_mask,
            span_labels=span_labels, cat_labels=cat_labels, rel_labels=rel_labels,
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
        final_loss, head_outputs = self._execute_forward_heads(
            shared=shared,
            flat_inputs_map=flat_inputs_map,
            batch_kwargs=batch_kwargs,
            rel_label_embeds=rel_label_embeds,
            joint_rel_flat_prompts=joint_rel_flat_prompts,
            joint_rel_flat_mask=joint_rel_flat_mask,
            structuring_count=structuring_count,
            manual_structuring_count=manual_structuring_count,
            runtime_kwargs=kwargs,
        )
        return self._collect_forward_output(
            final_loss=final_loss,
            head_outputs=head_outputs,
            flat_inputs_map=flat_inputs_map,
            batch_size=batch_size,
            words_embedding=words_embedding,
            mask=mask,
            prompts_embedding=prompts_embedding,
            prompts_embedding_mask=prompts_embedding_mask,
            vision_embedding=None,
            vision_mask=None,
            audio_embedding=None,
            audio_mask=None,
        )

    def _forward_all_tasks(self, *args, **kwargs) -> GLiNExTOutput:
        return self._forward_omni_task_heads(*args, **kwargs)

    def _forward_text_tasks(self, *args, **kwargs) -> GLiNExTTextOutput:
        return self._forward_text_task_heads(*args, **kwargs)

class GLiNExTTextModel(_GLiNExTJointForwardModel):
    """Text-only GLiNExT model."""

    enabled_task_names = TASK_REGISTRY.text_tasks
    output_cls = GLiNExTTextOutput

    @staticmethod
    def _reject_media_inputs(kwargs: dict) -> None:
        media_keys = (
            "pixel_values",
            "vision_attention_mask",
            "audio_values",
            "input_values",
            "audio_attention_mask",
        )
        present = [key for key in media_keys if kwargs.get(key) is not None]
        if present:
            raise ValueError(
                "GLiNExTTextModel supports text/layout inputs only; "
                f"received media arguments: {', '.join(present)}"
            )

    def forward(self, *args, **kwargs) -> GLiNExTTextOutput:
        self._reject_media_inputs(kwargs)
        output = self._forward_text_task_heads(*args, **kwargs)
        if type(output) is GLiNExTTextOutput:
            return output
        return _filtered_output(
            GLiNExTTextOutput,
            **{field.name: getattr(output, field.name, None) for field in fields(GLiNExTTextOutput)},
        )


class _MediaOnlyBiEncoderModel(BaseGLiNextModel):
    """Shared implementation for efficient single-media bi-encoder models."""

    media_task_names: Tuple[str, ...] = ()
    media_token_name: str = ""
    media_mask_name: str = ""
    bi_encoder_cls = None

    def _init_token_rep_layer(self, config, from_pretrained, cache_dir):
        if self.bi_encoder_cls is None:
            raise NotImplementedError("media-only model must define bi_encoder_cls")
        return self.bi_encoder_cls(config, from_pretrained=from_pretrained, cache_dir=cache_dir)

    def __init__(self, config, from_pretrained=False, cache_dir=None):
        super().__init__(config, from_pretrained=from_pretrained, cache_dir=cache_dir)
        self.media_parent_embeddings = nn.ParameterDict(
            {
                task_name: nn.Parameter(torch.zeros(config.hidden_size))
                for task_name in self.media_task_names
            }
        )
        for param in self.media_parent_embeddings.values():
            nn.init.normal_(param, std=0.02)

    @staticmethod
    def _mask_for_tokens(tokens: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None and mask.shape[-1] == tokens.shape[1]:
            return mask.to(device=tokens.device)
        return torch.ones(tokens.shape[:2], dtype=torch.long, device=tokens.device)

    def _encode_media_tokens(self, media_values: torch.Tensor, media_mask: Optional[torch.Tensor], **kwargs):
        raise NotImplementedError

    def _parent_inputs_for_task(
        self,
        classes_mapping,
        task_name: str,
        batch_size: int,
        device,
        dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        counts = classes_mapping.group_counts(task_name, batch_size)
        max_groups = max(max(counts, default=0), 1)
        parent = torch.zeros(batch_size, max_groups, self.config.hidden_size, device=device, dtype=dtype)
        parent_mask = torch.zeros(batch_size, max_groups, device=device, dtype=torch.long)
        task_parent = self.media_parent_embeddings[task_name].to(device=device, dtype=dtype)
        for batch_idx, count in enumerate(counts):
            if count > 0:
                parent[batch_idx, :count] = task_parent
                parent_mask[batch_idx, :count] = 1
        return parent, parent_mask

    def _encode_task_labels(
        self,
        task_name: str,
        batch_size: int,
        attention_dtype: torch.dtype,
        kwargs: dict,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        input_ids = kwargs.get(f"{task_name}_labels_input_ids")
        attention_mask = kwargs.get(f"{task_name}_labels_attention_mask")
        group_sizes = kwargs.get(f"{task_name}_labels_group_size")
        if input_ids is None or attention_mask is None or group_sizes is None:
            return None, None, None
        labels_embeds = self.token_rep_layer.encode_labels(input_ids, attention_mask)
        labels_embeds = labels_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        labels_mask = torch.ones(
            labels_embeds.shape[:-1],
            dtype=attention_dtype,
            device=labels_embeds.device,
        )
        return labels_embeds, labels_mask, group_sizes.to(device=labels_embeds.device)

    def _build_media_flat_inputs(
        self,
        classes_mapping,
        media_tokens: torch.Tensor,
        media_mask: torch.Tensor,
        kwargs: dict,
    ) -> Dict[str, TaskFlatInputs]:
        flat_inputs_map: Dict[str, TaskFlatInputs] = {}
        batch_size = media_tokens.shape[0]
        for task_name in self.media_task_names:
            if task_name not in self.heads:
                continue
            child_embeds, child_mask, group_sizes = self._encode_task_labels(
                task_name,
                batch_size,
                media_mask.dtype,
                kwargs,
            )
            if child_embeds is None:
                continue
            parent_embeds, parent_mask = self._parent_inputs_for_task(
                classes_mapping,
                task_name,
                batch_size,
                media_tokens.device,
                media_tokens.dtype,
            )
            flat_inputs = self._build_flat_inputs(
                parent_embeds,
                parent_mask,
                child_embeds,
                child_mask,
                media_tokens,
                media_mask,
                classes_mapping,
                task_name,
                label_group_sizes=group_sizes,
                per_task_parents=True,
            )
            if flat_inputs is not None:
                flat_inputs_map[task_name] = flat_inputs
        return flat_inputs_map

    def _forward_media_heads(
        self,
        shared: SharedRepresentations,
        flat_inputs_map: Dict[str, TaskFlatInputs],
        batch_kwargs: dict,
        runtime_kwargs: dict,
        media_tokens: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, TaskHeadOutput]]:
        total_loss = torch.tensor(0.0, device=media_tokens.device)
        head_outputs: Dict[str, TaskHeadOutput] = {}
        for name in TASK_REGISTRY.execution_order:
            if name not in self.heads or name not in flat_inputs_map:
                continue
            head = self.heads[name]
            call_kwargs = dict(batch_kwargs)
            call_kwargs["flat_inputs"] = flat_inputs_map[name]
            call_kwargs["base_loss_fn"] = self._make_task_loss_fn(name, runtime_kwargs)
            output = head(shared, dependency_outputs={}, **call_kwargs)
            head_outputs[name] = output
            if output.loss is not None:
                total_loss = total_loss + head.loss_coef * output.loss
        if any(output.loss is not None for output in head_outputs.values()):
            return total_loss, head_outputs
        return None, head_outputs

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement a modality-specific forward pass."
        )


class GLiNExTVisionModel(_MediaOnlyBiEncoderModel):
    """Vision-only GLiNExT model using a vision/text-label bi-encoder."""

    enabled_task_names = TASK_REGISTRY.vision_tasks
    output_cls = GLiNExTVisionOutput
    media_task_names = TASK_REGISTRY.vision_tasks
    media_token_name = "vision"
    media_mask_name = "vision_attention_mask"
    bi_encoder_cls = VisionBiEncoder

    def _encode_media_tokens(self, media_values: torch.Tensor, media_mask: Optional[torch.Tensor], **kwargs):
        return self.token_rep_layer.vision_encoder(media_values)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        vision_attention_mask: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        image_classification_labels: Optional[torch.Tensor] = None,
        object_detection_class_labels: Optional[torch.Tensor] = None,
        object_detection_bbox_labels: Optional[torch.Tensor] = None,
        object_detection_object_mask: Optional[torch.Tensor] = None,
        segmentation_class_labels: Optional[torch.Tensor] = None,
        segmentation_bbox_labels: Optional[torch.Tensor] = None,
        segmentation_object_mask: Optional[torch.Tensor] = None,
        segmentation_mask_labels: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        **kwargs,
    ) -> GLiNExTVisionOutput:
        classes_mapping = kwargs.get("classes_mapping")
        if pixel_values is None:
            raise ValueError("GLiNExTVisionModel requires pixel_values")
        if classes_mapping is None:
            raise ValueError("GLiNExTVisionModel requires classes_mapping to build vision task inputs")

        vision_tokens = self._encode_media_tokens(pixel_values, vision_attention_mask, **kwargs)
        vision_mask = self._mask_for_tokens(vision_tokens, vision_attention_mask)
        flat_inputs_map = self._build_media_flat_inputs(classes_mapping, vision_tokens, vision_mask, kwargs)

        shared = SharedRepresentations(
            token_embeds=vision_tokens,
            input_ids=None,
            attention_mask=vision_mask,
            words_embedding=vision_tokens,
            mask=vision_mask,
            prompts_embedding=None,
            prompts_embedding_mask=None,
            vision_embedding=vision_tokens,
            vision_mask=vision_mask,
            audio_embedding=None,
            audio_mask=None,
            image_sizes=image_sizes,
        )
        batch_kwargs = dict(
            image_classification_labels=image_classification_labels,
            object_detection_class_labels=object_detection_class_labels,
            object_detection_bbox_labels=object_detection_bbox_labels,
            object_detection_object_mask=object_detection_object_mask,
            segmentation_class_labels=segmentation_class_labels,
            segmentation_bbox_labels=segmentation_bbox_labels,
            segmentation_object_mask=segmentation_object_mask,
            segmentation_mask_labels=segmentation_mask_labels,
            threshold=threshold,
        )
        final_loss, head_outputs = self._forward_media_heads(
            shared,
            flat_inputs_map,
            batch_kwargs,
            kwargs,
            vision_tokens,
        )

        image_cls_out = head_outputs.get("image_classification", TaskHeadOutput())
        det_out = head_outputs.get("object_detection", TaskHeadOutput())
        seg_out = head_outputs.get("segmentation", TaskHeadOutput())
        return GLiNExTVisionOutput(
            loss=final_loss,
            batch_size=vision_tokens.shape[0],
            image_classification_logits=image_cls_out.logits,
            image_classification_batch_origin=flat_inputs_map["image_classification"].batch_origin if "image_classification" in flat_inputs_map else None,
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
            vision_embedding=vision_tokens,
            vision_mask=vision_mask,
            words_embedding=vision_tokens,
            mask=vision_mask,
        )


class GLiNExTAudioModel(_MediaOnlyBiEncoderModel):
    """Audio-only GLiNExT model using an audio/text-label bi-encoder."""

    enabled_task_names = TASK_REGISTRY.audio_tasks
    output_cls = GLiNExTAudioOutput
    media_task_names = TASK_REGISTRY.audio_tasks
    media_token_name = "audio"
    media_mask_name = "audio_attention_mask"
    bi_encoder_cls = AudioBiEncoder

    def _encode_media_tokens(self, media_values: torch.Tensor, media_mask: Optional[torch.Tensor], **kwargs):
        return self.token_rep_layer.audio_encoder(
            media_values,
            attention_mask=media_mask,
        )

    def forward(
        self,
        audio_values: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        audio_classification_labels: Optional[torch.Tensor] = None,
        audio_segmentation_class_labels: Optional[torch.Tensor] = None,
        audio_segmentation_segment_labels: Optional[torch.Tensor] = None,
        audio_segmentation_object_mask: Optional[torch.Tensor] = None,
        audio_segmentation_mask_labels: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        **kwargs,
    ) -> GLiNExTAudioOutput:
        classes_mapping = kwargs.get("classes_mapping")
        if audio_values is None:
            raise ValueError("GLiNExTAudioModel requires audio_values")
        if classes_mapping is None:
            raise ValueError("GLiNExTAudioModel requires classes_mapping to build audio task inputs")

        audio_tokens = self._encode_media_tokens(audio_values, audio_attention_mask, **kwargs)
        audio_mask = self._mask_for_tokens(audio_tokens, audio_attention_mask)
        flat_inputs_map = self._build_media_flat_inputs(classes_mapping, audio_tokens, audio_mask, kwargs)

        shared = SharedRepresentations(
            token_embeds=audio_tokens,
            input_ids=None,
            attention_mask=audio_mask,
            words_embedding=audio_tokens,
            mask=audio_mask,
            prompts_embedding=None,
            prompts_embedding_mask=None,
            vision_embedding=None,
            vision_mask=None,
            audio_embedding=audio_tokens,
            audio_mask=audio_mask,
            image_sizes=None,
        )
        batch_kwargs = dict(
            audio_classification_labels=audio_classification_labels,
            audio_segmentation_class_labels=audio_segmentation_class_labels,
            audio_segmentation_segment_labels=audio_segmentation_segment_labels,
            audio_segmentation_object_mask=audio_segmentation_object_mask,
            audio_segmentation_mask_labels=audio_segmentation_mask_labels,
            threshold=threshold,
        )
        final_loss, head_outputs = self._forward_media_heads(
            shared,
            flat_inputs_map,
            batch_kwargs,
            kwargs,
            audio_tokens,
        )

        audio_cls_out = head_outputs.get("audio_classification", TaskHeadOutput())
        audio_seg_out = head_outputs.get("audio_segmentation", TaskHeadOutput())
        return GLiNExTAudioOutput(
            loss=final_loss,
            batch_size=audio_tokens.shape[0],
            audio_classification_logits=audio_cls_out.logits,
            audio_classification_batch_origin=flat_inputs_map["audio_classification"].batch_origin if "audio_classification" in flat_inputs_map else None,
            audio_segmentation_logits=audio_seg_out.logits,
            audio_segmentation_batch_origin=flat_inputs_map["audio_segmentation"].batch_origin if "audio_segmentation" in flat_inputs_map else None,
            audio_segmentation_segments=audio_seg_out.extra.get("segment_preds"),
            audio_segmentation_objectness_logits=audio_seg_out.extra.get("objectness_logits"),
            audio_segmentation_anchor_mask=audio_seg_out.extra.get("anchor_mask"),
            audio_segmentation_mask_logits=audio_seg_out.extra.get("mask_logits"),
            audio_segmentation_prototypes=audio_seg_out.extra.get("prototypes"),
            audio_segmentation_coefficients=audio_seg_out.extra.get("coefficients"),
            audio_embedding=audio_tokens,
            audio_mask=audio_mask,
            words_embedding=audio_tokens,
            mask=audio_mask,
        )


class GLiNExTLayoutModel(_GLiNExTJointForwardModel):
    """Text + document-layout variant.

    Layout coordinates are supplied as ``bbox`` with shape
    ``(batch, sequence, 4)``. Optional ``pixel_values`` can also be supplied for
    layout backbones such as LayoutLMv3 that fuse text, boxes, and page images.
    """

    enabled_task_names = TASK_REGISTRY.text_tasks
    output_cls = GLiNExTLayoutOutput
    layout_encoder_cls = LayoutEncoder
    layout_bi_encoder_cls = LayoutBiEncoder
    unsupported_input_names = {
        "word_bboxes",
        "text_bbox",
        "text_word_bboxes",
        "text_pixel_values",
        "layout_bbox",
        "layout_pixel_values",
    }

    def _init_token_rep_layer(self, config, from_pretrained, cache_dir):
        if config.labels_encoder is not None:
            return self.layout_bi_encoder_cls(config, from_pretrained, cache_dir=cache_dir)
        return self.layout_encoder_cls(config, from_pretrained, cache_dir=cache_dir)

    @classmethod
    def _reject_unsupported_input_names(cls, kwargs: dict) -> None:
        unsupported = [key for key in kwargs if key in cls.unsupported_input_names]
        if unsupported:
            raise ValueError(
                "GLiNExTLayoutModel uses canonical input names only; "
                f"use bbox/pixel_values instead of: {', '.join(sorted(unsupported))}"
            )

    @classmethod
    def _layout_kwargs(cls, kwargs: dict) -> dict:
        cls._reject_unsupported_input_names(kwargs)
        allowed = {
            "packing_config",
            "pair_attention_mask",
            "token_type_ids",
            "position_ids",
            "head_mask",
            "output_attentions",
            "output_hidden_states",
            "return_dict",
            "bbox",
            "page_token_ids",
            "pixel_values",
            "vision_attention_mask",
            "image_batch_idx",
            "image_page_ids",
        }
        return {key: kwargs[key] for key in allowed if key in kwargs}

    def _append_layout_extra_tokens(
        self,
        token_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        words_embedding: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        extra_mask = getattr(self.token_rep_layer, "_last_layout_extra_mask", None)
        if extra_mask is None or token_embeds.shape[1] <= input_ids.shape[1]:
            return words_embedding, mask
        extra_tokens = token_embeds[:, input_ids.shape[1]:]
        extra_mask = extra_mask[:, :extra_tokens.shape[1]].to(device=mask.device, dtype=mask.dtype)
        if extra_tokens.shape[1] == 0:
            return words_embedding, mask
        return torch.cat([words_embedding, extra_tokens], dim=1), torch.cat([mask, extra_mask], dim=1)

    @staticmethod
    def _reject_unsupported_media_inputs(kwargs: dict) -> None:
        media_keys = (
            "audio_values",
            "input_values",
            "audio_attention_mask",
        )
        present = [key for key in media_keys if kwargs.get(key) is not None]
        if present:
            raise ValueError(
                "GLiNExTLayoutModel supports text/layout inputs only; "
                f"received unsupported media arguments: {', '.join(present)}"
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
        encoder_kwargs = self._layout_kwargs(kwargs)

        if _has_labels_encoder(self.token_rep_layer) and labels_input_ids is not None:
            token_embeds, labels_embeds = self.token_rep_layer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels_input_ids=labels_input_ids,
                labels_attention_mask=labels_attention_mask,
                **encoder_kwargs,
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
            words_embedding, mask = self._append_layout_extra_tokens(token_embeds, input_ids, words_embedding, mask)
            if hasattr(self, "rnn"):
                words_embedding = self.rnn(words_embedding, mask)
            return token_embeds, labels_embeds, labels_mask, words_embedding, mask

        token_embeds = self.token_rep_layer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **encoder_kwargs,
        )
        prompts_embedding, prompts_embedding_mask, words_embedding, mask = (
            extract_prompt_features_and_word_embeddings(
                self.config.class_token_index, token_embeds, input_ids, attention_mask,
                text_lengths, words_mask, self.config.embed_ent_token,
            )
        )
        words_embedding, mask = self._append_layout_extra_tokens(token_embeds, input_ids, words_embedding, mask)
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
            self.token_rep_layer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **self._layout_kwargs(kwargs),
            )
        )

    def forward(self, *args, **kwargs) -> GLiNExTLayoutOutput:
        self._reject_unsupported_media_inputs(kwargs)
        self._reject_unsupported_input_names(kwargs)
        output = self._forward_text_task_heads(*args, **kwargs)
        if type(output) is GLiNExTLayoutOutput:
            return output
        return _filtered_output(
            GLiNExTLayoutOutput,
            **{field.name: getattr(output, field.name, None) for field in fields(GLiNExTLayoutOutput)},
        )


class GLiNExTOmniModel(_GLiNExTJointForwardModel):
    """Text + vision + audio model with early fusion inside the text transformer."""

    output_cls = GLiNExTOmniOutput
    omni_encoder_cls = TriOmniEncoder
    omni_bi_encoder_cls = TriOmniBiEncoder
    unsupported_input_names = {
        "input_values",
        "word_bboxes",
        "text_bbox",
        "text_word_bboxes",
        "text_pixel_values",
        "layout_bbox",
        "layout_pixel_values",
        "vision_pixel_values",
        "audio_input_values",
    }

    @staticmethod
    def _infer_omni_modalities(config) -> Tuple[str, ...]:
        if getattr(config, "omni_modalities", None) is not None:
            return tuple(config.omni_modalities)

        variant = _normalize_model_variant(getattr(config, "model_variant", None))
        try:
            return _VARIANT_MODALITIES[variant]
        except KeyError as exc:
            raise ValueError(
                "GLiNExTOmniModel requires model_variant='omni'; "
                f"got {variant!r}"
            ) from exc

    def _init_token_rep_layer(self, config, from_pretrained, cache_dir):
        if getattr(config, "omni_modalities", None) is None:
            config.omni_modalities = self._infer_omni_modalities(config)
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

    @classmethod
    def _reject_unsupported_input_names(cls, kwargs: dict) -> None:
        unsupported = [key for key in kwargs if key in cls.unsupported_input_names]
        if unsupported:
            raise ValueError(
                "GLiNExTOmniModel uses canonical input names only; "
                f"use input_ids/pixel_values/bbox/audio_values instead of: {', '.join(sorted(unsupported))}"
            )

    @classmethod
    def _omni_kwargs(cls, kwargs: dict) -> dict:
        allowed = {
            "packing_config", "pair_attention_mask", "pixel_values",
            "vision_attention_mask", "audio_values", "audio_attention_mask",
            "vision_input_mask", "audio_input_mask", "bbox",
        }
        cls._reject_unsupported_input_names(kwargs)
        return {key: kwargs[key] for key in allowed if key in kwargs}

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
            if getattr(output, "vision_embeddings", None) is not None:
                _cache_forward_modality(self, "vision", output.vision_embeddings, output.vision_attention_mask)
            if getattr(output, "audio_embeddings", None) is not None:
                _cache_forward_modality(self, "audio", output.audio_embeddings, output.audio_attention_mask)
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
        if getattr(output, "vision_embeddings", None) is not None:
            _cache_forward_modality(self, "vision", output.vision_embeddings, output.vision_attention_mask)
        if getattr(output, "audio_embeddings", None) is not None:
            _cache_forward_modality(self, "audio", output.audio_embeddings, output.audio_attention_mask)
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

    def forward(self, *args, **kwargs) -> GLiNExTOmniOutput:
        self._reject_unsupported_input_names(kwargs)
        return self._forward_omni_task_heads(*args, **kwargs)


def resolve_glinext_model_class(config) -> type[BaseGLiNextModel]:
    variant = _normalize_model_variant(getattr(config, "model_variant", None))

    if variant == "text":
        return GLiNExTTextModel
    if variant == "omni":
        return GLiNExTOmniModel
    if variant == "vision":
        return GLiNExTVisionModel
    if variant == "layout":
        return GLiNExTLayoutModel
    if variant == "audio":
        return GLiNExTAudioModel
    raise ValueError(f"Unknown GLiNExT model_variant: {getattr(config, 'model_variant', None)!r}")


class GLiNExTModel(GLiNExTTextModel):
    """Backward-compatible text model class.

    The user-facing :class:`glinext.glinext.GLiNExT` class selects concrete
    multimodal variants via :func:`resolve_glinext_model_class`.
    """
