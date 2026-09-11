"""GLiFormer: unified multi-task model — thin orchestrator over modular task heads."""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from gliner.modeling.base import BaseModel
from gliner.modeling.layers import CrossFuser, LstmSeq2SeqEncoder
from gliner.modeling.utils import (
    extract_prompt_features,
    extract_prompt_features_and_word_embeddings,
    extract_word_embeddings,
)
from torch import nn

from .config import GLiFormerConfig
from .encoders.audio import AudioBiEncoder, audio_token_mask
from .encoders.media import apply_input_mask
from .encoders.omni import (
    LayoutBiEncoder,
    LayoutEncoder,
    OmniEncoderOutput,
    TriOmniBiEncoder,
    TriOmniEncoder,
)
from .encoders.text import TextBiEncoder, TextEncoder
from .encoders.vision import VisionBiEncoder, vision_encoder_kwargs, vision_token_mask
from .layers import AnchorCrossAttentionLayer, AnchorModeling
from .outputs import (
    GLiFormerAudioOutput,
    GLiFormerLayoutOutput,
    GLiFormerOmniOutput,
    GLiFormerOutput,
    GLiFormerTextOutput,
    GLiFormerVisionOutput,
)
from .tasks import TASK_REGISTRY, SharedRepresentations, TaskFlatInputs, TaskHeadOutput
from .tasks.losses import binary_focal_or_bce


def _normalize_model_variant(value: Optional[str]) -> str:
    return value or "text"


_VARIANT_MODALITIES = {
    "omni": ("text", "vision", "audio"),
}

_SET_PREDICTION_COUNT_KEYS = {
    "object_detection": "object_detection_count",
    "segmentation": "segmentation_count",
    "audio_segmentation": "audio_segmentation_count",
}


@dataclass(frozen=True)
class _FlatGroupLayout:
    """Indexes one task's groups in flattened BN order."""

    batch_origin: torch.Tensor
    group_index: torch.Tensor
    starts: torch.Tensor
    sizes: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.batch_origin.numel())


@dataclass(frozen=True)
class _FlatEmbeddingBatch:
    """Padded embeddings and validity mask aligned to a flat group layout."""

    embeddings: torch.Tensor
    mask: torch.Tensor


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


def _cache_forward_modality(
    module: nn.Module,
    name: str,
    tokens,
    mask,
    spatial_shape=None,
    prefix_tokens=None,
) -> None:
    cache = getattr(module, "_forward_modality_cache", None)
    if cache is not None and tokens is not None:
        if spatial_shape is None and prefix_tokens is None:
            cache[name] = (tokens, mask)
        else:
            cache[name] = (tokens, mask, spatial_shape, prefix_tokens)


class BaseGLiFormerModel(BaseModel):
    """Unified multi-task model composing optional task heads.

    Each task is a standalone TaskHead subclass. The model registers enabled heads
    via nn.ModuleDict and executes them in dependency order during forward().
    """

    enabled_task_names: Optional[Tuple[str, ...]] = None

    def __init__(
        self,
        config: GLiFormerConfig,
        from_pretrained: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        local_files_only: bool = False,
    ):
        super().__init__(config, from_pretrained, cache_dir)

        self.token_rep_layer = self._init_token_rep_layer(
            config, from_pretrained, cache_dir, local_files_only=local_files_only,
        )

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

        shared_refinement_spec = getattr(
            config,
            "shared_anchor_refinement",
            None,
        )
        if shared_refinement_spec is None and config.shared_anchor_refine_layers > 0:
            shared_refinement_spec = {
                "type": "cross_attention",
                "params": {
                    "num_heads": config.shared_anchor_refine_heads,
                    "num_layers": config.shared_anchor_refine_layers,
                    "dropout": config.dropout,
                },
            }
        if shared_refinement_spec is not None:
            shared_refinement = AnchorCrossAttentionLayer.from_config(
                shared_refinement_spec,
                config.hidden_size,
                dropout=config.dropout,
            )
            if shared_refinement is not None:
                self.shared_anchor_refine = shared_refinement
                shared_layers["anchor_refine"] = self.shared_anchor_refine

        self.heads = nn.ModuleDict()
        enabled_task_names = getattr(self, "enabled_task_names", None)
        enabled_task_names = set(enabled_task_names) if enabled_task_names is not None else None
        for task_definition in TASK_REGISTRY:
            HeadClass = task_definition.load_head_class()
            head_kwargs = {
                "from_pretrained": from_pretrained,
                "cache_dir": cache_dir,
                "shared_layers": shared_layers,
            }
            if HeadClass.__name__ == "SegmentationHead":
                head_kwargs["detection_head"] = (
                    self.heads["object_detection"]
                    if "object_detection" in self.heads
                    else None
                )
            if HeadClass.__name__ in {
                "JointRelexHead",
                "StructuringHead",
            }:
                head_kwargs["ner_head"] = (
                    self.heads["ner"]
                    if "ner" in self.heads
                    else None
                )
            head = HeadClass.from_config(config, **head_kwargs)
            if head is not None:
                canonical_name = task_definition.name
                if enabled_task_names is not None and canonical_name not in enabled_task_names:
                    continue
                self.heads[canonical_name] = head

    def _migrate_renamed_task_state_dict(
        self,
        state_dict,
        prefix: str,
        legacy_name: str,
        canonical_name: str,
    ) -> None:
        """Move shape-compatible weights from a retired task prefix."""

        if canonical_name not in self.heads:
            return

        target_state = self.heads[canonical_name].state_dict()
        legacy_prefix = f"{prefix}heads.{legacy_name}."
        target_prefix = f"{prefix}heads.{canonical_name}."
        for legacy_key in tuple(state_dict):
            if not legacy_key.startswith(legacy_prefix):
                continue
            suffix = legacy_key[len(legacy_prefix):]
            target_value = target_state.get(suffix)
            if target_value is None:
                continue
            source_value = state_dict[legacy_key]
            try:
                shapes_match = tuple(source_value.shape) == tuple(
                    target_value.shape
                )
            except (AttributeError, RuntimeError):
                shapes_match = False
            if not shapes_match:
                continue

            target_key = f"{target_prefix}{suffix}"
            if target_key not in state_dict:
                state_dict[target_key] = source_value
            state_dict.pop(legacy_key)

    def _migrate_legacy_joint_relex_ner_state_dict(
        self,
        state_dict,
        prefix: str,
    ) -> None:
        """Remove the NER copy stored by pre-reuse Joint Relex checkpoints."""
        if "ner" not in self.heads or "joint_relex" not in self.heads:
            return
        joint_head = self.heads["joint_relex"]
        if getattr(joint_head, "_owns_ner_head", True):
            return

        ner_state = self.heads["ner"].state_dict()
        joint_state = joint_head.state_dict()
        legacy_prefix = f"{prefix}heads.joint_relex."
        ner_prefix = f"{prefix}heads.ner."
        for suffix, target_value in ner_state.items():
            # A suffix that remains in the composed Joint head is relation
            # state, not an obsolete inherited NER tensor.
            if suffix in joint_state:
                continue
            legacy_key = f"{legacy_prefix}{suffix}"
            if legacy_key not in state_dict:
                continue
            source_value = state_dict[legacy_key]
            target_key = f"{ner_prefix}{suffix}"
            if target_key not in state_dict:
                try:
                    shapes_match = tuple(source_value.shape) == tuple(
                        target_value.shape
                    )
                except (AttributeError, RuntimeError):
                    shapes_match = False
                if shapes_match:
                    state_dict[target_key] = source_value
            state_dict.pop(legacy_key)

    def _migrate_reused_structuring_ner_state_dict(
        self,
        state_dict,
        prefix: str,
    ) -> None:
        """Remove the private NER copy from pre-reuse structuring state."""

        if "ner" not in self.heads or "structuring" not in self.heads:
            return
        structuring_head = self.heads["structuring"]
        if getattr(structuring_head, "_owns_ner_head", True):
            return

        ner_state = self.heads["ner"].state_dict()
        structuring_state = structuring_head.state_dict()
        legacy_prefixes = [
            f"{prefix}heads.structuring.",
            f"{prefix}heads.set_structuring.",
        ]
        ner_prefix = f"{prefix}heads.ner."

        for suffix, target_value in ner_state.items():
            # Any same-named stage-2 tensor remains owned by Structuring.
            if suffix in structuring_state:
                continue
            for legacy_prefix in legacy_prefixes:
                legacy_key = f"{legacy_prefix}{suffix}"
                if legacy_key not in state_dict:
                    continue
                source_value = state_dict[legacy_key]
                target_key = f"{ner_prefix}{suffix}"
                if target_key not in state_dict:
                    try:
                        shapes_match = tuple(source_value.shape) == tuple(
                            target_value.shape
                        )
                    except (AttributeError, RuntimeError):
                        shapes_match = False
                    if shapes_match:
                        state_dict[target_key] = source_value
                # Canonical standalone NER state wins when both copies exist.
                state_dict.pop(legacy_key)

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
        self._migrate_renamed_task_state_dict(
            state_dict,
            prefix,
            "set_open_relex",
            "open_relex",
        )
        self._migrate_renamed_task_state_dict(
            state_dict,
            prefix,
            "set_structuring",
            "structuring",
        )
        self._migrate_legacy_joint_relex_ner_state_dict(state_dict, prefix)
        self._migrate_reused_structuring_ner_state_dict(
            state_dict,
            prefix,
        )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _init_token_rep_layer(self, config, from_pretrained, cache_dir, local_files_only=False):
        if config.labels_encoder is not None:
            return TextBiEncoder(
                config, from_pretrained, cache_dir=cache_dir, local_files_only=local_files_only,
            )
        return TextEncoder(
            config, from_pretrained, cache_dir=cache_dir, local_files_only=local_files_only,
        )

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
        else:
            predicted = count_logits.squeeze(-1).round().long()

        predicted = predicted.clamp(min=0)
        if predicted.shape[0] < struct_bn:
            return None

        return predicted[-struct_bn:]

    def _make_task_loss_fn(self, task_name: str, runtime_kwargs: dict):
        task_cfg = self.config.get_task_config(task_name)
        loss_defaults = {}
        for name in (
            "focal_loss_alpha",
            "focal_loss_gamma",
            "focal_loss_prob_margin",
        ):
            value = getattr(task_cfg, name, None) if task_cfg is not None else None
            if value is None:
                value = runtime_kwargs.get(name)
            if value is not None:
                loss_defaults[name] = value
        # Focal is the project-wide binary-loss default.  BCE is an explicit
        # opt-out made by setting both alpha and gamma to non-positive values.
        loss_defaults.setdefault("focal_loss_alpha", 0.25)
        loss_defaults.setdefault("focal_loss_gamma", 2.0)
        for name in ("label_smoothing", "negatives", "masking"):
            # Structuring heads own their negative-sampling policy because
            # their unmatched anchor cells are part of the set objective, not
            # expendable background examples.  In particular, applying the
            # trainer's very small NER keep-rate to objectness and Hungarian
            # matching makes the assignment stochastic and leaves almost all
            # unused anchors unsupervised.
            value = (
                getattr(task_cfg, name, None)
                if task_cfg is not None
                else None
            )
            if value is None:
                value = runtime_kwargs.get(name)
            if value is not None:
                loss_defaults[name] = value

        def task_loss_fn(logits, labels, **call_kwargs):
            merged_kwargs = dict(loss_defaults)
            merged_kwargs.update(
                {key: value for key, value in call_kwargs.items() if value is not None}
            )
            # Task heads own their reductions and structural masks. Keep this
            # shared primitive elementwise even when the trainer is configured
            # with a global reduction for legacy text heads.
            merged_kwargs["reduction"] = "none"
            losses = binary_focal_or_bce(logits, labels, **merged_kwargs)

            negatives = float(merged_kwargs.get("negatives", 1.0))
            masking = merged_kwargs.get("masking", "none")
            if negatives >= 1.0 or masking in {None, False, "none"}:
                return losses
            if masking == "global":
                keep = torch.where(
                    labels == 0,
                    torch.rand_like(logits) < negatives,
                    torch.ones_like(logits, dtype=torch.bool),
                )
                return losses * keep
            if masking in {"label", "span"}:
                dimension = 1 if masking == "label" else 2
                negative_groups = labels.sum(dim=dimension, keepdim=True) == 0
                negative_groups = negative_groups.expand_as(labels)
                keep = torch.where(
                    negative_groups,
                    torch.rand_like(logits) < negatives,
                    torch.ones_like(logits, dtype=torch.bool),
                )
                return losses * keep
            return losses

        return task_loss_fn

    @staticmethod
    def _flatten_set_prediction_count(
        count,
        flat_inputs: TaskFlatInputs,
        task_name: str,
    ) -> Optional[torch.Tensor]:
        """Map public per-item query counts to flattened task-label groups."""

        if count is None:
            return None
        batch_origin = flat_inputs.batch_origin.long()
        count = torch.as_tensor(count, device=batch_origin.device)
        if count.ndim == 0:
            count = count.expand(
                int(batch_origin.max().item()) + 1 if batch_origin.numel() else 0
            )
        elif count.ndim == 2 and count.shape[1] == 1:
            count = count[:, 0]
        elif count.ndim != 1:
            raise ValueError(
                f"{task_name}_count must be a scalar or one value per input item"
            )
        if count.is_floating_point():
            if not torch.equal(count, count.round()):
                raise ValueError(f"{task_name}_count values must be integers")
            count = count.round()
        count = count.long()
        if (count < 0).any():
            raise ValueError(f"{task_name}_count values must be non-negative")
        if not batch_origin.numel():
            return count.new_empty(0)
        required_items = int(batch_origin.max().item()) + 1
        if count.numel() < required_items:
            raise ValueError(
                f"{task_name}_count has {count.numel()} values, but flattened "
                f"groups reference {required_items} input items"
            )
        return count.index_select(0, batch_origin)

    def _encode_label_inputs_batched(
        self,
        label_inputs: Dict[
            str,
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
        ],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Encode named label-token batches in one padded encoder pass."""

        encoded = {name: None for name in label_inputs}
        if not _has_labels_encoder(self.token_rep_layer):
            return encoded

        active_names = []
        input_ids_parts = []
        attention_mask_parts = []
        sizes = []
        for name, (input_ids, attention_mask) in label_inputs.items():
            if input_ids is None and attention_mask is None:
                continue
            if input_ids is None or attention_mask is None:
                raise ValueError(
                    f"{name} label input IDs and attention mask must be provided together"
                )
            if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
                raise ValueError(
                    f"{name} label input IDs and attention mask must have the same 2D shape"
                )
            if input_ids.shape[0] == 0:
                continue
            active_names.append(name)
            input_ids_parts.append(input_ids)
            attention_mask_parts.append(attention_mask)
            sizes.append(input_ids.shape[0])

        if not active_names:
            return encoded

        max_length = max(input_ids.shape[1] for input_ids in input_ids_parts)
        for index, (input_ids, attention_mask) in enumerate(
            zip(input_ids_parts, attention_mask_parts)
        ):
            padding = max_length - input_ids.shape[1]
            if padding > 0:
                input_ids_parts[index] = torch.nn.functional.pad(
                    input_ids,
                    (0, padding),
                    value=0,
                )
                attention_mask_parts[index] = torch.nn.functional.pad(
                    attention_mask,
                    (0, padding),
                    value=0,
                )

        all_embeddings = self.token_rep_layer.encode_labels(
            torch.cat(input_ids_parts, dim=0),
            torch.cat(attention_mask_parts, dim=0),
        )
        for name, embeddings in zip(
            active_names,
            all_embeddings.split(sizes, dim=0),
        ):
            encoded[name] = embeddings
        return encoded

    def _encode_all_labels_batched(
        self,
        cat_labels_input_ids: Optional[torch.Tensor] = None,
        cat_labels_attention_mask: Optional[torch.Tensor] = None,
        rel_labels_input_ids: Optional[torch.Tensor] = None,
        rel_labels_attention_mask: Optional[torch.Tensor] = None,
        child_labels_input_ids: Optional[torch.Tensor] = None,
        child_labels_attention_mask: Optional[torch.Tensor] = None,
        open_rel_labels_input_ids: Optional[torch.Tensor] = None,
        open_rel_labels_attention_mask: Optional[torch.Tensor] = None,
    ):
        """Batch text-task label inputs into one BiEncoder pass, then split results."""
        embeddings = self._encode_label_inputs_batched(
            {
                "classification": (
                    cat_labels_input_ids,
                    cat_labels_attention_mask,
                ),
                "joint_relex": (
                    rel_labels_input_ids,
                    rel_labels_attention_mask,
                ),
                "structuring": (
                    child_labels_input_ids,
                    child_labels_attention_mask,
                ),
                "open_relex": (
                    open_rel_labels_input_ids,
                    open_rel_labels_attention_mask,
                ),
            }
        )
        return (
            embeddings["classification"],
            embeddings["joint_relex"],
            embeddings["structuring"],
            embeddings["open_relex"],
        )

    def _encode_media_labels_batched(self, kwargs: dict) -> Dict[str, torch.Tensor]:
        """Encode media-task label inputs as flat per-label embeddings."""
        embeddings = self._encode_label_inputs_batched(
            {
                task_name: (
                    kwargs.get(f"{task_name}_labels_input_ids"),
                    kwargs.get(f"{task_name}_labels_attention_mask"),
                )
                for task_name in self._media_task_names
                if task_name in self.heads
            }
        )
        return {
            task_name: task_embeddings
            for task_name, task_embeddings in embeddings.items()
            if task_embeddings is not None
        }

    def _vision_feature_encoder(self):
        encoder = getattr(self, "vision_encoder", None)
        if encoder is None:
            feature_encoders = getattr(self.token_rep_layer, "feature_encoders", None)
            if feature_encoders is not None and "vision" in feature_encoders:
                encoder = feature_encoders["vision"]
        return encoder

    def _encode_vision_features(
        self,
        pixel_values: Optional[torch.Tensor],
        vision_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        encoder = self._vision_feature_encoder()
        cache = getattr(self, "_forward_modality_cache", None)
        if cache is not None and "vision" in cache:
            cached = cache["vision"]
            vision_tokens, vision_mask = cached[:2]
            if len(cached) >= 4:
                spatial_shape, prefix_tokens = cached[2:4]
            else:
                spatial_shape = prefix_tokens = None
            if (
                spatial_shape is None
                and encoder is not None
                and pixel_values is not None
                and hasattr(encoder, "_infer_spatial_metadata")
            ):
                spatial_shape, prefix_tokens = encoder._infer_spatial_metadata(
                    pixel_values,
                    vision_tokens,
                    explicit_shape=None,
                )
            return vision_tokens, vision_mask, spatial_shape, prefix_tokens
        if pixel_values is None:
            return None, None, None, None
        if encoder is None:
            return None, None, None, None
        if hasattr(encoder, "forward_features"):
            vision_features = encoder.forward_features(
                pixel_values,
                **vision_encoder_kwargs(kwargs),
            )
            vision_tokens = vision_features.token_embeddings
            spatial_shape = vision_features.spatial_shape
            prefix_tokens = vision_features.prefix_tokens
        else:
            vision_tokens = encoder(pixel_values)
            spatial_shape = prefix_tokens = None
        vision_mask = vision_token_mask(
            vision_tokens,
            vision_attention_mask,
            prefix_tokens,
            spatial_shape,
        )
        vision_mask = apply_input_mask(vision_mask, kwargs.get("vision_input_mask"))
        return vision_tokens, vision_mask, spatial_shape, prefix_tokens

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
        audio_mask = audio_token_mask(
            encoder,
            audio_tokens,
            audio_attention_mask,
        )
        audio_mask = apply_input_mask(audio_mask, kwargs.get("audio_input_mask"))
        return audio_tokens, audio_mask

    @staticmethod
    def _build_group_layout(
        classes_mapping,
        task_name: str,
        *,
        device,
        label_kind: str = "primary",
        explicit_sizes: Optional[torch.Tensor] = None,
    ) -> _FlatGroupLayout:
        """Describe task groups, their source slices, and original batch rows."""

        groups = list(classes_mapping.flat_iter(task_name))
        batch_origin = torch.tensor(
            [batch_idx for _, batch_idx, _, _ in groups],
            dtype=torch.long,
            device=device,
        )
        group_index = torch.tensor(
            [group_idx for _, _, group_idx, _ in groups],
            dtype=torch.long,
            device=device,
        )

        if explicit_sizes is not None:
            sizes = explicit_sizes.to(device=device, dtype=torch.long).flatten()
            if sizes.numel() != len(groups):
                raise ValueError(
                    f"{task_name} has {len(groups)} flat groups but received "
                    f"{sizes.numel()} label-group sizes"
                )
            if bool((sizes < 0).any()):
                raise ValueError("label-group sizes must be non-negative")
            starts = torch.cumsum(sizes, dim=0) - sizes
        else:
            offsets = {}
            starts_list = []
            sizes_list = []
            for _, batch_idx, group_idx, _ in groups:
                size = classes_mapping.label_size(
                    task_name,
                    batch_idx,
                    group_idx,
                    label_kind=label_kind,
                )
                start = offsets.get(batch_idx, 0)
                starts_list.append(start)
                sizes_list.append(size)
                offsets[batch_idx] = start + size
            starts = torch.tensor(starts_list, dtype=torch.long, device=device)
            sizes = torch.tensor(sizes_list, dtype=torch.long, device=device)

        return _FlatGroupLayout(
            batch_origin=batch_origin,
            group_index=group_index,
            starts=starts,
            sizes=sizes,
        )

    @staticmethod
    def _flatten_grouped_embeddings(
        embeddings: torch.Tensor,
        layout: _FlatGroupLayout,
        source_mask: Optional[torch.Tensor] = None,
        *,
        mask_dtype: torch.dtype = torch.float,
    ) -> _FlatEmbeddingBatch:
        """Slice packed 2D or batch-packed 3D embeddings into padded groups."""

        if embeddings.dim() not in {2, 3}:
            raise ValueError(
                "grouped embeddings must have shape (labels, D) or (B, labels, D)"
            )
        if source_mask is not None:
            expected_mask_shape = (
                embeddings.shape[:1]
                if embeddings.dim() == 2
                else embeddings.shape[:2]
            )
            if source_mask.shape != expected_mask_shape:
                raise ValueError(
                    "grouped embedding mask must match the embedding label axes"
                )

        max_size = int(layout.sizes.max().item()) if layout.count else 0
        flat_embeddings = embeddings.new_zeros(
            layout.count,
            max_size,
            embeddings.shape[-1],
        )
        flat_mask = torch.zeros(
            layout.count,
            max_size,
            device=embeddings.device,
            dtype=mask_dtype,
        )
        descriptors = zip(
            layout.batch_origin.tolist(),
            layout.starts.tolist(),
            layout.sizes.tolist(),
        )
        for flat_idx, (batch_idx, start, size) in enumerate(descriptors):
            if size <= 0:
                continue
            end = start + size
            if embeddings.dim() == 2:
                if end > embeddings.shape[0]:
                    continue
                flat_embeddings[flat_idx, :size] = embeddings[start:end]
            else:
                if batch_idx >= embeddings.shape[0] or end > embeddings.shape[1]:
                    continue
                flat_embeddings[flat_idx, :size] = embeddings[
                    batch_idx,
                    start:end,
                ]

            if source_mask is None:
                flat_mask[flat_idx, :size] = 1
            elif embeddings.dim() == 2:
                flat_mask[flat_idx, :size] = source_mask[start:end].to(
                    device=flat_mask.device,
                    dtype=mask_dtype,
                )
            else:
                flat_mask[flat_idx, :size] = source_mask[
                    batch_idx,
                    start:end,
                ].to(device=flat_mask.device, dtype=mask_dtype)

        return _FlatEmbeddingBatch(flat_embeddings, flat_mask)

    @staticmethod
    def _slice_prompt_features(
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        starts: List[int],
        sizes: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select one task's contiguous prompt features for every batch row."""

        if embeddings.dim() != 3 or mask.shape != embeddings.shape[:2]:
            raise ValueError(
                "prompt embeddings and mask must have shapes (B, C, D) and (B, C)"
            )
        if len(starts) != embeddings.shape[0] or len(sizes) != embeddings.shape[0]:
            raise ValueError("prompt slice bounds must have one entry per batch row")

        max_size = max(sizes, default=0)
        selected = embeddings.new_zeros(
            embeddings.shape[0],
            max_size,
            embeddings.shape[-1],
        )
        selected_mask = mask.new_zeros(embeddings.shape[0], max_size)
        source_width = embeddings.shape[1]
        for batch_idx, (raw_start, raw_size) in enumerate(zip(starts, sizes)):
            start = max(int(raw_start), 0)
            size = max(int(raw_size), 0)
            copy_size = min(size, max(source_width - start, 0))
            if copy_size == 0:
                continue
            end = start + copy_size
            selected[batch_idx, :copy_size] = embeddings[batch_idx, start:end]
            selected_mask[batch_idx, :copy_size] = mask[batch_idx, start:end]
        return selected, selected_mask

    @staticmethod
    def _relation_prompt_counts(classes_mapping, batch_size: int) -> Dict[str, List[int]]:
        """Count relation marker features contributed by each text task."""

        counts = {
            "joint_relex": [0 for _ in range(batch_size)],
            "open_relex": [0 for _ in range(batch_size)],
        }
        extraction_iter = getattr(classes_mapping, "flat_extraction_iter", None)
        if extraction_iter is not None:
            for _, batch_idx, _, item_mapping in extraction_iter():
                relation_mapping = getattr(item_mapping, "rel_class_to_id", None)
                class_to_id = getattr(relation_mapping, "class_to_id", None)
                if class_to_id and batch_idx < batch_size:
                    counts["joint_relex"][batch_idx] += len(class_to_id)

        for task_name, iterator_name in (
            ("open_relex", "flat_open_relex_iter"),
        ):
            flat_iter = getattr(classes_mapping, iterator_name, None)
            if flat_iter is None:
                continue
            for _, batch_idx, _, item_mapping in flat_iter():
                relation_mapping = getattr(item_mapping, "rel_class_to_id", None)
                class_to_id = getattr(relation_mapping, "class_to_id", None)
                if class_to_id and batch_idx < batch_size:
                    counts[task_name][batch_idx] += len(class_to_id)
        return counts

    @classmethod
    def _slice_relation_prompt_features(
        cls,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        classes_mapping,
        task_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Isolate relation prompts for one task from the shared marker stream."""

        task_order = ("joint_relex", "open_relex")
        if task_name not in task_order:
            raise ValueError(f"Unknown relation prompt task: {task_name!r}")
        counts = cls._relation_prompt_counts(
            classes_mapping,
            embeddings.shape[0],
        )
        task_position = task_order.index(task_name)
        starts = [
            sum(counts[name][batch_idx] for name in task_order[:task_position])
            for batch_idx in range(embeddings.shape[0])
        ]
        return cls._slice_prompt_features(
            embeddings,
            mask,
            starts,
            counts[task_name],
        )

    def _slice_structuring_prompt_features(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        classes_mapping,
        task_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Isolate structuring field markers from the prompt stream."""

        task_order = ("structuring",)
        if task_name not in task_order:
            raise ValueError(f"Unknown structuring prompt task: {task_name!r}")
        batch_size = embeddings.shape[0]
        counts = {
            name: [0 for _ in range(batch_size)]
            for name in task_order
        }
        for name in task_order:
            if name not in self.heads:
                continue
            flat_iter = getattr(classes_mapping, f"flat_{name}_iter", None)
            if flat_iter is None:
                continue
            for _, batch_idx, _, item_mapping in flat_iter():
                if batch_idx >= batch_size:
                    continue
                field_mapping = getattr(
                    item_mapping, "field_class_to_id", None
                )
                class_to_id = getattr(field_mapping, "class_to_id", None)
                if class_to_id:
                    counts[name][batch_idx] += len(class_to_id)

        task_position = task_order.index(task_name)
        starts = [
            sum(
                counts[name][batch_idx]
                for name in task_order[:task_position]
            )
            for batch_idx in range(batch_size)
        ]
        return self._slice_prompt_features(
            embeddings,
            mask,
            starts,
            counts[task_name],
        )

    @staticmethod
    def _gather_flat_parents(
        parent_embeddings: torch.Tensor,
        layout: _FlatGroupLayout,
        classes_mapping,
        task_name: str,
        *,
        per_task_parents: bool,
    ) -> torch.Tensor:
        """Gather one parent embedding for each flattened task group."""

        flat_parents = parent_embeddings.new_zeros(
            layout.count,
            parent_embeddings.shape[-1],
        )
        for flat_idx, (batch_idx, group_idx) in enumerate(
            zip(layout.batch_origin.tolist(), layout.group_index.tolist())
        ):
            parent_idx = group_idx
            if not per_task_parents:
                parent_idx += classes_mapping.parent_offset_for_item(
                    task_name,
                    batch_idx,
                )
            if (
                batch_idx < parent_embeddings.shape[0]
                and parent_idx < parent_embeddings.shape[1]
            ):
                flat_parents[flat_idx] = parent_embeddings[batch_idx, parent_idx]
        return flat_parents

    def _build_flat_rel_prompts(
        self,
        rel_prompts: torch.Tensor,
        classes_mapping,
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

        layout = self._build_group_layout(
            classes_mapping,
            "joint_relex",
            device=rel_prompts.device,
            label_kind="relation",
            explicit_sizes=label_group_sizes,
        )
        if layout.count == 0:
            return None, None
        flattened = self._flatten_grouped_embeddings(
            rel_prompts,
            layout,
            rel_prompts_mask,
            mask_dtype=(
                rel_prompts_mask.dtype
                if rel_prompts_mask is not None
                else torch.long
            ),
        )
        return flattened.embeddings, flattened.mask

    def _build_flat_inputs(
        self,
        parent_embeds: torch.Tensor,
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
        layout = self._build_group_layout(
            classes_mapping,
            task_name,
            device=device,
            explicit_sizes=label_group_sizes,
        )
        if layout.count == 0:
            return None

        flat_words = words_embedding[layout.batch_origin]
        flat_word_mask = word_mask[layout.batch_origin]
        flat_parent = self._gather_flat_parents(
            parent_embeds,
            layout,
            classes_mapping,
            task_name,
            per_task_parents=per_task_parents,
        )
        flat_children = self._flatten_grouped_embeddings(
            child_embeds,
            layout,
            child_mask,
            mask_dtype=torch.float,
        )

        return TaskFlatInputs(
            words_embedding=flat_words,
            mask=flat_word_mask,
            parent_embedding=flat_parent,
            child_embedding=flat_children.embeddings,
            child_mask=flat_children.mask,
            batch_origin=layout.batch_origin,
            feature_embedding=flat_words,
            feature_mask=flat_word_mask,
        )

    def _apply_word_rnn(self, words_embedding, mask):
        """Run the packed RNN only for rows containing at least one word."""

        if not hasattr(self, "rnn"):
            return words_embedding
        nonempty_rows = mask.bool().any(dim=1)
        if not nonempty_rows.any():
            return words_embedding
        if nonempty_rows.all():
            return self.rnn(words_embedding, mask)
        row_indices = torch.where(nonempty_rows)[0]
        encoded = self.rnn(
            words_embedding[row_indices],
            mask[row_indices],
        )
        output = encoded.new_zeros(
            words_embedding.shape[0],
            words_embedding.shape[1],
            encoded.shape[-1],
        )
        return output.index_copy(0, row_indices, encoded)

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
            words_embedding = self._apply_word_rnn(words_embedding, mask)
            return token_embeds, labels_embeds, labels_mask, words_embedding, mask
        else:
            token_embeds = self.token_rep_layer(input_ids, attention_mask, **encoder_kwargs)
            prompts_embedding, prompts_embedding_mask, words_embedding, mask = (
                extract_prompt_features_and_word_embeddings(
                    self.config.class_token_index, token_embeds, input_ids, attention_mask,
                    text_lengths, words_mask, self.config.embed_ent_token,
                )
            )
            words_embedding = self._apply_word_rnn(words_embedding, mask)
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


class _GLiFormerJointForwardModel(BaseGLiFormerModel):
    """Shared text/joint task orchestration for concrete GLiFormer models."""

    output_cls = GLiFormerOutput
    _media_task_names = (
        "image_classification", "object_detection", "segmentation",
        "audio_classification", "audio_segmentation",
    )
    _flat_input_task_names = (
        "ner", "joint_relex", "classification", "structuring",
        "open_relex",
        "image_classification", "object_detection", "segmentation",
        "audio_classification", "audio_segmentation",
    )
    _flat_required_task_names = _flat_input_task_names + ("count",)
    _focal_loss_task_names = (
        "ner", "classification", "joint_relex", "open_relex",
        "structuring",
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
            "layout_input_mask",
            "page_token_ids",
            "page_input_mask",
            "pixel_values",
            "audio_values",
            "vision_attention_mask",
            "audio_attention_mask",
            "vision_input_mask",
            "vision_encoder_kwargs",
            "interpolate_pos_encoding",
            "pixel_mask",
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
            (
                vision_embedding,
                vision_mask,
                vision_spatial_shape,
                vision_prefix_tokens,
            ) = self._encode_vision_features(
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
            vision_spatial_shape, vision_prefix_tokens = None, None
            audio_embedding, audio_mask = None, None

        return {
            "token_embeds": token_embeds,
            "prompts_embedding": prompts_embedding,
            "prompts_embedding_mask": prompts_embedding_mask,
            "words_embedding": words_embedding,
            "mask": mask,
            "vision_embedding": vision_embedding,
            "vision_mask": vision_mask,
            "vision_spatial_shape": vision_spatial_shape,
            "vision_prefix_tokens": vision_prefix_tokens,
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
        media_label_embeds: Optional[Dict[str, torch.Tensor]],
        vision_embedding: Optional[torch.Tensor],
        vision_mask: Optional[torch.Tensor],
        audio_embedding: Optional[torch.Tensor],
        audio_mask: Optional[torch.Tensor],
        include_media: bool,
        kwargs: dict,
        vision_spatial_shape: Optional[torch.Tensor] = None,
        vision_prefix_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, TaskFlatInputs], Optional[torch.Tensor], Optional[torch.Tensor]]:
        parent_embeds = None
        per_task_parent_embeds: Dict[str, torch.Tensor] = {}
        media_tasks = self._media_task_names if include_media else ()
        joint_relex_active = (
            "joint_relex" in self.heads
            and classes_mapping is not None
            and any(
                ext_mapping.rel_class_to_id is not None
                and bool(ext_mapping.rel_class_to_id.class_to_id)
                for _, _, _, ext_mapping in classes_mapping.flat_extraction_iter()
            )
        )

        if classes_mapping is not None:
            if self.config.uses_per_task_parents:
                task_parent_cfgs = {
                    "ner": self.config.ner_config,
                    "classification": self.config.classification_config,
                    "open_relex": self.config.open_relex_config,
                    "structuring": self.config.structuring_config,
                }
                if (
                    self.config.ner_config is None
                    and self.config.joint_relex_config is not None
                ):
                    task_parent_cfgs["joint_relex"] = (
                        self.config.joint_relex_config
                    )
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
                prompt_parent_cfgs = {
                    "classification": self.config.classification_config,
                    "ner": self.config.ner_config,
                    "open_relex": self.config.open_relex_config,
                    "structuring": self.config.structuring_config,
                }
                if (
                    self.config.ner_config is None
                    and self.config.joint_relex_config is not None
                ):
                    prompt_parent_cfgs["joint_relex"] = (
                        self.config.joint_relex_config
                    )
                if include_media:
                    prompt_parent_cfgs.update(
                        {
                            "image_classification": (
                                self.config.image_classification_config
                            ),
                            "object_detection": (
                                self.config.object_detection_config
                            ),
                            "segmentation": self.config.segmentation_config,
                            "audio_classification": (
                                self.config.audio_classification_config
                            ),
                            "audio_segmentation": (
                                self.config.audio_segmentation_config
                            ),
                        }
                    )
                prompt_task_names = tuple(prompt_parent_cfgs)
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
                        prompt_task = task_name
                        prompt_position = prompt_task_names.index(prompt_task)
                        marker_index = task_cfg.parent_token_index
                        preceding_tasks = [
                            name
                            for name in prompt_task_names[:prompt_position]
                            if prompt_parent_cfgs[name] is not None
                            and prompt_parent_cfgs[name].parent_token_index
                            == marker_index
                        ]
                        starts = [
                            sum(
                                classes_mapping.group_count(name, batch_idx)
                                for name in preceding_tasks
                            )
                            for batch_idx in range(batch_size)
                        ]
                        sizes = classes_mapping.group_counts(
                            prompt_task,
                            batch_size,
                        )
                        parent_e, _ = self._slice_prompt_features(
                            parent_e,
                            parent_m,
                            starts,
                            sizes,
                        )
                        per_task_parent_embeds[task_name] = parent_e
                if "ner" in per_task_parent_embeds:
                    per_task_parent_embeds["joint_relex"] = per_task_parent_embeds["ner"]
            elif self.config.parent_token_index > 0:
                parent_embeds, _ = extract_prompt_features(
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
        if joint_relex_active and rel_label_embeds is None:
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
                rel_prompts_mask=rel_prompts_batch_mask,
            )
        elif joint_relex_active and rel_label_embeds is not None:
            joint_rel_flat_prompts, joint_rel_flat_mask = self._build_flat_rel_prompts(
                rel_label_embeds,
                classes_mapping,
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
            open_rel_child_embeds, open_rel_child_mask = (
                self._slice_relation_prompt_features(
                    open_rel_child_embeds,
                    open_rel_child_mask,
                    classes_mapping,
                    "open_relex",
                )
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
            all_struct_child_embeds, all_struct_child_mask = (
                extract_prompt_features(
                s_cfg.child_token_index,
                token_embeds,
                input_ids,
                attention_mask,
                batch_size,
                embed_dim,
                s_cfg.embed_child_token,
                )
            )
            struct_child_embeds, struct_child_mask = (
                self._slice_structuring_prompt_features(
                    all_struct_child_embeds,
                    all_struct_child_mask,
                    classes_mapping,
                    "structuring",
                )
            )
        elif child_label_embeds is not None:
            struct_child_mask = torch.ones(
                child_label_embeds.shape[:-1],
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            struct_child_embeds = child_label_embeds

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

        media_label_embeds = media_label_embeds or {}
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
                "image_classification": (
                    media_label_embeds.get("image_classification"),
                    None,
                ) if "image_classification" in media_label_embeds else media_child_prompts.get("image_classification", (obj_child_embeds, obj_child_mask)),
                "audio_classification": (
                    media_label_embeds.get("audio_classification"),
                    None,
                ) if "audio_classification" in media_label_embeds else media_child_prompts.get("audio_classification", (obj_child_embeds, obj_child_mask)),
                "object_detection": (
                    media_label_embeds.get("object_detection"),
                    None,
                ) if "object_detection" in media_label_embeds else media_child_prompts.get("object_detection", (obj_child_embeds, obj_child_mask)),
                "segmentation": (
                    media_label_embeds.get("segmentation"),
                    None,
                ) if "segmentation" in media_label_embeds else media_child_prompts.get("segmentation", (obj_child_embeds, obj_child_mask)),
                "audio_segmentation": (
                    media_label_embeds.get("audio_segmentation"),
                    None,
                ) if "audio_segmentation" in media_label_embeds else media_child_prompts.get("audio_segmentation", (obj_child_embeds, obj_child_mask)),
            }

            task_names = (
                "ner", "joint_relex", "classification", "structuring",
                "open_relex",
                *media_tasks,
            )
            for task_name in task_names:
                if task_name not in self.heads:
                    continue
                if task_name == "joint_relex" and not joint_relex_active:
                    continue
                child_e, child_m = task_child_map.get(task_name, (None, None))
                if child_e is None:
                    continue

                if use_per_task:
                    if task_name not in per_task_parent_embeds:
                        continue
                    task_parent_e = per_task_parent_embeds[task_name]
                else:
                    task_parent_e = parent_embeds

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
                    if task_name in {"object_detection", "segmentation"}:
                        if vision_spatial_shape is not None:
                            flat_inputs.feature_spatial_shape = vision_spatial_shape[
                                flat_inputs.batch_origin
                            ]
                        if vision_prefix_tokens is not None:
                            flat_inputs.feature_prefix_tokens = vision_prefix_tokens[
                                flat_inputs.batch_origin
                            ]
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
        per_task_parent_embeds: Dict[str, torch.Tensor],
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
                task_parent_e = per_task_parent_embeds.get(task_name)
                if task_parent_e is None:
                    continue
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
                for runtime_name in (
                    "rel_focal_loss_alpha",
                    "rel_focal_loss_gamma",
                    "rel_focal_loss_prob_margin",
                    "rel_label_smoothing",
                    "rel_negatives",
                    "rel_masking",
                    "relation_flat_ner",
                    "relation_multi_label",
                ):
                    value = runtime_kwargs.get(runtime_name)
                    if value is not None:
                        extra_kwargs[runtime_name] = value

            if name in self._focal_loss_task_names:
                extra_kwargs["base_loss_fn"] = self._make_task_loss_fn(name, runtime_kwargs)

            call_kwargs = dict(batch_kwargs)
            call_kwargs.update(extra_kwargs)
            count_key = _SET_PREDICTION_COUNT_KEYS.get(name)
            if count_key is not None and name in flat_inputs_map:
                call_kwargs[count_key] = self._flatten_set_prediction_count(
                    call_kwargs.get(count_key),
                    flat_inputs_map[name],
                    name,
                )
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
    ) -> GLiFormerOutput:
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
            joint_rel_entity_class_idx=joint_rel_out.extra.get(
                "rel_entity_class_idx"
            ),
            open_rel_entity_logits=open_rel_out.extra.get("entity_logits"),
            open_rel_logits=open_rel_out.logits,
            open_rel_assignment_logits=open_rel_out.extra.get(
                "assignment_logits"
            ),
            open_rel_batch_origin=flat_inputs_map["open_relex"].batch_origin if "open_relex" in flat_inputs_map else None,
            open_rel_anchor_mask=open_rel_out.extra.get("anchor_mask"),
            open_rel_objectness_logits=open_rel_out.extra.get(
                "objectness_logits"
            ),
            open_rel_span_idx=open_rel_out.extra.get("span_idx"),
            open_rel_span_mask=open_rel_out.extra.get("span_mask"),
            count_logits=count_out.logits,
            count_batch_origin=flat_inputs_map["count"].batch_origin if "count" in flat_inputs_map else None,
            structuring_entity_logits=struct_out.logits,
            structuring_field_logits=struct_out.extra.get(
                "entity_field_logits"
            ),
            structuring_logits=struct_out.extra.get(
                "membership_logits",
                struct_out.extra.get("structuring_logits"),
            ),
            structuring_assignment_logits=struct_out.extra.get(
                "entity_assignment_logits"
            ),
            structuring_batch_origin=flat_inputs_map["structuring"].batch_origin if "structuring" in flat_inputs_map else None,
            structuring_anchor_mask=struct_out.extra.get("anchor_mask"),
            structuring_objectness_logits=struct_out.extra.get("objectness_logits"),
            structuring_anchor_relation_scores=struct_out.extra.get(
                "anchor_relation_scores"
            ),
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
            segmentation_mask_validity=seg_out.extra.get("mask_validity"),
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
        object_detection_count: Optional[torch.Tensor] = None,
        segmentation_class_labels: Optional[torch.Tensor] = None,
        segmentation_bbox_labels: Optional[torch.Tensor] = None,
        segmentation_object_mask: Optional[torch.Tensor] = None,
        segmentation_mask_labels: Optional[torch.Tensor] = None,
        segmentation_count: Optional[torch.Tensor] = None,
        audio_segmentation_class_labels: Optional[torch.Tensor] = None,
        audio_segmentation_segment_labels: Optional[torch.Tensor] = None,
        audio_segmentation_object_mask: Optional[torch.Tensor] = None,
        audio_segmentation_mask_labels: Optional[torch.Tensor] = None,
        audio_segmentation_count: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        vision_attention_mask: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        # Joint Relex
        rel_labels: Optional[torch.Tensor] = None,
        rel_pair_mask: Optional[torch.Tensor] = None,
        rel_mask: Optional[torch.Tensor] = None,
        rel_batch_idx: Optional[torch.Tensor] = None,
        rel_span_idx: Optional[torch.Tensor] = None,
        rel_span_mask: Optional[torch.Tensor] = None,
        rel_span_class_idx: Optional[torch.Tensor] = None,
        # Open Relex
        open_rel_entity_labels: Optional[torch.Tensor] = None,
        open_rel_labels: Optional[torch.Tensor] = None,
        open_rel_assignment_labels: Optional[torch.Tensor] = None,
        open_rel_span_idx: Optional[torch.Tensor] = None,
        open_rel_span_mask: Optional[torch.Tensor] = None,
        open_rel_mask: Optional[torch.Tensor] = None,
        open_rel_count: Optional[torch.Tensor] = None,
        # Count
        count_targets: Optional[torch.Tensor] = None,
        # Groups / Structuring
        count_val: Optional[torch.Tensor] = None,
        structuring_labels: Optional[torch.Tensor] = None,
        structuring_count: Optional[torch.Tensor] = None,
        structuring_relation_labels: Optional[torch.Tensor] = None,
        structuring_relation_group_mask: Optional[torch.Tensor] = None,
        structuring_span_idx: Optional[torch.Tensor] = None,
        structuring_span_mask: Optional[torch.Tensor] = None,
        structuring_span_labels: Optional[torch.Tensor] = None,
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
    ) -> GLiFormerOutput:

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
        encoded_label_embeds = self._encode_all_labels_batched(
            cat_labels_input_ids, cat_labels_attention_mask,
            rel_labels_input_ids, rel_labels_attention_mask,
            child_labels_input_ids, child_labels_attention_mask,
            open_rel_labels_input_ids, open_rel_labels_attention_mask,
        )
        encoded_label_embeds = tuple(encoded_label_embeds)
        encoded_label_embeds += (None,) * max(
            0, 4 - len(encoded_label_embeds)
        )
        (
            cat_label_embeds,
            rel_label_embeds,
            child_label_embeds,
            open_rel_label_embeds,
        ) = encoded_label_embeds[:4]
        media_label_embeds = self._encode_media_labels_batched(kwargs)

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
                media_label_embeds=media_label_embeds,
                vision_embedding=vision_embedding,
                vision_mask=vision_mask,
                audio_embedding=audio_embedding,
                audio_mask=audio_mask,
                include_media=include_media,
                kwargs=kwargs,
                vision_spatial_shape=representations["vision_spatial_shape"],
                vision_prefix_tokens=representations["vision_prefix_tokens"],
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
            object_detection_count=object_detection_count,
            segmentation_class_labels=segmentation_class_labels,
            segmentation_bbox_labels=segmentation_bbox_labels,
            segmentation_object_mask=segmentation_object_mask,
            segmentation_mask_labels=segmentation_mask_labels,
            segmentation_count=segmentation_count,
            audio_segmentation_class_labels=audio_segmentation_class_labels,
            audio_segmentation_segment_labels=audio_segmentation_segment_labels,
            audio_segmentation_object_mask=audio_segmentation_object_mask,
            audio_segmentation_mask_labels=audio_segmentation_mask_labels,
            audio_segmentation_count=audio_segmentation_count,
            rel_pair_mask=rel_pair_mask,
            rel_mask=rel_mask, rel_batch_idx=rel_batch_idx,
            rel_span_idx=rel_span_idx, rel_span_mask=rel_span_mask,
            rel_span_class_idx=rel_span_class_idx,
            open_rel_entity_labels=open_rel_entity_labels,
            open_rel_labels=open_rel_labels,
            open_rel_assignment_labels=open_rel_assignment_labels,
            open_rel_span_idx=open_rel_span_idx,
            open_rel_span_mask=open_rel_span_mask,
            open_rel_mask=open_rel_mask,
            open_rel_count=open_rel_count,
            count_targets=count_targets,
            count_val=count_val, structuring_labels=structuring_labels,
            structuring_count=structuring_count,
            structuring_relation_labels=structuring_relation_labels,
            structuring_relation_group_mask=structuring_relation_group_mask,
            structuring_span_idx=structuring_span_idx,
            structuring_span_mask=structuring_span_mask,
            structuring_span_labels=structuring_span_labels,
            embedding_labels=embedding_labels,
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

    def _forward_omni_task_heads(self, *args, **kwargs) -> GLiFormerOutput:
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
        rel_mask: Optional[torch.Tensor] = None,
        rel_batch_idx: Optional[torch.Tensor] = None,
        rel_span_idx: Optional[torch.Tensor] = None,
        rel_span_mask: Optional[torch.Tensor] = None,
        rel_span_class_idx: Optional[torch.Tensor] = None,
        # Open Relex
        open_rel_entity_labels: Optional[torch.Tensor] = None,
        open_rel_labels: Optional[torch.Tensor] = None,
        open_rel_assignment_labels: Optional[torch.Tensor] = None,
        open_rel_span_idx: Optional[torch.Tensor] = None,
        open_rel_span_mask: Optional[torch.Tensor] = None,
        open_rel_mask: Optional[torch.Tensor] = None,
        open_rel_count: Optional[torch.Tensor] = None,
        # Count
        count_targets: Optional[torch.Tensor] = None,
        # Groups / Structuring
        count_val: Optional[torch.Tensor] = None,
        structuring_labels: Optional[torch.Tensor] = None,
        structuring_count: Optional[torch.Tensor] = None,
        structuring_relation_labels: Optional[torch.Tensor] = None,
        structuring_relation_group_mask: Optional[torch.Tensor] = None,
        structuring_span_idx: Optional[torch.Tensor] = None,
        structuring_span_mask: Optional[torch.Tensor] = None,
        structuring_span_labels: Optional[torch.Tensor] = None,
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
    ) -> GLiFormerTextOutput:
        classes_mapping = kwargs.get("classes_mapping")
        representation_keys = {
            "packing_config",
            "pair_attention_mask",
            "bbox",
            "layout_input_mask",
            "page_token_ids",
            "page_input_mask",
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
        encoded_label_embeds = self._encode_all_labels_batched(
            cat_labels_input_ids, cat_labels_attention_mask,
            rel_labels_input_ids, rel_labels_attention_mask,
            child_labels_input_ids, child_labels_attention_mask,
            open_rel_labels_input_ids, open_rel_labels_attention_mask,
        )
        encoded_label_embeds = tuple(encoded_label_embeds)
        encoded_label_embeds += (None,) * max(
            0, 4 - len(encoded_label_embeds)
        )
        (
            cat_label_embeds,
            rel_label_embeds,
            child_label_embeds,
            open_rel_label_embeds,
        ) = encoded_label_embeds[:4]
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
                media_label_embeds=None,
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
            rel_mask=rel_mask, rel_batch_idx=rel_batch_idx,
            rel_span_idx=rel_span_idx, rel_span_mask=rel_span_mask,
            rel_span_class_idx=rel_span_class_idx,
            open_rel_entity_labels=open_rel_entity_labels,
            open_rel_labels=open_rel_labels,
            open_rel_assignment_labels=open_rel_assignment_labels,
            open_rel_span_idx=open_rel_span_idx,
            open_rel_span_mask=open_rel_span_mask,
            open_rel_mask=open_rel_mask,
            open_rel_count=open_rel_count,
            count_targets=count_targets,
            count_val=count_val, structuring_labels=structuring_labels,
            structuring_count=structuring_count,
            structuring_relation_labels=structuring_relation_labels,
            structuring_relation_group_mask=structuring_relation_group_mask,
            structuring_span_idx=structuring_span_idx,
            structuring_span_mask=structuring_span_mask,
            structuring_span_labels=structuring_span_labels,
            embedding_labels=embedding_labels,
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

    def _forward_all_tasks(self, *args, **kwargs) -> GLiFormerOutput:
        return self._forward_omni_task_heads(*args, **kwargs)

    def _forward_text_tasks(self, *args, **kwargs) -> GLiFormerTextOutput:
        return self._forward_text_task_heads(*args, **kwargs)

class GLiFormerTextModel(_GLiFormerJointForwardModel):
    """Text-only GLiFormer model."""

    enabled_task_names = TASK_REGISTRY.text_tasks
    output_cls = GLiFormerTextOutput

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
                "GLiFormerTextModel supports text/layout inputs only; "
                f"received media arguments: {', '.join(present)}"
            )

    def forward(self, *args, **kwargs) -> GLiFormerTextOutput:
        self._reject_media_inputs(kwargs)
        output = self._forward_text_task_heads(*args, **kwargs)
        if type(output) is GLiFormerTextOutput:
            return output
        return _filtered_output(
            GLiFormerTextOutput,
            **{field.name: getattr(output, field.name, None) for field in fields(GLiFormerTextOutput)},
        )


class _MediaOnlyBiEncoderModel(BaseGLiFormerModel):
    """Shared implementation for efficient single-media bi-encoder models."""

    media_task_names: Tuple[str, ...] = ()
    media_token_name: str = ""
    media_mask_name: str = ""
    bi_encoder_cls = None

    def _init_token_rep_layer(self, config, from_pretrained, cache_dir, local_files_only=False):
        if self.bi_encoder_cls is None:
            raise NotImplementedError("media-only model must define bi_encoder_cls")
        return self.bi_encoder_cls(
            config, from_pretrained=from_pretrained, cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

    def __init__(self, config, from_pretrained=False, cache_dir=None, local_files_only=False):
        super().__init__(
            config, from_pretrained=from_pretrained, cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.media_parent_embeddings = nn.ParameterDict(
            {
                task_name: nn.Parameter(torch.zeros(config.hidden_size))
                for task_name in self.media_task_names
            }
        )
        for param in self.media_parent_embeddings.values():
            nn.init.normal_(param, std=0.02)

    def _encode_media_tokens(self, media_values: torch.Tensor, media_mask: Optional[torch.Tensor], **kwargs):
        raise NotImplementedError

    def _parent_inputs_for_task(
        self,
        classes_mapping,
        task_name: str,
        batch_size: int,
        media_tokens: torch.Tensor,
        media_mask: torch.Tensor,
        device,
        dtype,
    ) -> torch.Tensor:
        counts = classes_mapping.group_counts(task_name, batch_size)
        max_groups = max(max(counts, default=0), 1)
        parent = torch.zeros(batch_size, max_groups, self.config.hidden_size, device=device, dtype=dtype)
        task_parent = self._media_parent_embedding_for_task(
            task_name,
            media_tokens,
            media_mask,
            device=device,
            dtype=dtype,
        )
        for batch_idx, count in enumerate(counts):
            if count > 0:
                parent[batch_idx, :count] = task_parent[batch_idx]
        return parent

    def _media_parent_embedding_for_task(
        self,
        task_name: str,
        media_tokens: torch.Tensor,
        media_mask: torch.Tensor,
        *,
        device,
        dtype,
    ) -> torch.Tensor:
        source = getattr(self.config, "media_parent_embedding_source", "fixed")
        if source == "fixed":
            task_parent = self.media_parent_embeddings[task_name].to(device=device, dtype=dtype)
            return task_parent.unsqueeze(0).expand(media_tokens.shape[0], -1)

        tokens = media_tokens.to(device=device, dtype=dtype)
        if media_mask is None or media_mask.shape[-1] != tokens.shape[1]:
            mask = torch.ones(tokens.shape[:2], dtype=tokens.dtype, device=device)
        else:
            mask = media_mask.to(device=device, dtype=tokens.dtype)

        if source == "first":
            valid = mask > 0
            first_idx = valid.float().argmax(dim=1)
            batch_idx = torch.arange(tokens.shape[0], device=device)
            parent = tokens[batch_idx, first_idx]
            return parent * valid.any(dim=1).to(dtype=dtype).unsqueeze(-1)

        weights = mask.unsqueeze(-1)
        pooled = (tokens * weights).sum(dim=1)
        if source == "mean":
            pooled = pooled / weights.sum(dim=1).clamp(min=1.0)
        elif source != "sum":
            raise ValueError(f"Unknown media_parent_embedding_source: {source!r}")
        return pooled

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
        labels_mask = torch.ones(
            labels_embeds.shape[:1],
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
        media_spatial_shape: Optional[torch.Tensor] = None,
        media_prefix_tokens: Optional[torch.Tensor] = None,
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
            parent_embeds = self._parent_inputs_for_task(
                classes_mapping,
                task_name,
                batch_size,
                media_tokens,
                media_mask,
                media_tokens.device,
                media_tokens.dtype,
            )
            flat_inputs = self._build_flat_inputs(
                parent_embeds,
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
                if media_spatial_shape is not None:
                    flat_inputs.feature_spatial_shape = media_spatial_shape[
                        flat_inputs.batch_origin
                    ]
                if media_prefix_tokens is not None:
                    flat_inputs.feature_prefix_tokens = media_prefix_tokens[
                        flat_inputs.batch_origin
                    ]
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
            count_key = _SET_PREDICTION_COUNT_KEYS.get(name)
            if count_key is not None:
                call_kwargs[count_key] = self._flatten_set_prediction_count(
                    call_kwargs.get(count_key),
                    flat_inputs_map[name],
                    name,
                )
            dep_outputs = {
                dependency: head_outputs[dependency]
                for dependency in head.dependencies
                if dependency in head_outputs
            }
            output = head(
                shared,
                dependency_outputs=dep_outputs,
                **call_kwargs,
            )
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


class GLiFormerVisionModel(_MediaOnlyBiEncoderModel):
    """Vision-only GLiFormer model using a vision/text-label bi-encoder."""

    enabled_task_names = TASK_REGISTRY.vision_tasks
    output_cls = GLiFormerVisionOutput
    media_task_names = TASK_REGISTRY.vision_tasks
    media_token_name = "vision"
    media_mask_name = "vision_attention_mask"
    bi_encoder_cls = VisionBiEncoder

    def _encode_media_tokens(self, media_values: torch.Tensor, media_mask: Optional[torch.Tensor], **kwargs):
        return self.token_rep_layer.vision_encoder(
            media_values,
            **vision_encoder_kwargs(kwargs),
        )

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        vision_attention_mask: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        image_classification_labels: Optional[torch.Tensor] = None,
        object_detection_class_labels: Optional[torch.Tensor] = None,
        object_detection_bbox_labels: Optional[torch.Tensor] = None,
        object_detection_object_mask: Optional[torch.Tensor] = None,
        object_detection_count: Optional[torch.Tensor] = None,
        segmentation_class_labels: Optional[torch.Tensor] = None,
        segmentation_bbox_labels: Optional[torch.Tensor] = None,
        segmentation_object_mask: Optional[torch.Tensor] = None,
        segmentation_mask_labels: Optional[torch.Tensor] = None,
        segmentation_count: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        **kwargs,
    ) -> GLiFormerVisionOutput:
        classes_mapping = kwargs.get("classes_mapping")
        if pixel_values is None:
            raise ValueError("GLiFormerVisionModel requires pixel_values")
        if classes_mapping is None:
            raise ValueError("GLiFormerVisionModel requires classes_mapping to build vision task inputs")

        vision_features = self.token_rep_layer.vision_encoder.forward_features(
            pixel_values,
            **vision_encoder_kwargs(kwargs),
        )
        vision_tokens = vision_features.token_embeddings
        vision_mask = vision_token_mask(
            vision_tokens,
            vision_attention_mask,
            vision_features.prefix_tokens,
            vision_features.spatial_shape,
        )
        flat_inputs_map = self._build_media_flat_inputs(
            classes_mapping,
            vision_tokens,
            vision_mask,
            kwargs,
            media_spatial_shape=vision_features.spatial_shape,
            media_prefix_tokens=vision_features.prefix_tokens,
        )

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
            object_detection_count=object_detection_count,
            segmentation_class_labels=segmentation_class_labels,
            segmentation_bbox_labels=segmentation_bbox_labels,
            segmentation_object_mask=segmentation_object_mask,
            segmentation_mask_labels=segmentation_mask_labels,
            segmentation_count=segmentation_count,
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
        return GLiFormerVisionOutput(
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
            segmentation_mask_validity=seg_out.extra.get("mask_validity"),
            segmentation_prototypes=seg_out.extra.get("prototypes"),
            segmentation_coefficients=seg_out.extra.get("coefficients"),
            vision_embedding=vision_tokens,
            vision_mask=vision_mask,
            words_embedding=vision_tokens,
            mask=vision_mask,
        )


class GLiFormerAudioModel(_MediaOnlyBiEncoderModel):
    """Audio-only GLiFormer model using an audio/text-label bi-encoder."""

    enabled_task_names = TASK_REGISTRY.audio_tasks
    output_cls = GLiFormerAudioOutput
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
        audio_segmentation_count: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        **kwargs,
    ) -> GLiFormerAudioOutput:
        classes_mapping = kwargs.get("classes_mapping")
        if audio_values is None:
            raise ValueError("GLiFormerAudioModel requires audio_values")
        if classes_mapping is None:
            raise ValueError("GLiFormerAudioModel requires classes_mapping to build audio task inputs")

        audio_tokens = self._encode_media_tokens(audio_values, audio_attention_mask, **kwargs)
        audio_mask = audio_token_mask(
            self.token_rep_layer.audio_encoder,
            audio_tokens,
            audio_attention_mask,
        )
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
            audio_segmentation_count=audio_segmentation_count,
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
        return GLiFormerAudioOutput(
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


class GLiFormerLayoutModel(_GLiFormerJointForwardModel):
    """Text + document-layout variant.

    Layout coordinates are supplied as ``bbox`` with shape
    ``(batch, sequence, 4)``. Optional ``pixel_values`` can also be supplied for
    layout backbones such as LayoutLMv3 that fuse text, boxes, and page images.
    """

    enabled_task_names = TASK_REGISTRY.text_tasks
    output_cls = GLiFormerLayoutOutput
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

    def _init_token_rep_layer(self, config, from_pretrained, cache_dir, local_files_only=False):
        if config.labels_encoder is not None:
            return self.layout_bi_encoder_cls(
                config, from_pretrained, cache_dir=cache_dir, local_files_only=local_files_only,
            )
        return self.layout_encoder_cls(
            config, from_pretrained, cache_dir=cache_dir, local_files_only=local_files_only,
        )

    @classmethod
    def _reject_unsupported_input_names(cls, kwargs: dict) -> None:
        unsupported = [key for key in kwargs if key in cls.unsupported_input_names]
        if unsupported:
            raise ValueError(
                "GLiFormerLayoutModel uses canonical input names only; "
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
            "layout_input_mask",
            "page_token_ids",
            "page_input_mask",
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
                "GLiFormerLayoutModel supports text/layout inputs only; "
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
            words_embedding = self._apply_word_rnn(words_embedding, mask)
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
        words_embedding = self._apply_word_rnn(words_embedding, mask)
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

    def forward(self, *args, **kwargs) -> GLiFormerLayoutOutput:
        self._reject_unsupported_media_inputs(kwargs)
        self._reject_unsupported_input_names(kwargs)
        output = self._forward_text_task_heads(*args, **kwargs)
        if type(output) is GLiFormerLayoutOutput:
            return output
        return _filtered_output(
            GLiFormerLayoutOutput,
            **{field.name: getattr(output, field.name, None) for field in fields(GLiFormerLayoutOutput)},
        )


class GLiFormerOmniModel(_GLiFormerJointForwardModel):
    """Text + vision + audio model with early fusion inside the text transformer."""

    output_cls = GLiFormerOmniOutput
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
                "GLiFormerOmniModel requires model_variant='omni'; "
                f"got {variant!r}"
            ) from exc

    def _init_token_rep_layer(self, config, from_pretrained, cache_dir, local_files_only=False):
        if getattr(config, "omni_modalities", None) is None:
            config.omni_modalities = self._infer_omni_modalities(config)
        if config.labels_encoder is not None:
            return self.omni_bi_encoder_cls(
                config,
                from_pretrained=from_pretrained,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        return self.omni_encoder_cls(
            config,
            from_pretrained=from_pretrained,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

    @classmethod
    def _reject_unsupported_input_names(cls, kwargs: dict) -> None:
        unsupported = [key for key in kwargs if key in cls.unsupported_input_names]
        if unsupported:
            raise ValueError(
                "GLiFormerOmniModel uses canonical input names only; "
                f"use input_ids/pixel_values/bbox/audio_values instead of: {', '.join(sorted(unsupported))}"
            )

    @classmethod
    def _omni_kwargs(cls, kwargs: dict) -> dict:
        allowed = {
            "packing_config", "pair_attention_mask", "pixel_values",
            "vision_attention_mask", "audio_values", "audio_attention_mask",
            "vision_input_mask", "audio_input_mask", "bbox", "layout_input_mask",
            "vision_encoder_kwargs", "interpolate_pos_encoding", "pixel_mask",
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
                _cache_forward_modality(
                    self,
                    "vision",
                    output.vision_embeddings,
                    output.vision_attention_mask,
                    output.vision_spatial_shape,
                    output.vision_prefix_tokens,
                )
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
            words_embedding = self._apply_word_rnn(words_embedding, mask)
            return token_embeds, labels_embeds, labels_mask, words_embedding, mask

        output = self.token_rep_layer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **omni_kwargs,
        )
        token_embeds = output.text_embeddings
        if getattr(output, "vision_embeddings", None) is not None:
            _cache_forward_modality(
                self,
                "vision",
                output.vision_embeddings,
                output.vision_attention_mask,
                output.vision_spatial_shape,
                output.vision_prefix_tokens,
            )
        if getattr(output, "audio_embeddings", None) is not None:
            _cache_forward_modality(self, "audio", output.audio_embeddings, output.audio_attention_mask)
        prompts_embedding, prompts_embedding_mask, words_embedding, mask = (
            extract_prompt_features_and_word_embeddings(
                self.config.class_token_index, token_embeds, input_ids, attention_mask,
                text_lengths, words_mask, self.config.embed_ent_token,
            )
        )
        words_embedding = self._apply_word_rnn(words_embedding, mask)
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

    def forward(self, *args, **kwargs) -> GLiFormerOmniOutput:
        self._reject_unsupported_input_names(kwargs)
        return self._forward_omni_task_heads(*args, **kwargs)


def resolve_gliformer_model_class(config) -> type[BaseGLiFormerModel]:
    variant = _normalize_model_variant(getattr(config, "model_variant", None))

    if variant == "text":
        return GLiFormerTextModel
    if variant == "omni":
        return GLiFormerOmniModel
    if variant == "vision":
        return GLiFormerVisionModel
    if variant == "layout":
        return GLiFormerLayoutModel
    if variant == "audio":
        return GLiFormerAudioModel
    raise ValueError(f"Unknown GLiFormer model_variant: {getattr(config, 'model_variant', None)!r}")


class GLiFormerModel(GLiFormerTextModel):
    """Backward-compatible text model class.

    The user-facing :class:`gliformer.gliformer.GLiFormer` class selects concrete
    multimodal variants via :func:`resolve_gliformer_model_class`.
    """
