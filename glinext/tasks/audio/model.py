"""Audio task heads: audio classification and temporal segmentation."""

import torch
import torch.nn.functional as F
from torch import nn

from ...encoders.audio import validate_audio_attention_mask
from ...layers import PositionEmbedding, normalized_grid_1d
from ...layers.mlp import create_mlp
from .. import TaskHeadOutput
from ..media import (
    MediaClassificationHead,
    MediaSetPredictionHead,
    flat_features,
    matched_classification_loss,
    matched_mask_loss,
    normalized_objectness_loss,
    score_anchor_labels,
)


def _segments_from_raw(raw: torch.Tensor) -> torch.Tensor:
    coords = raw.sigmoid()
    start = torch.minimum(coords[..., 0], coords[..., 1])
    end = torch.maximum(coords[..., 0], coords[..., 1])
    return torch.stack([start, end], dim=-1)


class AudioClassificationHead(MediaClassificationHead):
    """Audio-level classification over pooled audio tokens."""

    name = "audio_classification"
    config_attribute = "audio_classification_config"
    labels_key = "audio_classification_labels"


class AudioSegmentationHead(MediaSetPredictionHead):
    """Anchor-slot temporal segmentation head over audio tokens.

    Outputs normalized ``[start, end]`` segments, class scores, objectness, and
    optional 1D mask logits built from audio-token prototypes.
    """

    name = "audio_segmentation"
    dependencies = []

    def __init__(self, config, hidden_size, dropout, shared_layers=None):
        super().__init__()
        cfg = config.audio_segmentation_config
        self.seg_cfg = cfg
        self.loss_coef = cfg.loss_coef
        if shared_layers is None:
            shared_layers = {}
        self._init_set_prediction_pipeline(
            cfg,
            config,
            hidden_size,
            dropout,
            shared_layers,
            geometry_cost=getattr(cfg, "matcher_segment_cost", 5.0),
        )
        self.segment_head = create_mlp(
            input_dim=hidden_size,
            intermediate_dims=[hidden_size],
            output_dim=2,
            dropout=dropout,
            activation="gelu",
        )
        self.objectness_head = nn.Linear(hidden_size, 1)
        self.memory_position_embedding = None
        self.query_position_embedding = None
        if hasattr(self, "anchor_refine"):
            self.memory_position_embedding = PositionEmbedding.from_config(
                getattr(cfg, "memory_position_embedding_type", "sine1d"),
                hidden_size,
                **dict(
                    getattr(cfg, "memory_position_embedding_kwargs", None)
                    or {}
                ),
            )
            query_kwargs = dict(
                getattr(cfg, "query_position_embedding_kwargs", None) or {}
            )
            query_type = getattr(
                cfg,
                "query_position_embedding_type",
                "sine1d",
            )
            if PositionEmbedding.strategy_class(query_type).requires_num_embeddings:
                query_kwargs.setdefault("num_embeddings", cfg.num_fixed_slots)
            self.query_position_embedding = PositionEmbedding.from_config(
                query_type,
                hidden_size,
                **query_kwargs,
            )
        self.num_prototypes = int(getattr(cfg, "num_prototypes", 32))
        self.mask_size = int(getattr(cfg, "mask_size", 256))
        self.proto_proj = nn.Linear(hidden_size, self.num_prototypes)
        self.proto_refine = nn.Sequential(
            nn.Conv1d(self.num_prototypes, self.num_prototypes, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.num_prototypes, self.num_prototypes, kernel_size=1),
        )
        self.coeff_head = nn.Linear(hidden_size, self.num_prototypes)

    @classmethod
    def from_config(cls, config, shared_layers=None, **kwargs):
        if config.audio_segmentation_config is None:
            return None
        return cls(
            config,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            shared_layers=shared_layers,
        )

    def _memory_positions(self, audio_features, audio_mask):
        if self.memory_position_embedding is None:
            return None
        batch_size, token_count, hidden_size = audio_features.shape
        audio_mask = validate_audio_attention_mask(
            audio_mask,
            batch_size=batch_size,
            time_length=token_count,
            name="audio feature mask",
        )
        valid_lengths = (
            audio_mask.long().sum(dim=-1)
            if audio_mask is not None
            else torch.full(
                (batch_size,),
                token_count,
                dtype=torch.long,
                device=audio_features.device,
            )
        )
        positions = audio_features.new_zeros(
            batch_size,
            token_count,
            hidden_size,
        )
        for batch_idx, valid_length in enumerate(valid_lengths.tolist()):
            if valid_length <= 0:
                continue
            coordinates = normalized_grid_1d(
                valid_length,
                device=audio_features.device,
                dtype=torch.float32,
            )
            row_positions = self.memory_position_embedding(
                coordinates,
                count=valid_length,
                dtype=audio_features.dtype,
                device=audio_features.device,
            )
            if row_positions is None:
                return None
            if row_positions.dim() == 3 and row_positions.shape[0] == 1:
                row_positions = row_positions[0]
            if row_positions.shape != (valid_length, hidden_size):
                raise ValueError(
                    "Audio memory positions must have shape (valid_time, hidden_size)"
                )
            positions[batch_idx, :valid_length] = row_positions
        return positions

    def _query_positions(self, anchors):
        if self.query_position_embedding is None or anchors.shape[1] == 0:
            return None
        coordinates = normalized_grid_1d(
            anchors.shape[1],
            device=anchors.device,
            dtype=torch.float32,
        )
        positions = self.query_position_embedding(
            coordinates,
            count=anchors.shape[1],
            dtype=anchors.dtype,
            device=anchors.device,
        )
        if positions is None:
            return None
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        if positions.shape[-2:] != anchors.shape[-2:] or positions.shape[0] not in {
            1,
            anchors.shape[0],
        }:
            raise ValueError(
                "Audio query positions must have shape (1, slots, hidden_size) "
                "or (batch, slots, hidden_size)"
            )
        return positions

    def _compute_segmentation(self, flat_inputs, count=None, threshold=0.5):
        audio_features, audio_mask = flat_features(flat_inputs)
        anchors, anchor_mask = self.anchor_layer(
            flat_inputs.parent_embedding,
            audio_features,
            count=count,
            threshold=threshold,
            feature_mask=audio_mask,
        )
        if hasattr(self, "anchor_refine"):
            anchors = self.anchor_refine(
                anchors,
                audio_features,
                token_mask=audio_mask,
                query_mask=anchor_mask,
                query_pos_emb=self._query_positions(anchors),
                memory_pos_emb=self._memory_positions(
                    audio_features,
                    audio_mask,
                ),
                memory_position_in_values=getattr(
                    self.seg_cfg,
                    "memory_position_in_values",
                    True,
                ),
            )
        class_logits = score_anchor_labels(
            self.anchor_modeling,
            self.cls_head,
            anchors,
            flat_inputs.child_embedding,
            flat_inputs.child_mask,
        )
        segment_preds = _segments_from_raw(self.segment_head(anchors))
        objectness_logits = self.objectness_head(anchors).squeeze(-1)
        return class_logits, segment_preds, objectness_logits, anchors, anchor_mask

    def _prototype_masks(self, audio_tokens, audio_mask=None):
        if audio_mask is not None:
            audio_mask = validate_audio_attention_mask(
                audio_mask,
                batch_size=audio_tokens.shape[0],
                time_length=audio_tokens.shape[1],
                name="audio feature mask",
            )
            resized = []
            for batch_idx, valid_length in enumerate(
                audio_mask.long().sum(dim=-1).tolist()
            ):
                if valid_length <= 0:
                    resized.append(
                        audio_tokens.new_zeros(
                            1,
                            self.num_prototypes,
                            self.mask_size,
                        )
                    )
                    continue
                # Refine the valid prefix itself, so convolutional context at
                # its right boundary is identical regardless of batch padding.
                proto = self.proto_proj(
                    audio_tokens[batch_idx : batch_idx + 1, :valid_length]
                ).transpose(1, 2)
                proto = self.proto_refine(proto)
                resized.append(
                    F.interpolate(
                        proto,
                        size=self.mask_size,
                        mode="linear",
                        align_corners=False,
                    )
                )
            return torch.cat(resized, dim=0)
        proto = self.proto_proj(audio_tokens).transpose(1, 2)
        proto = self.proto_refine(proto)
        return F.interpolate(
            proto,
            size=self.mask_size,
            mode="linear",
            align_corners=False,
        )

    def _match_single(
        self,
        class_logits,
        segment_preds,
        gold_classes,
        gold_segments,
        valid_segments,
        prediction_mask=None,
    ):
        return self._match_geometry(
            class_logits,
            segment_preds,
            gold_classes,
            gold_segments,
            valid_segments,
            prediction_mask=prediction_mask,
        )

    def _segmentation_loss(
        self,
        class_logits,
        segment_preds,
        objectness_logits,
        anchor_mask,
        child_mask,
        class_labels,
        segment_labels,
        segment_mask,
        base_loss_fn=None,
    ) -> tuple[torch.Tensor, dict[int, list]]:
        B, A, C = class_logits.shape
        target_objectness = torch.zeros_like(objectness_logits)
        matched_segment_pred = []
        matched_segment_target = []
        matches_by_batch: dict[int, list] = {}

        for b in range(B):
            matches = self._match_single(
                class_logits[b],
                segment_preds[b],
                class_labels[b],
                segment_labels[b],
                segment_mask[b],
                anchor_mask[b],
            )
            matches_by_batch[b] = matches
            for anchor_idx, seg_idx in matches:
                target_objectness[b, anchor_idx] = 1.0
                matched_segment_pred.append(segment_preds[b, anchor_idx])
                matched_segment_target.append(segment_labels[b, seg_idx])

        loss_fn = base_loss_fn or self._default_binary_loss
        class_loss = matched_classification_loss(
            class_logits,
            class_labels,
            child_mask,
            matches_by_batch,
            loss_fn,
            class_probability=getattr(
                self.seg_cfg,
                "class_probability",
                "sigmoid",
            ),
        )
        objectness_loss = normalized_objectness_loss(
            objectness_logits,
            target_objectness,
            anchor_mask,
            loss_fn,
            positive_weight=getattr(
                self.seg_cfg,
                "objectness_positive_weight",
                1.0,
            ),
            negative_weight=getattr(
                self.seg_cfg,
                "objectness_negative_weight",
                1.0,
            ),
            loss_kwargs=self._objectness_loss_kwargs(),
        )

        if matched_segment_pred:
            segment_loss = F.l1_loss(
                torch.stack(matched_segment_pred),
                torch.stack(matched_segment_target).to(segment_preds.device),
                reduction="sum",
            ) / len(matched_segment_pred)
        else:
            segment_loss = class_logits.new_tensor(0.0)

        loss = (
            float(getattr(self.seg_cfg, "class_loss_coef", 1.0)) * class_loss
            + float(getattr(self.seg_cfg, "segment_loss_coef", 5.0)) * segment_loss
            + float(getattr(self.seg_cfg, "objectness_loss_coef", 1.0)) * objectness_loss
        )
        return loss, matches_by_batch

    def forward(self, shared, dependency_outputs, flat_inputs=None, **batch):
        class_labels = batch.get("audio_segmentation_class_labels")
        segment_labels = batch.get("audio_segmentation_segment_labels")
        segment_mask = batch.get("audio_segmentation_object_mask")
        mask_labels = batch.get("audio_segmentation_mask_labels")

        class_logits, segment_preds, objectness_logits, anchors, anchor_mask = (
            self._compute_segmentation(
                flat_inputs,
                count=batch.get("audio_segmentation_count"),
                threshold=batch.get("threshold", 0.5),
            )
        )
        audio_features, audio_mask = flat_features(flat_inputs)
        prototypes = self._prototype_masks(audio_features, audio_mask)
        coefficients = self.coeff_head(anchors)
        mask_logits = torch.einsum("bpt,bap->bat", prototypes, coefficients)

        loss = None
        matches = None
        if class_labels is not None and segment_labels is not None and segment_mask is not None:
            seg_loss, matches = self._segmentation_loss(
                class_logits,
                segment_preds,
                objectness_logits,
                anchor_mask,
                flat_inputs.child_mask,
                class_labels,
                segment_labels,
                segment_mask,
                base_loss_fn=batch.get("base_loss_fn"),
            )
            mask_loss = class_logits.new_tensor(0.0)
            if mask_labels is not None:
                mask_loss = matched_mask_loss(
                    mask_logits,
                    mask_labels,
                    matches,
                    batch.get("base_loss_fn") or self._default_binary_loss,
                )
            loss = seg_loss + float(getattr(self.seg_cfg, "mask_loss_coef", 1.0)) * mask_loss

        return TaskHeadOutput(
            loss=loss,
            logits=class_logits,
            extra={
                "segment_preds": segment_preds,
                "objectness_logits": objectness_logits,
                "anchor_mask": anchor_mask,
                "anchors": anchors,
                "matches": matches,
                "mask_logits": mask_logits,
                "prototypes": prototypes,
                "coefficients": coefficients,
            },
        )
