"""Audio task heads: audio classification and temporal segmentation."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from gliner.modeling.loss_functions import focal_loss_with_logits

from .. import TaskHead, TaskHeadOutput
from ...layers import Pooling
from ...layers.mlp import create_mlp
from ..classification.scorer import ClassificationScorer


def _segments_from_raw(raw: torch.Tensor) -> torch.Tensor:
    coords = raw.sigmoid()
    start = torch.minimum(coords[..., 0], coords[..., 1])
    end = torch.maximum(coords[..., 0], coords[..., 1])
    return torch.stack([start, end], dim=-1)


class AudioClassificationHead(TaskHead):
    """Audio-level classification over pooled audio tokens."""

    name = "audio_classification"
    dependencies = []

    def __init__(self, config, hidden_size, dropout, shared_layers=None):
        super().__init__()
        cfg = config.audio_classification_config
        self.loss_coef = cfg.loss_coef
        if shared_layers is None:
            shared_layers = {}
        self._init_anchor_pipeline(cfg, config, hidden_size, dropout, shared_layers)
        self.pooling = Pooling.from_config(
            pooling_type=getattr(cfg, "pooling_type", "mean"),
            hidden_size=hidden_size,
        )
        self.scorer = ClassificationScorer.from_config(
            scorer_type=getattr(cfg, "scorer_type", "dot"),
            hidden_size=hidden_size,
        )

    @classmethod
    def from_config(cls, config, shared_layers=None, **kwargs):
        if config.audio_classification_config is None:
            return None
        return cls(
            config,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            shared_layers=shared_layers,
        )

    def forward(self, shared, dependency_outputs, flat_inputs=None, **batch):
        labels = batch.get("audio_classification_labels")
        base_loss_fn = batch.get("base_loss_fn")
        audio_rep = self.pooling(flat_inputs.words_embedding, flat_inputs.mask)
        anchors, _ = self.anchor_layer(flat_inputs.parent_embedding, flat_inputs.words_embedding)
        if hasattr(self, "anchor_refine"):
            anchors = self.anchor_refine(anchors, flat_inputs.words_embedding, token_mask=flat_inputs.mask)
        fused = self.anchor_modeling(anchors, flat_inputs.child_embedding).squeeze(1)
        logits = self.scorer(audio_rep, fused)

        loss = None
        if labels is not None:
            loss_fn = base_loss_fn or focal_loss_with_logits
            min_c = min(logits.shape[1], labels.shape[1])
            losses = loss_fn(logits[:, :min_c], labels[:, :min_c])
            loss = (losses * flat_inputs.child_mask[:, :min_c].float()).sum()
        return TaskHeadOutput(loss=loss, logits=logits)


class AudioSegmentationHead(TaskHead):
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
        self._init_anchor_pipeline(cfg, config, hidden_size, dropout, shared_layers)
        self.class_head = nn.Linear(hidden_size, 1)
        self.segment_head = create_mlp(
            input_dim=hidden_size,
            intermediate_dims=[hidden_size],
            output_dim=2,
            dropout=dropout,
            activation="gelu",
        )
        self.objectness_head = nn.Linear(hidden_size, 1)
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

    def _compute_segmentation(self, flat_inputs, count=None, threshold=0.5):
        anchors, anchor_mask = self.anchor_layer(
            flat_inputs.parent_embedding,
            flat_inputs.words_embedding,
            count=count,
            threshold=threshold,
        )
        if hasattr(self, "anchor_refine"):
            anchors = self.anchor_refine(anchors, flat_inputs.words_embedding, token_mask=flat_inputs.mask)
        fused = self.anchor_modeling(anchors, flat_inputs.child_embedding)
        class_logits = self.class_head(fused).squeeze(-1)
        segment_preds = _segments_from_raw(self.segment_head(anchors))
        objectness_logits = self.objectness_head(anchors).squeeze(-1)
        return class_logits, segment_preds, objectness_logits, anchors, anchor_mask

    def _prototype_masks(self, audio_tokens):
        proto = self.proto_proj(audio_tokens).transpose(1, 2)
        proto = self.proto_refine(proto)
        return F.interpolate(proto, size=self.mask_size, mode="linear", align_corners=False)

    def _match_single(self, class_logits, segment_preds, gold_classes, gold_segments, valid_segments):
        valid_idx = torch.nonzero(valid_segments > 0, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0 or class_logits.shape[0] == 0:
            return []

        gold_classes = gold_classes[valid_idx].long()
        gold_segments = gold_segments[valid_idx]
        probs = class_logits.sigmoid()
        class_cost = []
        for cls in gold_classes:
            if 0 <= int(cls.item()) < probs.shape[1]:
                class_cost.append(-probs[:, int(cls.item())])
            else:
                class_cost.append(torch.zeros(probs.shape[0], device=probs.device))
        class_cost = torch.stack(class_cost, dim=1)
        segment_cost = torch.cdist(segment_preds, gold_segments, p=1)
        cost = (
            float(getattr(self.seg_cfg, "matcher_class_cost", 1.0)) * class_cost
            + float(getattr(self.seg_cfg, "matcher_segment_cost", 5.0)) * segment_cost
        )
        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
        return [(int(r), int(valid_idx[int(c)].item())) for r, c in zip(rows, cols)]

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
    ) -> Tuple[torch.Tensor, Dict[int, list]]:
        B, A, C = class_logits.shape
        target_classes = torch.zeros_like(class_logits)
        target_objectness = torch.zeros_like(objectness_logits)
        matched_segment_pred = []
        matched_segment_target = []
        matches_by_batch: Dict[int, list] = {}

        for b in range(B):
            matches = self._match_single(
                class_logits[b],
                segment_preds[b],
                class_labels[b],
                segment_labels[b],
                segment_mask[b],
            )
            matches_by_batch[b] = matches
            for anchor_idx, seg_idx in matches:
                cls = int(class_labels[b, seg_idx].item())
                if 0 <= cls < C:
                    target_classes[b, anchor_idx, cls] = 1.0
                    target_objectness[b, anchor_idx] = 1.0
                    matched_segment_pred.append(segment_preds[b, anchor_idx])
                    matched_segment_target.append(segment_labels[b, seg_idx])

        class_mask = anchor_mask.float().unsqueeze(-1) * child_mask[:, :C].float().unsqueeze(1)
        class_loss = F.binary_cross_entropy_with_logits(class_logits, target_classes, reduction="none")
        class_loss = (class_loss * class_mask).sum()
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness_logits, target_objectness, reduction="none",
        )
        objectness_loss = (objectness_loss * anchor_mask.float()).sum()

        if matched_segment_pred:
            segment_loss = F.l1_loss(
                torch.stack(matched_segment_pred),
                torch.stack(matched_segment_target).to(segment_preds.device),
                reduction="sum",
            )
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

        class_logits, segment_preds, objectness_logits, anchors, anchor_mask = self._compute_segmentation(
            flat_inputs,
            count=batch.get("audio_segmentation_count"),
            threshold=batch.get("threshold", 0.5),
        )
        prototypes = self._prototype_masks(flat_inputs.words_embedding)
        coefficients = self.coeff_head(anchors)
        mask_logits = torch.einsum("bpt,bap->bat", prototypes, coefficients)

        loss = None
        matches = None
        if class_labels is not None and segment_labels is not None and segment_mask is not None:
            seg_loss, matches = self._segmentation_loss(
                class_logits, segment_preds, objectness_logits, anchor_mask,
                flat_inputs.child_mask, class_labels, segment_labels, segment_mask,
            )
            mask_loss = class_logits.new_tensor(0.0)
            if mask_labels is not None:
                matched_pred = []
                matched_target = []
                for b, pairs in (matches or {}).items():
                    for anchor_idx, seg_idx in pairs:
                        matched_pred.append(mask_logits[b, anchor_idx])
                        matched_target.append(mask_labels[b, seg_idx])
                if matched_pred:
                    mask_loss = F.binary_cross_entropy_with_logits(
                        torch.stack(matched_pred),
                        torch.stack(matched_target).to(mask_logits.device),
                        reduction="sum",
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
