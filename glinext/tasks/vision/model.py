"""Vision task heads: image classification, object detection, segmentation."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from gliner.modeling.loss_functions import focal_loss_with_logits

from .. import TaskHead, TaskHeadOutput
from ..matcher import HungarianMatcher
from ...layers import Pooling
from ...layers.mlp import create_mlp
from ..classification.scorer import ClassificationScorer


def _flat_features(flat_inputs):
    features = getattr(flat_inputs, "feature_embedding", None)
    mask = getattr(flat_inputs, "feature_mask", None)
    return (
        features if features is not None else flat_inputs.words_embedding,
        mask if mask is not None else flat_inputs.mask,
    )


def _xyxy_from_raw(raw: torch.Tensor) -> torch.Tensor:
    coords = raw.sigmoid()
    x1 = torch.minimum(coords[..., 0], coords[..., 2])
    y1 = torch.minimum(coords[..., 1], coords[..., 3])
    x2 = torch.maximum(coords[..., 0], coords[..., 2])
    y2 = torch.maximum(coords[..., 1], coords[..., 3])
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _matched_mask_loss(mask_logits: torch.Tensor, mask_labels: torch.Tensor, matches) -> torch.Tensor:
    matched_pred = []
    matched_target = []
    for b, pairs in (matches or {}).items():
        for anchor_idx, obj_idx in pairs:
            matched_pred.append(mask_logits[b, anchor_idx])
            matched_target.append(mask_labels[b, obj_idx])
    if not matched_pred:
        return mask_logits.new_tensor(0.0)

    loss = F.binary_cross_entropy_with_logits(
        torch.stack(matched_pred),
        torch.stack(matched_target).to(mask_logits.device),
        reduction="none",
    )
    return loss.flatten(1).mean(dim=1).sum()


class ImageClassificationHead(TaskHead):
    name = "image_classification"
    dependencies = []

    def __init__(self, config, hidden_size, dropout, shared_layers=None):
        super().__init__()
        cfg = config.image_classification_config
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
        if config.image_classification_config is None:
            return None
        return cls(config, hidden_size=config.hidden_size, dropout=config.dropout,
                   shared_layers=shared_layers)

    def forward(self, shared, dependency_outputs, flat_inputs=None, **batch):
        labels = batch.get("image_classification_labels")
        base_loss_fn = batch.get("base_loss_fn")
        image_features, image_mask = _flat_features(flat_inputs)
        image_rep = self.pooling(image_features, image_mask)
        anchors, anchor_mask = self.anchor_layer(
            flat_inputs.parent_embedding,
            image_features,
            feature_mask=image_mask,
        )
        if hasattr(self, "anchor_refine"):
            anchors = self.anchor_refine(anchors, image_features, token_mask=image_mask)
        fused = self.anchor_modeling(anchors, flat_inputs.child_embedding).squeeze(1)
        if fused.dim() == 4:
            anchor_weights = anchor_mask.float()
            fused = (fused * anchor_weights[:, :, None, None]).sum(dim=1)
            fused = fused / anchor_weights.sum(dim=1).clamp(min=1)[:, None, None]
        logits = self.scorer(image_rep, fused)

        loss = None
        if labels is not None:
            loss_fn = base_loss_fn or focal_loss_with_logits
            min_c = min(logits.shape[1], labels.shape[1])
            losses = loss_fn(logits[:, :min_c], labels[:, :min_c])
            loss = (losses * flat_inputs.child_mask[:, :min_c].float()).sum()
        return TaskHeadOutput(loss=loss, logits=logits)


class ObjectDetectionHead(TaskHead):
    """Anchor-slot object detector over visual tokens.

    Class predictions use the same anchor + child-label fusion as the text
    heads. Box and objectness predictions are per anchor slot.
    """

    name = "object_detection"
    dependencies = []

    def __init__(self, config, hidden_size, dropout, shared_layers=None):
        super().__init__()
        cfg = config.object_detection_config
        self.det_cfg = cfg
        self.loss_coef = cfg.loss_coef
        if shared_layers is None:
            shared_layers = {}
        self.matcher = HungarianMatcher(
            cost_class=getattr(cfg, "matcher_class_cost", 1.0),
            cost_geometry=getattr(cfg, "matcher_bbox_cost", 5.0),
        )
        self._init_anchor_pipeline(cfg, config, hidden_size, dropout, shared_layers)
        self.class_head = nn.Linear(hidden_size, 1)
        self.bbox_head = create_mlp(
            input_dim=hidden_size,
            intermediate_dims=[hidden_size],
            output_dim=4,
            dropout=dropout,
            activation="gelu",
        )
        self.objectness_head = nn.Linear(hidden_size, 1)

    @classmethod
    def from_config(cls, config, shared_layers=None, **kwargs):
        if config.object_detection_config is None:
            return None
        return cls(config, hidden_size=config.hidden_size, dropout=config.dropout,
                   shared_layers=shared_layers)

    def _compute_detection(self, flat_inputs, count=None, threshold=0.5):
        image_features, image_mask = _flat_features(flat_inputs)
        anchors, anchor_mask = self.anchor_layer(
            flat_inputs.parent_embedding,
            image_features,
            count=count,
            threshold=threshold,
            feature_mask=image_mask,
        )
        if hasattr(self, "anchor_refine"):
            anchors = self.anchor_refine(anchors, image_features, token_mask=image_mask)

        fused = self.anchor_modeling(anchors, flat_inputs.child_embedding)
        class_logits = self.class_head(fused).squeeze(-1)
        bbox_preds = _xyxy_from_raw(self.bbox_head(anchors))
        objectness_logits = self.objectness_head(anchors).squeeze(-1)
        return class_logits, bbox_preds, objectness_logits, anchors, anchor_mask

    def _match_single(self, class_logits, bbox_preds, gold_classes, gold_boxes, valid_objects):
        return self.matcher(class_logits, bbox_preds, gold_classes, gold_boxes, valid_objects)

    def _detection_loss(
        self,
        class_logits,
        bbox_preds,
        objectness_logits,
        anchor_mask,
        child_mask,
        class_labels,
        bbox_labels,
        object_mask,
    ) -> Tuple[torch.Tensor, Dict[int, list]]:
        B, A, C = class_logits.shape
        target_classes = torch.zeros_like(class_logits)
        target_objectness = torch.zeros_like(objectness_logits)
        matched_bbox_pred = []
        matched_bbox_target = []
        matches_by_batch: Dict[int, list] = {}

        for b in range(B):
            matches = self._match_single(
                class_logits[b],
                bbox_preds[b],
                class_labels[b],
                bbox_labels[b],
                object_mask[b],
            )
            matches_by_batch[b] = matches
            for anchor_idx, obj_idx in matches:
                cls = int(class_labels[b, obj_idx].item())
                if 0 <= cls < C:
                    target_classes[b, anchor_idx, cls] = 1.0
                    target_objectness[b, anchor_idx] = 1.0
                    matched_bbox_pred.append(bbox_preds[b, anchor_idx])
                    matched_bbox_target.append(bbox_labels[b, obj_idx])

        class_mask = anchor_mask.float().unsqueeze(-1) * child_mask[:, :C].float().unsqueeze(1)
        class_loss = F.binary_cross_entropy_with_logits(
            class_logits, target_classes, reduction="none",
        )
        class_loss = (class_loss * class_mask).sum()
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness_logits, target_objectness, reduction="none",
        )
        objectness_loss = (objectness_loss * anchor_mask.float()).sum()

        if matched_bbox_pred:
            bbox_loss = F.l1_loss(
                torch.stack(matched_bbox_pred),
                torch.stack(matched_bbox_target).to(bbox_preds.device),
                reduction="sum",
            )
        else:
            bbox_loss = class_logits.new_tensor(0.0)

        loss = (
            float(getattr(self.det_cfg, "class_loss_coef", 1.0)) * class_loss
            + float(getattr(self.det_cfg, "bbox_loss_coef", 5.0)) * bbox_loss
            + float(getattr(self.det_cfg, "objectness_loss_coef", 1.0)) * objectness_loss
        )
        return loss, matches_by_batch

    def forward(self, shared, dependency_outputs, flat_inputs=None, **batch):
        class_labels = batch.get("object_detection_class_labels")
        bbox_labels = batch.get("object_detection_bbox_labels")
        object_mask = batch.get("object_detection_object_mask")
        class_logits, bbox_preds, objectness_logits, anchors, anchor_mask = self._compute_detection(
            flat_inputs,
            count=batch.get("object_detection_count"),
            threshold=batch.get("threshold", 0.5),
        )

        loss = None
        matches = None
        if class_labels is not None and bbox_labels is not None and object_mask is not None:
            loss, matches = self._detection_loss(
                class_logits, bbox_preds, objectness_logits, anchor_mask,
                flat_inputs.child_mask, class_labels, bbox_labels, object_mask,
            )

        return TaskHeadOutput(
            loss=loss,
            logits=class_logits,
            extra={
                "bbox_preds": bbox_preds,
                "objectness_logits": objectness_logits,
                "anchor_mask": anchor_mask,
                "anchors": anchors,
                "matches": matches,
            },
        )


class SegmentationHead(ObjectDetectionHead):
    """Detection head with prototype-mask segmentation outputs."""

    name = "segmentation"
    dependencies = []

    def __init__(self, config, hidden_size, dropout, shared_layers=None):
        nn.Module.__init__(self)
        cfg = config.segmentation_config
        self.det_cfg = cfg
        self.loss_coef = cfg.loss_coef
        if shared_layers is None:
            shared_layers = {}
        self.matcher = HungarianMatcher(
            cost_class=getattr(cfg, "matcher_class_cost", 1.0),
            cost_geometry=getattr(cfg, "matcher_bbox_cost", 5.0),
        )
        self._init_anchor_pipeline(cfg, config, hidden_size, dropout, shared_layers)
        self.class_head = nn.Linear(hidden_size, 1)
        self.bbox_head = create_mlp(
            input_dim=hidden_size,
            intermediate_dims=[hidden_size],
            output_dim=4,
            dropout=dropout,
            activation="gelu",
        )
        self.objectness_head = nn.Linear(hidden_size, 1)
        self.num_prototypes = int(getattr(cfg, "num_prototypes", 32))
        self.mask_size = int(getattr(cfg, "mask_size", 128))
        self.proto_proj = nn.Linear(hidden_size, self.num_prototypes)
        self.proto_refine = nn.Sequential(
            nn.Conv2d(self.num_prototypes, self.num_prototypes, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.num_prototypes, self.num_prototypes, kernel_size=1),
        )
        self.coeff_head = nn.Linear(hidden_size, self.num_prototypes)

    @classmethod
    def from_config(cls, config, shared_layers=None, **kwargs):
        if config.segmentation_config is None:
            return None
        return cls(config, hidden_size=config.hidden_size, dropout=config.dropout,
                   shared_layers=shared_layers)

    @staticmethod
    def _grid_shape(num_tokens: int) -> Tuple[int, int]:
        side = int(num_tokens ** 0.5)
        if side * side == num_tokens:
            return side, side
        return 1, num_tokens

    def _prototype_masks(self, vision_tokens):
        B, V, _ = vision_tokens.shape
        h, w = self._grid_shape(V)
        proto = self.proto_proj(vision_tokens).transpose(1, 2).reshape(B, self.num_prototypes, h, w)
        proto = self.proto_refine(proto)
        return F.interpolate(proto, size=(self.mask_size, self.mask_size), mode="bilinear", align_corners=False)

    def forward(self, shared, dependency_outputs, flat_inputs=None, **batch):
        class_labels = batch.get("segmentation_class_labels")
        bbox_labels = batch.get("segmentation_bbox_labels")
        object_mask = batch.get("segmentation_object_mask")
        mask_labels = batch.get("segmentation_mask_labels")

        class_logits, bbox_preds, objectness_logits, anchors, anchor_mask = self._compute_detection(
            flat_inputs,
            count=batch.get("segmentation_count"),
            threshold=batch.get("threshold", 0.5),
        )
        image_features, _ = _flat_features(flat_inputs)
        prototypes = self._prototype_masks(image_features)
        coefficients = self.coeff_head(anchors)
        mask_logits = torch.einsum("bphw,bap->bahw", prototypes, coefficients)

        loss = None
        matches = None
        if class_labels is not None and bbox_labels is not None and object_mask is not None:
            det_loss, matches = self._detection_loss(
                class_logits, bbox_preds, objectness_logits, anchor_mask,
                flat_inputs.child_mask, class_labels, bbox_labels, object_mask,
            )
            mask_loss = class_logits.new_tensor(0.0)
            if mask_labels is not None:
                mask_loss = _matched_mask_loss(mask_logits, mask_labels, matches)
            loss = det_loss + float(getattr(self.det_cfg, "mask_loss_coef", 1.0)) * mask_loss

        return TaskHeadOutput(
            loss=loss,
            logits=class_logits,
            extra={
                "bbox_preds": bbox_preds,
                "objectness_logits": objectness_logits,
                "anchor_mask": anchor_mask,
                "anchors": anchors,
                "matches": matches,
                "mask_logits": mask_logits,
                "prototypes": prototypes,
                "coefficients": coefficients,
            },
        )
