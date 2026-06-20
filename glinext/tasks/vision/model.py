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
    """Decode raw (cx, cy, w, h) logits to xyxy in [0,1].

    Using cxcywh avoids sigmoid saturation: cx/cy priors land in logit [-3, +3]
    (gradient ≥ 0.045) vs xyxy corner priors at ±4.6 (gradient ≈ 0.01). The
    bbox_head therefore receives healthy gradients for all anchor slots.
    """
    cxcywh = raw.sigmoid()
    cx = cxcywh[..., 0]
    cy = cxcywh[..., 1]
    w = cxcywh[..., 2]
    h = cxcywh[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _bbox_prior_grid(num_slots: int, margin: float = 0.1) -> torch.Tensor:
    """Return (num_slots, 4) logit-space cxcywh priors laid out on a spatial grid.

    Each slot starts predicting its grid-cell center and cell size. Using the
    cxcywh parameterization keeps all logit priors in [-3, +3], so the sigmoid
    gradient is ≥ 0.045 for every slot — roughly 25× stronger than xyxy corner
    priors which would sit at logit ±4.6 (gradient ≈ 0.01).

    margin controls the object-size prior: cell width = (1−2·margin)/cols.
    The bias is a frozen buffer so bbox_head learns pure cxcywh offsets.
    """
    cols = max(1, int(num_slots ** 0.5))
    rows = (num_slots + cols - 1) // cols
    priors = []
    for r in range(rows):
        for c in range(cols):
            if len(priors) == num_slots:
                break
            cx = (c + 0.5) / cols
            cy = (r + 0.5) / rows
            w = (1.0 - 2.0 * margin) / cols
            h = (1.0 - 2.0 * margin) / rows
            priors.append([cx, cy, w, h])
    t = torch.tensor(priors, dtype=torch.float).clamp(1e-4, 1 - 1e-4)
    return torch.log(t / (1 - t))  # inverse sigmoid → logit space


def _canonical_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x1 = torch.minimum(boxes[..., 0], boxes[..., 2])
    y1 = torch.minimum(boxes[..., 1], boxes[..., 3])
    x2 = torch.maximum(boxes[..., 0], boxes[..., 2])
    y2 = torch.maximum(boxes[..., 1], boxes[..., 3])
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _generalized_box_iou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    pred_boxes = _canonical_xyxy(pred_boxes)
    target_boxes = _canonical_xyxy(target_boxes)

    inter_x1 = torch.maximum(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.maximum(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.minimum(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.minimum(pred_boxes[:, 3], target_boxes[:, 3])
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0) * (
        pred_boxes[:, 3] - pred_boxes[:, 1]
    ).clamp(min=0)
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=0) * (
        target_boxes[:, 3] - target_boxes[:, 1]
    ).clamp(min=0)
    union = pred_area + target_area - inter_area
    iou = inter_area / union.clamp(min=1e-6)

    enc_x1 = torch.minimum(pred_boxes[:, 0], target_boxes[:, 0])
    enc_y1 = torch.minimum(pred_boxes[:, 1], target_boxes[:, 1])
    enc_x2 = torch.maximum(pred_boxes[:, 2], target_boxes[:, 2])
    enc_y2 = torch.maximum(pred_boxes[:, 3], target_boxes[:, 3])
    enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0)
    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
    return (1.0 - giou).sum()


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


def _binary_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    base_loss_fn=None,
) -> torch.Tensor:
    if base_loss_fn is not None:
        return base_loss_fn(logits, targets)
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="none")


def _drop_leading_cls_if_grid(features: torch.Tensor, mask: Optional[torch.Tensor] = None):
    if features is None or features.shape[1] <= 1:
        return features, mask
    patch_count = features.shape[1] - 1
    side = int(patch_count ** 0.5)
    if side * side != patch_count:
        return features, mask
    features = features[:, 1:]
    if mask is not None:
        mask = mask[:, 1:]
    return features, mask


def _patch_xyxy_coordinates(num_tokens: int, device: torch.device) -> Optional[torch.Tensor]:
    side = int(num_tokens ** 0.5)
    if side * side != num_tokens:
        return None
    y, x = torch.meshgrid(
        torch.arange(side, device=device),
        torch.arange(side, device=device),
        indexing="ij",
    )
    coords = torch.stack(
        [
            x.float() / side,
            y.float() / side,
            (x.float() + 1.0) / side,
            (y.float() + 1.0) / side,
        ],
        dim=-1,
    )
    return coords.reshape(num_tokens, 4)


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
        self.scorer = ClassificationScorer.from_config(
            scorer_type=getattr(cfg, "scorer_type", "dot"),
            hidden_size=hidden_size,
        )
        self.bbox_head = create_mlp(
            input_dim=hidden_size,
            intermediate_dims=[hidden_size],
            output_dim=4,
            dropout=dropout,
            activation="gelu",
        )
        self.objectness_head = nn.Linear(hidden_size, 1)
        self.class_conditioned_bbox = bool(getattr(cfg, "class_conditioned_bbox", True))
        self.drop_cls_token_for_dense = bool(getattr(cfg, "drop_cls_token_for_dense", True))
        self.dense_coord_features = bool(getattr(cfg, "dense_coord_features", True))
        self.coord_proj = nn.Linear(4, hidden_size) if self.dense_coord_features else None
        anchor_mode = getattr(cfg, "anchor_mode", "fixed")
        num_slots = getattr(cfg, "num_fixed_slots", 100)
        if anchor_mode == "fixed" and getattr(cfg, "bbox_prior_grid", False):
            margin = float(getattr(cfg, "bbox_prior_margin", 0.1))
            self.register_buffer("bbox_bias", _bbox_prior_grid(num_slots, margin=margin))
        else:
            self.bbox_bias = None
        # Learnable slot positional embeddings — added to Q and K at every
        # anchor_refine layer to prevent self-attention collapse across slots.
        self.slot_pos_embed = nn.Embedding(num_slots, hidden_size)

    @classmethod
    def from_config(cls, config, shared_layers=None, **kwargs):
        if config.object_detection_config is None:
            return None
        return cls(config, hidden_size=config.hidden_size, dropout=config.dropout,
                   shared_layers=shared_layers)

    def _score_anchor_labels(self, anchors: torch.Tensor, label_reps: torch.Tensor, label_mask: torch.Tensor):
        B, A, D = anchors.shape
        C = label_reps.shape[1]
        anchor_flat = anchors.reshape(B * A, D)
        label_flat = label_reps[:, None].expand(B, A, C, D).reshape(B * A, C, D)
        logits = self.scorer(anchor_flat, label_flat).reshape(B, A, C)
        if label_mask is not None:
            invalid = ~label_mask.to(device=logits.device).bool()
            logits = logits.masked_fill(invalid[:, None, :], -1e4)
        return logits

    def _dense_features(self, flat_inputs):
        image_features, image_mask = _flat_features(flat_inputs)
        if self.drop_cls_token_for_dense:
            image_features, image_mask = _drop_leading_cls_if_grid(image_features, image_mask)
        if self.coord_proj is not None and image_features is not None:
            coords = _patch_xyxy_coordinates(image_features.shape[1], image_features.device)
            if coords is not None:
                coord_emb = self.coord_proj(coords.to(dtype=self.coord_proj.weight.dtype))
                image_features = image_features + coord_emb.to(dtype=image_features.dtype).unsqueeze(0)
        return image_features, image_mask

    def _compute_detection(self, flat_inputs, count=None, threshold=0.5):
        image_features, image_mask = self._dense_features(flat_inputs)
        anchors, anchor_mask = self.anchor_layer(
            flat_inputs.parent_embedding,
            image_features,
            count=count,
            threshold=threshold,
            feature_mask=image_mask,
        )
        if hasattr(self, "anchor_refine"):
            A = anchors.shape[1]
            slot_ids = torch.arange(A, device=anchors.device)
            slot_pos = self.slot_pos_embed(slot_ids).unsqueeze(0)  # (1, A, D)
            anchors = self.anchor_refine(anchors, image_features, token_mask=image_mask,
                                         query_pos_emb=slot_pos)

        class_logits = self._score_anchor_labels(
            anchors,
            flat_inputs.child_embedding,
            flat_inputs.child_mask,
        )
        A = anchors.shape[1]
        if self.class_conditioned_bbox:
            bbox_features = self.anchor_modeling(anchors, flat_inputs.child_embedding)
            bbox_raw = self.bbox_head(bbox_features)
        else:
            bbox_raw = self.bbox_head(anchors)
        if self.bbox_bias is not None:
            bias = self.bbox_bias[:A].unsqueeze(0)
            if bbox_raw.dim() == 4:
                bias = bias.unsqueeze(2)
            bbox_raw = bbox_raw + bias
        bbox_preds = _xyxy_from_raw(bbox_raw)
        objectness_logits = self.objectness_head(anchors).squeeze(-1)
        return class_logits, bbox_preds, objectness_logits, anchors, anchor_mask

    def _match_single(
        self,
        class_logits,
        bbox_preds,
        gold_classes,
        gold_boxes,
        valid_objects,
        valid_predictions=None,
    ):
        return self.matcher(
            class_logits,
            bbox_preds,
            gold_classes,
            gold_boxes,
            valid_objects,
            prediction_mask=valid_predictions,
        )

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
        base_loss_fn=None,
    ) -> Tuple[torch.Tensor, Dict[int, list]]:
        B, A, C = class_logits.shape
        target_objectness = torch.zeros_like(objectness_logits)
        matched_class_logits = []
        matched_class_targets = []
        matched_child_masks = []
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
                anchor_mask[b],
            )
            matches_by_batch[b] = matches
            for anchor_idx, obj_idx in matches:
                cls = int(class_labels[b, obj_idx].item())
                if 0 <= cls < C and bool(child_mask[b, cls].item()):
                    target_objectness[b, anchor_idx] = 1.0
                    matched_class_logits.append(class_logits[b, anchor_idx])
                    matched_class_targets.append(cls)
                    matched_child_masks.append(child_mask[b])
                    if bbox_preds.dim() == 4:
                        matched_bbox_pred.append(bbox_preds[b, anchor_idx, cls])
                    else:
                        matched_bbox_pred.append(bbox_preds[b, anchor_idx])
                    matched_bbox_target.append(bbox_labels[b, obj_idx])

        normalizer = target_objectness.sum().clamp(min=1.0)
        if matched_class_logits:
            matched_logits = torch.stack(matched_class_logits)
            matched_valid_labels = torch.stack(matched_child_masks).to(device=matched_logits.device).bool()
            matched_logits = matched_logits.masked_fill(~matched_valid_labels, -1e4)
            class_loss = F.cross_entropy(
                matched_logits,
                torch.as_tensor(
                    matched_class_targets,
                    dtype=torch.long,
                    device=class_logits.device,
                ),
                reduction="mean",
            )
        else:
            class_loss = class_logits.new_tensor(0.0)

        objectness_element_loss = _binary_loss_with_logits(
            objectness_logits,
            target_objectness,
            base_loss_fn,
        )
        objectness_mask = anchor_mask.float()
        # Normalize over all valid anchor slots. The base_loss_fn (focal loss with
        # alpha=0.75) already handles the foreground/background imbalance; no
        # additional manual positive_weight is applied to avoid double-weighting.
        objectness_loss = (objectness_element_loss * objectness_mask).sum() / (
            objectness_mask.sum().clamp(min=1.0)
        )

        if matched_bbox_pred:
            pred_boxes = torch.stack(matched_bbox_pred)
            target_boxes = torch.stack(matched_bbox_target).to(bbox_preds.device)
            bbox_l1_loss = F.l1_loss(pred_boxes, target_boxes, reduction="sum")
            bbox_giou_loss = _generalized_box_iou_loss(pred_boxes, target_boxes)
            bbox_loss = (bbox_l1_loss + bbox_giou_loss) / normalizer
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
                base_loss_fn=batch.get("base_loss_fn"),
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
        self.scorer = ClassificationScorer.from_config(
            scorer_type=getattr(cfg, "scorer_type", "dot"),
            hidden_size=hidden_size,
        )
        self.bbox_head = create_mlp(
            input_dim=hidden_size,
            intermediate_dims=[hidden_size],
            output_dim=4,
            dropout=dropout,
            activation="gelu",
        )
        self.objectness_head = nn.Linear(hidden_size, 1)
        self.class_conditioned_bbox = bool(getattr(cfg, "class_conditioned_bbox", True))
        self.drop_cls_token_for_dense = bool(getattr(cfg, "drop_cls_token_for_dense", True))
        self.dense_coord_features = bool(getattr(cfg, "dense_coord_features", True))
        self.coord_proj = nn.Linear(4, hidden_size) if self.dense_coord_features else None
        anchor_mode = getattr(cfg, "anchor_mode", "fixed")
        num_slots = getattr(cfg, "num_fixed_slots", 100)
        if anchor_mode == "fixed" and getattr(cfg, "bbox_prior_grid", False):
            margin = float(getattr(cfg, "bbox_prior_margin", 0.1))
            self.register_buffer("bbox_bias", _bbox_prior_grid(num_slots, margin=margin))
        else:
            self.bbox_bias = None
        self.slot_pos_embed = nn.Embedding(num_slots, hidden_size)
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
        image_features, _ = self._dense_features(flat_inputs)
        prototypes = self._prototype_masks(image_features)
        coefficients = self.coeff_head(anchors)
        mask_logits = torch.einsum("bphw,bap->bahw", prototypes, coefficients)

        loss = None
        matches = None
        if class_labels is not None and bbox_labels is not None and object_mask is not None:
            det_loss, matches = self._detection_loss(
                class_logits, bbox_preds, objectness_logits, anchor_mask,
                flat_inputs.child_mask, class_labels, bbox_labels, object_mask,
                base_loss_fn=batch.get("base_loss_fn"),
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
