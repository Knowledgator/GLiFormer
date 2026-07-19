"""Vision task heads: image classification, object detection, segmentation."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from gliner.modeling.loss_functions import focal_loss_with_logits

from .. import TaskHead, TaskHeadOutput
from ..matcher import HungarianMatcher
from ...layers import AnchorCrossAttentionLayer, AnchorLayer, Pooling
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
    """Decode sigmoid ``cxcywh`` parameters without clipping the corners.

    The center and size remain bounded, but a box near an image edge may have a
    corner outside ``[0, 1]``. Keeping that corner differentiable is important:
    clamping here gives it a zero gradient and leaves boxes stuck on image
    boundaries. Public decoders clip the final inference result instead.
    """
    coords = raw.sigmoid()
    cx, cy, w, h = coords.unbind(dim=-1)
    half_w = 0.5 * w
    half_h = 0.5 * h
    x1 = cx - half_w
    y1 = cy - half_h
    x2 = cx + half_w
    y2 = cy + half_h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(box1[..., 0], box2[..., 0])
    y1 = torch.maximum(box1[..., 1], box2[..., 1])
    x2 = torch.minimum(box1[..., 2], box2[..., 2])
    y2 = torch.minimum(box1[..., 3], box2[..., 3])
    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    box1_area = (box1[..., 2] - box1[..., 0]).clamp(min=0) * (box1[..., 3] - box1[..., 1]).clamp(min=0)
    box2_area = (box2[..., 2] - box2[..., 0]).clamp(min=0) * (box2[..., 3] - box2[..., 1]).clamp(min=0)
    return inter_area / (box1_area + box2_area - inter_area + 1e-6)


def bbox_giou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Generalized IoU (Rezatofighi et al.). Unlike plain IoU it stays
    differentiable when the boxes do not overlap — the enclosing-box penalty
    keeps pulling them together. This matters for the small objects, whose
    predicted boxes rarely overlap the target early in training, leaving plain
    IoU with a zero gradient and only the scale-sensitive L1 term to localize."""
    x1 = torch.maximum(box1[..., 0], box2[..., 0])
    y1 = torch.maximum(box1[..., 1], box2[..., 1])
    x2 = torch.minimum(box1[..., 2], box2[..., 2])
    y2 = torch.minimum(box1[..., 3], box2[..., 3])
    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    box1_area = (box1[..., 2] - box1[..., 0]).clamp(min=0) * (box1[..., 3] - box1[..., 1]).clamp(min=0)
    box2_area = (box2[..., 2] - box2[..., 0]).clamp(min=0) * (box2[..., 3] - box2[..., 1]).clamp(min=0)
    union = box1_area + box2_area - inter_area + 1e-6
    iou = inter_area / union
    # Smallest axis-aligned box enclosing both.
    cx1 = torch.minimum(box1[..., 0], box2[..., 0])
    cy1 = torch.minimum(box1[..., 1], box2[..., 1])
    cx2 = torch.maximum(box1[..., 2], box2[..., 2])
    cy2 = torch.maximum(box1[..., 3], box2[..., 3])
    enclosing = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + 1e-6
    return iou - (enclosing - union) / enclosing


def _inverse_sigmoid(value: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    value = value.clamp(min=eps, max=1.0 - eps)
    return torch.log(value / (1.0 - value))


def _sine_pos_2d(coords: torch.Tensor, dim: int, temperature: float = 10000.0) -> torch.Tensor:
    """2D sinusoidal positional encoding for normalized ``(x, y)`` in [0, 1].

    ``coords`` ``(..., 2)`` -> ``(..., dim)`` (dim divisible by 4). As in DETR,
    the dot product of two such encodings peaks when their positions coincide,
    so adding it to both the patch tokens and a slot's reference-point query
    makes the cross-attention spatially selective — the mechanism the plain
    learnable coord projection lacked.
    """
    half = dim // 2
    dim_t = torch.arange(half, device=coords.device, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / half)
    feats = []
    for axis in range(2):
        pos = coords[..., axis:axis + 1].float() * (2 * math.pi) / dim_t
        pos = torch.stack((pos[..., 0::2].sin(), pos[..., 1::2].cos()), dim=-1).flatten(-2)
        feats.append(pos)
    return torch.cat(feats, dim=-1)


def _bbox_prior_grid(num_slots: int, margin: float = 0.1) -> torch.Tensor:
    side = int(num_slots ** 0.5)
    cols = max(side, 1)
    rows = (num_slots + cols - 1) // cols
    ys, xs = torch.meshgrid(
        torch.arange(rows, dtype=torch.float),
        torch.arange(cols, dtype=torch.float),
        indexing="ij",
    )
    centers = torch.stack([(xs.flatten() + 0.5) / cols, (ys.flatten() + 0.5) / rows], dim=-1)
    centers = centers[:num_slots]
    cell_w = (1.0 - 2.0 * margin) / cols
    cell_h = (1.0 - 2.0 * margin) / rows
    centers = margin + centers * (1.0 - 2.0 * margin)
    sizes = centers.new_full((num_slots, 2), max(min(cell_w, cell_h), 1e-3))
    return _inverse_sigmoid(torch.cat([centers, sizes], dim=-1))


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
        fused = self._reduce_fused_anchors(
            self.anchor_modeling(anchors, flat_inputs.child_embedding),
            anchor_mask,
        )
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
            cost_giou=getattr(cfg, "matcher_giou_cost", 2.0),
        )
        self._init_detection_anchor_pipeline(cfg, hidden_size, dropout, shared_layers)
        self.cls_head = ClassificationScorer.from_config(
            scorer_type=getattr(cfg, "scorer_type", "dot"),
            hidden_size=hidden_size,
        )

        self.bbox_head = create_mlp(
            input_dim=hidden_size,
            intermediate_dims=[hidden_size//2],
            output_dim=4,
            dropout=dropout,
            activation="gelu",
        )
        self.objectness_head = create_mlp(
            input_dim=hidden_size,
            intermediate_dims=[hidden_size//2],
            output_dim=1,
            dropout=dropout,
            activation="gelu",
        )
        if getattr(cfg, "dense_coord_features", True):
            self.coord_proj = nn.Linear(2, hidden_size, bias=False)
        else:
            self.coord_proj = None
        if getattr(cfg, "bbox_prior_grid", False):
            self.register_buffer(
                "bbox_bias",
                _bbox_prior_grid(int(getattr(cfg, "num_fixed_slots", 100)), float(getattr(cfg, "bbox_prior_margin", 0.1))),
                persistent=False,
            )
        else:
            self.bbox_bias = None

    @classmethod
    def from_config(cls, config, shared_layers=None, **kwargs):
        if config.object_detection_config is None:
            return None
        return cls(config, hidden_size=config.hidden_size, dropout=config.dropout,
                   shared_layers=shared_layers)

    def _init_detection_anchor_pipeline(self, task_cfg, hidden_size, dropout, shared_layers):
        anchor_mode = getattr(task_cfg, "anchor_mode", "fixed")
        if anchor_mode == "rnn":
            anchor_mode = "rotary"
        self.anchor_mode = anchor_mode
        self.anchor_layer = AnchorLayer.from_config(
            anchor_mode,
            hidden_size,
            max_count=getattr(task_cfg, "max_count", 20),
            num_slots=getattr(task_cfg, "num_fixed_slots", 10),
            num_heads=getattr(task_cfg, "anchor_num_heads", 4),
            num_layers=getattr(task_cfg, "anchor_num_layers", 2),
            dropout=dropout,
            feature_mlp=getattr(task_cfg, "feature_anchor_mlp", False),
            feature_mlp_hidden_multiplier=getattr(task_cfg, "feature_anchor_mlp_hidden_multiplier", 1),
        )
        if "anchor_refine" in shared_layers:
            self.anchor_refine = shared_layers["anchor_refine"]
        else:
            refine_layers = getattr(task_cfg, "anchor_refine_layers", 0)
            if refine_layers > 0:
                self.anchor_refine = AnchorCrossAttentionLayer(
                    hidden_size,
                    num_heads=getattr(task_cfg, "anchor_refine_heads", 8),
                    num_layers=refine_layers,
                    dropout=dropout,
                )

        # DETR-style per-slot positional embeddings, re-added to the queries at
        # every refinement layer. Without them the fixed slots are interchangeable
        # (permutation-symmetric): they start near-identical and the refine
        # self-attention averages them into a single query under training, so every
        # anchor predicts the same class/box. These embeddings must be *strong* to
        # break the symmetry — a weak init (e.g. std 0.02, norm ~0.6) is swamped by
        # the anchor magnitude and training still collapses the slots (measured
        # after-refine cosine ~0.99). Initialize at unit variance (norm ~sqrt(D),
        # as in DETR's learned query embeddings); this keeps slots distinct under
        # training (cosine ~0.62) and reaches a lower loss.
        # Reference-point (DAB/Conditional-DETR) conditioning. Each slot owns a
        # learnable (cx, cy, w, h) reference in logit space, laid out on a grid so
        # the slots start spatially spread. The same reference both (a) supplies a
        # sinusoidal positional query that steers the slot's cross-attention to its
        # own image region and (b) biases the box regression as an offset. Sharing
        # one parameter couples attention and box: box-loss gradients move the
        # reference, which moves where the slot looks. This replaces the random
        # slot positional embedding (which broke symmetry but carried no spatial
        # meaning, so boxes collapsed to the dataset-mean box).
        self.use_reference_points = bool(getattr(task_cfg, "reference_points", False))
        self.pos_emb_scale = float(getattr(task_cfg, "pos_emb_scale", 1.0))
        self.reference_points = None
        self.slot_pos_emb = None
        if self.use_reference_points and anchor_mode in {"fixed", "fixed_rnn", "fixed_transformer"}:
            num_slots = int(getattr(task_cfg, "num_fixed_slots", 10))
            grid = _bbox_prior_grid(num_slots, float(getattr(task_cfg, "bbox_prior_margin", 0.1)))
            default_size = float(getattr(task_cfg, "default_box_size", 0.1))
            grid[:, 2:] = _inverse_sigmoid(torch.full_like(grid[:, 2:], default_size))
            self.reference_points = nn.Parameter(grid)
        elif hasattr(self, "anchor_refine") and anchor_mode in {"fixed", "fixed_rnn", "fixed_transformer"}:
            # Legacy path: per-slot positional embeddings, re-added to the queries
            # at every refinement layer to break the permutation symmetry of the
            # interchangeable fixed slots. Must be *strong* (norm ~sqrt(D), DETR
            # convention) or the refine self-attention averages the slots into one.
            self.slot_pos_emb = nn.Embedding(num_slots := int(getattr(task_cfg, "num_fixed_slots", 10)), hidden_size)
            nn.init.normal_(self.slot_pos_emb.weight, std=float(getattr(task_cfg, "slot_pos_emb_std", 1.0)))

    def _slot_query_pos(self, anchors: torch.Tensor) -> Optional[torch.Tensor]:
        if getattr(self, "use_reference_points", False) and self.reference_points is not None:
            if anchors.shape[1] != self.reference_points.shape[0]:
                return None
            centers = self.reference_points[:, :2].sigmoid()
            pe = self.pos_emb_scale * _sine_pos_2d(centers, anchors.shape[-1])
            return pe.unsqueeze(0).to(dtype=anchors.dtype, device=anchors.device)
        if self.slot_pos_emb is None or anchors.shape[1] != self.slot_pos_emb.num_embeddings:
            return None
        return self.slot_pos_emb.weight.unsqueeze(0).to(dtype=anchors.dtype, device=anchors.device)

    def _dense_features(self, flat_inputs):
        image_features, image_mask = _flat_features(flat_inputs)
        length = image_features.shape[1]
        if getattr(self.det_cfg, "drop_cls_token_for_dense", True) and length > 1:
            side = int(length ** 0.5)
            # ViT-style backbones prepend a CLS token, making the patch count
            # non-square (e.g. 197 = 1 + 14*14). Drop it so feature anchors and
            # the injected coordinate grid resolve to the true 2D patch layout
            # instead of collapsing to a degenerate 1xN row.
            cls_breaks_grid = side * side != length and int((length - 1) ** 0.5) ** 2 == length - 1
            if self.anchor_mode in {"features", "feature"} or cls_breaks_grid:
                image_features = image_features[:, 1:]
                image_mask = image_mask[:, 1:] if image_mask is not None else None

        use_sine = getattr(self, "use_reference_points", False)
        if (use_sine or self.coord_proj is not None) and image_features.shape[1] > 0:
            length = image_features.shape[1]
            side = int(length ** 0.5)
            if side * side == length:
                rows = cols = side
            else:
                rows, cols = 1, length
            yy, xx = torch.meshgrid(
                torch.arange(rows, device=image_features.device, dtype=image_features.dtype),
                torch.arange(cols, device=image_features.device, dtype=image_features.dtype),
                indexing="ij",
            )
            coords = torch.stack([(xx.flatten() + 0.5) / cols, (yy.flatten() + 0.5) / rows], dim=-1)[:length]
            if use_sine:
                # Sinusoidal patch PE in the same basis as the slot reference query
                # so the cross-attention dot product is spatially peaked.
                pe = self.pos_emb_scale * _sine_pos_2d(coords, image_features.shape[-1])
                image_features = image_features + pe.unsqueeze(0).to(image_features.dtype)
            else:
                image_features = image_features + self.coord_proj(coords).unsqueeze(0)
        return image_features, image_mask

    def _score_anchor_labels(self, anchors, label_reps, label_mask=None):
        B, A, D = anchors.shape
        _, C, _ = label_reps.shape
        flat_anchors = anchors.reshape(B * A, D)
        flat_labels = label_reps[:, None].expand(B, A, C, D).reshape(B * A, C, D)
        logits = self.cls_head(flat_anchors, flat_labels).reshape(B, A, C)
        if label_mask is not None:
            mask = label_mask[:, None, :C].to(device=logits.device).bool()
            logits = logits.masked_fill(~mask, -1e4)
        return logits

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
            anchors = self.anchor_refine(
                anchors,
                image_features,
                token_mask=image_mask,
                query_pos_emb=self._slot_query_pos(anchors),
            )

        class_logits = self._score_anchor_labels(anchors, flat_inputs.child_embedding, flat_inputs.child_mask)
        bbox_raw = self.bbox_head(anchors)
        if getattr(self, "use_reference_points", False) and self.reference_points is not None \
                and bbox_raw.shape[1] == self.reference_points.shape[0]:
            # Box = sigmoid(delta + reference): the head predicts an offset from
            # the slot's learnable (cx, cy, w, h) reference, anchoring each slot to
            # its own region/scale so they cannot collapse to one mean box.
            bbox_raw = bbox_raw + self.reference_points.unsqueeze(0).to(device=bbox_raw.device, dtype=bbox_raw.dtype)
        elif self.bbox_bias is not None and bbox_raw.shape[1] <= self.bbox_bias.shape[0]:
            bbox_raw = bbox_raw + self.bbox_bias[:bbox_raw.shape[1]].to(device=bbox_raw.device, dtype=bbox_raw.dtype)
        bbox_preds = _xyxy_from_raw(bbox_raw)
        objectness_logits = self.objectness_head(anchors).squeeze(-1)
        return class_logits, bbox_preds, objectness_logits, anchors, anchor_mask

    def _match_single(self, class_logits, bbox_preds, gold_classes, gold_boxes, valid_objects, prediction_mask=None):
        return self.matcher(class_logits, bbox_preds, gold_classes, gold_boxes, valid_objects, prediction_mask)

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
        device = class_logits.device
        target_objectness = torch.zeros_like(objectness_logits)
        matches_by_batch: Dict[int, list] = {}

        # Flat (batch, anchor, target) indices of all matched pairs.
        src_batch_idx: list = []
        src_anchor_idx: list = []
        tgt_obj_idx: list = []
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
            if matches:
                src_idx = torch.tensor([a for a, _ in matches], dtype=torch.long, device=device)
                target_objectness[b, src_idx] = 1.0
                for anchor_idx, obj_idx in matches:
                    src_batch_idx.append(b)
                    src_anchor_idx.append(anchor_idx)
                    tgt_obj_idx.append(obj_idx)

        # DETR-style normalization: divide by the number of matched objects so the
        # objectness signal is not diluted by the ~num_anchors background slots.
        num_matched = max(len(src_batch_idx), 1)
        loss_fn = base_loss_fn or focal_loss_with_logits

        if src_batch_idx:
            sb = torch.tensor(src_batch_idx, dtype=torch.long, device=device)
            sa = torch.tensor(src_anchor_idx, dtype=torch.long, device=device)
            tt = torch.tensor(tgt_obj_idx, dtype=torch.long, device=device)

            # Classification: softmax cross-entropy over matched anchors. Per-anchor
            # multi-label sigmoid-focal over every class is numerically unstable here
            # — the shared dot-product scorer ties correlated label embeddings, so
            # pushing the GT class up drags neighbours up into false positives and the
            # negative focal mass explodes. Softmax couples the classes, so raising the
            # GT logit lowers the rest, and the loss stays well-conditioned.
            matched_logits = class_logits[sb, sa]  # (M, C)
            label_mask = child_mask[sb, :C].to(device=device).bool()  # (M, C)
            matched_logits = matched_logits.masked_fill(~label_mask, -1e4)
            gt_classes = class_labels[sb, tt].long()
            valid = (gt_classes >= 0) & (gt_classes < C)
            if valid.any():
                class_loss = F.cross_entropy(matched_logits[valid], gt_classes[valid])
            else:
                class_loss = class_logits.new_tensor(0.0)

            matched_bbox_pred = bbox_preds[sb, sa]
            matched_bbox_target = bbox_labels[sb, tt].to(device=device, dtype=bbox_preds.dtype)
            bbox_loss = F.l1_loss(matched_bbox_pred, matched_bbox_target, reduction="sum") / num_matched
            iou_loss = (1.0 - bbox_giou(matched_bbox_pred, matched_bbox_target)).sum() / num_matched
        else:
            class_loss = class_logits.new_tensor(0.0)
            bbox_loss = class_logits.new_tensor(0.0)
            iou_loss = class_logits.new_tensor(0.0)

        # Objectness: binary foreground/background over all anchor slots, with matched
        # slots up-weighted, normalized by matched objects.
        objectness_weight = anchor_mask.float()
        pos_weight = float(getattr(self.det_cfg, "objectness_positive_weight", 1.0))
        if pos_weight != 1.0:
            objectness_weight = objectness_weight * torch.where(
                target_objectness > 0,
                objectness_weight.new_tensor(pos_weight),
                objectness_weight.new_tensor(1.0),
            )
        objectness_loss = (
            loss_fn(objectness_logits, target_objectness) * objectness_weight
        ).sum() / num_matched

        loss = (
            float(getattr(self.det_cfg, "class_loss_coef", 1.0)) * class_loss
            + float(getattr(self.det_cfg, "bbox_loss_coef", 5.0)) * bbox_loss
            + float(getattr(self.det_cfg, "iou_loss_coef", 0.0)) * iou_loss
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
            cost_giou=getattr(cfg, "matcher_giou_cost", 2.0),
        )
        self._init_detection_anchor_pipeline(cfg, hidden_size, dropout, shared_layers)
        self.cls_head = ClassificationScorer.from_config(
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
        if getattr(cfg, "dense_coord_features", True):
            self.coord_proj = nn.Linear(2, hidden_size, bias=False)
        else:
            self.coord_proj = None
        if getattr(cfg, "bbox_prior_grid", False):
            self.register_buffer(
                "bbox_bias",
                _bbox_prior_grid(int(getattr(cfg, "num_fixed_slots", 100)), float(getattr(cfg, "bbox_prior_margin", 0.1))),
                persistent=False,
            )
        else:
            self.bbox_bias = None
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
