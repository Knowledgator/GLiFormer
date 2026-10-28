"""Vision task heads: image classification, object detection, segmentation."""

from dataclasses import dataclass
import weakref
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from .. import TaskHead, TaskHeadOutput
from ..box_ops import (
    aligned_generalized_box_iou,
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
)
from ...layers import covering_grid_2d, normalized_grid_2d, ragged_normalized_grid_2d
from ...layers.mlp import create_mlp
from ..media import (
    MediaClassificationHead,
    MediaSetPredictionHead,
    flat_features,
    matched_classification_loss,
    matched_mask_loss,
    normalized_objectness_loss,
)


def _xyxy_from_raw(raw: torch.Tensor) -> torch.Tensor:
    """Decode sigmoid ``cxcywh`` parameters without clipping the corners.

    The center and size remain bounded, but a box near an image edge may have a
    corner outside ``[0, 1]``. Keeping that corner differentiable is important:
    clamping here gives it a zero gradient and leaves boxes stuck on image
    boundaries. Public decoders clip the final inference result instead.
    """
    return box_cxcywh_to_xyxy(raw.sigmoid())


def _inverse_sigmoid(value: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    value = value.clamp(min=eps, max=1.0 - eps)
    return torch.log(value / (1.0 - value))


def _reference_box_grid(
    num_slots: int,
    margin: float = 0.1,
    box_size: float = 0.1,
) -> torch.Tensor:
    rows, cols = covering_grid_2d(num_slots)
    ys, xs = torch.meshgrid(
        torch.arange(rows, dtype=torch.float),
        torch.arange(cols, dtype=torch.float),
        indexing="ij",
    )
    centers = torch.stack([(xs.flatten() + 0.5) / cols, (ys.flatten() + 0.5) / rows], dim=-1)
    centers = centers[:num_slots]
    centers = margin + centers * (1.0 - 2.0 * margin)
    sizes = centers.new_full((num_slots, 2), box_size)
    return _inverse_sigmoid(torch.cat([centers, sizes], dim=-1))


def _random_reference_boxes(
    num_slots: int,
    margin: float = 0.1,
    box_size: float = 0.1,
) -> torch.Tensor:
    """Initialize learned references without a deterministic spatial codebook."""

    centers = torch.rand(num_slots, 2)
    centers = margin + centers * (1.0 - 2.0 * margin)
    sizes = centers.new_full((num_slots, 2), box_size)
    return _inverse_sigmoid(torch.cat([centers, sizes], dim=-1))


class ImageClassificationHead(MediaClassificationHead):
    """Vision specialization of the shared media classification head."""

    name = "image_classification"
    config_attribute = "image_classification_config"
    labels_key = "image_classification_labels"


@dataclass
class DetectionStagePredictions:
    """Predictions emitted by one iterative detector-decoder layer."""

    class_logits: torch.Tensor
    boxes_xyxy: torch.Tensor
    boxes_cxcywh: torch.Tensor
    objectness_logits: torch.Tensor
    anchors: torch.Tensor


@dataclass
class DetectionPredictions:
    """Named intermediate values shared by detection and segmentation."""

    class_logits: torch.Tensor
    boxes_xyxy: torch.Tensor
    boxes_cxcywh: torch.Tensor
    objectness_logits: torch.Tensor
    anchors: torch.Tensor
    anchor_mask: torch.Tensor
    dense_features: torch.Tensor
    dense_mask: Optional[torch.Tensor]
    # One (height, width) for the batch, or a (BN, 2) tensor when rows differ.
    spatial_shape: Union[Tuple[int, int], torch.Tensor]
    auxiliary_predictions: tuple[DetectionStagePredictions, ...] = ()


@dataclass
class DenseGrid:
    """Dense vision tokens with the geometry of each row's patch grid.

    Rows either share one grid (``uniform_shape``) or flatten their own
    ``(h, w)`` grid row-major into the first ``h * w`` token slots, e.g. a
    dynamic-resolution backbone such as Qwen3.5. Coordinates are normalized to
    each row's own image, so boxes stay in one ``[0, 1]`` frame either way.
    """

    features: torch.Tensor  # (BN, N, D)
    mask: Optional[torch.Tensor]  # (BN, N)
    coordinates: torch.Tensor  # (N, 2) shared, or (BN, N, 2) per row
    row_shapes: torch.Tensor  # (BN, 2)
    uniform_shape: Optional[Tuple[int, int]]

    @property
    def spatial_shape(self) -> Union[Tuple[int, int], torch.Tensor]:
        return self.uniform_shape if self.uniform_shape is not None else self.row_shapes


class ObjectDetectionHead(MediaSetPredictionHead):
    """Anchor-slot object detector over visual tokens.

    Class predictions use the same anchor + child-label fusion as the text
    heads. Box and objectness predictions are per anchor slot.
    """

    name = "object_detection"
    dependencies = []
    config_attribute = "object_detection_config"

    def __init__(self, config, hidden_size, dropout, shared_layers=None):
        super().__init__()
        cfg = getattr(config, self.config_attribute)
        self.det_cfg = cfg
        self.loss_coef = cfg.loss_coef
        if shared_layers is None:
            shared_layers = {}
        self._init_set_prediction_pipeline(
            cfg,
            config,
            hidden_size,
            dropout,
            shared_layers,
            geometry_cost=getattr(cfg, "matcher_bbox_cost", 5.0),
            giou_cost=getattr(cfg, "matcher_giou_cost", 2.0),
        )
        self.anchor_mode = cfg.effective_anchor_mode()

        self.bbox_head = create_mlp(
            input_dim=hidden_size,
            intermediate_dims=[hidden_size//2],
            output_dim=4,
            dropout=dropout,
            activation="gelu",
        )
        self.auxiliary_bbox_heads = nn.ModuleList()
        if (
            getattr(cfg, "iterative_box_refinement", False)
            and hasattr(self, "anchor_refine")
        ):
            self.auxiliary_bbox_heads.extend(
                create_mlp(
                    input_dim=hidden_size,
                    intermediate_dims=[hidden_size // 2],
                    output_dim=4,
                    dropout=dropout,
                    activation="gelu",
                )
                for _ in range(max(self.anchor_refine.num_layers - 1, 0))
            )
        if getattr(cfg, "bbox_head_zero_init", False):
            self._zero_output_layer(self.bbox_head)
            for bbox_head in self.auxiliary_bbox_heads:
                self._zero_output_layer(bbox_head)
        self.objectness_head = create_mlp(
            input_dim=hidden_size,
            intermediate_dims=[hidden_size//2],
            output_dim=1,
            dropout=dropout,
            activation="gelu",
        )
        self.reference_boxes = None
        if getattr(cfg, "reference_box_mode", "learned") == "learned":
            fixed_modes = {"fixed", "fixed_rnn", "fixed_transformer"}
            if self.anchor_mode not in fixed_modes:
                raise ValueError(
                    "reference_box_mode='learned' requires a fixed anchor mode; "
                    f"got {self.anchor_mode!r}"
                )
            num_slots = cfg.effective_anchor_num_slots()
            initializer = (
                _random_reference_boxes
                if getattr(cfg, "reference_box_initialization", "grid") == "random"
                else _reference_box_grid
            )
            references = initializer(
                num_slots,
                float(getattr(cfg, "reference_box_grid_margin", 0.1)),
                box_size=float(getattr(cfg, "reference_box_initial_size", 0.1)),
            )
            self.reference_boxes = nn.Parameter(references)

    @staticmethod
    def _zero_output_layer(module: nn.Module) -> None:
        """Start a refinement head as an identity update in logit space."""

        output_layer = next(
            (layer for layer in reversed(list(module.modules())) if isinstance(layer, nn.Linear)),
            None,
        )
        if output_layer is None:
            raise TypeError("A bounding-box MLP must end in a linear layer")
        nn.init.zeros_(output_layer.weight)
        if output_layer.bias is not None:
            nn.init.zeros_(output_layer.bias)

    @classmethod
    def from_config(cls, config, shared_layers=None, **kwargs):
        if getattr(config, cls.config_attribute) is None:
            return None
        return cls(config, hidden_size=config.hidden_size, dropout=config.dropout,
                   shared_layers=shared_layers)

    def _slot_query_pos(
        self,
        anchors: torch.Tensor,
        reference_boxes: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.query_position_embedding is None:
            return None
        if anchors.shape[1] == 0:
            return anchors.new_empty(1, 0, anchors.shape[-1])
        coordinates = None
        if reference_boxes is None and self.reference_boxes is not None:
            reference_boxes = self.reference_boxes.sigmoid()
        if reference_boxes is not None:
            if anchors.shape[1] != reference_boxes.shape[-2]:
                raise ValueError(
                    "Anchor count does not match configured reference boxes: "
                    f"{anchors.shape[1]} vs {reference_boxes.shape[-2]}"
                )
            coordinate_dimensions = getattr(
                self.query_position_embedding,
                "coordinate_dimensions",
                None,
            )
            if coordinate_dimensions is not None:
                coordinates = reference_boxes[..., :coordinate_dimensions]
        positions = self.query_position_embedding(
            coordinates,
            count=anchors.shape[1],
            spatial_shape=covering_grid_2d(anchors.shape[1]),
            dtype=anchors.dtype,
            device=anchors.device,
        )
        if positions is None:
            return None
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        if positions.dim() != 3 or positions.shape[0] not in {
            1,
            anchors.shape[0],
        }:
            raise ValueError(
                "Query positions must have shape (A, D), (1, A, D), or (B, A, D)"
            )
        if positions.shape[1] < anchors.shape[1]:
            raise ValueError(
                "The configured query position grid has fewer cells than anchors"
            )
        return positions[:, : anchors.shape[1]]

    def _slot_query_coordinates(
        self,
        anchors: torch.Tensor,
        reference_boxes: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Return raw query geometry for positions and attention biases."""

        if reference_boxes is None and self.reference_boxes is not None:
            reference_boxes = self.reference_boxes.sigmoid()
        if reference_boxes is None:
            return None
        if anchors.shape[1] != reference_boxes.shape[-2]:
            raise ValueError(
                "Anchor count does not match configured reference boxes: "
                f"{anchors.shape[1]} vs {reference_boxes.shape[-2]}"
            )
        coordinate_dimensions = getattr(
            getattr(self.anchor_refine_positions, "query_strategy", None),
            "coordinate_dimensions",
            None,
        )
        if coordinate_dimensions is None:
            return reference_boxes
        return reference_boxes[..., :coordinate_dimensions]

    def _spatial_attention_bias(
        self,
        reference_boxes: Optional[torch.Tensor],
        spatial_shape: Tuple[int, int],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Return an additive query-to-patch geometry prior.

        Positional vectors alone can be ignored by learned projections.  The
        Gaussian prior makes every reference attend to its own image region
        while leaving content attention free to choose patches within it.
        """

        bias_type = getattr(self.det_cfg, "spatial_attention_bias_type", "none")
        if bias_type == "none":
            return None
        if reference_boxes is None:
            raise ValueError("Spatial attention bias requires reference boxes")
        if bias_type != "gaussian":
            raise ValueError(f"Unsupported spatial attention bias {bias_type!r}")

        if isinstance(spatial_shape, torch.Tensor):
            patch_centers, _ = ragged_normalized_grid_2d(
                spatial_shape,
                int(spatial_shape.prod(-1).max()),
                device=device,
            )
            patch_centers = patch_centers[:, None]  # (BN, 1, N, 2)
        else:
            patch_centers = normalized_grid_2d(
                *spatial_shape,
                device=device,
                dtype=torch.float32,
            )[None, None]
        reference_centers = reference_boxes[..., :2].to(
            device=device,
            dtype=torch.float32,
        )
        squared_distance = (
            patch_centers - reference_centers[..., None, :]
        ).square().sum(dim=-1)
        sigma = float(getattr(self.det_cfg, "spatial_attention_sigma", 0.2))
        weight = float(
            getattr(self.det_cfg, "spatial_attention_bias_weight", 1.0)
        )
        return (-weight * squared_distance / (2.0 * sigma * sigma)).to(dtype=dtype)

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
        # Detector v3 used names tied to one positional implementation. Map
        # those tensors into the strategy-backed hierarchy when the configured
        # destination exists; discard only legacy tensors with no live consumer.
        current_keys = set(self.state_dict())
        for old_name, new_name in (
            ("reference_points", "reference_boxes"),
            (
                "query_position_embedding.embedding.weight",
                "anchor_refine_positions.query_embedding.embedding.weight",
            ),
            (
                "memory_position_embedding.projection.weight",
                "anchor_refine_positions.memory_embedding.projection.weight",
            ),
            (
                "slot_pos_emb.weight",
                "anchor_refine_positions.query_embedding.embedding.weight",
            ),
            (
                "coord_proj.weight",
                "anchor_refine_positions.memory_embedding.projection.weight",
            ),
        ):
            old_key = f"{prefix}{old_name}"
            new_key = f"{prefix}{new_name}"
            if old_key not in state_dict:
                continue
            if new_name in current_keys and new_key not in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)
            else:
                state_dict.pop(old_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """Emit the two pre-component position aliases for old tooling."""

        super()._save_to_state_dict(destination, prefix, keep_vars)
        positions = self.anchor_refine_positions
        if positions is None:
            return
        for role in ("query", "memory"):
            module = getattr(positions, f"{role}_embedding")
            for name, value in module.state_dict(keep_vars=keep_vars).items():
                destination[f"{prefix}{role}_position_embedding.{name}"] = value

    def _dense_grid(self, flat_inputs) -> DenseGrid:
        image_features, image_mask = flat_features(flat_inputs)
        spatial_shapes = getattr(flat_inputs, "feature_spatial_shape", None)
        prefix_tokens = getattr(flat_inputs, "feature_prefix_tokens", None)
        if spatial_shapes is None or prefix_tokens is None:
            raise ValueError(
                "Dense vision heads require feature_spatial_shape and "
                "feature_prefix_tokens from VisionEncoderOutput"
            )
        if not torch.equal(prefix_tokens, prefix_tokens[:1].expand_as(prefix_tokens)):
            raise ValueError("A dense token tensor must have one prefix-token count per padded batch")
        spatial_shapes = spatial_shapes.to(device=image_features.device, dtype=torch.long)
        uniform = bool(torch.equal(spatial_shapes, spatial_shapes[:1].expand_as(spatial_shapes)))
        dense_count = int(spatial_shapes.prod(-1).max())
        if dense_count == 0:
            raise ValueError("Dense vision heads received no image tokens")

        prefix_count = int(prefix_tokens[0].item())
        end = prefix_count + dense_count
        if prefix_count < 0 or end > image_features.shape[1]:
            raise ValueError(
                f"Vision metadata describes tokens [{prefix_count}:{end}], but "
                f"the sequence has length {image_features.shape[1]}"
            )
        image_features = image_features[:, prefix_count:end]
        if image_mask is not None:
            image_mask = image_mask[:, prefix_count:end]

        if uniform:
            rows, cols = (int(value) for value in spatial_shapes[0].tolist())
            coordinates = normalized_grid_2d(
                rows,
                cols,
                device=image_features.device,
                dtype=torch.float32,
            )
            return DenseGrid(image_features, image_mask, coordinates, spatial_shapes, (rows, cols))

        # Rows carry grids of different sizes: each row's tokens fill the front
        # of the padded sequence, and slots past its own grid are masked out.
        coordinates, in_grid = ragged_normalized_grid_2d(
            spatial_shapes,
            dense_count,
            device=image_features.device,
        )
        if image_mask is None:
            image_mask = in_grid.long()
        else:
            image_mask = (image_mask.to(device=in_grid.device).bool() & in_grid).to(image_mask.dtype)
        return DenseGrid(image_features, image_mask, coordinates, spatial_shapes, None)

    def _grid_memory_positions(self, grid: DenseGrid) -> Optional[torch.Tensor]:
        positions = self.anchor_refine_positions
        if positions is None:
            return None
        if grid.uniform_shape is not None or not positions.memory_strategy.requires_grid_size:
            return positions.memory_positions(
                grid.features,
                grid.mask,
                coordinates=grid.coordinates,
                spatial_shape=grid.uniform_shape,
            )
        # Grid-table positions resize to one shape per call: encode each
        # distinct grid on its own rows and place them back in the padded layout.
        output = None
        for shape in torch.unique(grid.row_shapes, dim=0):
            height, width = (int(value) for value in shape.tolist())
            count = height * width
            if count == 0:
                continue
            rows = (grid.row_shapes == shape).all(dim=-1).nonzero().squeeze(-1)
            part = positions.memory_positions(
                grid.features[rows, :count],
                grid.mask[rows, :count] if grid.mask is not None else None,
                coordinates=grid.coordinates[rows, :count],
                spatial_shape=(height, width),
            )
            if part is None:
                return None
            if output is None:
                output = part.new_zeros(*grid.features.shape[:2], part.shape[-1])
            output[rows, :count] = part
        return output

    def _dense_features(self, flat_inputs):
        grid = self._dense_grid(flat_inputs)
        return grid.features, grid.mask, self._grid_memory_positions(grid), grid.spatial_shape

    def _compute_detection(self, flat_inputs, count=None, threshold=0.5) -> DetectionPredictions:
        grid = self._dense_grid(flat_inputs)
        image_features, image_mask = grid.features, grid.mask
        memory_positions = self._grid_memory_positions(grid)
        memory_coordinates = grid.coordinates
        # Grid-table position strategies need one shape; per-row grids pass None.
        memory_spatial_shape = grid.uniform_shape
        spatial_shape = grid.spatial_shape
        anchors, anchor_mask = self._generate_anchors(
            flat_inputs.parent_embedding,
            image_features,
            count=count,
            threshold=threshold,
            feature_mask=image_mask,
        )
        iterative_refinement = bool(
            getattr(self.det_cfg, "iterative_box_refinement", False)
        )
        if hasattr(self, "anchor_refine") and iterative_refinement:
            if self.reference_boxes is None:
                raise ValueError("Iterative box refinement requires reference boxes")
            num_layers = self.anchor_refine.num_layers
            if len(self.auxiliary_bbox_heads) != max(num_layers - 1, 0):
                raise RuntimeError(
                    "The number of per-layer box heads does not match the decoder"
                )
            memory = self.anchor_refine.prepare_memory(
                image_features,
                image_mask,
                memory_positions,
                memory_coordinates=memory_coordinates,
                memory_spatial_shape=memory_spatial_shape,
                position_encoding=self.anchor_refine_positions,
            )
            reference_logits = self.reference_boxes.unsqueeze(0).expand(
                anchors.shape[0],
                -1,
                -1,
            ).float()
            stage_predictions: list[DetectionStagePredictions] = []
            for layer_index in range(num_layers):
                reference_boxes = reference_logits.sigmoid()
                anchors = self.anchor_refine.forward_layer(
                    layer_index,
                    anchors,
                    memory,
                    query_mask=anchor_mask,
                    query_coordinates=self._slot_query_coordinates(
                        anchors,
                        reference_boxes,
                    ),
                    query_spatial_shape=covering_grid_2d(anchors.shape[1]),
                    position_encoding=self.anchor_refine_positions,
                    self_attention_bias_module=self.anchor_self_attention_bias,
                    cross_attention_bias_module=self.anchor_cross_attention_bias,
                )
                bbox_head = (
                    self.bbox_head
                    if layer_index == num_layers - 1
                    else self.auxiliary_bbox_heads[layer_index]
                )
                updated_reference_logits = bbox_head(anchors).float() + reference_logits
                boxes_cxcywh = updated_reference_logits.sigmoid()
                stage_predictions.append(
                    DetectionStagePredictions(
                        class_logits=self._score_anchor_labels(
                            anchors,
                            flat_inputs.child_embedding,
                            flat_inputs.child_mask,
                            anchor_mask,
                        ),
                        boxes_xyxy=box_cxcywh_to_xyxy(boxes_cxcywh),
                        boxes_cxcywh=boxes_cxcywh,
                        objectness_logits=self.objectness_head(anchors).squeeze(-1),
                        anchors=anchors,
                    )
                )
                reference_logits = (
                    updated_reference_logits.detach()
                    if getattr(self.det_cfg, "bbox_refinement_detach", True)
                    else updated_reference_logits
                )

            final = stage_predictions[-1]
            return DetectionPredictions(
                class_logits=final.class_logits,
                boxes_xyxy=final.boxes_xyxy,
                boxes_cxcywh=final.boxes_cxcywh,
                objectness_logits=final.objectness_logits,
                anchors=final.anchors,
                anchor_mask=anchor_mask,
                dense_features=image_features,
                dense_mask=image_mask,
                spatial_shape=spatial_shape,
                auxiliary_predictions=tuple(stage_predictions[:-1]),
            )

        static_references = None
        if self.reference_boxes is not None:
            static_references = self.reference_boxes.sigmoid().unsqueeze(0).expand(
                anchors.shape[0],
                -1,
                -1,
            )
        anchors = self._refine_anchors(
            anchors,
            image_features,
            memory_mask=image_mask,
            anchor_mask=anchor_mask,
            query_coordinates=self._slot_query_coordinates(
                anchors,
                static_references,
            ),
            memory_coordinates=memory_coordinates,
            memory_positions=memory_positions,
            query_spatial_shape=covering_grid_2d(anchors.shape[1]),
            memory_spatial_shape=memory_spatial_shape,
        )

        class_logits = self._score_anchor_labels(
            anchors,
            flat_inputs.child_embedding,
            flat_inputs.child_mask,
            anchor_mask,
        )
        bbox_raw = self.bbox_head(anchors).float()
        if self.reference_boxes is not None:
            if bbox_raw.shape[1] != self.reference_boxes.shape[0]:
                raise ValueError("Reference-box count must match the number of anchors")
            bbox_raw = bbox_raw + self.reference_boxes.unsqueeze(0).float()
        bbox_cxcywh = bbox_raw.sigmoid()
        bbox_preds = _xyxy_from_raw(bbox_raw)
        objectness_logits = self.objectness_head(anchors).squeeze(-1)
        return DetectionPredictions(
            class_logits=class_logits,
            boxes_xyxy=bbox_preds,
            boxes_cxcywh=bbox_cxcywh,
            objectness_logits=objectness_logits,
            anchors=anchors,
            anchor_mask=anchor_mask,
            dense_features=image_features,
            dense_mask=image_mask,
            spatial_shape=spatial_shape,
        )

    def _match_single(
        self,
        class_logits,
        bbox_preds,
        gold_classes,
        gold_boxes,
        valid_objects,
        prediction_mask=None,
        bbox_cxcywh=None,
    ):
        if (
            getattr(self.det_cfg, "bbox_l1_format", "cxcywh") == "cxcywh"
            and bbox_cxcywh is not None
        ):
            geometry_preds = bbox_cxcywh
            target_geometry = box_xyxy_to_cxcywh(gold_boxes)
        else:
            geometry_preds = bbox_preds
            target_geometry = gold_boxes
        return self._match_geometry(
            class_logits,
            geometry_preds,
            gold_classes,
            target_geometry,
            valid_objects,
            prediction_mask=prediction_mask,
            giou_geometry_preds=bbox_preds,
            target_giou_geometry=gold_boxes,
        )

    def _detection_loss(
        self,
        class_logits,
        bbox_preds,
        bbox_cxcywh,
        objectness_logits,
        anchor_mask,
        child_mask,
        class_labels,
        bbox_labels,
        object_mask,
        base_loss_fn=None,
        auxiliary_predictions: tuple[DetectionStagePredictions, ...] = (),
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
                bbox_cxcywh[b],
            )
            matches_by_batch[b] = matches
            if matches:
                src_idx = torch.tensor([a for a, _ in matches], dtype=torch.long, device=device)
                target_objectness[b, src_idx] = 1.0
                for anchor_idx, obj_idx in matches:
                    src_batch_idx.append(b)
                    src_anchor_idx.append(anchor_idx)
                    tgt_obj_idx.append(obj_idx)

        num_matched = max(len(src_batch_idx), 1)
        loss_fn = base_loss_fn or self._default_binary_loss

        if src_batch_idx:
            sb = torch.tensor(src_batch_idx, dtype=torch.long, device=device)
            sa = torch.tensor(src_anchor_idx, dtype=torch.long, device=device)
            tt = torch.tensor(tgt_obj_idx, dtype=torch.long, device=device)

            class_loss = matched_classification_loss(
                class_logits,
                class_labels,
                child_mask,
                matches_by_batch,
                loss_fn,
                class_probability=getattr(
                    self.det_cfg,
                    "class_probability",
                    "sigmoid",
                ),
            )

            matched_bbox_pred = bbox_preds[sb, sa]
            matched_bbox_target = bbox_labels[sb, tt].to(device=device, dtype=bbox_preds.dtype)
            if getattr(self.det_cfg, "bbox_l1_format", "cxcywh") == "cxcywh":
                matched_l1_pred = bbox_cxcywh[sb, sa]
                matched_l1_target = box_xyxy_to_cxcywh(matched_bbox_target)
            else:
                matched_l1_pred = matched_bbox_pred
                matched_l1_target = matched_bbox_target
            bbox_loss = F.l1_loss(
                matched_l1_pred,
                matched_l1_target,
                reduction="sum",
            ) / num_matched
            iou_loss = (
                1.0 - aligned_generalized_box_iou(
                    matched_bbox_pred,
                    matched_bbox_target,
                )
            ).sum() / num_matched
        else:
            class_loss = class_logits.new_tensor(0.0)
            bbox_loss = class_logits.new_tensor(0.0)
            iou_loss = class_logits.new_tensor(0.0)

        # Positive and background terms are means over their own populations.
        # Consequently, changing the query count or image density cannot silently
        # rescale objectness relative to the box and classification objectives.
        objectness_loss = normalized_objectness_loss(
            objectness_logits,
            target_objectness,
            anchor_mask,
            loss_fn,
            positive_weight=getattr(
                self.det_cfg,
                "objectness_positive_weight",
                1.0,
            ),
            negative_weight=getattr(
                self.det_cfg,
                "objectness_negative_weight",
                1.0,
            ),
            loss_kwargs=self._objectness_loss_kwargs(),
        )

        loss = (
            float(getattr(self.det_cfg, "class_loss_coef", 1.0)) * class_loss
            + float(getattr(self.det_cfg, "bbox_loss_coef", 5.0)) * bbox_loss
            + float(getattr(self.det_cfg, "iou_loss_coef", 0.0)) * iou_loss
            + float(getattr(self.det_cfg, "objectness_loss_coef", 1.0)) * objectness_loss
        )
        auxiliary_coef = float(
            getattr(self.det_cfg, "auxiliary_detection_loss_coef", 0.0)
        )
        if auxiliary_coef > 0.0 and auxiliary_predictions:
            auxiliary_losses = []
            for auxiliary in auxiliary_predictions:
                auxiliary_loss, _ = self._detection_loss(
                    auxiliary.class_logits,
                    auxiliary.boxes_xyxy,
                    auxiliary.boxes_cxcywh,
                    auxiliary.objectness_logits,
                    anchor_mask,
                    child_mask,
                    class_labels,
                    bbox_labels,
                    object_mask,
                    base_loss_fn=base_loss_fn,
                )
                auxiliary_losses.append(auxiliary_loss)
            loss = loss + auxiliary_coef * torch.stack(auxiliary_losses).mean()
        return loss, matches_by_batch

    def forward(self, shared, dependency_outputs, flat_inputs=None, **batch):
        prefix = self.name
        class_labels = batch.get(f"{prefix}_class_labels")
        bbox_labels = batch.get(f"{prefix}_bbox_labels")
        object_mask = batch.get(f"{prefix}_object_mask")
        count = batch.get(f"{prefix}_count")
        threshold = batch.get("threshold", 0.5)
        predictions = self._compute_detection(
            flat_inputs,
            count=count,
            threshold=threshold,
        )

        loss = None
        matches = None
        if class_labels is not None and bbox_labels is not None and object_mask is not None:
            loss, matches = self._detection_loss(
                predictions.class_logits,
                predictions.boxes_xyxy,
                predictions.boxes_cxcywh,
                predictions.objectness_logits,
                predictions.anchor_mask,
                flat_inputs.child_mask, class_labels, bbox_labels, object_mask,
                base_loss_fn=batch.get("base_loss_fn"),
                auxiliary_predictions=predictions.auxiliary_predictions,
            )

        return TaskHeadOutput(
            loss=loss,
            logits=predictions.class_logits,
            extra={
                "bbox_preds": predictions.boxes_xyxy,
                "bbox_cxcywh": predictions.boxes_cxcywh,
                "objectness_logits": predictions.objectness_logits,
                "anchor_mask": predictions.anchor_mask,
                "anchors": predictions.anchors,
                "matches": matches,
                "_predictions": predictions,
                "_flat_inputs": flat_inputs,
                "_count": count,
                "_threshold": threshold,
            },
        )


class SegmentationHead(ObjectDetectionHead):
    """Detection head with prototype-mask segmentation outputs."""

    name = "segmentation"
    dependencies = ["object_detection"]
    config_attribute = "segmentation_config"

    def __init__(
        self,
        config,
        hidden_size,
        dropout,
        shared_layers=None,
        detection_head=None,
    ):
        if detection_head is None:
            super().__init__(
                config,
                hidden_size,
                dropout,
                shared_layers=shared_layers,
            )
            self._uses_shared_detection_head = False
        else:
            TaskHead.__init__(self)
            self.det_cfg = getattr(config, self.config_attribute)
            self.loss_coef = self.det_cfg.loss_coef
            self._init_set_prediction_matcher(
                self.det_cfg,
                geometry_cost=getattr(
                    self.det_cfg,
                    "matcher_bbox_cost",
                    5.0,
                ),
                giou_cost=getattr(
                    self.det_cfg,
                    "matcher_giou_cost",
                    2.0,
                ),
            )
            # A weak reference avoids registering the same module twice in the
            # state dict; the canonical owner remains heads.object_detection.
            object.__setattr__(
                self,
                "_detection_head_ref",
                weakref.ref(detection_head),
            )
            self._uses_shared_detection_head = True
        cfg = self.det_cfg
        self.num_prototypes = int(getattr(cfg, "num_prototypes", 32))
        self.mask_size = int(getattr(cfg, "mask_size", 128))
        self.proto_proj = nn.Linear(hidden_size, self.num_prototypes)
        self.proto_refine = nn.Sequential(
            nn.Conv2d(self.num_prototypes, self.num_prototypes, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.num_prototypes, self.num_prototypes, kernel_size=1),
        )
        self.coeff_head = nn.Linear(hidden_size, self.num_prototypes)

    @staticmethod
    def _compatible_detection_configs(detection_config, segmentation_config):
        if detection_config is None or segmentation_config is None:
            return False
        # Matcher, focal/objectness policy, and loss weights remain owned by
        # segmentation and must not force a duplicate detector parameter tree.
        return (
            detection_config.prediction_architecture_signature()
            == segmentation_config.prediction_architecture_signature()
        )

    @classmethod
    def from_config(
        cls,
        config,
        shared_layers=None,
        detection_head=None,
        **kwargs,
    ):
        segmentation_config = getattr(config, cls.config_attribute)
        if segmentation_config is None:
            return None
        can_reuse = (
            detection_head is not None
            and getattr(segmentation_config, "reuse_detection_head", True)
            and cls._compatible_detection_configs(
                getattr(config, "object_detection_config", None),
                segmentation_config,
            )
        )
        return cls(
            config,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            shared_layers=shared_layers,
            detection_head=detection_head if can_reuse else None,
        )

    def _shared_detection_head(self):
        reference = getattr(self, "_detection_head_ref", None)
        return reference() if reference is not None else None

    def _compute_detection(self, flat_inputs, count=None, threshold=0.5):
        detection_head = self._shared_detection_head()
        if detection_head is None:
            return super()._compute_detection(flat_inputs, count, threshold)
        return detection_head._compute_detection(flat_inputs, count, threshold)

    @staticmethod
    def _same_flat_inputs(left, right):
        if left is None or right is None:
            return False
        for name in (
            "batch_origin",
            "parent_embedding",
            "child_embedding",
            "child_mask",
            "feature_embedding",
            "feature_mask",
            "feature_spatial_shape",
            "feature_prefix_tokens",
        ):
            left_value = getattr(left, name, None)
            right_value = getattr(right, name, None)
            if left_value is None or right_value is None:
                if left_value is not right_value:
                    return False
            elif left_value.shape != right_value.shape or not torch.equal(
                left_value,
                right_value,
            ):
                return False
        return True

    @staticmethod
    def _same_optional_value(left, right):
        if left is None or right is None:
            return left is right
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            left = torch.as_tensor(left)
            right = torch.as_tensor(right, device=left.device)
            return left.shape == right.shape and torch.equal(left, right)
        return left == right

    def _refine_prototype_grid(self, tokens, height, width, coarse_validity=None):
        """Project ``(B, height * width, D)`` grid tokens to refined prototype maps."""

        proto = self.proto_proj(tokens).transpose(1, 2).reshape(
            tokens.shape[0],
            self.num_prototypes,
            height,
            width,
        )
        if coarse_validity is not None:
            # Linear and convolutional biases can recreate values in padded
            # cells, so preserve validity at every refinement boundary.
            proto = proto * coarse_validity
        for layer in self.proto_refine:
            proto = layer(proto)
            if coarse_validity is not None:
                proto = proto * coarse_validity
        return proto

    def _upsample_prototypes(self, proto, coarse_validity=None):
        proto = F.interpolate(
            proto,
            size=(self.mask_size, self.mask_size),
            mode="bilinear",
            align_corners=False,
        )
        output_validity = None
        if coarse_validity is not None:
            output_validity = F.interpolate(
                coarse_validity,
                size=(self.mask_size, self.mask_size),
                mode="nearest",
            ).squeeze(1).bool()
            proto = proto * output_validity.unsqueeze(1)
        return proto, output_validity

    def _prototype_masks(
        self,
        vision_tokens,
        spatial_shape,
        vision_mask=None,
        *,
        return_validity=False,
    ):
        if isinstance(spatial_shape, torch.Tensor):
            proto, output_validity = self._ragged_prototype_masks(
                vision_tokens,
                spatial_shape,
                vision_mask,
            )
            return (proto, output_validity) if return_validity else proto

        batch_size, token_count, _ = vision_tokens.shape
        height, width = spatial_shape
        if height * width != token_count:
            raise ValueError(
                f"Segmentation received {token_count} dense tokens for spatial "
                f"shape {(height, width)}"
            )
        coarse_validity = None
        if vision_mask is not None:
            if vision_mask.shape != vision_tokens.shape[:2]:
                raise ValueError(
                    "vision feature mask must match the prototype token sequence"
                )
            valid_tokens = vision_mask.to(
                device=vision_tokens.device,
                dtype=vision_tokens.dtype,
            )
            vision_tokens = vision_tokens * valid_tokens.unsqueeze(-1)
            coarse_validity = valid_tokens.reshape(
                batch_size,
                1,
                height,
                width,
            )
        proto = self._refine_prototype_grid(vision_tokens, height, width, coarse_validity)
        proto, output_validity = self._upsample_prototypes(proto, coarse_validity)
        if return_validity:
            return proto, output_validity
        return proto

    def _ragged_prototype_masks(self, vision_tokens, spatial_shapes, vision_mask=None):
        """Prototype masks for rows whose tokens flatten grids of different sizes.

        Each row's grid is laid onto a shared zero canvas, so the convolutions
        see exactly the zero padding they would see on the row's own grid. The
        row's own crop is then upsampled to the full mask, which keeps masks in
        the same normalized image frame as the boxes.
        """

        batch_size, token_count, hidden_size = vision_tokens.shape
        device = vision_tokens.device
        shapes = spatial_shapes.to(device=device, dtype=torch.long)
        if shapes.shape != (batch_size, 2):
            raise ValueError(
                f"spatial_shapes must have shape {(batch_size, 2)}, got {tuple(shapes.shape)}"
            )
        if int(shapes.prod(-1).max()) > token_count:
            raise ValueError(
                f"Segmentation received {token_count} dense tokens for grids up to "
                f"{int(shapes.prod(-1).max())} cells"
            )
        _, valid = ragged_normalized_grid_2d(shapes, token_count, device=device)
        if vision_mask is not None:
            if vision_mask.shape != vision_tokens.shape[:2]:
                raise ValueError(
                    "vision feature mask must match the prototype token sequence"
                )
            valid = valid & vision_mask.to(device=device).bool()

        height = int(shapes[:, 0].max())
        width = int(shapes[:, 1].max())
        canvas_cells = height * width
        slots = torch.arange(token_count, device=device).unsqueeze(0)
        row_widths = shapes[:, 1:].clamp(min=1)
        canvas_index = (slots // row_widths) * width + slots % row_widths
        # Padding slots all land in one extra cell that is dropped afterwards.
        canvas_index = torch.where(valid, canvas_index, torch.full_like(canvas_index, canvas_cells))
        valid_values = valid.to(dtype=vision_tokens.dtype)
        canvas = vision_tokens.new_zeros(batch_size, canvas_cells + 1, hidden_size).scatter_add(
            1,
            canvas_index.unsqueeze(-1).expand(-1, -1, hidden_size),
            vision_tokens * valid_values.unsqueeze(-1),
        )[:, :canvas_cells]
        coarse_validity = valid_values.new_zeros(batch_size, canvas_cells + 1).scatter_add(
            1,
            canvas_index,
            valid_values,
        )[:, :canvas_cells].reshape(batch_size, 1, height, width)

        proto = self._refine_prototype_grid(canvas, height, width, coarse_validity)
        output = None
        output_validity = torch.zeros(
            batch_size,
            self.mask_size,
            self.mask_size,
            dtype=torch.bool,
            device=device,
        )
        for shape in torch.unique(shapes, dim=0):
            row_height, row_width = (int(value) for value in shape.tolist())
            if row_height * row_width == 0:
                continue
            rows = (shapes == shape).all(dim=-1).nonzero().squeeze(-1)
            upsampled, upsampled_validity = self._upsample_prototypes(
                proto[rows, :, :row_height, :row_width],
                coarse_validity[rows, :, :row_height, :row_width],
            )
            if output is None:
                # Interpolation may change dtype under autocast; follow it.
                output = upsampled.new_zeros(batch_size, *upsampled.shape[1:])
            output[rows] = upsampled
            output_validity[rows] = upsampled_validity
        if output is None:
            output = proto.new_zeros(batch_size, self.num_prototypes, self.mask_size, self.mask_size)
        return output, output_validity

    def forward(self, shared, dependency_outputs, flat_inputs=None, **batch):
        class_labels = batch.get("segmentation_class_labels")
        bbox_labels = batch.get("segmentation_bbox_labels")
        object_mask = batch.get("segmentation_object_mask")
        mask_labels = batch.get("segmentation_mask_labels")

        detection_output = dependency_outputs.get("object_detection")
        detection_extra = (
            detection_output.extra
            if detection_output is not None and detection_output.extra is not None
            else {}
        )
        count = batch.get("segmentation_count")
        threshold = batch.get("threshold", 0.5)
        if (
            self._uses_shared_detection_head
            and self._same_flat_inputs(
                flat_inputs,
                detection_extra.get("_flat_inputs"),
            )
            and self._same_optional_value(count, detection_extra.get("_count"))
            and self._same_optional_value(
                threshold,
                detection_extra.get("_threshold"),
            )
        ):
            predictions = detection_extra["_predictions"]
        else:
            predictions = self._compute_detection(
                flat_inputs,
                count=count,
                threshold=threshold,
            )
        prototypes, mask_validity = self._prototype_masks(
            predictions.dense_features,
            predictions.spatial_shape,
            predictions.dense_mask,
            return_validity=True,
        )
        coefficients = self.coeff_head(predictions.anchors)
        mask_logits = torch.einsum("bphw,bap->bahw", prototypes, coefficients)

        loss = None
        matches = None
        if class_labels is not None and bbox_labels is not None and object_mask is not None:
            det_loss, matches = self._detection_loss(
                predictions.class_logits,
                predictions.boxes_xyxy,
                predictions.boxes_cxcywh,
                predictions.objectness_logits,
                predictions.anchor_mask,
                flat_inputs.child_mask, class_labels, bbox_labels, object_mask,
                base_loss_fn=batch.get("base_loss_fn"),
                auxiliary_predictions=predictions.auxiliary_predictions,
            )
            mask_loss = predictions.class_logits.new_tensor(0.0)
            if mask_labels is not None:
                mask_loss = matched_mask_loss(
                    mask_logits,
                    mask_labels,
                    matches,
                    batch.get("base_loss_fn") or self._default_binary_loss,
                    validity_mask=mask_validity,
                )
            loss = det_loss + float(getattr(self.det_cfg, "mask_loss_coef", 1.0)) * mask_loss

        return TaskHeadOutput(
            loss=loss,
            logits=predictions.class_logits,
            extra={
                "bbox_preds": predictions.boxes_xyxy,
                "bbox_cxcywh": predictions.boxes_cxcywh,
                "objectness_logits": predictions.objectness_logits,
                "anchor_mask": predictions.anchor_mask,
                "anchors": predictions.anchors,
                "matches": matches,
                "mask_logits": mask_logits,
                "mask_validity": mask_validity,
                "prototypes": prototypes,
                "coefficients": coefficients,
            },
        )
