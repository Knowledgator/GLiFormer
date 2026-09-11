"""Reusable building blocks for vision and audio task heads."""

import torch
import torch.nn.functional as F

from ..layers import Pooling
from . import TaskDecoder, TaskHead, TaskHeadOutput
from .classification.scorer import ClassificationScorer
from .losses import binary_focal_or_bce, configured_binary_loss
from .matcher import HungarianMatcher
from ..processing.decoder import unflatten_by_batch_origin


def flat_features(flat_inputs):
    """Return the task feature sequence and mask from flattened inputs.

    Media-only models populate ``feature_embedding`` and ``feature_mask``.
    Falling back to the historical text-shaped fields keeps multimodal and old
    checkpoint call paths compatible without duplicating this decision in each
    task head.
    """

    features = getattr(flat_inputs, "feature_embedding", None)
    mask = getattr(flat_inputs, "feature_mask", None)
    return (
        features if features is not None else flat_inputs.words_embedding,
        mask if mask is not None else flat_inputs.mask,
    )


def matched_mask_loss(
    mask_logits: torch.Tensor,
    mask_labels: torch.Tensor,
    matches,
    loss_fn=None,
    validity_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute configured binary loss over matched masks, normalized per element.

    Pixel/time resolution and the number of matched instances therefore do not
    change the scale relative to the normalized detection/segmentation losses.
    """

    matched_pred = []
    matched_target = []
    matched_validity = []
    for batch_idx, pairs in (matches or {}).items():
        for prediction_idx, target_idx in pairs:
            matched_pred.append(mask_logits[batch_idx, prediction_idx])
            matched_target.append(mask_labels[batch_idx, target_idx])
            if validity_mask is not None:
                matched_validity.append(validity_mask[batch_idx])
    if not matched_pred:
        return mask_logits.new_tensor(0.0)

    loss_fn = loss_fn or binary_focal_or_bce
    loss = loss_fn(
        torch.stack(matched_pred),
        torch.stack(matched_target).to(
            device=mask_logits.device,
            dtype=mask_logits.dtype,
        ),
    )
    flat_loss = loss.flatten(1)
    if matched_validity:
        valid = torch.stack(matched_validity).to(
            device=flat_loss.device,
            dtype=flat_loss.dtype,
        ).flatten(1)
        if valid.shape != flat_loss.shape:
            raise ValueError(
                "mask validity must match the spatial or temporal mask-logit shape"
            )
        return (
            (flat_loss * valid).sum(dim=1)
            / valid.sum(dim=1).clamp(min=1.0)
        ).mean()
    return flat_loss.mean(dim=1).mean()


def matched_classification_loss(
    class_logits: torch.Tensor,
    class_labels: torch.Tensor,
    child_mask: torch.Tensor,
    matches: dict[int, list[tuple[int, int]]],
    loss_fn,
    *,
    class_probability: str = "sigmoid",
) -> torch.Tensor:
    """Classify matched set-prediction slots with class-count-invariant scale."""

    batch_indices = []
    anchor_indices = []
    target_indices = []
    for batch_idx, pairs in matches.items():
        for anchor_idx, target_idx in pairs:
            batch_indices.append(batch_idx)
            anchor_indices.append(anchor_idx)
            target_indices.append(target_idx)
    if not batch_indices:
        return class_logits.new_tensor(0.0)

    device = class_logits.device
    sb = torch.tensor(batch_indices, dtype=torch.long, device=device)
    sa = torch.tensor(anchor_indices, dtype=torch.long, device=device)
    tt = torch.tensor(target_indices, dtype=torch.long, device=device)
    matched_logits = class_logits[sb, sa]
    label_mask = child_mask[sb, : class_logits.shape[-1]].to(device=device).bool()
    target_classes = class_labels[sb, tt].to(device=device).long()
    valid = (target_classes >= 0) & (target_classes < class_logits.shape[-1])
    if not valid.any():
        return class_logits.new_tensor(0.0)

    matched_logits = matched_logits[valid]
    label_mask = label_mask[valid]
    target_classes = target_classes[valid]
    if class_probability == "softmax":
        return F.cross_entropy(
            matched_logits.masked_fill(~label_mask, -1e4),
            target_classes,
        )
    if class_probability != "sigmoid":
        raise ValueError(f"Unsupported class probability: {class_probability!r}")

    targets = torch.zeros_like(matched_logits)
    targets.scatter_(1, target_classes[:, None], 1.0)
    losses = loss_fn(matched_logits.float(), targets.float())
    valid_class_count = label_mask.sum(dim=-1).clamp(min=1)
    return ((losses * label_mask).sum(dim=-1) / valid_class_count).mean()


def score_anchor_labels(
    anchor_modeling,
    scorer,
    anchors: torch.Tensor,
    label_representations: torch.Tensor,
    label_mask: torch.Tensor | None = None,
    anchor_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse and score anchor/label pairs through the shared anchor hierarchy."""

    batch_size, anchor_count, hidden_size = anchors.shape
    class_count = label_representations.shape[1]
    fused_labels = anchor_modeling(
        anchors,
        label_representations,
        anchor_mask=anchor_mask,
        child_mask=label_mask,
    )
    logits = scorer(
        anchors.reshape(batch_size * anchor_count, hidden_size),
        fused_labels.reshape(
            batch_size * anchor_count,
            class_count,
            hidden_size,
        ),
    ).reshape(batch_size, anchor_count, class_count)
    if label_mask is not None:
        valid_labels = label_mask[:, None, :class_count].to(
            device=logits.device,
        ).bool()
        logits = logits.masked_fill(~valid_labels, -1e4)
    return logits


def normalized_objectness_loss(
    objectness_logits: torch.Tensor,
    target_objectness: torch.Tensor,
    prediction_mask: torch.Tensor,
    loss_fn,
    *,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
    loss_kwargs: dict | None = None,
) -> torch.Tensor:
    """Average foreground and background query losses independently."""

    losses = loss_fn(
        objectness_logits.float(),
        target_objectness.float(),
        **(loss_kwargs or {}),
    )
    valid = prediction_mask.to(device=losses.device).bool()
    positive = valid & target_objectness.to(device=losses.device).bool()
    negative = valid & ~target_objectness.to(device=losses.device).bool()
    zero = losses.new_tensor(0.0)
    positive_loss = losses[positive].mean() if positive.any() else zero
    negative_loss = losses[negative].mean() if negative.any() else zero
    return float(positive_weight) * positive_loss + float(negative_weight) * negative_loss


def id_maps(classes_mapping, attribute: str) -> list[dict]:
    """Extract per-flat-group reverse class mappings for a task decoder."""

    maps = []
    if classes_mapping is None:
        return maps
    for class_mapping in getattr(classes_mapping, attribute, []):
        for item in class_mapping.items:
            maps.append(item.class_to_id.get_reverse_mapping())
    return maps


class MediaSetPredictionHead(TaskHead):
    """Shared anchor/matcher/classification core for media set prediction."""

    def _init_set_prediction_matcher(
        self,
        task_config,
        *,
        geometry_cost: float,
        giou_cost: float = 0.0,
    ) -> None:
        self.set_prediction_config = task_config
        self.matcher = HungarianMatcher(
            cost_class=getattr(task_config, "matcher_class_cost", 1.0),
            cost_geometry=geometry_cost,
            cost_giou=giou_cost,
            class_probability=getattr(
                task_config,
                "class_probability",
                "sigmoid",
            ),
        )

    def _init_set_prediction_pipeline(
        self,
        task_config,
        model_config,
        hidden_size: int,
        dropout: float,
        shared_layers,
        *,
        geometry_cost: float,
        giou_cost: float = 0.0,
    ) -> None:
        self._init_set_prediction_matcher(
            task_config,
            geometry_cost=geometry_cost,
            giou_cost=giou_cost,
        )
        self._init_anchor_components(
            task_config,
            model_config,
            hidden_size,
            dropout,
            shared_layers or {},
        )
        self.cls_head = ClassificationScorer.from_config(
            scorer_type=getattr(task_config, "scorer_type", "dot"),
            hidden_size=hidden_size,
        )

    def _score_anchor_labels(
        self,
        anchors,
        label_reps,
        label_mask=None,
        anchor_mask=None,
    ):
        return score_anchor_labels(
            self.anchor_modeling,
            self.cls_head,
            anchors,
            label_reps,
            label_mask,
            anchor_mask,
        )

    def _match_geometry(
        self,
        class_logits,
        geometry_preds,
        gold_classes,
        gold_geometry,
        valid_geometry,
        prediction_mask=None,
        **matcher_kwargs,
    ):
        return self.matcher(
            class_logits,
            geometry_preds,
            gold_classes,
            gold_geometry,
            valid_geometry,
            prediction_mask,
            **matcher_kwargs,
        )

    def _default_binary_loss(self, logits, targets, **kwargs):
        return configured_binary_loss(
            self.set_prediction_config,
            logits,
            targets,
            **kwargs,
        )

    def _objectness_loss_kwargs(self) -> dict[str, float]:
        """Return focal overrides owned specifically by objectness."""

        loss_kwargs = {}
        for config_name, loss_name in (
            ("objectness_focal_loss_alpha", "focal_loss_alpha"),
            ("objectness_focal_loss_gamma", "focal_loss_gamma"),
            ("objectness_focal_loss_prob_margin", "focal_loss_prob_margin"),
        ):
            value = getattr(self.set_prediction_config, config_name, None)
            if value is not None:
                loss_kwargs[loss_name] = float(value)
        return loss_kwargs


class MediaSetPredictionDecoder(TaskDecoder):
    """Shared class/objectness decoding for spatial and temporal set heads."""

    logits_name: str = ""
    origin_name: str = ""
    geometry_name: str = ""
    geometry_key: str = "geometry"
    objectness_name: str = ""
    anchor_mask_name: str = ""
    mapping_name: str = ""
    config_attribute: str = ""

    def _prepare_geometry(self, geometry: torch.Tensor) -> torch.Tensor | None:
        geometry = geometry.detach().float()
        return geometry if torch.isfinite(geometry).all() else None

    def decode(
        self,
        model_output,
        classes_mapping=None,
        threshold=0.5,
        multi_label=None,
        **kwargs,
    ):
        logits = getattr(model_output, self.logits_name, None)
        geometry = getattr(model_output, self.geometry_name, None)
        origin = getattr(model_output, self.origin_name, None)
        objectness = getattr(model_output, self.objectness_name, None)
        anchor_mask = getattr(model_output, self.anchor_mask_name, None)
        if logits is None or geometry is None or origin is None:
            return []

        head_config = getattr(self.config, self.config_attribute, None)
        probability_type = getattr(head_config, "class_probability", "sigmoid")
        if multi_label is None:
            multi_label = bool(getattr(head_config, "multi_label", True))
        if multi_label and probability_type != "sigmoid":
            raise ValueError(
                "multi-label set decoding requires class_probability='sigmoid'"
            )
        if probability_type == "sigmoid":
            class_probabilities = torch.sigmoid(logits.float())
        elif probability_type == "softmax":
            class_probabilities = torch.softmax(logits.float(), dim=-1)
        else:
            raise ValueError(
                f"Unsupported {self.config_attribute} class_probability: "
                f"{probability_type!r}"
            )
        objectness_probabilities = (
            torch.sigmoid(objectness.float())
            if objectness is not None
            else torch.ones_like(class_probabilities[..., 0])
        )
        id_to_class_maps = id_maps(classes_mapping, self.mapping_name)
        flat_results = []
        for batch_idx in range(class_probabilities.shape[0]):
            id_to_class = (
                id_to_class_maps[batch_idx]
                if batch_idx < len(id_to_class_maps)
                else {}
            )
            class_count = (
                len(id_to_class)
                if id_to_class
                else class_probabilities.shape[-1]
            )
            predictions = []
            for anchor_idx in range(class_probabilities.shape[1]):
                if (
                    anchor_mask is not None
                    and not bool(anchor_mask[batch_idx, anchor_idx].item())
                ):
                    continue
                scores = (
                    class_probabilities[batch_idx, anchor_idx, :class_count]
                    * objectness_probabilities[batch_idx, anchor_idx]
                )
                if multi_label:
                    class_indices = torch.nonzero(
                        scores > threshold,
                        as_tuple=False,
                    ).flatten().tolist()
                else:
                    best_idx = int(scores.argmax().item())
                    class_indices = (
                        [best_idx] if scores[best_idx] > threshold else []
                    )
                if not class_indices:
                    continue
                decoded_geometry = self._prepare_geometry(
                    geometry[batch_idx, anchor_idx]
                )
                if decoded_geometry is None:
                    continue
                objectness_score = float(
                    objectness_probabilities[batch_idx, anchor_idx].item()
                )
                for class_idx in class_indices:
                    predictions.append(
                        {
                            "label": id_to_class.get(class_idx, str(class_idx)),
                            "score": float(scores[class_idx].item()),
                            "class_score": float(
                                class_probabilities[
                                    batch_idx,
                                    anchor_idx,
                                    class_idx,
                                ].item()
                            ),
                            "objectness_score": objectness_score,
                            self.geometry_key: decoded_geometry.cpu().tolist(),
                            "anchor": anchor_idx,
                        }
                    )
            predictions.sort(
                key=lambda prediction: prediction["score"],
                reverse=True,
            )
            flat_results.append(predictions)
        return unflatten_by_batch_origin(
            flat_results,
            origin,
            model_output.batch_size,
        )


class MediaClassificationHead(TaskHead):
    """Shared pooled classification head for feature-sequence modalities.

    Concrete modality heads only provide ``config_attribute`` and
    ``labels_key``. It composes the same independent anchor components used by
    the project's text classification heads.
    """

    config_attribute: str = ""
    labels_key: str = ""
    dependencies = []

    def __init__(self, config, hidden_size, dropout, shared_layers=None):
        super().__init__()
        head_config = self._head_config(config)
        self.head_config = head_config
        self.loss_coef = head_config.loss_coef
        if shared_layers is None:
            shared_layers = {}
        self._init_anchor_components(
            head_config,
            config,
            hidden_size,
            dropout,
            shared_layers,
        )
        self.pooling = Pooling.from_config(
            pooling_type=getattr(head_config, "pooling_type", "mean"),
            hidden_size=hidden_size,
        )
        self.scorer = ClassificationScorer.from_config(
            scorer_type=getattr(head_config, "scorer_type", "dot"),
            hidden_size=hidden_size,
        )

    @classmethod
    def _head_config(cls, config):
        if not cls.config_attribute:
            raise TypeError(f"{cls.__name__} must define config_attribute")
        return getattr(config, cls.config_attribute)

    @classmethod
    def from_config(cls, config, shared_layers=None, **kwargs):
        if cls._head_config(config) is None:
            return None
        return cls(
            config,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            shared_layers=shared_layers,
        )

    def forward(self, shared, dependency_outputs, flat_inputs=None, **batch):
        labels = batch.get(self.labels_key)
        base_loss_fn = batch.get("base_loss_fn")
        features, feature_mask = flat_features(flat_inputs)
        media_representation = self.pooling(features, feature_mask)
        anchors, anchor_mask = self._generate_anchors(
            flat_inputs.parent_embedding,
            features,
            feature_mask=feature_mask,
        )
        anchors = self._refine_anchors(
            anchors,
            features,
            memory_mask=feature_mask,
            anchor_mask=anchor_mask,
        )
        fused = self._reduce_fused_anchors(
            self._model_anchors(
                anchors,
                flat_inputs.child_embedding,
                anchor_mask=anchor_mask,
                child_mask=flat_inputs.child_mask,
            ),
            anchor_mask,
        )
        logits = self.scorer(media_representation, fused)

        loss = None
        if labels is not None:
            loss_fn = base_loss_fn or (
                lambda logits, targets: configured_binary_loss(
                    self.head_config,
                    logits,
                    targets,
                )
            )
            class_count = min(logits.shape[1], labels.shape[1])
            losses = loss_fn(logits[:, :class_count], labels[:, :class_count])
            valid_classes = flat_inputs.child_mask[:, :class_count].to(losses.dtype)
            loss = (losses * valid_classes).sum() / valid_classes.sum().clamp(
                min=1.0
            )
        return TaskHeadOutput(loss=loss, logits=logits)


class MediaClassificationDecoder(TaskDecoder):
    """Shared sigmoid decoder for image- and audio-level classification."""

    logits_name: str = ""
    origin_name: str = ""
    mapping_name: str = ""

    def decode(
        self,
        model_output,
        classes_mapping=None,
        threshold=0.5,
        multi_label=True,
        **kwargs,
    ) -> list[list[dict]]:
        logits = getattr(model_output, self.logits_name, None)
        origin = getattr(model_output, self.origin_name, None)
        if logits is None or origin is None:
            return []
        if multi_label is None:
            multi_label = True

        probabilities = torch.sigmoid(logits.float())
        id_to_class_maps = id_maps(classes_mapping, self.mapping_name)
        flat_results = []
        for batch_idx in range(probabilities.shape[0]):
            id_to_class = (
                id_to_class_maps[batch_idx]
                if batch_idx < len(id_to_class_maps)
                else {}
            )
            class_count = (
                len(id_to_class) if id_to_class else probabilities.shape[1]
            )
            valid_probabilities = probabilities[batch_idx, :class_count]
            if multi_label:
                class_indices = torch.nonzero(
                    valid_probabilities > threshold,
                    as_tuple=False,
                ).flatten().tolist()
            else:
                best_idx = int(valid_probabilities.argmax().item())
                class_indices = (
                    [best_idx]
                    if valid_probabilities[best_idx] > threshold
                    else []
                )
            predictions = [
                {
                    "label": id_to_class.get(class_idx, str(class_idx)),
                    "score": float(valid_probabilities[class_idx].item()),
                }
                for class_idx in class_indices
            ]
            predictions.sort(key=lambda prediction: prediction["score"], reverse=True)
            flat_results.append(predictions)
        return unflatten_by_batch_origin(
            flat_results,
            origin,
            model_output.batch_size,
        )


__all__ = [
    "MediaClassificationHead",
    "MediaClassificationDecoder",
    "MediaSetPredictionHead",
    "MediaSetPredictionDecoder",
    "flat_features",
    "id_maps",
    "matched_mask_loss",
    "matched_classification_loss",
    "score_anchor_labels",
    "normalized_objectness_loss",
]
