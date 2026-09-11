"""GLiFormer trainer — extends the GLiNER Trainer for multi-task training."""

import logging
import math
import random
from collections.abc import Mapping
from typing import Any

import torch
from gliner.training.trainer import Trainer as GLiNERTrainer
from gliner.training.trainer import TrainingArguments as TrainingArguments
from torch import nn
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import nested_detach

from .processing.label_augmentation import (
    LABEL_AUGMENTATION_INDEX_KEY,
    LABEL_AUGMENTATION_MARKER_KEY,
)

logger = logging.getLogger(__name__)

class TrainingLabelAugmentationDataset(Dataset):
    """Mark records that may receive training-only label augmentation.

    Hugging Face uses one data collator for both the training and evaluation
    dataloaders.  A dataset marker lets the shared collator distinguish those
    paths without consulting mutable model state.  The stable source index is
    also available to the augmenter when it derives a deterministic batch
    seed.  Source records are never modified.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        record = self.dataset[index]
        if not isinstance(record, Mapping):
            return record

        marked = dict(record)
        marked[LABEL_AUGMENTATION_MARKER_KEY] = True
        marked[LABEL_AUGMENTATION_INDEX_KEY] = int(index)
        return marked


class ClassificationParentNameDropoutDataset(Dataset):
    """Expose named and unnamed singleton classification prompts in training.

    The public inference API represents a single unnamed label group as a
    list and a named group as a mapping.  Dropping the name dynamically lets
    the model see both prompt forms without duplicating records or modifying
    the source dataset.  Multi-group examples retain their names because the
    groups would otherwise be ambiguous and cannot be represented as multiple
    unnamed groups by the public API.
    """

    def __init__(self, dataset, probability: float):
        probability = float(probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                "classification_parent_name_dropout must be between 0 and 1"
            )
        self.dataset = dataset
        self.probability = probability

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        record = self.dataset[index]
        if self.probability == 0.0 or not isinstance(record, Mapping):
            return record

        groups = record.get("classification")
        if not isinstance(groups, list | tuple) or len(groups) != 1:
            return record
        group = groups[0]
        if not isinstance(group, Mapping) or not group.get("name"):
            return record
        if self.probability < 1.0 and random.random() >= self.probability:
            return record

        unnamed_group = dict(group)
        unnamed_group["name"] = None
        augmented = dict(record)
        augmented["classification"] = [unnamed_group]
        return augmented


# Label keys that indicate training data is present in the batch.
# GLiFormer does not use a single "labels" key — each task has its own.
_LABEL_KEYS = frozenset(
    {
        # Text tasks
        "ner_labels",
        "span_labels",
        "cat_labels",
        "rel_labels",
        "open_rel_entity_labels",
        "open_rel_labels",
        "open_rel_assignment_labels",
        "open_rel_count",
        "structuring_labels",
        "structuring_span_labels",
        "structuring_count",
        "structuring_relation_labels",
        "embedding_labels",
        "count_targets",
        # Vision tasks
        "image_classification_labels",
        "object_detection_class_labels",
        "object_detection_bbox_labels",
        "object_detection_object_mask",
        "segmentation_class_labels",
        "segmentation_bbox_labels",
        "segmentation_object_mask",
        "segmentation_mask_labels",
        # Audio tasks
        "audio_classification_labels",
        "audio_segmentation_class_labels",
        "audio_segmentation_segment_labels",
        "audio_segmentation_object_mask",
        "audio_segmentation_mask_labels",
    }
)


# Ordered prediction fields associated with each supervision key.  GLiFormer's
# public outputs intentionally use task-specific names rather than one ambiguous
# ``logits`` field; evaluation therefore has to select the relevant fields from
# the labels present in the batch.  Geometry/objectness/masks are included so a
# detector metric can evaluate the same values exposed by inference.
_PREDICTION_FIELDS_BY_LABEL = {
    "ner_labels": ("ner_logits",),
    "span_labels": ("span_logits",),
    "cat_labels": ("cat_logits",),
    "rel_labels": ("joint_rel_logits",),
    "open_rel_entity_labels": ("open_rel_entity_logits",),
    "open_rel_labels": (
        "open_rel_logits",
        "open_rel_objectness_logits",
        "open_rel_anchor_mask",
    ),
    "open_rel_assignment_labels": (
        "open_rel_assignment_logits",
        "open_rel_span_idx",
        "open_rel_span_mask",
    ),
    "open_rel_count": (
        "open_rel_objectness_logits",
        "open_rel_anchor_mask",
    ),
    "structuring_labels": (
        "structuring_entity_logits",
        "structuring_field_logits",
    ),
    "structuring_span_labels": (
        "structuring_assignment_logits",
        "structuring_span_idx",
        "structuring_span_mask",
        "structuring_objectness_logits",
        "structuring_anchor_mask",
    ),
    "structuring_count": (
        "structuring_objectness_logits",
        "structuring_anchor_mask",
    ),
    "structuring_relation_labels": (
        "structuring_anchor_relation_scores",
    ),
    "embedding_labels": ("embedding_logits",),
    "count_targets": ("count_logits",),
    "image_classification_labels": ("image_classification_logits",),
    "object_detection_class_labels": (
        "object_detection_logits",
        "object_detection_boxes",
        "object_detection_objectness_logits",
        "object_detection_anchor_mask",
    ),
    "object_detection_bbox_labels": (
        "object_detection_logits",
        "object_detection_boxes",
        "object_detection_objectness_logits",
        "object_detection_anchor_mask",
    ),
    "object_detection_object_mask": (
        "object_detection_logits",
        "object_detection_boxes",
        "object_detection_objectness_logits",
        "object_detection_anchor_mask",
    ),
    "segmentation_class_labels": (
        "segmentation_logits",
        "segmentation_boxes",
        "segmentation_objectness_logits",
        "segmentation_anchor_mask",
        "segmentation_mask_logits",
        "segmentation_mask_validity",
    ),
    "segmentation_bbox_labels": (
        "segmentation_logits",
        "segmentation_boxes",
        "segmentation_objectness_logits",
        "segmentation_anchor_mask",
        "segmentation_mask_logits",
        "segmentation_mask_validity",
    ),
    "segmentation_object_mask": (
        "segmentation_logits",
        "segmentation_boxes",
        "segmentation_objectness_logits",
        "segmentation_anchor_mask",
        "segmentation_mask_logits",
        "segmentation_mask_validity",
    ),
    "segmentation_mask_labels": (
        "segmentation_logits",
        "segmentation_boxes",
        "segmentation_objectness_logits",
        "segmentation_anchor_mask",
        "segmentation_mask_logits",
        "segmentation_mask_validity",
    ),
    "audio_classification_labels": ("audio_classification_logits",),
    "audio_segmentation_class_labels": (
        "audio_segmentation_logits",
        "audio_segmentation_segments",
        "audio_segmentation_objectness_logits",
        "audio_segmentation_anchor_mask",
        "audio_segmentation_mask_logits",
    ),
    "audio_segmentation_segment_labels": (
        "audio_segmentation_logits",
        "audio_segmentation_segments",
        "audio_segmentation_objectness_logits",
        "audio_segmentation_anchor_mask",
        "audio_segmentation_mask_logits",
    ),
    "audio_segmentation_object_mask": (
        "audio_segmentation_logits",
        "audio_segmentation_segments",
        "audio_segmentation_objectness_logits",
        "audio_segmentation_anchor_mask",
        "audio_segmentation_mask_logits",
    ),
    "audio_segmentation_mask_labels": (
        "audio_segmentation_logits",
        "audio_segmentation_segments",
        "audio_segmentation_objectness_logits",
        "audio_segmentation_anchor_mask",
        "audio_segmentation_mask_logits",
    ),
}

_STRUCTURING_RELATION_PREDICTION_FIELDS = frozenset(
    {"structuring_anchor_relation_scores"}
)

_OPEN_REL_ASSIGNMENT_PREDICTION_FIELDS = frozenset(
    {"open_rel_assignment_logits"}
)


def _present_label_keys(inputs: dict[str, Any]) -> tuple[str, ...]:
    """Return task supervision keys that carry a value in this batch."""
    return tuple(sorted(key for key in _LABEL_KEYS if inputs.get(key) is not None))


def _get_output_value(outputs: Any, name: str) -> Any:
    value = getattr(outputs, name, None)
    if value is None and isinstance(outputs, dict):
        value = outputs.get(name)
    return value


def _prediction_values(
    outputs: Any,
    label_keys: tuple[str, ...],
    ignore_keys: list[str] | None,
) -> Any:
    """Extract task-relevant predictions while preserving deterministic order."""

    ignored = set(ignore_keys or ())
    field_names = []
    for label_key in label_keys:
        for field_name in _PREDICTION_FIELDS_BY_LABEL.get(label_key, ()):
            if field_name not in ignored and field_name not in field_names:
                field_names.append(field_name)

    values = []
    for field_name in field_names:
        value = _get_output_value(outputs, field_name)
        if value is None:
            continue
        if (
            field_name in _STRUCTURING_RELATION_PREDICTION_FIELDS
            and isinstance(value, torch.Tensor)
            and value.ndim >= 3
        ):
            # HF's evaluation accumulator pads only dimension 1. Flatten the
            # two anchor axes so batches with different anchor counts can be
            # concatenated. This preserves the model's row-major slot order;
            # it does not attempt to apply training-time anchor matching.
            value = value.flatten(start_dim=1)
        elif (
            field_name in _OPEN_REL_ASSIGNMENT_PREDICTION_FIELDS
            and isinstance(value, torch.Tensor)
            and value.ndim == 4
        ):
            # Assignment logits are canonically (BN, A, E, 2), where both
            # anchor count A and recognized-entity count E can vary by batch.
            # Hugging Face pads only dimension 1 during evaluation, so flatten
            # both variable axes there while retaining the endpoint roles.
            value = value.flatten(start_dim=1, end_dim=-2)
        values.append(value)
    if not values:
        fallback = _get_output_value(outputs, "logits")
        return None if "logits" in ignored else fallback
    return values[0] if len(values) == 1 else tuple(values)


class GLiFormerTrainer(GLiNERTrainer):
    """Trainer for GLiFormer multi-task model.

    Differences from the base GLiNER Trainer:

    * **Label check**: The base trainer requires a ``labels`` key in every
      batch.  GLiFormer uses task-specific label keys (``ner_labels``,
      ``cat_labels``, etc.), so the guardrail is adjusted accordingly.

    * **Column removal disabled**: HuggingFace Trainer strips batch keys
      that don't appear in the model's ``forward`` signature.  GLiFormer
      passes ``classes_mapping`` (a non-tensor object) through ``**kwargs``,
      so column removal is disabled to keep it intact.

    * **Freezable heads**: Works with
      :meth:`GLiFormer._get_freezable_components` which exposes individual
      task heads for selective freezing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Prevent HF Trainer from stripping keys not in forward() signature
        # (classes_mapping, tokens, etc. travel through **kwargs)
        self.args.remove_unused_columns = False

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        """Call the model with the same focal-loss names used by task configs."""

        del num_items_in_batch
        rel_focal_loss_alpha = (
            self.args.rel_focal_loss_alpha
            if self.args.rel_focal_loss_alpha is not None
            else self.args.focal_loss_alpha
        )
        rel_focal_loss_gamma = (
            self.args.rel_focal_loss_gamma
            if self.args.rel_focal_loss_gamma is not None
            else self.args.focal_loss_gamma
        )
        outputs = model(
            focal_loss_alpha=self.args.focal_loss_alpha,
            focal_loss_gamma=self.args.focal_loss_gamma,
            rel_focal_loss_alpha=rel_focal_loss_alpha,
            rel_focal_loss_gamma=rel_focal_loss_gamma,
            focal_loss_prob_margin=self.args.focal_loss_prob_margin,
            label_smoothing=self.args.label_smoothing,
            reduction=self.args.loss_reduction,
            negatives=self.args.negatives,
            masking=self.args.masking,
            **inputs,
        )
        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        """Training step with multi-task label validation.

        Replaces the base trainer's ``labels`` key check with a check for
        any of the task-specific label keys.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # A malformed or unresolvable annotation must not terminate a long
        # multi-task run.  The processors normally filter these records, but
        # retaining this last-line guard makes training robust to new source
        # formats and rare alignment failures.
        label_keys = _present_label_keys(inputs)
        if not label_keys:
            skipped = getattr(self, "_label_free_batches_skipped", 0) + 1
            self._label_free_batches_skipped = skipped
            logger.warning(
                "Skipping label-free training batch #%d. Expected at least "
                "one of %s; got keys: %s",
                skipped,
                sorted(_LABEL_KEYS),
                sorted(inputs),
            )
            return torch.zeros((), device=self.args.device)

        try:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

            if loss is None:
                raise RuntimeError(
                    "Model returned loss=None. Check that at least one task "
                    "head received labels and computed a loss."
                )

            if self.args.n_gpu > 1:
                loss = loss.mean()

            if (
                self.args.gradient_accumulation_steps > 1
                and getattr(self, "deepspeed", None) is None
            ):
                loss = loss / self.args.gradient_accumulation_steps

            self.accelerator.backward(loss)
            return loss.detach()

        except torch.cuda.OutOfMemoryError as e:
            logger.warning("Skipping batch due to CUDA OOM: %s", e)
            model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return torch.zeros((), device=self.args.device)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("Skipping batch due to OOM RuntimeError: %s", e)
                model.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return torch.zeros((), device=self.args.device)
            raise

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, Any | None]:
        """Evaluate batches supervised by any GLiFormer task label.

        Hugging Face cannot infer ``label_names`` from GLiFormer's
        ``forward(*args, **kwargs)`` signatures.  Detect supervision from the
        batch instead so evaluation always executes ``compute_loss`` and
        reports ``eval_loss`` for text, vision, audio, and mixed-task data.
        """
        model.eval()
        inputs = self._prepare_inputs(inputs)
        label_keys = _present_label_keys(inputs)

        with torch.no_grad():
            if label_keys:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                if loss is None:
                    raise RuntimeError(
                        "Model returned loss=None during evaluation despite receiving "
                        f"task labels: {list(label_keys)}"
                    )
                loss = loss.detach().mean()
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)

        if prediction_loss_only:
            return (loss, None, None)

        logits = _prediction_values(outputs, label_keys, ignore_keys)
        logits = nested_detach(logits) if logits is not None else None

        labels = {key: nested_detach(inputs[key]) for key in label_keys}
        assignment_labels = labels.get("open_rel_assignment_labels")
        if (
            isinstance(assignment_labels, torch.Tensor)
            and assignment_labels.ndim == 4
        ):
            # Gold targets are (BN, G, E, 2); both G and E may vary.
            labels["open_rel_assignment_labels"] = (
                assignment_labels.flatten(start_dim=1, end_dim=-2)
            )
        relation_labels = labels.get("structuring_relation_labels")
        if isinstance(relation_labels, torch.Tensor) and relation_labels.ndim >= 3:
            labels["structuring_relation_labels"] = relation_labels.flatten(start_dim=1)
        if "structuring_relation_labels" in labels:
            relation_group_mask = inputs.get("structuring_relation_group_mask")
            if relation_group_mask is not None:
                # Auxiliary metric metadata, deliberately not a supervision
                # key: its presence alone must not make a batch "labelled".
                labels["structuring_relation_group_mask"] = nested_detach(relation_group_mask)
        return (loss, logits, labels or None)
