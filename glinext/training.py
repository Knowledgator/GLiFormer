"""GLiNExT trainer — extends the GLiNER Trainer for multi-task training."""

import logging
from typing import Any

import torch
from gliner.training.trainer import Trainer as GLiNERTrainer
from gliner.training.trainer import TrainingArguments as TrainingArguments
from torch import nn
from transformers.trainer_pt_utils import nested_detach

logger = logging.getLogger(__name__)

# Label keys that indicate training data is present in the batch.
# GLiNExT does not use a single "labels" key — each task has its own.
_LABEL_KEYS = frozenset(
    {
        # Text tasks
        "ner_labels",
        "span_labels",
        "cat_labels",
        "rel_labels",
        "open_rel_labels",
        "open_rel_span_labels",
        "structuring_labels",
        "structuring_span_labels",
        "structuring_count",
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


# Ordered prediction fields associated with each supervision key.  GLiNExT's
# public outputs intentionally use task-specific names rather than one ambiguous
# ``logits`` field; evaluation therefore has to select the relevant fields from
# the labels present in the batch.  Geometry/objectness/masks are included so a
# detector metric can evaluate the same values exposed by inference.
_PREDICTION_FIELDS_BY_LABEL = {
    "ner_labels": ("ner_logits",),
    "span_labels": ("span_logits",),
    "cat_labels": ("cat_logits",),
    "rel_labels": ("joint_rel_logits",),
    "open_rel_labels": ("open_rel_logits",),
    "open_rel_span_labels": ("open_rel_span_logits",),
    "structuring_labels": ("structuring_logits", "structuring_objectness_logits"),
    "structuring_span_labels": ("structuring_span_logits",),
    "structuring_count": ("count_logits",),
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

    values = [
        value
        for field_name in field_names
        if (value := _get_output_value(outputs, field_name)) is not None
    ]
    if not values:
        fallback = _get_output_value(outputs, "logits")
        return None if "logits" in ignored else fallback
    return values[0] if len(values) == 1 else tuple(values)


class GLiNExTTrainer(GLiNERTrainer):
    """Trainer for GLiNExT multi-task model.

    Differences from the base GLiNER Trainer:

    * **Label check**: The base trainer requires a ``labels`` key in every
      batch.  GLiNExT uses task-specific label keys (``ner_labels``,
      ``cat_labels``, etc.), so the guardrail is adjusted accordingly.

    * **Column removal disabled**: HuggingFace Trainer strips batch keys
      that don't appear in the model's ``forward`` signature.  GLiNExT
      passes ``classes_mapping`` (a non-tensor object) through ``**kwargs``,
      so column removal is disabled to keep it intact.

    * **Freezable heads**: Works with
      :meth:`GLiNExT._get_freezable_components` which exposes individual
      task heads for selective freezing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Prevent HF Trainer from stripping keys not in forward() signature
        # (classes_mapping, tokens, etc. travel through **kwargs)
        self.args.remove_unused_columns = False

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

        # Guardrail: at least one task-specific label key must be present
        label_keys = _present_label_keys(inputs)
        if not label_keys:
            raise KeyError(
                f"Batch has no task label keys. Expected at least one of "
                f"{sorted(_LABEL_KEYS)}. Got keys: {sorted(inputs.keys())}"
            )

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
        """Evaluate batches supervised by any GLiNExT task label.

        Hugging Face cannot infer ``label_names`` from GLiNExT's
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
        return (loss, logits, labels or None)
