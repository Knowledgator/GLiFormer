"""GLiNExT trainer — extends the GLiNER Trainer for multi-task training."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from gliner.training.trainer import Trainer as GLiNERTrainer, TrainingArguments

logger = logging.getLogger(__name__)

# Label keys that indicate training data is present in the batch.
# GLiNExT does not use a single "labels" key — each task has its own.
_LABEL_KEYS = frozenset({
    "ner_labels", "cat_labels", "rel_labels",
    "open_rel_labels", "structuring_labels",
    "embedding_labels", "count_targets",
})


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
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """Training step with multi-task label validation.

        Replaces the base trainer's ``labels`` key check with a check for
        any of the task-specific label keys.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Guardrail: at least one task-specific label key must be present
        has_labels = any(k in inputs for k in _LABEL_KEYS)
        if not has_labels:
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

            if self.args.gradient_accumulation_steps > 1 and not hasattr(self, 'deepspeed'):
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
