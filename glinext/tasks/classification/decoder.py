"""Classification task decoder — post-processing logits into label predictions."""

from typing import List

import torch

from .. import TaskDecoder


class ClassificationDecoder(TaskDecoder):
    """Decodes classification logits into predicted labels."""

    def __init__(self, config):
        super().__init__(config)
        self.threshold = 0.5

    def decode(self, model_output, classes_mapping=None, threshold=None, **kwargs) -> List[List[dict]]:
        """Decode classification logits into predicted labels.

        Args:
            model_output: GLiNExTOutput with cat_logits (BN, C).
            classes_mapping: BatchClassesMapping for label resolution.
            threshold: Override detection threshold.

        Returns:
            If batch_origin available: List[List[List[dict]]] — per batch item, per group.
            Otherwise: List[List[dict]] — per group (flat BN).
        """
        if model_output.cat_logits is None:
            return []

        threshold = threshold or self.threshold
        probs = torch.sigmoid(model_output.cat_logits)

        flat_results = []
        for b in range(probs.shape[0]):
            predictions = []
            for c in range(probs.shape[1]):
                score = probs[b, c].item()
                if score > threshold:
                    predictions.append({
                        "class_id": c,
                        "score": score,
                    })
            flat_results.append(predictions)

        # Unflatten BN → B if batch_origin is available
        if model_output.cat_batch_origin is not None and model_output.batch_size is not None:
            from ...decoder import unflatten_by_batch_origin
            return unflatten_by_batch_origin(
                flat_results, model_output.cat_batch_origin, model_output.batch_size,
            )

        return flat_results
