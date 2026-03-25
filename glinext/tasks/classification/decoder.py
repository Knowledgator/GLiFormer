"""Classification task decoder — post-processing logits into label predictions."""

from typing import Dict, List, Optional

import torch


class ClassificationDecoder:
    """Decodes classification logits into predicted labels."""

    def __init__(self, config):
        self.config = config
        self.threshold = 0.5

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def decode(self, model_output, classes_mapping=None, threshold=None, **kwargs) -> List[List[dict]]:
        """Decode classification logits into predicted labels.

        Args:
            model_output: GLiNExTOutput with cat_logits (B*N, C).
            classes_mapping: BatchClassesMapping for label resolution.
            threshold: Override detection threshold.

        Returns:
            List of lists of label dicts: [{class_id, score}, ...] per group.
        """
        if model_output.cat_logits is None:
            return []

        threshold = threshold or self.threshold
        probs = torch.sigmoid(model_output.cat_logits)

        all_predictions = []
        for b in range(probs.shape[0]):
            predictions = []
            for c in range(probs.shape[1]):
                score = probs[b, c].item()
                if score > threshold:
                    predictions.append({
                        "class_id": c,
                        "score": score,
                    })
            all_predictions.append(predictions)

        return all_predictions
