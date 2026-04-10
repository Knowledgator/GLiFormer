"""Classification task decoder — post-processing logits into label predictions."""

from typing import List

import torch

from .. import TaskDecoder


class ClassificationDecoder(TaskDecoder):
    """Decodes classification logits into predicted labels."""

    def __init__(self, config):
        super().__init__(config)
        self.threshold = 0.5

    def decode(self, model_output, classes_mapping=None, threshold=None,
               multi_label=True, **kwargs) -> List[List[dict]]:
        """Decode classification logits into predicted labels.

        Args:
            model_output: GLiNExTOutput with cat_logits (BN, C).
            classes_mapping: BatchClassesMapping for label resolution.
            threshold: Override detection threshold.
            multi_label: If True, return all classes above threshold.
                If False, return only the top-scoring class (if above threshold).

        Returns:
            If batch_origin available: List[List[List[dict]]] — per batch item, per group.
            Otherwise: List[List[dict]] — per group (flat BN).
        """
        if model_output.cat_logits is None:
            return []

        threshold = threshold or self.threshold
        probs = torch.sigmoid(model_output.cat_logits)

        # Build per-flat-group id→name mappings from classes_mapping
        id_to_class_maps = []
        if classes_mapping is not None and hasattr(classes_mapping, 'cat_mapping'):
            for cm in classes_mapping.cat_mapping:
                for base_map in cm.cat_class_to_id:
                    id_to_class_maps.append(base_map.get_reverse_mapping())

        flat_results = []
        for b in range(probs.shape[0]):
            id_to_class = id_to_class_maps[b] if b < len(id_to_class_maps) else {}
            num_classes = len(id_to_class) if id_to_class else probs.shape[1]

            if multi_label:
                predictions = []
                for c in range(num_classes):
                    score = probs[b, c].item()
                    if score > threshold:
                        predictions.append({
                            "class_name": id_to_class.get(c, str(c)),
                            "score": score,
                        })
            else:
                # Single-label: pick the highest-scoring class
                valid_probs = probs[b, :num_classes]
                best_idx = valid_probs.argmax().item()
                best_score = valid_probs[best_idx].item()
                if best_score > threshold:
                    predictions = [{
                        "class_name": id_to_class.get(best_idx, str(best_idx)),
                        "score": best_score,
                    }]
                else:
                    predictions = []

            flat_results.append(predictions)

        # Unflatten BN → B if batch_origin is available
        if model_output.cat_batch_origin is not None and model_output.batch_size is not None:
            from ...processing.decoder import unflatten_by_batch_origin
            return unflatten_by_batch_origin(
                flat_results, model_output.cat_batch_origin, model_output.batch_size,
            )

        return flat_results
