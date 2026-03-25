"""Structuring task decoder — post-processing logits into structured output."""

from typing import Dict, List, Optional

import torch


class StructuringDecoder:
    """Decodes structuring logits into field-value assignments per instance."""

    def __init__(self, config):
        self.config = config
        self.threshold = 0.5

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def decode(self, model_output, classes_mapping=None, threshold=None, **kwargs) -> List[List[dict]]:
        """Decode structuring logits into field-value spans per instance.

        Args:
            model_output: GLiNExTOutput with structuring_logits (B, X, L, C, 3).
            classes_mapping: BatchClassesMapping for label resolution.
            threshold: Override detection threshold.

        Returns:
            List of lists of instance dicts per group.
        """
        if model_output.structuring_logits is None:
            return []

        threshold = threshold or self.threshold
        logits = model_output.structuring_logits
        probs = torch.sigmoid(logits)
        anchor_mask = model_output.structuring_anchor_mask

        all_instances = []
        B = probs.shape[0]

        for b in range(B):
            instances = []
            for x in range(probs.shape[1]):
                if anchor_mask is not None and not anchor_mask[b, x]:
                    continue
                fields = []
                for c in range(probs.shape[3]):
                    start_probs = probs[b, x, :, c, 0]
                    end_probs = probs[b, x, :, c, 1]

                    start_positions = (start_probs > threshold).nonzero(as_tuple=True)[0]
                    for start in start_positions:
                        end_candidates = (end_probs[start:] > threshold).nonzero(as_tuple=True)[0]
                        if len(end_candidates) > 0:
                            end = start + end_candidates[0]
                            score = (start_probs[start] + end_probs[end]).item() / 2
                            fields.append({
                                "field_id": c,
                                "start": start.item(),
                                "end": end.item(),
                                "score": score,
                            })
                if fields:
                    instances.append(fields)
            all_instances.append(instances)

        return all_instances
