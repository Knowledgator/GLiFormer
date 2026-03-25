"""Joint relex decoder — post-processing logits into relation triples."""

from typing import List

import torch


class JointRelexDecoder:
    """Decodes relation logits into (head, relation, tail) triples."""

    def __init__(self, config):
        self.config = config
        self.threshold = 0.5

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def decode(self, model_output, classes_mapping=None, threshold=None, **kwargs) -> List[List[dict]]:
        """Decode relation logits into triples.

        Args:
            model_output: GLiNExTOutput with joint_rel_logits, joint_rel_idx, joint_rel_mask.
            classes_mapping: BatchClassesMapping for label resolution.
            threshold: Override detection threshold.

        Returns:
            List of lists of triple dicts per group.
        """
        if model_output.joint_rel_logits is None or model_output.joint_rel_idx is None:
            return []

        threshold = threshold or self.threshold
        probs = torch.sigmoid(model_output.joint_rel_logits)
        pair_idx = model_output.joint_rel_idx
        pair_mask = model_output.joint_rel_mask

        all_triples = []
        B = probs.shape[0]

        for b in range(B):
            triples = []
            for p in range(probs.shape[1]):
                if pair_mask is not None and not pair_mask[b, p]:
                    continue
                for c in range(probs.shape[2]):
                    score = probs[b, p, c].item()
                    if score > threshold:
                        triples.append({
                            "head_id": pair_idx[b, p, 0].item(),
                            "tail_id": pair_idx[b, p, 1].item(),
                            "rel_class_id": c,
                            "score": score,
                        })
            all_triples.append(triples)

        return all_triples
