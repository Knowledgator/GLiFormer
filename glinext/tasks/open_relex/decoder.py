"""Open relex decoder — post-processing logits into relation triples with spans."""

from typing import List

import torch


class OpenRelexDecoder:
    """Decodes anchor-based relation logits into triples with head/tail spans.

    Input logits shape: (B, X, C, L, 2, 3)
        X = anchors, C = rel classes, L = seq len, 2 = [head, tail], 3 = [start, inside, end]
    """

    def __init__(self, config):
        self.config = config
        self.threshold = 0.5

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def _extract_span(self, probs_sie, threshold):
        """Extract a span from start/inside/end probabilities.

        Args:
            probs_sie: (L, 3) — start/inside/end probabilities for one span.
            threshold: minimum probability to consider.

        Returns:
            (start, end, score) or None if no valid span found.
        """
        start_probs = probs_sie[:, 0]
        end_probs = probs_sie[:, 1]

        # Find best start position above threshold
        start_mask = start_probs > threshold
        if not start_mask.any():
            return None

        start_idx = start_probs.argmax().item()
        if start_probs[start_idx] < threshold:
            return None

        # Find best end position at or after start
        end_scores = end_probs.clone()
        end_scores[:start_idx] = -1.0
        end_idx = end_scores.argmax().item()
        if end_probs[end_idx] < threshold:
            return None

        score = (start_probs[start_idx] + end_probs[end_idx]).item() / 2.0
        return start_idx, end_idx, score

    def decode(self, model_output, classes_mapping=None, threshold=None, **kwargs) -> List[List[dict]]:
        """Decode open relex logits into relation triples with spans.

        Returns:
            List of lists of dicts per batch item:
            [{anchor_id, rel_class_id, head_start, head_end, tail_start, tail_end, score}]
        """
        if model_output.open_rel_logits is None:
            return []

        threshold = threshold or self.threshold
        logits = model_output.open_rel_logits  # (B, X, C, L, 2, 3)
        probs = torch.sigmoid(logits)
        anchor_mask = model_output.open_rel_anchor_mask  # (B, X)

        all_results = []
        B, X, C, L, _, _ = probs.shape

        for b in range(B):
            results = []
            for x in range(X):
                if anchor_mask is not None and not anchor_mask[b, x]:
                    continue
                for c in range(C):
                    head_probs = probs[b, x, c, :, 0, :]  # (L, 3)
                    tail_probs = probs[b, x, c, :, 1, :]  # (L, 3)

                    head_span = self._extract_span(head_probs, threshold)
                    tail_span = self._extract_span(tail_probs, threshold)

                    if head_span is not None and tail_span is not None:
                        score = (head_span[2] + tail_span[2]) / 2.0
                        results.append({
                            "anchor_id": x,
                            "rel_class_id": c,
                            "head_start": head_span[0],
                            "head_end": head_span[1],
                            "tail_start": tail_span[0],
                            "tail_end": tail_span[1],
                            "score": score,
                        })
            all_results.append(results)

        return all_results
