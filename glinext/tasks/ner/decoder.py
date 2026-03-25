"""NER task decoder — post-processing logits into entity spans.

Follows GLiNER TokenDecoder pattern: supports both token-level BIO decoding
and span-level decoding (when represent_spans is enabled in NERHead).
"""

from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import torch


@dataclass
class Span:
    """Detected entity span."""
    start: int
    end: int
    entity_type: str
    score: float
    class_probs: Optional[Dict[str, float]] = None


def _has_overlapping(idx1, idx2, multi_label=False):
    if idx1[:2] == idx2[:2]:
        return not multi_label
    return not (idx1[0] > idx2[1] or idx2[0] > idx1[1])


def _has_overlapping_nested(idx1, idx2, multi_label=False):
    if idx1[:2] == idx2[:2]:
        return not multi_label
    nested = ((idx1[0] <= idx2[0] and idx1[1] >= idx2[1]) or
              (idx2[0] <= idx1[0] and idx2[1] >= idx1[1]))
    return not ((idx1[0] > idx2[1] or idx2[0] > idx1[1]) or nested)


class NERDecoder:
    """Decodes NER logits into entity spans with labels.

    Two decoding modes:
    1. Token-level BIO: uses ner_logits (B*N, L, C+1, 3) with start/end/inside
    2. Span-level: uses span_logits (B, S, C) + span_idx (B, S, 2) + span_mask (B, S)
    """

    def __init__(self, config):
        self.config = config
        self.threshold = 0.5

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def _get_id_to_class(self, id_to_classes, sample_idx):
        if isinstance(id_to_classes, list):
            return id_to_classes[sample_idx]
        return id_to_classes

    def _get_indices_above_threshold(self, scores: torch.Tensor, threshold: float):
        """Get (position, class) indices where sigmoid(scores) > threshold."""
        scores = torch.sigmoid(scores)
        return [k.tolist() for k in torch.where(scores > threshold)]

    def _calculate_span_score(
        self,
        start_idx: tuple,
        end_idx: tuple,
        scores_inside: torch.Tensor,
        scores_start: torch.Tensor,
        scores_end: torch.Tensor,
        id_to_classes: Dict[int, str],
        threshold: float,
    ) -> List[Span]:
        """Match start/end positions of the same class, validate inside scores."""
        spans = []
        for st, cls_st in zip(*start_idx):
            for ed, cls_ed in zip(*end_idx):
                if ed >= st and cls_st == cls_ed:
                    ins = scores_inside[st:ed + 1, cls_st]
                    if (ins < threshold).any():
                        continue
                    start_score = scores_start[st, cls_st]
                    end_score = scores_end[ed, cls_ed]
                    combined = torch.cat([ins, start_score.unsqueeze(0), end_score.unsqueeze(0)])
                    score = combined.min().item()
                    # id_to_classes uses 1-indexed keys (0 is parent)
                    entity_type = id_to_classes.get(cls_st + 1, str(cls_st))
                    spans.append(Span(
                        start=st,
                        end=ed,
                        entity_type=entity_type,
                        score=score,
                    ))
        return spans

    def _decode_token_level(
        self,
        logits: torch.Tensor,
        id_to_classes: Union[Dict[int, str], List[Dict[int, str]]],
        batch_size: int,
        flat_ner: bool,
        threshold: float,
        multi_label: bool,
    ) -> List[List[Span]]:
        """Decode from token-level BIO logits (B*N, L, C+1, 3)."""
        # Permute to (3, B*N, L, C+1) so we can split start/end/inside
        model_output = logits.permute(3, 0, 1, 2)
        scores_start, scores_end, scores_inside = model_output

        all_spans = []
        for i in range(batch_size):
            id_to_class_i = self._get_id_to_class(id_to_classes, i)
            span_scores = self._calculate_span_score(
                self._get_indices_above_threshold(scores_start[i], threshold),
                self._get_indices_above_threshold(scores_end[i], threshold),
                torch.sigmoid(scores_inside[i]),
                torch.sigmoid(scores_start[i]),
                torch.sigmoid(scores_end[i]),
                id_to_class_i,
                threshold,
            )
            all_spans.append(self.greedy_search(span_scores, flat_ner, multi_label))
        return all_spans

    def _decode_from_spans(
        self,
        span_logits: torch.Tensor,
        span_idx: torch.Tensor,
        span_mask: torch.Tensor,
        id_to_classes: Union[Dict[int, str], List[Dict[int, str]]],
        flat_ner: bool,
        threshold: float,
        multi_label: bool,
    ) -> List[List[Span]]:
        """Decode from span-level predictions (B, S, C)."""
        batch_size = span_logits.size(0)
        span_probs = torch.sigmoid(span_logits)
        all_spans = []

        for i in range(batch_size):
            id_to_class_i = self._get_id_to_class(id_to_classes, i)
            spans = []
            valid_indices = torch.where(span_mask[i])[0]

            for span_pos in valid_indices:
                span_start = span_idx[i, span_pos, 0].item()
                span_end = span_idx[i, span_pos, 1].item()
                probs = span_probs[i, span_pos]
                class_indices = torch.where(probs > threshold)[0]

                for class_idx in class_indices:
                    class_id = class_idx.item() + 1  # 1-indexed (0 is parent)
                    if class_id in id_to_class_i:
                        spans.append(Span(
                            start=span_start,
                            end=span_end,
                            entity_type=id_to_class_i[class_id],
                            score=probs[class_idx].item(),
                        ))

            all_spans.append(self.greedy_search(spans, flat_ner, multi_label))
        return all_spans

    def greedy_search(
        self, spans: List[Span], flat_ner: bool = True, multi_label: bool = False,
    ) -> List[Span]:
        """Remove overlapping spans, keeping highest-scoring ones."""
        if not spans:
            return []

        has_ov = partial(
            _has_overlapping if flat_ner else _has_overlapping_nested,
            multi_label=multi_label,
        )

        selected = []
        selected_tuples = []
        for span in sorted(spans, key=lambda x: -x.score):
            span_tuple = (span.start, span.end, span.entity_type)
            if not any(has_ov(span_tuple, existing) for existing in selected_tuples):
                selected.append(span)
                selected_tuples.append(span_tuple)

        selected.sort(key=lambda x: x.start)
        return selected

    def decode(
        self,
        model_output,
        classes_mapping=None,
        threshold=None,
        flat_ner=True,
        multi_label=False,
        **kwargs,
    ) -> List[List[Span]]:
        """Decode NER predictions.

        Automatically selects span-level decoding when span_logits are available,
        otherwise falls back to token-level BIO decoding.

        Args:
            model_output: GLiNExTOutput with ner_logits and optionally span_logits/span_idx/span_mask.
            classes_mapping: BatchClassesMapping for label resolution (id_to_classes).
            threshold: Detection threshold (default 0.5).
            flat_ner: If True, enforce non-overlapping spans.
            multi_label: If True, allow multiple labels per span position.

        Returns:
            List of lists of Span objects per group.
        """
        if model_output.ner_logits is None and model_output.span_logits is None:
            return []

        threshold = threshold or self.threshold
        id_to_classes = classes_mapping if classes_mapping is not None else {}

        # Prefer span-level decoding when available
        if (model_output.span_logits is not None
                and model_output.span_idx is not None
                and model_output.span_mask is not None):
            return self._decode_from_spans(
                span_logits=model_output.span_logits,
                span_idx=model_output.span_idx,
                span_mask=model_output.span_mask,
                id_to_classes=id_to_classes,
                flat_ner=flat_ner,
                threshold=threshold,
                multi_label=multi_label,
            )

        # Fall back to token-level BIO decoding
        logits = model_output.ner_logits
        batch_size = logits.shape[0]
        return self._decode_token_level(
            logits=logits,
            id_to_classes=id_to_classes,
            batch_size=batch_size,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
        )