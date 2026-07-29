"""Shared span decoding utilities for BIO-based extraction tasks.

Provides SpanDecoder base class with:
- BIO token-level span extraction (start/inside/end)
- Span-level decoding (pre-computed spans + class logits)
- Greedy overlap removal (flat and nested modes)
- Span text resolution
"""

from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import torch

from . import TaskDecoder


@dataclass
class Span:
    """Detected span with type and score."""
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


class SpanDecoder(TaskDecoder):
    """Base class for BIO-based span decoding.

    Provides shared logic for extracting spans from (L, C, 3) BIO logits,
    greedy overlap removal, and span text resolution. Used by NER, open relex,
    and structuring decoders.
    """

    def __init__(self, config):
        super().__init__(config)
        self.threshold = 0.5

    def decode(self, model_output, classes_mapping=None, **kwargs):
        raise NotImplementedError("Subclasses must implement decode()")

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
        """Match start/end positions and validate inside/boundary scores.

        ``inside`` is trained as zero immediately outside a gold span.  Using
        that signal when ranking candidates prevents a high-scoring suffix
        from suppressing the complete value.  This is especially important
        for punctuation-heavy values: a model can predict both ``$1,000`` and
        ``000``, with the shorter suffix receiving a marginally higher local
        score even though the comma immediately before it is confidently
        inside the full value.
        """
        spans = []
        for st, cls_st in zip(*start_idx):
            for ed, cls_ed in zip(*end_idx):
                if ed >= st and cls_st == cls_ed:
                    ins = scores_inside[st:ed + 1, cls_st]
                    if (ins < threshold).any():
                        continue
                    start_score = scores_start[st, cls_st]
                    end_score = scores_end[ed, cls_ed]
                    score_parts = [
                        ins,
                        start_score.unsqueeze(0),
                        end_score.unsqueeze(0),
                    ]
                    if st > 0:
                        score_parts.append(
                            (1.0 - scores_inside[st - 1, cls_st]).unsqueeze(0)
                        )
                    if ed + 1 < scores_inside.shape[0]:
                        score_parts.append(
                            (1.0 - scores_inside[ed + 1, cls_st]).unsqueeze(0)
                        )
                    combined = torch.cat(score_parts)
                    score = combined.min().item()
                    if id_to_classes and cls_st not in id_to_classes:
                        continue
                    entity_type = id_to_classes.get(cls_st, str(cls_st))
                    spans.append(Span(
                        start=st,
                        end=ed,
                        entity_type=entity_type,
                        score=score,
                    ))
        return spans

    def decode_bio_spans(
        self,
        logits: torch.Tensor,
        id_to_classes: Dict[int, str],
        threshold: float,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> List[Span]:
        """Decode BIO logits for a single sample into spans.

        Args:
            logits: (L, C, 3) — sequence length, num classes, start/end/inside.
            id_to_classes: mapping from class id (1-indexed) to class name.
            threshold: minimum probability for detection.
            flat_ner: enforce non-overlapping spans.
            multi_label: allow multiple labels per span position.

        Returns:
            List of Span objects after greedy overlap removal.
        """
        # Permute to (3, L, C)
        model_output = logits.permute(2, 0, 1)
        scores_start, scores_end, scores_inside = model_output

        span_scores = self._calculate_span_score(
            self._get_indices_above_threshold(scores_start, threshold),
            self._get_indices_above_threshold(scores_end, threshold),
            torch.sigmoid(scores_inside),
            torch.sigmoid(scores_start),
            torch.sigmoid(scores_end),
            id_to_classes,
            threshold,
        )
        return self.greedy_search(span_scores, flat_ner, multi_label)

    def decode_bio_spans_single_class(
        self,
        probs: torch.Tensor,
        threshold: float,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> Optional[List[Span]]:
        """Decode BIO probabilities for a single class (e.g. one role in relex).

        Args:
            probs: (L, 3) — already sigmoid'd probabilities for start/end/inside.
            threshold: minimum probability.
            flat_ner: enforce non-overlapping spans.
            multi_label: allow multiple labels per position.

        Returns:
            List of Span objects, or None if no spans found.
        """
        scores_start = probs[:, 0]   # (L,)
        scores_end = probs[:, 1]     # (L,)
        scores_inside = probs[:, 2] if probs.shape[1] > 2 else probs[:, 0]  # (L,)

        # _get_indices_above_threshold expects raw logits, convert back
        start_idx = self._get_indices_above_threshold(
            scores_start.unsqueeze(-1).logit(), threshold,
        )
        end_idx = self._get_indices_above_threshold(
            scores_end.unsqueeze(-1).logit(), threshold,
        )

        # Single class: id_to_classes maps 0→""
        spans = self._calculate_span_score(
            start_idx, end_idx,
            scores_inside.unsqueeze(-1),
            scores_start.unsqueeze(-1),
            scores_end.unsqueeze(-1),
            {0: ""},
            threshold,
        )
        if not spans:
            return None
        return self.greedy_search(spans, flat_ner, multi_label)

    def decode_bio_spans_batch(
        self,
        logits: torch.Tensor,
        id_to_classes: Union[Dict[int, str], List[Dict[int, str]]],
        batch_size: int,
        threshold: float,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> List[List[Span]]:
        """Decode BIO logits for a batch into spans.

        Args:
            logits: (B, L, C, 3) — batch, seq length, num classes, start/end/inside.
            id_to_classes: per-sample or shared class mapping.
            batch_size: number of samples.
            threshold: minimum probability.
            flat_ner: enforce non-overlapping spans.
            multi_label: allow multiple labels per span position.

        Returns:
            List of lists of Span objects per sample.
        """
        all_spans = []
        for i in range(batch_size):
            id_to_class_i = self._get_id_to_class(id_to_classes, i)
            spans = self.decode_bio_spans(
                logits[i], id_to_class_i, threshold, flat_ner, multi_label,
            )
            all_spans.append(spans)
        return all_spans

    def decode_span_level(
        self,
        span_logits: torch.Tensor,
        span_idx: torch.Tensor,
        span_mask: torch.Tensor,
        id_to_classes: Union[Dict[int, str], List[Dict[int, str]]],
        threshold: float,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> List[List[Span]]:
        """Decode from pre-computed span representations (B, S, C).

        Args:
            span_logits: (B, S, C) — logits per span per class.
            span_idx: (B, S, 2) — start/end token indices per span.
            span_mask: (B, S) — valid span mask.
            id_to_classes: per-sample or shared class mapping.
            threshold: minimum probability.
            flat_ner: enforce non-overlapping spans.
            multi_label: allow multiple labels per span position.

        Returns:
            List of lists of Span objects per sample.
        """
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
                    class_id = class_idx.item()
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

    @staticmethod
    def resolve_span_text(
        texts: Optional[List[List[str]]],
        batch_idx: int,
        start: int,
        end: int,
    ) -> str:
        """Resolve span token indices to text string."""
        if texts is None or batch_idx >= len(texts):
            return ""
        tokens = texts[batch_idx]
        span_tokens = tokens[start:end + 1]
        return " ".join(span_tokens)
