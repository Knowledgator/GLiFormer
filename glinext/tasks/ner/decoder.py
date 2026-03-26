"""NER task decoder — post-processing logits into entity spans.

Follows GLiNER TokenDecoder pattern: supports both token-level BIO decoding
and span-level decoding (when represent_spans is enabled in NERHead).
"""

from typing import List

from ..span_decoder import Span, SpanDecoder


class NERDecoder(SpanDecoder):
    """Decodes NER logits into entity spans with labels.

    Two decoding modes:
    1. Token-level BIO: uses ner_logits (B*N, L, C+1, 3) with start/end/inside
    2. Span-level: uses span_logits (B, S, C) + span_idx (B, S, 2) + span_mask (B, S)
    """

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
            return self.decode_span_level(
                span_logits=model_output.span_logits,
                span_idx=model_output.span_idx,
                span_mask=model_output.span_mask,
                id_to_classes=id_to_classes,
                threshold=threshold,
                flat_ner=flat_ner,
                multi_label=multi_label,
            )

        # Fall back to token-level BIO decoding
        logits = model_output.ner_logits
        batch_size = logits.shape[0]
        return self.decode_bio_spans_batch(
            logits=logits,
            id_to_classes=id_to_classes,
            batch_size=batch_size,
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
        )
