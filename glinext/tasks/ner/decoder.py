"""NER task decoder — post-processing logits into entity spans.

Follows GLiNER TokenDecoder pattern: supports both token-level BIO decoding
and span-level decoding (when represent_spans is enabled in NERHead).
"""

from typing import List

from ..span_decoder import Span, SpanDecoder


class NERDecoder(SpanDecoder):
    """Decodes NER logits into entity spans with labels.

    Two decoding modes:
    1. Token-level BIO: uses ner_logits (BN, L, C+1, 3) with start/end/inside
    2. Span-level: uses span_logits (BN, S, C) + span_idx (BN, S, 2) + span_mask (BN, S)

    When batch_origin is available in model_output, results are grouped back
    to per-batch-item lists of lists. Otherwise returns flat per-group lists.
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

        Returns:
            If batch_origin available: List[List[List[Span]]] — per batch item, per group.
            Otherwise: List[List[Span]] — per group (flat BN).
        """
        if model_output.ner_logits is None and model_output.span_logits is None:
            return []

        threshold = threshold or self.threshold
        id_to_classes = classes_mapping if classes_mapping is not None else {}

        # Prefer span-level decoding when available
        if (model_output.span_logits is not None
                and model_output.span_idx is not None
                and model_output.span_mask is not None):
            flat_results = self.decode_span_level(
                span_logits=model_output.span_logits,
                span_idx=model_output.span_idx,
                span_mask=model_output.span_mask,
                id_to_classes=id_to_classes,
                threshold=threshold,
                flat_ner=flat_ner,
                multi_label=multi_label,
            )
        else:
            # Token-level BIO decoding
            logits = model_output.ner_logits
            bn_size = logits.shape[0]
            flat_results = self.decode_bio_spans_batch(
                logits=logits,
                id_to_classes=id_to_classes,
                batch_size=bn_size,
                threshold=threshold,
                flat_ner=flat_ner,
                multi_label=multi_label,
            )

        # Unflatten BN → B if batch_origin is available
        if model_output.ner_batch_origin is not None and model_output.batch_size is not None:
            from ...decoder import unflatten_by_batch_origin
            return unflatten_by_batch_origin(
                flat_results, model_output.ner_batch_origin, model_output.batch_size,
            )

        return flat_results
