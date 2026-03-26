"""Structuring task decoder — post-processing logits into structured output.

Uses SpanDecoder for BIO span extraction and greedy overlap removal.
Supports both token-level BIO decoding and span-level decoding (represent_spans).
"""

from typing import Dict, List

import torch

from ..span_decoder import Span, SpanDecoder


class StructuringDecoder(SpanDecoder):
    """Decodes structuring logits into field-value assignments per instance.

    Two decoding modes:
    1. Token-level BIO: structuring_logits (B, X, L, C, 3)
    2. Span-level: structuring_span_logits (B, X, S, C) + span_idx/span_mask
    """

    def decode(
        self,
        model_output,
        classes_mapping=None,
        threshold=None,
        flat_ner=True,
        multi_label=False,
        texts=None,
        **kwargs,
    ) -> List[List[dict]]:
        """Decode structuring predictions.

        Automatically selects span-level decoding when span_logits are available,
        otherwise falls back to token-level BIO decoding.
        """
        if model_output.structuring_logits is None and model_output.structuring_span_logits is None:
            return []

        threshold = threshold or self.threshold
        anchor_mask = model_output.structuring_anchor_mask

        # Determine batch size from whichever output is available
        if model_output.structuring_logits is not None:
            B = model_output.structuring_logits.shape[0]
        else:
            B = model_output.structuring_span_logits.shape[0]

        id_to_fields = self._build_field_class_maps(classes_mapping, B)

        # Prefer span-level decoding when available
        if (model_output.structuring_span_logits is not None
                and model_output.structuring_span_idx is not None
                and model_output.structuring_span_mask is not None):
            return self._decode_from_spans(
                model_output.structuring_span_logits,
                model_output.structuring_span_idx,
                model_output.structuring_span_mask,
                anchor_mask,
                id_to_fields,
                threshold,
                flat_ner,
                multi_label,
                texts,
            )

        # Fall back to token-level BIO decoding
        return self._decode_token_level(
            model_output.structuring_logits,
            anchor_mask,
            id_to_fields,
            threshold,
            flat_ner,
            multi_label,
            texts,
        )

    def _decode_token_level(self, logits, anchor_mask, id_to_fields, threshold,
                            flat_ner, multi_label, texts):
        """Decode from token-level BIO logits (B, X, L, C, 3)."""
        B, X, L, C, _ = logits.shape
        all_instances = []

        for b in range(B):
            instances = []
            for x in range(X):
                if anchor_mask is not None and not anchor_mask[b, x]:
                    continue

                instance_logits = logits[b, x]  # (L, C, 3)
                field_id_to_class = id_to_fields[b] if id_to_fields[b] else {
                    i + 1: str(i) for i in range(C)
                }
                spans = self.decode_bio_spans(
                    instance_logits, field_id_to_class, threshold, flat_ner, multi_label,
                )

                if spans:
                    fields = self._spans_to_fields(spans, texts, b)
                    instances.append(fields)
            all_instances.append(instances)

        return all_instances

    def _decode_from_spans(self, span_logits, span_idx, span_mask, anchor_mask,
                           id_to_fields, threshold, flat_ner, multi_label, texts):
        """Decode from span-level predictions (B, X, S, C)."""
        B, X, S, C = span_logits.shape
        span_probs = torch.sigmoid(span_logits)
        all_instances = []

        for b in range(B):
            instances = []
            for x in range(X):
                if anchor_mask is not None and not anchor_mask[b, x]:
                    continue

                field_id_to_class = id_to_fields[b] if id_to_fields[b] else {
                    i + 1: str(i) for i in range(C)
                }
                spans = []
                valid_indices = torch.where(span_mask[b])[0]

                for span_pos in valid_indices:
                    span_start = span_idx[b, span_pos, 0].item()
                    span_end = span_idx[b, span_pos, 1].item()
                    probs = span_probs[b, x, span_pos]
                    class_indices = torch.where(probs > threshold)[0]

                    for class_idx in class_indices:
                        class_id = class_idx.item() + 1  # 1-indexed
                        if class_id in field_id_to_class:
                            spans.append(Span(
                                start=span_start,
                                end=span_end,
                                entity_type=field_id_to_class[class_id],
                                score=probs[class_idx].item(),
                            ))

                spans = self.greedy_search(spans, flat_ner, multi_label)
                if spans:
                    fields = self._spans_to_fields(spans, texts, b)
                    instances.append(fields)
            all_instances.append(instances)

        return all_instances

    def _spans_to_fields(self, spans, texts, batch_idx):
        """Convert Span objects to field dicts with text."""
        fields = []
        for span in spans:
            text = self.resolve_span_text(texts, batch_idx, span.start, span.end)
            fields.append({
                "field": span.entity_type,
                "start": span.start,
                "end": span.end,
                "text": text,
                "score": span.score,
            })
        return fields

    def _build_field_class_maps(self, classes_mapping, batch_size: int) -> List[Dict[int, str]]:
        """Build per-batch-item id->field_name mappings from BatchClassesMapping."""
        if classes_mapping is None:
            return [{} for _ in range(batch_size)]

        maps = [{} for _ in range(batch_size)]
        if not hasattr(classes_mapping, 'structuring_mapping'):
            return maps

        for batch_idx, sm in enumerate(classes_mapping.structuring_mapping):
            if batch_idx >= batch_size:
                break
            for item in sm.items:
                if hasattr(item, 'field_class_to_id') and item.field_class_to_id is not None:
                    reverse = item.field_class_to_id.get_reverse_mapping()
                    maps[batch_idx].update(reverse)

        return maps
