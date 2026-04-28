"""Structuring task decoder — post-processing logits into structured output.

Uses SpanDecoder for BIO span extraction and greedy overlap removal.
Supports both token-level BIO decoding and span-level decoding (represent_spans).
"""

from typing import Dict, List, Optional

import torch

from ..span_decoder import Span, SpanDecoder
from ...processing.decoder import unflatten_by_batch_origin


class StructuringDecoder(SpanDecoder):
    """Decodes structuring logits into field-value assignments per instance.

    Two decoding modes:
    1. Token-level BIO: structuring_logits (B, X, L, C, 3)
    2. Span-level: structuring_span_logits (B, X, S, C) + span_idx/span_mask
    """

    def __init__(self, config):
        super().__init__(config)
        struct_cfg = getattr(config, "structuring_config", None)
        self.objectness_threshold = (
            getattr(struct_cfg, "anchor_objectness_threshold", 0.5)
            if struct_cfg is not None else 0.5
        )

    def _resolve_anchor_mask(self, anchor_mask, objectness_logits,
                              objectness_threshold):
        """Combine the structural anchor mask with objectness gating.

        Returns a boolean mask of the same shape as ``anchor_mask`` (or the
        objectness mask, when anchor_mask is None). Anchors where objectness
        falls below the threshold are dropped from decoding.
        """
        threshold = objectness_threshold if objectness_threshold is not None \
            else self.objectness_threshold

        obj_mask = None
        if objectness_logits is not None:
            obj_mask = torch.sigmoid(objectness_logits) > threshold

        if anchor_mask is None and obj_mask is None:
            return None
        if anchor_mask is None:
            return obj_mask
        if obj_mask is None:
            return anchor_mask.bool() if anchor_mask.dtype != torch.bool else anchor_mask

        # Align anchor counts (objectness comes from post-refine anchors which
        # should match anchor_mask but be defensive about it).
        min_A = min(anchor_mask.shape[1], obj_mask.shape[1])
        combined = anchor_mask[:, :min_A].bool() & obj_mask[:, :min_A]
        return combined

    def decode(
        self,
        model_output,
        classes_mapping=None,
        threshold=None,
        flat_ner=True,
        multi_label=False,
        texts=None,
        objectness_threshold=None,
        **kwargs,
    ) -> List[List[dict]]:
        """Decode structuring predictions.

        Automatically selects span-level decoding when span_logits are available,
        otherwise falls back to token-level BIO decoding. ``required_fields``
        are not consulted here — they only affect post-processing, where
        instances missing a required field are dropped.

        When the model has an anchor-objectness head, anchors below
        ``objectness_threshold`` are filtered before BIO decoding so that
        empty slots do not leak into the output.
        """
        if model_output.structuring_logits is None and model_output.structuring_span_logits is None:
            return []

        threshold = threshold or self.threshold
        anchor_mask = self._resolve_anchor_mask(
            model_output.structuring_anchor_mask,
            getattr(model_output, "structuring_objectness_logits", None),
            objectness_threshold,
        )

        # Determine batch size from whichever output is available
        if model_output.structuring_logits is not None:
            B = model_output.structuring_logits.shape[0]
        else:
            B = model_output.structuring_span_logits.shape[0]

        id_to_fields = self._build_field_class_maps(classes_mapping, B)

        batch_origin = model_output.structuring_batch_origin
        batch_size = model_output.batch_size

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
                batch_origin=batch_origin,
                batch_size=batch_size,
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
            batch_origin=batch_origin,
            batch_size=batch_size,
        )

    def _decode_token_level(self, logits, anchor_mask, id_to_fields, threshold,
                            flat_ner, multi_label, texts, batch_origin=None, batch_size=None):
        """Decode from token-level BIO logits (BN, X, L, C, 3)."""
        BN, X, L, C, _ = logits.shape
        flat_results = []
        print(f"[DEBUG structuring_decoder] token-level logits.shape=(BN={BN}, X={X}, L={L}, C={C}, 3) "
              f"anchor_mask.sum_per_group={anchor_mask.sum(dim=-1).tolist() if anchor_mask is not None else 'None'} "
              f"threshold={threshold}")

        for b in range(BN):
            text_bi = batch_origin[b].item()
            instances = []
            anchors_with_spans = 0
            field_id_to_class = id_to_fields[b] if b < len(id_to_fields) and id_to_fields[b] else {
                i: str(i) for i in range(C)
            }
            for x in range(X):
                if anchor_mask is not None and not anchor_mask[b, x]:
                    continue

                instance_logits = logits[b, x]  # (L, C, 3)
                spans = self.decode_bio_spans(
                    instance_logits, field_id_to_class, threshold, flat_ner, multi_label,
                )

                if spans:
                    anchors_with_spans += 1
                    fields = self._spans_to_fields(spans, texts, text_bi)
                    instances.append(fields)
            print(f"[DEBUG structuring_decoder]   group={b} text_bi={text_bi} "
                  f"anchors_with_spans={anchors_with_spans}/{X} → instances={len(instances)}")
            flat_results.append(instances)

        return unflatten_by_batch_origin(flat_results, batch_origin, batch_size)

    def _decode_from_spans(self, span_logits, span_idx, span_mask, anchor_mask,
                           id_to_fields, threshold, flat_ner, multi_label, texts,
                           batch_origin=None, batch_size=None):
        """Decode from span-level predictions (BN, X, S, C)."""
        BN, X, S, C = span_logits.shape
        span_probs = torch.sigmoid(span_logits)
        flat_results = []
        print(f"[DEBUG structuring_decoder] span-level span_logits.shape=(BN={BN}, X={X}, S={S}, C={C}) "
              f"anchor_mask.sum_per_group={anchor_mask.sum(dim=-1).tolist() if anchor_mask is not None else 'None'} "
              f"threshold={threshold}")

        for b in range(BN):
            text_bi = batch_origin[b].item()
            instances = []
            anchors_with_spans = 0
            field_id_to_class = id_to_fields[b] if b < len(id_to_fields) and id_to_fields[b] else {
                i: str(i) for i in range(C)
            }
            valid_indices = torch.where(span_mask[b])[0]
            for x in range(X):
                if anchor_mask is not None and not anchor_mask[b, x]:
                    continue

                spans = []

                for span_pos in valid_indices:
                    span_start = span_idx[b, span_pos, 0].item()
                    span_end = span_idx[b, span_pos, 1].item()
                    probs = span_probs[b, x, span_pos]
                    class_indices = torch.where(probs > threshold)[0]

                    for class_idx in class_indices:
                        class_id = class_idx.item()
                        if class_id in field_id_to_class:
                            spans.append(Span(
                                start=span_start,
                                end=span_end,
                                entity_type=field_id_to_class[class_id],
                                score=probs[class_idx].item(),
                            ))

                spans = self.greedy_search(spans, flat_ner, multi_label)

                if spans:
                    anchors_with_spans += 1
                    fields = self._spans_to_fields(spans, texts, text_bi)
                    instances.append(fields)
            print(f"[DEBUG structuring_decoder]   group={b} text_bi={text_bi} "
                  f"anchors_with_spans={anchors_with_spans}/{X} → instances={len(instances)}")
            flat_results.append(instances)

        return unflatten_by_batch_origin(flat_results, batch_origin, batch_size)

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
        """Build per-batch-item id->field_name mappings from BatchClassesMapping.

        Indexing matches the flat structuring batch axis (BN), where each
        entry corresponds to a single (batch_item, schema) group.
        """
        if classes_mapping is None or not hasattr(classes_mapping, 'structuring_mapping'):
            return [{} for _ in range(batch_size)]

        maps: List[Dict[int, str]] = []
        for sm in classes_mapping.structuring_mapping:
            for item in sm.items:
                if hasattr(item, 'field_class_to_id') and item.field_class_to_id is not None:
                    maps.append(item.field_class_to_id.get_reverse_mapping())
                else:
                    maps.append({})

        # Pad/truncate to batch_size so callers can index by group safely.
        if len(maps) < batch_size:
            maps.extend({} for _ in range(batch_size - len(maps)))
        return maps[:batch_size]
