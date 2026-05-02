"""NER task decoder — post-processing logits into entity spans.

Follows GLiNER TokenDecoder pattern: supports both token-level BIO decoding
and span-level decoding (when represent_spans is enabled in NERHead).
"""

from typing import Dict, List, Union

from ..span_decoder import Span, SpanDecoder
from ...processing.decoder import unflatten_by_batch_origin


class NERDecoder(SpanDecoder):
    """Decodes NER logits into entity spans with labels.

    Two decoding modes:
    1. Token-level BIO: uses ner_logits (BN, L, C+1, 3) with start/end/inside
    2. Span-level: uses span_logits (BN, S, C) + span_idx (BN, S, 2) + span_mask (BN, S)

    Model outputs are always BN-indexed; results are unflattened back to
    per-batch-item lists via batch_origin.
    """

    def _get_ner_id_to_classes(
        self, classes_mapping,
    ) -> Union[Dict[int, str], List[Dict[int, str]]]:
        """Extract NER id->class mappings from BatchClassesMapping.

        Returns a BN-level list of 0-indexed dicts matching the label layout
        (entity types at index 0+, no parent class).
        """
        if classes_mapping is None:
            return {}
        if not hasattr(classes_mapping, 'extraction_mapping'):
            return classes_mapping  # already a dict/list

        maps = []
        for em in classes_mapping.extraction_mapping:
            for item in em.items:
                maps.append(item.ner_class_to_id.get_reverse_mapping())
        return maps

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
            List[List[List[Span]]] — per batch item, per group, list of spans.
        """
        if model_output.ner_logits is None and model_output.span_logits is None:
            return []

        threshold = threshold or self.threshold
        id_to_classes = self._get_ner_id_to_classes(classes_mapping)

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

        # Unflatten BN → B
        return unflatten_by_batch_origin(
            flat_results, model_output.ner_batch_origin, model_output.batch_size,
        )

    def map_results(
        self,
        task_results: list,
        valid_to_orig_idx: List[int],
        all_start_maps: List[List[int]],
        all_end_maps: List[List[int]],
        valid_texts: List[str],
        num_original: int,
        **kwargs,
    ) -> List[List[Dict]]:
        output = [[] for _ in range(num_original)]

        for valid_i, per_text_groups in enumerate(task_results):
            orig_i = valid_to_orig_idx[valid_i]
            start_map = all_start_maps[valid_i]
            end_map = all_end_maps[valid_i]
            text = valid_texts[valid_i]

            entities = []
            groups = per_text_groups if isinstance(per_text_groups, list) else [per_text_groups]
            for group in groups:
                if not isinstance(group, list):
                    group = [group]
                for span in group:
                    mapped = self._map_span(span, start_map, end_map, text)
                    if mapped is not None:
                        entities.append(mapped)

            output[orig_i] = entities
        return output

    @staticmethod
    def _map_span(span, start_map, end_map, text):
        if hasattr(span, 'start') and hasattr(span, 'entity_type'):
            if span.start >= len(start_map) or span.end >= len(end_map):
                return None
            start_char = start_map[span.start]
            end_char = end_map[span.end]
            entity = {
                "start": start_char,
                "end": end_char,
                "text": text[start_char:end_char],
                "label": span.entity_type,
                "score": span.score,
            }
            if span.class_probs is not None:
                entity["class_probs"] = span.class_probs
            return entity
        if isinstance(span, dict):
            return span
        return None
