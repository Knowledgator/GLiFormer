"""Open relex decoder — post-processing logits into relation triples with spans.

Uses SpanDecoder for BIO span extraction logic.
Supports both token-level BIO decoding and span-level decoding (represent_spans).
"""

from typing import Dict, List

import torch

from ..span_decoder import Span, SpanDecoder
from ...processing.decoder import unflatten_by_batch_origin


class OpenRelexDecoder(SpanDecoder):
    """Decodes anchor-based relation logits into triples with head/tail spans.

    Two decoding modes:
    1. Token-level BIO: open_rel_logits (B, X, C, L, 2, 3)
    2. Span-level: open_rel_span_logits (B, S, X, C, 2) + span_idx/span_mask
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
        """Decode open relex predictions.

        Automatically selects span-level decoding when span_logits are available,
        otherwise falls back to token-level BIO decoding.
        """
        if model_output.open_rel_logits is None and model_output.open_rel_span_logits is None:
            return []

        threshold = threshold or self.threshold
        anchor_mask = model_output.open_rel_anchor_mask
        batch_origin = model_output.open_rel_batch_origin
        batch_size = model_output.batch_size

        # Prefer span-level decoding when available
        if (model_output.open_rel_span_logits is not None
                and model_output.open_rel_span_idx is not None
                and model_output.open_rel_span_mask is not None):
            return self._decode_from_spans(
                model_output.open_rel_span_logits,
                model_output.open_rel_span_idx,
                model_output.open_rel_span_mask,
                anchor_mask,
                classes_mapping,
                threshold,
                flat_ner,
                multi_label,
                texts,
                batch_origin=batch_origin,
                batch_size=batch_size,
            )

        # Fall back to token-level BIO decoding
        return self._decode_token_level(
            model_output.open_rel_logits,
            anchor_mask,
            classes_mapping,
            threshold,
            flat_ner,
            multi_label,
            texts,
            batch_origin=batch_origin,
            batch_size=batch_size,
        )

    def _decode_token_level(self, logits, anchor_mask, classes_mapping,
                            threshold, flat_ner, multi_label, texts,
                            batch_origin=None, batch_size=None):
        """Decode from token-level BIO logits (BN, X, C, L, 2, 3)."""
        probs = torch.sigmoid(logits)
        BN, X, C, _, _, _ = probs.shape
        id_to_rel_classes = self._build_rel_class_maps(classes_mapping, BN)

        all_results = []

        for b in range(BN):
            text_bi = batch_origin[b].item()
            triples = []

            for x in range(X):
                if anchor_mask is not None and not anchor_mask[b, x]:
                    continue
                for c in range(C):
                    if id_to_rel_classes[b] and c not in id_to_rel_classes[b]:
                        continue
                    head_probs = probs[b, x, c, :, 0, :]  # (L, 3)
                    tail_probs = probs[b, x, c, :, 1, :]  # (L, 3)

                    head_spans = self.decode_bio_spans_single_class(
                        head_probs, threshold, flat_ner, multi_label,
                    )
                    tail_spans = self.decode_bio_spans_single_class(
                        tail_probs, threshold, flat_ner, multi_label,
                    )

                    if not head_spans or not tail_spans:
                        continue

                    rel_name = id_to_rel_classes[b].get(c, str(c))
                    self._add_triples(triples, head_spans, tail_spans, rel_name, texts, text_bi)

            all_results.append(triples)

        return unflatten_by_batch_origin(all_results, batch_origin, batch_size)

    def _decode_from_spans(self, span_logits, span_idx, span_mask, anchor_mask,
                           classes_mapping, threshold, flat_ner, multi_label, texts,
                           batch_origin=None, batch_size=None):
        """Decode from span-level predictions.

        span_logits: (BN, S, X, C, 2) — per span, per anchor, per rel class, head/tail score
        """
        BN, S, X, C, _ = span_logits.shape
        span_probs = torch.sigmoid(span_logits)
        id_to_rel_classes = self._build_rel_class_maps(classes_mapping, BN)

        all_results = []

        for b in range(BN):
            text_bi = batch_origin[b].item()
            triples = []
            valid_indices = torch.where(span_mask[b])[0]

            for x in range(X):
                if anchor_mask is not None and not anchor_mask[b, x]:
                    continue
                for c in range(C):
                    if id_to_rel_classes[b] and c not in id_to_rel_classes[b]:
                        continue
                    rel_name = id_to_rel_classes[b].get(c, str(c))

                    # Collect head and tail spans above threshold
                    head_spans = []
                    tail_spans = []
                    for span_pos in valid_indices:
                        s = span_pos.item()
                        start = span_idx[b, s, 0].item()
                        end = span_idx[b, s, 1].item()

                        head_score = span_probs[b, s, x, c, 0].item()
                        tail_score = span_probs[b, s, x, c, 1].item()

                        if head_score > threshold:
                            head_spans.append(Span(start=start, end=end,
                                                   entity_type="", score=head_score))
                        if tail_score > threshold:
                            tail_spans.append(Span(start=start, end=end,
                                                   entity_type="", score=tail_score))

                    head_spans = self.greedy_search(head_spans, flat_ner, multi_label)
                    tail_spans = self.greedy_search(tail_spans, flat_ner, multi_label)

                    if not head_spans or not tail_spans:
                        continue

                    self._add_triples(triples, head_spans, tail_spans, rel_name, texts, text_bi)

            all_results.append(triples)

        return unflatten_by_batch_origin(all_results, batch_origin, batch_size)

    def _add_triples(self, triples, head_spans, tail_spans, rel_name, texts, batch_idx):
        """Build relation triple dicts from head/tail span lists."""
        for head_span in head_spans:
            for tail_span in tail_spans:
                head_text = self.resolve_span_text(
                    texts, batch_idx, head_span.start, head_span.end,
                )
                tail_text = self.resolve_span_text(
                    texts, batch_idx, tail_span.start, tail_span.end,
                )
                score = (head_span.score + tail_span.score) / 2.0
                triples.append({
                    "head": {
                        "start": head_span.start,
                        "end": head_span.end,
                        "text": head_text,
                    },
                    "tail": {
                        "start": tail_span.start,
                        "end": tail_span.end,
                        "text": tail_text,
                    },
                    "relation": rel_name,
                    "score": score,
                })

    def _build_rel_class_maps(self, classes_mapping, batch_size: int) -> List[Dict[int, str]]:
        """Build per-batch-item id->relation_name mappings from BatchClassesMapping."""
        if classes_mapping is None:
            return [{} for _ in range(batch_size)]

        maps = [{} for _ in range(batch_size)]
        if not hasattr(classes_mapping, 'open_relex_mapping'):
            return maps

        for batch_idx, om in enumerate(classes_mapping.open_relex_mapping):
            if batch_idx >= batch_size:
                break
            for item in om.items:
                reverse = item.rel_class_to_id.get_reverse_mapping()
                maps[batch_idx].update(reverse)

        return maps
