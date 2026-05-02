"""Joint relex decoder — post-processing logits into relation triples.

Inherits from NERDecoder to first decode entities, then resolve relation
triples with full head/tail entity span information.
"""

from typing import Dict, List, Optional, Union

import torch

from ..span_decoder import Span  # noqa: F401 — re-exported
from ..ner.decoder import NERDecoder
from ..open_relex.decoder import OpenRelexDecoder
from ...processing.decoder import unflatten_by_batch_origin


class JointRelexDecoder(NERDecoder):
    """Decodes joint NER + relation logits into relation triples with entity spans.

    First decodes NER entities (inherited), then resolves entity pair indices
    from relation logits into full span information.
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
        """Decode joint relex predictions into relation triples with entity spans.

        Args:
            model_output: GLiNExTOutput with ner_logits, joint_rel_logits, joint_rel_idx, joint_rel_mask.
            classes_mapping: BatchClassesMapping for label resolution.
            threshold: Detection threshold (default 0.5).
            flat_ner: If True, enforce non-overlapping NER spans.
            multi_label: If True, allow multiple labels per span position.
            texts: Optional list of token lists per batch item for resolving span text.

        Returns:
            List[List[dict]] — per batch item, list of relation triple dicts:
            {
                "head": {"start": int, "end": int, "text": str, "type": str, "entity_idx": int},
                "tail": {"start": int, "end": int, "text": str, "type": str, "entity_idx": int},
                "relation": str,
                "score": float,
            }
        """
        if model_output.joint_rel_logits is None or model_output.joint_rel_idx is None:
            return []

        threshold = threshold or self.threshold

        # 1. Decode NER entities first (returns B × groups × spans)
        ner_id_to_classes = self._get_ner_id_to_classes(classes_mapping)
        entities = super().decode(
            model_output,
            classes_mapping=ner_id_to_classes,
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
            **kwargs,
        )

        # Flatten back to BN-indexed list of span lists for per-group entity lookup
        flat_entities = [spans for batch_groups in entities for spans in batch_groups]

        # 2. Build relation class mappings
        rel_id_to_classes = self._get_rel_id_to_classes(classes_mapping)
        entity_index_maps = self._build_entity_index_maps(
            flat_entities, getattr(model_output, "joint_rel_entity_spans", None),
        )

        # 3. Decode relation triples
        probs = torch.sigmoid(model_output.joint_rel_logits)
        pair_idx = model_output.joint_rel_idx
        pair_mask = model_output.joint_rel_mask
        BN = probs.shape[0]

        flat_triples = []
        for bn in range(BN):
            triples = []
            batch_entities = flat_entities[bn] if bn < len(flat_entities) else []
            rel_map = (
                rel_id_to_classes[bn]
                if isinstance(rel_id_to_classes, list) and bn < len(rel_id_to_classes)
                else rel_id_to_classes
            )

            for p in range(probs.shape[1]):
                if pair_mask is not None and not pair_mask[bn, p]:
                    continue
                for c in range(probs.shape[2]):
                    if rel_map and c not in rel_map:
                        continue
                    score = probs[bn, p, c].item()
                    if score <= threshold:
                        continue

                    head_id = pair_idx[bn, p, 0].item()
                    tail_id = pair_idx[bn, p, 1].item()
                    entity_index_map = entity_index_maps[bn] if bn < len(entity_index_maps) else None
                    if entity_index_map is not None:
                        head_id = entity_index_map.get(head_id)
                        tail_id = entity_index_map.get(tail_id)
                        if head_id is None or tail_id is None:
                            continue

                    head_span = self._resolve_entity(
                        batch_entities, head_id, texts, bn,
                    )
                    tail_span = self._resolve_entity(
                        batch_entities, tail_id, texts, bn,
                    )

                    rel_name = rel_map.get(c, str(c))

                    triples.append({
                        "head": head_span,
                        "tail": tail_span,
                        "relation": rel_name,
                        "score": score,
                    })
            flat_triples.append(triples)

        # Unflatten BN → B
        return unflatten_by_batch_origin(
            flat_triples, model_output.joint_rel_batch_origin, model_output.batch_size,
        )

    def _resolve_entity(
        self,
        entities: List[Span],
        entity_id: int,
        texts: Optional[List[List[str]]],
        batch_idx: int,
    ) -> dict:
        """Resolve an entity index to a span dict with text."""
        if entity_id < len(entities):
            entity = entities[entity_id]
            text = self.resolve_span_text(texts, batch_idx, entity.start, entity.end)
            return {
                "start": entity.start,
                "end": entity.end,
                "text": text,
                "type": entity.entity_type,
                "entity_idx": entity_id,
            }
        # Fallback for out-of-range indices
        return {
            "start": -1,
            "end": -1,
            "text": "",
            "type": "",
            "entity_idx": entity_id,
        }

    def _build_entity_index_maps(self, flat_entities: List[List[Span]], entity_spans) -> List[Optional[Dict[int, int]]]:
        """Map model entity indices to decoded entity indices by span boundary."""
        if entity_spans is None:
            return [None] * len(flat_entities)

        maps: List[Optional[Dict[int, int]]] = []
        for bn, entities in enumerate(flat_entities):
            if bn >= entity_spans.shape[0]:
                maps.append(None)
                continue
            boundary_to_idx = {}
            for decoded_idx, span in enumerate(entities):
                boundary_to_idx.setdefault((span.start, span.end), decoded_idx)

            model_to_decoded = {}
            for entity_idx in range(entity_spans.shape[1]):
                start = int(entity_spans[bn, entity_idx, 0].item())
                end = int(entity_spans[bn, entity_idx, 1].item())
                decoded_idx = boundary_to_idx.get((start, end))
                if decoded_idx is not None:
                    model_to_decoded[entity_idx] = decoded_idx
            maps.append(model_to_decoded)
        return maps

    def map_results(self, task_results: list, **kwargs):
        return OpenRelexDecoder.map_results(self, task_results, **kwargs)

    def _get_ner_id_to_classes(self, classes_mapping) -> Union[Dict[int, str], List[Dict[int, str]]]:
        """Extract NER id→class mappings from BatchClassesMapping.

        Returns BN-level list of 0-indexed dicts (entity types at index 0+).
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

    def _get_rel_id_to_classes(self, classes_mapping) -> Union[Dict[int, str], List[Dict[int, str]]]:
        """Extract BN-aligned relation id→class mappings from BatchClassesMapping."""
        if classes_mapping is None:
            return {}
        if not hasattr(classes_mapping, 'extraction_mapping'):
            return {}

        maps = []
        for em in classes_mapping.extraction_mapping:
            for item in em.items:
                if item.rel_class_to_id is not None:
                    maps.append(item.rel_class_to_id.get_reverse_mapping())
                else:
                    maps.append({})
        return maps
