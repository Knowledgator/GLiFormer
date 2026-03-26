"""Joint relex decoder — post-processing logits into relation triples.

Inherits from NERDecoder to first decode entities, then resolve relation
triples with full head/tail entity span information.
"""

from typing import Dict, List, Optional, Union

import torch

from ..span_decoder import Span  # noqa: F401 — re-exported
from ..ner.decoder import NERDecoder


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

        # 1. Decode NER entities first
        ner_id_to_classes = self._get_ner_id_to_classes(classes_mapping)
        entities = super().decode(
            model_output,
            classes_mapping=ner_id_to_classes,
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
            **kwargs,
        )

        # 2. Build relation class mappings
        rel_id_to_classes = self._get_rel_id_to_classes(classes_mapping)

        # 3. Decode relation triples
        probs = torch.sigmoid(model_output.joint_rel_logits)
        pair_idx = model_output.joint_rel_idx
        pair_mask = model_output.joint_rel_mask
        B = probs.shape[0]

        all_triples = []
        for b in range(B):
            triples = []
            batch_entities = entities[b] if b < len(entities) else []

            for p in range(probs.shape[1]):
                if pair_mask is not None and not pair_mask[b, p]:
                    continue
                for c in range(probs.shape[2]):
                    score = probs[b, p, c].item()
                    if score <= threshold:
                        continue

                    head_id = pair_idx[b, p, 0].item()
                    tail_id = pair_idx[b, p, 1].item()

                    head_span = self._resolve_entity(
                        batch_entities, head_id, texts, b,
                    )
                    tail_span = self._resolve_entity(
                        batch_entities, tail_id, texts, b,
                    )

                    rel_name = rel_id_to_classes.get(c, str(c))

                    triples.append({
                        "head": head_span,
                        "tail": tail_span,
                        "relation": rel_name,
                        "score": score,
                    })
            all_triples.append(triples)

        return all_triples

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

    def _get_ner_id_to_classes(self, classes_mapping) -> Union[Dict[int, str], List[Dict[int, str]]]:
        """Extract NER id→class mappings from BatchClassesMapping."""
        if classes_mapping is None:
            return {}
        if not hasattr(classes_mapping, 'extraction_mapping'):
            return classes_mapping  # already a dict/list

        maps = []
        for em in classes_mapping.extraction_mapping:
            merged = {}
            for item in em.items:
                merged.update(item.ner_class_to_id.get_reverse_mapping())
            maps.append(merged)
        return maps

    def _get_rel_id_to_classes(self, classes_mapping) -> Dict[int, str]:
        """Extract relation id→class mappings from BatchClassesMapping."""
        if classes_mapping is None:
            return {}
        if not hasattr(classes_mapping, 'extraction_mapping'):
            return {}

        # Joint relex: relation classes are in extraction_mapping items
        merged = {}
        for em in classes_mapping.extraction_mapping:
            for item in em.items:
                if item.rel_class_to_id is not None:
                    merged.update(item.rel_class_to_id.get_reverse_mapping())
        return merged

