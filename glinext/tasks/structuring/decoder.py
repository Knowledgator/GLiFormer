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

    config_attr = "structuring_config"
    token_logits_attr = "structuring_logits"
    span_logits_attr = "structuring_span_logits"
    batch_origin_attr = "structuring_batch_origin"
    anchor_mask_attr = "structuring_anchor_mask"
    objectness_logits_attr = "structuring_objectness_logits"
    span_idx_attr = "structuring_span_idx"
    span_mask_attr = "structuring_span_mask"

    def __init__(self, config):
        super().__init__(config)
        struct_cfg = getattr(config, self.config_attr, None)
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
        token_logits = getattr(model_output, self.token_logits_attr, None)
        span_logits = getattr(model_output, self.span_logits_attr, None)
        span_idx = getattr(model_output, self.span_idx_attr, None)
        span_mask = getattr(model_output, self.span_mask_attr, None)
        if token_logits is None and span_logits is None:
            return []

        threshold = threshold or self.threshold
        anchor_mask = self._resolve_anchor_mask(
            getattr(model_output, self.anchor_mask_attr, None),
            getattr(model_output, self.objectness_logits_attr, None),
            objectness_threshold,
        )

        # Determine batch size from whichever output is available
        if token_logits is not None:
            B = token_logits.shape[0]
        else:
            B = span_logits.shape[0]

        id_to_fields = self._build_field_class_maps(classes_mapping, B)

        batch_origin = getattr(model_output, self.batch_origin_attr, None)
        batch_size = model_output.batch_size

        # Prefer span-level decoding when available
        if span_logits is not None and span_idx is not None and span_mask is not None:
            return self._decode_from_spans(
                span_logits,
                span_idx,
                span_mask,
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
            token_logits,
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

        for b in range(BN):
            text_bi = batch_origin[b].item()
            instances = []
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
                    fields = self._spans_to_fields(spans, texts, text_bi)
                    instances.append(fields)
            flat_results.append(instances)

        return unflatten_by_batch_origin(flat_results, batch_origin, batch_size)

    def _decode_from_spans(self, span_logits, span_idx, span_mask, anchor_mask,
                           id_to_fields, threshold, flat_ner, multi_label, texts,
                           batch_origin=None, batch_size=None):
        """Decode from span-level predictions (BN, X, S, C)."""
        BN, X, S, C = span_logits.shape
        span_probs = torch.sigmoid(span_logits)
        flat_results = []

        for b in range(BN):
            text_bi = batch_origin[b].item()
            instances = []
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
                    fields = self._spans_to_fields(spans, texts, text_bi)
                    instances.append(fields)
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

    def map_results(
        self,
        task_results: list,
        valid_to_orig_idx: List[int],
        all_start_maps: List[List[int]],
        all_end_maps: List[List[int]],
        valid_texts: List[str],
        num_original: int,
        all_classes_mappings: Optional[list] = None,
        structures=None,
        structuring_dedup: bool = True,
        **kwargs,
    ) -> List[Dict[str, List[Dict]]]:
        output: List[Dict[str, List[Dict]]] = [{} for _ in range(num_original)]
        schema_fields, schema_required_fields = self._extract_structuring_meta(structures)

        for valid_i, per_text_groups in enumerate(task_results):
            orig_i = valid_to_orig_idx[valid_i]
            start_map = all_start_maps[valid_i]
            end_map = all_end_maps[valid_i]
            text = valid_texts[valid_i]
            schema_names = self._get_structuring_schema_names(
                all_classes_mappings, valid_i,
            )

            result_dict: Dict[str, List[Dict]] = {}
            groups = per_text_groups if isinstance(per_text_groups, list) else [per_text_groups]

            for group_idx, group in enumerate(groups):
                schema_name = (
                    schema_names[group_idx]
                    if group_idx < len(schema_names)
                    else f"schema_{group_idx}"
                )
                result_dict.setdefault(schema_name, [])

                schema_field_list = schema_fields.get(schema_name, [])
                required_for_schema = schema_required_fields.get(schema_name, [])

                instances = group if isinstance(group, list) else [group]
                schema_instances: List[Dict[str, object]] = []
                for instance in instances:
                    if isinstance(instance, list):
                        instance_dict = self._assemble_instance_dict(
                            instance, start_map, end_map, text, schema_field_list,
                        )
                        if instance_dict is not None:
                            schema_instances.append(instance_dict)
                    elif isinstance(instance, dict):
                        schema_instances.append(self._fill_missing_fields(instance, schema_field_list))

                if required_for_schema:
                    schema_instances = [
                        inst for inst in schema_instances
                        if all(inst.get(f) is not None for f in required_for_schema)
                    ]

                if structuring_dedup and len(schema_instances) > 1:
                    schema_instances = self._dedup_similar_instances(
                        schema_instances, schema_field_list,
                    )

                result_dict[schema_name].extend(schema_instances)

            output[orig_i] = result_dict
        return output

    @staticmethod
    def _extract_structuring_meta(structures):
        if not structures:
            return {}, {}

        schema_fields: Dict[str, List[str]] = {}
        schema_required: Dict[str, List[str]] = {}
        for schema_name, spec in structures.items():
            if isinstance(spec, list):
                fields = list(spec)
                required = []
            elif isinstance(spec, dict):
                fields = list(spec.get("fields", []))
                required = list(spec.get("required_fields") or [])
            else:
                fields = []
                required = []
            schema_fields[schema_name] = fields
            schema_required[schema_name] = [f for f in required if f in fields]
        return schema_fields, schema_required

    @staticmethod
    def _fill_missing_fields(instance: Dict[str, object], schema_field_list: List[str]) -> Dict[str, object]:
        if not schema_field_list:
            return dict(instance)
        filled = {f: instance.get(f, None) for f in schema_field_list}
        for k, v in instance.items():
            if k not in filled:
                filled[k] = v
        return filled

    def _assemble_instance_dict(
        self,
        instance_fields: list,
        start_map: List[int],
        end_map: List[int],
        text: str,
        schema_field_list: List[str],
    ) -> Optional[Dict[str, object]]:
        instance_dict: Dict[str, object] = {}
        ordered_fields = sorted(
            instance_fields,
            key=lambda f: -(f.get("score", 0.0) if isinstance(f, dict) else 0.0),
        )
        for field in ordered_fields:
            mapped = self._map_field_to_value(field, start_map, end_map, text)
            if mapped is None:
                continue
            field_name = mapped["field"]
            value = mapped["value"]
            if field_name in instance_dict and instance_dict[field_name] is not None:
                existing = instance_dict[field_name]
                if not isinstance(existing, list):
                    instance_dict[field_name] = [existing, value]
                else:
                    existing.append(value)
            else:
                instance_dict[field_name] = value

        if schema_field_list:
            instance_dict = self._fill_missing_fields(instance_dict, schema_field_list)
            if all(v is None for v in instance_dict.values()):
                return None
            return instance_dict

        if not instance_dict:
            return None
        return instance_dict

    @staticmethod
    def _value_dedup_key(value):
        if value is None:
            return None
        raw = value if isinstance(value, list) else [value]
        items = []
        for v in raw:
            if v is None:
                continue
            if isinstance(v, dict):
                if "text" in v:
                    v = v["text"]
                else:
                    items.append((v.get("start"), v.get("end")))
                    continue
            if isinstance(v, str):
                v = v.strip().lower()
            items.append(v)
        if not items:
            return None
        return frozenset(items)

    @classmethod
    def _dominates(
        cls, richer: Dict[str, object], poorer: Dict[str, object],
        schema_field_list: List[str],
    ) -> bool:
        keys = schema_field_list or list({*richer.keys(), *poorer.keys()})
        for k in keys:
            kp = cls._value_dedup_key(poorer.get(k))
            if kp is None:
                continue
            kr = cls._value_dedup_key(richer.get(k))
            if kr is None:
                return False
            if not kp.issubset(kr):
                return False
        return True

    @classmethod
    def _dedup_similar_instances(
        cls, instances: List[Dict[str, object]], schema_field_list: List[str],
    ) -> List[Dict[str, object]]:
        def richness(inst):
            total = 0
            for v in inst.values():
                key = cls._value_dedup_key(v)
                if key is None:
                    continue
                total += len(key) if isinstance(key, frozenset) else 1
            return total

        ordered = sorted(
            enumerate(instances),
            key=lambda iv: (-richness(iv[1]), iv[0]),
        )
        kept: List[Dict[str, object]] = []
        for _, inst in ordered:
            if any(cls._dominates(k, inst, schema_field_list) for k in kept):
                continue
            kept.append(inst)
        return kept

    @staticmethod
    def _map_field_to_value(field, start_map, end_map, text):
        if not isinstance(field, dict) or "field" not in field:
            return None
        st = field.get("start", 0)
        ed = field.get("end", 0)
        if st < len(start_map) and ed < len(end_map):
            start_char = start_map[st]
            end_char = end_map[ed]
            return {
                "field": field["field"],
                "value": text[start_char:end_char],
            }
        return {
            "field": field["field"],
            "value": field.get("text", ""),
        }

    @staticmethod
    def _get_structuring_schema_names(all_classes_mappings: Optional[list], valid_idx: int) -> List[str]:
        if not all_classes_mappings or valid_idx >= len(all_classes_mappings):
            return []
        entry = all_classes_mappings[valid_idx]
        if isinstance(entry, tuple):
            cm, item_idx = entry
        else:
            cm, item_idx = entry, None
        if cm is None or not hasattr(cm, "structuring_mapping"):
            return []

        names = []
        mappings = cm.structuring_mapping
        if item_idx is not None and item_idx < len(mappings):
            mappings = [mappings[item_idx]]
        for sm in mappings:
            for item in sm.items:
                names.append(getattr(item, "name", None) or f"schema_{len(names)}")
        return names
