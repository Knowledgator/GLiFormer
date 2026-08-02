"""Structuring task decoder — post-processing logits into structured output.

Uses SpanDecoder for BIO span extraction and greedy overlap removal.
Supports both token-level BIO decoding and span-level decoding (represent_spans).
"""

from typing import Dict, List, Optional

import torch

from ...processing.decoder import unflatten_by_batch_origin
from ..span_decoder import Span, SpanDecoder
from .multilevel import MULTI_LEVEL_ROOT_KEY
from .multilevel_decoder import (
    MultiLevelStructuringDecoder,
    is_multi_level_group_result,
    make_multi_level_group_result,
)


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
    relation_scores_attr = "structuring_anchor_relation_scores"
    span_idx_attr = "structuring_span_idx"
    span_mask_attr = "structuring_span_mask"

    def __init__(self, config):
        super().__init__(config)
        struct_cfg = getattr(config, self.config_attr, None)
        self.objectness_threshold = (
            getattr(struct_cfg, "anchor_objectness_threshold", None)
            if struct_cfg is not None else None
        )
        self.multi_level = bool(
            getattr(struct_cfg, "multi_level", False)
            if struct_cfg is not None else False
        )
        self.anchor_relations_threshold = float(
            getattr(struct_cfg, "anchor_relations_threshold", 0.5)
            if struct_cfg is not None else 0.5
        )
        self.multi_level_decoder = MultiLevelStructuringDecoder(
            self.anchor_relations_threshold
        )

    def _resolve_anchor_mask(
        self,
        anchor_mask,
        objectness_logits,
        objectness_threshold,
        *,
        relation_scores=None,
        expected_shape=None,
    ):
        """Combine the structural anchor mask with objectness gating.

        Returns a boolean mask of the same shape as ``anchor_mask`` (or the
        objectness mask, when anchor_mask is None). Anchors where objectness
        falls below the threshold are dropped from decoding.
        """
        for value, name in (
            (anchor_mask, self.anchor_mask_attr),
            (objectness_logits, self.objectness_logits_attr),
        ):
            if (
                value is not None
                and expected_shape is not None
                and tuple(value.shape) != tuple(expected_shape)
            ):
                raise ValueError(
                    f"{name} must have shape (BN, A), got "
                    f"{tuple(value.shape)}"
                )

        obj_mask = None
        if objectness_logits is not None:
            # Objectness is the authoritative prediction that a slot exists.
            # Relation logits are conditional attributes of existing slots;
            # allowing them to revive a rejected slot makes an explicit high
            # objectness threshold ineffective and creates phantom records.
            obj_mask = torch.sigmoid(objectness_logits) > objectness_threshold

        if anchor_mask is None and obj_mask is None:
            return None
        if anchor_mask is None:
            return obj_mask
        if obj_mask is None:
            return anchor_mask.bool() if anchor_mask.dtype != torch.bool else anchor_mask

        return anchor_mask.bool() & obj_mask

    def _rescue_nested_relation_anchors(
        self,
        anchor_mask,
        raw_anchor_mask,
        relation_scores,
        multi_level_contexts,
    ):
        """Retain relation-linked containers only inside real hierarchies.

        A relation may bridge a fieldless parent/child container, but it must
        be connected to at least one objectness-selected slot. Relations alone
        never create a graph, and root-only schemas never use relation rescue.
        """

        if anchor_mask is None or relation_scores is None:
            return anchor_mask
        rescued = anchor_mask.bool().clone()
        for batch_idx, context in enumerate(multi_level_contexts or []):
            mapping = context.get("mapping") if context else None
            hierarchy = list(getattr(mapping, "hierarchy", None) or [])
            if not any(tuple(node.get("path") or ()) for node in hierarchy):
                continue
            edges = relation_scores[batch_idx] >= self.anchor_relations_threshold
            edges = edges.clone()
            edges.fill_diagonal_(False)
            valid = (
                raw_anchor_mask[batch_idx].bool()
                if raw_anchor_mask is not None
                else torch.ones_like(rescued[batch_idx])
            )
            active = rescued[batch_idx]
            if not active.any():
                continue
            while True:
                neighbours = (
                    edges[active].any(dim=0)
                    | edges[:, active].any(dim=1)
                )
                expanded = valid & (active | neighbours)
                if torch.equal(expanded, active):
                    break
                active = expanded
            rescued[batch_idx] = active
        return rescued

    def decode(
        self,
        model_output,
        classes_mapping=None,
        threshold=None,
        flat_ner=True,
        multi_label=False,
        texts=None,
        objectness_threshold=None,
        preserve_empty_records=False,
        **kwargs,
    ) -> List[List[dict]]:
        """Decode structuring predictions.

        Automatically selects span-level decoding when span_logits are available,
        otherwise falls back to token-level BIO decoding. ``required_fields``
        are not consulted here — they only affect post-processing, where
        instances missing a required field are dropped.

        When the model has an anchor-objectness head, anchors below
        ``objectness_threshold`` are filtered before BIO decoding so that
        empty slots do not leak into the output. When it is omitted, the main
        ``threshold`` is reused for objectness. Objectness-selected records
        without field evidence are dropped unless ``preserve_empty_records``
        is explicitly enabled.
        """
        token_logits = getattr(model_output, self.token_logits_attr, None)
        span_logits = getattr(model_output, self.span_logits_attr, None)
        span_idx = getattr(model_output, self.span_idx_attr, None)
        span_mask = getattr(model_output, self.span_mask_attr, None)
        if token_logits is None and span_logits is None:
            return []

        # A call-time objectness threshold is most specific. Otherwise reuse
        # the call's main threshold; the configured value is retained only as
        # a fallback for direct decoder calls that omit both.
        if objectness_threshold is None and threshold is not None:
            objectness_threshold = threshold
        if objectness_threshold is None:
            objectness_threshold = self.objectness_threshold
        threshold = self.threshold if threshold is None else threshold
        if objectness_threshold is None:
            objectness_threshold = threshold
        # Determine batch size from whichever output is available
        if token_logits is not None:
            B = token_logits.shape[0]
        else:
            B = span_logits.shape[0]

        id_to_fields = self._build_field_class_maps(classes_mapping, B)
        multi_level_contexts = self._build_multi_level_contexts(
            classes_mapping, B,
        )
        relation_scores = getattr(
            model_output, self.relation_scores_attr, None,
        )
        expected_anchor_count = (
            span_logits.shape[1]
            if span_logits is not None and span_idx is not None and span_mask is not None
            else token_logits.shape[1]
        )
        if relation_scores is not None and (
            relation_scores.dim() != 3
            or relation_scores.shape != (
                B,
                expected_anchor_count,
                expected_anchor_count,
            )
        ):
            raise ValueError(
                f"{self.relation_scores_attr} must have shape (BN, A, A), "
                f"got {tuple(relation_scores.shape)}"
            )
        objectness_logits = getattr(
            model_output, self.objectness_logits_attr, None,
        )
        raw_anchor_mask = getattr(
            model_output, self.anchor_mask_attr, None,
        )
        anchor_mask = self._resolve_anchor_mask(
            raw_anchor_mask,
            objectness_logits,
            objectness_threshold,
            relation_scores=relation_scores,
            expected_shape=(B, expected_anchor_count),
        )
        anchor_mask = self._rescue_nested_relation_anchors(
            anchor_mask,
            raw_anchor_mask,
            relation_scores,
            multi_level_contexts,
        )
        reliable_presence_mask = None
        if objectness_logits is not None:
            reliable_presence_mask = (
                torch.sigmoid(objectness_logits) > objectness_threshold
            )
            if raw_anchor_mask is not None:
                reliable_presence_mask &= raw_anchor_mask.bool()

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
                multi_level_contexts=multi_level_contexts,
                relation_scores=relation_scores,
                reliable_presence_mask=reliable_presence_mask,
                preserve_empty_records=preserve_empty_records,
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
            multi_level_contexts=multi_level_contexts,
            relation_scores=relation_scores,
            reliable_presence_mask=reliable_presence_mask,
            preserve_empty_records=preserve_empty_records,
        )

    def _decode_token_level(self, logits, anchor_mask, id_to_fields, threshold,
                            flat_ner, multi_label, texts, batch_origin=None,
                            batch_size=None, multi_level_contexts=None,
                            relation_scores=None,
                            reliable_presence_mask=None,
                            preserve_empty_records=False):
        """Decode from token-level BIO logits (BN, X, L, C, 3)."""
        BN, X, L, C, _ = logits.shape
        flat_results = []

        for b in range(BN):
            text_bi = batch_origin[b].item()
            instances = []
            nodes = []
            context = (
                multi_level_contexts[b]
                if multi_level_contexts and b < len(multi_level_contexts)
                else None
            )
            is_multi_level = bool(
                context and getattr(context["mapping"], "multi_level", False)
            )
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

                fields = self._spans_to_fields(spans, texts, text_bi)
                if is_multi_level:
                    nodes.append({
                        "anchor_index": x,
                        "fields": fields,
                        "presence_is_reliable": bool(
                            reliable_presence_mask is not None
                            and reliable_presence_mask[b, x]
                        ),
                    })
                elif spans:
                    instances.append(fields)
            if is_multi_level:
                flat_results.append(make_multi_level_group_result(
                    nodes,
                    relation_scores[b] if relation_scores is not None else None,
                    context["mapping"],
                    context["output_mode"],
                    preserve_empty_records=preserve_empty_records,
                ))
            else:
                flat_results.append(instances)

        return unflatten_by_batch_origin(flat_results, batch_origin, batch_size)

    def _decode_from_spans(self, span_logits, span_idx, span_mask, anchor_mask,
                           id_to_fields, threshold, flat_ner, multi_label, texts,
                           batch_origin=None, batch_size=None,
                           multi_level_contexts=None, relation_scores=None,
                           reliable_presence_mask=None,
                           preserve_empty_records=False):
        """Decode from span-level predictions (BN, X, S, C)."""
        BN, X, S, C = span_logits.shape
        span_probs = torch.sigmoid(span_logits)
        flat_results = []

        for b in range(BN):
            text_bi = batch_origin[b].item()
            instances = []
            nodes = []
            context = (
                multi_level_contexts[b]
                if multi_level_contexts and b < len(multi_level_contexts)
                else None
            )
            is_multi_level = bool(
                context and getattr(context["mapping"], "multi_level", False)
            )
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

                fields = self._spans_to_fields(spans, texts, text_bi)
                if is_multi_level:
                    nodes.append({
                        "anchor_index": x,
                        "fields": fields,
                        "presence_is_reliable": bool(
                            reliable_presence_mask is not None
                            and reliable_presence_mask[b, x]
                        ),
                    })
                elif spans:
                    instances.append(fields)
            if is_multi_level:
                flat_results.append(make_multi_level_group_result(
                    nodes,
                    relation_scores[b] if relation_scores is not None else None,
                    context["mapping"],
                    context["output_mode"],
                    preserve_empty_records=preserve_empty_records,
                ))
            else:
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

    @staticmethod
    def _build_multi_level_contexts(classes_mapping, batch_size: int) -> List[dict]:
        if classes_mapping is None or not hasattr(
            classes_mapping, "structuring_mapping"
        ):
            return [{} for _ in range(batch_size)]
        contexts = []
        for structuring_mapping in classes_mapping.structuring_mapping:
            for item in structuring_mapping.items:
                contexts.append({
                    "mapping": item,
                    "output_mode": getattr(
                        structuring_mapping, "output_mode", "schemas"
                    ),
                })
        contexts.extend({} for _ in range(max(0, batch_size - len(contexts))))
        return contexts[:batch_size]

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
        anchor_diagnostics_output: Optional[list] = None,
        **kwargs,
    ) -> List[Dict[str, List[Dict]]]:
        raw_root_spec = (
            structures.get(MULTI_LEVEL_ROOT_KEY)
            if isinstance(structures, dict)
            and set(structures) == {MULTI_LEVEL_ROOT_KEY}
            else None
        )
        is_root_list = self.multi_level and (
            isinstance(structures, list)
            or isinstance(raw_root_spec, list)
        )
        output = [([] if is_root_list else {}) for _ in range(num_original)]
        diagnostics = (
            [{"summary": {}, "groups": []} for _ in range(num_original)]
            if anchor_diagnostics_output is not None
            else None
        )
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
            multi_level_mode = None

            for group_idx, group in enumerate(groups):
                schema_name = (
                    schema_names[group_idx]
                    if group_idx < len(schema_names)
                    else f"schema_{group_idx}"
                )
                if is_multi_level_group_result(group):
                    group_mapping = group.get("mapping")
                    schema_name = (
                        getattr(group_mapping, "name", None) or schema_name
                    )
                result_dict.setdefault(schema_name, [])
                required_for_schema = schema_required_fields.get(
                    schema_name, []
                )

                if is_multi_level_group_result(group):
                    group_diagnostics = {} if diagnostics is not None else None
                    roots = self.multi_level_decoder.reconstruct_group(
                        group,
                        start_map,
                        end_map,
                        text,
                        diagnostics=group_diagnostics,
                    )
                    if group_diagnostics is not None:
                        group_diagnostics = {
                            "schema": schema_name,
                            "output_mode": group.get(
                                "output_mode", "schemas",
                            ),
                            **group_diagnostics,
                        }
                        diagnostics[orig_i]["groups"].append(
                            group_diagnostics
                        )
                    group_spec = self._multi_level_group_spec(
                        structures, schema_name,
                    )
                    roots = [
                        filtered
                        for root in roots
                        if (
                            filtered := self._filter_nested_required_fields(
                                root, group_spec,
                            )
                        ) is not None
                    ]
                    if required_for_schema:
                        roots = [
                            root
                            for root in roots
                            if all(
                                self._required_value(root, field) is not None
                                for field in required_for_schema
                            )
                        ]
                    result_dict[schema_name].extend(roots)
                    multi_level_mode = group.get("output_mode", "schemas")
                    continue

                schema_field_list = schema_fields.get(schema_name, [])
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

            if multi_level_mode == "object":
                root_values = next(iter(result_dict.values()), [])
                merged_root = {}
                for root_value in root_values:
                    self._merge_nested_result(merged_root, root_value)
                output[orig_i] = merged_root
            elif multi_level_mode == "list":
                output[orig_i] = next(iter(result_dict.values()), [])
            else:
                output[orig_i] = result_dict

        if diagnostics is not None:
            for per_text in diagnostics:
                groups = per_text["groups"]
                per_text["summary"] = {
                    "schema_group_count": len(groups),
                    "activated_anchor_count": sum(
                        int(group.get("active_anchor_count", 0))
                        for group in groups
                    ),
                    "logical_anchor_count": sum(
                        len(group.get("logical_nodes", []))
                        for group in groups
                    ),
                    "selected_connection_count": sum(
                        len(group.get("connections", []))
                        for group in groups
                    ),
                    "raw_relation_connection_count": sum(
                        len(group.get("raw_relation_connections", []))
                        for group in groups
                    ),
                }
            anchor_diagnostics_output.extend(diagnostics)
        return output

    @classmethod
    def _merge_nested_result(cls, target: dict, source: dict) -> None:
        """Merge disconnected predictions for a single raw-object root."""

        for key, value in source.items():
            if key not in target or target[key] is None:
                target[key] = value
                continue
            existing = target[key]
            if isinstance(existing, dict) and isinstance(value, dict):
                cls._merge_nested_result(existing, value)
            elif isinstance(existing, list) and isinstance(value, list):
                existing.extend(value)
            elif existing != value:
                target[key] = (
                    existing + [value]
                    if isinstance(existing, list)
                    else [existing, value]
                )

    @staticmethod
    def _extract_structuring_meta(structures):
        if not structures or not isinstance(structures, dict):
            return {}, {}

        if set(structures) == {MULTI_LEVEL_ROOT_KEY}:
            root_spec = structures[MULTI_LEVEL_ROOT_KEY]
            structures = {"root": root_spec}

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
    def _required_value(instance: dict, field: str):
        """Resolve a required field, preferring an exact literal-dot key."""

        if field in instance:
            return instance[field]
        value = instance
        for segment in str(field).split("."):
            if not isinstance(value, dict) or segment not in value:
                return None
            value = value[segment]
        return value

    @staticmethod
    def _multi_level_group_spec(structures, schema_name: str):
        if not isinstance(structures, dict):
            return None
        if set(structures) == {MULTI_LEVEL_ROOT_KEY}:
            return structures[MULTI_LEVEL_ROOT_KEY]
        return structures.get(schema_name)

    @classmethod
    def _filter_nested_required_fields(cls, instance: dict, spec):
        """Apply descriptor requirements recursively to reconstructed JSON."""

        if not isinstance(spec, dict):
            return instance
        descriptor_keys = {
            "fields", "children", "required_fields", "description",
        }
        is_descriptor = set(spec).issubset(descriptor_keys) and (
            isinstance(spec.get("fields"), (list, dict))
            or "required_fields" in spec
            or (
                "fields" in spec
                and isinstance(spec.get("children"), dict)
            )
        )
        if not is_descriptor:
            return instance

        children = spec.get("children") or {}
        if isinstance(children, dict):
            for child_name, child_spec in children.items():
                value = instance.get(str(child_name))
                if isinstance(value, list):
                    instance[str(child_name)] = [
                        filtered
                        for child in value
                        if isinstance(child, dict)
                        and (
                            filtered := cls._filter_nested_required_fields(
                                child, child_spec,
                            )
                        ) is not None
                    ]
                elif isinstance(value, dict):
                    instance[str(child_name)] = (
                        cls._filter_nested_required_fields(value, child_spec)
                    )

        required = spec.get("required_fields") or []
        if any(cls._required_value(instance, field) is None for field in required):
            return None
        return instance

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
