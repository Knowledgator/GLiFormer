"""Shared hierarchy, mapping, and formatting support for structuring."""

from dataclasses import dataclass

import torch

from ..tasks.span_decoder import SpanDecoder
from ._structuring_alignment import (
    MULTI_LEVEL_RESULT_KEY,
    ReconstructedStructuringGroup,
    StructuringAnchorEntry,
    StructuringDecoderComponent,
    align_structuring_anchors,
    is_multi_level_group_result,
    make_multi_level_group_result,
    resolve_structuring_decoder,
)
from .structuring_processor import MULTI_LEVEL_ROOT_KEY
from .structuring_types import is_structuring_descriptor


@dataclass(frozen=True)
class StructuringDecodeContext:
    """Validated batch/mask state shared by both structuring decoders."""

    threshold: float
    objectness_threshold: float
    raw_anchor_mask: torch.Tensor | None
    anchor_mask: torch.Tensor | None
    reliable_presence_mask: torch.Tensor | None
    relation_scores: torch.Tensor | None
    batch_origin: torch.Tensor
    batch_size: int
    id_to_fields: list[dict[int, str]]
    multi_level_contexts: list[dict]


class StructuringDecoder(SpanDecoder):
    """Shared support for the canonical entity-first task decoder.

    Concrete task decoding lives in :mod:`gliformer.tasks.structuring.decoder`.
    This class owns common threshold/mask validation plus hierarchy mapping and
    result formatting; it deliberately does not accept the retired 5-D
    anchor-BIO output contract.
    """

    config_attr = "structuring_config"
    mapping_attr = "structuring_mapping"
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
        legacy_multi_level = bool(
            getattr(struct_cfg, "multi_level", False)
            if struct_cfg is not None else False
        )
        mode = (
            struct_cfg.effective_structure_mode()
            if struct_cfg is not None
            and hasattr(struct_cfg, "effective_structure_mode")
            else None
        )
        configured_multi_level = (
            bool(mode.is_multi_level)
            if mode is not None else legacy_multi_level
        )
        decoder_spec = (
            mode.decoder_spec()
            if mode is not None and hasattr(mode, "decoder_spec")
            else getattr(mode, "decoder", None) if mode is not None else None
        )
        self.anchor_relations_threshold = float(
            getattr(struct_cfg, "anchor_relations_threshold", 0.5)
            if struct_cfg is not None else 0.5
        )
        self.component = resolve_structuring_decoder(
            decoder_spec,
            multi_level=configured_multi_level,
            relation_threshold=self.anchor_relations_threshold,
        )
        self.multi_level = bool(
            getattr(self.component, "multi_level", configured_multi_level)
        )

    def _finalize_anchor_group(
        self,
        entries,
        *,
        relation_scores,
        context,
        preserve_empty_records,
    ):
        mapping = context.get("mapping") if context else None
        output_mode = context.get("output_mode", "schemas") if context else "schemas"
        return self.component.finalize_group(
            entries,
            relation_scores=relation_scores,
            mapping=mapping,
            output_mode=output_mode,
            preserve_empty_records=preserve_empty_records,
        )

    def _resolve_anchor_mask(
        self,
        anchor_mask,
        objectness_logits,
        objectness_threshold,
        *,
        expected_shape=None,
    ):
        """Combine the structural anchor mask with objectness gating.

        Returns a boolean mask of the same shape as ``anchor_mask`` (or the
        objectness mask, when anchor_mask is None). Objectness must be strictly
        above the threshold; parent-child relations cannot override this gate.
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

    def _prepare_decode_context(
        self,
        model_output,
        classes_mapping,
        *,
        batch_groups: int,
        anchor_count: int,
        device: torch.device,
        threshold: float | None,
        objectness_threshold: float | None,
    ) -> StructuringDecodeContext:
        """Resolve thresholds and validate common anchor/batch tensors."""

        if objectness_threshold is None and threshold is not None:
            objectness_threshold = threshold
        if objectness_threshold is None:
            objectness_threshold = self.objectness_threshold
        threshold = self.threshold if threshold is None else threshold
        if objectness_threshold is None:
            objectness_threshold = threshold

        id_to_fields = self._build_field_class_maps(
            classes_mapping,
            batch_groups,
        )
        multi_level_contexts = self._build_multi_level_contexts(
            classes_mapping,
            batch_groups,
        )
        relation_scores = getattr(
            model_output,
            self.relation_scores_attr,
            None,
        )
        if relation_scores is not None and (
            relation_scores.dim() != 3
            or relation_scores.shape
            != (batch_groups, anchor_count, anchor_count)
        ):
            raise ValueError(
                f"{self.relation_scores_attr} must have shape (BN, A, A), "
                f"got {tuple(relation_scores.shape)}"
            )

        objectness_logits = getattr(
            model_output,
            self.objectness_logits_attr,
            None,
        )
        raw_anchor_mask = getattr(
            model_output,
            self.anchor_mask_attr,
            None,
        )
        anchor_mask = self._resolve_anchor_mask(
            raw_anchor_mask,
            objectness_logits,
            float(objectness_threshold),
            expected_shape=(batch_groups, anchor_count),
        )
        reliable_presence_mask = None
        if objectness_logits is not None:
            reliable_presence_mask = (
                torch.sigmoid(objectness_logits) > objectness_threshold
            )
            if raw_anchor_mask is not None:
                reliable_presence_mask &= raw_anchor_mask.bool()

        batch_origin = getattr(model_output, self.batch_origin_attr, None)
        if batch_origin is None:
            batch_origin = torch.arange(batch_groups, device=device)
        elif batch_origin.shape != (batch_groups,):
            raise ValueError(
                f"{self.batch_origin_attr} must have shape (BN,), got "
                f"{tuple(batch_origin.shape)}"
            )
        batch_size = getattr(model_output, "batch_size", None)
        if batch_size is None:
            batch_size = int(batch_origin.max().item()) + 1

        return StructuringDecodeContext(
            threshold=float(threshold),
            objectness_threshold=float(objectness_threshold),
            raw_anchor_mask=raw_anchor_mask,
            anchor_mask=anchor_mask,
            reliable_presence_mask=reliable_presence_mask,
            relation_scores=relation_scores,
            batch_origin=batch_origin,
            batch_size=int(batch_size),
            id_to_fields=id_to_fields,
            multi_level_contexts=multi_level_contexts,
        )

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

    @classmethod
    def _task_structuring_mappings(cls, classes_mapping):
        if classes_mapping is None:
            return None
        return getattr(classes_mapping, cls.mapping_attr, None)

    def _build_field_class_maps(self, classes_mapping, batch_size: int) -> list[dict[int, str]]:
        """Build per-batch-item id->field_name mappings from BatchClassesMapping.

        Indexing matches the flat structuring batch axis (BN), where each
        entry corresponds to a single (batch_item, schema) group.
        """
        structuring_mappings = self._task_structuring_mappings(classes_mapping)
        if structuring_mappings is None:
            return [{} for _ in range(batch_size)]

        maps: list[dict[int, str]] = []
        for sm in structuring_mappings or []:
            for item in sm.items:
                if hasattr(item, 'field_class_to_id') and item.field_class_to_id is not None:
                    maps.append(item.field_class_to_id.get_reverse_mapping())
                else:
                    maps.append({})

        # Pad/truncate to batch_size so callers can index by group safely.
        if len(maps) < batch_size:
            maps.extend({} for _ in range(batch_size - len(maps)))
        return maps[:batch_size]

    @classmethod
    def _build_multi_level_contexts(cls, classes_mapping, batch_size: int) -> list[dict]:
        structuring_mappings = cls._task_structuring_mappings(classes_mapping)
        if structuring_mappings is None:
            return [{} for _ in range(batch_size)]
        contexts = []
        for structuring_mapping in structuring_mappings or []:
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
        valid_to_orig_idx: list[int],
        all_start_maps: list[list[int]],
        all_end_maps: list[list[int]],
        valid_texts: list[str],
        num_original: int,
        all_classes_mappings: list | None = None,
        structures=None,
        structuring_dedup: bool = True,
        anchor_diagnostics_output: list | None = None,
        **kwargs,
    ) -> list[dict[str, list[dict]]]:
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

            result_dict: dict[str, list[dict]] = {}
            groups = per_text_groups if isinstance(per_text_groups, list) else [per_text_groups]
            multi_level_mode = None

            for group_idx, group in enumerate(groups):
                schema_name = (
                    schema_names[group_idx]
                    if group_idx < len(schema_names)
                    else f"schema_{group_idx}"
                )
                is_hierarchy = is_multi_level_group_result(group)
                if is_hierarchy:
                    group_mapping = (
                        group.get("mapping")
                        if isinstance(group, dict) else None
                    )
                    schema_name = (
                        getattr(group_mapping, "name", None) or schema_name
                    )
                result_dict.setdefault(schema_name, [])
                required_for_schema = schema_required_fields.get(
                    schema_name, []
                )

                schema_field_list = schema_fields.get(schema_name, [])
                reconstructed = self.component.reconstruct_group(
                    group,
                    start_map,
                    end_map,
                    text,
                    schema_fields=schema_field_list,
                    assemble_instance=self._assemble_instance_dict,
                    fill_missing_fields=self._fill_missing_fields,
                    diagnostics={} if diagnostics is not None else None,
                )
                schema_instances = reconstructed.instances

                if (
                    diagnostics is not None
                    and reconstructed.diagnostics is not None
                ):
                    group_diagnostics = {
                        "schema": schema_name,
                        "output_mode": (
                            reconstructed.output_mode or "schemas"
                        ),
                        **reconstructed.diagnostics,
                    }
                    diagnostics[orig_i]["groups"].append(group_diagnostics)

                if reconstructed.multi_level:
                    group_spec = self._multi_level_group_spec(
                        structures, schema_name,
                    )
                    schema_instances = [
                        filtered
                        for root in schema_instances
                        if (
                            filtered := self._filter_nested_required_fields(
                                root, group_spec,
                            )
                        ) is not None
                    ]

                if required_for_schema:
                    if reconstructed.multi_level:
                        schema_instances = [
                            instance
                            for instance in schema_instances
                            if all(
                                self._required_value(instance, field) is not None
                                for field in required_for_schema
                            )
                        ]
                    else:
                        schema_instances = [
                            instance
                            for instance in schema_instances
                            if all(
                                instance.get(field) is not None
                                for field in required_for_schema
                            )
                        ]

                result_dict[schema_name].extend(schema_instances)
                if reconstructed.output_mode is not None:
                    multi_level_mode = reconstructed.output_mode

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
        if structuring_dedup:
            output = [self._postprocess_structuring(item) for item in output]
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

        schema_fields: dict[str, list[str]] = {}
        schema_required: dict[str, list[str]] = {}
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
        if not is_structuring_descriptor(spec):
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
    def _fill_missing_fields(instance: dict[str, object], schema_field_list: list[str]) -> dict[str, object]:
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
        start_map: list[int],
        end_map: list[int],
        text: str,
        schema_field_list: list[str],
    ) -> dict[str, object] | None:
        instance_dict: dict[str, object] = {}
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

    @classmethod
    def _postprocess_structuring(cls, value):
        """Recursively remove dictionaries duplicated by a fuller peer."""

        if isinstance(value, dict):
            return {
                key: cls._postprocess_structuring(item)
                for key, item in value.items()
            }
        if not isinstance(value, list):
            return value

        items = [cls._postprocess_structuring(item) for item in value]
        dict_indices = [
            index for index, item in enumerate(items)
            if isinstance(item, dict)
        ]
        ranked = sorted(
            dict_indices,
            key=lambda index: (
                -cls._non_null_output_count(items[index]),
                index,
            ),
        )
        kept = []
        for index in ranked:
            candidate = items[index]
            if any(
                cls._contains_non_null_outputs(items[other], candidate)
                for other in kept
            ):
                continue
            kept.append(index)
        kept = set(kept)
        return [
            item for index, item in enumerate(items)
            if not isinstance(item, dict) or index in kept
        ]

    @classmethod
    def _non_null_output_count(cls, value):
        if value is None:
            return 0
        if isinstance(value, dict):
            return sum(cls._non_null_output_count(item) for item in value.values())
        return 1

    @classmethod
    def _contains_non_null_outputs(cls, reference: dict, candidate: dict):
        for key, item in candidate.items():
            if item is None:
                continue
            reference_item = reference.get(key)
            if isinstance(item, dict):
                if not isinstance(reference_item, dict) or not cls._contains_non_null_outputs(
                    reference_item, item,
                ):
                    return False
            elif reference_item != item:
                return False
        return True

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

    @classmethod
    def _get_structuring_schema_names(cls, all_classes_mappings: list | None, valid_idx: int) -> list[str]:
        if not all_classes_mappings or valid_idx >= len(all_classes_mappings):
            return []
        entry = all_classes_mappings[valid_idx]
        if isinstance(entry, tuple):
            cm, item_idx = entry
        else:
            cm, item_idx = entry, None
        structuring_mappings = cls._task_structuring_mappings(cm)
        if structuring_mappings is None:
            return []

        names = []
        mappings = structuring_mappings or []
        if item_idx is not None and item_idx < len(mappings):
            mappings = [mappings[item_idx]]
        for sm in mappings:
            for item in sm.items:
                names.append(getattr(item, "name", None) or f"schema_{len(names)}")
        return names


__all__ = [
    "MULTI_LEVEL_RESULT_KEY",
    "ReconstructedStructuringGroup",
    "StructuringAnchorEntry",
    "StructuringDecoder",
    "StructuringDecoderComponent",
    "align_structuring_anchors",
    "is_multi_level_group_result",
    "make_multi_level_group_result",
    "resolve_structuring_decoder",
]
