"""Structuring task processor."""

import random
from copy import deepcopy

import torch

from ...processing.mappings import (
    BaseClassMapping,
    StructuringClassMapping,
    StructuringItemMapping,
)
from ..span_processor import SpanProcessor
from .multilevel import (
    MULTI_LEVEL_META_KEY,
    MULTI_LEVEL_ROOT_KEY,
    NODE_ID_KEY,
    NODE_KEEP_KEY,
    MultiLevelStructuringProcessor,
    is_internal_instance_key,
)


class StructuringProcessor(SpanProcessor):
    """Processor for structuring task."""

    def __init__(self, config, tokenizer=None, words_splitter=None, **kwargs):
        super().__init__(config, tokenizer, words_splitter,
                         parent_token=getattr(config, 'struct_parent_token', None), **kwargs)
        self.child_token = config.child_token
        # Fixed-width anchors always emit ``num_slots`` anchors; labels must
        # be padded to that size so Hungarian sees unmatched slots and trains
        # them as negatives. This includes learned slots and parameter-free
        # token selectors.
        struct_configs = [
            struct_cfg
            for struct_cfg in (
                getattr(config, 'structuring_config', None),
                getattr(config, 'set_structuring_config', None),
            )
            if struct_cfg is not None
        ]
        self.has_structuring_head = (
            getattr(config, 'structuring_config', None) is not None
        )
        self.has_set_structuring_head = (
            getattr(config, 'set_structuring_config', None) is not None
        )
        multi_level_flags = {
            bool(getattr(struct_cfg, 'multi_level', False))
            for struct_cfg in struct_configs
        }
        if len(multi_level_flags) > 1:
            raise ValueError(
                "structuring_config and set_structuring_config must use the "
                "same multi_level value because they share one processor"
            )
        self.multi_level = bool(multi_level_flags and next(iter(multi_level_flags)))
        self.multi_level_processor = (
            MultiLevelStructuringProcessor(
                child_token=config.structuring_child_token,
                end_token=config.structuring_end_token,
            )
            if self.multi_level
            else None
        )
        set_structuring_config = getattr(
            config,
            'set_structuring_config',
            None,
        )
        # Entity spans are mandatory targets for the independent second
        # stage, not an optional auxiliary representation.  Regular
        # structuring still honors its ``represent_spans`` switch.
        self._span_config = set_structuring_config or next(
            (
                struct_cfg
                for struct_cfg in struct_configs
                if getattr(struct_cfg, 'represent_spans', False)
            ),
            None,
        )
        # Only the classical structuring head consumes dense labels whose
        # record axis must match its fixed query width. Set structuring uses a
        # rectangular assignment (predicted slots versus gold records), so
        # padding its gold axis to ``num_slots`` only wastes memory.
        fixed_slot_configs = [
            struct_cfg
            for struct_cfg in (getattr(config, 'structuring_config', None),)
            if struct_cfg is not None
        ]
        self._fixed_slot_pad = max(
            (
                struct_cfg.effective_anchor_num_slots()
                if struct_cfg.effective_anchor_mode()
                in (
                    'fixed',
                    'fixed_rnn',
                    'fixed_transformer',
                    'position_buckets',
                    'topk_norm',
                    'topk_distinct',
                    'topk_parent',
                    'topk_density_distinct',
                )
                else 0
            )
            for struct_cfg in fixed_slot_configs
        ) if fixed_slot_configs else 0

    @staticmethod
    def _instance_sort_key(instance):
        """Return the earliest span start position across all fields in an instance."""
        min_start = float('inf')
        for key, value in instance.items():
            if is_internal_instance_key(key):
                continue
            if isinstance(value, dict):
                st = value.get('start', -1)
                if st >= 0:
                    min_start = min(min_start, st)
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, dict):
                        st = v.get('start', -1)
                        if st >= 0:
                            min_start = min(min_start, st)
        return min_start

    @staticmethod
    def _instance_has_valid_span(
        instance,
        max_seq_len=0,
        *,
        enforce_limit=False,
    ):
        if instance.get(NODE_KEEP_KEY, False) and not enforce_limit:
            return True
        for key, value in instance.items():
            if is_internal_instance_key(key):
                continue
            values = value if isinstance(value, list) else [value]
            for field_value in values:
                if not isinstance(field_value, dict):
                    continue
                st = field_value.get('start', -1)
                ed = field_value.get('end', -1)
                if st < 0 or ed < st:
                    continue
                if (
                    (enforce_limit or max_seq_len > 0)
                    and (st >= max_seq_len or ed >= max_seq_len)
                ):
                    continue
                return True
        return False

    @staticmethod
    def _normalize_sequence_lengths(batch_list, sequence_lengths, max_seq_len):
        """Validate optional per-item source lengths retained by tokenization."""

        if sequence_lengths is None:
            return None, int(max_seq_len)
        if torch.is_tensor(sequence_lengths):
            values = sequence_lengths.detach().reshape(-1).cpu().tolist()
        else:
            values = list(sequence_lengths)
        if len(values) != len(batch_list):
            raise ValueError(
                "sequence_lengths must contain one value per batch item"
            )
        normalized = []
        for value in values:
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(
                    "sequence_lengths values must be non-negative integers"
                )
            normalized.append(int(value))
        return normalized, max(normalized, default=0)

    def _filter_instances(
        self,
        item,
        data_key,
        instances,
        max_seq_len,
        *,
        enforce_limit=False,
    ):
        """Keep visible records and, for hierarchies, their ancestors."""

        visible = [
            instance
            for instance in instances
            if self._instance_has_valid_span(
                instance,
                max_seq_len,
                enforce_limit=enforce_limit,
            )
        ]
        if not enforce_limit or not self.multi_level:
            return visible

        multi_meta = item.get(MULTI_LEVEL_META_KEY) or {}
        group_meta = next(
            (
                group
                for group in multi_meta.get('groups', [])
                if group.get('data_key') == data_key
            ),
            {},
        )
        retained_node_ids = {
            instance.get(NODE_ID_KEY)
            for instance in visible
            if instance.get(NODE_ID_KEY) is not None
        }
        changed = True
        relations = group_meta.get('relations', [])
        while changed:
            changed = False
            for parent_node, child_node in relations:
                if (
                    child_node in retained_node_ids
                    and parent_node not in retained_node_ids
                ):
                    retained_node_ids.add(parent_node)
                    changed = True

        visible_object_ids = {id(instance) for instance in visible}
        return [
            instance
            for instance in instances
            if id(instance) in visible_object_ids
            or instance.get(NODE_ID_KEY) in retained_node_ids
        ]

    @staticmethod
    def _iter_field_values(value):
        """Flatten nested value arrays into groundable scalar occurrences."""

        if isinstance(value, list):
            for child in value:
                yield from StructuringProcessor._iter_field_values(child)
        else:
            yield value

    def get_classes_mapping(self, batch_list, shuffle_labels=False, **kwargs):
        structuring_mapping = []
        for item in batch_list:
            if self.multi_level_processor is not None:
                self.multi_level_processor.normalize_item(item)
            structuring_data = item.get('structuring', {})
            structuring_schema = item.get('structuring_schema') or {}
            multi_meta = item.get(MULTI_LEVEL_META_KEY) or {}
            meta_groups = {
                group.get('data_key'): group
                for group in multi_meta.get('groups', [])
            }
            prompt = item.get('prompt')
            item_mappings = []
            for data_key, instances in structuring_data.items():
                group_meta = meta_groups.get(data_key, {})
                schema_name = group_meta.get('name', data_key)
                field_names = []
                seen = set()
                # Inference stubs and explicitly supplied schemas carry the
                # complete field list separately from their (empty) values.
                # Prefer that list so invalid training annotations can be
                # pruned without making inference depend on placeholder
                # strings surviving span resolution.
                schema_field_names = self._structure_fields(
                    structuring_schema.get(data_key, [])
                )
                for field_name in schema_field_names:
                    if field_name not in seen:
                        field_names.append(field_name)
                        seen.add(field_name)
                for instance in instances:
                    for field_name in instance:
                        if is_internal_instance_key(field_name):
                            continue
                        if field_name not in seen:
                            field_names.append(field_name)
                            seen.add(field_name)
                if shuffle_labels:
                    random.shuffle(field_names)
                field_class_to_id = {name: idx for idx, name in enumerate(field_names)}
                if isinstance(prompt, dict):
                    description = prompt.get(schema_name)
                elif isinstance(prompt, str):
                    description = prompt
                else:
                    description = None
                if not isinstance(description, str) or not description.strip():
                    description = None
                else:
                    description = description.strip()
                item_mappings.append(StructuringItemMapping(
                    field_class_to_id=BaseClassMapping(
                        class_to_id=field_class_to_id,
                        name=schema_name,
                        description=description,
                    ),
                    name=schema_name,
                    description=description,
                    data_key=data_key,
                    hierarchy=deepcopy(group_meta.get('hierarchy') or []),
                    multi_level=bool(group_meta),
                ))
            structuring_mapping.append(StructuringClassMapping(
                items=item_mappings,
                output_mode=multi_meta.get('output_mode', 'schemas'),
                multi_level=bool(multi_meta),
            ))
        return structuring_mapping

    def contribute_prompt(self, classes_mapping, batch_idx, use_labels_encoder=False):
        if not hasattr(classes_mapping, 'structuring_mapping'):
            return []
        if batch_idx >= len(classes_mapping.structuring_mapping):
            return []
        prompt = []
        for struct_item in classes_mapping.structuring_mapping[batch_idx].items:
            if struct_item.multi_level and self.multi_level_processor is not None:
                prompt.extend(self.multi_level_processor.contribute_prompt(
                    struct_item,
                    parent_token=self.parent_token,
                    field_token=self.child_token,
                    sep_token=self.sep_token,
                    use_labels_encoder=use_labels_encoder,
                ))
                continue
            prompt.append(self.parent_token)
            field_map = struct_item.field_class_to_id
            if field_map.name:
                prompt.append(field_map.name)
            if field_map.description:
                prompt.append(field_map.description)
            if not use_labels_encoder:
                for field_name in field_map.class_to_id:
                    prompt.append(f"{self.child_token} {field_name}")
            prompt.append(self.sep_token)
        return prompt

    @staticmethod
    def _structure_fields(fields):
        if isinstance(fields, list):
            return fields
        if isinstance(fields, dict):
            return fields.get("fields", [])
        return []

    def contribute_inference_input(self, item, structures=None, **kwargs):
        if structures is None:
            return

        if self.multi_level_processor is not None:
            self.multi_level_processor.contribute_inference_input(item, structures)
            return

        if isinstance(structures, list):
            raise ValueError(
                "Root-list or nested structuring schemas require "
                "multi_level=True on the structuring head"
            )
        if not isinstance(structures, dict):
            raise TypeError("structures must be a dictionary or list")

        structuring = {}
        structuring_schema = {}
        for schema_name, fields in structures.items():
            if isinstance(fields, dict) and "fields" not in fields:
                raise ValueError(
                    "Nested structuring schemas require multi_level=True on "
                    "the structuring head"
                )
            field_list = self._structure_fields(fields)
            if not isinstance(field_list, list) or not all(
                isinstance(field, str) for field in field_list
            ):
                raise TypeError(
                    "Flat structuring schema fields must be a list of strings"
                )
            structuring[schema_name] = [dict.fromkeys(field_list, "")] if field_list else []
            structuring_schema[schema_name] = field_list
        item["structuring"] = structuring
        item["structuring_schema"] = structuring_schema

    def empty_inference_result(self, num_texts: int, structures=None, **kwargs):
        if structures is None:
            return None
        root_spec = (
            structures.get(MULTI_LEVEL_ROOT_KEY)
            if isinstance(structures, dict)
            and set(structures) == {MULTI_LEVEL_ROOT_KEY}
            else None
        )
        is_root_list = self.multi_level and (
            isinstance(structures, list) or isinstance(root_spec, list)
        )

        def empty_values():
            return [([] if is_root_list else {}) for _ in range(num_texts)]

        result = {}
        if self.has_structuring_head:
            result["structuring"] = empty_values()
        if self.has_set_structuring_head:
            result["set_structuring"] = empty_values()
        return result

    def _normalize_field_value(
        self,
        text,
        tokens_with_spans,
        field_name,
        value,
        first_only=True,
        used_spans=None,
    ):
        value_text = value.get('text') if isinstance(value, dict) else value
        if value_text is None and isinstance(value, dict) and 'start' in value and 'end' in value:
            try:
                value_text = text[int(value['start']):int(value['end'])]
            except (TypeError, ValueError):
                value_text = ''

        spans = self._resolve_labeled_span(
            text,
            tokens_with_spans,
            value,
            label=field_name,
            # Resolve all candidates here so repeated scalar values can be
            # assigned to successive records instead of copied onto every
            # record containing that value.
            first_only=False,
        )
        if used_spans is not None:
            spans = [
                span for span in spans
                if (span[0], span[1]) not in used_spans
            ]
        if first_only:
            spans = spans[:1]
        if not spans:
            return []

        if used_spans is not None:
            used_spans.update((start, end) for start, end, _ in spans)

        return [
            {
                'text': str(value_text),
                'start': start,
                'end': end,
            }
            for start, end, _ in spans
        ]

    def resolve_spans(self, item):
        if self.multi_level_processor is not None:
            self.multi_level_processor.normalize_item(item)
        if item.get('_glinext_structuring_spans_resolved'):
            return
        structuring = item.get('structuring', {})
        if not structuring:
            return

        text = item.get('text', '')
        tokens_with_spans, _ = self._tokenize_text(item)
        if tokens_with_spans is None:
            return

        for schema_name, instances in list(structuring.items()):
            used_spans_by_field = {}
            resolved_instances = []
            for instance in instances:
                if not isinstance(instance, dict):
                    continue
                for field_name, value in list(instance.items()):
                    if is_internal_instance_key(field_name):
                        continue
                    used_spans = used_spans_by_field.setdefault(field_name, set())
                    if isinstance(value, list):
                        resolved_list = []
                        for v in self._iter_field_values(value):
                            resolved_list.extend(self._normalize_field_value(
                                text,
                                tokens_with_spans,
                                field_name,
                                v,
                                used_spans=used_spans,
                            ))
                        if resolved_list:
                            instance[field_name] = resolved_list
                        else:
                            # An annotated value that cannot be grounded must
                            # not become an all-zero (false-negative) target.
                            instance.pop(field_name, None)
                    else:
                        normalized = self._normalize_field_value(
                            text,
                            tokens_with_spans,
                            field_name,
                            value,
                            used_spans=used_spans,
                        )
                        if normalized:
                            instance[field_name] = normalized[0]
                        else:
                            instance.pop(field_name, None)
                if any(
                    not is_internal_instance_key(key)
                    for key in instance
                ) or instance.get(NODE_KEEP_KEY, False):
                    resolved_instances.append(instance)
            structuring[schema_name] = resolved_instances
        item['_glinext_structuring_spans_resolved'] = True

    def create_labels(self, batch_list, classes_mapping, max_seq_len=0, **kwargs):
        sequence_lengths, max_seq_len = self._normalize_sequence_lengths(
            batch_list,
            kwargs.get("sequence_lengths"),
            max_seq_len,
        )
        source_sequence_lengths, _ = self._normalize_sequence_lengths(
            batch_list,
            kwargs.get("source_sequence_lengths"),
            max_seq_len,
        )
        for item in batch_list:
            self.resolve_spans(item)

        total_groups = classes_mapping.total_structuring_groups()
        if total_groups == 0:
            return None

        max_instances = 0
        max_fields = 0
        has_schema = False

        for flat_idx, batch_idx, group_idx, struct_item in classes_mapping.flat_structuring_iter():
            structuring_data = batch_list[batch_idx].get('structuring', {})
            data_key = struct_item.data_key or struct_item.name
            if data_key not in structuring_data:
                continue
            instances = structuring_data[data_key]
            if instances:
                has_schema = True
                item_limit = (
                    sequence_lengths[batch_idx]
                    if sequence_lengths is not None
                    else max_seq_len
                )
                enforce_limit = sequence_lengths is not None and (
                    source_sequence_lengths is None
                    or item_limit < source_sequence_lengths[batch_idx]
                )
                valid_instances = self._filter_instances(
                    batch_list[batch_idx],
                    data_key,
                    instances,
                    item_limit,
                    enforce_limit=enforce_limit,
                )
                max_instances = max(max_instances, len(valid_instances))
                max_fields = max(max_fields, len(struct_item.field_class_to_id.class_to_id))

        if not has_schema or max_fields == 0:
            return None
        if max_instances == 0:
            max_instances = 1

        if self._fixed_slot_pad:
            max_instances = max(max_instances, self._fixed_slot_pad)

        structuring_labels = torch.zeros(
            total_groups, max_instances, max_seq_len, max_fields, 3,
            dtype=torch.float,
        )
        structuring_mask = torch.zeros(total_groups, dtype=torch.bool)
        structuring_batch_idx = torch.zeros(total_groups, dtype=torch.long)
        structuring_count = torch.zeros(total_groups, dtype=torch.long)
        structuring_relation_labels = (
            torch.zeros(
                total_groups, max_instances, max_instances, dtype=torch.float,
            )
            if self.multi_level
            else None
        )
        structuring_relation_group_mask = (
            torch.zeros(total_groups, dtype=torch.bool)
            if self.multi_level
            else None
        )

        for flat_idx, batch_idx, group_idx, struct_item in classes_mapping.flat_structuring_iter():
            structuring_batch_idx[flat_idx] = batch_idx
            structuring_data = batch_list[batch_idx].get('structuring', {})
            data_key = struct_item.data_key or struct_item.name
            if data_key not in structuring_data:
                continue

            item_limit = (
                sequence_lengths[batch_idx]
                if sequence_lengths is not None
                else max_seq_len
            )
            enforce_limit = sequence_lengths is not None and (
                source_sequence_lengths is None
                or item_limit < source_sequence_lengths[batch_idx]
            )
            instances = self._filter_instances(
                batch_list[batch_idx],
                data_key,
                structuring_data[data_key],
                item_limit,
                enforce_limit=enforce_limit,
            )
            instances = sorted(instances, key=self._instance_sort_key)
            field_to_id = struct_item.field_class_to_id.class_to_id
            structuring_mask[flat_idx] = True
            structuring_count[flat_idx] = len(instances)
            if structuring_relation_group_mask is not None:
                # A root-only multi-level group has no meaningful directed
                # hierarchy to supervise. In particular, repeated flat records
                # must not turn every fixed-slot pair into an all-negative
                # relation target merely because they share the multi-level
                # processor with genuinely nested examples.
                hierarchy = list(getattr(struct_item, 'hierarchy', None) or [])
                structuring_relation_group_mask[flat_idx] = bool(
                    struct_item.multi_level
                    and any(
                        node.get('parent_path') is not None
                        for node in hierarchy
                        if isinstance(node, dict)
                    )
                )

            if struct_item.multi_level:
                multi_meta = batch_list[batch_idx].get(MULTI_LEVEL_META_KEY) or {}
                group_meta = next(
                    (
                        group for group in multi_meta.get('groups', [])
                        if group.get('data_key') == data_key
                    ),
                    {},
                )
                node_to_instance = {
                    instance.get(NODE_ID_KEY): inst_idx
                    for inst_idx, instance in enumerate(instances[:max_instances])
                    if instance.get(NODE_ID_KEY) is not None
                }
                for parent_node, child_node in group_meta.get('relations', []):
                    parent_idx = node_to_instance.get(parent_node)
                    child_idx = node_to_instance.get(child_node)
                    if parent_idx is not None and child_idx is not None:
                        structuring_relation_labels[
                            flat_idx, parent_idx, child_idx
                        ] = 1.0

            for inst_idx, instance in enumerate(instances):
                if inst_idx >= max_instances:
                    break
                for field_name, value in instance.items():
                    if is_internal_instance_key(field_name):
                        continue
                    if field_name not in field_to_id:
                        continue
                    field_id = field_to_id[field_name]
                    if field_id >= max_fields:
                        continue
                    values = value if isinstance(value, list) else [value]
                    for field_value in values:
                        if not isinstance(field_value, dict):
                            continue
                        st = field_value.get('start', -1)
                        ed = field_value.get('end', -1)
                        if (
                            st < 0
                            or ed < st
                            or st >= item_limit
                            or ed >= item_limit
                        ):
                            continue
                        structuring_labels[flat_idx, inst_idx, st, field_id, 0] = 1.0
                        structuring_labels[flat_idx, inst_idx, ed, field_id, 1] = 1.0
                        structuring_labels[flat_idx, inst_idx, st:ed + 1, field_id, 2] = 1.0

        result = {
            "structuring_labels": structuring_labels,
            "structuring_mask": structuring_mask,
            "structuring_batch_idx": structuring_batch_idx,
            "structuring_count": structuring_count,
        }
        if structuring_relation_labels is not None:
            result["structuring_relation_labels"] = structuring_relation_labels
            result["structuring_relation_group_mask"] = (
                structuring_relation_group_mask
            )
        return result

    def create_span_labels(self, batch_list, classes_mapping, max_seq_len=0, **kwargs):
        """Create span-level labels for structuring when represent_spans is enabled.

        Returns dict with:
            structuring_span_idx: (total_groups, max_spans, 2)
            structuring_span_labels: (total_groups, max_spans, max_instances, max_fields)
            structuring_span_mask: (total_groups, max_spans)
            structuring_span_batch_idx: (total_groups,)
        """
        struct_cfg = self._span_config
        if struct_cfg is None:
            return None

        sequence_lengths, max_seq_len = self._normalize_sequence_lengths(
            batch_list,
            kwargs.get("sequence_lengths"),
            max_seq_len,
        )
        source_sequence_lengths, _ = self._normalize_sequence_lengths(
            batch_list,
            kwargs.get("source_sequence_lengths"),
            max_seq_len,
        )

        for item in batch_list:
            self.resolve_spans(item)

        total_groups = classes_mapping.total_structuring_groups()
        if total_groups == 0:
            return None

        neg_ratio = getattr(struct_cfg, 'neg_spans_ratio', 1.0)
        max_instances = 0
        max_fields = 0
        has_any = False

        all_group_spans = []
        batch_indices = []

        for flat_idx, batch_idx, group_idx, struct_item in classes_mapping.flat_structuring_iter():
            structuring_data = batch_list[batch_idx].get('structuring', {})
            data_key = struct_item.data_key or struct_item.name
            field_to_id = struct_item.field_class_to_id.class_to_id
            max_fields = max(max_fields, len(field_to_id))
            batch_indices.append(batch_idx)

            # One pooled candidate per unique entity boundary. The value is a
            # set of (record, field) targets so multi-field/multi-record
            # annotations remain expressible without sending duplicate span
            # representations into the second stage.
            span_targets = {}
            item_limit = (
                sequence_lengths[batch_idx]
                if sequence_lengths is not None
                else max_seq_len
            )
            enforce_limit = sequence_lengths is not None and (
                source_sequence_lengths is None
                or item_limit < source_sequence_lengths[batch_idx]
            )

            if data_key in structuring_data:
                instances = self._filter_instances(
                    batch_list[batch_idx],
                    data_key,
                    structuring_data[data_key],
                    item_limit,
                    enforce_limit=enforce_limit,
                )
                instances = sorted(instances, key=self._instance_sort_key)
                max_instances = max(max_instances, len(instances))

                for inst_idx, instance in enumerate(instances):
                    for field_name, value in instance.items():
                        if is_internal_instance_key(field_name):
                            continue
                        if field_name not in field_to_id:
                            continue
                        field_id = field_to_id[field_name]
                        values = value if isinstance(value, list) else [value]
                        for field_value in values:
                            if not isinstance(field_value, dict):
                                continue
                            st = field_value.get('start', -1)
                            ed = field_value.get('end', -1)
                            if 0 <= st <= ed < item_limit:
                                span_targets.setdefault((st, ed), set()).add(
                                    (inst_idx, field_id)
                                )
                                has_any = True

            group_spans = [
                (start, end, targets)
                for (start, end), targets in span_targets.items()
            ]
            neg_count = int(len(group_spans) * neg_ratio)
            if neg_count > 0 and item_limit > 0:
                negatives = self._generate_negative_spans(
                    set(span_targets),
                    item_limit,
                    neg_count,
                )
                for st, ed in negatives:
                    group_spans.append((st, ed, set()))

            all_group_spans.append(group_spans)

        if not has_any or max_instances == 0 or max_fields == 0:
            return None

        if self._fixed_slot_pad:
            max_instances = max(max_instances, self._fixed_slot_pad)

        max_spans = max((len(s) for s in all_group_spans), default=0)
        if max_spans == 0:
            return None

        span_idx = torch.zeros(total_groups, max_spans, 2, dtype=torch.long)
        span_labels = torch.zeros(total_groups, max_spans, max_instances, max_fields, dtype=torch.float)
        span_mask = torch.zeros(total_groups, max_spans, dtype=torch.bool)
        span_batch_idx = torch.tensor(batch_indices, dtype=torch.long)

        for g, group_spans in enumerate(all_group_spans):
            for s, (st, ed, targets) in enumerate(group_spans):
                span_idx[g, s, 0] = st
                span_idx[g, s, 1] = ed
                span_mask[g, s] = True
                for inst_idx, field_id in targets:
                    span_labels[g, s, inst_idx, field_id] = 1.0

        return {
            "structuring_span_idx": span_idx,
            "structuring_span_labels": span_labels,
            "structuring_span_mask": span_mask,
            "structuring_span_batch_idx": span_batch_idx,
        }

    def prepare_label_encoder_inputs(self, classes_mapping, labels_tokenizer):
        if labels_tokenizer is None:
            return None

        all_label_strings = []
        group_sizes = []
        has_any = False

        total_groups = classes_mapping.total_structuring_groups()
        if total_groups == 0:
            return None

        for _, _, _, struct_item in classes_mapping.flat_structuring_iter():
            labels = list(struct_item.field_class_to_id.class_to_id.keys())
            if labels:
                has_any = True
            group_sizes.append(len(labels))
            all_label_strings.extend(labels)

        if not has_any or not all_label_strings:
            return None

        tokenized = labels_tokenizer(
            all_label_strings, return_tensors="pt", truncation=True,
            padding="longest", add_special_tokens=True
        )
        return {
            "child_labels_input_ids": tokenized["input_ids"],
            "child_labels_attention_mask": tokenized["attention_mask"],
            "child_labels_group_size": torch.LongTensor(group_sizes),
        }
