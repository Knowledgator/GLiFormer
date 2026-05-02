"""Structuring task processor."""

import random
from typing import Dict, List, Optional

import torch

from ..span_processor import SpanProcessor
from ...processing.mappings import (
    BaseClassMapping, StructuringItemMapping, StructuringClassMapping, BatchClassesMapping,
)


class StructuringProcessor(SpanProcessor):
    """Processor for structuring task."""

    def __init__(self, config, tokenizer=None, words_splitter=None, **kwargs):
        super().__init__(config, tokenizer, words_splitter,
                         parent_token=getattr(config, 'struct_parent_token', None), **kwargs)
        self.child_token = config.child_token
        # Fixed-slot anchors always emit ``num_slots`` anchors; labels must be
        # padded to that size so Hungarian sees unmatched slots and trains
        # them as negatives.
        struct_cfg = getattr(config, 'structuring_config', None)
        anchor_mode = getattr(struct_cfg, 'anchor_mode', '') if struct_cfg else ''
        self._fixed_slot_pad = (
            getattr(struct_cfg, 'num_fixed_slots', 0)
            if anchor_mode in ('fixed', 'fixed_rnn', 'fixed_transformer')
            else 0
        )

    @staticmethod
    def _instance_sort_key(instance):
        """Return the earliest span start position across all fields in an instance."""
        min_start = float('inf')
        for value in instance.values():
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
    def _instance_has_valid_span(instance, max_seq_len=0):
        for value in instance.values():
            values = value if isinstance(value, list) else [value]
            for field_value in values:
                if not isinstance(field_value, dict):
                    continue
                st = field_value.get('start', -1)
                ed = field_value.get('end', -1)
                if st < 0 or ed < 0:
                    continue
                if max_seq_len > 0 and (st >= max_seq_len or ed >= max_seq_len):
                    continue
                return True
        return False

    def get_classes_mapping(self, batch_list, shuffle_labels=False, **kwargs):
        structuring_mapping = []
        for item in batch_list:
            structuring_data = item.get('structuring', {})
            prompt = item.get('prompt')
            item_mappings = []
            for schema_name, instances in structuring_data.items():
                field_names = []
                seen = set()
                for instance in instances:
                    for field_name in instance:
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
                ))
            structuring_mapping.append(StructuringClassMapping(items=item_mappings))
        return structuring_mapping

    def contribute_prompt(self, classes_mapping, batch_idx, use_labels_encoder=False):
        if not hasattr(classes_mapping, 'structuring_mapping'):
            return []
        if batch_idx >= len(classes_mapping.structuring_mapping):
            return []
        prompt = []
        for struct_item in classes_mapping.structuring_mapping[batch_idx].items:
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
        return fields if isinstance(fields, list) else fields.get("fields", [])

    def contribute_inference_input(self, item, structures=None, **kwargs):
        if not structures:
            return

        structuring = {}
        structuring_schema = {}
        for schema_name, fields in structures.items():
            field_list = self._structure_fields(fields)
            structuring[schema_name] = [dict.fromkeys(field_list, "")] if field_list else []
            structuring_schema[schema_name] = field_list
        item["structuring"] = structuring
        item["structuring_schema"] = structuring_schema

    def empty_inference_result(self, num_texts: int, structures=None, **kwargs):
        if structures is None:
            return None
        return {"structuring": [{} for _ in range(num_texts)]}

    def _normalize_field_value(self, text, tokens_with_spans, field_name, value, first_only=False):
        value_text = value.get('text') if isinstance(value, dict) else value
        if value_text is None and isinstance(value, dict) and 'start' in value and 'end' in value:
            try:
                value_text = text[int(value['start']):int(value['end'])]
            except (TypeError, ValueError):
                value_text = ''

        spans = self._resolve_labeled_span(
            text, tokens_with_spans, value, label=field_name, first_only=first_only,
        )
        if not spans:
            return [{
                'text': str(value_text),
                'start': -1,
                'end': -1,
            }]

        return [
            {
                'text': str(value_text),
                'start': start,
                'end': end,
            }
            for start, end, _ in spans
        ]

    def resolve_spans(self, item):
        if item.get('_glinext_structuring_spans_resolved'):
            return
        structuring = item.get('structuring', {})
        if not structuring:
            return

        text = item.get('text', '')
        tokens_with_spans, _ = self._tokenize_text(item)
        if tokens_with_spans is None:
            return

        for schema_name, instances in structuring.items():
            for instance in instances:
                for field_name, value in list(instance.items()):
                    if isinstance(value, list):
                        resolved_list = []
                        for v in value:
                            resolved_list.extend(self._normalize_field_value(
                                text, tokens_with_spans, field_name, v,
                            ))
                        instance[field_name] = resolved_list
                    else:
                        normalized = self._normalize_field_value(
                            text, tokens_with_spans, field_name, value,
                        )
                        instance[field_name] = normalized[0] if len(normalized) == 1 else normalized
        item['_glinext_structuring_spans_resolved'] = True

    def create_labels(self, batch_list, classes_mapping, max_seq_len=0, **kwargs):
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
            schema_name = struct_item.name
            if schema_name not in structuring_data:
                continue
            instances = structuring_data[schema_name]
            if instances:
                has_schema = True
                valid_instances = [
                    instance for instance in instances
                    if self._instance_has_valid_span(instance, max_seq_len)
                ]
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

        for flat_idx, batch_idx, group_idx, struct_item in classes_mapping.flat_structuring_iter():
            structuring_batch_idx[flat_idx] = batch_idx
            structuring_data = batch_list[batch_idx].get('structuring', {})
            schema_name = struct_item.name
            if schema_name not in structuring_data:
                continue

            instances = [
                instance for instance in structuring_data[schema_name]
                if self._instance_has_valid_span(instance, max_seq_len)
            ]
            instances = sorted(instances, key=self._instance_sort_key)
            field_to_id = struct_item.field_class_to_id.class_to_id
            structuring_mask[flat_idx] = True
            structuring_count[flat_idx] = len(instances)

            for inst_idx, instance in enumerate(instances):
                if inst_idx >= max_instances:
                    break
                for field_name, value in instance.items():
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
                        if st < 0 or ed < 0 or st >= max_seq_len or ed >= max_seq_len:
                            continue
                        structuring_labels[flat_idx, inst_idx, st, field_id, 0] = 1.0
                        structuring_labels[flat_idx, inst_idx, ed, field_id, 1] = 1.0
                        structuring_labels[flat_idx, inst_idx, st:ed + 1, field_id, 2] = 1.0

        return {
            "structuring_labels": structuring_labels,
            "structuring_mask": structuring_mask,
            "structuring_batch_idx": structuring_batch_idx,
            "structuring_count": structuring_count,
        }

    def create_span_labels(self, batch_list, classes_mapping, max_seq_len=0, **kwargs):
        """Create span-level labels for structuring when represent_spans is enabled.

        Returns dict with:
            structuring_span_idx: (total_groups, max_spans, 2)
            structuring_span_labels: (total_groups, max_spans, max_instances, max_fields)
            structuring_span_mask: (total_groups, max_spans)
            structuring_span_batch_idx: (total_groups,)
        """
        struct_cfg = getattr(self.config, 'structuring_config', None)
        if struct_cfg is None or not getattr(struct_cfg, 'represent_spans', False):
            return None

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
            schema_name = struct_item.name
            field_to_id = struct_item.field_class_to_id.class_to_id
            max_fields = max(max_fields, len(field_to_id))
            batch_indices.append(batch_idx)

            group_spans = []  # (start, end, inst_idx, field_id)
            positive_spans = set()

            if schema_name in structuring_data:
                instances = sorted(structuring_data[schema_name], key=self._instance_sort_key)
                max_instances = max(max_instances, len(instances))

                for inst_idx, instance in enumerate(instances):
                    for field_name, value in instance.items():
                        if field_name not in field_to_id:
                            continue
                        field_id = field_to_id[field_name]
                        values = value if isinstance(value, list) else [value]
                        for field_value in values:
                            if not isinstance(field_value, dict):
                                continue
                            st = field_value.get('start', -1)
                            ed = field_value.get('end', -1)
                            if 0 <= st < max_seq_len and 0 <= ed < max_seq_len:
                                group_spans.append((st, ed, inst_idx, field_id))
                                positive_spans.add((st, ed))
                                has_any = True

            neg_count = int(len(group_spans) * neg_ratio)
            if neg_count > 0 and max_seq_len > 0:
                negatives = self._generate_negative_spans(positive_spans, max_seq_len, neg_count)
                for st, ed in negatives:
                    group_spans.append((st, ed, -1, -1))

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
            for s, (st, ed, inst_idx, field_id) in enumerate(group_spans):
                span_idx[g, s, 0] = st
                span_idx[g, s, 1] = ed
                span_mask[g, s] = True
                if inst_idx >= 0 and field_id >= 0:
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
