"""Structuring task processor."""

import re
import random
from typing import Dict, List, Optional

import torch

from .. import TaskProcessor
from ...mappings import (
    BaseClassMapping, StructuringItemMapping, StructuringClassMapping, BatchClassesMapping,
)


class StructuringProcessor(TaskProcessor):
    """Processor for structuring task."""

    def __init__(self, config, tokenizer=None, words_splitter=None, **kwargs):
        super().__init__(config, tokenizer, words_splitter)
        self.words_splitter = words_splitter
        self.child_token = config.child_token
        self.parent_token = config.parent_token
        self.sep_token = config.sep_token

    def get_classes_mapping(self, batch_list, shuffle_labels=False, **kwargs):
        structuring_mapping = []
        for item in batch_list:
            structuring_data = item.get('structuring', {})
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
                item_mappings.append(StructuringItemMapping(
                    field_class_to_id=BaseClassMapping(
                        class_to_id=field_class_to_id, name=schema_name,
                    ),
                    name=schema_name,
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

    def resolve_spans(self, item):
        text = item.get('text', '')
        structuring = item.get('structuring', {})
        if not structuring or not text:
            return

        tokens_with_spans = list(self.words_splitter(text))
        if 'tokenized_text' not in item:
            item['tokenized_text'] = [tok for tok, _, _ in tokens_with_spans]

        for schema_name, instances in structuring.items():
            for instance in instances:
                for field_name, value in list(instance.items()):
                    if isinstance(value, dict) and 'text' in value:
                        if 'start' not in value or not isinstance(value['start'], int):
                            text_val = str(value['text'])
                            resolved = self._resolve_entity_spans(
                                text, tokens_with_spans, [[text_val, field_name]]
                            )
                            if resolved:
                                value['start'] = resolved[0][0]
                                value['end'] = resolved[0][1]
                            else:
                                value['start'] = -1
                                value['end'] = -1
                    elif isinstance(value, list):
                        resolved_list = []
                        for v in value:
                            v_str = str(v)
                            resolved = self._resolve_entity_spans(
                                text, tokens_with_spans, [[v_str, field_name]]
                            )
                            if resolved:
                                resolved_list.append({
                                    'text': v_str,
                                    'start': resolved[0][0],
                                    'end': resolved[0][1],
                                })
                            else:
                                resolved_list.append({
                                    'text': v_str, 'start': -1, 'end': -1,
                                })
                        instance[field_name] = resolved_list
                    else:
                        text_val = str(value)
                        resolved = self._resolve_entity_spans(
                            text, tokens_with_spans, [[text_val, field_name]]
                        )
                        if resolved:
                            instance[field_name] = {
                                'text': text_val,
                                'start': resolved[0][0],
                                'end': resolved[0][1],
                            }
                        else:
                            instance[field_name] = {
                                'text': text_val, 'start': -1, 'end': -1,
                            }

    @staticmethod
    def _resolve_entity_spans(text, tokens_with_spans, ner):
        if not ner:
            return []
        s2t = {s: idx for idx, (_, s, _) in enumerate(tokens_with_spans)}
        e2t = {e: idx for idx, (_, _, e) in enumerate(tokens_with_spans)}
        resolved = []
        for ent in ner:
            ent_text, label = ent[0], ent[-1]
            try:
                for match in re.finditer(re.escape(ent_text), text, re.IGNORECASE):
                    s, e = match.start(), match.end()
                    if s in s2t and e in e2t:
                        resolved.append([s2t[s], e2t[e], label])
            except (ValueError, re.error):
                continue
        return resolved

    def create_labels(self, batch_list, classes_mapping, max_seq_len=0, **kwargs):
        total_groups = classes_mapping.total_structuring_groups()
        if total_groups == 0:
            return None

        max_instances = 0
        max_fields = 0
        has_any = False

        for flat_idx, batch_idx, group_idx, struct_item in classes_mapping.flat_structuring_iter():
            structuring_data = batch_list[batch_idx].get('structuring', {})
            schema_name = struct_item.name
            if schema_name not in structuring_data:
                continue
            instances = structuring_data[schema_name]
            if instances:
                has_any = True
                max_instances = max(max_instances, len(instances))
                max_fields = max(max_fields, len(struct_item.field_class_to_id.class_to_id))

        if not has_any or max_instances == 0 or max_fields == 0:
            return None

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

            instances = structuring_data[schema_name]
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
                    if isinstance(value, dict):
                        st = value.get('start', -1)
                        ed = value.get('end', -1)
                    else:
                        continue
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
