"""Classification task processor."""

import random
from typing import Dict, List, Optional

import torch

from .. import TaskProcessor
from ...processing.mappings import BaseClassMapping, CatClassMapping, BatchClassesMapping


class ClassificationProcessor(TaskProcessor):
    """Processor for classification task."""

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.cat_token = config.cat_token
        self.parent_token = config.parent_token
        self.sep_token = config.sep_token

    @staticmethod
    def _build_class_to_id(labels, negatives, sample_neg, shuffle_labels):
        if negatives is not None:
            label_set = set(labels)
            labels.extend(
                label for label in negatives[:sample_neg + len(labels)]
                if label not in label_set
            )
            if len(labels) > len(label_set) + sample_neg:
                labels = labels[:len(label_set) + sample_neg]
        if shuffle_labels:
            random.shuffle(labels)
        return {label: idx for idx, label in enumerate(labels)}

    def get_classes_mapping(self, batch_list, cat_negatives=None, sample_neg=100,
                            shuffle_labels=False, **kwargs):
        cat_mapping = []
        for item in batch_list:
            cat_examples = item.get('classification', [])
            class_mapping = []
            for example in cat_examples:
                all_labels = list(example['all_labels'])
                class_to_id = self._build_class_to_id(all_labels, cat_negatives, sample_neg, shuffle_labels)
                class_mapping.append(BaseClassMapping(
                    class_to_id=class_to_id,
                    name=example.get('name', None),
                    description=example.get('description', None),
                ))
            cat_mapping.append(CatClassMapping(cat_class_to_id=class_mapping))
        return cat_mapping

    def contribute_prompt(self, classes_mapping, batch_idx, use_labels_encoder=False):
        cat_mapping = classes_mapping.cat_mapping[batch_idx]
        prompt = []
        for cat_map in cat_mapping.cat_class_to_id:
            prompt.append(self.parent_token)
            if cat_map.name:
                prompt.append(cat_map.name)
            if cat_map.description:
                prompt.append(cat_map.description)
            if not use_labels_encoder:
                for cat in cat_map.class_to_id:
                    prompt.append(f"{self.cat_token} {cat}")
            prompt.append(self.sep_token)
        return prompt

    def create_labels(self, batch_list, classes_mapping, **kwargs):
        total_groups = classes_mapping.total_cat_groups()
        if total_groups == 0:
            return None

        max_num_classes = max(
            len(m.class_to_id)
            for cm in classes_mapping.cat_mapping
            for m in cm.cat_class_to_id
        ) if any(cm.cat_class_to_id for cm in classes_mapping.cat_mapping) else 0

        if max_num_classes == 0:
            return None

        cat_labels = torch.zeros(total_groups, max_num_classes, dtype=torch.float)
        cat_batch_idx = torch.zeros(total_groups, dtype=torch.long)

        for flat_idx, batch_idx, group_idx, mapping in classes_mapping.flat_cat_iter():
            cat_batch_idx[flat_idx] = batch_idx
            cat_examples = batch_list[batch_idx].get('classification', [])
            if group_idx < len(cat_examples):
                for label in cat_examples[group_idx]['true_labels']:
                    if label in mapping.class_to_id:
                        cat_labels[flat_idx, mapping.class_to_id[label]] = 1.0

        return {"cat_labels": cat_labels, "cat_batch_idx": cat_batch_idx}

    def prepare_label_encoder_inputs(self, classes_mapping, labels_tokenizer):
        if labels_tokenizer is None:
            return None

        all_label_strings = []
        group_sizes = []
        has_any = False

        total_groups = classes_mapping.total_cat_groups()
        if total_groups == 0:
            return None

        for _, _, _, mapping in classes_mapping.flat_cat_iter():
            labels = list(mapping.class_to_id.keys())
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
            "cat_labels_input_ids": tokenized["input_ids"],
            "cat_labels_attention_mask": tokenized["attention_mask"],
            "cat_labels_group_size": torch.LongTensor(group_sizes),
        }
