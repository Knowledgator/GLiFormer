"""Joint relex task processor."""

import random
from typing import Dict, List, Optional

import torch

from ..ner.processor import NERProcessor


class JointRelexProcessor(NERProcessor):
    """Processor for joint NER + relation extraction task.

    Inherits NERProcessor for shared span resolution and extraction mapping utilities.
    Relation classes are derived from extraction data by NERProcessor.
    REL tokens are already included by NERProcessor within extraction groups.
    """

    def __init__(self, config, tokenizer=None, words_splitter=None, **kwargs):
        super().__init__(config, tokenizer, words_splitter, **kwargs)
        self.rel_token = config.rel_token

    def get_classes_mapping(self, batch_list, **kwargs):
        # Relation mappings are derived from extraction data by NERProcessor
        return None

    def contribute_prompt(self, classes_mapping, batch_idx, use_labels_encoder=False):
        # REL tokens are already included by NERProcessor within extraction groups
        return []

    def create_labels(self, batch_list, classes_mapping, **kwargs):
        total_groups = classes_mapping.total_extraction_groups()
        if total_groups == 0:
            return None

        max_entities = 0
        max_rel_classes = 0
        has_any_relations = False
        for flat_idx, batch_idx, group_idx, ext_mapping in classes_mapping.flat_extraction_iter():
            if ext_mapping.rel_class_to_id is not None:
                extraction_examples = batch_list[batch_idx].get('extraction', [])
                if group_idx < len(extraction_examples):
                    has_any_relations = True
                    max_entities = max(max_entities, len(extraction_examples[group_idx].get('ner', [])))
                    max_rel_classes = max(max_rel_classes, len(ext_mapping.rel_class_to_id.class_to_id))

        if not has_any_relations or max_entities == 0 or max_rel_classes == 0:
            return None

        rel_labels = torch.zeros(
            total_groups, max_entities, max_entities, max_rel_classes,
            dtype=torch.float
        )
        # Candidate-pair adjacency used by the GLiNER-relex algorithm:
        # positives plus sampled no-relation pairs. This is intentionally
        # distinct from rel_labels.sum(-1), which marks only positive relations.
        rel_pair_mask = torch.zeros(total_groups, max_entities, max_entities, dtype=torch.float)
        rel_mask = torch.zeros(total_groups, dtype=torch.bool)
        rel_batch_idx = torch.zeros(total_groups, dtype=torch.long)
        rel_span_idx = torch.zeros(total_groups, max_entities, 2, dtype=torch.long)
        rel_span_mask = torch.zeros(total_groups, max_entities, dtype=torch.bool)

        max_seq_len = kwargs.get("max_seq_len", 0)
        add_reversed_negatives = kwargs.get("add_reversed_negatives", True)
        add_random_negatives = kwargs.get("add_random_negatives", True)
        negative_ratio = kwargs.get("negative_ratio", (1.0, 10.0))

        for flat_idx, batch_idx, group_idx, ext_mapping in classes_mapping.flat_extraction_iter():
            rel_batch_idx[flat_idx] = batch_idx
            if ext_mapping.rel_class_to_id is None:
                continue
            extraction_examples = batch_list[batch_idx].get('extraction', [])
            if group_idx >= len(extraction_examples):
                continue
            rel_mask[flat_idx] = True
            mapping = ext_mapping.rel_class_to_id
            example = extraction_examples[group_idx]

            # Compact valid entities into 0..n-1 so build_all_entity_pairs
            # (which assumes contiguous indices) stays in sync with rel_labels.
            # Track the remap so relation head/tail ids land in the right slot.
            old_to_new = {}
            for orig_id, ent in enumerate(example.get('ner', [])):
                new_id = len(old_to_new)
                if new_id >= max_entities:
                    break
                start, end = ent[0], ent[1]
                if max_seq_len and (start >= max_seq_len or end >= max_seq_len):
                    continue
                rel_span_idx[flat_idx, new_id, 0] = start
                rel_span_idx[flat_idx, new_id, 1] = end
                rel_span_mask[flat_idx, new_id] = True
                old_to_new[orig_id] = new_id

            positive_pairs = set()
            for head_id, rel_type, tail_id in example.get('relations', []):
                if rel_type not in mapping.class_to_id:
                    continue
                new_head = old_to_new.get(head_id)
                new_tail = old_to_new.get(tail_id)
                if new_head is None or new_tail is None:
                    continue
                rel_class_idx = mapping.class_to_id[rel_type]
                rel_labels[flat_idx, new_head, new_tail, rel_class_idx] = 1.0
                if new_head != new_tail:
                    positive_pairs.add((new_head, new_tail))

            n_valid = len(old_to_new)
            negative_pairs = set()
            if n_valid > 1:
                if add_reversed_negatives:
                    for head, tail in positive_pairs:
                        reversed_pair = (tail, head)
                        if reversed_pair not in positive_pairs:
                            negative_pairs.add(reversed_pair)

                if add_random_negatives:
                    if isinstance(negative_ratio, (tuple, list)):
                        ratio = random.uniform(float(negative_ratio[0]), float(negative_ratio[1]))
                    else:
                        ratio = float(negative_ratio)
                    target_negatives = max(1, int(len(positive_pairs) * ratio))
                    attempts = 0
                    max_attempts = max(target_negatives * 10, 0)
                    while len(negative_pairs) < target_negatives and attempts < max_attempts:
                        attempts += 1
                        head = random.randint(0, n_valid - 1)
                        tail = random.randint(0, n_valid - 1)
                        if head == tail:
                            continue
                        pair = (head, tail)
                        if pair in positive_pairs or pair in negative_pairs:
                            continue
                        negative_pairs.add(pair)

            for head, tail in positive_pairs | negative_pairs:
                rel_pair_mask[flat_idx, head, tail] = 1.0

        return {
            "rel_labels": rel_labels,
            "rel_pair_mask": rel_pair_mask,
            "rel_mask": rel_mask,
            "rel_batch_idx": rel_batch_idx,
            "rel_span_idx": rel_span_idx,
            "rel_span_mask": rel_span_mask,
        }

    def prepare_label_encoder_inputs(self, classes_mapping, labels_tokenizer):
        if labels_tokenizer is None:
            return None

        all_label_strings = []
        group_sizes = []
        has_any = False

        for _, _, _, ext_mapping in classes_mapping.flat_extraction_iter():
            if ext_mapping.rel_class_to_id is not None:
                labels = list(ext_mapping.rel_class_to_id.class_to_id.keys())
                has_any = True
            else:
                labels = []
            group_sizes.append(len(labels))
            all_label_strings.extend(labels)

        if not has_any or not all_label_strings:
            return None

        tokenized = labels_tokenizer(
            all_label_strings, return_tensors="pt", truncation=True,
            padding="longest", add_special_tokens=True
        )
        return {
            "rel_labels_input_ids": tokenized["input_ids"],
            "rel_labels_attention_mask": tokenized["attention_mask"],
            "rel_labels_group_size": torch.LongTensor(group_sizes),
        }
