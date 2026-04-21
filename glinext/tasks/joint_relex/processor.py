"""Joint relex task processor."""

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
        rel_mask = torch.zeros(total_groups, dtype=torch.bool)
        rel_batch_idx = torch.zeros(total_groups, dtype=torch.long)
        rel_span_idx = torch.zeros(total_groups, max_entities, 2, dtype=torch.long)
        rel_span_mask = torch.zeros(total_groups, max_entities, dtype=torch.bool)

        max_seq_len = kwargs.get("max_seq_len", 0)

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

            for ent_id, ent in enumerate(example.get('ner', [])):
                if ent_id >= max_entities:
                    break
                start, end = ent[0], ent[1]
                if max_seq_len and (start >= max_seq_len or end >= max_seq_len):
                    continue
                rel_span_idx[flat_idx, ent_id, 0] = start
                rel_span_idx[flat_idx, ent_id, 1] = end
                rel_span_mask[flat_idx, ent_id] = True

            for head_id, rel_type, tail_id in example.get('relations', []):
                if rel_type in mapping.class_to_id:
                    rel_class_idx = mapping.class_to_id[rel_type]
                    if head_id < max_entities and tail_id < max_entities:
                        rel_labels[flat_idx, head_id, tail_id, rel_class_idx] = 1.0

        return {
            "rel_labels": rel_labels,
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
