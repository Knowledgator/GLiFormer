"""NER task processor."""

import random
from typing import Dict, List, Optional

import torch

from ..span_processor import SpanProcessor
from ...processing.mappings import (
    BaseClassMapping, ExtractionItemMapping, ExtractionClassMapping, BatchClassesMapping,
)


class NERProcessor(SpanProcessor):
    """Processor for NER task: class mappings, prompts, labels, span resolution."""

    def __init__(self, config, tokenizer=None, words_splitter=None, **kwargs):
        super().__init__(config, tokenizer, words_splitter,
                         parent_token=getattr(config, 'ner_parent_token', None), **kwargs)
        self.ent_token = config.ent_token
        self.rel_token = getattr(config, 'rel_token', None)

    @staticmethod
    def _build_class_to_id(labels, negatives, sample_neg, shuffle_labels):
        labels = list(dict.fromkeys(labels))
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

    def get_classes_mapping(self, batch_list, ner_negatives=None, rel_negatives=None,
                            sample_neg=100, shuffle_labels=False, **kwargs):
        extraction_mapping = []
        for item in batch_list:
            extraction_examples = item.get('extraction', [])
            item_mappings = []
            for example in extraction_examples:
                if 'all_labels' in example:
                    ner_labels = list(example['all_labels'])
                else:
                    ner_labels = list({ent[-1] for ent in example.get('ner', [])})
                ner_class_to_id = self._build_class_to_id(ner_labels, ner_negatives, sample_neg, shuffle_labels)
                name = example.get('name', None)
                description = example.get('description', None)
                ner_mapping = BaseClassMapping(
                    class_to_id=ner_class_to_id, name=name, description=description
                )

                rel_mapping = None
                if 'all_rel_labels' in example:
                    rel_labels = list(example['all_rel_labels'])
                else:
                    # Relations are (head_id, rel_type, tail_id); rel_type lives at index 1.
                    rel_labels = list({rel[1] for rel in example.get('relations', [])})
                if rel_labels:
                    rel_class_to_id = self._build_class_to_id(rel_labels, rel_negatives, sample_neg, shuffle_labels)
                    rel_mapping = BaseClassMapping(
                        class_to_id=rel_class_to_id, name=name, description=description
                    )

                item_mappings.append(ExtractionItemMapping(
                    ner_class_to_id=ner_mapping,
                    rel_class_to_id=rel_mapping,
                ))
            extraction_mapping.append(ExtractionClassMapping(items=item_mappings))
        return extraction_mapping

    def contribute_prompt(self, classes_mapping, batch_idx, use_labels_encoder=False):
        extraction_mapping = classes_mapping.extraction_mapping[batch_idx]
        prompt = []
        for ext_item in extraction_mapping.items:
            prompt.append(self.parent_token)
            ner_map = ext_item.ner_class_to_id
            if ner_map.name:
                prompt.append(ner_map.name)
            if ner_map.description:
                prompt.append(ner_map.description)
            if not use_labels_encoder:
                if ner_map.class_to_id is not None:
                    for ent in ner_map.class_to_id:
                        prompt.append(f"{self.ent_token} {ent}")
                else:
                    prompt.append(f"{self.ent_token} ENTITY")
                rel_map = ext_item.rel_class_to_id
                if self.rel_token is not None and rel_map is not None and rel_map.class_to_id:
                    for rel in rel_map.class_to_id:
                        prompt.append(f"{self.rel_token} {rel}")
            prompt.append(self.sep_token)
        return prompt

    def resolve_spans(self, item):
        if not item.get('extraction') or not item.get('text'):
            return
        text = item.get('text', '')
        tokens_with_spans, tokens = self._tokenize_text(item)
        if tokens_with_spans is None:
            return

        for ext_example in item.get('extraction', []):
            ner = ext_example.get('ner', [])
            if not ner or (len(ner[0]) == 3 and isinstance(ner[0][0], int)):
                continue
            # Resolve per-entity so we can track which originals survived.
            # Relation head_id/tail_id reference positions in the input ner
            # list — if resolution drops an entity, the remaining indices
            # shift and labels get misaligned.
            resolved = []
            old_to_new = {}
            for orig_idx, ent in enumerate(ner):
                single = self._resolve_entity_spans(
                    text, tokens_with_spans, [ent]
                )
                if single:
                    old_to_new[orig_idx] = len(resolved)
                    resolved.append(single[0])
            ext_example['ner'] = resolved

            relations = ext_example.get('relations', [])
            if relations:
                remapped = []
                for rel in relations:
                    new_h = old_to_new.get(rel[0])
                    new_t = old_to_new.get(rel[2])
                    if new_h is None or new_t is None:
                        continue
                    if isinstance(rel, tuple):
                        remapped.append((new_h, rel[1], new_t))
                    else:
                        new_rel = list(rel)
                        new_rel[0] = new_h
                        new_rel[2] = new_t
                        remapped.append(new_rel)
                ext_example['relations'] = remapped

        self._sort_extraction_data(item)

    @staticmethod
    def _sort_extraction_data(item):
        for ext_example in item.get('extraction', []):
            ner = ext_example.get('ner', [])
            relations = ext_example.get('relations', [])
            if ner:
                # Sort entities by (start, end); remap head/tail ids in relations
                # so indices keep pointing to the same entity after reordering.
                indexed = sorted(
                    enumerate(ner), key=lambda pair: (pair[1][0], pair[1][1])
                )
                old_to_new = {old: new for new, (old, _) in enumerate(indexed)}
                ext_example['ner'] = [ent for _, ent in indexed]
                if relations:
                    remapped = []
                    for rel in relations:
                        new_h = old_to_new.get(rel[0], rel[0])
                        new_t = old_to_new.get(rel[2], rel[2])
                        if isinstance(rel, tuple):
                            remapped.append((new_h, rel[1], new_t))
                        else:
                            new_rel = list(rel)
                            new_rel[0] = new_h
                            new_rel[2] = new_t
                            remapped.append(new_rel)
                    relations = remapped
            if relations:
                ext_example['relations'] = sorted(relations, key=lambda x: (x[0], x[2]))

    def create_labels(self, batch_list, classes_mapping, max_seq_len=0, **kwargs):
        total_groups = classes_mapping.total_extraction_groups()
        if total_groups == 0:
            return None

        max_num_classes = max(
            len(item.ner_class_to_id.class_to_id)
            for em in classes_mapping.extraction_mapping
            for item in em.items
        ) if any(em.items for em in classes_mapping.extraction_mapping) else 0

        if max_num_classes == 0:
            return None

        ner_labels = torch.zeros(
            total_groups, max_seq_len, max_num_classes, 3,
            dtype=torch.float
        )
        ner_batch_idx = torch.zeros(total_groups, dtype=torch.long)

        for flat_idx, batch_idx, group_idx, ext_mapping in classes_mapping.flat_extraction_iter():
            ner_batch_idx[flat_idx] = batch_idx
            extraction_examples = batch_list[batch_idx].get('extraction', [])
            if group_idx >= len(extraction_examples):
                continue

            ner_data = extraction_examples[group_idx].get('ner', [])
            mapping = ext_mapping.ner_class_to_id

            for ent in ner_data:
                start, end, label = ent[0], ent[1], ent[-1]
                if start >= max_seq_len or end >= max_seq_len:
                    continue
                if label in mapping.class_to_id:
                    class_idx = mapping.class_to_id[label]
                    ner_labels[flat_idx, start, class_idx, 0] = 1
                    ner_labels[flat_idx, end, class_idx, 1] = 1
                    ner_labels[flat_idx, start:end + 1, class_idx, 2] = 1

        return {"ner_labels": ner_labels, "ner_batch_idx": ner_batch_idx}

    def prepare_span_idx(self, ner, classes_to_id, num_tokens):
        if ner is not None and getattr(self.config, 'represent_spans', False):
            span_idx_list = []
            span_label_list = []
            positive_spans = set()
            for start, end, label in ner:
                if label in classes_to_id and end < num_tokens:
                    span_idx_list.append([start, end])
                    # 1-indexed: 0 is reserved for negative spans so
                    # create_span_labels can distinguish them via `> 0`.
                    span_label_list.append(classes_to_id[label] + 1)
                    positive_spans.add((start, end))
            neg_spans_ratio = getattr(self.config, 'neg_spans_ratio', 0)
            neg_spans_count = int(len(span_idx_list) * neg_spans_ratio)
            if neg_spans_count > 0 and num_tokens > 0:
                max_width = getattr(self.config, "max_width", 10)
                negative_spans = self._generate_negative_spans(
                    positive_spans, num_tokens, neg_spans_count, max_width
                )
                for start, end in negative_spans:
                    span_idx_list.append([start, end])
                    span_label_list.append(0)
            if span_idx_list:
                span_idx = torch.LongTensor(span_idx_list)
                span_label = torch.LongTensor(span_label_list)
            else:
                span_idx = torch.zeros(0, 2, dtype=torch.long)
                span_label = torch.zeros(0, dtype=torch.long)
        else:
            span_idx, span_label = None, None
        return span_idx, span_label

    def prepare_label_encoder_inputs(self, classes_mapping, labels_tokenizer):
        if labels_tokenizer is None:
            return None

        all_label_strings = []
        group_sizes = []
        has_any = False

        total_groups = classes_mapping.total_extraction_groups()
        if total_groups == 0:
            return None

        for _, _, _, ext_mapping in classes_mapping.flat_extraction_iter():
            labels = list(ext_mapping.ner_class_to_id.class_to_id.keys())
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
            "ner_labels_input_ids": tokenized["input_ids"],
            "ner_labels_attention_mask": tokenized["attention_mask"],
            "ner_labels_group_size": torch.LongTensor(group_sizes),
        }
