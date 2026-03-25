"""NER task processor."""

import re
import random
from typing import Dict, List, Optional

import torch

from .. import TaskProcessor
from ...mappings import (
    BaseClassMapping, ExtractionItemMapping, ExtractionClassMapping, BatchClassesMapping,
)


class NERProcessor(TaskProcessor):
    """Processor for NER task: class mappings, prompts, labels, span resolution."""

    def __init__(self, config, tokenizer=None, words_splitter=None, **kwargs):
        super().__init__(config, tokenizer, words_splitter)
        self.words_splitter = words_splitter
        self.ent_token = config.ent_token
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

    def get_classes_mapping(self, batch_list, ner_negatives=None, rel_negatives=None,
                            sample_neg=100, shuffle_labels=False, **kwargs):
        extraction_mapping = []
        for item in batch_list:
            extraction_examples = item.get('extraction', [])
            item_mappings = []
            for example in extraction_examples:
                ner_labels = list({ent[-1] for ent in example.get('ner', [])})
                ner_class_to_id = self._build_class_to_id(ner_labels, ner_negatives, sample_neg, shuffle_labels)
                name = example.get('name', None)
                description = example.get('description', None)
                ner_mapping = BaseClassMapping(
                    class_to_id=ner_class_to_id, name=name, description=description
                )

                rel_mapping = None
                relations = example.get('relations', [])
                if relations:
                    rel_labels = list({rel[-1] for rel in relations})
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
            prompt.append(self.sep_token)
        return prompt

    def resolve_spans(self, item):
        if not item.get('extraction') or not item.get('text'):
            return
        text = item.get('text', '')
        tokens_with_spans = list(self.words_splitter(text))
        tokens = [tok for tok, _, _ in tokens_with_spans]
        if 'tokenized_text' not in item:
            item['tokenized_text'] = tokens

        for ext_example in item.get('extraction', []):
            ner = ext_example.get('ner', [])
            if ner and not (len(ner[0]) == 3 and isinstance(ner[0][0], int)):
                ext_example['ner'] = self._resolve_entity_spans(
                    text, tokens_with_spans, ner
                )

        self._sort_extraction_data(item)

    @staticmethod
    def _resolve_entity_spans(text, tokens_with_spans, ner):
        if not ner:
            return []
        s2t = {s: idx for idx, (_, s, _) in enumerate(tokens_with_spans)}
        e2t = {e: idx for idx, (_, _, e) in enumerate(tokens_with_spans)}
        resolved = []
        for ent in ner:
            if len(ent) == 3 and isinstance(ent[0], int):
                resolved.append(list(ent))
            else:
                ent_text, label = ent[0], ent[-1]
                try:
                    for match in re.finditer(re.escape(ent_text), text, re.IGNORECASE):
                        s, e = match.start(), match.end()
                        if s in s2t and e in e2t:
                            resolved.append([s2t[s], e2t[e], label])
                except (ValueError, re.error):
                    continue
        return resolved

    @staticmethod
    def _sort_extraction_data(item):
        for ext_example in item.get('extraction', []):
            ner = ext_example.get('ner', [])
            if ner:
                ext_example['ner'] = sorted(ner, key=lambda x: (x[0], x[1]))
            relations = ext_example.get('relations', [])
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
            total_groups, max_seq_len, max_num_classes + 1, 3,
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
                # Parent class (index 0)
                ner_labels[flat_idx, start, 0, 0] = 1
                ner_labels[flat_idx, end, 0, 1] = 1
                ner_labels[flat_idx, start:end + 1, 0, 2] = 1
                # Child class labels (1-indexed)
                if label in mapping.class_to_id:
                    class_idx = mapping.class_to_id[label] + 1
                    ner_labels[flat_idx, start, class_idx, 0] = 1
                    ner_labels[flat_idx, end, class_idx, 1] = 1
                    ner_labels[flat_idx, start:end + 1, class_idx, 2] = 1

        return {"ner_labels": ner_labels, "ner_batch_idx": ner_batch_idx}

    def _generate_negative_spans(self, positive_spans, num_tokens, num_negatives, max_width=None):
        if max_width is None:
            max_width = getattr(self.config, "max_width", 10)
        negative_spans = []
        attempts = 0
        max_attempts = num_negatives * 20
        while len(negative_spans) < num_negatives and attempts < max_attempts:
            attempts += 1
            start = random.randint(0, num_tokens - 1)
            width = random.randint(1, min(max_width, num_tokens - start))
            end = start + width - 1
            span = (start, end)
            if span in positive_spans:
                continue
            overlaps = False
            for pos_start, pos_end in positive_spans:
                if not (end < pos_start or start > pos_end):
                    overlaps = True
                    break
            if not overlaps and span not in negative_spans:
                negative_spans.append(span)
        return negative_spans

    def prepare_span_idx(self, ner, classes_to_id, num_tokens):
        if ner is not None and getattr(self.config, 'represent_spans', False):
            span_idx_list = []
            span_label_list = []
            positive_spans = set()
            for start, end, label in ner:
                if label in classes_to_id and end < num_tokens:
                    span_idx_list.append([start, end])
                    span_label_list.append(classes_to_id[label])
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
