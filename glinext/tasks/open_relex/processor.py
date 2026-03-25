"""Open relex task processor."""

import re
from typing import Dict, List, Optional

import torch

from .. import TaskProcessor
from ...mappings import (
    BaseClassMapping, OpenRelexItemMapping, OpenRelexClassMapping, BatchClassesMapping,
)


class OpenRelexProcessor(TaskProcessor):
    """Processor for anchor-based open relation extraction.

    Builds its own extraction groups with [P] and [REL] tokens.
    Input data uses the 'open_relex' key with text-based head/tail mentions.
    """

    def __init__(self, config, tokenizer=None, words_splitter=None, **kwargs):
        super().__init__(config, tokenizer, words_splitter)
        self.words_splitter = words_splitter
        self.rel_token = config.rel_token
        self.parent_token = config.parent_token
        self.sep_token = config.sep_token

    def get_classes_mapping(self, batch_list, shuffle_labels=False, **kwargs):
        open_relex_mapping = []
        for item in batch_list:
            open_relex_data = item.get('open_relex', [])
            item_mappings = []
            for group in open_relex_data:
                # Collect unique relation types
                rel_types = []
                seen = set()
                for rel in group.get('relations', []):
                    rel_type = rel.get('relation', '')
                    if rel_type and rel_type not in seen:
                        rel_types.append(rel_type)
                        seen.add(rel_type)
                if shuffle_labels:
                    import random
                    random.shuffle(rel_types)
                rel_class_to_id = {rt: idx for idx, rt in enumerate(rel_types)}
                item_mappings.append(OpenRelexItemMapping(
                    rel_class_to_id=BaseClassMapping(
                        class_to_id=rel_class_to_id,
                        name=group.get('name'),
                    ),
                    name=group.get('name'),
                ))
            open_relex_mapping.append(OpenRelexClassMapping(items=item_mappings))
        return open_relex_mapping

    def contribute_prompt(self, classes_mapping, batch_idx, use_labels_encoder=False):
        if not hasattr(classes_mapping, 'open_relex_mapping'):
            return []
        if batch_idx >= len(classes_mapping.open_relex_mapping):
            return []
        prompt = []
        for relex_item in classes_mapping.open_relex_mapping[batch_idx].items:
            prompt.append(self.parent_token)
            rel_map = relex_item.rel_class_to_id
            if rel_map.name:
                prompt.append(rel_map.name)
            if not use_labels_encoder:
                for rel_name in rel_map.class_to_id:
                    prompt.append(f"{self.rel_token} {rel_name}")
            prompt.append(self.sep_token)
        return prompt

    def resolve_spans(self, item):
        """Resolve head/tail text mentions to token indices."""
        text = item.get('text', '')
        open_relex_data = item.get('open_relex', [])
        if not open_relex_data or not text:
            return

        tokens_with_spans = list(self.words_splitter(text))
        if 'tokenized_text' not in item:
            item['tokenized_text'] = [tok for tok, _, _ in tokens_with_spans]

        for group in open_relex_data:
            for rel in group.get('relations', []):
                for role in ('head', 'tail'):
                    value = rel.get(role)
                    if value is None:
                        continue
                    if isinstance(value, str):
                        # Resolve text → {text, start, end}
                        resolved = self._resolve_text_span(text, tokens_with_spans, value)
                        rel[role] = resolved
                    elif isinstance(value, dict) and 'text' in value:
                        if 'start' not in value or not isinstance(value['start'], int):
                            resolved = self._resolve_text_span(
                                text, tokens_with_spans, str(value['text']),
                            )
                            value.update(resolved)

    @staticmethod
    def _resolve_text_span(text, tokens_with_spans, mention_text):
        """Resolve a text mention to token start/end indices."""
        s2t = {s: idx for idx, (_, s, _) in enumerate(tokens_with_spans)}
        e2t = {e: idx for idx, (_, _, e) in enumerate(tokens_with_spans)}
        try:
            for match in re.finditer(re.escape(mention_text), text, re.IGNORECASE):
                s, e = match.start(), match.end()
                if s in s2t and e in e2t:
                    return {'text': mention_text, 'start': s2t[s], 'end': e2t[e]}
        except (ValueError, re.error):
            pass
        return {'text': mention_text, 'start': -1, 'end': -1}

    def create_labels(self, batch_list, classes_mapping, max_seq_len=0, **kwargs):
        """Create label tensors for open relex.

        Output shape: (total_groups, max_anchors, max_rel_classes, max_seq_len, 2, 3)
            2 = [head, tail], 3 = [start, inside, end]

        Anchor assignment: relations are grouped by unique (head_text, tail_text) pairs
        within each group, with each unique pair becoming an anchor slot.
        """
        total_groups = classes_mapping.total_open_relex_groups()
        if total_groups == 0:
            return None

        max_anchors = 0
        max_rel_classes = 0
        has_any = False

        # First pass: determine dimensions
        group_relations = []
        for flat_idx, batch_idx, group_idx, relex_item in classes_mapping.flat_open_relex_iter():
            open_relex_data = batch_list[batch_idx].get('open_relex', [])
            if group_idx >= len(open_relex_data):
                group_relations.append([])
                continue

            group = open_relex_data[group_idx]
            relations = group.get('relations', [])
            rel_to_id = relex_item.rel_class_to_id.class_to_id

            # Group relations by anchor (unique entity pairs)
            anchor_rels = {}  # anchor_key → list of (rel_class_id, head_span, tail_span)
            for rel in relations:
                head = rel.get('head', {})
                tail = rel.get('tail', {})
                if isinstance(head, dict):
                    h_start, h_end = head.get('start', -1), head.get('end', -1)
                else:
                    continue
                if isinstance(tail, dict):
                    t_start, t_end = tail.get('start', -1), tail.get('end', -1)
                else:
                    continue

                rel_type = rel.get('relation', '')
                if rel_type not in rel_to_id:
                    continue
                if h_start < 0 or h_end < 0 or t_start < 0 or t_end < 0:
                    continue

                # Each unique (head_span, tail_span) pair is an anchor
                anchor_key = (h_start, h_end, t_start, t_end)
                if anchor_key not in anchor_rels:
                    anchor_rels[anchor_key] = []
                anchor_rels[anchor_key].append(rel_to_id[rel_type])

            if anchor_rels:
                has_any = True
                max_anchors = max(max_anchors, len(anchor_rels))
            max_rel_classes = max(max_rel_classes, len(rel_to_id))
            group_relations.append(anchor_rels)

        if not has_any or max_anchors == 0 or max_rel_classes == 0:
            return None

        # Allocate label tensor: (total_groups, max_anchors, max_rel_classes, max_seq_len, 2, 3)
        open_rel_labels = torch.zeros(
            total_groups, max_anchors, max_rel_classes, max_seq_len, 2, 3,
            dtype=torch.float,
        )
        open_rel_mask = torch.zeros(total_groups, dtype=torch.bool)
        open_rel_batch_idx = torch.zeros(total_groups, dtype=torch.long)
        open_rel_count = torch.zeros(total_groups, dtype=torch.long)

        # Second pass: fill labels
        for flat_idx, batch_idx, group_idx, relex_item in classes_mapping.flat_open_relex_iter():
            open_rel_batch_idx[flat_idx] = batch_idx
            if flat_idx >= len(group_relations):
                continue

            anchor_rels = group_relations[flat_idx]
            if not anchor_rels:
                continue

            open_rel_mask[flat_idx] = True
            open_rel_count[flat_idx] = len(anchor_rels)

            for anchor_idx, (anchor_key, rel_class_ids) in enumerate(anchor_rels.items()):
                if anchor_idx >= max_anchors:
                    break
                h_start, h_end, t_start, t_end = anchor_key

                for rel_class_id in rel_class_ids:
                    if rel_class_id >= max_rel_classes:
                        continue

                    # Head span: dim=-2 index 0
                    if h_start < max_seq_len and h_end < max_seq_len:
                        open_rel_labels[flat_idx, anchor_idx, rel_class_id, h_start, 0, 0] = 1.0
                        open_rel_labels[flat_idx, anchor_idx, rel_class_id, h_end, 0, 1] = 1.0
                        open_rel_labels[flat_idx, anchor_idx, rel_class_id, h_start:h_end + 1, 0, 2] = 1.0

                    # Tail span: dim=-2 index 1
                    if t_start < max_seq_len and t_end < max_seq_len:
                        open_rel_labels[flat_idx, anchor_idx, rel_class_id, t_start, 1, 0] = 1.0
                        open_rel_labels[flat_idx, anchor_idx, rel_class_id, t_end, 1, 1] = 1.0
                        open_rel_labels[flat_idx, anchor_idx, rel_class_id, t_start:t_end + 1, 1, 2] = 1.0

        return {
            "open_rel_labels": open_rel_labels,
            "open_rel_mask": open_rel_mask,
            "open_rel_batch_idx": open_rel_batch_idx,
            "open_rel_count": open_rel_count,
        }

    def prepare_label_encoder_inputs(self, classes_mapping, labels_tokenizer):
        if labels_tokenizer is None:
            return None

        all_label_strings = []
        group_sizes = []
        has_any = False

        total_groups = classes_mapping.total_open_relex_groups()
        if total_groups == 0:
            return None

        for _, _, _, relex_item in classes_mapping.flat_open_relex_iter():
            labels = list(relex_item.rel_class_to_id.class_to_id.keys())
            if labels:
                has_any = True
            group_sizes.append(len(labels))
            all_label_strings.extend(labels)

        if not has_any or not all_label_strings:
            return None

        tokenized = labels_tokenizer(
            all_label_strings, return_tensors="pt", truncation=True,
            padding="longest", add_special_tokens=True,
        )
        return {
            "open_rel_labels_input_ids": tokenized["input_ids"],
            "open_rel_labels_attention_mask": tokenized["attention_mask"],
            "open_rel_labels_group_size": torch.LongTensor(group_sizes),
        }
