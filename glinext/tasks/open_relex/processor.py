"""Open relex task processor."""

from typing import Dict, List, Optional

import torch

from ..span_processor import SpanProcessor
from ...processing.mappings import (
    BaseClassMapping, OpenRelexItemMapping, OpenRelexClassMapping, BatchClassesMapping,
)


class OpenRelexProcessor(SpanProcessor):
    """Processor for anchor-based open relation extraction.

    Builds its own extraction groups with [P] and [REL] tokens.
    Input data uses the 'open_relex' key with text-based head/tail mentions.
    """

    def __init__(self, config, tokenizer=None, words_splitter=None, **kwargs):
        super().__init__(config, tokenizer, words_splitter,
                         parent_token=getattr(config, 'open_rel_parent_token', None), **kwargs)
        self.rel_token = config.rel_token

    def get_classes_mapping(self, batch_list, shuffle_labels=False, **kwargs):
        open_relex_mapping = []
        for item in batch_list:
            open_relex_data = item.get('open_relex', [])
            item_mappings = []
            for group in open_relex_data:
                # Use all_labels when available (inference), else extract from annotations
                if 'all_labels' in group:
                    rel_types = list(group['all_labels'])
                else:
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
        open_relex_data = item.get('open_relex', [])
        if not open_relex_data:
            return

        text = item.get('text', '')
        tokens_with_spans, _ = self._tokenize_text(item)
        if tokens_with_spans is None:
            return

        for group in open_relex_data:
            for rel in group.get('relations', []):
                for role in ('head', 'tail'):
                    value = rel.get(role)
                    if value is None:
                        continue
                    if isinstance(value, str):
                        start, end = self._resolve_text_span(text, tokens_with_spans, value)
                        rel[role] = {'text': value, 'start': start, 'end': end}
                    elif isinstance(value, dict) and 'text' in value:
                        if 'start' not in value or not isinstance(value['start'], int):
                            start, end = self._resolve_text_span(
                                text, tokens_with_spans, str(value['text']),
                            )
                            value['start'] = start
                            value['end'] = end

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

            for anchor_idx, (anchor_key, rel_class_ids) in enumerate(sorted(anchor_rels.items())):
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

    def create_span_labels(self, batch_list, classes_mapping, max_seq_len=0, **kwargs):
        """Create span-level labels for open relex when represent_spans is enabled.

        Returns dict with:
            open_rel_span_idx: (total_groups, max_spans, 2)
            open_rel_span_labels: (total_groups, max_spans, max_anchors, max_rel_classes, 2)
                2 = [head_match, tail_match] per span per (anchor, rel_class) pair
            open_rel_span_mask: (total_groups, max_spans)
            open_rel_span_batch_idx: (total_groups,)
        """
        cfg = getattr(self.config, 'open_relex_config', None)
        if cfg is None or not getattr(cfg, 'represent_spans', False):
            return None

        total_groups = classes_mapping.total_open_relex_groups()
        if total_groups == 0:
            return None

        neg_ratio = getattr(cfg, 'neg_spans_ratio', 1.0)
        max_anchors = 0
        max_rel_classes = 0
        has_any = False

        # Collect per-group: anchor assignments and all unique spans
        all_group_data = []  # list of (anchor_rels_dict, all_spans, positive_spans_set)
        batch_indices = []

        for flat_idx, batch_idx, group_idx, relex_item in classes_mapping.flat_open_relex_iter():
            batch_indices.append(batch_idx)
            open_relex_data = batch_list[batch_idx].get('open_relex', [])
            if group_idx >= len(open_relex_data):
                all_group_data.append(({}, [], set()))
                continue

            group = open_relex_data[group_idx]
            relations = group.get('relations', [])
            rel_to_id = relex_item.rel_class_to_id.class_to_id
            max_rel_classes = max(max_rel_classes, len(rel_to_id))

            # Collect all head/tail spans and group by anchor
            keyed_anchor_rels = {}  # anchor_key → list of (rel_class_id, h_start, h_end, t_start, t_end)
            positive_spans = set()

            for rel in relations:
                head = rel.get('head', {})
                tail = rel.get('tail', {})
                if not isinstance(head, dict) or not isinstance(tail, dict):
                    continue
                h_start, h_end = head.get('start', -1), head.get('end', -1)
                t_start, t_end = tail.get('start', -1), tail.get('end', -1)
                rel_type = rel.get('relation', '')
                if rel_type not in rel_to_id or h_start < 0 or t_start < 0:
                    continue
                if h_start >= max_seq_len or h_end >= max_seq_len:
                    continue
                if t_start >= max_seq_len or t_end >= max_seq_len:
                    continue

                anchor_key = (h_start, h_end, t_start, t_end)
                if anchor_key not in keyed_anchor_rels:
                    keyed_anchor_rels[anchor_key] = []
                keyed_anchor_rels[anchor_key].append((rel_to_id[rel_type], h_start, h_end, t_start, t_end))
                positive_spans.add((h_start, h_end))
                positive_spans.add((t_start, t_end))
                has_any = True

            # Sort anchors by position (head start, head end, tail start, tail end)
            anchor_rels = {}
            for anchor_idx, (_, rels) in enumerate(sorted(keyed_anchor_rels.items())):
                anchor_rels[anchor_idx] = rels

            max_anchors = max(max_anchors, len(anchor_rels))

            # Generate negative spans
            all_spans = list(positive_spans)
            neg_count = int(len(all_spans) * neg_ratio)
            if neg_count > 0 and max_seq_len > 0:
                negatives = self._generate_negative_spans(positive_spans, max_seq_len, neg_count)
                all_spans.extend(negatives)

            all_group_data.append((anchor_rels, all_spans, positive_spans))

        if not has_any or max_anchors == 0 or max_rel_classes == 0:
            return None

        max_spans = max((len(d[1]) for d in all_group_data), default=0)
        if max_spans == 0:
            return None

        span_idx = torch.zeros(total_groups, max_spans, 2, dtype=torch.long)
        # Labels: for each span, for each (anchor, rel_class): does span match as head (0) or tail (1)?
        span_labels = torch.zeros(
            total_groups, max_spans, max_anchors, max_rel_classes, 2,
            dtype=torch.float,
        )
        span_mask = torch.zeros(total_groups, max_spans, dtype=torch.bool)
        span_batch_idx = torch.tensor(batch_indices, dtype=torch.long)

        for g, (anchor_rels, all_spans, _) in enumerate(all_group_data):
            # Build span→index mapping
            span_to_idx = {}
            for s, (st, ed) in enumerate(all_spans):
                span_idx[g, s, 0] = st
                span_idx[g, s, 1] = ed
                span_mask[g, s] = True
                span_to_idx[(st, ed)] = s

            # Fill labels
            for a_idx, rels in anchor_rels.items():
                if a_idx >= max_anchors:
                    break
                for rel_class_id, h_start, h_end, t_start, t_end in rels:
                    if rel_class_id >= max_rel_classes:
                        continue
                    h_span_idx = span_to_idx.get((h_start, h_end))
                    t_span_idx = span_to_idx.get((t_start, t_end))
                    if h_span_idx is not None:
                        span_labels[g, h_span_idx, a_idx, rel_class_id, 0] = 1.0
                    if t_span_idx is not None:
                        span_labels[g, t_span_idx, a_idx, rel_class_id, 1] = 1.0

        return {
            "open_rel_span_idx": span_idx,
            "open_rel_span_labels": span_labels,
            "open_rel_span_mask": span_mask,
            "open_rel_span_batch_idx": span_batch_idx,
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
