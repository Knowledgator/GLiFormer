"""Open relex task processor."""

import random

import torch

from ...processing.label_augmentation import AugmentableLabelGroup
from ...processing.mappings import (
    BaseClassMapping,
    OpenRelexClassMapping,
    OpenRelexItemMapping,
)
from ..ner.processor import NERProcessor


class OpenRelexProcessor(NERProcessor):
    """Processor for anchor-based open relation extraction.

    Builds its own extraction groups with [SCHEMA] and [RELATION] tokens.
    Training accepts the canonical ``extraction`` representation used by the
    relation datasets as well as the dedicated ``open_relex`` endpoint format.
    """

    data_key = "open_relex"
    extraction_data_key = "extraction"

    def __init__(self, config, tokenizer=None, words_splitter=None, **kwargs):
        super().__init__(config, tokenizer, words_splitter, **kwargs)
        self.parent_token = (
            getattr(config, "open_rel_parent_token", None)
            or config.parent_token
        )
        self.rel_token = config.rel_token

    @staticmethod
    def _is_indexed_relation(relation):
        return (
            isinstance(relation, (list, tuple))
            and len(relation) >= 3
            and isinstance(relation[0], int)
            and isinstance(relation[2], int)
        )

    @classmethod
    def _is_extraction_group(cls, group):
        if not isinstance(group, dict) or "ner" not in group:
            return False
        return "all_rel_labels" in group or any(
            cls._is_indexed_relation(relation)
            for relation in group.get("relations", [])
        )

    @classmethod
    def _extraction_groups(cls, item):
        return [
            group
            for group in item.get(cls.extraction_data_key, []) or []
            if cls._is_extraction_group(group)
        ]

    @classmethod
    def _groups(cls, item):
        explicit_groups = item.get(cls.data_key) or []
        return explicit_groups if explicit_groups else cls._extraction_groups(item)

    @staticmethod
    def _relation_name(relation):
        if isinstance(relation, (list, tuple)):
            return relation[1] if len(relation) >= 2 else ""
        if not isinstance(relation, dict):
            return ""
        return relation.get(
            "relation",
            relation.get("label", relation.get("type", "")),
        )

    @classmethod
    def _relation_types(cls, group):
        if cls._is_extraction_group(group) and "all_rel_labels" in group:
            candidates = group["all_rel_labels"]
        elif "all_labels" in group:
            candidates = group["all_labels"]
        else:
            candidates = (
                cls._relation_name(relation)
                for relation in group.get("relations", [])
            )
        return list(dict.fromkeys(label for label in candidates if label))

    @classmethod
    def _mapped_groups(cls, item):
        """Return groups that occupy a flattened model/prompt slot."""

        return [
            group for group in cls._groups(item) if cls._relation_types(group)
        ]

    def has_training_annotations(self, item):
        """Return whether an item provides a usable relation label space."""

        return any(self._relation_types(group) for group in self._groups(item))

    def get_classes_mapping(self, batch_list, shuffle_labels=False, **kwargs):
        open_relex_mapping = []
        for item in batch_list:
            item_mappings = []
            for group in self._mapped_groups(item):
                rel_types = self._relation_types(group)
                if shuffle_labels:
                    random.shuffle(rel_types)
                rel_class_to_id = {rt: idx for idx, rt in enumerate(rel_types)}
                item_mappings.append(OpenRelexItemMapping(
                    rel_class_to_id=BaseClassMapping(
                        class_to_id=rel_class_to_id,
                        name=group.get('name'),
                        description=group.get('description'),
                    ),
                    name=group.get('name'),
                ))
            open_relex_mapping.append(OpenRelexClassMapping(items=item_mappings))
        return open_relex_mapping

    def get_augmentable_label_groups(self, batch_list, classes_mapping):
        groups = []
        for batch_idx, item_mapping in enumerate(
            classes_mapping.open_relex_mapping
        ):
            source_groups = self._mapped_groups(batch_list[batch_idx])
            for group_idx, relex_mapping in enumerate(item_mapping.items):
                mapping = relex_mapping.rel_class_to_id
                if not mapping.class_to_id:
                    continue
                source_group = (
                    source_groups[group_idx]
                    if group_idx < len(source_groups)
                    else {}
                )
                positives = list(dict.fromkeys(
                    label
                    for relation in source_group.get("relations", [])
                    if (label := self._relation_name(relation))
                ))
                groups.append(AugmentableLabelGroup(
                    task="open_relex",
                    batch_idx=batch_idx,
                    group_idx=group_idx,
                    mapping=mapping,
                    positive_labels=positives,
                    parent_name=mapping.name,
                ))
        return groups

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
            if rel_map.description:
                prompt.append(rel_map.description)
            if not use_labels_encoder:
                for rel_name in rel_map.class_to_id:
                    prompt.append(f"{self.rel_token} {rel_name}")
            prompt.append(self.sep_token)
        return prompt

    def contribute_inference_input(self, item, relations=None, **kwargs):
        relation_groups = self._normalize_label_groups(relations)
        if not relation_groups:
            return

        item["open_relex"] = [
            {
                "name": parent_name,
                "relations": [],
                "all_labels": rel_labels,
            }
            for parent_name, rel_labels in relation_groups.items()
        ]

    def empty_inference_result(self, num_texts: int, relations=None, **kwargs):
        if relations is None:
            return None
        return {"open_relex": [[] for _ in range(num_texts)]}

    def _normalize_endpoint(self, text, tokens_with_spans, value):
        if isinstance(value, dict):
            endpoint_text = value.get('text')
            if endpoint_text is None and 'start' in value and 'end' in value:
                try:
                    endpoint_text = text[int(value['start']):int(value['end'])]
                except (TypeError, ValueError):
                    endpoint_text = ''
            spans = self._resolve_labeled_span(
                text, tokens_with_spans, value, label='entity',
            )
        else:
            endpoint_text = value
            spans = self._resolve_labeled_span(
                text, tokens_with_spans, {'text': str(value)}, label='entity',
            )

        if not spans:
            return {'text': str(endpoint_text), 'start': -1, 'end': -1}

        start, end, _ = spans[0]
        return {'text': str(endpoint_text), 'start': start, 'end': end}

    def resolve_spans(self, item):
        """Resolve head/tail text mentions to token indices."""
        if item.get('_glinext_open_relex_spans_resolved'):
            return

        open_relex_data = item.get(self.data_key, []) or []
        if not open_relex_data and self._extraction_groups(item):
            # Canonical extraction relations index the group's NER array.
            # Reuse its offset resolver so dropped/sorted entities also remap
            # relation indices correctly.
            super().resolve_spans(item)
            item['_glinext_open_relex_spans_resolved'] = True
            return
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
                    rel[role] = self._normalize_endpoint(text, tokens_with_spans, value)
        item['_glinext_open_relex_spans_resolved'] = True

    @classmethod
    def _resolved_span(cls, relation, role, group):
        if cls._is_indexed_relation(relation):
            entity_idx = relation[0 if role == "head" else 2]
            entities = group.get("ner", [])
            if entity_idx < 0 or entity_idx >= len(entities):
                return None
            entity = entities[entity_idx]
            if isinstance(entity, dict):
                start, end = entity.get("start", -1), entity.get("end", -1)
            elif isinstance(entity, (list, tuple)) and len(entity) >= 2:
                start, end = entity[0], entity[1]
            else:
                return None
        elif isinstance(relation, dict):
            endpoint = relation.get(role, {})
            if not isinstance(endpoint, dict):
                return None
            start, end = endpoint.get("start", -1), endpoint.get("end", -1)
        else:
            return None
        try:
            return int(start), int(end)
        except (TypeError, ValueError):
            return None

    def create_labels(self, batch_list, classes_mapping, max_seq_len=0, **kwargs):
        """Create label tensors for open relex.

        Output shape: (total_groups, max_anchors, max_rel_classes, max_seq_len, 2, 3)
            2 = [head, tail], 3 = [start, inside, end]

        Anchor assignment: relations are grouped by unique (head_text, tail_text) pairs
        within each group, with each unique pair becoming an anchor slot.
        """
        for item in batch_list:
            self.resolve_spans(item)

        total_groups = classes_mapping.total_open_relex_groups()
        if total_groups == 0:
            return None

        max_anchors = 0
        max_rel_classes = 0

        # First pass: determine dimensions
        group_relations = []
        for flat_idx, batch_idx, group_idx, relex_item in classes_mapping.flat_open_relex_iter():
            open_relex_data = self._mapped_groups(batch_list[batch_idx])
            if group_idx >= len(open_relex_data):
                group_relations.append({})
                continue

            group = open_relex_data[group_idx]
            relations = group.get('relations', [])
            rel_to_id = relex_item.rel_class_to_id.class_to_id

            # Group relations by anchor (unique entity pairs)
            anchor_rels = {}  # anchor_key → list of (rel_class_id, head_span, tail_span)
            for rel in relations:
                head_span = self._resolved_span(rel, "head", group)
                tail_span = self._resolved_span(rel, "tail", group)
                if head_span is None or tail_span is None:
                    continue
                h_start, h_end = head_span
                t_start, t_end = tail_span

                rel_type = self._relation_name(rel)
                if rel_type not in rel_to_id:
                    continue
                if not (
                    0 <= h_start <= h_end < max_seq_len
                    and 0 <= t_start <= t_end < max_seq_len
                ):
                    continue

                # Each unique (head_span, tail_span) pair is an anchor
                anchor_key = (h_start, h_end, t_start, t_end)
                if anchor_key not in anchor_rels:
                    anchor_rels[anchor_key] = []
                anchor_rels[anchor_key].append(rel_to_id[rel_type])

            max_anchors = max(max_anchors, len(anchor_rels))
            max_rel_classes = max(max_rel_classes, len(rel_to_id))
            group_relations.append(anchor_rels)

        if max_rel_classes == 0:
            return None
        # Keep a placeholder for all-negative groups. The head pads this axis
        # to its configured query capacity and supervises every unused slot.
        max_anchors = max(max_anchors, 1)

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
            open_rel_mask[flat_idx] = True
            if flat_idx >= len(group_relations):
                continue

            anchor_rels = group_relations[flat_idx]
            if not anchor_rels:
                continue

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

        for item in batch_list:
            self.resolve_spans(item)

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
            open_relex_data = self._mapped_groups(batch_list[batch_idx])
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
                head_span = self._resolved_span(rel, "head", group)
                tail_span = self._resolved_span(rel, "tail", group)
                if head_span is None or tail_span is None:
                    continue
                h_start, h_end = head_span
                t_start, t_end = tail_span
                rel_type = self._relation_name(rel)
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
