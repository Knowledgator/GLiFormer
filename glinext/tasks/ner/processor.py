"""NER task processor."""

import random
import warnings

import torch

from ...processing.label_augmentation import AugmentableLabelGroup
from ...processing.mappings import (
    BaseClassMapping,
    ExtractionClassMapping,
    ExtractionItemMapping,
)
from ..span_processor import SpanProcessor


class NERProcessor(SpanProcessor):
    """Processor for NER task: class mappings, prompts, labels, span resolution."""

    def __init__(self, config, tokenizer=None, words_splitter=None, **kwargs):
        super().__init__(config, tokenizer, words_splitter,
                         parent_token=getattr(config, 'ner_parent_token', None), **kwargs)
        self.ent_token = config.ent_token
        self.rel_token = getattr(config, 'rel_token', None)

    @staticmethod
    def _entity_label(entity):
        if isinstance(entity, dict):
            return entity.get('label', entity.get('type', entity.get('entity_type')))
        if isinstance(entity, (list, tuple)) and entity:
            return entity[-1]
        return None

    @staticmethod
    def _build_class_to_id(labels, negatives, sample_neg, shuffle_labels):
        labels = [label for label in dict.fromkeys(labels) if label is not None]
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
                    ner_labels = list(dict.fromkeys(
                        label
                        for ent in example.get('ner', [])
                        if (label := self._entity_label(ent)) is not None
                    ))
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
                    rel_labels = list(dict.fromkeys(
                        rel[1] for rel in example.get('relations', [])
                    ))
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

    def get_augmentable_label_groups(self, batch_list, classes_mapping):
        groups = []
        for batch_idx, item_mapping in enumerate(
            classes_mapping.extraction_mapping
        ):
            examples = batch_list[batch_idx].get("extraction", [])
            for group_idx, extraction_mapping in enumerate(item_mapping.items):
                mapping = extraction_mapping.ner_class_to_id
                if mapping is None or not mapping.class_to_id:
                    continue
                example = examples[group_idx] if group_idx < len(examples) else {}
                positives = list(dict.fromkeys(
                    label
                    for entity in example.get("ner", [])
                    if (label := self._entity_label(entity)) is not None
                ))
                groups.append(AugmentableLabelGroup(
                    task="ner",
                    batch_idx=batch_idx,
                    group_idx=group_idx,
                    mapping=mapping,
                    positive_labels=positives,
                    parent_name=mapping.name,
                ))
        return groups

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

    def contribute_inference_input(self, item, entities=None, **kwargs):
        entity_groups = self._normalize_label_groups(entities)
        if not entity_groups:
            return

        extraction = item.setdefault('extraction', [])
        by_name = {
            group.get('name'): group
            for group in extraction
        }
        for parent_name, ent_labels in entity_groups.items():
            entry = by_name.get(parent_name)
            if entry is None:
                entry = {
                    "name": parent_name,
                    "ner": [],
                }
                extraction.append(entry)
                by_name[parent_name] = entry
            entry["all_labels"] = ent_labels

    def empty_inference_result(self, num_texts: int, entities=None, **kwargs):
        if entities is None:
            return None
        return {"ner": [[] for _ in range(num_texts)]}

    def resolve_spans(self, item):
        if item.get('_glinext_extraction_spans_resolved'):
            return
        if not item.get('extraction'):
            return
        text = item.get('text', '')
        has_tokenized_text = bool(item.get('tokenized_text'))
        if has_tokenized_text:
            tokens = list(item.get('tokenized_text') or [])
            tokens_with_spans = self._align_tokens_to_text(tokens, text) if text else None
        else:
            tokens_with_spans, tokens = self._tokenize_text(item)
        if not tokens:
            return

        for ext_example in item.get('extraction', []):
            ner = ext_example.get('ner', [])
            if not ner:
                continue
            # Compact integer triplets are ambiguous when ``text`` is
            # present: current datasets use character offsets, while older
            # datasets used inclusive token indices.  Preserve the legacy
            # interpretation only when every entity is a valid token span and
            # at least one span cannot be a character-boundary span.  Fully
            # character-aligned compact offsets continue down the current
            # character-resolution path.
            legacy_token_offsets = (
                not has_tokenized_text
                and self._uses_legacy_token_offsets(
                    ner, tokens_with_spans, num_tokens=len(tokens)
                )
            )
            # Resolve per-entity so we can track which originals survived.
            # Relation head_id/tail_id reference positions in the input ner
            # list — if resolution drops an entity, the remaining indices
            # shift and labels get misaligned.
            resolved = []
            old_to_new = {}
            for orig_idx, ent in enumerate(ner):
                single = self._resolve_ner_span(
                    text,
                    tokens_with_spans,
                    ent,
                    tokenized_offsets=has_tokenized_text or legacy_token_offsets,
                    num_tokens=len(tokens),
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
        item['_glinext_extraction_spans_resolved'] = True

    @staticmethod
    def _uses_legacy_token_offsets(ner, tokens_with_spans, num_tokens: int) -> bool:
        """Detect the old compact token-index representation conservatively.

        Numeric list spans that align cleanly to character boundaries retain
        the documented character-offset semantics.  If all spans fit inside
        the token sequence but any one does not align to text boundaries, the
        group can only be interpreted consistently as legacy token indices.
        Dict annotations remain unambiguous character-offset inputs.
        """
        if not tokens_with_spans or num_tokens <= 0:
            return False

        spans = []
        for value in ner:
            if not (
                isinstance(value, (list, tuple))
                and len(value) >= 3
                and isinstance(value[0], int)
                and isinstance(value[1], int)
            ):
                return False
            start, end = value[0], value[1]
            if start < 0 or end < start or start >= num_tokens or end >= num_tokens:
                return False
            spans.append((start, end))

        char_starts = {start for _, start, _ in tokens_with_spans}
        char_ends = {end for _, _, end in tokens_with_spans}
        return any(
            start not in char_starts
            or (end not in char_ends and end + 1 not in char_ends)
            for start, end in spans
        )

    @classmethod
    def _resolve_ner_span(
        cls,
        text,
        tokens_with_spans,
        value,
        tokenized_offsets: bool,
        num_tokens: int,
    ):
        if not tokenized_offsets:
            if tokens_with_spans is None:
                return []
            return cls._resolve_labeled_span(text, tokens_with_spans, value)

        label = cls._entity_label(value)
        if isinstance(value, dict):
            if "start" in value and "end" in value and label is not None:
                return cls._token_span(value.get("start"), value.get("end"), label, num_tokens)
            mention_text = value.get("text")
        elif isinstance(value, (list, tuple)):
            if len(value) >= 4 and isinstance(value[0], str) and isinstance(value[1], int) and isinstance(value[2], int):
                label = value[-1]
                return cls._token_span(value[1], value[2], label, num_tokens)
            if len(value) >= 3 and isinstance(value[0], int) and isinstance(value[1], int):
                label = value[-1]
                return cls._token_span(value[0], value[1], label, num_tokens)
            mention_text = value[0] if value else None
        else:
            mention_text = value

        if mention_text is None or label is None or tokens_with_spans is None:
            return []
        return cls._resolve_labeled_span(text, tokens_with_spans, mention_text, label=label)

    @staticmethod
    def _token_span(start, end, label, num_tokens: int):
        try:
            start = int(start)
            end = int(end)
        except (TypeError, ValueError):
            return []
        if label is None or start < 0 or end < start or start >= num_tokens or end >= num_tokens:
            return []
        return [[start, end, label]]

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
        for item in batch_list:
            self.resolve_spans(item)

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

    def preprocess_example(self, item, extraction_mapping):
        text = item.get("text", "")
        if "tokenized_text" in item:
            tokens = list(item["tokenized_text"])
        else:
            raw_tokens = list(self.words_splitter(text))
            if raw_tokens and isinstance(raw_tokens[0], (list, tuple)):
                tokens = [tok[0] for tok in raw_tokens]
            else:
                tokens = raw_tokens
        if len(tokens) == 0:
            tokens = ["[PAD]"]
        max_len = self.config.max_len
        if len(tokens) > max_len:
            warnings.warn(
                f"Sentence of length {len(tokens)} has been truncated to {max_len}",
                stacklevel=2
            )
            tokens = tokens[:max_len]

        num_tokens = len(tokens)
        item_span_idx = []
        item_span_label = []

        for i, ext_example in enumerate(item.get('extraction', [])):
            ner = ext_example.get('ner', [])
            if i >= len(extraction_mapping.items):
                item_span_idx.append(None)
                item_span_label.append(None)
                continue
            classes_to_id = extraction_mapping.items[i].ner_class_to_id.class_to_id
            span_idx, span_label = self.prepare_span_idx(ner, classes_to_id, num_tokens)
            item_span_idx.append(span_idx)
            item_span_label.append(span_label)

        return {
            "tokens": tokens,
            "seq_length": len(tokens),
            "entities": item.get('extraction', []),
            "span_idx": item_span_idx,
            "span_label": item_span_label,
        }

    def add_span_batch_fields(self, batch_dict, classes_mapping):
        total_groups = classes_mapping.total_extraction_groups()
        if total_groups == 0:
            return batch_dict

        span_idx_nested = batch_dict.get("span_idx")
        span_label_nested = batch_dict.get("span_label")

        if span_idx_nested is None:
            return batch_dict

        has_spans = any(
            si is not None and si.numel() > 0
            for item_spans in span_idx_nested
            for si in item_spans
        )
        if not has_spans:
            return batch_dict

        flat_span_idx = []
        flat_span_label = []
        max_spans = 0

        for _, batch_idx, group_idx, _ in classes_mapping.flat_extraction_iter():
            if batch_idx < len(span_idx_nested) and group_idx < len(span_idx_nested[batch_idx]):
                si = span_idx_nested[batch_idx][group_idx]
                sl = span_label_nested[batch_idx][group_idx]
                if si is not None and si.numel() > 0:
                    max_spans = max(max_spans, si.size(0))
                    flat_span_idx.append(si)
                    flat_span_label.append(sl)
                    continue
            flat_span_idx.append(torch.zeros(0, 2, dtype=torch.long))
            flat_span_label.append(torch.zeros(0, dtype=torch.long))

        if max_spans == 0:
            return batch_dict

        span_idx = torch.zeros(total_groups, max_spans, 2, dtype=torch.long)
        span_label = torch.full((total_groups, max_spans), -1, dtype=torch.long)
        span_mask = torch.zeros(total_groups, max_spans, dtype=torch.bool)

        for idx in range(total_groups):
            si = flat_span_idx[idx]
            sl = flat_span_label[idx]
            if si.numel() > 0:
                count = si.size(0)
                span_idx[idx, :count] = si
                span_label[idx, :count] = sl
                span_mask[idx, :count] = True

        batch_dict["span_idx"] = span_idx
        batch_dict["span_label"] = span_label
        batch_dict["span_mask"] = span_mask

        return batch_dict

    def create_span_labels(self, batch, classes_mapping=None):
        if "span_label" not in batch or "span_mask" not in batch:
            return None

        span_label = batch["span_label"]
        span_mask = batch["span_mask"]
        classes_mapping = classes_mapping or batch["classes_mapping"]

        max_num_classes = max(
            len(item.ner_class_to_id.class_to_id)
            for em in classes_mapping.extraction_mapping
            for item in em.items
        ) if any(em.items for em in classes_mapping.extraction_mapping) else 0

        if max_num_classes == 0:
            return None

        total_groups, max_spans = span_label.shape
        labels_one_hot = torch.zeros(total_groups, max_spans, max_num_classes, dtype=torch.float)

        valid = span_mask & (span_label > 0)
        class_indices = (span_label - 1).clamp(min=0)

        if valid.any():
            flat_indices = valid.nonzero(as_tuple=False)
            row = flat_indices[:, 0]
            col = flat_indices[:, 1]
            cls = class_indices[row, col]
            in_range = cls < max_num_classes
            labels_one_hot[row[in_range], col[in_range], cls[in_range]] = 1.0

        return {
            "span_labels": labels_one_hot,
            "span_mask": span_mask,
            "span_idx": batch["span_idx"],
        }

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
