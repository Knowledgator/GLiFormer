import re
import random
import warnings
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from gliner.data_processing import BaseProcessor


@dataclass
class BaseClassMapping:
    class_to_id: dict
    name: Optional[str] = None
    description: Optional[str] = None

    def get_reverse_mapping(self) -> Dict[int, str]:
        return {v: k for k, v in self.class_to_id.items()}


@dataclass
class CatClassMapping:
    cat_class_to_id: List[BaseClassMapping]


@dataclass
class ExtractionItemMapping:
    """Mapping for a single extraction item containing NER and optional relation classes."""
    ner_class_to_id: BaseClassMapping
    rel_class_to_id: Optional[BaseClassMapping] = None


@dataclass
class ExtractionClassMapping:
    """Per-example extraction mappings: list of items each with NER + optional REL."""
    items: List[ExtractionItemMapping] = field(default_factory=list)


@dataclass
class StructuringItemMapping:
    """Mapping for a single structuring schema (e.g. 'person' with fields 'name', 'age')."""
    field_class_to_id: BaseClassMapping  # field names → ids
    name: Optional[str] = None
    description: Optional[str] = None


@dataclass
class StructuringClassMapping:
    """Per-example structuring mappings: list of schemas each with field mappings."""
    items: List[StructuringItemMapping] = field(default_factory=list)


@dataclass
class BatchClassesMapping:
    cat_mapping: List[CatClassMapping]
    extraction_mapping: List[ExtractionClassMapping]
    structuring_mapping: List[StructuringClassMapping] = field(default_factory=list)

    def get_item_mapping(self, index: int) -> Tuple[CatClassMapping, ExtractionClassMapping]:
        return self.cat_mapping[index], self.extraction_mapping[index]

    def total_cat_groups(self) -> int:
        """Total number of classification groups across the batch."""
        return sum(len(cm.cat_class_to_id) for cm in self.cat_mapping)

    def total_extraction_groups(self) -> int:
        """Total number of extraction groups across the batch."""
        return sum(len(em.items) for em in self.extraction_mapping)

    def total_structuring_groups(self) -> int:
        """Total number of structuring groups (schemas) across the batch."""
        return sum(len(sm.items) for sm in self.structuring_mapping)

    def flat_cat_iter(self):
        """Iterate (flat_idx, batch_idx, group_idx, mapping) over all cat groups."""
        flat_idx = 0
        for batch_idx, cm in enumerate(self.cat_mapping):
            for group_idx, mapping in enumerate(cm.cat_class_to_id):
                yield flat_idx, batch_idx, group_idx, mapping
                flat_idx += 1

    def flat_extraction_iter(self):
        """Iterate (flat_idx, batch_idx, group_idx, item_mapping) over all extraction groups."""
        flat_idx = 0
        for batch_idx, em in enumerate(self.extraction_mapping):
            for group_idx, item_mapping in enumerate(em.items):
                yield flat_idx, batch_idx, group_idx, item_mapping
                flat_idx += 1

    def flat_structuring_iter(self):
        """Iterate (flat_idx, batch_idx, group_idx, item_mapping) over all structuring groups."""
        flat_idx = 0
        for batch_idx, sm in enumerate(self.structuring_mapping):
            for group_idx, item_mapping in enumerate(sm.items):
                yield flat_idx, batch_idx, group_idx, item_mapping
                flat_idx += 1


class GLiNextProcessor(BaseProcessor):
    def __init__(self, config, tokenizer, words_splitter,
                 decoder_tokenizer: Optional[object] = None,
                 labels_tokenizer: Optional[object] = None):
        super().__init__(config, tokenizer, words_splitter)
        self.decoder_tokenizer = decoder_tokenizer
        self.labels_tokenizer = labels_tokenizer

        self.seq_token = config.seq_token
        self.cat_token = config.cat_token
        self.ent_token = config.ent_token
        self.sep_token = config.sep_token
        self.rel_token = config.rel_token
        self.parent_token = config.parent_token
        self.child_token = config.child_token

    @staticmethod
    def _build_class_to_id(labels: List[str],
                           negatives: Optional[List[str]],
                           sample_neg: int,
                           shuffle_labels: bool) -> dict:
        if negatives is not None:
            label_set = set(labels)
            labels.extend(
                label for label in negatives[:sample_neg + len(labels)]
                if label not in label_set
            )
            # trim to exact count of negatives requested
            # (the slice above over-fetches to account for overlap)
            if len(labels) > len(label_set) + sample_neg:
                labels = labels[:len(label_set) + sample_neg]

        if shuffle_labels:
            random.shuffle(labels)

        return {label: idx for idx, label in enumerate(labels)}

    def get_cat_classes_mapping(self, batch_list: List[Dict],
                                cat_negatives: Optional[List[str]] = None,
                                sample_neg=100,
                                shuffle_labels=False) -> List[CatClassMapping]:

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

    def get_extraction_classes_mapping(self, batch_list: List[Dict],
                                       ner_negatives: Optional[List[str]] = None,
                                       rel_negatives: Optional[List[str]] = None,
                                       sample_neg=100,
                                       shuffle_labels=False) -> List[ExtractionClassMapping]:

        extraction_mapping = []
        for item in batch_list:
            extraction_examples = item.get('extraction', [])

            item_mappings = []
            for example in extraction_examples:
                # NER classes
                ner_labels = list({ent[-1] for ent in example.get('ner', [])})
                ner_class_to_id = self._build_class_to_id(ner_labels, ner_negatives, sample_neg, shuffle_labels)

                name = example.get('name', None)
                description = example.get('description', None)

                ner_mapping = BaseClassMapping(
                    class_to_id=ner_class_to_id, name=name, description=description
                )

                # Relation classes (optional)
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
                    rel_class_to_id=rel_mapping
                ))

            extraction_mapping.append(ExtractionClassMapping(items=item_mappings))

        return extraction_mapping

    def get_structuring_classes_mapping(self, batch_list: List[Dict],
                                        shuffle_labels=False) -> List[StructuringClassMapping]:
        """Build structuring class mappings from structuring data.

        Structuring data format::

            "structuring": {
                "person": [
                    {"name": "John Smith", "age": "25"},
                    {"name": "Jane Doe", "age": "30"}
                ]
            }

        Each key in the dict is a schema name, each list element is an instance,
        and the dict keys within each instance are field names.

        Returns:
            Per-example StructuringClassMapping with field name → id mappings.
        """
        structuring_mapping = []
        for item in batch_list:
            structuring_data = item.get('structuring', {})

            item_mappings = []
            for schema_name, instances in structuring_data.items():
                # Collect all field names across instances
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
                        class_to_id=field_class_to_id,
                        name=schema_name,
                    ),
                    name=schema_name,
                ))

            structuring_mapping.append(StructuringClassMapping(items=item_mappings))

        return structuring_mapping

    def batch_generate_class_mappings(self, batch_list: List[Dict],
                                      cat_negatives: Optional[List[str]] = None,
                                      rel_negatives: Optional[List[str]] = None,
                                      ner_negatives: Optional[List[str]] = None,
                                      sample_neg=100,
                                      shuffle_labels=False) -> BatchClassesMapping:

        cat_mapping = self.get_cat_classes_mapping(batch_list, cat_negatives, sample_neg, shuffle_labels)
        extraction_mapping = self.get_extraction_classes_mapping(
            batch_list, ner_negatives, rel_negatives, sample_neg, shuffle_labels
        )
        structuring_mapping = self.get_structuring_classes_mapping(batch_list, shuffle_labels)

        return BatchClassesMapping(
            cat_mapping=cat_mapping,
            extraction_mapping=extraction_mapping,
            structuring_mapping=structuring_mapping,
        )

    def resolve_entity_spans(self, text: str, tokens_with_spans: List[Tuple],
                             ner: List) -> List[List]:
        """Resolve NER entities to token-level [start, end, label] format.

        Entities can be provided as:
            - [text, label] — positions are deduced via regex matching
            - [text, start, end, label] — character positions mapped to token indices

        Args:
            text: Original text string.
            tokens_with_spans: List of (token, char_start, char_end) from words_splitter.
            ner: List of entities in either format.

        Returns:
            List of [token_start, token_end, label] triples.
        """
        if not ner:
            return []

        # Build char-to-token mappings
        s2t = {s: idx for idx, (_, s, _) in enumerate(tokens_with_spans)}
        e2t = {e: idx for idx, (_, _, e) in enumerate(tokens_with_spans)}

        resolved = []
        for ent in ner:
            if len(ent) == 3 and isinstance(ent[0], int):
                # Already in [start_token, end_token, label] format
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

    def resolve_extraction_spans(self, item: Dict) -> Dict:
        """Resolve all entity spans in an item's extraction data.

        Tokenizes text once and resolves spans for all extraction items.

        Args:
            item: Dict with 'tokenized_text' or 'text', and 'extraction' list.

        Returns:
            Item with resolved NER spans and 'tokenized_text' set.
        """
        text = item.get('text', '')
        tokens_with_spans = list(self.words_splitter(text))
        tokens = [tok for tok, _, _ in tokens_with_spans]

        if 'tokenized_text' not in item:
            item['tokenized_text'] = tokens

        for ext_example in item.get('extraction', []):
            ner = ext_example.get('ner', [])
            if ner and not (len(ner[0]) == 3 and isinstance(ner[0][0], int)):
                ext_example['ner'] = self.resolve_entity_spans(
                    text, tokens_with_spans, ner
                )

        self.sort_extraction_data(item)
        return item

    def resolve_structuring_spans(self, item: Dict) -> Dict:
        """Resolve structuring field values to token-level spans.

        For each schema instance, resolves field value text to (start, end) token
        indices using the same logic as NER entity resolution.

        Modifies the item in-place: each field value becomes a dict with
        'text', 'start', 'end' keys (token-level indices).

        Args:
            item: Dict with 'text', 'tokenized_text', and 'structuring' data.

        Returns:
            The item with resolved structuring spans.
        """
        text = item.get('text', '')
        structuring = item.get('structuring', {})
        if not structuring or not text:
            return item

        tokens_with_spans = list(self.words_splitter(text))
        if 'tokenized_text' not in item:
            item['tokenized_text'] = [tok for tok, _, _ in tokens_with_spans]

        for schema_name, instances in structuring.items():
            for instance in instances:
                for field_name, value in list(instance.items()):
                    if isinstance(value, dict) and 'text' in value:
                        # Has text + optional char offsets; resolve if no token spans yet
                        if 'start' not in value or not isinstance(value['start'], int):
                            text_val = str(value['text'])
                            ner_like = [[text_val, field_name]]
                            resolved = self.resolve_entity_spans(
                                text, tokens_with_spans, ner_like
                            )
                            if resolved:
                                value['start'] = resolved[0][0]
                                value['end'] = resolved[0][1]
                            else:
                                value['start'] = -1
                                value['end'] = -1
                    elif isinstance(value, list):
                        # List of values — resolve each as a separate span
                        resolved_list = []
                        for v in value:
                            v_str = str(v)
                            ner_like = [[v_str, field_name]]
                            resolved = self.resolve_entity_spans(
                                text, tokens_with_spans, ner_like
                            )
                            if resolved:
                                resolved_list.append({
                                    'text': v_str,
                                    'start': resolved[0][0],
                                    'end': resolved[0][1],
                                })
                            else:
                                resolved_list.append({
                                    'text': v_str,
                                    'start': -1,
                                    'end': -1,
                                })
                        instance[field_name] = resolved_list
                    else:
                        # Scalar value (str, int, float, bool) — convert to str and resolve
                        text_val = str(value)
                        ner_like = [[text_val, field_name]]
                        resolved = self.resolve_entity_spans(
                            text, tokens_with_spans, ner_like
                        )
                        if resolved:
                            instance[field_name] = {
                                'text': text_val,
                                'start': resolved[0][0],
                                'end': resolved[0][1],
                            }
                        else:
                            instance[field_name] = {
                                'text': text_val,
                                'start': -1,
                                'end': -1,
                            }

        return item

    def sort_extraction_data(self, item: Dict) -> None:
        """Sort NER entities and relations within each extraction example.

        Entities are sorted by (start, end) position. Relations are sorted by
        (head_id, tail_id) to ensure consistent ordering across all downstream
        processing (label creation, decoder label generation, etc.).

        Args:
            item: Dict with 'extraction' list containing NER and relation data.
        """
        for ext_example in item.get('extraction', []):
            ner = ext_example.get('ner', [])
            if ner:
                ext_example['ner'] = sorted(ner, key=lambda x: (x[0], x[1]))

            relations = ext_example.get('relations', [])
            if relations:
                ext_example['relations'] = sorted(relations, key=lambda x: (x[0], x[2]))

    def prepare_inputs(
        self,
        texts: List[List[str]],
        classes_mapping: BatchClassesMapping,
        blank: Optional[str] = None,
        add_entities: Optional[bool] = True,
        **kwargs,
    ) -> Tuple[List[List[str]], List[int]]:
        """Prepare input texts with multi-task prompts.

        Builds prompts with classification and extraction (NER + optional relation)
        type tokens organized by parent groups.

        Prompt structure:
            [SEQ] [PARENT] name desc [CAT] label1 [CAT] label2 [SEP]
                  [PARENT] name desc [ENT] entity1 [ENT] entity2 [REL] rel1 [REL] rel2 [SEP]
                  [SEP]
                  text tokens...

        Args:
            texts: Sequences of token strings, one per example.
            classes_mapping: Multi-task class mappings for the batch.
            blank: Optional blank entity token for zero-shot scenarios.
            add_entities: Whether to add entity/label text strings to the prompt.

        Returns:
            Tuple of (input text sequences with prompts, prompt lengths).
        """
        input_texts: List[List[str]] = []
        prompt_lengths: List[int] = []

        for i, text in enumerate(texts):
            cat_mapping, extraction_mapping = classes_mapping.get_item_mapping(i)

            prompt: List[str] = [self.seq_token]

            # Classification parent groups
            for cat_map in cat_mapping.cat_class_to_id:
                prompt.append(self.parent_token)
                if cat_map.name:
                    prompt.append(cat_map.name)
                if cat_map.description:
                    prompt.append(cat_map.description)
                for cat in cat_map.class_to_id:
                    prompt.append(f"{self.cat_token} {cat}")
                prompt.append(self.sep_token)

            # Extraction parent groups (NER + optional relations together)
            for ext_item in extraction_mapping.items:
                prompt.append(self.parent_token)
                ner_map = ext_item.ner_class_to_id
                if ner_map.name:
                    prompt.append(ner_map.name)
                if ner_map.description:
                    prompt.append(ner_map.description)
                if ner_map.class_to_id is not None:
                    for ent in ner_map.class_to_id:
                        prompt.append(f"{self.ent_token} {ent}")
                else:
                    prompt.append(f"{self.ent_token} {blank or 'ENTITY'}")

                if ext_item.rel_class_to_id is not None:
                    for rel in ext_item.rel_class_to_id.class_to_id:
                        prompt.append(f"{self.rel_token} {rel}")
                prompt.append(self.sep_token)

            # Structuring parent groups (schema name + field names as [CHILD] tokens)
            if hasattr(classes_mapping, 'structuring_mapping') and i < len(classes_mapping.structuring_mapping):
                for struct_item in classes_mapping.structuring_mapping[i].items:
                    prompt.append(self.parent_token)
                    field_map = struct_item.field_class_to_id
                    if field_map.name:
                        prompt.append(field_map.name)
                    if field_map.description:
                        prompt.append(field_map.description)
                    for field_name in field_map.class_to_id:
                        prompt.append(f"{self.child_token} {field_name}")
                    prompt.append(self.sep_token)

            prompt.append(self.sep_token)
            prompt_lengths.append(len(prompt))
            input_texts.append(prompt + list(text))

        return input_texts, prompt_lengths

    def _prepare_label_encoder_inputs(
        self, classes_mapping: BatchClassesMapping, label_type: str
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Encode label strings using the labels tokenizer (bi-encoder style).

        Collects label strings across all extraction groups and tokenizes them
        with labels_tokenizer, producing per-group encoder input IDs.

        Args:
            classes_mapping: Multi-task class mappings.
            label_type: Either "ner" or "rel".

        Returns:
            Dictionary with tokenized label input IDs, attention mask, and
            per-group label counts. Returns None if labels_tokenizer is not
            available or no relevant labels exist.
        """
        if self.labels_tokenizer is None:
            return None

        if label_type == "ner":
            total_groups = classes_mapping.total_extraction_groups()
            if total_groups == 0:
                return None

        all_label_strings = []
        group_sizes = []
        has_any = False

        for _, _, _, ext_mapping in classes_mapping.flat_extraction_iter():
            if label_type == "ner":
                labels = list(ext_mapping.ner_class_to_id.class_to_id.keys())
                has_any = True
            else:
                if ext_mapping.rel_class_to_id is not None:
                    labels = list(ext_mapping.rel_class_to_id.class_to_id.keys())
                    has_any = True
                else:
                    labels = []
            group_sizes.append(len(labels))
            all_label_strings.extend(labels)

        if not has_any or not all_label_strings:
            return None

        tokenized = self.labels_tokenizer(
            all_label_strings, return_tensors="pt", truncation=True,
            padding="longest", add_special_tokens=True
        )

        return {
            f"{label_type}_labels_input_ids": tokenized["input_ids"],
            f"{label_type}_labels_attention_mask": tokenized["attention_mask"],
            f"{label_type}_labels_group_size": torch.LongTensor(group_sizes),
        }

    def prepare_labels_encoder_inputs(self, classes_mapping: BatchClassesMapping
                                      ) -> Optional[Dict[str, torch.Tensor]]:
        """Encode NER label strings using the labels tokenizer (bi-encoder style)."""
        return self._prepare_label_encoder_inputs(classes_mapping, "ner")

    def prepare_rel_labels_encoder_inputs(self, classes_mapping: BatchClassesMapping
                                           ) -> Optional[Dict[str, torch.Tensor]]:
        """Encode relation label strings using the labels tokenizer (bi-encoder style)."""
        return self._prepare_label_encoder_inputs(classes_mapping, "rel")

    def tokenize_inputs(self, texts, classes_mapping, **kwargs):
        """Tokenize input texts with multi-task prompts.

        Overrides BaseProcessor.tokenize_inputs to work with BatchClassesMapping
        instead of simple entity type lists.

        Args:
            texts: Sequences of token strings.
            classes_mapping: BatchClassesMapping with per-task class mappings.

        Returns:
            Dictionary with input_ids, attention_mask, and words_mask tensors.
            Optionally includes ner_labels_input_ids/attention_mask and
            rel_labels_input_ids/attention_mask if labels_tokenizer is available.
        """
        input_texts, prompt_lengths = self.prepare_inputs(texts, classes_mapping, **kwargs)

        tokenized_inputs = self.transformer_tokenizer(
            input_texts,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding="longest",
        )
        words_masks = self.prepare_word_mask(texts, tokenized_inputs, prompt_lengths)
        tokenized_inputs["words_mask"] = torch.tensor(words_masks)

        # Optional: encode NER label strings via labels_tokenizer
        ner_labels_enc = self.prepare_labels_encoder_inputs(classes_mapping)
        if ner_labels_enc is not None:
            tokenized_inputs["ner_labels_input_ids"] = ner_labels_enc["ner_labels_input_ids"]
            tokenized_inputs["ner_labels_attention_mask"] = ner_labels_enc["ner_labels_attention_mask"]
            tokenized_inputs["ner_labels_group_size"] = ner_labels_enc["ner_labels_group_size"]

        # Optional: encode relation label strings via labels_tokenizer
        rel_labels_enc = self.prepare_rel_labels_encoder_inputs(classes_mapping)
        if rel_labels_enc is not None:
            tokenized_inputs["rel_labels_input_ids"] = rel_labels_enc["rel_labels_input_ids"]
            tokenized_inputs["rel_labels_attention_mask"] = rel_labels_enc["rel_labels_attention_mask"]
            tokenized_inputs["rel_labels_group_size"] = rel_labels_enc["rel_labels_group_size"]

        return tokenized_inputs

    def _generate_negative_spans(self, positive_spans, num_tokens, num_negatives, max_width=None):
        """Generate random negative spans that don't overlap with positive spans."""
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
        """Prepare span indices and labels for span-based representation."""
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

    def preprocess_example(self, item, classes_mapping):
        """Preprocess a single NER example with token truncation and span preparation."""
        text = item.get("text", "")
        tokens = self.words_splitter(text)
        
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
            classes_to_id = classes_mapping.extraction_mapping[i].items[0].ner_class_to_id.class_to_id
            span_idx, span_label = self.prepare_span_idx(ner, classes_to_id, num_tokens)
            item_span_idx.append(span_idx)
            item_span_label.append(span_label)

        return {
            "tokens": tokens,
            "seq_length": len(tokens),
            "entities": ner,
            "span_idx": item_span_idx,
            "span_label": item_span_label,
        }

    def create_cat_labels(self, batch_list: List[Dict],
                          classes_mapping: BatchClassesMapping) -> Optional[torch.Tensor]:
        """Create classification labels.

        Args:
            batch_list: Raw batch items with 'classification' data.
            classes_mapping: Multi-task class mappings.

        Returns:
            Tuple of:
                - cat_labels: Tensor (total_cat_groups, C)
                - cat_batch_idx: LongTensor (total_cat_groups,) mapping each group
                  to its batch item index.
            Returns None if no classification data exists.
        """
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

        return cat_labels, cat_batch_idx

    def create_ner_labels(self, batch_list: List[Dict],
                          classes_mapping: BatchClassesMapping,
                          max_seq_len: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Create NER labels with begin/inside/end markers.

        Creates token-level labels for each extraction item, including a parent
        class (index 0) that captures all entities regardless of type. This parent
        class is useful for relation extraction which needs all entity positions.

        Args:
            batch_list: Raw batch items with 'extraction' data.
            classes_mapping: Multi-task class mappings.
            max_seq_len: Maximum sequence length in the batch.

        Returns:
            Tuple of:
                - ner_labels: Tensor (total_extraction_groups, L, C+1, 3)
                - ner_batch_idx: LongTensor (total_extraction_groups,) mapping each
                  group to its batch item index.
            Returns None if no extraction data exists.
        """
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

        # Shape: (total_groups, L, C+1, 3) - C+1 includes parent class at index 0
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

                # Parent class (index 0) - captures all entities
                ner_labels[flat_idx, start, 0, 0] = 1  # start
                ner_labels[flat_idx, end, 0, 1] = 1    # end
                ner_labels[flat_idx, start:end + 1, 0, 2] = 1  # inside

                # Child class labels (1-indexed in C+1 dimension)
                if label in mapping.class_to_id:
                    class_idx = mapping.class_to_id[label] + 1  # +1 for parent offset
                    ner_labels[flat_idx, start, class_idx, 0] = 1  # start
                    ner_labels[flat_idx, end, class_idx, 1] = 1    # end
                    ner_labels[flat_idx, start:end + 1, class_idx, 2] = 1  # inside

        return ner_labels, ner_batch_idx

    def create_rel_labels(self, batch_list: List[Dict],
                          classes_mapping: BatchClassesMapping) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Create relation extraction labels.

        Creates an entity-pair matrix for each extraction item that has relations,
        where entry (head, tail, rel_type) = 1 indicates a relation.

        Args:
            batch_list: Raw batch items with 'extraction' data containing relations.
            classes_mapping: Multi-task class mappings.

        Returns:
            Tuple of:
                - rel_labels: Tensor (total_extraction_groups, E, E, C)
                - rel_mask: Boolean tensor (total_extraction_groups,) indicating
                  which groups have relation data.
                - rel_batch_idx: LongTensor (total_extraction_groups,) mapping each
                  group to its batch item index.
            Returns None if no relation data exists.
        """
        total_groups = classes_mapping.total_extraction_groups()
        if total_groups == 0:
            return None

        # Find max entities and max relation classes
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

            for head_id, rel_type, tail_id in example.get('relations', []):
                if rel_type in mapping.class_to_id:
                    rel_class_idx = mapping.class_to_id[rel_type]
                    if head_id < max_entities and tail_id < max_entities:
                        rel_labels[flat_idx, head_id, tail_id, rel_class_idx] = 1.0

        return rel_labels, rel_mask, rel_batch_idx

    def create_count_labels(self, batch_list: List[Dict],
                            classes_mapping: BatchClassesMapping) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Create count labels for the CountHead — one per parent group.

        Each parent group gets a count target:
          - Classification groups: always 1
          - Extraction groups: always 1
          - Structuring groups: number of instances in the schema

        Args:
            batch_list: Raw batch items with task data.
            classes_mapping: Multi-task class mappings.

        Returns:
            Tuple of:
                - count_targets: FloatTensor (total_parents,) with per-group counts.
                - count_batch_idx: LongTensor (total_parents,) mapping group → batch item.
            Returns None if no groups exist.
        """
        total_cat = classes_mapping.total_cat_groups()
        total_ext = classes_mapping.total_extraction_groups()
        total_struct = classes_mapping.total_structuring_groups()
        total_parents = total_cat + total_ext + total_struct

        if total_parents == 0:
            return None

        count_targets = torch.zeros(total_parents, dtype=torch.float)
        count_batch_idx = torch.zeros(total_parents, dtype=torch.long)
        offset = 0

        # Classification: count is always 1
        for flat_idx, batch_idx, group_idx, _ in classes_mapping.flat_cat_iter():
            count_targets[offset] = 1.0
            count_batch_idx[offset] = batch_idx
            offset += 1

        # Extraction: count is always 1
        for flat_idx, batch_idx, group_idx, _ in classes_mapping.flat_extraction_iter():
            count_targets[offset] = 1.0
            count_batch_idx[offset] = batch_idx
            offset += 1

        # Structuring: count is number of instances in the schema
        for flat_idx, batch_idx, group_idx, struct_item in classes_mapping.flat_structuring_iter():
            structuring_data = batch_list[batch_idx].get('structuring', {})
            schema_name = struct_item.name
            instances = structuring_data.get(schema_name, [])
            count_targets[offset] = float(len(instances))
            count_batch_idx[offset] = batch_idx
            offset += 1

        return count_targets, count_batch_idx

    def create_embedding_labels(self, batch_list: List[Dict]) -> Optional[Dict[str, torch.Tensor]]:
        """Create embedding similarity labels.

        Each batch item may have an ``embedding`` field containing a list of
        ``(text1, text2, score)`` tuples.  For training, both texts in a pair
        must already be present in the batch as separate items (the processor
        duplicates items so that text1 and text2 occupy consecutive batch
        positions).

        This method assigns pair indices into the batch and collects target
        similarity scores.

        Returns:
            Dict with:
                - embedding_pair_idx: LongTensor (N, 2) — batch indices for each pair.
                - embedding_labels: FloatTensor (N,) — target similarity scores.
            Returns None if no embedding data exists.
        """

        # TODO: think how to handle training
        pair_indices = []
        scores = []
        offset = 0

        for item in batch_list:
            embedding_pairs = item.get('embedding', [])
            for pair in embedding_pairs:
                # Each pair occupies two consecutive batch positions
                # starting at the current offset
                pair_indices.append([offset, offset + 1])
                scores.append(float(pair[2]))
                offset += 2
            if not embedding_pairs:
                offset += 1

        if not pair_indices:
            return None

        return {
            'embedding_pair_idx': torch.tensor(pair_indices, dtype=torch.long),
            'embedding_labels': torch.tensor(scores, dtype=torch.float),
        }

    def create_structuring_labels(self, batch_list: List[Dict],
                                   classes_mapping: BatchClassesMapping,
                                   max_seq_len: int) -> Optional[Tuple[torch.Tensor, ...]]:
        """Create structuring labels for anchor-based span extraction.

        For each structuring schema, for each instance, for each field, marks
        start/inside/end positions in text for the field's value span.

        Data format::

            "structuring": {
                "person": [
                    {"name": {"text": "John", "start": 0, "end": 0},
                     "age": {"text": "25", "start": 3, "end": 3}},
                ]
            }

        Returns:
            Tuple of:
                - structuring_labels: FloatTensor (total_groups, X, L, C, 3)
                  where X=max instances, L=max_seq_len, C=max fields, 3=start/end/inside
                - structuring_mask: BoolTensor (total_groups,) indicating which
                  groups have structuring data.
                - structuring_batch_idx: LongTensor (total_groups,) mapping each
                  group to its batch item index.
                - structuring_count: LongTensor (total_groups,) number of instances
                  per schema.
            Returns None if no structuring data exists.
        """
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

                    # Get token span
                    if isinstance(value, dict):
                        st = value.get('start', -1)
                        ed = value.get('end', -1)
                    else:
                        continue

                    if st < 0 or ed < 0 or st >= max_seq_len or ed >= max_seq_len:
                        continue

                    # Mark start/end/inside (same as NER)
                    structuring_labels[flat_idx, inst_idx, st, field_id, 0] = 1.0  # start
                    structuring_labels[flat_idx, inst_idx, ed, field_id, 1] = 1.0  # end
                    structuring_labels[flat_idx, inst_idx, st:ed + 1, field_id, 2] = 1.0  # inside

        return structuring_labels, structuring_mask, structuring_batch_idx, structuring_count

    def create_structuring_span_labels(self, batch_list: List[Dict],
                                        classes_mapping: BatchClassesMapping,
                                        max_seq_len: int) -> Optional[Tuple[torch.Tensor, ...]]:
        """Create span-level structuring labels with shape (gB, S, X, C).

        For each structuring group, collects all field value spans across all
        instances, then builds a label tensor where each span is assigned to
        its (instance, field) slot.

        Args:
            batch_list: List of per-example dicts with 'structuring' data.
            classes_mapping: BatchClassesMapping for the batch.
            max_seq_len: Maximum sequence length (for negative span generation).

        Returns:
            Tuple of:
                - structuring_span_idx: LongTensor (gB, S, 2) span start/end pairs
                - structuring_span_labels: FloatTensor (gB, S, X, C) one-hot labels
                - structuring_span_mask: BoolTensor (gB, S) valid span mask
                - structuring_span_batch_idx: LongTensor (gB,) batch index per group
            Returns None if no structuring data exists.
        """
        total_groups = classes_mapping.total_structuring_groups()
        if total_groups == 0:
            return None

        max_instances = 0
        max_fields = 0
        has_any = False

        # First pass: collect spans per group, find dimensions
        all_group_spans = []  # list of list of (start, end, instance_idx, field_id)
        batch_indices = []

        for flat_idx, batch_idx, group_idx, struct_item in classes_mapping.flat_structuring_iter():
            structuring_data = batch_list[batch_idx].get('structuring', {})
            schema_name = struct_item.name
            field_to_id = struct_item.field_class_to_id.class_to_id
            max_fields = max(max_fields, len(field_to_id))
            batch_indices.append(batch_idx)

            group_spans = []
            positive_spans = set()

            if schema_name in structuring_data:
                instances = structuring_data[schema_name]
                max_instances = max(max_instances, len(instances))

                for inst_idx, instance in enumerate(instances):
                    for field_name, value in instance.items():
                        if field_name not in field_to_id:
                            continue
                        field_id = field_to_id[field_name]
                        if not isinstance(value, dict):
                            continue
                        st = value.get('start', -1)
                        ed = value.get('end', -1)
                        if 0 <= st < max_seq_len and 0 <= ed < max_seq_len:
                            group_spans.append((st, ed, inst_idx, field_id))
                            positive_spans.add((st, ed))
                            has_any = True

            # Add negative spans
            neg_ratio = getattr(self.config, 'neg_spans_ratio', 0)
            neg_count = int(len(group_spans) * neg_ratio)
            if neg_count > 0 and max_seq_len > 0:
                max_width = getattr(self.config, "max_width", 10)
                negatives = self._generate_negative_spans(
                    positive_spans, max_seq_len, neg_count, max_width
                )
                for st, ed in negatives:
                    group_spans.append((st, ed, -1, -1))  # sentinel for negative

            all_group_spans.append(group_spans)

        if not has_any or max_instances == 0 or max_fields == 0:
            return None

        max_spans = max((len(s) for s in all_group_spans), default=0)
        if max_spans == 0:
            return None

        span_idx = torch.zeros(total_groups, max_spans, 2, dtype=torch.long)
        span_labels = torch.zeros(total_groups, max_spans, max_instances, max_fields, dtype=torch.float)
        span_mask = torch.zeros(total_groups, max_spans, dtype=torch.bool)
        span_batch_idx = torch.tensor(batch_indices, dtype=torch.long)

        for g, group_spans in enumerate(all_group_spans):
            for s, (st, ed, inst_idx, field_id) in enumerate(group_spans):
                span_idx[g, s, 0] = st
                span_idx[g, s, 1] = ed
                span_mask[g, s] = True
                if inst_idx >= 0 and field_id >= 0:
                    span_labels[g, s, inst_idx, field_id] = 1.0

        return span_idx, span_labels, span_mask, span_batch_idx

    def prepare_decoder_labels(self, decoder_label_strings):
        """Tokenize decoder label strings using the decoder tokenizer.

        Args:
            decoder_label_strings: List of label strings to tokenize.

        Returns:
            Dictionary with input_ids, attention_mask, and labels tensors.
            Returns None if no decoder_tokenizer is available.
        """
        if self.decoder_tokenizer is None:
            return None

        if not decoder_label_strings:
            decoder_label_strings = ["other"]

        decoder_tokenized_input = self.decoder_tokenizer(
            decoder_label_strings, return_tensors="pt", truncation=True,
            padding="longest", add_special_tokens=True
        )
        decoder_input_ids = decoder_tokenized_input["input_ids"]
        decoder_attention_mask = decoder_tokenized_input["attention_mask"]
        decoder_labels = decoder_input_ids.clone()
        decoder_labels.masked_fill(~decoder_attention_mask.bool(), -100)
        decoder_tokenized_input["labels"] = decoder_labels
        return decoder_tokenized_input

    def _collect_decoder_items(self, batch_list: List[Dict],
                               classes_mapping: BatchClassesMapping,
                               label_type: str,
                               max_seq_len: int = 0) -> Optional[Dict]:
        """Collect and tokenize decoder label strings for NER or relation labels.

        Iterates over extraction groups, collects label strings and metadata,
        then tokenizes via the decoder tokenizer.

        Args:
            batch_list: Raw batch items with 'extraction' data.
            classes_mapping: Multi-task class mappings.
            label_type: Either "ner" or "rel".
            max_seq_len: Maximum sequence length (used for NER span filtering).

        Returns:
            Dictionary with tokenized labels and group/pair indices,
            or None if no decoder_tokenizer or no labels exist.
        """
        if self.decoder_tokenizer is None:
            return None

        decoder_label_strings = []
        decoder_group_idx = []
        decoder_pair_idx = []

        for flat_idx, batch_idx, group_idx, ext_mapping in classes_mapping.flat_extraction_iter():
            extraction_examples = batch_list[batch_idx].get('extraction', [])
            if group_idx >= len(extraction_examples):
                continue

            example = extraction_examples[group_idx]

            if label_type == "ner":
                mapping = ext_mapping.ner_class_to_id
                # NER data is already sorted by sort_extraction_data
                for ent in example.get('ner', []):
                    start, end, label = ent[0], ent[1], ent[-1]
                    if start >= max_seq_len or end >= max_seq_len:
                        continue
                    if label in mapping.class_to_id:
                        decoder_label_strings.append(label)
                        decoder_group_idx.append(flat_idx)
            else:
                if ext_mapping.rel_class_to_id is None:
                    continue
                mapping = ext_mapping.rel_class_to_id
                num_entities = len(example.get('ner', []))
                for head_id, rel_type, tail_id in example.get('relations', []):
                    if rel_type in mapping.class_to_id and head_id < num_entities and tail_id < num_entities:
                        decoder_label_strings.append(rel_type)
                        decoder_group_idx.append(flat_idx)
                        decoder_pair_idx.append([head_id, tail_id])

        if not decoder_label_strings:
            if label_type == "ner":
                # NER decoder needs a fallback group_idx even when empty
                decoder_tokenized = self.prepare_decoder_labels([])
                if decoder_tokenized is None:
                    return None
                decoder_tokenized["decoder_group_idx"] = torch.LongTensor([0])
                return decoder_tokenized
            return None

        decoder_tokenized = self.prepare_decoder_labels(decoder_label_strings)
        if decoder_tokenized is None:
            return None

        if label_type == "ner":
            decoder_tokenized["decoder_group_idx"] = torch.LongTensor(decoder_group_idx)
        else:
            decoder_tokenized["rel_decoder_group_idx"] = torch.LongTensor(decoder_group_idx)
            decoder_tokenized["rel_decoder_pair_idx"] = torch.LongTensor(decoder_pair_idx)

        return decoder_tokenized

    def create_decoder_labels(self, batch_list: List[Dict],
                              classes_mapping: BatchClassesMapping,
                              max_seq_len: int) -> Optional[Dict]:
        """Create decoder labels from entity type strings matching each span."""
        return self._collect_decoder_items(batch_list, classes_mapping, "ner", max_seq_len)

    def create_rel_decoder_labels(self, batch_list: List[Dict],
                                  classes_mapping: BatchClassesMapping) -> Optional[Dict]:
        """Create decoder labels for relation type strings with source/target entity pairs."""
        return self._collect_decoder_items(batch_list, classes_mapping, "rel")

    def create_span_labels(self, batch):
        """Create one-hot encoded span labels from padded flat tensors.

        Converts integer span labels into one-hot encoded vectors per extraction group.

        Args:
            batch: Batch dictionary containing:
                - span_label: Tensor (total_extraction_groups, max_spans)
                - span_mask: Tensor (total_extraction_groups, max_spans)
                - classes_mapping: BatchClassesMapping

        Returns:
            Tuple of:
                - labels_one_hot: Tensor (total_extraction_groups, max_spans, max_num_classes)
                - span_mask: Tensor (total_extraction_groups, max_spans)
            Returns None if span data is missing.
        """
        if "span_label" not in batch or "span_mask" not in batch:
            return None

        span_label = batch["span_label"]   # (total_groups, max_spans)
        span_mask = batch["span_mask"]      # (total_groups, max_spans)
        classes_mapping = batch["classes_mapping"]

        max_num_classes = max(
            len(item.ner_class_to_id.class_to_id)
            for em in classes_mapping.extraction_mapping
            for item in em.items
        ) if any(em.items for em in classes_mapping.extraction_mapping) else 0

        if max_num_classes == 0:
            return None

        total_groups, max_spans = span_label.shape
        labels_one_hot = torch.zeros(total_groups, max_spans, max_num_classes, dtype=torch.float)

        # Vectorized: mask valid spans with positive class ids
        valid = span_mask & (span_label > 0)
        class_indices = (span_label - 1).clamp(min=0)

        # Scatter into one-hot
        if valid.any():
            flat_indices = valid.nonzero(as_tuple=False)  # (N, 2)
            row = flat_indices[:, 0]
            col = flat_indices[:, 1]
            cls = class_indices[row, col]
            in_range = cls < max_num_classes
            labels_one_hot[row[in_range], col[in_range], cls[in_range]] = 1.0

        return labels_one_hot, span_mask

    def create_labels(self, batch):
        """Create token-level NER labels with begin/inside/end markers.

        Compatibility method for single-task NER. For multi-task use create_ner_labels.

        Args:
            batch: Batch dict with tokens, seq_length, entities, classes_to_id.

        Returns:
            Tensor of shape (batch_size, seq_len, num_classes, 3).
        """
        batch_size = len(batch["tokens"])
        seq_len = batch["seq_length"].max().item()
        num_classes = max(len(cid) for cid in batch["classes_to_id"])

        word_labels = torch.zeros(batch_size, seq_len, num_classes, 3, dtype=torch.float)

        for i, sentence_entities in enumerate(batch["entities"]):
            for st, ed, sp_label in sentence_entities:
                if sp_label not in batch["classes_to_id"][i]:
                    continue
                lbl = batch["classes_to_id"][i][sp_label]
                class_idx = lbl - 1

                if st >= seq_len or ed >= seq_len:
                    continue

                word_labels[i, st, class_idx, 0] = 1  # start
                word_labels[i, ed, class_idx, 1] = 1  # end
                word_labels[i, st:ed + 1, class_idx, 2] = 1  # inside

        return word_labels

    def create_batch_dict(self, batch, classes_mapping):
        """Create a batch dictionary with padded tensors for label creation and model input.

        Flattens the per-example, per-extraction-item span data into tensors
        indexed by (B * N_ext) to match the flat indexing used by NER/REL labels.

        Args:
            batch: Batch dict containing tokens, seq_length, classes_mapping,
                   classification, extraction, embedding, and optionally
                   span_idx/span_label from preprocess_example.
            classes_mapping: BatchClassesMapping for the batch.

        Returns:
            Batch dictionary with:
                - tokens, seq_length, classes_mapping, classification, extraction, embedding
                - span_idx: Tensor (total_extraction_groups, max_spans, 2)
                - span_label: Tensor (total_extraction_groups, max_spans)
                - span_mask: Tensor (total_extraction_groups, max_spans)
        """
        batch_size = len(batch["tokens"])
        total_groups = classes_mapping.total_extraction_groups()

        batch_dict = {
            "tokens": batch["tokens"],
            "seq_length": batch["seq_length"],
            "classes_mapping": classes_mapping,
            "classification": batch.get("classification", [[] for _ in range(batch_size)]),
            "extraction": batch.get("extraction", [[] for _ in range(batch_size)]),
            "embedding": batch.get("embedding", [[] for _ in range(batch_size)]),
            "structuring": batch.get("structuring", [{} for _ in range(batch_size)]),
        }

        if total_groups == 0:
            return batch_dict

        # Collect all span tensors from preprocess_example results into flat list
        # span_idx/span_label are List[List[Tensor]] (batch x extraction items)
        span_idx_nested = batch.get("span_idx")
        span_label_nested = batch.get("span_label")

        if span_idx_nested is None:
            return batch_dict

        has_spans = any(
            si is not None and si.numel() > 0
            for item_spans in span_idx_nested
            for si in item_spans
        )
        if not has_spans:
            return batch_dict

        # Flatten following the same order as flat_extraction_iter
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

        # Pad into (total_groups, max_spans, ...) tensors
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

    def collate_raw_batch(self, batch_list: List[Dict],
                          cat_negatives: Optional[List[str]] = None,
                          ner_negatives: Optional[List[str]] = None,
                          rel_negatives: Optional[List[str]] = None,
                          sample_neg: int = 100,
                          shuffle_labels: bool = False) -> Dict:
        """Collate a raw multi-task batch.

        Generates class mappings and assembles an intermediate batch dictionary
        containing all task data needed for tokenization and label creation.

        Args:
            batch_list: List of raw example dicts. Each dict should contain:
                - 'tokenized_text': List[str] - pre-tokenized text
                - 'classification': List[Dict] - classification tasks (optional)
                - 'extraction': List[Dict] - extraction tasks with NER and optional
                  relations. Each dict has 'name', 'ner' (list of entities), and
                  optionally 'relations' (list of (head_id, relation, tail_id) tuples).
                - 'embedding': List[Tuple] - embedding pairs (optional)
            cat_negatives: Optional negative labels for classification.
            ner_negatives: Optional negative labels for NER.
            rel_negatives: Optional negative labels for relations.
            sample_neg: Number of negative samples.
            shuffle_labels: Whether to shuffle label order.

        Returns:
            Multi-task batch dictionary.
        """
        # Resolve entity spans (text -> token indices) and sort extraction data
        for item in batch_list:
            if item.get('extraction') and item.get('text'):
                # resolve_extraction_spans also calls sort_extraction_data
                self.resolve_extraction_spans(item)
            elif item.get('extraction'):
                self.sort_extraction_data(item)
            # Resolve structuring field values to token spans
            if item.get('structuring') and item.get('text'):
                self.resolve_structuring_spans(item)

        classes_mapping = self.batch_generate_class_mappings(
            batch_list,
            cat_negatives=cat_negatives,
            ner_negatives=ner_negatives,
            rel_negatives=rel_negatives,
            sample_neg=sample_neg,
            shuffle_labels=shuffle_labels,
        )

        texts = [item['tokenized_text'] for item in batch_list]
        max_len = self.config.max_len
        truncated_texts = []
        for t in texts:
            if len(t) == 0:
                t = ["[PAD]"]
            if len(t) > max_len:
                warnings.warn(
                    f"Sentence of length {len(t)} has been truncated to {max_len}",
                    stacklevel=2
                )
                t = t[:max_len]
            truncated_texts.append(t)

        seq_lengths = [len(t) for t in truncated_texts]

        batch_dict = {
            'tokens': truncated_texts,
            'seq_length': torch.LongTensor(seq_lengths).unsqueeze(-1),
            'classes_mapping': classes_mapping,
            'classification': [item.get('classification', []) for item in batch_list],
            'extraction': [item.get('extraction', []) for item in batch_list],
            'embedding': [item.get('embedding', []) for item in batch_list],
            'structuring': [item.get('structuring', {}) for item in batch_list],
        }

        return self.create_batch_dict(batch_dict, classes_mapping)

    def tokenize_and_prepare_labels(self, batch, prepare_labels=True, *args, **kwargs):
        """Tokenize inputs and prepare multi-task labels.

        Args:
            batch: Multi-task batch dict from collate_raw_batch.
            prepare_labels: Whether to create label tensors.

        Returns:
            Dictionary containing:
                - input_ids, attention_mask, words_mask: Tokenized input tensors
                - classes_mapping: BatchClassesMapping
                - cat_labels, cat_batch_idx: Classification labels (if applicable)
                - ner_labels, ner_batch_idx: NER labels (if applicable)
                - rel_labels, rel_mask, rel_batch_idx: Relation labels (if applicable)
                - count_targets, gold_count_val: Count labels (if applicable)

        """
        classes_mapping = batch['classes_mapping']

        tokenized_input = self.tokenize_inputs(batch['tokens'], classes_mapping)
        tokenized_input['classes_mapping'] = classes_mapping

        if prepare_labels:
            max_seq_len = batch['seq_length'].max().item()

            # Reconstruct batch_list-like structure for label creation
            batch_list = []
            for i in range(len(batch['tokens'])):
                item = {
                    'classification': batch['classification'][i],
                    'extraction': batch['extraction'][i],
                    'embedding': batch['embedding'][i],
                }
                if 'structuring' in batch and i < len(batch['structuring']):
                    item['structuring'] = batch['structuring'][i]
                batch_list.append(item)

            # Classification labels: (total_cat_groups, C)
            cat_result = self.create_cat_labels(batch_list, classes_mapping)
            if cat_result is not None:
                tokenized_input['cat_labels'], tokenized_input['cat_batch_idx'] = cat_result

            # NER labels: (total_extraction_groups, L, C+1, 3)
            ner_result = self.create_ner_labels(batch_list, classes_mapping, max_seq_len)
            if ner_result is not None:
                tokenized_input['ner_labels'], tokenized_input['ner_batch_idx'] = ner_result

            # Relation labels: (total_extraction_groups, E, E, C)
            rel_result = self.create_rel_labels(batch_list, classes_mapping)
            if rel_result is not None:
                tokenized_input['rel_labels'], tokenized_input['rel_mask'] = rel_result[0], rel_result[1]
                tokenized_input['rel_batch_idx'] = rel_result[2]

            # Decoder labels: optional per-span entity type strings
            decoder_result = self.create_decoder_labels(batch_list, classes_mapping, max_seq_len)
            if decoder_result is not None:
                tokenized_input['decoder_labels_ids'] = decoder_result['input_ids']
                tokenized_input['decoder_labels_mask'] = decoder_result['attention_mask']
                tokenized_input['decoder_labels'] = decoder_result['labels']
                tokenized_input['decoder_group_idx'] = decoder_result['decoder_group_idx']

            # Relation decoder labels: optional per-relation type strings with entity pairs
            rel_decoder_result = self.create_rel_decoder_labels(batch_list, classes_mapping)
            if rel_decoder_result is not None:
                tokenized_input['rel_decoder_labels_ids'] = rel_decoder_result['input_ids']
                tokenized_input['rel_decoder_labels_mask'] = rel_decoder_result['attention_mask']
                tokenized_input['rel_decoder_labels'] = rel_decoder_result['labels']
                tokenized_input['rel_decoder_group_idx'] = rel_decoder_result['rel_decoder_group_idx']
                tokenized_input['rel_decoder_pair_idx'] = rel_decoder_result['rel_decoder_pair_idx']

            # Count labels: (batch_size,) — for CountHead
            count_result = self.create_count_labels(batch_list, classes_mapping)
            if count_result is not None:
                tokenized_input['count_targets'] = count_result[0]
                tokenized_input['gold_count_val'] = count_result[0].clone()

            # Embedding similarity labels
            embedding_result = self.create_embedding_labels(batch_list)
            if embedding_result is not None:
                tokenized_input['embedding_labels'] = embedding_result['embedding_labels']
                tokenized_input['embedding_pair_idx'] = embedding_result['embedding_pair_idx']

            # Span labels — only when represent_spans is enabled
            if getattr(self.config, 'represent_spans', False):
                # NER spans: (gB_ext, S, C) — per extraction group
                ner_span_result = self.create_span_labels(batch)
                if ner_span_result is not None:
                    tokenized_input['ner_span_labels'] = ner_span_result[0]
                    tokenized_input['ner_span_mask'] = ner_span_result[1]
                    tokenized_input['ner_span_idx'] = batch['span_idx']

                # Structuring spans: (gB_struct, S, X, C) — per structuring group
                struct_span_result = self.create_structuring_span_labels(
                    batch_list, classes_mapping, max_seq_len
                )
                if struct_span_result is not None:
                    tokenized_input['structuring_span_idx'] = struct_span_result[0]
                    tokenized_input['structuring_span_labels'] = struct_span_result[1]
                    tokenized_input['structuring_span_mask'] = struct_span_result[2]
                    tokenized_input['structuring_span_batch_idx'] = struct_span_result[3]

            # Structuring labels: (total_structuring_groups, X, L, C, 3) — anchor-based span extraction
            structuring_result = self.create_structuring_labels(batch_list, classes_mapping, max_seq_len)
            if structuring_result is not None:
                tokenized_input['structuring_labels'] = structuring_result[0]
                tokenized_input['structuring_mask'] = structuring_result[1]
                tokenized_input['structuring_batch_idx'] = structuring_result[2]
                tokenized_input['structuring_count'] = structuring_result[3]

        return tokenized_input

    def collate_fn(self, batch_list, prepare_labels=True, *args, **kwargs):
        """Collate function for DataLoader.

        End-to-end collation: takes raw examples from the dataset, generates
        class mappings, builds multi-task prompts, tokenizes, and creates labels.

        Args:
            batch_list: List of raw example dicts from the dataset.
            prepare_labels: Whether to prepare label tensors.

        Returns:
            Dictionary containing model inputs and multi-task labels.
        """
        batch = self.collate_raw_batch(batch_list, **kwargs)
        return self.tokenize_and_prepare_labels(batch, prepare_labels)