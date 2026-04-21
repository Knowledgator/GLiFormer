"""GLiNExT processor — thin orchestrator delegating to per-task processors."""

import warnings
import torch
from typing import Dict, Optional

from gliner.data_processing import BaseProcessor

from .mappings import (
    CatClassMapping,
    ExtractionClassMapping,
    StructuringClassMapping,
    OpenRelexClassMapping,
    BatchClassesMapping,
)
from ..tasks.ner.processor import NERProcessor
from ..tasks.classification.processor import ClassificationProcessor
from ..tasks.joint_relex.processor import JointRelexProcessor
from ..tasks.open_relex.processor import OpenRelexProcessor
from ..tasks.count.processor import CountProcessor
from ..tasks.structuring.processor import StructuringProcessor
from ..tasks.embedding.processor import EmbeddingProcessor


class GLiNextProcessor(BaseProcessor):
    def __init__(self, config, tokenizer, words_splitter,
                 labels_tokenizer: Optional[object] = None):
        super().__init__(config, tokenizer, words_splitter)
        self.labels_tokenizer = labels_tokenizer

        self.seq_token = config.seq_token
        self.cat_token = config.cat_token
        self.ent_token = config.ent_token
        self.sep_token = config.sep_token
        self.rel_token = config.rel_token
        self.parent_token = config.parent_token
        self.child_token = config.child_token

        # ── Register per-task processors ────────────────────────────────
        self.task_processors: Dict[str, object] = {}

        if config.ner_config is not None:
            self.task_processors["ner"] = NERProcessor(config, tokenizer, words_splitter)
        if config.classification_config is not None:
            self.task_processors["classification"] = ClassificationProcessor(config)
        if config.joint_relex_config is not None:
            self.task_processors["joint_relex"] = JointRelexProcessor(config, tokenizer, words_splitter)
        if config.open_relex_config is not None:
            self.task_processors["open_relex"] = OpenRelexProcessor(config, tokenizer, words_splitter)
        if config.count_config is not None:
            self.task_processors["count"] = CountProcessor(config)
        if config.structuring_config is not None:
            self.task_processors["structuring"] = StructuringProcessor(config, tokenizer, words_splitter)
        if config.embedding_config is not None:
            self.task_processors["embedding"] = EmbeddingProcessor(config)

    # ── Class mappings ──────────────────────────────────────────────────

    def batch_generate_class_mappings(self, batch_list, **kwargs):
        cat_mapping = []
        extraction_mapping = []
        structuring_mapping = []
        open_relex_mapping = []

        if "classification" in self.task_processors:
            cat_mapping = self.task_processors["classification"].get_classes_mapping(
                batch_list, **kwargs
            )
        else:
            cat_mapping = [CatClassMapping(cat_class_to_id=[]) for _ in batch_list]

        if "ner" in self.task_processors:
            extraction_mapping = self.task_processors["ner"].get_classes_mapping(
                batch_list, **kwargs
            )
        else:
            extraction_mapping = [ExtractionClassMapping() for _ in batch_list]

        if "structuring" in self.task_processors:
            structuring_mapping = self.task_processors["structuring"].get_classes_mapping(
                batch_list, **kwargs
            )
        else:
            structuring_mapping = [StructuringClassMapping() for _ in batch_list]

        if "open_relex" in self.task_processors:
            open_relex_mapping = self.task_processors["open_relex"].get_classes_mapping(
                batch_list, **kwargs
            )
        else:
            open_relex_mapping = [OpenRelexClassMapping() for _ in batch_list]

        return BatchClassesMapping(
            cat_mapping=cat_mapping,
            extraction_mapping=extraction_mapping,
            structuring_mapping=structuring_mapping,
            open_relex_mapping=open_relex_mapping,
        )

    # ── Prompt construction ─────────────────────────────────────────────

    def prepare_inputs(self, texts, classes_mapping, blank=None, add_entities=True, **kwargs):
        use_labels_encoder = self.labels_tokenizer is not None
        input_texts = []
        prompt_lengths = []

        # Fixed ordering: classification -> extraction (NER+REL) -> structuring
        for i, text in enumerate(texts):
            prompt = [self.seq_token]

            if "classification" in self.task_processors:
                prompt.extend(self.task_processors["classification"].contribute_prompt(
                    classes_mapping, i, use_labels_encoder,
                ))

            if "ner" in self.task_processors:
                prompt.extend(self.task_processors["ner"].contribute_prompt(
                    classes_mapping, i, use_labels_encoder,
                ))

            if "open_relex" in self.task_processors:
                prompt.extend(self.task_processors["open_relex"].contribute_prompt(
                    classes_mapping, i, use_labels_encoder,
                ))

            if "structuring" in self.task_processors:
                prompt.extend(self.task_processors["structuring"].contribute_prompt(
                    classes_mapping, i, use_labels_encoder,
                ))

            prompt.append(self.sep_token)
            prompt_lengths.append(len(prompt))
            input_texts.append(prompt + list(text))

        return input_texts, prompt_lengths

    # ── Labels encoder inputs ───────────────────────────────────────────

    def prepare_all_label_encoder_inputs(self, classes_mapping):
        """Collect label encoder inputs from all task processors."""
        if self.labels_tokenizer is None:
            return {}

        result = {}
        for name, proc in self.task_processors.items():
            enc = proc.prepare_label_encoder_inputs(classes_mapping, self.labels_tokenizer)
            if enc is not None:
                result.update(enc)
        return result

    # ── Tokenization ────────────────────────────────────────────────────

    def tokenize_inputs(self, texts, classes_mapping, **kwargs):
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

        # Label encoder inputs
        label_enc = self.prepare_all_label_encoder_inputs(classes_mapping)
        tokenized_inputs.update(label_enc)

        return tokenized_inputs

    # ── Span resolution (delegates to task processors) ──────────────────

    def resolve_extraction_spans(self, item):
        if "ner" in self.task_processors:
            self.task_processors["ner"].resolve_spans(item)
        return item

    def resolve_structuring_spans(self, item):
        if "structuring" in self.task_processors:
            self.task_processors["structuring"].resolve_spans(item)
        return item

    def resolve_open_relex_spans(self, item):
        if "open_relex" in self.task_processors:
            self.task_processors["open_relex"].resolve_spans(item)
        return item

    # ── Span index preparation ──────────────────────────────────────────

    def prepare_span_idx(self, ner, classes_to_id, num_tokens):
        if "ner" in self.task_processors:
            return self.task_processors["ner"].prepare_span_idx(ner, classes_to_id, num_tokens)
        return None, None

    def _generate_negative_spans(self, positive_spans, num_tokens, num_negatives, max_width=None):
        if "ner" in self.task_processors:
            return self.task_processors["ner"]._generate_negative_spans(
                positive_spans, num_tokens, num_negatives, max_width
            )
        return []

    # ── Generic label creation ────────────────────────────────────────────

    def create_all_labels(self, batch_list, classes_mapping, max_seq_len=0):
        """Create labels from all task processors in a single call.

        Returns a flat dict of all label tensors from all active tasks.
        """
        all_labels = {}
        for name, proc in self.task_processors.items():
            result = proc.create_labels(
                batch_list, classes_mapping, max_seq_len=max_seq_len,
            )
            if result is not None:
                all_labels.update(result)
        return all_labels

    # ── Legacy compatibility methods ────────────────────────────────────

    def sort_extraction_data(self, item):
        if "ner" in self.task_processors:
            NERProcessor._sort_extraction_data(item)

    def resolve_entity_spans(self, text, tokens_with_spans, ner):
        return NERProcessor._resolve_entity_spans(text, tokens_with_spans, ner)

    # ── Label creation (delegates to task processors) ───────────────────

    def create_cat_labels(self, batch_list, classes_mapping):
        if "classification" in self.task_processors:
            result = self.task_processors["classification"].create_labels(batch_list, classes_mapping)
            if result is not None:
                return result["cat_labels"], result["cat_batch_idx"]
        return None

    def create_ner_labels(self, batch_list, classes_mapping, max_seq_len):
        if "ner" in self.task_processors:
            result = self.task_processors["ner"].create_labels(
                batch_list, classes_mapping, max_seq_len=max_seq_len
            )
            if result is not None:
                return result["ner_labels"], result["ner_batch_idx"]
        return None

    def create_rel_labels(self, batch_list, classes_mapping, max_seq_len=0):
        return self.create_joint_rel_labels(batch_list, classes_mapping, max_seq_len=max_seq_len)

    def create_joint_rel_labels(self, batch_list, classes_mapping, max_seq_len=0):
        if "joint_relex" in self.task_processors:
            result = self.task_processors["joint_relex"].create_labels(
                batch_list, classes_mapping, max_seq_len=max_seq_len,
            )
            if result is not None:
                return result
        return None

    def create_open_rel_labels(self, batch_list, classes_mapping, max_seq_len):
        if "open_relex" in self.task_processors:
            return self.task_processors["open_relex"].create_labels(
                batch_list, classes_mapping, max_seq_len=max_seq_len,
            )
        return None

    def create_count_labels(self, batch_list, classes_mapping):
        if "count" in self.task_processors:
            result = self.task_processors["count"].create_labels(batch_list, classes_mapping)
            if result is not None:
                return result["count_targets"], result["count_targets"]
        return None

    def create_embedding_labels(self, batch_list):
        if "embedding" in self.task_processors:
            return self.task_processors["embedding"].create_labels(batch_list, None)
        return None

    def create_structuring_labels(self, batch_list, classes_mapping, max_seq_len):
        if "structuring" in self.task_processors:
            result = self.task_processors["structuring"].create_labels(
                batch_list, classes_mapping, max_seq_len=max_seq_len
            )
            if result is not None:
                return (result["structuring_labels"], result["structuring_mask"],
                        result["structuring_batch_idx"], result["structuring_count"])
        return None

    # ── Span labels (NER + structuring) ─────────────────────────────────

    def create_span_labels(self, batch):
        if "span_label" not in batch or "span_mask" not in batch:
            return None

        span_label = batch["span_label"]
        span_mask = batch["span_mask"]
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

        valid = span_mask & (span_label > 0)
        class_indices = (span_label - 1).clamp(min=0)

        if valid.any():
            flat_indices = valid.nonzero(as_tuple=False)
            row = flat_indices[:, 0]
            col = flat_indices[:, 1]
            cls = class_indices[row, col]
            in_range = cls < max_num_classes
            labels_one_hot[row[in_range], col[in_range], cls[in_range]] = 1.0

        return labels_one_hot, span_mask

    def create_structuring_span_labels(self, batch_list, classes_mapping, max_seq_len):
        """Create span-level structuring labels — kept in orchestrator for backward compat."""
        total_groups = classes_mapping.total_structuring_groups()
        if total_groups == 0:
            return None

        max_instances = 0
        max_fields = 0
        has_any = False

        all_group_spans = []
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

            neg_ratio = getattr(self.config, 'neg_spans_ratio', 0)
            neg_count = int(len(group_spans) * neg_ratio)
            if neg_count > 0 and max_seq_len > 0:
                negatives = self._generate_negative_spans(
                    positive_spans, max_seq_len, neg_count
                )
                for st, ed in negatives:
                    group_spans.append((st, ed, -1, -1))

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

    # ── Legacy create_labels for single-task NER ────────────────────────

    def create_labels(self, batch):
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
                word_labels[i, st, class_idx, 0] = 1
                word_labels[i, ed, class_idx, 1] = 1
                word_labels[i, st:ed + 1, class_idx, 2] = 1

        return word_labels

    # ── Preprocessing ───────────────────────────────────────────────────

    def preprocess_example(self, item, classes_mapping):
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
            "entities": ext_example.get('ner', []) if item.get('extraction') else [],
            "span_idx": item_span_idx,
            "span_label": item_span_label,
        }

    # ── Batch dict creation ─────────────────────────────────────────────

    def create_batch_dict(self, batch, classes_mapping):
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
            "open_relex": batch.get("open_relex", [[] for _ in range(batch_size)]),
        }

        if total_groups == 0:
            return batch_dict

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

    # ── Raw batch collation ─────────────────────────────────────────────

    def collate_raw_batch(self, batch_list, **kwargs):
        # Resolve spans via task processors
        for item in batch_list:
            for proc in self.task_processors.values():
                proc.resolve_spans(item)

        classes_mapping = self.batch_generate_class_mappings(batch_list, **kwargs)

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
            'open_relex': [item.get('open_relex', []) for item in batch_list],
        }

        return self.create_batch_dict(batch_dict, classes_mapping)

    # ── Tokenize + prepare labels ───────────────────────────────────────

    def tokenize_and_prepare_labels(self, batch, prepare_labels=True, *args, **kwargs):
        classes_mapping = batch['classes_mapping']
        tokenized_input = self.tokenize_inputs(batch['tokens'], classes_mapping)
        tokenized_input['classes_mapping'] = classes_mapping

        if prepare_labels:
            max_seq_len = batch['seq_length'].max().item()

            batch_list = []
            for i in range(len(batch['tokens'])):
                item = {
                    'classification': batch['classification'][i],
                    'extraction': batch['extraction'][i],
                    'embedding': batch['embedding'][i],
                }
                if 'structuring' in batch and i < len(batch['structuring']):
                    item['structuring'] = batch['structuring'][i]
                if 'open_relex' in batch and i < len(batch['open_relex']):
                    item['open_relex'] = batch['open_relex'][i]
                batch_list.append(item)

            # Delegate label creation to task processors via wrapper methods
            cat_result = self.create_cat_labels(batch_list, classes_mapping)
            if cat_result is not None:
                tokenized_input['cat_labels'], tokenized_input['cat_batch_idx'] = cat_result

            ner_result = self.create_ner_labels(batch_list, classes_mapping, max_seq_len)
            if ner_result is not None:
                tokenized_input['ner_labels'], tokenized_input['ner_batch_idx'] = ner_result

            rel_result = self.create_joint_rel_labels(batch_list, classes_mapping, max_seq_len=max_seq_len)
            if rel_result is not None:
                tokenized_input['rel_labels'] = rel_result['rel_labels']
                tokenized_input['rel_mask'] = rel_result['rel_mask']
                tokenized_input['rel_batch_idx'] = rel_result['rel_batch_idx']
                tokenized_input['rel_span_idx'] = rel_result['rel_span_idx']
                tokenized_input['rel_span_mask'] = rel_result['rel_span_mask']

            open_rel_result = self.create_open_rel_labels(batch_list, classes_mapping, max_seq_len)
            if open_rel_result is not None:
                tokenized_input['open_rel_labels'] = open_rel_result['open_rel_labels']
                tokenized_input['open_rel_mask'] = open_rel_result['open_rel_mask']
                tokenized_input['open_rel_batch_idx'] = open_rel_result['open_rel_batch_idx']
                tokenized_input['open_rel_count'] = open_rel_result['open_rel_count']

            count_result = self.create_count_labels(batch_list, classes_mapping)
            if count_result is not None:
                tokenized_input['count_targets'] = count_result[0]
                tokenized_input['count_val'] = count_result[1]

            embedding_result = self.create_embedding_labels(batch_list)
            if embedding_result is not None:
                emb_texts = embedding_result.pop('embedding_texts')
                emb_tokenized = self.transformer_tokenizer(
                    emb_texts,
                    is_split_into_words=True,
                    return_tensors="pt",
                    truncation=True,
                    padding="longest",
                )
                tokenized_input['embedding_input_ids'] = emb_tokenized['input_ids']
                tokenized_input['embedding_attention_mask'] = emb_tokenized['attention_mask']
                tokenized_input['embedding_labels'] = embedding_result['embedding_labels']
                tokenized_input['embedding_pair_idx'] = embedding_result['embedding_pair_idx']

            if getattr(self.config, 'represent_spans', False):
                ner_span_result = self.create_span_labels(batch)
                if ner_span_result is not None:
                    tokenized_input['ner_span_labels'] = ner_span_result[0]
                    tokenized_input['ner_span_mask'] = ner_span_result[1]
                    tokenized_input['ner_span_idx'] = batch['span_idx']

            if "structuring" in self.task_processors:
                struct_span_result = self.task_processors["structuring"].create_span_labels(
                    batch_list, classes_mapping, max_seq_len=max_seq_len,
                )
                if struct_span_result is not None:
                    tokenized_input.update(struct_span_result)

            if "open_relex" in self.task_processors:
                open_rel_span_result = self.task_processors["open_relex"].create_span_labels(
                    batch_list, classes_mapping, max_seq_len=max_seq_len,
                )
                if open_rel_span_result is not None:
                    tokenized_input.update(open_rel_span_result)

            structuring_result = self.create_structuring_labels(batch_list, classes_mapping, max_seq_len)
            if structuring_result is not None:
                tokenized_input['structuring_labels'] = structuring_result[0]
                tokenized_input['structuring_mask'] = structuring_result[1]
                tokenized_input['structuring_batch_idx'] = structuring_result[2]
                tokenized_input['structuring_count'] = structuring_result[3]

        return tokenized_input

    def collate_fn(self, batch_list, prepare_labels=True, *args, **kwargs):
        batch = self.collate_raw_batch(batch_list, **kwargs)
        return self.tokenize_and_prepare_labels(batch, prepare_labels)
