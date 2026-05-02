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

    def empty_inference_results(self, num_texts: int, **kwargs):
        results = {}
        for proc in self.task_processors.values():
            empty = proc.empty_inference_result(num_texts, **kwargs)
            if empty:
                results.update(empty)
        return results

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

    # ── Generic label creation ────────────────────────────────────────────

    def create_labels(self, batch_list, classes_mapping=None, max_seq_len=0):
        if classes_mapping is None:
            raise ValueError("classes_mapping is required; use create_all_labels for raw batches.")
        return self.create_all_labels(
            batch_list, classes_mapping, max_seq_len=max_seq_len,
        )

    def create_all_labels(self, batch_list, classes_mapping, max_seq_len=0):
        """Create labels from all task processors in a single call.

        Returns a flat dict of all label tensors from all active tasks.
        """
        for item in batch_list:
            for proc in self.task_processors.values():
                if hasattr(proc, "resolve_spans"):
                    proc.resolve_spans(item)

        all_labels = {}
        for name, proc in self.task_processors.items():
            result = proc.create_labels(
                batch_list, classes_mapping, max_seq_len=max_seq_len,
            )
            if result is not None:
                all_labels.update(result)
        return all_labels

    # ── Label creation (delegates to task processors) ───────────────────

    def create_cat_labels(self, batch_list, classes_mapping):
        if "classification" in self.task_processors:
            result = self.task_processors["classification"].create_labels(batch_list, classes_mapping)
            if result is not None:
                return result["cat_labels"], result["cat_batch_idx"]
        return None

    def create_ner_labels(self, batch_list, classes_mapping, max_seq_len):
        for item in batch_list:
            self.resolve_extraction_spans(item)
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
        for item in batch_list:
            self.resolve_extraction_spans(item)
        if "joint_relex" in self.task_processors:
            result = self.task_processors["joint_relex"].create_labels(
                batch_list, classes_mapping, max_seq_len=max_seq_len,
            )
            if result is not None:
                return result
        return None

    def create_open_rel_labels(self, batch_list, classes_mapping, max_seq_len):
        for item in batch_list:
            self.resolve_open_relex_spans(item)
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
        for item in batch_list:
            self.resolve_structuring_spans(item)
        if "structuring" in self.task_processors:
            result = self.task_processors["structuring"].create_labels(
                batch_list, classes_mapping, max_seq_len=max_seq_len
            )
            if result is not None:
                return (result["structuring_labels"], result["structuring_mask"],
                        result["structuring_batch_idx"], result["structuring_count"])
        return None

    # ── Preprocessing ───────────────────────────────────────────────────

    def preprocess_example(self, item, extraction_mapping=None):
        if "ner" in self.task_processors and extraction_mapping is not None:
            return self.task_processors["ner"].preprocess_example(
                item, extraction_mapping,
            )
        return self._preprocess_text_only(item)

    def _preprocess_text_only(self, item):
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

        return {
            "tokens": tokens,
            "seq_length": len(tokens),
        }

    # ── Batch dict creation ─────────────────────────────────────────────

    def create_batch_dict(self, batch, classes_mapping):
        batch_size = len(batch["tokens"])

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

        if "ner" in self.task_processors:
            batch_dict["span_idx"] = batch.get("span_idx")
            batch_dict["span_label"] = batch.get("span_label")
            batch_dict = self.task_processors["ner"].add_span_batch_fields(
                batch_dict, classes_mapping,
            )

        return batch_dict

    # ── Raw batch collation ─────────────────────────────────────────────

    def collate_raw_batch(self, batch_list, **kwargs):
        # Resolve spans via task processors
        for item in batch_list:
            for proc in self.task_processors.values():
                proc.resolve_spans(item)

        classes_mapping = self.batch_generate_class_mappings(batch_list, **kwargs)

        if "ner" in self.task_processors:
            preprocessed = [
                self.task_processors["ner"].preprocess_example(
                    item, classes_mapping.extraction_mapping[i],
                )
                for i, item in enumerate(batch_list)
            ]
        else:
            preprocessed = [self._preprocess_text_only(item) for item in batch_list]

        texts = [item['tokens'] for item in preprocessed]
        seq_lengths = [len(t) for t in texts]

        batch_dict = {
            'tokens': texts,
            'seq_length': torch.LongTensor(seq_lengths).unsqueeze(-1),
            'classes_mapping': classes_mapping,
            'classification': [item.get('classification', []) for item in batch_list],
            'extraction': [item.get('extraction', []) for item in batch_list],
            'embedding': [item.get('embedding', []) for item in batch_list],
            'structuring': [item.get('structuring', {}) for item in batch_list],
            'open_relex': [item.get('open_relex', []) for item in batch_list],
            'span_idx': [item.get('span_idx') for item in preprocessed],
            'span_label': [item.get('span_label') for item in preprocessed],
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
                tokenized_input['rel_pair_mask'] = rel_result.get('rel_pair_mask')
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
                    return_tensors="pt",
                    truncation=True,
                    padding="longest",
                )
                tokenized_input['embedding_input_ids'] = emb_tokenized['input_ids']
                tokenized_input['embedding_attention_mask'] = emb_tokenized['attention_mask']
                tokenized_input['embedding_labels'] = embedding_result['embedding_labels']
                tokenized_input['embedding_pair_idx'] = embedding_result['embedding_pair_idx']

            if getattr(self.config, 'represent_spans', False) and "ner" in self.task_processors:
                ner_span_result = self.task_processors["ner"].create_span_labels(
                    batch, classes_mapping,
                )
                if ner_span_result is not None:
                    tokenized_input.update(ner_span_result)

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
