"""Decoder task processor."""

import torch

from .. import TaskProcessor


class DecoderProcessor(TaskProcessor):
    """Processor for decoder task."""

    def __init__(self, config, decoder_tokenizer=None, **kwargs):
        super().__init__(config)
        self.decoder_tokenizer = decoder_tokenizer

    def get_classes_mapping(self, batch_list, **kwargs):
        return None

    def contribute_prompt(self, classes_mapping, batch_idx, use_labels_encoder=False):
        return []

    def create_labels(self, batch_list, classes_mapping, max_seq_len=0, **kwargs):
        return self._collect_decoder_items(batch_list, classes_mapping, "ner", max_seq_len)

    def create_rel_decoder_labels(self, batch_list, classes_mapping):
        return self._collect_decoder_items(batch_list, classes_mapping, "rel")

    def _prepare_decoder_labels(self, decoder_label_strings):
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

    def _collect_decoder_items(self, batch_list, classes_mapping, label_type, max_seq_len=0):
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
                decoder_tokenized = self._prepare_decoder_labels([])
                if decoder_tokenized is None:
                    return None
                return {
                    "decoder_labels_ids": decoder_tokenized["input_ids"],
                    "decoder_labels_mask": decoder_tokenized["attention_mask"],
                    "decoder_labels": decoder_tokenized["labels"],
                    "decoder_group_idx": torch.LongTensor([0]),
                }
            return None

        decoder_tokenized = self._prepare_decoder_labels(decoder_label_strings)
        if decoder_tokenized is None:
            return None

        result = {
            "decoder_labels_ids": decoder_tokenized["input_ids"],
            "decoder_labels_mask": decoder_tokenized["attention_mask"],
            "decoder_labels": decoder_tokenized["labels"],
        }
        if label_type == "ner":
            result["decoder_group_idx"] = torch.LongTensor(decoder_group_idx)
        else:
            result["rel_decoder_group_idx"] = torch.LongTensor(decoder_group_idx)
            result["rel_decoder_pair_idx"] = torch.LongTensor(decoder_pair_idx)
        return result
