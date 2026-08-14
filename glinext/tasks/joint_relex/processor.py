"""Joint relex task processor."""

import random
import warnings

import torch

from ...processing.label_augmentation import AugmentableLabelGroup
from ..ner.processor import NERProcessor


class JointRelexProcessor(NERProcessor):
    """Processor for joint NER + relation extraction task.

    Inherits NERProcessor for shared span resolution and extraction mapping utilities.
    Relation classes are derived from extraction data by NERProcessor.
    REL tokens are already included by NERProcessor within extraction groups.
    """

    def __init__(self, config, tokenizer=None, words_splitter=None, **kwargs):
        super().__init__(config, tokenizer, words_splitter, **kwargs)
        self.rel_token = config.rel_token
        self._self_relation_warning_emitted = False

    def get_classes_mapping(self, batch_list, **kwargs):
        # This is used as the extraction-mapping fallback when no standalone
        # NER task is configured. With a standalone NER processor, the shared
        # processor selects that one and avoids duplicate work.
        if getattr(self.config, "ner_config", None) is not None:
            return None
        return super().get_classes_mapping(batch_list, **kwargs)

    @staticmethod
    def _relation_label(relation):
        if isinstance(relation, (list, tuple)):
            return relation[1] if len(relation) >= 2 else None
        if isinstance(relation, dict):
            return relation.get(
                "relation",
                relation.get("label", relation.get("type")),
            )
        return None

    def get_augmentable_label_groups(self, batch_list, classes_mapping):
        groups = (
            super().get_augmentable_label_groups(batch_list, classes_mapping)
            if getattr(self.config, "ner_config", None) is None
            else []
        )
        for batch_idx, item_mapping in enumerate(
            classes_mapping.extraction_mapping
        ):
            examples = batch_list[batch_idx].get("extraction", [])
            for group_idx, extraction_mapping in enumerate(item_mapping.items):
                mapping = extraction_mapping.rel_class_to_id
                if mapping is None or not mapping.class_to_id:
                    continue
                example = examples[group_idx] if group_idx < len(examples) else {}
                positives = list(dict.fromkeys(
                    label
                    for relation in example.get("relations", [])
                    if (label := self._relation_label(relation)) is not None
                ))
                groups.append(AugmentableLabelGroup(
                    task="joint_relex",
                    batch_idx=batch_idx,
                    group_idx=group_idx,
                    mapping=mapping,
                    positive_labels=positives,
                    parent_name=mapping.name,
                ))
        return groups

    def contribute_prompt(self, classes_mapping, batch_idx, use_labels_encoder=False):
        # NERProcessor emits the joint [SCHEMA] + [ENT] + [REL] prompt. The
        # common processor calls exactly one extraction prompt contributor,
        # preferring standalone NER when it exists.
        if getattr(self.config, "ner_config", None) is not None:
            return []
        return super().contribute_prompt(
            classes_mapping,
            batch_idx,
            use_labels_encoder,
        )

    def create_ner_labels(
        self,
        batch_list,
        classes_mapping,
        max_seq_len=0,
        **kwargs,
    ):
        """Build owned-NER labels without replacing relation label creation."""
        return super().create_labels(
            batch_list,
            classes_mapping,
            max_seq_len=max_seq_len,
            **kwargs,
        )

    def contribute_inference_input(self, item, joint_relations=None, **kwargs):
        if not joint_relations:
            return

        extraction = item.setdefault('extraction', [])
        by_name = {
            group.get('name'): group
            for group in extraction
        }
        for parent_name, jconf in joint_relations.items():
            entry = by_name.get(parent_name)
            if entry is None:
                entry = {
                    "name": parent_name,
                    "ner": [],
                }
                extraction.append(entry)
                by_name[parent_name] = entry
            entry["relations"] = []
            entry["all_labels"] = jconf.get("entities", [])
            entry["all_rel_labels"] = jconf.get("relations", [])

    def empty_inference_result(self, num_texts: int, joint_relations=None, **kwargs):
        if joint_relations is None:
            return None
        return {"joint_relex": [[] for _ in range(num_texts)]}

    @staticmethod
    def _normalize_sequence_lengths(batch_list, sequence_lengths):
        """Return one post-tokenization source length per batch item."""
        if sequence_lengths is None:
            return None
        if torch.is_tensor(sequence_lengths):
            values = sequence_lengths.detach().reshape(-1).cpu().tolist()
        else:
            values = list(sequence_lengths)
        if len(values) != len(batch_list):
            raise ValueError(
                "sequence_lengths must contain one value per batch item"
            )
        normalized = []
        for value in values:
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(
                    "sequence_lengths values must be non-negative integers"
                )
            normalized.append(int(value))
        return normalized

    def create_labels(self, batch_list, classes_mapping, **kwargs):
        total_groups = classes_mapping.total_extraction_groups()
        if total_groups == 0:
            return None

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
        # Candidate-pair adjacency used by the GLiNER-relex algorithm:
        # positives plus sampled no-relation pairs. This is intentionally
        # distinct from rel_labels.sum(-1), which marks only positive relations.
        rel_pair_mask = torch.zeros(total_groups, max_entities, max_entities, dtype=torch.float)
        rel_mask = torch.zeros(total_groups, dtype=torch.bool)
        rel_batch_idx = torch.zeros(total_groups, dtype=torch.long)
        rel_span_idx = torch.zeros(total_groups, max_entities, 2, dtype=torch.long)
        rel_span_mask = torch.zeros(total_groups, max_entities, dtype=torch.bool)
        # Entity class ids use the same group-local mapping as the NER logits.
        # Keeping them next to the span axis lets the model/decoder distinguish
        # equal boundaries predicted for different entity types.
        rel_span_class_idx = torch.full(
            (total_groups, max_entities), -1, dtype=torch.long,
        )

        max_seq_len = kwargs.get("max_seq_len", 0)
        sequence_lengths = self._normalize_sequence_lengths(
            batch_list,
            kwargs.get("sequence_lengths"),
        )
        add_reversed_negatives = kwargs.get("add_reversed_negatives", True)
        add_random_negatives = kwargs.get("add_random_negatives", True)
        negative_ratio = kwargs.get("negative_ratio", (1.0, 10.0))

        for flat_idx, batch_idx, group_idx, ext_mapping in classes_mapping.flat_extraction_iter():
            rel_batch_idx[flat_idx] = batch_idx
            if ext_mapping.rel_class_to_id is None:
                continue
            extraction_examples = batch_list[batch_idx].get('extraction', [])
            if group_idx >= len(extraction_examples):
                continue
            mapping = ext_mapping.rel_class_to_id
            if not mapping.class_to_id:
                continue
            rel_mask[flat_idx] = True
            example = extraction_examples[group_idx]
            item_max_seq_len = (
                sequence_lengths[batch_idx]
                if sequence_lengths is not None
                else max_seq_len
            )
            enforce_length = sequence_lengths is not None or item_max_seq_len > 0

            # Compact valid entities into 0..n-1 so build_all_entity_pairs
            # (which assumes contiguous indices) stays in sync with rel_labels.
            # Track the remap so relation head/tail ids land in the right slot.
            old_to_new = {}
            for orig_id, ent in enumerate(example.get('ner', [])):
                entity_label = self._entity_label(ent)
                entity_class_idx = ext_mapping.ner_class_to_id.class_to_id.get(
                    entity_label,
                )
                # A relation endpoint that the local NER label space cannot
                # predict would disappear at inference and therefore cannot
                # have a stable relation-entity index.
                if entity_class_idx is None:
                    continue
                new_id = len(old_to_new)
                if new_id >= max_entities:
                    break
                start, end = ent[0], ent[1]
                if enforce_length and (
                    start >= item_max_seq_len or end >= item_max_seq_len
                ):
                    continue
                rel_span_idx[flat_idx, new_id, 0] = start
                rel_span_idx[flat_idx, new_id, 1] = end
                rel_span_mask[flat_idx, new_id] = True
                rel_span_class_idx[flat_idx, new_id] = entity_class_idx
                old_to_new[orig_id] = new_id

            positive_pairs = set()
            for head_id, rel_type, tail_id in example.get('relations', []):
                if rel_type not in mapping.class_to_id:
                    continue
                new_head = old_to_new.get(head_id)
                new_tail = old_to_new.get(tail_id)
                if new_head is None or new_tail is None:
                    continue
                # Every supported pair builder excludes the diagonal. Do not
                # leave an unreachable positive in rel_labels.
                if new_head == new_tail:
                    if not self._self_relation_warning_emitted:
                        warnings.warn(
                            "Joint Relex does not score self-relations; "
                            "self-relation annotations are ignored.",
                            UserWarning,
                            stacklevel=2,
                        )
                        self._self_relation_warning_emitted = True
                    continue
                rel_class_idx = mapping.class_to_id[rel_type]
                rel_labels[flat_idx, new_head, new_tail, rel_class_idx] = 1.0
                positive_pairs.add((new_head, new_tail))

            n_valid = len(old_to_new)
            negative_pairs = set()
            if n_valid > 1:
                if add_reversed_negatives:
                    for head, tail in positive_pairs:
                        reversed_pair = (tail, head)
                        if reversed_pair not in positive_pairs:
                            negative_pairs.add(reversed_pair)

                if add_random_negatives:
                    if isinstance(negative_ratio, (tuple, list)):
                        ratio = random.uniform(float(negative_ratio[0]), float(negative_ratio[1]))
                    else:
                        ratio = float(negative_ratio)
                    target_negatives = max(1, int(len(positive_pairs) * ratio))
                    attempts = 0
                    max_attempts = max(target_negatives * 10, 0)
                    while len(negative_pairs) < target_negatives and attempts < max_attempts:
                        attempts += 1
                        head = random.randint(0, n_valid - 1)
                        tail = random.randint(0, n_valid - 1)
                        if head == tail:
                            continue
                        pair = (head, tail)
                        if pair in positive_pairs or pair in negative_pairs:
                            continue
                        negative_pairs.add(pair)

            for head, tail in positive_pairs | negative_pairs:
                rel_pair_mask[flat_idx, head, tail] = 1.0

        return {
            "rel_labels": rel_labels,
            "rel_pair_mask": rel_pair_mask,
            "rel_mask": rel_mask,
            "rel_batch_idx": rel_batch_idx,
            "rel_span_idx": rel_span_idx,
            "rel_span_mask": rel_span_mask,
            "rel_span_class_idx": rel_span_class_idx,
        }

    def prepare_label_encoder_inputs(self, classes_mapping, labels_tokenizer):
        if labels_tokenizer is None:
            return None

        encoded = {}
        if self.config.ner_config is None:
            # A self-contained Joint Relex head also needs the entity label
            # embeddings that a standalone NER processor would normally add.
            ner_encoded = super().prepare_label_encoder_inputs(
                classes_mapping,
                labels_tokenizer,
            )
            if ner_encoded is not None:
                encoded.update(ner_encoded)

        all_label_strings = []
        group_sizes = []
        has_any = False

        for _, _, _, ext_mapping in classes_mapping.flat_extraction_iter():
            if ext_mapping.rel_class_to_id is not None:
                labels = list(ext_mapping.rel_class_to_id.class_to_id.keys())
                has_any = True
            else:
                labels = []
            group_sizes.append(len(labels))
            all_label_strings.extend(labels)

        if not has_any or not all_label_strings:
            return encoded or None

        tokenized = labels_tokenizer(
            all_label_strings, return_tensors="pt", truncation=True,
            padding="longest", add_special_tokens=True
        )
        encoded.update({
            "rel_labels_input_ids": tokenized["input_ids"],
            "rel_labels_attention_mask": tokenized["attention_mask"],
            "rel_labels_group_size": torch.LongTensor(group_sizes),
        })
        return encoded
