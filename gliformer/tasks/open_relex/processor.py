"""Processor for entity, pair, and relation set extraction."""

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
    """Build entity, directed-pair, and relation targets for open relex.

    The task first recognizes relation-conditioned endpoint entities.  Its
    second stage groups source/target entities into unique directed pairs,
    independently of relation type and predicted anchor-query capacity.  The
    third stage assigns every applicable relation label to each gold pair.

    Training uses the same canonical ``extraction`` groups as joint NER and
    relation extraction: ``ner`` contains the entity mentions and every
    relation is ``[head_entity_index, relation_label, tail_entity_index]``.
    The dedicated ``open_relex`` endpoint-object format is also supported.
    """

    data_key = "open_relex"
    extraction_data_key = "extraction"
    resolved_flag = "_gliformer_open_relex_spans_resolved"
    resolved_group_flag = "_gliformer_open_relex_group_spans_resolved"

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

    def _group_source(self, item):
        explicit_groups = item.get(self.data_key) or []
        if explicit_groups:
            return self.data_key, explicit_groups

        extraction_groups = self._extraction_groups(item)
        if extraction_groups:
            return self.extraction_data_key, extraction_groups

        return None, []

    def _groups(self, item):
        return self._group_source(item)[1]

    def _relation_types(self, group):
        if self._is_extraction_group(group):
            if "all_rel_labels" in group:
                candidates = group["all_rel_labels"]
            else:
                candidates = (
                    self._relation_name(relation)
                    for relation in group.get("relations", [])
                )
        elif "all_labels" in group:
            candidates = group["all_labels"]
        else:
            candidates = (
                self._relation_name(relation)
                for relation in group.get("relations", [])
            )
        return list(dict.fromkeys(
            label for label in candidates if label
        ))

    def _mapped_groups(self, item):
        """Return groups that occupy a flattened model/prompt slot."""

        return [
            group
            for group in self._groups(item)
            if self._relation_types(group)
        ]

    def has_training_annotations(self, item):
        """Whether an item supplies a usable set-relation label space."""

        return bool(self._mapped_groups(item))

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

    @staticmethod
    def _endpoint_entry(relation, role):
        keys = ("source", "head") if role == 0 else ("target", "tail")
        if not isinstance(relation, dict):
            return keys[0], None
        for key in keys:
            if key in relation:
                return key, relation.get(key)
        return keys[0], None

    def get_classes_mapping(self, batch_list, shuffle_labels=False, **kwargs):
        mappings = []
        for item in batch_list:
            item_mappings = []
            for group in self._mapped_groups(item):
                relation_types = self._relation_types(group)
                if shuffle_labels:
                    random.shuffle(relation_types)
                item_mappings.append(
                    OpenRelexItemMapping(
                        rel_class_to_id=BaseClassMapping(
                            class_to_id={
                                relation_type: index
                                for index, relation_type in enumerate(
                                    relation_types
                                )
                            },
                            name=group.get("name"),
                            description=group.get("description"),
                        ),
                        name=group.get("name"),
                    )
                )
            mappings.append(OpenRelexClassMapping(items=item_mappings))
        return mappings

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

    def contribute_prompt(
        self,
        classes_mapping,
        batch_idx,
        use_labels_encoder=False,
    ):
        mappings = getattr(classes_mapping, "open_relex_mapping", None)
        if mappings is None or batch_idx >= len(mappings):
            return []

        prompt = []
        for item_mapping in mappings[batch_idx].items:
            prompt.append(self.parent_token)
            relation_mapping = item_mapping.rel_class_to_id
            if relation_mapping.name:
                prompt.append(relation_mapping.name)
            if relation_mapping.description:
                prompt.append(relation_mapping.description)
            if not use_labels_encoder:
                prompt.extend(
                    f"{self.rel_token} {relation_name}"
                    for relation_name in relation_mapping.class_to_id
                )
            prompt.append(self.sep_token)
        return prompt

    def contribute_inference_input(
        self,
        item,
        relations=None,
        **kwargs,
    ):
        relation_groups = self._normalize_label_groups(relations)
        if not relation_groups:
            return
        item[self.data_key] = [
            {
                "name": parent_name,
                "relations": [],
                "all_labels": relation_labels,
            }
            for parent_name, relation_labels in relation_groups.items()
        ]

    def empty_inference_result(
        self,
        num_texts,
        relations=None,
        **kwargs,
    ):
        if relations is None:
            return None
        return {self.data_key: [[] for _ in range(num_texts)]}

    def _normalize_endpoint(self, text, tokens_with_spans, value):
        if isinstance(value, dict):
            endpoint_text = value.get("text")
            if (
                endpoint_text is None
                and "start" in value
                and "end" in value
            ):
                try:
                    endpoint_text = text[
                        int(value["start"]):int(value["end"])
                    ]
                except (TypeError, ValueError):
                    endpoint_text = ""
            spans = self._resolve_labeled_span(
                text,
                tokens_with_spans,
                value,
                label="entity",
            )
        else:
            endpoint_text = value
            spans = self._resolve_labeled_span(
                text,
                tokens_with_spans,
                {"text": str(value)},
                label="entity",
            )

        if not spans:
            return {
                "text": str(endpoint_text),
                "start": -1,
                "end": -1,
            }
        start, end, _ = spans[0]
        return {
            "text": str(endpoint_text),
            "start": start,
            "end": end,
        }

    def resolve_spans(self, item):
        """Resolve source/target (or legacy head/tail) mentions to words."""
        if item.get(self.resolved_flag):
            return
        source, groups = self._group_source(item)
        if not groups:
            return

        if source == self.extraction_data_key:
            # Reuse the canonical joint NER/relex resolver. It converts entity
            # character offsets to inclusive token spans, drops invalid
            # entities, and remaps relation indices after filtering/sorting.
            super().resolve_spans(item)
            item[self.resolved_flag] = True
            return

        unresolved = [
            (group_idx, group)
            for group_idx, group in enumerate(groups)
            if not group.get(self.resolved_group_flag)
        ]
        if not unresolved:
            item[self.resolved_flag] = True
            return

        extraction_groups = [
            (group_idx, group)
            for group_idx, group in unresolved
            if self._is_extraction_group(group)
        ]
        if extraction_groups:
            # Internal collation may transport normalized extraction groups in
            # the dedicated task slot. Resolve only previously unseen groups:
            # endpoint token indices are not character offsets and resolving
            # them twice can select a different repeated text mention.
            extraction_item = {
                "text": item.get("text", ""),
                self.extraction_data_key: [
                    group for _, group in extraction_groups
                ],
            }
            if "tokenized_text" in item:
                extraction_item["tokenized_text"] = item["tokenized_text"]
            super().resolve_spans(extraction_item)
            for (group_idx, _), resolved_group in zip(
                extraction_groups,
                extraction_item[self.extraction_data_key],
                strict=True,
            ):
                resolved_group[self.resolved_group_flag] = True
                groups[group_idx] = resolved_group

        text = item.get("text", "")
        tokens_with_spans, _ = self._tokenize_text(item)
        for _, group in unresolved:
            if self._is_extraction_group(group):
                continue
            if tokens_with_spans is None:
                continue
            for relation in group.get("relations", []):
                for role in (0, 1):
                    key, value = self._endpoint_entry(relation, role)
                    if value is not None:
                        relation[key] = self._normalize_endpoint(
                            text,
                            tokens_with_spans,
                            value,
                        )
            group[self.resolved_group_flag] = True
        if all(group.get(self.resolved_group_flag) for group in groups):
            item[self.resolved_flag] = True

    @classmethod
    def _resolved_span(cls, relation, role, group=None):
        if cls._is_indexed_relation(relation):
            entity_position = 0 if role == 0 else 2
            entity_idx = relation[entity_position]
            entities = group.get("ner", []) if isinstance(group, dict) else []
            if entity_idx < 0 or entity_idx >= len(entities):
                return None
            entity = entities[entity_idx]
            if isinstance(entity, dict):
                start = entity.get("start", -1)
                end = entity.get("end", -1)
            elif isinstance(entity, (list, tuple)) and len(entity) >= 2:
                start, end = entity[0], entity[1]
            else:
                return None
            try:
                return int(start), int(end)
            except (TypeError, ValueError):
                return None

        _, endpoint = OpenRelexProcessor._endpoint_entry(relation, role)
        if not isinstance(endpoint, dict):
            return None
        try:
            start = int(endpoint.get("start", -1))
            end = int(endpoint.get("end", -1))
        except (TypeError, ValueError):
            return None
        return start, end

    def _collect_group_targets(
        self,
        group,
        relation_to_id,
        max_seq_len,
    ):
        """Coalesce repeated triples into directed pair-level targets."""
        pair_relations = {}
        for relation in group.get("relations", []):
            relation_name = self._relation_name(relation)
            if relation_name not in relation_to_id:
                continue
            source_span = self._resolved_span(relation, 0, group)
            target_span = self._resolved_span(relation, 1, group)
            if source_span is None or target_span is None:
                continue
            source_start, source_end = source_span
            target_start, target_end = target_span
            if not (
                0 <= source_start <= source_end < max_seq_len
                and 0 <= target_start <= target_end < max_seq_len
            ):
                continue
            pair = (
                source_start,
                source_end,
                target_start,
                target_end,
            )
            pair_relations.setdefault(pair, set()).add(
                relation_to_id[relation_name]
            )

        ordered_pairs = [
            (pair, pair_relations[pair])
            for pair in sorted(pair_relations)
        ]
        boundaries = sorted({
            boundary
            for pair, _ in ordered_pairs
            for boundary in ((pair[0], pair[1]), (pair[2], pair[3]))
        })
        return ordered_pairs, boundaries

    def create_labels(self, batch_list, classes_mapping, max_seq_len=0, **kwargs):
        """Create three-stage set targets without query-capacity padding.

        Shapes use ``BN`` flattened groups, ``G`` unique directed gold pairs,
        ``R`` relation classes, and ``E`` unique endpoint boundaries.

        Entity targets remain relation-conditioned at ``[BN, L, R, 3]``.
        Pair relation targets are multi-label ``[BN, G, R]`` tensors, while
        pair endpoint assignments are relation-independent ``[BN, G, E, 2]``
        tensors whose final axis denotes source and target respectively.
        """
        max_seq_len = int(max_seq_len)
        if max_seq_len <= 0:
            return None
        for item in batch_list:
            self.resolve_spans(item)

        total_groups = classes_mapping.total_open_relex_groups()
        if total_groups == 0:
            return None

        group_targets = []
        max_pairs = 0
        max_relations = 0
        max_entities = 0
        for _, batch_idx, group_idx, item_mapping in (
            classes_mapping.flat_open_relex_iter()
        ):
            groups = self._mapped_groups(batch_list[batch_idx])
            relation_to_id = (
                item_mapping.rel_class_to_id.class_to_id
            )
            max_relations = max(max_relations, len(relation_to_id))
            if group_idx < len(groups):
                pairs, boundaries = self._collect_group_targets(
                    groups[group_idx],
                    relation_to_id,
                    max_seq_len,
                )
            else:
                pairs, boundaries = [], []
            group_targets.append((pairs, boundaries, group_idx < len(groups)))
            max_pairs = max(max_pairs, len(pairs))
            max_entities = max(max_entities, len(boundaries))

        if max_relations == 0:
            return None

        # Keep one masked placeholder along the variable gold/entity axes for
        # all-negative batches.  The zero pair count tells Hungarian matching
        # that there is no gold set element, while the dense zero tensors still
        # supervise entity negatives, unused relation slots, and objectness.
        max_pairs = max(max_pairs, 1)
        max_entities = max(max_entities, 1)

        entity_labels = torch.zeros(
            total_groups,
            max_seq_len,
            max_relations,
            3,
            dtype=torch.float,
        )
        relation_labels = torch.zeros(
            total_groups,
            max_pairs,
            max_relations,
            dtype=torch.float,
        )
        assignment_labels = torch.zeros(
            total_groups,
            max_pairs,
            max_entities,
            2,
            dtype=torch.float,
        )
        span_idx = torch.zeros(
            total_groups,
            max_entities,
            2,
            dtype=torch.long,
        )
        span_mask = torch.zeros(
            total_groups,
            max_entities,
            dtype=torch.bool,
        )
        group_mask = torch.zeros(total_groups, dtype=torch.bool)
        batch_indices = torch.zeros(total_groups, dtype=torch.long)
        pair_counts = torch.zeros(total_groups, dtype=torch.long)

        for flat_idx, batch_idx, _, _ in (
            classes_mapping.flat_open_relex_iter()
        ):
            pairs, boundaries, group_present = group_targets[flat_idx]
            batch_indices[flat_idx] = batch_idx
            group_mask[flat_idx] = group_present
            pair_counts[flat_idx] = len(pairs)

            boundary_to_id = {
                boundary: entity_idx
                for entity_idx, boundary in enumerate(boundaries)
            }
            for entity_idx, boundary in enumerate(boundaries):
                span_idx[flat_idx, entity_idx] = torch.tensor(
                    boundary,
                    dtype=torch.long,
                )
                span_mask[flat_idx, entity_idx] = True

            for pair_idx, (pair, relation_ids) in enumerate(pairs):
                source = (pair[0], pair[1])
                target = (pair[2], pair[3])
                source_entity = boundary_to_id[source]
                target_entity = boundary_to_id[target]
                assignment_labels[
                    flat_idx,
                    pair_idx,
                    source_entity,
                    0,
                ] = 1.0
                assignment_labels[
                    flat_idx,
                    pair_idx,
                    target_entity,
                    1,
                ] = 1.0
                for relation_id in relation_ids:
                    relation_labels[
                        flat_idx, pair_idx, relation_id
                    ] = 1.0
                    for start, end in (source, target):
                        entity_labels[
                            flat_idx, start, relation_id, 0
                        ] = 1.0
                        entity_labels[
                            flat_idx, end, relation_id, 1
                        ] = 1.0
                        entity_labels[
                            flat_idx, start:end + 1, relation_id, 2
                        ] = 1.0

        return {
            "open_rel_entity_labels": entity_labels,
            "open_rel_labels": relation_labels,
            "open_rel_assignment_labels": assignment_labels,
            "open_rel_span_idx": span_idx,
            "open_rel_span_mask": span_mask,
            "open_rel_mask": group_mask,
            "open_rel_batch_idx": batch_indices,
            "open_rel_count": pair_counts,
        }

    def create_span_labels(self, *args, **kwargs):
        """Disable the retired direct-span target path.

        Entity-first Open Relex already returns its entity spans and endpoint
        assignment targets from :meth:`create_labels`. The inherited NER
        helper consumes extraction-class tensors and has a different call
        contract from the unified Open Relex routing.
        """

        return None

    def prepare_label_encoder_inputs(self, classes_mapping, labels_tokenizer):
        if labels_tokenizer is None:
            return None
        if classes_mapping.total_open_relex_groups() == 0:
            return None

        label_strings = []
        group_sizes = []
        for _, _, _, item_mapping in (
            classes_mapping.flat_open_relex_iter()
        ):
            labels = list(
                item_mapping.rel_class_to_id.class_to_id.keys()
            )
            group_sizes.append(len(labels))
            label_strings.extend(labels)
        if not label_strings:
            return None

        tokenized = labels_tokenizer(
            label_strings,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            add_special_tokens=True,
        )
        return {
            "open_rel_labels_input_ids": tokenized["input_ids"],
            "open_rel_labels_attention_mask": tokenized[
                "attention_mask"
            ],
            "open_rel_labels_group_size": torch.LongTensor(
                group_sizes
            ),
        }


__all__ = ["OpenRelexProcessor"]
