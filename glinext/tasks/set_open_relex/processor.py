"""Processor for independent entity-first set open relation extraction."""

import random

import torch

from ...processing.mappings import (
    BaseClassMapping,
    OpenRelexClassMapping,
    OpenRelexItemMapping,
)
from ..span_processor import SpanProcessor


class SetOpenRelexProcessor(SpanProcessor):
    """Build entity, pair, and assignment targets for set open relex.

    The task first recognizes relation-conditioned endpoint entities.  Its
    second stage treats every unique directed endpoint pair as one gold set
    element, independently of the number of predicted anchor queries.
    """

    data_key = "set_open_relex"
    legacy_data_key = "open_relex"

    def __init__(self, config, tokenizer=None, words_splitter=None, **kwargs):
        super().__init__(
            config,
            tokenizer,
            words_splitter,
            parent_token=getattr(config, "open_rel_parent_token", None),
            **kwargs,
        )
        self.rel_token = config.rel_token
        # Before set-open-relex became an independent task, its annotations
        # used ``open_relex``.  Reuse that spelling only when a normal open
        # head is not active, otherwise the two tasks must remain isolated.
        self.allow_legacy_data = (
            getattr(config, "open_relex_config", None) is None
        )

    def _groups(self, item):
        if self.data_key in item:
            return item.get(self.data_key) or []
        if self.allow_legacy_data:
            return item.get(self.legacy_data_key) or []
        return []

    @staticmethod
    def _relation_name(relation):
        return relation.get(
            "relation",
            relation.get("label", relation.get("type", "")),
        )

    @staticmethod
    def _endpoint_entry(relation, role):
        keys = ("source", "head") if role == 0 else ("target", "tail")
        for key in keys:
            if key in relation:
                return key, relation.get(key)
        return keys[0], None

    def get_classes_mapping(self, batch_list, shuffle_labels=False, **kwargs):
        mappings = []
        for item in batch_list:
            item_mappings = []
            for group in self._groups(item):
                if "all_labels" in group:
                    relation_types = list(dict.fromkeys(group["all_labels"]))
                else:
                    relation_types = []
                    seen = set()
                    for relation in group.get("relations", []):
                        relation_type = self._relation_name(relation)
                        if relation_type and relation_type not in seen:
                            relation_types.append(relation_type)
                            seen.add(relation_type)
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
                        ),
                        name=group.get("name"),
                    )
                )
            mappings.append(OpenRelexClassMapping(items=item_mappings))
        return mappings

    def contribute_prompt(
        self,
        classes_mapping,
        batch_idx,
        use_labels_encoder=False,
    ):
        mappings = getattr(classes_mapping, "set_open_relex_mapping", None)
        if mappings is None or batch_idx >= len(mappings):
            return []

        prompt = []
        for item_mapping in mappings[batch_idx].items:
            prompt.append(self.parent_token)
            relation_mapping = item_mapping.rel_class_to_id
            if relation_mapping.name:
                prompt.append(relation_mapping.name)
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
        set_relations=None,
        **kwargs,
    ):
        requested_relations = (
            set_relations
            if set_relations is not None
            else (relations if self.allow_legacy_data else None)
        )
        relation_groups = self._normalize_label_groups(requested_relations)
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
        set_relations=None,
        **kwargs,
    ):
        requested_relations = (
            set_relations
            if set_relations is not None
            else (relations if self.allow_legacy_data else None)
        )
        if requested_relations is None:
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
        if item.get("_glinext_set_open_relex_spans_resolved"):
            return
        groups = self._groups(item)
        if not groups:
            return

        text = item.get("text", "")
        tokens_with_spans, _ = self._tokenize_text(item)
        if tokens_with_spans is None:
            return

        for group in groups:
            for relation in group.get("relations", []):
                for role in (0, 1):
                    key, value = self._endpoint_entry(relation, role)
                    if value is not None:
                        relation[key] = self._normalize_endpoint(
                            text,
                            tokens_with_spans,
                            value,
                        )
        item["_glinext_set_open_relex_spans_resolved"] = True

    @staticmethod
    def _resolved_span(relation, role):
        _, endpoint = SetOpenRelexProcessor._endpoint_entry(relation, role)
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
            source_span = self._resolved_span(relation, 0)
            target_span = self._resolved_span(relation, 1)
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
        """Create entity-first set targets without query-capacity padding.

        Shapes use ``BN`` flattened groups, ``G`` unique directed gold pairs,
        ``R`` relation classes, and ``E`` unique endpoint boundaries.
        """
        max_seq_len = int(max_seq_len)
        if max_seq_len <= 0:
            return None
        for item in batch_list:
            self.resolve_spans(item)

        total_groups = classes_mapping.total_set_open_relex_groups()
        if total_groups == 0:
            return None

        group_targets = []
        max_pairs = 0
        max_relations = 0
        max_entities = 0
        for _, batch_idx, group_idx, item_mapping in (
            classes_mapping.flat_set_open_relex_iter()
        ):
            groups = self._groups(batch_list[batch_idx])
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
            2,
            dtype=torch.float,
        )
        assignment_labels = torch.zeros(
            total_groups,
            max_pairs,
            max_relations,
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
            classes_mapping.flat_set_open_relex_iter()
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
                for relation_id in relation_ids:
                    relation_labels[
                        flat_idx, pair_idx, relation_id, :
                    ] = 1.0
                    assignment_labels[
                        flat_idx,
                        pair_idx,
                        relation_id,
                        source_entity,
                        0,
                    ] = 1.0
                    assignment_labels[
                        flat_idx,
                        pair_idx,
                        relation_id,
                        target_entity,
                        1,
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
            "set_open_rel_entity_labels": entity_labels,
            "set_open_rel_labels": relation_labels,
            "set_open_rel_assignment_labels": assignment_labels,
            "set_open_rel_span_idx": span_idx,
            "set_open_rel_span_mask": span_mask,
            "set_open_rel_mask": group_mask,
            "set_open_rel_batch_idx": batch_indices,
            "set_open_rel_count": pair_counts,
        }

    def prepare_label_encoder_inputs(self, classes_mapping, labels_tokenizer):
        if labels_tokenizer is None:
            return None
        if classes_mapping.total_set_open_relex_groups() == 0:
            return None

        label_strings = []
        group_sizes = []
        for _, _, _, item_mapping in (
            classes_mapping.flat_set_open_relex_iter()
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
            "set_open_rel_labels_input_ids": tokenized["input_ids"],
            "set_open_rel_labels_attention_mask": tokenized[
                "attention_mask"
            ],
            "set_open_rel_labels_group_size": torch.LongTensor(
                group_sizes
            ),
        }


__all__ = ["SetOpenRelexProcessor"]
