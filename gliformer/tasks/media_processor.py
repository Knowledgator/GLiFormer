"""Shared class-mapping and prompt mechanics for media task processors."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from ..processing.label_augmentation import AugmentableLabelGroup
from ..processing.mappings import (
    BaseClassMapping,
    BatchClassesMapping,
    VisionClassMapping,
    VisionItemMapping,
)
from . import TaskProcessor


def unique_labels(items: Iterable[Any]) -> list[str]:
    """Return non-null string labels in stable first-seen order."""

    return list(dict.fromkeys(str(item) for item in items if item is not None))


class MediaTaskProcessor(TaskProcessor):
    """Reusable schema handling for vision and audio task processors.

    Media tasks share the same grouped-label mapping, prompt, classification
    target, and labels-encoder contracts. Concrete processors only decide how
    groups and positive labels are obtained from their modality annotations.
    """

    skip_item_flag: str = ""

    def __init__(self, config, task_name: str, **kwargs):
        super().__init__(config, **kwargs)
        self.task_name = task_name
        self.obj_token = config.obj_token
        self.parent_token = config.parent_token
        self.sep_token = config.sep_token

    def _task_label_groups(
        self,
        item: dict[str, Any],
    ) -> list[dict[str, Any]] | None:
        """Normalize a task payload to a list of label-group dictionaries."""

        groups = item.get(self.task_name)
        if groups is None:
            return None
        if isinstance(groups, dict):
            groups = [groups]
        elif isinstance(groups, list | tuple) and (
            not groups or not isinstance(groups[0], dict)
        ):
            groups = [{"all_labels": list(groups)}]
        return list(groups)

    @staticmethod
    def _groups_for_task(
        item: dict[str, Any],
        task_name: str,
    ) -> list[dict[str, Any]] | None:
        """Return explicit dictionary groups for another media task, if any."""

        groups = item.get(task_name)
        if groups is None:
            return None
        if isinstance(groups, dict):
            return [groups]
        if isinstance(groups, list | tuple) and groups and isinstance(groups[0], dict):
            return list(groups)
        return None

    @classmethod
    def _instances(cls, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Return modality instances used to infer labels from an item."""

        return []

    @classmethod
    def labels_from_item(cls, item: dict[str, Any]) -> list[str]:
        labels = item.get("labels") or item.get("classes") or item.get("all_labels")
        if labels is not None:
            return unique_labels(labels)
        return unique_labels(instance.get("label") for instance in cls._instances(item))

    @classmethod
    def true_labels_from_item(cls, item: dict[str, Any]) -> list[str]:
        labels = item.get("true_labels")
        if labels is not None:
            return unique_labels(labels)
        return unique_labels(instance.get("label") for instance in cls._instances(item))

    def _fallback_label_group(self, item: dict[str, Any]) -> dict[str, Any]:
        group = {
            "name": item.get("name", self.task_name),
            "all_labels": self.labels_from_item(item),
        }
        true_labels = self.true_labels_from_item(item)
        if true_labels:
            group["true_labels"] = true_labels
        return group

    def _label_groups_for_item(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        groups = self._task_label_groups(item)
        return groups if groups is not None else [self._fallback_label_group(item)]

    def _mapping_groups(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Return groups used to build the per-item class mapping."""

        return self._label_groups_for_item(item)

    def _mapping_labels(self, group: dict[str, Any]) -> list[str]:
        labels = group.get("all_labels") or group.get("labels") or group.get("classes")
        return unique_labels(labels or [])

    def _mapping_group_entries(
        self,
        item: dict[str, Any],
    ) -> list[tuple[int, dict[str, Any]]]:
        """Return source-indexed groups that actually produce mappings.

        Explicit media payloads may contain empty schema groups.  Mapping
        creation skips those groups, so all later mapping-indexed operations
        must use the same compacted view before looking up source annotations.
        """

        return [
            (source_idx, group)
            for source_idx, group in enumerate(self._mapping_groups(item))
            if self._mapping_labels(group)
        ]

    def _mapping_group_entry(
        self,
        item: dict[str, Any],
        group_idx: int,
    ) -> tuple[int, dict[str, Any]]:
        entries = self._mapping_group_entries(item)
        if group_idx < len(entries):
            return entries[group_idx]
        return group_idx, {}

    def get_classes_mapping(self, batch_list, **kwargs):
        mappings = []
        for item in batch_list:
            if self.skip_item_flag and item.get(self.skip_item_flag):
                mappings.append(VisionClassMapping())
                continue

            item_mappings = []
            for _, group in self._mapping_group_entries(item):
                labels = self._mapping_labels(group)
                name = group.get("name", item.get("name", self.task_name))
                item_mappings.append(
                    VisionItemMapping(
                        class_to_id=BaseClassMapping(
                            class_to_id={label: idx for idx, label in enumerate(labels)},
                            name=name,
                        ),
                        name=name,
                    )
                )

            mappings.append(VisionClassMapping(items=item_mappings))
        return mappings

    def get_augmentable_label_groups(self, batch_list, classes_mapping):
        groups = []
        mapping_list = getattr(
            classes_mapping,
            f"{self.task_name}_mapping",
        )
        classification_tasks = {
            "image_classification",
            "audio_classification",
        }
        for batch_idx, item_mapping in enumerate(mapping_list):
            item = batch_list[batch_idx]
            source_groups = self._mapping_group_entries(item)
            for group_idx, mapping_item in enumerate(item_mapping.items):
                mapping = mapping_item.class_to_id
                if not mapping.class_to_id:
                    continue
                if group_idx < len(source_groups):
                    source_idx, source_group = source_groups[group_idx]
                else:
                    source_idx, source_group = group_idx, {}

                if self.task_name in classification_tasks:
                    positives = self.true_labels_from_item(source_group)
                elif hasattr(self, "_group_for_mapping"):
                    positives = self.true_labels_from_item(
                        self._group_for_mapping(item, source_idx)
                    )
                else:
                    positives = self.true_labels_from_item(source_group)
                positives = unique_labels(positives)

                groups.append(AugmentableLabelGroup(
                    task=self.task_name,
                    batch_idx=batch_idx,
                    group_idx=group_idx,
                    mapping=mapping,
                    positive_labels=positives,
                    parent_name=mapping.name,
                ))
        return groups

    def contribute_prompt(
        self,
        classes_mapping: BatchClassesMapping,
        batch_idx,
        use_labels_encoder=False,
    ):
        mapping_list = getattr(classes_mapping, f"{self.task_name}_mapping")
        if batch_idx >= len(mapping_list):
            return []

        prompt = []
        for mapping in mapping_list[batch_idx].items:
            prompt.append(self.parent_token)
            if mapping.name:
                prompt.append(mapping.name)
            if not use_labels_encoder:
                prompt.extend(
                    f"{self.obj_token} {label}"
                    for label in mapping.class_to_id.class_to_id
                )
            prompt.append(self.sep_token)
        return prompt

    def _mapping_iter(self, classes_mapping):
        return getattr(classes_mapping, f"flat_{self.task_name}_iter")()

    def _total_groups(self, classes_mapping):
        return getattr(classes_mapping, f"total_{self.task_name}_groups")()

    def _classification_true_labels(
        self,
        item: dict[str, Any],
        group_idx: int,
    ) -> Iterable[Any]:
        """Return positive labels for one classification group."""

        groups = self._label_groups_for_item(item)
        group = groups[group_idx] if group_idx < len(groups) else item
        return self.true_labels_from_item(group)

    def _create_classification_labels(
        self,
        batch_list,
        classes_mapping,
        output_key: str,
    ):
        total = self._total_groups(classes_mapping)
        if total == 0:
            return None

        max_classes = max(
            len(mapping.class_to_id.class_to_id)
            for _, _, _, mapping in self._mapping_iter(classes_mapping)
        )
        labels = torch.zeros(total, max_classes, dtype=torch.float)
        for flat_idx, batch_idx, group_idx, mapping in self._mapping_iter(classes_mapping):
            label_to_id = mapping.class_to_id.class_to_id
            _, source_group = self._mapping_group_entry(
                batch_list[batch_idx],
                group_idx,
            )
            positives = unique_labels(
                self.true_labels_from_item(source_group)
            )
            for label in positives:
                if label in label_to_id:
                    labels[flat_idx, label_to_id[label]] = 1.0
        return {output_key: labels}

    def prepare_label_encoder_inputs(self, classes_mapping, labels_tokenizer):
        if labels_tokenizer is None:
            return None

        all_labels = []
        group_sizes = []
        for _, _, _, mapping in self._mapping_iter(classes_mapping):
            labels = list(mapping.class_to_id.class_to_id)
            all_labels.extend(labels)
            group_sizes.append(len(labels))
        if not all_labels:
            return None

        tokenized = labels_tokenizer(
            all_labels,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            add_special_tokens=True,
        )
        return {
            f"{self.task_name}_labels_input_ids": tokenized["input_ids"],
            f"{self.task_name}_labels_attention_mask": tokenized["attention_mask"],
            f"{self.task_name}_labels_group_size": torch.LongTensor(group_sizes),
        }


__all__ = ["MediaTaskProcessor", "unique_labels"]
