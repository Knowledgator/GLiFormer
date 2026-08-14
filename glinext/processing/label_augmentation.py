"""Training-only augmentation of runtime label schemas.

The augmenter operates on the ephemeral ``BaseClassMapping`` objects created
for one collator microbatch.  Source annotations are intentionally left
untouched: task processors continue to be the single source of truth for gold
targets, added labels therefore receive zero targets, and removed labels are
treated as schema-level supervision dropout.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

LABEL_AUGMENTATION_MARKER_KEY = "_glinext_label_augmentation"
LABEL_AUGMENTATION_INDEX_KEY = "_glinext_label_augmentation_index"

SUPPORTED_LABEL_AUGMENTATION_TASKS = (
    "classification",
    "ner",
    "joint_relex",
    "open_relex",
    "set_open_relex",
    "structuring",
    "set_structuring",
    "image_classification",
    "object_detection",
    "segmentation",
    "audio_classification",
    "audio_segmentation",
)

_ROOT_KEYS = frozenset({"enabled", "defaults", "tasks", "seed"})
_POLICY_KEYS = frozenset({
    "enabled",
    "shuffle_probability",
    "drop_probability",
    "add_probability",
    "preserve_positive_labels",
    "min_labels_per_group",
    "max_added_labels",
    "pool_scope",
    "seed",
})


def _as_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")


def _validate_unknown_keys(values: Mapping[str, Any], allowed, *, name: str):
    unknown = set(values).difference(allowed)
    if unknown:
        formatted = ", ".join(sorted(map(str, unknown)))
        raise ValueError(f"Unknown {name} option(s): {formatted}")


def _probability(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number between 0 and 1")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a number between 0 and 1") from exc
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return normalized


def _non_negative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _optional_seed(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer or null")
    return value


@dataclass(frozen=True)
class LabelAugmentationPolicy:
    """Resolved policy for one task-specific label namespace."""

    enabled: bool = True
    shuffle_probability: float = 0.0
    drop_probability: float = 0.0
    add_probability: float = 0.0
    preserve_positive_labels: bool = True
    min_labels_per_group: int = 1
    max_added_labels: int = 16
    pool_scope: str = "task"
    seed: int | None = None

    @classmethod
    def from_value(
        cls,
        value: Any = None,
        *,
        base: LabelAugmentationPolicy | None = None,
        name: str = "label augmentation policy",
    ) -> LabelAugmentationPolicy:
        if isinstance(value, cls) and base is None:
            return value
        values = _as_mapping(value, name=name)
        _validate_unknown_keys(values, _POLICY_KEYS, name=name)

        inherited = base or cls()
        resolved = {
            key: values.get(key, getattr(inherited, key))
            for key in _POLICY_KEYS
        }
        if not isinstance(resolved["enabled"], bool):
            raise TypeError(f"{name}.enabled must be a boolean")
        if not isinstance(resolved["preserve_positive_labels"], bool):
            raise TypeError(
                f"{name}.preserve_positive_labels must be a boolean"
            )
        for probability_name in (
            "shuffle_probability",
            "drop_probability",
            "add_probability",
        ):
            resolved[probability_name] = _probability(
                resolved[probability_name],
                name=f"{name}.{probability_name}",
            )
        resolved["min_labels_per_group"] = _non_negative_integer(
            resolved["min_labels_per_group"],
            name=f"{name}.min_labels_per_group",
        )
        resolved["max_added_labels"] = _non_negative_integer(
            resolved["max_added_labels"],
            name=f"{name}.max_added_labels",
        )
        if resolved["pool_scope"] not in {"task", "task_and_parent"}:
            raise ValueError(
                f"{name}.pool_scope must be 'task' or 'task_and_parent'"
            )
        resolved["seed"] = _optional_seed(
            resolved["seed"],
            name=f"{name}.seed",
        )
        return cls(**resolved)

    @property
    def is_active(self) -> bool:
        return self.enabled and any((
            self.shuffle_probability > 0.0,
            self.drop_probability > 0.0,
            self.add_probability > 0.0 and self.max_added_labels > 0,
        ))


_DISABLED_POLICY = LabelAugmentationPolicy(enabled=False)


@dataclass(frozen=True)
class LabelAugmentationConfig:
    """Validated global configuration with resolved per-task policies.

    When an explicit ``tasks`` mapping is supplied it is an allowlist.  When
    ``tasks`` is omitted or null, the resolved defaults apply to every
    supported task.
    """

    enabled: bool = False
    defaults: LabelAugmentationPolicy = field(
        default_factory=LabelAugmentationPolicy
    )
    tasks: Mapping[str, LabelAugmentationPolicy] = field(
        default_factory=dict
    )
    seed: int | None = None

    @classmethod
    def from_value(
        cls,
        value: Any = None,
        *,
        default_seed: int | None = None,
    ) -> LabelAugmentationConfig:
        if isinstance(value, cls):
            if value.seed is None and default_seed is not None:
                return replace(
                    value,
                    seed=_optional_seed(default_seed, name="default_seed"),
                )
            return value
        if value is None or value is False:
            return cls(seed=_optional_seed(default_seed, name="default_seed"))
        if value is True:
            values: dict[str, Any] = {"enabled": True}
        else:
            values = _as_mapping(value, name="label_augmentation")
        _validate_unknown_keys(
            values,
            _ROOT_KEYS,
            name="label_augmentation",
        )

        enabled = values.get("enabled", False)
        if not isinstance(enabled, bool):
            raise TypeError("label_augmentation.enabled must be a boolean")
        configured_seed = values.get("seed")
        seed = _optional_seed(
            default_seed if configured_seed is None else configured_seed,
            name="label_augmentation.seed",
        )
        defaults = LabelAugmentationPolicy.from_value(
            values.get("defaults"),
            name="label_augmentation.defaults",
        )

        raw_tasks = values.get("tasks", None)
        if raw_tasks is None:
            task_values = {
                task: defaults for task in SUPPORTED_LABEL_AUGMENTATION_TASKS
            }
        else:
            task_mapping = _as_mapping(
                raw_tasks,
                name="label_augmentation.tasks",
            )
            unknown_tasks = set(task_mapping).difference(
                SUPPORTED_LABEL_AUGMENTATION_TASKS
            )
            if unknown_tasks:
                formatted = ", ".join(sorted(map(str, unknown_tasks)))
                raise ValueError(
                    "Unsupported label augmentation task(s): " + formatted
                )
            task_values = {
                task: LabelAugmentationPolicy.from_value(
                    policy,
                    base=defaults,
                    name=f"label_augmentation.tasks.{task}",
                )
                for task, policy in task_mapping.items()
            }

        return cls(
            enabled=enabled,
            defaults=defaults,
            tasks=dict(task_values),
            seed=seed,
        )

    def policy_for(self, task_name: str) -> LabelAugmentationPolicy:
        if not self.enabled:
            return _DISABLED_POLICY
        return self.tasks.get(task_name, _DISABLED_POLICY)

    @property
    def is_active(self) -> bool:
        return self.enabled and any(
            policy.is_active for policy in self.tasks.values()
        )


@dataclass
class AugmentableLabelGroup:
    """A mutable mapping reference plus immutable task/gold metadata."""

    task: str
    batch_idx: int
    group_idx: int
    mapping: Any
    positive_labels: Sequence[str] = ()
    parent_name: str | None = None
    apply_labels: Callable[[list[str]], None] | None = None

    def __post_init__(self):
        if self.task not in SUPPORTED_LABEL_AUGMENTATION_TASKS:
            raise ValueError(f"Unsupported label augmentation task: {self.task}")
        if (
            isinstance(self.batch_idx, bool)
            or not isinstance(self.batch_idx, int)
            or self.batch_idx < 0
        ):
            raise ValueError("batch_idx must be a non-negative integer")
        if (
            isinstance(self.group_idx, bool)
            or not isinstance(self.group_idx, int)
            or self.group_idx < 0
        ):
            raise ValueError("group_idx must be a non-negative integer")
        if not hasattr(self.mapping, "class_to_id"):
            raise TypeError("mapping must expose class_to_id")
        positive_labels = (
            [self.positive_labels]
            if isinstance(self.positive_labels, str)
            else self.positive_labels
        )
        self.positive_labels = frozenset(positive_labels)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.mapping.class_to_id)

    def replace_labels(self, labels: Sequence[str]) -> None:
        normalized = list(dict.fromkeys(labels))
        if self.apply_labels is not None:
            self.apply_labels(normalized)
        else:
            self.mapping.class_to_id = {
                label: index for index, label in enumerate(normalized)
            }


def _stable_seed(seed: int | None, task: str, batch_ids: Sequence[Any]) -> int | None:
    if seed is None:
        return None
    payload = json.dumps(
        [seed, task, list(batch_ids)],
        ensure_ascii=False,
        sort_keys=True,
        default=repr,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=16).digest(),
        byteorder="big",
    )


def _empty_stats() -> dict[str, int]:
    return {
        "groups": 0,
        "original_labels": 0,
        "dropped_labels": 0,
        "added_labels": 0,
        "shuffled_groups": 0,
        "empty_donor_groups": 0,
    }


class BatchLabelAugmenter:
    """Apply resolved policies to label groups from one local microbatch."""

    def __init__(self, config: LabelAugmentationConfig | Mapping | None):
        self.config = LabelAugmentationConfig.from_value(config)

    def augment(
        self,
        groups: Sequence[AugmentableLabelGroup],
        *,
        batch_ids: Sequence[Any] | None = None,
    ) -> dict[str, dict[str, int]]:
        if not self.config.is_active:
            return {}

        groups = list(groups)
        if not groups:
            return {}
        minimum_batch_size = max(group.batch_idx for group in groups) + 1
        if batch_ids is None:
            batch_ids = list(range(minimum_batch_size))
        else:
            batch_ids = list(batch_ids)
            if len(batch_ids) < minimum_batch_size:
                raise ValueError(
                    "label augmentation batch_ids must contain one value "
                    "for every referenced physical batch item"
                )

        # All donor pools come from this immutable snapshot. Newly added labels
        # can never cascade into another group in the same batch.
        snapshots = [(group, group.labels) for group in groups]
        task_groups: dict[str, list[tuple[AugmentableLabelGroup, tuple[str, ...]]]] = {}
        for group, labels in snapshots:
            task_groups.setdefault(group.task, []).append((group, labels))

        all_stats: dict[str, dict[str, int]] = {}
        for task, task_snapshots in task_groups.items():
            policy = self.config.policy_for(task)
            if not policy.is_active:
                continue
            rng = random.Random(_stable_seed(
                policy.seed if policy.seed is not None else self.config.seed,
                task,
                batch_ids,
            ))
            stats = _empty_stats()
            all_stats[task] = stats

            for group, original_tuple in task_snapshots:
                original = list(original_tuple)
                if not original:
                    continue
                stats["groups"] += 1
                stats["original_labels"] += len(original)
                positive_set = set(group.positive_labels)

                retained = []
                dropped = []
                for label in original:
                    protected = (
                        policy.preserve_positive_labels
                        and label in positive_set
                    )
                    if (
                        not protected
                        and policy.drop_probability > 0.0
                        and rng.random() < policy.drop_probability
                    ):
                        dropped.append(label)
                    else:
                        retained.append(label)

                donor_candidates = []
                seen = set(original)
                seen.update(positive_set)
                for donor_group, donor_labels in task_snapshots:
                    if donor_group.batch_idx == group.batch_idx:
                        continue
                    if (
                        policy.pool_scope == "task_and_parent"
                        and donor_group.parent_name != group.parent_name
                    ):
                        continue
                    for label in donor_labels:
                        if label in seen:
                            continue
                        seen.add(label)
                        donor_candidates.append(label)

                added = []
                if policy.add_probability > 0.0 and policy.max_added_labels > 0:
                    passing = [
                        label for label in donor_candidates
                        if rng.random() < policy.add_probability
                    ]
                    if len(passing) > policy.max_added_labels:
                        chosen = set(rng.sample(
                            range(len(passing)),
                            policy.max_added_labels,
                        ))
                        passing = [
                            label for index, label in enumerate(passing)
                            if index in chosen
                        ]
                    added = passing
                    if not donor_candidates:
                        stats["empty_donor_groups"] += 1

                final = retained + added
                minimum = min(policy.min_labels_per_group, len(original))
                if len(final) < minimum:
                    restore_count = minimum - len(final)
                    restored = dropped[:restore_count]
                    final.extend(restored)
                    restored_set = set(restored)
                    dropped = [
                        label for label in dropped if label not in restored_set
                    ]

                shuffled = False
                if (
                    len(final) > 1
                    and policy.shuffle_probability > 0.0
                    and rng.random() < policy.shuffle_probability
                ):
                    rng.shuffle(final)
                    shuffled = True

                stats["dropped_labels"] += len(dropped)
                stats["added_labels"] += len(added)
                stats["shuffled_groups"] += int(shuffled)
                if final != original:
                    group.replace_labels(final)

        return all_stats


__all__ = [
    "LABEL_AUGMENTATION_INDEX_KEY",
    "LABEL_AUGMENTATION_MARKER_KEY",
    "SUPPORTED_LABEL_AUGMENTATION_TASKS",
    "AugmentableLabelGroup",
    "BatchLabelAugmenter",
    "LabelAugmentationConfig",
    "LabelAugmentationPolicy",
]
