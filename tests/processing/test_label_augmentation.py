import pickle
from types import SimpleNamespace

import pytest

from glinext.processing.label_augmentation import (
    AugmentableLabelGroup,
    BatchLabelAugmenter,
    LabelAugmentationConfig,
)


def _group(task, batch_idx, labels, positives=(), parent=None, group_idx=0):
    mapping = SimpleNamespace(
        class_to_id={label: index for index, label in enumerate(labels)}
    )
    return AugmentableLabelGroup(
        task=task,
        batch_idx=batch_idx,
        group_idx=group_idx,
        mapping=mapping,
        positive_labels=positives,
        parent_name=parent,
    )


def _augmenter(task="classification", **policy):
    return BatchLabelAugmenter({
        "enabled": True,
        "seed": 17,
        "tasks": {task: policy},
    })


def test_disabled_configuration_is_a_strict_identity_path():
    group = _group("classification", 0, ["a", "b"])
    original_mapping = group.mapping.class_to_id

    stats = BatchLabelAugmenter(None).augment([group])

    assert stats == {}
    assert group.mapping.class_to_id is original_mapping


def test_explicit_tasks_are_an_allowlist_with_per_task_overrides():
    config = LabelAugmentationConfig.from_value({
        "enabled": True,
        "defaults": {
            "enabled": True,
            "shuffle_probability": 0.25,
            "add_probability": 0.5,
        },
        "tasks": {
            "ner": {"add_probability": 1.0},
            "joint_relex": {"enabled": False},
        },
    })

    assert config.policy_for("ner").shuffle_probability == 0.25
    assert config.policy_for("ner").add_probability == 1.0
    assert config.policy_for("joint_relex").enabled is False
    assert config.policy_for("classification").enabled is False


def test_null_tasks_apply_defaults_to_every_supported_task():
    config = LabelAugmentationConfig.from_value({
        "enabled": True,
        "defaults": {"shuffle_probability": 1.0},
        "tasks": None,
    })

    assert config.policy_for("classification").is_active
    assert config.policy_for("audio_segmentation").is_active


def test_resolved_configuration_is_picklable_for_dataloader_workers():
    config = LabelAugmentationConfig.from_value({
        "enabled": True,
        "tasks": {"classification": {"add_probability": 1.0}},
    })

    restored = pickle.loads(pickle.dumps(config))

    assert restored.policy_for("classification").add_probability == 1.0


@pytest.mark.parametrize(
    "value",
    [
        {"enabled": True, "seed": None},
        LabelAugmentationConfig(enabled=True, seed=None),
    ],
)
def test_null_seed_inherits_the_training_seed(value):
    config = LabelAugmentationConfig.from_value(value, default_seed=123)

    assert config.seed == 123


@pytest.mark.parametrize(
    "value, message",
    [
        ({"enabled": True, "unknown": 1}, "Unknown"),
        ({"enabled": True, "tasks": {"count": {}}}, "Unsupported"),
        (
            {"enabled": True, "tasks": {"ner": {"add_probability": 2}}},
            "between 0 and 1",
        ),
        (
            {"enabled": True, "tasks": {"ner": {"pool_scope": "batch"}}},
            "task_and_parent",
        ),
    ],
)
def test_configuration_rejects_invalid_values(value, message):
    with pytest.raises((TypeError, ValueError), match=message):
        LabelAugmentationConfig.from_value(value)


def test_add_uses_other_physical_examples_and_never_same_item_groups():
    first = _group("classification", 0, ["a"])
    same_item = _group(
        "classification", 0, ["same-item"], group_idx=1
    )
    donor = _group("classification", 1, ["donor"])

    stats = _augmenter(add_probability=1.0).augment(
        [first, same_item, donor],
        batch_ids=[10, 11],
    )

    assert list(first.mapping.class_to_id) == ["a", "donor"]
    assert "same-item" not in first.mapping.class_to_id
    assert stats["classification"]["added_labels"] == 4


def test_donor_pools_do_not_cross_tasks():
    classification = _group("classification", 0, ["topic"])
    entity = _group("ner", 1, ["person"])
    augmenter = BatchLabelAugmenter({
        "enabled": True,
        "tasks": {
            "classification": {"add_probability": 1.0},
            "ner": {"add_probability": 1.0},
        },
    })

    augmenter.augment([classification, entity])

    assert list(classification.mapping.class_to_id) == ["topic"]
    assert list(entity.mapping.class_to_id) == ["person"]


def test_task_and_parent_scope_filters_donors():
    target = _group("structuring", 0, ["name"], parent="person")
    compatible = _group("structuring", 1, ["age"], parent="person")
    incompatible = _group("structuring", 2, ["price"], parent="product")

    _augmenter(
        "structuring",
        add_probability=1.0,
        pool_scope="task_and_parent",
    ).augment([target, compatible, incompatible])

    assert list(target.mapping.class_to_id) == ["name", "age"]


def test_drop_preserves_positives_and_minimum_group_size():
    group = _group(
        "classification",
        0,
        ["positive", "negative"],
        positives=["positive"],
    )

    stats = _augmenter(
        drop_probability=1.0,
        preserve_positive_labels=True,
        min_labels_per_group=1,
    ).augment([group])

    assert list(group.mapping.class_to_id) == ["positive"]
    assert stats["classification"]["dropped_labels"] == 1


def test_addition_is_capped_and_excludes_target_positive_labels():
    target = _group(
        "ner",
        0,
        ["person"],
        positives=["organization"],
    )
    donor = _group(
        "ner",
        1,
        ["organization", "location", "event"],
    )

    _augmenter(
        "ner",
        add_probability=1.0,
        max_added_labels=1,
    ).augment([target, donor], batch_ids=[2, 3])

    labels = list(target.mapping.class_to_id)
    assert len(labels) == 2
    assert "organization" not in labels


def test_seeded_augmentation_is_reproducible():
    def run():
        target = _group("classification", 0, ["a", "b", "c"])
        donor = _group("classification", 1, ["d", "e", "f"])
        _augmenter(
            add_probability=0.5,
            drop_probability=0.5,
            shuffle_probability=1.0,
        ).augment([target, donor], batch_ids=[101, 202])
        return list(target.mapping.class_to_id)

    assert run() == run()


def test_task_adapter_callback_controls_the_final_mapping_order():
    applied = []
    mapping = SimpleNamespace(class_to_id={"a": 0})
    group = AugmentableLabelGroup(
        task="structuring",
        batch_idx=0,
        group_idx=0,
        mapping=mapping,
        apply_labels=lambda labels: (
            applied.append(list(labels)),
            setattr(mapping, "class_to_id", {
                label: index for index, label in enumerate(reversed(labels))
            }),
        ),
    )
    donor = _group("structuring", 1, ["b"])

    _augmenter("structuring", add_probability=1.0).augment([group, donor])

    assert applied == [["a", "b"]]
    assert list(mapping.class_to_id) == ["b", "a"]


def test_batch_ids_must_cover_every_referenced_physical_batch_item():
    groups = [
        _group("classification", 0, ["a"]),
        _group("classification", 1, ["b"]),
    ]
    with pytest.raises(ValueError, match="every referenced physical batch item"):
        _augmenter(add_probability=1.0).augment(groups, batch_ids=[1])


def test_batch_ids_may_include_items_without_augmentable_groups():
    group = _group("classification", 0, ["a"])

    stats = _augmenter(shuffle_probability=1.0).augment(
        [group],
        batch_ids=[1, 2, 3],
    )

    assert stats["classification"]["groups"] == 1
