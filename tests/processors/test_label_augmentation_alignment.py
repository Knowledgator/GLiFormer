"""Target-alignment contracts for training-time label augmentation.

These tests exercise the boundary between runtime mapping augmentation and
task label construction.  Gold annotations stay unchanged while the target
group drops its annotated label and receives two labels from another physical
batch item.  The prompt and mapping must agree, and every added class must
remain a zero/ignored target.
"""

from dataclasses import asdict

import pytest

from glinext.config import (
    AudioClassificationHeadConfig,
    AudioSegmentationHeadConfig,
    ImageClassificationHeadConfig,
    ObjectDetectionHeadConfig,
    SegmentationHeadConfig,
)
from glinext.processing.label_augmentation import BatchLabelAugmenter
from glinext.processing.mappings import (
    BatchClassesMapping,
    CatClassMapping,
    ExtractionClassMapping,
    OpenRelexClassMapping,
    StructuringClassMapping,
)
from glinext.tasks.audio.processor import AudioProcessor
from glinext.tasks.classification.processor import ClassificationProcessor
from glinext.tasks.joint_relex.processor import JointRelexProcessor
from glinext.tasks.ner.processor import NERProcessor
from glinext.tasks.open_relex.processor import OpenRelexProcessor
from glinext.tasks.structuring.processor import StructuringProcessor
from glinext.tasks.vision.processor import VisionProcessor
from tests.conftest import FakeWordsSplitter, make_config


def _classes_mapping(task, mappings):
    """Place one task's per-example mappings in a complete batch mapping."""

    batch_size = len(mappings)
    cat_mapping = (
        mappings
        if task == "classification"
        else [CatClassMapping([]) for _ in range(batch_size)]
    )
    extraction_mapping = (
        mappings
        if task in {"ner", "joint_relex"}
        else [ExtractionClassMapping() for _ in range(batch_size)]
    )
    kwargs = {}
    mapping_attributes = {
        "open_relex": "open_relex_mapping",
        "structuring": "structuring_mapping",
        "image_classification": "image_classification_mapping",
        "object_detection": "object_detection_mapping",
        "segmentation": "segmentation_mapping",
        "audio_classification": "audio_classification_mapping",
        "audio_segmentation": "audio_segmentation_mapping",
    }
    if task in mapping_attributes:
        kwargs[mapping_attributes[task]] = mappings
    if task != "structuring":
        kwargs.setdefault(
            "structuring_mapping",
            [StructuringClassMapping() for _ in range(batch_size)],
        )
    if task != "open_relex":
        kwargs.setdefault(
            "open_relex_mapping",
            [OpenRelexClassMapping() for _ in range(batch_size)],
        )
    return BatchClassesMapping(
        cat_mapping=cat_mapping,
        extraction_mapping=extraction_mapping,
        **kwargs,
    )


def _augment(
    processor,
    task,
    items,
    classes_mapping,
    *,
    target_label="target",
    donor_labels=("donor_a", "donor_b"),
):
    """Swap the first item's gold label for donor-only batch labels."""

    groups = processor.get_augmentable_label_groups(items, classes_mapping)
    target_group = next(
        group
        for group in groups
        if group.task == task and group.batch_idx == 0
    )
    stats = BatchLabelAugmenter({
        "enabled": True,
        "seed": 31,
        "tasks": {
            task: {
                "drop_probability": 1.0,
                "add_probability": 1.0,
                "shuffle_probability": 1.0,
                "preserve_positive_labels": False,
                "min_labels_per_group": 1,
            },
        },
    }).augment(groups, batch_ids=[100, 200])

    assert set(target_group.mapping.class_to_id) == set(donor_labels)
    assert target_label not in target_group.mapping.class_to_id
    assert stats[task]["dropped_labels"] >= 1
    assert stats[task]["added_labels"] >= 2
    assert stats[task]["shuffled_groups"] >= 1
    return target_group.mapping


def _assert_prompt_matches_mapping(processor, classes_mapping, mapping, token):
    prompted_labels = [
        value.removeprefix(f"{token} ")
        for value in processor.contribute_prompt(classes_mapping, 0)
        if value.startswith(f"{token} ")
    ]
    assert prompted_labels == list(mapping.class_to_id)


def test_classification_added_and_dropped_labels_stay_target_aligned():
    processor = ClassificationProcessor(
        make_config(classification_config={}),
    )
    items = [
        {
            "classification": [{
                "name": "topic",
                "all_labels": ["target"],
                "true_labels": ["target"],
            }],
        },
        {
            "classification": [{
                "name": "topic",
                "all_labels": ["donor_a", "donor_b"],
                "true_labels": ["donor_a"],
            }],
        },
    ]
    mappings = processor.get_classes_mapping(items)
    classes_mapping = _classes_mapping("classification", mappings)
    baseline = processor.create_labels(items, classes_mapping)
    assert baseline["cat_labels"][0, 0].item() == 1.0
    assert baseline["cat_labels"][0].sum().item() == 1.0

    mapping = _augment(
        processor,
        "classification",
        items,
        classes_mapping,
    )
    result = processor.create_labels(items, classes_mapping)

    assert result["cat_labels"][0].tolist() == [0.0, 0.0]
    _assert_prompt_matches_mapping(processor, classes_mapping, mapping, "[CAT]")


def test_ner_added_and_dropped_labels_stay_target_aligned():
    processor = NERProcessor(
        make_config(),
        words_splitter=FakeWordsSplitter(),
    )
    items = [
        {
            "text": "Alice",
            "extraction": [{
                "name": "entities",
                "all_labels": ["target"],
                "ner": [[0, 0, "target"]],
            }],
        },
        {
            "text": "Bob",
            "extraction": [{
                "name": "entities",
                "all_labels": ["donor_a", "donor_b"],
                "ner": [[0, 0, "donor_a"]],
            }],
        },
    ]
    mappings = processor.get_classes_mapping(items)
    classes_mapping = _classes_mapping("ner", mappings)
    baseline = processor.create_labels(
        items,
        classes_mapping,
        max_seq_len=1,
    )
    assert baseline["ner_labels"][0].any()

    mapping = _augment(processor, "ner", items, classes_mapping)
    result = processor.create_labels(items, classes_mapping, max_seq_len=1)

    assert result["ner_labels"][0].shape[-2] == 2
    assert not result["ner_labels"][0].any()
    _assert_prompt_matches_mapping(processor, classes_mapping, mapping, "[ENT]")


def _joint_item(relation_label, all_relation_labels):
    return {
        "text": "Alice Acme",
        "extraction": [{
            "name": "relations",
            "all_labels": ["entity"],
            "all_rel_labels": all_relation_labels,
            "ner": [[0, 0, "entity"], [1, 1, "entity"]],
            "relations": [[0, relation_label, 1]],
        }],
    }


def test_joint_relex_added_and_dropped_labels_stay_target_aligned():
    processor = JointRelexProcessor(
        make_config(
            default_ner_config=False,
            ner_config=None,
            joint_relex_config={},
        ),
        words_splitter=FakeWordsSplitter(),
    )
    items = [
        _joint_item("target", ["target"]),
        _joint_item("donor_a", ["donor_a", "donor_b"]),
    ]
    mappings = processor.get_classes_mapping(items)
    classes_mapping = _classes_mapping("joint_relex", mappings)
    baseline = processor.create_labels(
        items,
        classes_mapping,
        max_seq_len=2,
        add_random_negatives=False,
        add_reversed_negatives=False,
    )
    assert baseline["rel_labels"][0].any()

    mapping = _augment(processor, "joint_relex", items, classes_mapping)
    result = processor.create_labels(
        items,
        classes_mapping,
        max_seq_len=2,
        add_random_negatives=False,
        add_reversed_negatives=False,
    )

    assert result["rel_labels"][0].shape[-1] == 2
    assert not result["rel_labels"][0].any()
    _assert_prompt_matches_mapping(processor, classes_mapping, mapping, "[REL]")


def _endpoint_relation_item(task, relation_label, all_relation_labels):
    source_key, target_key = "source", "target"
    return {
        "text": "Alice Acme",
        task: [{
            "name": "relations",
            "all_labels": all_relation_labels,
            "relations": [{
                "relation": relation_label,
                source_key: {"text": "Alice", "start": 0, "end": 0},
                target_key: {"text": "Acme", "start": 1, "end": 1},
            }],
        }],
    }


def test_open_relex_keeps_augmented_class_axis_aligned():
    task = "open_relex"
    tensor_key = "open_rel_labels"
    count_key = "open_rel_count"
    processor = OpenRelexProcessor(
        make_config(default_ner_config=False, open_relex_config={}),
        words_splitter=FakeWordsSplitter(),
    )
    items = [
        _endpoint_relation_item(task, "target", ["target"]),
        _endpoint_relation_item(
            task,
            "donor_a",
            ["donor_a", "donor_b"],
        ),
    ]
    mappings = processor.get_classes_mapping(items)
    classes_mapping = _classes_mapping(task, mappings)
    baseline = processor.create_labels(
        items,
        classes_mapping,
        max_seq_len=2,
    )
    assert baseline[tensor_key][0].any()
    assert baseline[count_key][0].item() == 1

    mapping = _augment(processor, task, items, classes_mapping)
    result = processor.create_labels(
        items,
        classes_mapping,
        max_seq_len=2,
    )

    assert result[tensor_key][0].shape[1] == 2
    assert not result[tensor_key][0].any()
    assert result[count_key][0].item() == 0
    _assert_prompt_matches_mapping(processor, classes_mapping, mapping, "[REL]")


def _flat_structuring_items(task):
    return [
        {
            "text": "Alice",
            task: {
                "person": [{
                    "target": {"text": "Alice", "start": 0, "end": 0},
                }],
            },
        },
        {
            "text": "42 Paris",
            task: {
                "person": [{
                    "donor_a": {"text": "42", "start": 0, "end": 0},
                    "donor_b": {"text": "Paris", "start": 1, "end": 1},
                }],
            },
        },
    ]


def test_flat_structuring_keeps_augmented_fields_aligned():
    task = "structuring"
    tensor_key = "structuring_labels"
    processor = StructuringProcessor(
        make_config(default_ner_config=False, structuring_config={}),
        words_splitter=FakeWordsSplitter(),
    )
    items = _flat_structuring_items(task)
    mappings = processor.get_classes_mapping(items)
    classes_mapping = _classes_mapping(task, mappings)
    baseline = processor.create_labels(
        items,
        classes_mapping,
        max_seq_len=2,
    )
    assert baseline[tensor_key][0].any()

    mapping = _augment(processor, task, items, classes_mapping)
    result = processor.create_labels(
        items,
        classes_mapping,
        max_seq_len=2,
    )

    assert result[tensor_key][0].shape[-2] == 2
    assert not result[tensor_key][0].any()
    _assert_prompt_matches_mapping(processor, classes_mapping, mapping, "[CHILD]")


def test_multilevel_structuring_keeps_hierarchy_prompt_and_targets_aligned():
    processor = StructuringProcessor(
        make_config(structuring_config={"multi_level": True}),
        words_splitter=FakeWordsSplitter(),
    )
    items = [
        {
            "text": "Alice",
            "structuring": {
                "person": [{"profile": {"target": "Alice"}}],
            },
        },
        {
            "text": "42 Paris",
            "structuring": {
                "person": [{
                    "profile": {
                        "donor_a": "42",
                        "donor_b": "Paris",
                    },
                }],
            },
        },
    ]
    mappings = processor.get_classes_mapping(items)
    classes_mapping = _classes_mapping("structuring", mappings)
    baseline = processor.create_labels(
        items,
        classes_mapping,
        max_seq_len=2,
    )
    assert baseline["structuring_labels"][0].any()

    mapping = _augment(
        processor,
        "structuring",
        items,
        classes_mapping,
        target_label="profile.target",
        donor_labels=("profile.donor_a", "profile.donor_b"),
    )
    result = processor.create_labels(
        items,
        classes_mapping,
        max_seq_len=2,
    )

    assert not result["structuring_labels"][0].any()
    _assert_prompt_matches_mapping(processor, classes_mapping, mapping, "[CHILD]")
    hierarchy_labels = [
        field["label"]
        for node in mappings[0].items[0].hierarchy
        for field in node.get("fields", [])
    ]
    assert hierarchy_labels == list(mapping.class_to_id)


@pytest.mark.parametrize(
    ("processor_class", "task", "config_name", "head_config", "tensor_key"),
    [
        (
            VisionProcessor,
            "image_classification",
            "image_classification_config",
            ImageClassificationHeadConfig(),
            "image_classification_labels",
        ),
        (
            AudioProcessor,
            "audio_classification",
            "audio_classification_config",
            AudioClassificationHeadConfig(),
            "audio_classification_labels",
        ),
    ],
)
def test_media_classification_variants_keep_augmented_targets_aligned(
    processor_class,
    task,
    config_name,
    head_config,
    tensor_key,
):
    processor = processor_class(
        make_config(
            model_variant="vision" if processor_class is VisionProcessor else "audio",
            default_ner_config=False,
            **{config_name: asdict(head_config)},
        ),
        task,
    )
    items = [
        {
            task: [{
                "name": "labels",
                "all_labels": ["target"],
                "true_labels": ["target"],
            }],
        },
        {
            task: [{
                "name": "labels",
                "all_labels": ["donor_a", "donor_b"],
                "true_labels": ["donor_a"],
            }],
        },
    ]
    mappings = processor.get_classes_mapping(items)
    classes_mapping = _classes_mapping(task, mappings)
    baseline = processor.create_labels(items, classes_mapping)
    assert baseline[tensor_key][0, 0].item() == 1.0
    assert baseline[tensor_key][0].sum().item() == 1.0

    mapping = _augment(processor, task, items, classes_mapping)
    result = processor.create_labels(items, classes_mapping)

    assert result[tensor_key][0].tolist() == [0.0, 0.0]
    _assert_prompt_matches_mapping(processor, classes_mapping, mapping, "[OBJECT]")


def _localization_items(task, instance_key, position_key, target_position):
    def item(labels, label, position):
        instance = {"label": label, position_key: position}
        if task == "segmentation":
            instance["mask"] = [[1.0]]
        return {
            task: [{
                "name": "labels",
                "all_labels": labels,
                instance_key: [instance],
            }],
        }

    return [
        item(["target"], "target", target_position),
        item(["donor_a", "donor_b"], "donor_a", target_position),
    ]


@pytest.mark.parametrize(
    (
        "processor_class",
        "task",
        "config_name",
        "head_config",
        "instance_key",
        "position_key",
        "target_position",
        "class_key",
        "mask_key",
    ),
    [
        (
            VisionProcessor,
            "object_detection",
            "object_detection_config",
            ObjectDetectionHeadConfig(),
            "objects",
            "bbox",
            [0.0, 0.0, 1.0, 1.0],
            "object_detection_class_labels",
            "object_detection_object_mask",
        ),
        (
            VisionProcessor,
            "segmentation",
            "segmentation_config",
            SegmentationHeadConfig(mask_size=4),
            "objects",
            "bbox",
            [0.0, 0.0, 1.0, 1.0],
            "segmentation_class_labels",
            "segmentation_object_mask",
        ),
        (
            AudioProcessor,
            "audio_segmentation",
            "audio_segmentation_config",
            AudioSegmentationHeadConfig(mask_size=4),
            "segments",
            "segment",
            [0.0, 1.0],
            "audio_segmentation_class_labels",
            "audio_segmentation_object_mask",
        ),
    ],
)
def test_media_localization_variants_ignore_dropped_gold_classes(
    processor_class,
    task,
    config_name,
    head_config,
    instance_key,
    position_key,
    target_position,
    class_key,
    mask_key,
):
    processor = processor_class(
        make_config(
            model_variant="vision" if processor_class is VisionProcessor else "audio",
            default_ner_config=False,
            **{config_name: asdict(head_config)},
        ),
        task,
    )
    items = _localization_items(
        task,
        instance_key,
        position_key,
        target_position,
    )
    mappings = processor.get_classes_mapping(items)
    classes_mapping = _classes_mapping(task, mappings)
    baseline = processor.create_labels(items, classes_mapping)
    assert baseline[class_key][0, 0].item() == 0
    assert baseline[mask_key][0, 0].item() == 1.0

    mapping = _augment(processor, task, items, classes_mapping)
    result = processor.create_labels(items, classes_mapping)

    assert result[class_key][0].tolist() == [-1]
    assert not result[mask_key][0].any()
    _assert_prompt_matches_mapping(processor, classes_mapping, mapping, "[OBJECT]")
