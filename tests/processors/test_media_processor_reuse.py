"""Contract tests for the shared vision/audio processor mechanics."""

from dataclasses import asdict

import pytest
import torch

from gliformer.config import (
    AudioClassificationHeadConfig,
    AudioSegmentationHeadConfig,
    ImageClassificationHeadConfig,
    MediaClassificationHeadConfig,
)
from gliformer.processing.mappings import BatchClassesMapping
from gliformer.tasks.audio.processor import AudioProcessor
from gliformer.tasks.media_processor import MediaTaskProcessor
from gliformer.tasks.vision.processor import VisionProcessor
from tests.conftest import make_config
from tests.processors.test_unified_processor import FakeTokenizer


def test_media_classification_configs_share_one_implementation():
    assert ImageClassificationHeadConfig is MediaClassificationHeadConfig
    assert AudioClassificationHeadConfig is MediaClassificationHeadConfig


@pytest.mark.parametrize(
    ("processor_class", "task_name", "config_name", "head_config", "labels_key", "labels"),
    [
        (
            VisionProcessor,
            "image_classification",
            "image_classification_config",
            ImageClassificationHeadConfig(),
            "image_classification_labels",
            ("cat", "dog"),
        ),
        (
            AudioProcessor,
            "audio_classification",
            "audio_classification_config",
            AudioClassificationHeadConfig(),
            "audio_classification_labels",
            ("speech", "music"),
        ),
    ],
)
def test_media_processors_share_mapping_prompt_and_label_encoder_contract(
    processor_class,
    task_name,
    config_name,
    head_config,
    labels_key,
    labels,
):
    config = make_config(
        model_variant="vision" if processor_class is VisionProcessor else "audio",
        default_ner_config=False,
        **{config_name: asdict(head_config)},
    )
    processor = processor_class(config, task_name)
    assert isinstance(processor, MediaTaskProcessor)

    first_label, second_label = labels
    items = [{
        task_name: [{
            "name": "primary",
            "all_labels": [first_label, second_label, first_label],
            "true_labels": [second_label],
        }],
    }]
    task_mapping = processor.get_classes_mapping(items)
    classes_mapping = BatchClassesMapping(
        cat_mapping=[],
        extraction_mapping=[],
        **{f"{task_name}_mapping": task_mapping},
    )

    assert list(task_mapping[0].items[0].class_to_id.class_to_id) == [
        first_label,
        second_label,
    ]
    assert processor.contribute_prompt(classes_mapping, 0) == [
        "[P]",
        "primary",
        f"[OBJECT] {first_label}",
        f"[OBJECT] {second_label}",
        "[SEP]",
    ]
    assert processor.contribute_prompt(
        classes_mapping,
        0,
        use_labels_encoder=True,
    ) == ["[P]", "primary", "[SEP]"]

    result = processor.create_labels(items, classes_mapping)
    assert torch.equal(result[labels_key], torch.tensor([[0.0, 1.0]]))

    encoded = processor.prepare_label_encoder_inputs(classes_mapping, FakeTokenizer())
    assert encoded[f"{task_name}_labels_input_ids"].shape[0] == 2
    assert encoded[f"{task_name}_labels_attention_mask"].shape[0] == 2
    assert encoded[f"{task_name}_labels_group_size"].tolist() == [2]

    augmentable = processor.get_augmentable_label_groups(
        items,
        classes_mapping,
    )
    assert len(augmentable) == 1
    assert augmentable[0].task == task_name
    assert augmentable[0].positive_labels == frozenset({second_label})
    augmentable[0].replace_labels([second_label, "batch_negative"])
    assert list(
        task_mapping[0].items[0].class_to_id.class_to_id
    ) == [second_label, "batch_negative"]


@pytest.mark.parametrize(
    ("processor_class", "task_name", "skip_flag"),
    [
        (VisionProcessor, "image_classification", "_skip_vision_tasks"),
        (AudioProcessor, "audio_classification", "_skip_audio_tasks"),
    ],
)
def test_media_processors_preserve_modality_specific_skip_flags(
    processor_class,
    task_name,
    skip_flag,
):
    processor = processor_class(make_config(default_ner_config=False), task_name)

    mappings = processor.get_classes_mapping([
        {skip_flag: True, "labels": ["ignored"]},
    ])

    assert mappings[0].items == []


@pytest.mark.parametrize(
    ("processor_class", "task_name", "config_name", "payload", "positive"),
    [
        (
            VisionProcessor,
            "object_detection",
            "object_detection_config",
            {
                "all_labels": ["cat", "dog"],
                "objects": [{"label": "cat", "bbox": [0.0, 0.0, 1.0, 1.0]}],
            },
            "cat",
        ),
        (
            VisionProcessor,
            "segmentation",
            "segmentation_config",
            {
                "all_labels": ["car", "road"],
                "objects": [{"label": "car", "bbox": [0.0, 0.0, 1.0, 1.0]}],
            },
            "car",
        ),
        (
            AudioProcessor,
            "audio_segmentation",
            "audio_segmentation_config",
            {
                "all_labels": ["speech", "music"],
                "segments": [{"label": "speech", "start": 0.0, "end": 0.5}],
            },
            "speech",
        ),
    ],
)
def test_media_localization_tasks_expose_augmentation_groups(
    processor_class,
    task_name,
    config_name,
    payload,
    positive,
):
    processor = processor_class(
        make_config(
            default_ner_config=False,
            **{config_name: {}},
        ),
        task_name,
    )
    items = [{task_name: [payload]}]
    task_mapping = processor.get_classes_mapping(items)
    classes_mapping = BatchClassesMapping(
        cat_mapping=[],
        extraction_mapping=[],
        **{f"{task_name}_mapping": task_mapping},
    )

    groups = processor.get_augmentable_label_groups(items, classes_mapping)

    assert len(groups) == 1
    assert groups[0].task == task_name
    assert groups[0].positive_labels == frozenset({positive})


def test_audio_segmentation_targets_are_not_truncated_by_query_capacity():
    config = make_config(
        model_variant="audio",
        default_ner_config=False,
        audio_segmentation_config=asdict(
            AudioSegmentationHeadConfig(max_count=1)
        ),
    )
    processor = AudioProcessor(config, "audio_segmentation")
    item = {
        "audio_segmentation": [
            {
                "all_labels": ["speech"],
                "segments": [
                    {"label": "speech", "start": 0.0, "end": 0.2},
                    {"label": "speech", "start": 0.3, "end": 0.5},
                    {"label": "speech", "start": 0.6, "end": 0.9},
                ],
            }
        ]
    }
    mapping = processor.get_classes_mapping([item])
    classes_mapping = BatchClassesMapping(
        cat_mapping=[],
        extraction_mapping=[],
        audio_segmentation_mapping=mapping,
    )

    result = processor.create_labels([item], classes_mapping)

    assert result["audio_segmentation_object_mask"].sum().item() == 3
    assert result["audio_segmentation_class_labels"].shape == (1, 3)


@pytest.mark.parametrize(
    ("processor_class", "task_name", "config_name", "labels_key"),
    [
        (
            VisionProcessor,
            "image_classification",
            "image_classification_config",
            "image_classification_labels",
        ),
        (
            AudioProcessor,
            "audio_classification",
            "audio_classification_config",
            "audio_classification_labels",
        ),
    ],
)
def test_media_classification_targets_skip_leading_and_middle_empty_groups(
    processor_class,
    task_name,
    config_name,
    labels_key,
):
    processor = processor_class(
        make_config(default_ner_config=False, **{config_name: {}}),
        task_name,
    )
    items = [{
        task_name: [
            {
                "name": "leading-empty",
                "all_labels": [],
            },
            {
                "name": "first",
                "all_labels": ["first-negative", "first-positive"],
                "true_labels": ["first-positive"],
            },
            {
                "name": "middle-empty",
                "all_labels": [],
            },
            {
                "name": "second",
                "all_labels": ["second-positive", "second-negative"],
                "true_labels": ["second-positive"],
            },
        ],
    }]
    mapping = processor.get_classes_mapping(items)
    classes_mapping = BatchClassesMapping(
        cat_mapping=[],
        extraction_mapping=[],
        **{f"{task_name}_mapping": mapping},
    )

    result = processor.create_labels(items, classes_mapping)
    augmentable = processor.get_augmentable_label_groups(
        items,
        classes_mapping,
    )

    assert [group.name for group in mapping[0].items] == [
        "first",
        "second",
    ]
    assert result[labels_key].tolist() == [
        [0.0, 1.0],
        [1.0, 0.0],
    ]
    assert [group.positive_labels for group in augmentable] == [
        frozenset({"first-positive"}),
        frozenset({"second-positive"}),
    ]


@pytest.mark.parametrize(
    (
        "processor_class",
        "task_name",
        "config_name",
        "instance_key",
        "position_key",
        "position",
        "class_labels_key",
        "object_mask_key",
    ),
    [
        (
            VisionProcessor,
            "object_detection",
            "object_detection_config",
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
            "segments",
            "segment",
            [0.0, 1.0],
            "audio_segmentation_class_labels",
            "audio_segmentation_object_mask",
        ),
    ],
)
def test_media_localization_targets_skip_leading_and_middle_empty_groups(
    processor_class,
    task_name,
    config_name,
    instance_key,
    position_key,
    position,
    class_labels_key,
    object_mask_key,
):
    processor = processor_class(
        make_config(default_ner_config=False, **{config_name: {}}),
        task_name,
    )

    def instance(label):
        return {"label": label, position_key: position}

    items = [{
        task_name: [
            {
                "name": "leading-empty",
                "all_labels": [],
            },
            {
                "name": "first",
                "all_labels": ["first-negative", "first-positive"],
                instance_key: [instance("first-positive")],
            },
            {
                "name": "middle-empty",
                "all_labels": [],
            },
            {
                "name": "second",
                "all_labels": ["second-positive", "second-negative"],
                instance_key: [instance("second-positive")],
            },
        ],
    }]
    mapping = processor.get_classes_mapping(items)
    classes_mapping = BatchClassesMapping(
        cat_mapping=[],
        extraction_mapping=[],
        **{f"{task_name}_mapping": mapping},
    )

    result = processor.create_labels(items, classes_mapping)
    augmentable = processor.get_augmentable_label_groups(
        items,
        classes_mapping,
    )

    assert [group.name for group in mapping[0].items] == [
        "first",
        "second",
    ]
    assert result[class_labels_key].tolist() == [[1], [0]]
    assert result[object_mask_key].tolist() == [[1.0], [1.0]]
    assert [group.positive_labels for group in augmentable] == [
        frozenset({"first-positive"}),
        frozenset({"second-positive"}),
    ]
