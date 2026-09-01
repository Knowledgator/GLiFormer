import wave
from dataclasses import asdict

import pytest
import torch

from glinext.config import (
    AudioClassificationHeadConfig,
    AudioSegmentationHeadConfig,
    ClassificationHeadConfig,
    ImageClassificationHeadConfig,
    ObjectDetectionHeadConfig,
    SegmentationHeadConfig,
)
from glinext.glinext import (
    GLiNExTAudio,
    GLiNExTLayout,
    GLiNExTOmni,
    GLiNExTText,
    GLiNExTVision,
)
from glinext.processing.collator import (
    GLiNExTAudioDataCollator,
    GLiNExTLayoutDataCollator,
    GLiNExTOmniDataCollator,
    GLiNExTTextDataCollator,
    GLiNExTVisionDataCollator,
    resolve_glinext_collator_class,
)
from glinext.processing.label_augmentation import (
    LABEL_AUGMENTATION_INDEX_KEY,
    LABEL_AUGMENTATION_MARKER_KEY,
)
from glinext.processing.processor import (
    GLiNextAudioProcessor,
    GLiNextLayoutProcessor,
    GLiNextOmniProcessor,
    GLiNextTextProcessor,
    GLiNextVisionProcessor,
)
from tests.conftest import FakeWordsSplitter, make_config
from tests.processors.test_unified_processor import FakeTokenizer


def _marked_training_item(index, label):
    return {
        LABEL_AUGMENTATION_MARKER_KEY: True,
        LABEL_AUGMENTATION_INDEX_KEY: index,
        "text": f"example {label}",
        "classification": [{
            "name": "topic",
            "all_labels": [label],
            "true_labels": [label],
        }],
    }


def test_text_collator_applies_same_batch_labels_only_to_marked_training_rows():
    config = make_config(
        default_ner_config=False,
        ner_config=None,
        classification_config={},
    )
    processor = GLiNextTextProcessor(
        config,
        FakeTokenizer(),
        FakeWordsSplitter(),
    )
    collator = GLiNExTTextDataCollator(
        config,
        processor,
        label_augmentation={
            "enabled": True,
            "seed": 7,
            "tasks": {
                "classification": {"add_probability": 1.0},
            },
        },
    )

    training_batch = collator([
        _marked_training_item(10, "science"),
        _marked_training_item(11, "sports"),
    ])
    training_mappings = training_batch["classes_mapping"].cat_mapping

    assert list(training_mappings[0].cat_class_to_id[0].class_to_id) == [
        "science",
        "sports",
    ]
    assert list(training_mappings[1].cat_class_to_id[0].class_to_id) == [
        "sports",
        "science",
    ]
    assert training_batch["cat_labels"].tolist() == [
        [1.0, 0.0],
        [1.0, 0.0],
    ]

    evaluation_batch = collator([
        {
            key: value
            for key, value in _marked_training_item(10, "science").items()
            if key not in {
                LABEL_AUGMENTATION_MARKER_KEY,
                LABEL_AUGMENTATION_INDEX_KEY,
            }
        },
        {
            key: value
            for key, value in _marked_training_item(11, "sports").items()
            if key not in {
                LABEL_AUGMENTATION_MARKER_KEY,
                LABEL_AUGMENTATION_INDEX_KEY,
            }
        },
    ])
    evaluation_mappings = evaluation_batch["classes_mapping"].cat_mapping
    assert list(evaluation_mappings[0].cat_class_to_id[0].class_to_id) == [
        "science"
    ]
    assert list(evaluation_mappings[1].cat_class_to_id[0].class_to_id) == [
        "sports"
    ]


def test_collator_rejects_mixed_training_and_evaluation_markers():
    config = make_config(
        default_ner_config=False,
        ner_config=None,
        classification_config={},
    )
    processor = GLiNextTextProcessor(
        config,
        FakeTokenizer(),
        FakeWordsSplitter(),
    )
    collator = GLiNExTTextDataCollator(
        config,
        processor,
        label_augmentation={
            "enabled": True,
            "tasks": {
                "classification": {"add_probability": 1.0},
            },
        },
    )

    with pytest.raises(ValueError, match="cannot mix"):
        collator([
            _marked_training_item(0, "science"),
            {
                "text": "example sports",
                "classification": [{
                    "all_labels": ["sports"],
                    "true_labels": ["sports"],
                }],
            },
        ])


def test_resolve_glinext_collator_class():
    assert resolve_glinext_collator_class(make_config(model_variant="text")) is GLiNExTTextDataCollator
    assert resolve_glinext_collator_class(make_config(model_variant="layout")) is GLiNExTLayoutDataCollator
    assert resolve_glinext_collator_class(make_config(model_variant="vision")) is GLiNExTVisionDataCollator
    assert resolve_glinext_collator_class(make_config(model_variant="audio")) is GLiNExTAudioDataCollator
    assert resolve_glinext_collator_class(make_config(model_variant="omni")) is GLiNExTOmniDataCollator


def test_user_facing_wrappers_pin_collator_classes():
    assert GLiNExTText.data_collator_class is GLiNExTTextDataCollator
    assert GLiNExTLayout.data_collator_class is GLiNExTLayoutDataCollator
    assert GLiNExTVision.data_collator_class is GLiNExTVisionDataCollator
    assert GLiNExTAudio.data_collator_class is GLiNExTAudioDataCollator
    assert GLiNExTOmni.data_collator_class is GLiNExTOmniDataCollator


def test_layout_collator_masks_text_only_rows_in_mixed_batch():
    config = make_config(
        model_variant="layout",
        default_ner_config=False,
        ner_config=None,
        classification_config={},
    )
    processor = GLiNextLayoutProcessor(
        config,
        FakeTokenizer(),
        FakeWordsSplitter(),
    )
    collator = GLiNExTLayoutDataCollator(
        config,
        processor,
        prepare_labels=False,
    )
    classification = [{
        "name": "topic",
        "all_labels": ["invoice"],
        "true_labels": ["invoice"],
    }]

    batch = collator([
        {
            "text": "Invoice total",
            "tokenized_text": ["Invoice", "total"],
            "bboxes": [[10, 20, 40, 40], [50, 20, 80, 40]],
            "page": 3,
            "classification": classification,
        },
        {
            "text": "A text only example",
            "classification": classification,
        },
    ])

    assert batch["layout_input_mask"].dtype == torch.bool
    assert batch["layout_input_mask"].tolist() == [True, False]
    assert batch["page_input_mask"].dtype == torch.bool
    assert batch["page_input_mask"].tolist() == [True, False]
    assert torch.count_nonzero(batch["page_token_ids"][0] == 3).item() > 0
    assert torch.count_nonzero(batch["page_token_ids"][1]).item() == 0
    assert torch.count_nonzero(batch["bbox"][0]).item() > 0
    assert torch.count_nonzero(batch["bbox"][1]).item() == 0


def test_layout_collator_omits_bbox_for_text_only_batch():
    config = make_config(
        model_variant="layout",
        default_ner_config=False,
        ner_config=None,
        classification_config={},
    )
    processor = GLiNextLayoutProcessor(
        config,
        FakeTokenizer(),
        FakeWordsSplitter(),
    )
    collator = GLiNExTLayoutDataCollator(
        config,
        processor,
        prepare_labels=False,
    )

    batch = collator([{
        "text": "A text only example",
        "classification": [{
            "name": "topic",
            "all_labels": ["plain"],
            "true_labels": ["plain"],
        }],
    }])

    assert "bbox" not in batch
    assert "layout_input_mask" not in batch
    assert "page_token_ids" not in batch
    assert "page_input_mask" not in batch


def test_text_collator_uses_source_length_retained_after_prompt_truncation():
    config = make_config(
        max_len=8,
        default_ner_config=False,
        ner_config=None,
        structuring_config={"neg_spans_ratio": 0.0},
    )
    processor = GLiNextTextProcessor(
        config,
        FakeTokenizer(),
        FakeWordsSplitter(),
    )
    collator = GLiNExTTextDataCollator(config, processor)
    batch = collator([{
        "text": "A B C D",
        "structuring": {
            "schema": [{
                "field": [
                    {"text": "A", "start": 0, "end": 0},
                    {"text": "D", "start": 3, "end": 3},
                ],
            }],
        },
    }])

    retained_length = int(batch["words_mask"].amax().item())
    assert 0 < retained_length < 4
    assert batch["text_lengths"].tolist() == [[retained_length]]
    assert batch["structuring_labels"].shape[2] == retained_length
    assert batch["structuring_span_mask"].sum().item() == 1
    assert batch["structuring_span_idx"][0, 0].tolist() == [0, 0]


def test_text_collator_filters_structuring_spans_per_item_not_batch_max():
    config = make_config(
        max_len=10,
        default_ner_config=False,
        ner_config=None,
        structuring_config={"neg_spans_ratio": 0.0},
    )
    processor = GLiNextTextProcessor(
        config,
        FakeTokenizer(),
        FakeWordsSplitter(),
    )
    collator = GLiNExTTextDataCollator(config, processor)

    def span(text, index):
        return {"text": text, "start": index, "end": index}

    batch = collator([
        {
            "text": "A B C D",
            "structuring": {"schema": [{
                "first": span("A", 0),
                "second": span("B", 1),
                "truncated": span("D", 3),
            }]},
        },
        {
            "text": "E F G",
            "structuring": {
                "schema": [{"last": span("G", 2)}],
            },
        },
        {
            "text": "H I",
            "structuring": {"schema": [{
                f"field_{index}": span("H", 0)
                for index in range(5)
            }]},
        },
    ])

    assert batch["text_lengths"].tolist() == [[2], [3], [0]]
    assert batch["word_lengths"] == [2, 3, 0]
    assert batch["structuring_labels"].shape[2] == 3
    assert batch["structuring_count"].tolist() == [1, 1, 0]
    assert batch["structuring_span_mask"].sum(dim=1).tolist() == [2, 1, 0]
    assert batch["structuring_span_idx"][0, :2].tolist() == [[0, 0], [1, 1]]
    assert batch["structuring_span_idx"][1, 0].tolist() == [2, 2]


def test_text_collator_filters_joint_relations_by_retained_source_length():
    config = make_config(
        max_len=8,
        joint_relex_config={},
    )
    processor = GLiNextTextProcessor(
        config,
        FakeTokenizer(),
        FakeWordsSplitter(),
    )
    collator = GLiNExTTextDataCollator(config, processor)
    batch = collator([{
        "text": "A B C",
        "extraction": [{
            "ner": [
                {"text": "A", "start": 0, "end": 1, "label": "entity"},
                {"text": "C", "start": 4, "end": 5, "label": "entity"},
            ],
            "relations": [[0, "related_to", 1]],
        }],
    }])

    assert batch["text_lengths"].tolist() == [[2]]
    assert batch["rel_span_mask"].tolist() == [[True, False]]
    assert not batch["rel_pair_mask"].any()
    assert not batch["rel_labels"].any()


def test_vision_collator_does_not_add_text_fields():
    config = make_config(
        model_variant="vision",
        default_ner_config=False,
        image_classification_config=asdict(ImageClassificationHeadConfig()),
    )
    processor = GLiNextVisionProcessor(config, FakeTokenizer(), FakeWordsSplitter())
    collator = GLiNExTVisionDataCollator(config, processor)

    batch = collator([
        {
            "pixel_values": torch.zeros(3, 8, 8),
            "labels": ["cat"],
            "true_labels": ["cat"],
        }
    ])

    assert "pixel_values" in batch
    assert "image_classification_labels" in batch
    assert "input_ids" not in batch
    assert "text_lengths" not in batch


def test_vision_task_specific_inference_labels_only_activate_requested_head():
    config = make_config(
        model_variant="vision",
        default_ner_config=False,
        image_classification_config=asdict(ImageClassificationHeadConfig()),
        object_detection_config=asdict(ObjectDetectionHeadConfig()),
    )
    processor = GLiNextVisionProcessor(config, FakeTokenizer(), FakeWordsSplitter())
    collator = GLiNExTVisionDataCollator(config, processor, prepare_labels=False)

    batch = collator([
        {
            "pixel_values": torch.zeros(3, 8, 8),
            "image_classification": [
                {"name": "scene", "all_labels": ["indoor", "outdoor"]},
            ],
        }
    ])

    mapping = batch["classes_mapping"]
    assert mapping.total_image_classification_groups() == 1
    assert mapping.total_object_detection_groups() == 0


def test_vision_grouped_task_annotations_create_group_specific_labels():
    config = make_config(
        model_variant="vision",
        default_ner_config=False,
        image_classification_config=asdict(ImageClassificationHeadConfig()),
        object_detection_config=asdict(ObjectDetectionHeadConfig()),
        segmentation_config=asdict(SegmentationHeadConfig()),
    )
    processor = GLiNextVisionProcessor(config, FakeTokenizer(), FakeWordsSplitter())
    collator = GLiNExTVisionDataCollator(config, processor)

    batch = collator([
        {
            "pixel_values": torch.zeros(3, 8, 8),
            "image_classification": [
                {"name": "animal", "all_labels": ["cat", "dog"], "true_labels": ["cat"]},
                {"name": "place", "all_labels": ["indoor", "outdoor"], "true_labels": ["indoor"]},
            ],
            "object_detection": [
                {
                    "name": "animals",
                    "all_labels": ["cat", "dog"],
                    "objects": [{"label": "cat", "bbox": [0, 0, 4, 4]}],
                },
                {
                    "name": "vehicles",
                    "all_labels": ["person", "car"],
                    "objects": [{"label": "car", "bbox": [1, 1, 6, 6]}],
                },
            ],
            "segmentation": [
                {
                    "name": "things",
                    "all_labels": ["cat"],
                    "objects": [{"label": "cat", "bbox": [0, 0, 4, 4]}],
                }
            ],
        }
    ])

    assert batch["image_classification_labels"].tolist() == [[1.0, 0.0], [1.0, 0.0]]
    assert batch["object_detection_class_labels"].tolist() == [[0], [1]]
    assert batch["object_detection_object_mask"].tolist() == [[1.0], [1.0]]
    assert batch["segmentation_class_labels"].tolist() == [[0]]


def test_detection_targets_retain_all_valid_objects_in_source_order():
    config = make_config(
        model_variant="vision",
        default_ner_config=False,
        image_size=100,
        object_detection_config=asdict(ObjectDetectionHeadConfig(
            num_fixed_slots=3,
        )),
    )
    processor = GLiNextVisionProcessor(config, FakeTokenizer(), FakeWordsSplitter())
    collator = GLiNExTVisionDataCollator(config, processor)

    batch = collator([{
        "pixel_values": torch.zeros(3, 100, 100),
        "object_detection": [{
            "name": "objects",
            "all_labels": ["cat", "dog"],
            "objects": [
                # Small and near-duplicate boxes are still valid supervision.
                {"label": "cat", "bbox": [0, 0, 5, 50]},
                {"label": "cat", "bbox": [0, 0, 60, 60]},
                {"label": "cat", "bbox": [1, 1, 59, 59]},
                {"label": "cat", "bbox": [70, 70, 90, 90]},
                {"label": "dog", "bbox": [0, 60, 50, 100]},
                {"label": "dog", "bbox": [60, 0, 80, 20]},
                # Invalid or out-of-scope entries remain excluded.
                {"label": "dog", "bbox": [10, 10, 10, 20]},
                {"label": "horse", "bbox": [0, 0, 20, 20]},
                {"label": "cat"},
            ],
        }],
    }])

    assert batch["object_detection_class_labels"].tolist() == [[0, 0, 0, 0, 1, 1]]
    assert batch["object_detection_object_mask"].tolist() == [[1.0] * 6]
    assert torch.allclose(
        batch["object_detection_bbox_labels"],
        torch.tensor([[[0.0, 0.0, 0.05, 0.5],
                       [0.0, 0.0, 0.6, 0.6],
                       [0.01, 0.01, 0.59, 0.59],
                       [0.7, 0.7, 0.9, 0.9],
                       [0.0, 0.6, 0.5, 1.0],
                       [0.6, 0.0, 0.8, 0.2]]]),
    )


def test_vision_classification_is_derived_from_object_groups():
    config = make_config(
        model_variant="vision",
        default_ner_config=False,
        image_classification_config=asdict(ImageClassificationHeadConfig()),
        object_detection_config=asdict(ObjectDetectionHeadConfig()),
    )
    processor = GLiNextVisionProcessor(config, FakeTokenizer(), FakeWordsSplitter())
    collator = GLiNExTVisionDataCollator(config, processor)

    batch = collator([
        {
            "pixel_values": torch.zeros(3, 8, 8),
            "object_detection": [
                {
                    "name": "lvis",
                    "all_labels": ["cat", "dog", "horse"],
                    "objects": [
                        {"label": "cat", "bbox": [0, 0, 4, 4]},
                        {"label": "cat", "bbox": [1, 1, 3, 3]},
                        {"label": "dog", "bbox": [2, 2, 6, 6]},
                    ],
                },
            ],
        }
    ])

    mapping = batch["classes_mapping"]
    labels = mapping.image_classification_mapping[0].items[0].class_to_id.class_to_id
    assert list(labels) == ["cat", "dog", "horse"]
    assert batch["image_classification_labels"].tolist() == [[1.0, 1.0, 0.0]]


def test_audio_collator_does_not_add_text_fields():
    config = make_config(
        model_variant="audio",
        default_ner_config=False,
        audio_classification_config=asdict(AudioClassificationHeadConfig()),
    )
    processor = GLiNextAudioProcessor(config, FakeTokenizer(), FakeWordsSplitter())
    collator = GLiNExTAudioDataCollator(config, processor)

    batch = collator([
        {
            "audio_values": torch.zeros(160),
            "labels": ["speech"],
            "true_labels": ["speech"],
        }
    ])

    assert "audio_values" in batch
    assert "audio_classification_labels" in batch
    assert "input_ids" not in batch
    assert "text_lengths" not in batch


def test_audio_task_specific_inference_labels_only_activate_requested_head():
    config = make_config(
        model_variant="audio",
        default_ner_config=False,
        audio_classification_config=asdict(AudioClassificationHeadConfig()),
        audio_segmentation_config=asdict(AudioSegmentationHeadConfig()),
    )
    processor = GLiNextAudioProcessor(config, FakeTokenizer(), FakeWordsSplitter())
    collator = GLiNExTAudioDataCollator(config, processor, prepare_labels=False)

    batch = collator([
        {
            "audio_values": torch.zeros(160),
            "audio_segmentation": [
                {"name": "events", "all_labels": ["speech", "music"]},
            ],
        }
    ])

    mapping = batch["classes_mapping"]
    assert mapping.total_audio_classification_groups() == 0
    assert mapping.total_audio_segmentation_groups() == 1


def test_audio_grouped_task_annotations_create_group_specific_labels():
    config = make_config(
        model_variant="audio",
        default_ner_config=False,
        audio_classification_config=asdict(AudioClassificationHeadConfig()),
        audio_segmentation_config=asdict(AudioSegmentationHeadConfig()),
    )
    processor = GLiNextAudioProcessor(config, FakeTokenizer(), FakeWordsSplitter())
    collator = GLiNExTAudioDataCollator(config, processor)

    batch = collator([
        {
            "audio_values": torch.zeros(160),
            "audio_classification": [
                {"name": "kind", "all_labels": ["speech", "music"], "true_labels": ["speech"]},
                {"name": "quality", "all_labels": ["clean", "noisy"], "true_labels": ["noisy"]},
            ],
            "audio_segmentation": [
                {
                    "name": "events",
                    "all_labels": ["speech", "music"],
                    "segments": [{"label": "music", "start": 0.0, "end": 0.5}],
                }
            ],
            "duration": 1.0,
        }
    ])

    assert batch["audio_classification_labels"].tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert batch["audio_segmentation_class_labels"].tolist() == [[1]]
    assert batch["audio_segmentation_object_mask"].tolist() == [[1.0]]


def test_audio_segmentation_preserves_wav_duration_before_processing(tmp_path):
    wav_path = tmp_path / "four-seconds.wav"
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * 32000)

    config = make_config(
        model_variant="audio",
        default_ner_config=False,
        audio_sampling_rate=None,
        audio_do_resample=False,
        audio_segmentation_config=asdict(AudioSegmentationHeadConfig()),
    )
    processor = GLiNextAudioProcessor(config, FakeTokenizer(), FakeWordsSplitter())

    class LengthDoublingProcessor:
        def __call__(self, audio, **kwargs):
            return {"audio_values": torch.zeros(1, len(audio) * 2)}

    processor.audio_input_processor = LengthDoublingProcessor()
    batch = processor.collate_fn([
        {
            "audio": str(wav_path),
            "audio_segmentation": [
                {
                    "all_labels": ["speech"],
                    "segments": [{"label": "speech", "start": 2.0, "end": 4.0}],
                }
            ],
        }
    ])

    assert batch["audio_values"].shape == (1, 64000)
    torch.testing.assert_close(
        batch["audio_segmentation_segment_labels"][0, 0],
        torch.tensor([0.5, 1.0]),
    )


@pytest.mark.parametrize(
    ("audio_payload", "audio_config"),
    [
        ({"audio_values": torch.zeros(20), "sample_rate": 5}, {}),
        (
            {"audio_features": torch.zeros(3, 8)},
            {"audio_sampling_rate": 4, "audio_hop_length": 2},
        ),
    ],
    ids=["waveform", "features"],
)
def test_audio_segmentation_infers_precomputed_input_duration(audio_payload, audio_config):
    config_kwargs = dict(
        model_variant="audio",
        default_ner_config=False,
        audio_sampling_rate=None,
        audio_do_resample=False,
        audio_segmentation_config=asdict(AudioSegmentationHeadConfig()),
    )
    config_kwargs.update(audio_config)
    config = make_config(**config_kwargs)
    processor = GLiNextAudioProcessor(config, FakeTokenizer(), FakeWordsSplitter())
    item = {
        **audio_payload,
        "audio_segmentation": [
            {
                "all_labels": ["speech"],
                "segments": [{"label": "speech", "start": 2.0, "end": 4.0}],
            }
        ],
    }

    raw_batch = processor.collate_raw_batch([item])
    assert raw_batch["_audio_duration_seconds"] == [4.0]

    batch = processor.tokenize_and_prepare_labels(raw_batch)
    torch.testing.assert_close(
        batch["audio_segmentation_segment_labels"][0, 0],
        torch.tensor([0.5, 1.0]),
    )


def test_omni_audio_duration_metadata_survives_optional_media_collation():
    config = make_config(
        model_variant="omni",
        default_ner_config=False,
        audio_sampling_rate=None,
        audio_do_resample=False,
        audio_segmentation_config=asdict(AudioSegmentationHeadConfig()),
    )
    processor = GLiNextOmniProcessor(config, FakeTokenizer(), FakeWordsSplitter())
    raw_batch = processor.collate_raw_batch([
        {
            "text": "feature row",
            "audio_features": torch.zeros(3, 8),
            "sample_rate": 4,
            "frame_rate": 2,
            "audio_segmentation": [
                {
                    "all_labels": ["speech"],
                    "segments": [{"label": "speech", "start": 2.0, "end": 4.0}],
                }
            ],
        }
    ])
    label_items = processor._build_label_batch_list(raw_batch)
    batch = processor.task_processors["audio_segmentation"].create_labels(
        label_items,
        raw_batch["classes_mapping"],
    )

    assert raw_batch["_audio_duration_seconds"] == [4.0]
    assert label_items[0]["_audio_duration_seconds"] == 4.0
    torch.testing.assert_close(
        batch["audio_segmentation_segment_labels"][0, 0],
        torch.tensor([0.5, 1.0]),
    )


def test_omni_collator_allows_rows_without_every_media_type():
    config = make_config(
        model_variant="omni",
        classification_config=asdict(ClassificationHeadConfig()),
        image_classification_config=asdict(ImageClassificationHeadConfig()),
        audio_classification_config=asdict(AudioClassificationHeadConfig()),
        audio_sampling_rate=None,
        audio_do_resample=False,
    )
    processor = GLiNextOmniProcessor(config, FakeTokenizer(), FakeWordsSplitter())
    collator = GLiNExTOmniDataCollator(config, processor)

    batch = collator([
        {
            "text": "plain text row",
            "classification": [{"all_labels": ["news"], "true_labels": ["news"]}],
        },
        {
            "text": "image row",
            "pixel_values": torch.ones(3, 8, 8),
            "labels": ["cat"],
            "true_labels": ["cat"],
        },
        {
            "text": "audio row",
            "audio_values": torch.ones(16),
            "labels": ["speech"],
            "true_labels": ["speech"],
        },
    ])

    mapping = batch["classes_mapping"]
    assert mapping.total_cat_groups() == 1
    assert mapping.total_image_classification_groups() == 1
    assert mapping.total_audio_classification_groups() == 1
    assert batch["pixel_values"].shape == (3, 3, 8, 8)
    assert batch["vision_input_mask"].tolist() == [0, 1, 0]
    assert batch["audio_values"].shape == (3, 16)
    assert batch["audio_input_mask"].tolist() == [0, 0, 1]
    assert batch["audio_attention_mask"].sum(dim=1).tolist() == [0, 0, 16]


def test_omni_collator_rejects_explicit_media_task_without_payload():
    config = make_config(
        model_variant="omni",
        image_classification_config=asdict(ImageClassificationHeadConfig()),
    )
    processor = GLiNextOmniProcessor(config, FakeTokenizer(), FakeWordsSplitter())
    collator = GLiNExTOmniDataCollator(config, processor)

    try:
        collator([
            {
                "text": "missing image",
                "image_classification": [{"all_labels": ["cat"]}],
            }
        ])
    except ValueError as exc:
        assert "Omni vision task rows require" in str(exc)
    else:
        raise AssertionError("Expected an explicit vision task without image payload to fail")


def test_omni_collator_rejects_spatial_padding_that_would_misalign_boxes():
    config = make_config(
        model_variant="omni",
        image_classification_config=asdict(ImageClassificationHeadConfig()),
    )
    processor = GLiNextOmniProcessor(config, FakeTokenizer(), FakeWordsSplitter())
    collator = GLiNExTOmniDataCollator(config, processor)

    with pytest.raises(ValueError, match="one processed tensor shape"):
        collator([
            {
                "text": "small",
                "pixel_values": torch.ones(3, 8, 8),
                "labels": ["cat"],
            },
            {
                "text": "wide",
                "pixel_values": torch.ones(3, 8, 12),
                "labels": ["cat"],
            },
        ])
