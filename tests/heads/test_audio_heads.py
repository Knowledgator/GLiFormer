"""Tests for multimodal audio heads."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from gliformer.config import (
    AudioClassificationHeadConfig,
    AudioSegmentationHeadConfig,
)
from gliformer.tasks import TaskFlatInputs
from gliformer.tasks.audio.model import (
    AudioClassificationHead,
    AudioSegmentationHead,
)
from gliformer.tasks.audio.decoder import AudioSegmentationDecoder
from gliformer.tasks.media import MediaClassificationHead, matched_mask_loss
from gliformer.encoders.audio import ConvAudioEncoder
from tests.heads.conftest import B, C, D, W, make_config


def _audio_flat_inputs():
    return TaskFlatInputs(
        words_embedding=torch.randn(B, W, D),
        mask=torch.ones(B, W, dtype=torch.long),
        parent_embedding=torch.randn(B, D),
        child_embedding=torch.randn(B, C, D),
        child_mask=torch.ones(B, C, dtype=torch.long),
        batch_origin=torch.arange(B),
    )


def test_audio_classification_head_forward(shared):
    config = make_config(
        audio_classification_config=asdict(AudioClassificationHeadConfig()),
    )
    head = AudioClassificationHead.from_config(config)
    assert isinstance(head, MediaClassificationHead)
    flat_inputs = _audio_flat_inputs()
    labels = torch.zeros(B, C)
    labels[:, 0] = 1.0

    out = head(shared, {}, flat_inputs=flat_inputs, audio_classification_labels=labels)

    assert out.logits.shape == (B, C)
    assert out.loss is not None
    assert out.loss.item() >= 0


def test_media_classification_honors_explicit_bce_configuration(shared):
    config = make_config(
        audio_classification_config=asdict(
            AudioClassificationHeadConfig(
                focal_loss_alpha=-1.0,
                focal_loss_gamma=0.0,
            )
        ),
    )
    head = AudioClassificationHead.from_config(config)
    flat_inputs = _audio_flat_inputs()
    labels = torch.zeros(B, C)
    labels[:, 0] = 1.0

    out = head(
        shared,
        {},
        flat_inputs=flat_inputs,
        audio_classification_labels=labels,
    )

    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        out.logits,
        labels,
        reduction="none",
    ).mean()
    assert torch.allclose(out.loss, expected)


def test_audio_classification_uses_feature_embedding_not_words(shared):
    config = make_config(
        audio_classification_config=asdict(AudioClassificationHeadConfig()),
    )
    head = AudioClassificationHead.from_config(config)
    head.eval()
    flat_inputs = _audio_flat_inputs()
    flat_inputs.feature_embedding = torch.randn_like(flat_inputs.words_embedding)
    flat_inputs.feature_mask = torch.ones_like(flat_inputs.mask)
    flat_inputs.words_embedding = torch.zeros_like(flat_inputs.words_embedding)
    flat_inputs.mask = torch.zeros_like(flat_inputs.mask)

    first = head(shared, {}, flat_inputs=flat_inputs).logits
    flat_inputs.feature_embedding = flat_inputs.feature_embedding + 1.0
    second = head(shared, {}, flat_inputs=flat_inputs).logits

    assert not torch.allclose(first, second)


def test_audio_segmentation_head_outputs_temporal_segments_and_masks(shared):
    config = make_config(
        audio_segmentation_config=asdict(
            AudioSegmentationHeadConfig(num_fixed_slots=3, num_prototypes=8, mask_size=32),
        ),
    )
    head = AudioSegmentationHead.from_config(config)
    assert head.matcher.class_probability == "sigmoid"
    flat_inputs = _audio_flat_inputs()
    class_labels = torch.tensor([[0], [1]])
    segment_labels = torch.tensor(
        [
            [[0.10, 0.45]],
            [[0.35, 0.80]],
        ],
        dtype=torch.float,
    )
    object_mask = torch.ones(B, 1)
    mask_labels = torch.zeros(B, 1, 32)
    mask_labels[:, :, 8:24] = 1.0

    out = head(
        shared,
        {},
        flat_inputs=flat_inputs,
        audio_segmentation_class_labels=class_labels,
        audio_segmentation_segment_labels=segment_labels,
        audio_segmentation_object_mask=object_mask,
        audio_segmentation_mask_labels=mask_labels,
    )

    assert out.logits.shape == (B, 3, C)
    assert out.extra["segment_preds"].shape == (B, 3, 2)
    assert out.extra["objectness_logits"].shape == (B, 3)
    assert out.extra["mask_logits"].shape == (B, 3, 32)
    assert out.extra["prototypes"].shape == (B, 8, 32)
    assert out.loss is not None


def test_audio_segmentation_uses_temporal_features_and_positions(shared):
    config = make_config(
        audio_segmentation_config=asdict(
            AudioSegmentationHeadConfig(
                num_fixed_slots=3,
                num_prototypes=8,
                mask_size=32,
            )
        ),
    )
    head = AudioSegmentationHead.from_config(config).eval()
    flat_inputs = _audio_flat_inputs()

    first = head._compute_segmentation(flat_inputs)[1]
    flat_inputs.words_embedding = flat_inputs.words_embedding + torch.randn_like(
        flat_inputs.words_embedding
    )
    second = head._compute_segmentation(flat_inputs)[1]

    assert not torch.allclose(first, second)

    tokens = torch.randn(1, 9, D)
    short_positions = head._memory_positions(
        tokens[:, :5],
        torch.ones(1, 5, dtype=torch.long),
    )
    padded_positions = head._memory_positions(
        tokens,
        torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0]]),
    )
    assert torch.allclose(
        short_positions,
        padded_positions[:, :5],
        atol=1e-6,
        rtol=1e-6,
    )


def test_audio_segmentation_routes_memory_positions_into_values(monkeypatch):
    config = make_config(
        audio_segmentation_config=asdict(
            AudioSegmentationHeadConfig(
                num_fixed_slots=3,
                memory_position_in_values=True,
            )
        ),
    )
    head = AudioSegmentationHead.from_config(config).eval()
    forwarded = {}
    original_forward = head.anchor_refine.forward

    def capture_forward(*args, **kwargs):
        forwarded.update(kwargs)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(head.anchor_refine, "forward", capture_forward)
    head._compute_segmentation(_audio_flat_inputs())

    assert forwarded["memory_position_in_values"] is True


def test_audio_segmentation_mask_loss_is_normalized_per_mask_step():
    matches = {0: [(0, 0), (1, 1)]}
    small_logits = torch.zeros(1, 2, 8)
    small_labels = torch.zeros(1, 2, 8)
    large_logits = torch.zeros(1, 2, 64)
    large_labels = torch.zeros(1, 2, 64)

    small_loss = matched_mask_loss(small_logits, small_labels, matches)
    large_loss = matched_mask_loss(large_logits, large_labels, matches)

    assert torch.allclose(small_loss, large_loss)


def test_local_audio_encoder_maps_valid_lengths_through_stride():
    encoder = ConvAudioEncoder(hidden_size=D, stride=4)

    assert encoder.output_lengths(torch.tensor([16, 9, 1])).tolist() == [4, 3, 1]


def test_audio_prototypes_resize_each_valid_prefix_independently(shared):
    config = make_config(
        audio_segmentation_config=asdict(
            AudioSegmentationHeadConfig(
                num_fixed_slots=3,
                num_prototypes=8,
                mask_size=32,
            )
        ),
    )
    head = AudioSegmentationHead.from_config(config).eval()
    short_tokens = torch.randn(1, 5, D)
    padded_tokens = torch.cat(
        [short_tokens, torch.randn(1, 4, D) * 10_000],
        dim=1,
    )

    short = head._prototype_masks(
        short_tokens,
        torch.ones(1, 5, dtype=torch.long),
    )
    padded = head._prototype_masks(
        padded_tokens,
        torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0]]),
    )

    assert torch.allclose(short, padded, atol=1e-6, rtol=1e-6)


def test_audio_prototypes_return_zero_for_an_empty_feature_row(shared):
    config = make_config(
        audio_segmentation_config=asdict(
            AudioSegmentationHeadConfig(
                num_fixed_slots=3,
                num_prototypes=8,
                mask_size=32,
            )
        ),
    )
    head = AudioSegmentationHead.from_config(config).eval()

    prototypes = head._prototype_masks(
        torch.randn(2, 5, D),
        torch.tensor([[1, 1, 1, 0, 0], [0, 0, 0, 0, 0]]),
    )

    assert prototypes.shape == (2, 8, 32)
    assert torch.equal(prototypes[1], torch.zeros_like(prototypes[1]))


def test_audio_prototypes_reject_non_right_padded_feature_mask(shared):
    config = make_config(
        audio_segmentation_config=asdict(
            AudioSegmentationHeadConfig(
                num_fixed_slots=3,
                num_prototypes=8,
                mask_size=32,
            )
        ),
    )
    head = AudioSegmentationHead.from_config(config).eval()

    with pytest.raises(ValueError, match="audio feature mask must be right-padded"):
        head._prototype_masks(
            torch.randn(1, 4, D),
            torch.tensor([[1, 0, 1, 0]]),
        )


def test_audio_objectness_scale_is_invariant_to_background_query_count(shared):
    config = make_config(
        audio_segmentation_config=asdict(
            AudioSegmentationHeadConfig(
                num_fixed_slots=12,
                class_loss_coef=0.0,
                segment_loss_coef=0.0,
                objectness_loss_coef=1.0,
                focal_loss_alpha=-1.0,
                focal_loss_gamma=0.0,
                objectness_focal_loss_alpha=-1.0,
                objectness_focal_loss_gamma=0.0,
            )
        ),
    )
    head = AudioSegmentationHead.from_config(config)

    def loss_for(query_count):
        segments = torch.zeros(1, query_count, 2)
        segments[:, 0] = torch.tensor([0.2, 0.6])
        objectness = torch.full((1, query_count), -0.4)
        objectness[:, 0] = 0.3
        loss, _ = head._segmentation_loss(
            torch.zeros(1, query_count, C),
            segments,
            objectness,
            torch.ones(1, query_count, dtype=torch.bool),
            torch.ones(1, C),
            torch.tensor([[0]]),
            torch.tensor([[[0.2, 0.6]]]),
            torch.ones(1, 1),
        )
        return loss

    assert torch.allclose(loss_for(3), loss_for(12))


def test_audio_objectness_forwards_its_focal_overrides_only_to_objectness(shared):
    config = make_config(
        audio_segmentation_config=asdict(
            AudioSegmentationHeadConfig(
                num_fixed_slots=3,
                class_loss_coef=0.0,
                segment_loss_coef=0.0,
                objectness_loss_coef=1.0,
                objectness_focal_loss_alpha=0.2,
                objectness_focal_loss_gamma=1.5,
                objectness_focal_loss_prob_margin=0.1,
            )
        ),
    )
    head = AudioSegmentationHead.from_config(config)
    calls = []

    def recording_loss(logits, targets, **kwargs):
        calls.append(kwargs)
        return torch.zeros_like(logits)

    head._segmentation_loss(
        torch.zeros(1, 3, C),
        torch.tensor([[[0.2, 0.6], [0.0, 0.0], [0.0, 0.0]]]),
        torch.zeros(1, 3),
        torch.ones(1, 3, dtype=torch.bool),
        torch.ones(1, C),
        torch.tensor([[0]]),
        torch.tensor([[[0.2, 0.6]]]),
        torch.ones(1, 1),
        base_loss_fn=recording_loss,
    )

    assert calls[0] == {}
    assert calls[1] == {
        "focal_loss_alpha": 0.2,
        "focal_loss_gamma": 1.5,
        "focal_loss_prob_margin": 0.1,
    }


def test_audio_segmentation_decoder_is_multilabel_and_float32_by_default():
    config = SimpleNamespace(
        audio_segmentation_config=AudioSegmentationHeadConfig(),
    )
    decoder = AudioSegmentationDecoder(config)
    output = SimpleNamespace(
        audio_segmentation_logits=torch.tensor([[[8.0, 7.0]]], dtype=torch.float16),
        audio_segmentation_segments=torch.tensor([[[0.2, 0.6]]]),
        audio_segmentation_batch_origin=torch.tensor([0]),
        audio_segmentation_objectness_logits=torch.tensor([[8.0]], dtype=torch.float16),
        audio_segmentation_anchor_mask=torch.ones(1, 1, dtype=torch.bool),
        batch_size=1,
    )

    decoded = decoder.decode(output, threshold=0.9)

    assert [prediction["label"] for prediction in decoded[0][0]] == ["0", "1"]
    assert all(prediction["score"] > 0.9 for prediction in decoded[0][0])
