"""Shared behavior for single-modality media bi-encoders."""

from types import SimpleNamespace

import torch
from torch import nn

import glinext.encoders.media as media_encoders
from glinext.encoders.audio import AudioBiEncoder
from glinext.encoders.media import MediaBiEncoder
from glinext.encoders.vision import VisionBiEncoder


def test_vision_label_mean_pooling_preserves_embedding_dtype():
    token_embeddings = torch.randn(2, 4, 8, dtype=torch.float16)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]])

    pooled = VisionBiEncoder.mean_pooling(token_embeddings, attention_mask)

    assert pooled.dtype == torch.float16


def test_audio_label_mean_pooling_preserves_embedding_dtype():
    token_embeddings = torch.randn(2, 4, 8, dtype=torch.float16)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]])

    pooled = AudioBiEncoder.mean_pooling(token_embeddings, attention_mask)

    assert pooled.dtype == torch.float16


def test_single_media_encoders_share_label_encoder_contract():
    assert issubclass(VisionBiEncoder, MediaBiEncoder)
    assert issubclass(AudioBiEncoder, MediaBiEncoder)
    assert VisionBiEncoder.encode_labels is MediaBiEncoder.encode_labels
    assert AudioBiEncoder.encode_labels is MediaBiEncoder.encode_labels
    assert VisionBiEncoder.mean_pooling is MediaBiEncoder.mean_pooling
    assert AudioBiEncoder.mean_pooling is MediaBiEncoder.mean_pooling


class _FakeLabelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=6)
        self.embeddings = nn.Embedding(16, 6)

    def get_input_embeddings(self):
        return self.embeddings


class _FakeTextTransformer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = _FakeLabelModel()
        self.last_kwargs = None

    def forward(self, input_ids, attention_mask=None, **kwargs):
        self.last_kwargs = kwargs
        return self.model.embeddings(input_ids)


def _media_config():
    return SimpleNamespace(
        hidden_size=4,
        image_size=16,
        vision_patch_size=8,
        vision_position_embedding_type="none",
        vision_encoder_type="patch",
        vision_model_name=None,
        audio_encoder_type="conv",
        audio_model_name=None,
        audio_in_channels=1,
        audio_num_layers=1,
        audio_stride=2,
        labels_encoder="fake-labels",
        model_name="unused",
    )


def test_media_biencoder_preserves_modality_specific_checkpoint_names(monkeypatch):
    monkeypatch.setattr(media_encoders, "TextTransformer", _FakeTextTransformer)
    config = _media_config()

    vision_encoder = VisionBiEncoder(config)
    audio_encoder = AudioBiEncoder(config)

    vision_keys = set(vision_encoder.state_dict())
    audio_keys = set(audio_encoder.state_dict())
    assert any(key.startswith("vision_encoder.") for key in vision_keys)
    assert any(key.startswith("audio_encoder.") for key in audio_keys)
    assert any(key.startswith("labels_encoder.") for key in vision_keys)
    assert any(key.startswith("labels_projection.") for key in vision_keys)
    assert not any(key.startswith("media_encoder.") for key in vision_keys | audio_keys)


def test_shared_media_biencoder_forward_and_label_kwarg_filtering(monkeypatch):
    monkeypatch.setattr(media_encoders, "TextTransformer", _FakeTextTransformer)
    config = _media_config()
    label_ids = torch.tensor([[1, 2, 0], [3, 0, 0]])
    label_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    vision_encoder = VisionBiEncoder(config)
    vision_tokens, vision_labels = vision_encoder(
        torch.randn(2, 3, 16, 16),
        labels_input_ids=label_ids,
        labels_attention_mask=label_mask,
    )
    audio_encoder = AudioBiEncoder(config)
    audio_tokens, audio_labels = audio_encoder(
        torch.randn(2, 16),
        audio_attention_mask=torch.ones(2, 16),
        labels_input_ids=label_ids,
        labels_attention_mask=label_mask,
    )

    assert vision_tokens.shape == (2, 4, 4)
    assert audio_tokens.shape == (2, 8, 4)
    assert vision_labels.shape == audio_labels.shape == (2, 4)

    vision_encoder.encode_labels(
        label_ids,
        label_mask,
        packing_config={"enabled": True},
        pair_attention_mask=torch.ones(2, 3, 3),
        output_hidden_states=True,
    )
    assert vision_encoder.labels_encoder.last_kwargs == {"output_hidden_states": True}
