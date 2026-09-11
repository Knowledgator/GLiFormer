from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

import gliformer.encoders.media as media_module
from gliformer.config import AudioClassificationHeadConfig, GLiFormerConfig
from gliformer.encoders.media import MediaBackboneEncoder
from gliformer.encoders.audio import (
    AudioEncoder,
    ConvAudioEncoder,
    MelConvAudioEncoder,
    audio_token_mask,
)
from gliformer.processing.processor import GLiFormerProcessor
from tests.conftest import FakeWordsSplitter


class FakeTokenizer:
    pad_token = "[PAD]"
    unk_token = "[UNK]"


class _ExternalAudioBackbone(torch.nn.Module):
    def __init__(self, hidden_size=12):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.attention_mask = None

    def forward(self, input_values, attention_mask=None, **kwargs):
        self.attention_mask = attention_mask
        return (
            torch.zeros(
                input_values.shape[0],
                input_values.shape[-1],
                self.config.hidden_size,
                device=input_values.device,
            ),
        )


def test_external_audio_backbone_uses_shared_construction_and_projection(
    monkeypatch,
):
    external_config = SimpleNamespace(hidden_size=12)
    backbone = _ExternalAudioBackbone(hidden_size=12)

    class FakeAutoModel:
        @staticmethod
        def from_config(model_config, **kwargs):
            assert model_config is external_config
            assert kwargs == {"trust_remote_code": True}
            return backbone

    monkeypatch.setattr(media_module, "AutoModel", FakeAutoModel)
    encoder = AudioEncoder(
        SimpleNamespace(
            hidden_size=8,
            audio_model_name=None,
            audio_encoder_type="auto",
            audio_encoder_config=external_config,
        )
    )
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])

    token_embeddings = encoder(
        torch.randn(2, 4),
        attention_mask=attention_mask,
    )

    assert isinstance(encoder, MediaBackboneEncoder)
    assert encoder.model is backbone
    assert token_embeddings.shape == (2, 4, 8)
    assert torch.equal(backbone.attention_mask, attention_mask.bool())
    assert encoder.projection.in_features == 12
    assert encoder.projection.out_features == 8


def test_mel_conv_audio_encoder_returns_time_tokens():
    config = GLiFormerConfig(
        model_name="unused",
        hidden_size=8,
        default_ner_config=False,
        audio_encoder_type="mel",
        audio_num_layers=2,
        audio_freq_stride=2,
        audio_time_stride=2,
    )
    encoder = AudioEncoder(config)

    token_embeddings = encoder(torch.randn(2, 80, 31))

    assert isinstance(encoder.model, MelConvAudioEncoder)
    assert token_embeddings.shape == (2, 16, 8)


def test_audio_token_mask_uses_encoder_output_lengths():
    encoder = ConvAudioEncoder(hidden_size=8, stride=4)
    tokens = torch.zeros(2, 4, 8)
    input_mask = torch.tensor(
        [
            [1] * 16,
            [1] * 9 + [0] * 7,
        ]
    )

    mask = audio_token_mask(encoder, tokens, input_mask)

    assert mask.tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]


def test_audio_token_mask_fails_closed_without_length_transform():
    with pytest.raises(ValueError, match="output-length transform"):
        audio_token_mask(
            torch.nn.Identity(),
            torch.zeros(1, 4, 8),
            torch.ones(1, 16),
        )


def test_audio_token_mask_rejects_non_right_padded_input():
    with pytest.raises(ValueError, match="must be right-padded.*non-prefix rows: \\[0\\]"):
        audio_token_mask(
            ConvAudioEncoder(hidden_size=8, stride=1),
            torch.zeros(1, 4, 8),
            torch.tensor([[1, 0, 1, 0]]),
        )


@pytest.mark.parametrize(
    ("encoder", "inputs"),
    [
        (ConvAudioEncoder(hidden_size=8, stride=1), torch.randn(1, 4)),
        (MelConvAudioEncoder(hidden_size=8, time_stride=1), torch.randn(1, 8, 4)),
    ],
)
def test_local_audio_encoders_reject_non_right_padded_input(encoder, inputs):
    with pytest.raises(ValueError, match="must be right-padded"):
        encoder(inputs, attention_mask=torch.tensor([[1, 0, 1, 0]]))


@pytest.mark.parametrize("encoder_kind", ["waveform", "mel"])
def test_local_audio_encoder_valid_tokens_are_batch_padding_invariant(
    encoder_kind,
):
    torch.manual_seed(7)
    if encoder_kind == "waveform":
        encoder = ConvAudioEncoder(
            hidden_size=8,
            num_layers=3,
            stride=4,
        ).eval()
        short = torch.randn(1, 17)
        padded = torch.nn.functional.pad(short, (0, 16))
    else:
        encoder = MelConvAudioEncoder(
            hidden_size=8,
            num_layers=3,
            time_stride=2,
        ).eval()
        short = torch.randn(1, 12, 17)
        padded = torch.nn.functional.pad(short, (0, 16))
    short_mask = torch.ones(1, 17, dtype=torch.long)
    padded_mask = torch.tensor([[1] * 17 + [0] * 16])

    short_tokens = encoder(short, attention_mask=short_mask)[0]
    padded_tokens = encoder(padded, attention_mask=padded_mask)[0]
    valid_length = int(encoder.output_lengths(torch.tensor([17]))[0].item())

    assert torch.allclose(
        short_tokens[:, :valid_length],
        padded_tokens[:, :valid_length],
        atol=1e-6,
        rtol=1e-6,
    )


def test_audio_processor_keeps_mel_features_2d():
    config = GLiFormerConfig(
        model_name="unused",
        hidden_size=8,
        model_variant="audio",
        audio_encoder_type="mel",
        audio_classification_config=asdict(AudioClassificationHeadConfig()),
    )
    processor = GLiFormerProcessor(config, FakeTokenizer(), FakeWordsSplitter())
    batch = processor.collate_raw_batch(
        [
            {
                "tokenized_text": ["[PAD]"],
                "audio_values": torch.ones(80, 12),
                "labels": ["music"],
                "true_labels": ["music"],
            },
            {
                "tokenized_text": ["[PAD]"],
                "audio_values": torch.ones(64, 8),
                "labels": ["speech"],
                "true_labels": ["speech"],
            },
        ]
    )

    assert batch["audio_values"].shape == (2, 80, 12)
    assert batch["audio_attention_mask"].shape == (2, 12)
    assert batch["audio_attention_mask"][0].sum().item() == 12
    assert batch["audio_attention_mask"][1].sum().item() == 8
