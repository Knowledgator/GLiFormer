from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from .base import hidden_size
from .text import TextTransformer


def _get_config_value(config: Any, name: str, default: Any = None) -> Any:
    return getattr(config, name, default) if config is not None else default


def _hidden_size(model_config: Any, default: int) -> int:
    for name in ("hidden_size", "audio_hidden_size", "d_model", "encoder_embed_dim"):
        value = getattr(model_config, name, None)
        if value is not None:
            return int(value)
    return int(default)


class ConvAudioEncoder(nn.Module):
    """Small local waveform encoder returning frame-level embeddings."""

    def __init__(
        self,
        hidden_size: int,
        in_channels: int = 1,
        num_layers: int = 3,
        stride: int = 4,
    ) -> None:
        super().__init__()
        layers = []
        channels = int(in_channels)
        for layer_idx in range(int(num_layers)):
            out_channels = int(hidden_size)
            layers.extend(
                [
                    nn.Conv1d(
                        channels,
                        out_channels,
                        kernel_size=5,
                        stride=stride if layer_idx == 0 else 1,
                        padding=2,
                    ),
                    nn.GELU(),
                    nn.GroupNorm(1, out_channels),
                ]
            )
            channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.config = type("ConvAudioEncoderConfig", (), {"hidden_size": int(hidden_size)})()

    def forward(self, input_values: torch.Tensor, **_: Any) -> tuple[torch.Tensor]:
        if input_values.dim() == 2:
            input_values = input_values.unsqueeze(1)
        features = self.conv(input_values)
        token_embeddings = features.transpose(1, 2).contiguous()
        return (token_embeddings,)


class MelConvAudioEncoder(nn.Module):
    """Small local spectrogram/mel encoder returning time-frame embeddings.

    Inputs are expected as ``(batch, mel_bins, frames)`` by default, or
    ``(batch, channels, mel_bins, frames)`` when ``in_channels`` is greater than
    one. Set ``input_format="time_first"`` for ``(batch, frames, mel_bins)``.
    The 2D convolution preserves the time axis as the token sequence and pools
    the frequency axis before returning ``(batch, frames', hidden_size)``.
    """

    def __init__(
        self,
        hidden_size: int,
        in_channels: int = 1,
        num_layers: int = 3,
        freq_stride: int = 2,
        time_stride: int = 2,
        input_format: str = "freq_first",
    ) -> None:
        super().__init__()
        if input_format not in {"freq_first", "time_first"}:
            raise ValueError("audio_input_format must be 'freq_first' or 'time_first'")
        self.input_format = input_format

        layers = []
        channels = int(in_channels)
        first_stride = (int(freq_stride), int(time_stride))
        for layer_idx in range(int(num_layers)):
            out_channels = int(hidden_size)
            layers.extend(
                [
                    nn.Conv2d(
                        channels,
                        out_channels,
                        kernel_size=3,
                        stride=first_stride if layer_idx == 0 else 1,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.GroupNorm(1, out_channels),
                ]
            )
            channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.config = type("MelConvAudioEncoderConfig", (), {"hidden_size": int(hidden_size)})()

    def _normalize_input(self, input_values: torch.Tensor) -> torch.Tensor:
        if input_values.dim() == 3:
            # B x F x T by default, or B x T x F for time_first.
            if self.input_format == "time_first":
                input_values = input_values.transpose(1, 2)
            return input_values.unsqueeze(1)
        if input_values.dim() == 4:
            # B x C x F x T by default, or B x C x T x F for time_first.
            if self.input_format == "time_first":
                input_values = input_values.transpose(2, 3)
            return input_values
        raise ValueError(
            "MelConvAudioEncoder expects input_values with shape "
            "(batch, mel_bins, frames) or (batch, channels, mel_bins, frames)"
        )

    def forward(self, input_values: torch.Tensor, **_: Any) -> tuple[torch.Tensor]:
        input_values = self._normalize_input(input_values)
        features = self.conv(input_values)
        features = features.mean(dim=2)
        token_embeddings = features.transpose(1, 2).contiguous()
        return (token_embeddings,)


class AudioEncoder(nn.Module):
    """Audio encoder with a GLiNER-style token embedding interface.

    The forward pass accepts waveform ``input_values`` for ``conv`` or mel /
    spectrogram features for ``mel``/``spectrogram``/``conv2d`` and returns
    embeddings with shape ``(batch, audio_frames, config.hidden_size)``.
    """

    def __init__(
        self,
        config: Any,
        model_name: Optional[str] = None,
        from_pretrained: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = int(_get_config_value(config, "hidden_size"))
        self.model_name = model_name or _get_config_value(config, "audio_model_name")
        self.encoder_type = _get_config_value(config, "audio_encoder_type", None) or (
            "auto" if self.model_name else "conv"
        )
        if isinstance(self.encoder_type, str):
            self.encoder_type = self.encoder_type.lower().replace("-", "_")
            if self.encoder_type in {"mel_conv", "spectrogram_conv", "conv_2d"}:
                self.encoder_type = "conv2d"

        self.model = self._build_model(from_pretrained=from_pretrained, cache_dir=cache_dir)
        model_hidden_size = _hidden_size(getattr(self.model, "config", None), self.hidden_size)
        if model_hidden_size != self.hidden_size:
            self.projection = nn.Linear(model_hidden_size, self.hidden_size)

    def _build_model(
        self,
        from_pretrained: bool,
        cache_dir: Optional[Union[str, Path]],
    ) -> nn.Module:
        if self.encoder_type == "conv":
            return ConvAudioEncoder(
                hidden_size=self.hidden_size,
                in_channels=int(_get_config_value(self.config, "audio_in_channels", 1)),
                num_layers=int(_get_config_value(self.config, "audio_num_layers", 3)),
                stride=int(_get_config_value(self.config, "audio_stride", 4)),
            )
        if self.encoder_type in {"mel", "spectrogram", "conv2d"}:
            time_stride = _get_config_value(self.config, "audio_time_stride")
            if time_stride is None:
                time_stride = 2
            return MelConvAudioEncoder(
                hidden_size=self.hidden_size,
                in_channels=int(_get_config_value(self.config, "audio_in_channels", 1)),
                num_layers=int(_get_config_value(self.config, "audio_num_layers", 3)),
                freq_stride=int(_get_config_value(self.config, "audio_freq_stride", 2)),
                time_stride=int(time_stride),
                input_format=str(_get_config_value(self.config, "audio_input_format", "freq_first")),
            )

        audio_config = _get_config_value(self.config, "audio_encoder_config")
        if audio_config is None:
            if self.model_name is None:
                raise ValueError(
                    "audio_model_name is required when audio_encoder_type is not "
                    "'conv', 'mel', 'spectrogram', or 'conv2d'"
                )
            audio_config = AutoConfig.from_pretrained(
                self.model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )

        if from_pretrained:
            if self.model_name is None:
                raise ValueError("audio_model_name is required to load pretrained audio weights")
            return AutoModel.from_pretrained(
                self.model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
        return AutoModel.from_config(audio_config, trust_remote_code=True)

    @staticmethod
    def _extract_sequence(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            return output.last_hidden_state
        if isinstance(output, (tuple, list)) and output:
            return output[0]
        raise ValueError("Audio backbone did not return token embeddings")

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        model_kwargs = dict(kwargs)
        if attention_mask is not None and self.encoder_type != "conv":
            model_kwargs["attention_mask"] = attention_mask
        output = self.model(input_values=input_values, **model_kwargs)
        token_embeddings = self._extract_sequence(output)
        if hasattr(self, "projection"):
            token_embeddings = self.projection(token_embeddings)
        return token_embeddings


class AudioBiEncoder(nn.Module):
    """Bi-encoder for audio GLiNExT models.

    Audio inputs are encoded by ``AudioEncoder``. Label names are encoded by a
    text transformer and mean-pooled into label embeddings.
    """

    def __init__(
        self,
        config: Any,
        from_pretrained: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.audio_encoder = AudioEncoder(
            config,
            from_pretrained=from_pretrained,
            cache_dir=cache_dir,
        )
        label_model_name = _get_config_value(config, "labels_encoder") or _get_config_value(config, "model_name")
        if label_model_name is None:
            raise ValueError("AudioBiEncoder requires config.labels_encoder or config.model_name for label encoding")
        self.labels_encoder = TextTransformer(
            label_model_name,
            config,
            from_pretrained=from_pretrained,
            labels_encoder=True,
            cache_dir=cache_dir,
        )
        label_hidden_size = hidden_size(self.labels_encoder.model.config)
        if int(_get_config_value(config, "hidden_size")) != label_hidden_size:
            self.labels_projection = nn.Linear(label_hidden_size, int(_get_config_value(config, "hidden_size")))

    @staticmethod
    def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1),
            min=1e-9,
        )

    def resize_token_embeddings(
        self,
        new_num_tokens: int,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        return self.labels_encoder.model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.labels_encoder.model.get_input_embeddings()

    def encode_labels(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        label_kwargs = dict(kwargs)
        label_kwargs.pop("packing_config", None)
        label_kwargs.pop("pair_attention_mask", None)
        label_tokens = self.labels_encoder(input_ids, attention_mask=attention_mask, **label_kwargs)
        if hasattr(self, "labels_projection"):
            label_tokens = self.labels_projection(label_tokens)
        return self.mean_pooling(label_tokens, attention_mask)

    def forward(
        self,
        audio_values: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        labels_input_ids: Optional[torch.Tensor] = None,
        labels_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        audio_tokens = self.audio_encoder(audio_values, attention_mask=audio_attention_mask, **kwargs)
        if labels_input_ids is None or labels_attention_mask is None:
            return audio_tokens
        labels_embeddings = self.encode_labels(labels_input_ids, labels_attention_mask)
        return audio_tokens, labels_embeddings
