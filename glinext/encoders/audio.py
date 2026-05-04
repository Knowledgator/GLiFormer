from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


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


class AudioEncoder(nn.Module):
    """Audio encoder with a GLiNER-style token embedding interface.

    The forward pass accepts ``input_values``/waveforms and returns embeddings with
    shape ``(batch, audio_frames, config.hidden_size)``.
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

        audio_config = _get_config_value(self.config, "audio_encoder_config")
        if audio_config is None:
            if self.model_name is None:
                raise ValueError("audio_model_name is required when audio_encoder_type is not 'conv'")
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
