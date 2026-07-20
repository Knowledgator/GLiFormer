from typing import Any, Optional

import torch
from torch import nn

from .media import MediaBackboneEncoder, MediaBiEncoder, get_config_value


def validate_audio_attention_mask(
    attention_mask: Optional[torch.Tensor],
    *,
    batch_size: int,
    time_length: Optional[int] = None,
    name: str = "audio_attention_mask",
) -> Optional[torch.Tensor]:
    """Validate and normalize a right-padded audio attention mask.

    Audio length transforms and temporal heads represent each row by its valid
    prefix length. Reject masks with holes instead of silently treating their
    sum as that prefix length.
    """

    if attention_mask is None:
        return None
    if attention_mask.dim() != 2 or attention_mask.shape[0] != batch_size:
        expected = f"({batch_size}, input_time)"
        raise ValueError(f"{name} must have shape {expected}")
    if time_length is not None and attention_mask.shape[1] != time_length:
        raise ValueError(f"{name} must have shape ({batch_size}, {time_length})")

    valid = attention_mask.bool()
    if valid.shape[1] > 1:
        non_prefix_rows = ((~valid[:, :-1]) & valid[:, 1:]).any(dim=-1)
        if non_prefix_rows.any().item():
            row_indices = non_prefix_rows.nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                f"{name} must be right-padded (all valid positions before all "
                f"padding positions); non-prefix rows: {row_indices}"
            )
    return valid


def _convolution_output_lengths(
    input_lengths: torch.Tensor,
    convolution: nn.Module,
    dimension: int,
) -> torch.Tensor:
    kernel_size = convolution.kernel_size[dimension]
    stride = convolution.stride[dimension]
    padding = convolution.padding[dimension]
    dilation = convolution.dilation[dimension]
    return torch.div(
        input_lengths
        + 2 * padding
        - dilation * (kernel_size - 1)
        - 1,
        stride,
        rounding_mode="floor",
    ) + 1


class _ChannelLayerNorm1d(nn.LayerNorm):
    """Layer-normalize channels independently at each temporal position."""

    def __init__(self, channels: int):
        super().__init__(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return super().forward(inputs.transpose(1, 2)).transpose(1, 2)


class _ChannelLayerNorm2d(nn.LayerNorm):
    """Layer-normalize channels independently at each time/frequency cell."""

    def __init__(self, channels: int):
        super().__init__(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return super().forward(inputs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def audio_token_mask(
    encoder: nn.Module,
    token_embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Align an input-time audio mask with an encoder's output token sequence.

    Encoders that change temporal resolution must expose ``output_lengths``;
    silently unmasking padded frames would let padding affect every downstream
    audio head and makes temporal coordinates batch-dependent.
    """

    batch_size, token_count = token_embeddings.shape[:2]
    if attention_mask is None:
        return torch.ones(
            (batch_size, token_count),
            dtype=torch.long,
            device=token_embeddings.device,
        )
    attention_mask = validate_audio_attention_mask(
        attention_mask,
        batch_size=batch_size,
    ).to(device=token_embeddings.device)
    if attention_mask.shape[1] == token_count:
        return attention_mask.long()
    output_lengths_fn = getattr(encoder, "output_lengths", None)
    if not callable(output_lengths_fn):
        raise ValueError(
            "The configured audio encoder changes sequence length but does not "
            "expose an output-length transform"
        )
    input_lengths = attention_mask.long().sum(dim=-1)
    output_lengths = output_lengths_fn(input_lengths).to(
        device=token_embeddings.device,
        dtype=torch.long,
    )
    if output_lengths.shape != input_lengths.shape:
        raise ValueError("audio encoder output_lengths must preserve the batch shape")
    output_lengths = output_lengths.clamp(min=0, max=token_count)
    return (
        torch.arange(token_count, device=token_embeddings.device)[None]
        < output_lengths[:, None]
    ).long()


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
                    _ChannelLayerNorm1d(out_channels),
                ]
            )
            channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.config = type("ConvAudioEncoderConfig", (), {"hidden_size": int(hidden_size)})()

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> tuple[torch.Tensor]:
        if input_values.dim() == 2:
            input_values = input_values.unsqueeze(1)
        valid_lengths = None
        if attention_mask is not None:
            attention_mask = validate_audio_attention_mask(
                attention_mask,
                batch_size=input_values.shape[0],
                time_length=input_values.shape[-1],
                name="waveform attention_mask",
            ).to(device=input_values.device)
            input_values = input_values * attention_mask[:, None].to(input_values.dtype)
            valid_lengths = attention_mask.long().sum(dim=-1)
        features = input_values
        for layer_idx in range(0, len(self.conv), 3):
            convolution = self.conv[layer_idx]
            features = convolution(features)
            features = self.conv[layer_idx + 1](features)
            features = self.conv[layer_idx + 2](features)
            if valid_lengths is not None:
                valid_lengths = _convolution_output_lengths(
                    valid_lengths,
                    convolution,
                    0,
                ).clamp(min=0, max=features.shape[-1])
                valid = (
                    torch.arange(features.shape[-1], device=features.device)[None]
                    < valid_lengths[:, None]
                )
                features = features * valid[:, None].to(features.dtype)
        token_embeddings = features.transpose(1, 2).contiguous()
        return (token_embeddings,)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        lengths = input_lengths
        for layer_idx in range(0, len(self.conv), 3):
            lengths = _convolution_output_lengths(
                lengths,
                self.conv[layer_idx],
                0,
            )
        return lengths


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
                    _ChannelLayerNorm2d(out_channels),
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

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> tuple[torch.Tensor]:
        input_values = self._normalize_input(input_values)
        valid_lengths = None
        if attention_mask is not None:
            attention_mask = validate_audio_attention_mask(
                attention_mask,
                batch_size=input_values.shape[0],
                time_length=input_values.shape[-1],
                name="feature attention_mask",
            ).to(device=input_values.device)
            input_values = input_values * attention_mask[:, None, None].to(
                input_values.dtype
            )
            valid_lengths = attention_mask.long().sum(dim=-1)
        features = input_values
        for layer_idx in range(0, len(self.conv), 3):
            convolution = self.conv[layer_idx]
            features = convolution(features)
            features = self.conv[layer_idx + 1](features)
            features = self.conv[layer_idx + 2](features)
            if valid_lengths is not None:
                valid_lengths = _convolution_output_lengths(
                    valid_lengths,
                    convolution,
                    1,
                ).clamp(min=0, max=features.shape[-1])
                valid = (
                    torch.arange(features.shape[-1], device=features.device)[None]
                    < valid_lengths[:, None]
                )
                features = features * valid[:, None, None].to(features.dtype)
        features = features.mean(dim=2)
        token_embeddings = features.transpose(1, 2).contiguous()
        return (token_embeddings,)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        lengths = input_lengths
        for layer_idx in range(0, len(self.conv), 3):
            lengths = _convolution_output_lengths(
                lengths,
                self.conv[layer_idx],
                1,
            )
        return lengths


class AudioEncoder(MediaBackboneEncoder):
    """Audio encoder with a GLiNER-style token embedding interface.

    The forward pass accepts waveform ``input_values`` for ``conv`` or mel /
    spectrogram features for ``mel``/``spectrogram``/``conv2d`` and returns
    embeddings with shape ``(batch, audio_frames, config.hidden_size)``.
    """

    modality = "audio"
    model_name_config_key = "audio_model_name"
    encoder_type_config_key = "audio_encoder_type"
    encoder_config_key = "audio_encoder_config"
    default_local_encoder_type = "conv"
    additional_hidden_size_names = ("audio_hidden_size",)

    def _normalize_encoder_type(self, encoder_type: Any) -> Any:
        if isinstance(encoder_type, str):
            encoder_type = encoder_type.lower().replace("-", "_")
            if encoder_type in {"mel_conv", "spectrogram_conv", "conv_2d"}:
                encoder_type = "conv2d"
        return encoder_type

    def _build_local_model(self) -> nn.Module | None:
        if self.encoder_type == "conv":
            return ConvAudioEncoder(
                hidden_size=self.hidden_size,
                in_channels=int(get_config_value(self.config, "audio_in_channels", 1)),
                num_layers=int(get_config_value(self.config, "audio_num_layers", 3)),
                stride=int(get_config_value(self.config, "audio_stride", 4)),
            )
        if self.encoder_type in {"mel", "spectrogram", "conv2d"}:
            time_stride = get_config_value(self.config, "audio_time_stride")
            if time_stride is None:
                time_stride = 2
            return MelConvAudioEncoder(
                hidden_size=self.hidden_size,
                in_channels=int(get_config_value(self.config, "audio_in_channels", 1)),
                num_layers=int(get_config_value(self.config, "audio_num_layers", 3)),
                freq_stride=int(get_config_value(self.config, "audio_freq_stride", 2)),
                time_stride=int(time_stride),
                input_format=str(get_config_value(self.config, "audio_input_format", "freq_first")),
            )
        return None

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        model_kwargs = dict(kwargs)
        if attention_mask is not None:
            attention_mask = validate_audio_attention_mask(
                attention_mask,
                batch_size=input_values.shape[0],
            )
            model_kwargs["attention_mask"] = attention_mask
        output = self.model(input_values=input_values, **model_kwargs)
        token_embeddings = self._extract_token_sequence(output)
        return self._project_token_sequence(token_embeddings)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Map valid input lengths to this backbone's token sequence lengths."""

        if hasattr(self.model, "output_lengths"):
            return self.model.output_lengths(input_lengths).long()
        length_fn = getattr(
            self.model,
            "_get_feat_extract_output_lengths",
            None,
        )
        if callable(length_fn):
            return length_fn(input_lengths).long()
        raise ValueError(
            "The configured audio backbone changes sequence length but does not "
            "expose an output-length transform"
        )


class AudioBiEncoder(MediaBiEncoder):
    """Bi-encoder for audio GLiNExT models.

    Audio inputs are encoded by ``AudioEncoder``. Label names are encoded by a
    text transformer and mean-pooled into label embeddings.
    """

    media_encoder_cls = AudioEncoder
    media_encoder_attribute = "audio_encoder"

    def forward(
        self,
        audio_values: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        labels_input_ids: Optional[torch.Tensor] = None,
        labels_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        return self.forward_media(
            audio_values,
            media_attention_mask=audio_attention_mask,
            labels_input_ids=labels_input_ids,
            labels_attention_mask=labels_attention_mask,
            **kwargs,
        )
