from pathlib import Path
from typing import Any, Optional, Union
import collections.abc

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel


def _get_config_value(config: Any, name: str, default: Any = None) -> Any:
    return getattr(config, name, default) if config is not None else default


def _hidden_size(model_config: Any, default: int) -> int:
    for name in ("hidden_size", "vision_hidden_size", "projection_dim", "d_model"):
        value = getattr(model_config, name, None)
        if value is not None:
            return int(value)
    return int(default)


def _pair(value: Any) -> tuple[int, int]:
    if isinstance(value, collections.abc.Iterable) and not isinstance(value, (str, bytes)):
        values = tuple(value)
        if len(values) != 2:
            raise ValueError(f"Expected a 2-item size, got {value!r}")
        return int(values[0]), int(values[1])
    return int(value), int(value)


class VisionPathEmbeddings(nn.Module):
    """Patch-token vision embeddings for the local GLiNExT vision path."""

    def __init__(self, config: Any):
        super().__init__()
        image_size = _pair(_get_config_value(config, "vision_input_size", _get_config_value(config, "image_size", 224)))
        patch_size = _pair(_get_config_value(config, "vision_patch_size", _get_config_value(config, "patch_size", 16)))
        num_channels = int(_get_config_value(config, "vision_in_channels", _get_config_value(config, "num_channels", 3)))
        hidden_size = int(_get_config_value(config, "hidden_size"))

        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]
        self.proj = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.config = type(
            "VisionPathEmbeddingsConfig",
            (),
            {
                "hidden_size": hidden_size,
                "image_size": image_size,
                "patch_size": patch_size,
                "num_channels": num_channels,
            },
        )()

        if bool(_get_config_value(config, "vision_position_embeddings", True)):
            self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
            nn.init.trunc_normal_(self.position_embeddings, std=0.02)
        else:
            self.position_embeddings = None

    def _position_embeddings(self, height: int, width: int, position_embedding: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if position_embedding is None:
            position_embedding = self.position_embeddings
        if position_embedding is None:
            return None

        position_embedding = position_embedding.view(1, self.patch_shape[0], self.patch_shape[1], -1)
        position_embedding = position_embedding.permute(0, 3, 1, 2)
        position_embedding = F.interpolate(position_embedding, size=(height, width), mode="bicubic", align_corners=False)
        return position_embedding.flatten(2).transpose(1, 2)

    def forward(
        self,
        pixel_values: torch.Tensor,
        position_embedding: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> torch.Tensor:
        embeddings = self.proj(pixel_values)
        patch_height, patch_width = embeddings.shape[2], embeddings.shape[3]
        embeddings = embeddings.flatten(2).transpose(1, 2)
        position_embedding = self._position_embeddings(patch_height, patch_width, position_embedding)
        if position_embedding is not None:
            embeddings = embeddings + position_embedding.to(device=embeddings.device, dtype=embeddings.dtype)
        return embeddings.contiguous()


class VisionEncoder(nn.Module):
    """Vision encoder with a GLiNER-style token embedding interface.

    The forward pass accepts ``pixel_values`` and returns token embeddings with
    shape ``(batch, visual_tokens, config.hidden_size)``. For Hugging Face vision
    models, ``last_hidden_state`` is used when available. For the local patch
    fallback, image patches are projected into visual tokens.
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
        self.model_name = model_name or _get_config_value(config, "vision_model_name")
        self.encoder_type = _get_config_value(config, "vision_encoder_type", None) or (
            "auto" if self.model_name else "patch"
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
        if self.encoder_type in {"patch", "path", "cnn"}:
            return VisionPathEmbeddings(self.config)

        vision_config = _get_config_value(self.config, "vision_encoder_config")
        if vision_config is None:
            if self.model_name is None:
                raise ValueError("vision_model_name is required when vision_encoder_type is not 'patch'")
            vision_config = AutoConfig.from_pretrained(
                self.model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )

        if from_pretrained:
            if self.model_name is None:
                raise ValueError("vision_model_name is required to load pretrained vision weights")
            return AutoModel.from_pretrained(
                self.model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
        return AutoModel.from_config(vision_config, trust_remote_code=True)

    @staticmethod
    def _extract_sequence(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            return output.last_hidden_state
        if isinstance(output, (tuple, list)) and output:
            return output[0]
        raise ValueError("Vision backbone did not return token embeddings")

    def forward(self, pixel_values: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        output = self.model(pixel_values=pixel_values, **kwargs)
        token_embeddings = self._extract_sequence(output)
        if hasattr(self, "projection"):
            token_embeddings = self.projection(token_embeddings)
        return token_embeddings
