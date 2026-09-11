from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..layers.position import PositionEmbedding, normalized_grid_2d
from ..utils import pair_2d
from .media import MediaBackboneEncoder, MediaBiEncoder, get_config_value


_VISION_ENCODER_KWARGS = {
    "bool_masked_pos",
    "head_mask",
    "interpolate_pos_encoding",
    "output_attentions",
    "output_hidden_states",
    "pixel_mask",
    "return_dict",
}


def vision_encoder_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Select explicit vision-backbone kwargs from a heterogeneous model batch."""

    selected = {
        key: value
        for key, value in kwargs.items()
        if key in _VISION_ENCODER_KWARGS and value is not None
    }
    nested = kwargs.get("vision_encoder_kwargs")
    if nested is not None:
        if not isinstance(nested, dict):
            raise TypeError("vision_encoder_kwargs must be a dictionary")
        selected.update(nested)
    return selected


def vision_token_mask(
    token_embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    prefix_tokens: Optional[torch.Tensor] = None,
    spatial_shape: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Align either full-sequence or dense-only masks with vision tokens."""

    if attention_mask is not None:
        attention_mask = attention_mask.to(device=token_embeddings.device)
        if attention_mask.dim() == 3:
            if spatial_shape is None or spatial_shape.numel() == 0:
                raise ValueError(
                    "A spatial vision mask requires feature_spatial_shape metadata"
                )
            rows, cols = (int(value) for value in spatial_shape[0].tolist())
            attention_mask = F.interpolate(
                attention_mask[:, None].float(),
                size=(rows, cols),
                mode="nearest",
            )[:, 0].flatten(1).to(dtype=torch.long)
        elif attention_mask.dim() != 2:
            raise ValueError(
                "vision_attention_mask must have shape (B, tokens) or (B, H, W)"
            )
        if attention_mask.shape[-1] == token_embeddings.shape[1]:
            return attention_mask
        if prefix_tokens is not None and prefix_tokens.numel() > 0:
            prefix_count = int(prefix_tokens[0].item())
            uniform_prefix = torch.all(prefix_tokens == prefix_count)
            if (
                bool(uniform_prefix)
                and attention_mask.shape[-1] + prefix_count
                == token_embeddings.shape[1]
            ):
                prefix_mask = torch.ones(
                    attention_mask.shape[0],
                    prefix_count,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                return torch.cat([prefix_mask, attention_mask], dim=-1)
        raise ValueError(
            "vision_attention_mask cannot be aligned with the backbone token sequence"
        )
    return torch.ones(
        token_embeddings.shape[:2],
        dtype=torch.long,
        device=token_embeddings.device,
    )


@dataclass
class VisionEncoderOutput:
    """Vision tokens with a contiguous ``[prefix | dense grid]`` contract."""

    token_embeddings: torch.Tensor
    spatial_shape: Optional[torch.Tensor] = None  # (B, 2), height/width
    prefix_tokens: Optional[torch.Tensor] = None  # (B,), non-spatial token count


class VisionPathEmbeddings(nn.Module):
    """Patch-token vision embeddings for the local GLiFormer vision path."""

    def __init__(self, config: Any):
        super().__init__()
        image_size = pair_2d(get_config_value(config, "image_size", 224))
        patch_size = pair_2d(get_config_value(config, "vision_patch_size", get_config_value(config, "patch_size", 16)))
        num_channels = int(get_config_value(config, "vision_in_channels", get_config_value(config, "num_channels", 3)))
        hidden_size = int(get_config_value(config, "hidden_size"))

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

        position_type = get_config_value(
            config,
            "vision_position_embedding_type",
            None,
        )
        if position_type is None:
            # Plain namespaces and old checkpoints may still expose only the
            # former boolean switch.  GLiFormerConfig itself migrates this value.
            position_type = (
                "none"
                if get_config_value(config, "vision_position_embeddings", True) is False
                else "learned_grid2d"
            )
        position_kwargs = dict(
            get_config_value(config, "vision_position_embedding_kwargs", None) or {}
        )
        if PositionEmbedding.strategy_class(position_type).requires_grid_size:
            position_kwargs.setdefault("grid_size", self.patch_shape)
        self.position_embedding = PositionEmbedding.from_config(
            position_type,
            hidden_size,
            **position_kwargs,
        )

    def _position_embeddings(
        self,
        height: int,
        width: int,
        position_embedding: Optional[torch.Tensor],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if position_embedding is not None:
            # Preserve the historical forward override for callers that inject a
            # base-grid tensor directly.
            position_embedding = position_embedding.view(
                1,
                self.patch_shape[0],
                self.patch_shape[1],
                -1,
            )
            position_embedding = position_embedding.permute(0, 3, 1, 2)
            position_embedding = F.interpolate(
                position_embedding,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )
            return position_embedding.flatten(2).transpose(1, 2).to(
                device=device,
                dtype=dtype,
            )

        strategy = self.position_embedding
        coordinates = None
        if strategy.requires_coordinates:
            coordinates = normalized_grid_2d(
                height,
                width,
                device=device,
                dtype=torch.float32,
            )
        positions = strategy(
            coordinates,
            count=height * width,
            spatial_shape=(height, width),
            dtype=dtype,
            device=device,
        )
        if positions is not None and positions.dim() == 2:
            positions = positions.unsqueeze(0)
        return positions

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Local-patch checkpoints created before the position hierarchy stored
        # the table directly at ``position_embeddings``.
        old_key = f"{prefix}position_embeddings"
        new_key = f"{prefix}position_embedding.embedding"
        if old_key in state_dict and new_key not in state_dict:
            state_dict[new_key] = state_dict.pop(old_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        position_embedding: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> torch.Tensor:
        embeddings = self.proj(pixel_values)
        patch_height, patch_width = embeddings.shape[2], embeddings.shape[3]
        embeddings = embeddings.flatten(2).transpose(1, 2)
        position_embedding = self._position_embeddings(
            patch_height,
            patch_width,
            position_embedding,
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        if position_embedding is not None:
            embeddings = embeddings + position_embedding.to(device=embeddings.device, dtype=embeddings.dtype)
        return embeddings.contiguous()


class VisionEncoder(MediaBackboneEncoder):
    """Vision encoder with a GLiNER-style token embedding interface.

    The forward pass accepts ``pixel_values`` and returns token embeddings with
    shape ``(batch, visual_tokens, config.hidden_size)``. For Hugging Face vision
    models, ``last_hidden_state`` is used when available. For the local patch
    fallback, image patches are projected into visual tokens.
    """

    modality = "vision"
    model_name_config_key = "vision_model_name"
    encoder_type_config_key = "vision_encoder_type"
    encoder_config_key = "vision_encoder_config"
    default_local_encoder_type = "patch"
    additional_hidden_size_names = ("vision_hidden_size",)

    def _build_local_model(self) -> nn.Module | None:
        if self.encoder_type in {"patch", "path", "cnn"}:
            return VisionPathEmbeddings(self.config)
        return None

    def _extract_sequence_and_shape(
        self,
        output: Any,
    ) -> tuple[torch.Tensor, Optional[tuple[int, int]]]:
        spatial_shape = None
        for name in ("spatial_shape", "feature_spatial_shape"):
            value = (
                output.get(name)
                if isinstance(output, dict)
                else getattr(output, name, None)
            )
            if value is None:
                continue
            value = torch.as_tensor(value).reshape(-1, 2)
            if not torch.equal(value, value[:1].expand_as(value)):
                raise ValueError(
                    "A padded vision tensor must expose one uniform spatial shape"
                )
            spatial_shape = tuple(int(item) for item in value[0].tolist())
            break
        token_embeddings = self._extract_token_sequence(output)

        if token_embeddings.dim() == 4:
            spatial_shape = (int(token_embeddings.shape[-2]), int(token_embeddings.shape[-1]))
            token_embeddings = token_embeddings.flatten(2).transpose(1, 2)
        return token_embeddings, spatial_shape

    def _infer_spatial_metadata(
        self,
        pixel_values: torch.Tensor,
        token_embeddings: torch.Tensor,
        explicit_shape: Optional[tuple[int, int]],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch_size, token_count = token_embeddings.shape[:2]
        spatial_shape = explicit_shape

        configured_shape = get_config_value(
            self.config,
            "vision_feature_spatial_shape",
            None,
        )
        configured_stride = get_config_value(
            self.config,
            "vision_feature_stride",
            None,
        )
        configured_prefix = get_config_value(
            self.config,
            "vision_feature_prefix_tokens",
            None,
        )

        if spatial_shape is None and configured_shape is not None:
            spatial_shape = pair_2d(configured_shape)
        if spatial_shape is None and configured_stride is not None:
            stride_h, stride_w = pair_2d(configured_stride)
            spatial_shape = (
                int(pixel_values.shape[-2]) // stride_h,
                int(pixel_values.shape[-1]) // stride_w,
            )

        if spatial_shape is None and isinstance(self.model, VisionPathEmbeddings):
            patch_h, patch_w = self.model.patch_size
            spatial_shape = (
                int(pixel_values.shape[-2]) // patch_h,
                int(pixel_values.shape[-1]) // patch_w,
            )

        if spatial_shape is None:
            model_config = getattr(self.model, "config", None)
            patch_size = getattr(model_config, "patch_size", None)
            if patch_size is None:
                vision_config = getattr(model_config, "vision_config", None)
                patch_size = getattr(vision_config, "patch_size", None)
            if patch_size is not None:
                patch_h, patch_w = pair_2d(patch_size)
                candidate = (
                    int(pixel_values.shape[-2]) // patch_h,
                    int(pixel_values.shape[-1]) // patch_w,
                )
                if candidate[0] * candidate[1] <= token_count:
                    spatial_shape = candidate

        if spatial_shape is None:
            return None, None

        dense_count = spatial_shape[0] * spatial_shape[1]
        inferred_prefix = token_count - dense_count
        prefix_count = (
            inferred_prefix
            if configured_prefix is None
            else int(configured_prefix)
        )
        if prefix_count < 0 or prefix_count + dense_count != token_count:
            if configured_shape is not None or configured_stride is not None or configured_prefix is not None:
                raise ValueError(
                    "Configured vision spatial contract describes "
                    f"{prefix_count + dense_count} tokens, but the backbone returned "
                    f"{token_count}"
                )
            return None, None
        shapes = torch.tensor(
            spatial_shape,
            dtype=torch.long,
            device=token_embeddings.device,
        ).unsqueeze(0).expand(batch_size, -1).clone()
        prefixes = torch.full(
            (batch_size,),
            prefix_count,
            dtype=torch.long,
            device=token_embeddings.device,
        )
        return shapes, prefixes

    def forward_features(self, pixel_values: torch.Tensor, **kwargs: Any) -> VisionEncoderOutput:
        """Encode images while retaining exact dense-grid and prefix-token metadata."""
        output = self.model(pixel_values=pixel_values, **kwargs)
        token_embeddings, explicit_shape = self._extract_sequence_and_shape(output)
        token_embeddings = self._project_token_sequence(token_embeddings)
        spatial_shape, prefix_tokens = self._infer_spatial_metadata(
            pixel_values,
            token_embeddings,
            explicit_shape,
        )
        return VisionEncoderOutput(
            token_embeddings=token_embeddings,
            spatial_shape=spatial_shape,
            prefix_tokens=prefix_tokens,
        )

    def forward(self, pixel_values: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward_features(pixel_values, **kwargs).token_embeddings


class VisionBiEncoder(MediaBiEncoder):
    """Bi-encoder for vision GLiFormer models.

    Images are encoded by ``VisionEncoder``. Label names are encoded by a text
    transformer and mean-pooled, matching the GLiNER bi-encoder label contract.
    """

    media_encoder_cls = VisionEncoder
    media_encoder_attribute = "vision_encoder"

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels_input_ids: Optional[torch.Tensor] = None,
        labels_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        return self.forward_media(
            pixel_values,
            labels_input_ids=labels_input_ids,
            labels_attention_mask=labels_attention_mask,
            **kwargs,
        )
