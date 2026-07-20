from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from .base import hidden_size
from .text import TextTransformer


def get_config_value(config: Any, name: str, default: Any = None) -> Any:
    """Read an optional encoder configuration value."""
    return getattr(config, name, default) if config is not None else default


def encoder_hidden_size(
    model_config: Any,
    default: int,
    *additional_names: str,
) -> int:
    """Infer a media backbone's output width from common config fields."""
    names = ("hidden_size", *additional_names, "projection_dim", "d_model", "encoder_embed_dim")
    for name in names:
        value = getattr(model_config, name, None)
        if value is not None:
            return int(value)
    hidden_sizes = getattr(model_config, "hidden_sizes", None)
    if hidden_sizes:
        return int(hidden_sizes[-1])
    return int(default)


def extract_token_sequence(output: Any, modality: str) -> torch.Tensor:
    """Extract a token tensor from common backbone output containers."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise ValueError(f"{modality} backbone did not return token embeddings")


def apply_input_mask(
    mask: torch.Tensor,
    input_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Disable complete modality rows after constructing their token mask."""

    if input_mask is None:
        return mask
    input_mask = input_mask.to(device=mask.device, dtype=mask.dtype)
    if input_mask.dim() == 1:
        input_mask = input_mask[:, None]
    return mask * input_mask


class MediaBackboneEncoder(nn.Module):
    """Shared lifecycle for vision and audio backbone wrappers.

    A media encoder can provide a local backbone in ``_build_local_model`` or
    fall through to the common Hugging Face ``AutoModel`` construction path.
    Subclasses retain responsibility for their input keyword, token geometry,
    masks, and any output metadata.
    """

    modality: ClassVar[str] = ""
    model_name_config_key: ClassVar[str] = ""
    encoder_type_config_key: ClassVar[str] = ""
    encoder_config_key: ClassVar[str] = ""
    default_local_encoder_type: ClassVar[str] = ""
    additional_hidden_size_names: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        config: Any,
        model_name: str | None = None,
        from_pretrained: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self._validate_class_contract()
        self.config = config
        self.hidden_size = int(get_config_value(config, "hidden_size"))
        self.model_name = model_name or get_config_value(
            config,
            self.model_name_config_key,
        )
        encoder_type = get_config_value(
            config,
            self.encoder_type_config_key,
            None,
        ) or ("auto" if self.model_name else self.default_local_encoder_type)
        self.encoder_type = self._normalize_encoder_type(encoder_type)

        self.model = self._build_model(
            from_pretrained=from_pretrained,
            cache_dir=cache_dir,
        )
        model_hidden_size = encoder_hidden_size(
            getattr(self.model, "config", None),
            self.hidden_size,
            *self.additional_hidden_size_names,
        )
        if model_hidden_size != self.hidden_size:
            # Keep this conditional attribute for checkpoint compatibility with
            # the former modality-specific wrappers.
            self.projection = nn.Linear(model_hidden_size, self.hidden_size)

    def _validate_class_contract(self) -> None:
        required = {
            "modality": self.modality,
            "model_name_config_key": self.model_name_config_key,
            "encoder_type_config_key": self.encoder_type_config_key,
            "encoder_config_key": self.encoder_config_key,
            "default_local_encoder_type": self.default_local_encoder_type,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise TypeError(
                f"{type(self).__name__} must define {', '.join(missing)}"
            )

    def _normalize_encoder_type(self, encoder_type: Any) -> Any:
        """Normalize modality-specific aliases before backbone selection."""

        return encoder_type

    def _build_local_model(self) -> nn.Module | None:
        """Return a local backbone, or ``None`` for the external-model path."""

        return None

    def _build_model(
        self,
        from_pretrained: bool,
        cache_dir: str | Path | None,
    ) -> nn.Module:
        local_model = self._build_local_model()
        if local_model is not None:
            return local_model

        model_config = get_config_value(self.config, self.encoder_config_key)
        if model_config is None:
            if self.model_name is None:
                raise ValueError(
                    f"{self.model_name_config_key} is required when "
                    f"{self.encoder_type_config_key} selects an external backbone"
                )
            model_config = AutoConfig.from_pretrained(
                self.model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )

        if from_pretrained:
            if self.model_name is None:
                raise ValueError(
                    f"{self.model_name_config_key} is required to load pretrained "
                    f"{self.modality} weights"
                )
            return AutoModel.from_pretrained(
                self.model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
        return AutoModel.from_config(model_config, trust_remote_code=True)

    def _extract_token_sequence(self, output: Any) -> torch.Tensor:
        return extract_token_sequence(output, self.modality.capitalize())

    def _project_token_sequence(
        self,
        token_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(self, "projection"):
            token_embeddings = self.projection(token_embeddings)
        return token_embeddings


class MediaBiEncoder(nn.Module):
    """Shared label-encoder contract for single-modality media encoders.

    Subclasses provide a feature encoder class and the checkpoint-compatible
    attribute under which it is registered. Label names are encoded with the
    same text transformer and masked mean-pooling path for every modality.
    """

    media_encoder_cls: ClassVar[type[nn.Module] | None] = None
    media_encoder_attribute: ClassVar[str] = ""
    resizes_labels_encoder_only = True

    def __init__(
        self,
        config: Any,
        from_pretrained: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        if self.media_encoder_cls is None or not self.media_encoder_attribute:
            raise TypeError(
                f"{type(self).__name__} must define media_encoder_cls and "
                "media_encoder_attribute"
            )
        setattr(
            self,
            self.media_encoder_attribute,
            self.media_encoder_cls(
                config,
                from_pretrained=from_pretrained,
                cache_dir=cache_dir,
            ),
        )

        label_model_name = get_config_value(config, "labels_encoder") or get_config_value(
            config,
            "model_name",
        )
        if label_model_name is None:
            raise ValueError(
                f"{type(self).__name__} requires config.labels_encoder or "
                "config.model_name for label encoding"
            )
        self.labels_encoder = TextTransformer(
            label_model_name,
            config,
            from_pretrained=from_pretrained,
            labels_encoder=True,
            cache_dir=cache_dir,
        )
        output_hidden_size = int(get_config_value(config, "hidden_size"))
        label_hidden_size = hidden_size(self.labels_encoder.model.config)
        if output_hidden_size != label_hidden_size:
            self.labels_projection = nn.Linear(label_hidden_size, output_hidden_size)

    @property
    def media_encoder(self) -> nn.Module:
        return getattr(self, self.media_encoder_attribute)

    @staticmethod
    def mean_pooling(
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        input_mask_expanded = input_mask_expanded.to(dtype=token_embeddings.dtype)
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1),
            min=1,
        )

    def resize_token_embeddings(
        self,
        new_num_tokens: int,
        pad_to_multiple_of: int | None = None,
    ) -> nn.Embedding:
        # Media-only models resize the tokenizer externally; their only token
        # embedding table belongs to the labels encoder.
        return self.get_input_embeddings()

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
        label_tokens = self.labels_encoder(
            input_ids,
            attention_mask=attention_mask,
            **label_kwargs,
        )
        if hasattr(self, "labels_projection"):
            label_tokens = self.labels_projection(label_tokens)
        return self.mean_pooling(label_tokens, attention_mask)

    def encode_media(
        self,
        media_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        media_kwargs = dict(kwargs)
        if attention_mask is not None:
            media_kwargs["attention_mask"] = attention_mask
        return self.media_encoder(media_values, **media_kwargs)

    def forward_media(
        self,
        media_values: torch.Tensor,
        media_attention_mask: torch.Tensor | None = None,
        labels_input_ids: torch.Tensor | None = None,
        labels_attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        media_tokens = self.encode_media(
            media_values,
            attention_mask=media_attention_mask,
            **kwargs,
        )
        if labels_input_ids is None or labels_attention_mask is None:
            return media_tokens
        labels_embeddings = self.encode_labels(labels_input_ids, labels_attention_mask)
        return media_tokens, labels_embeddings
