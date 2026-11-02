import math
from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch import nn

from gliner.modeling.encoder import BiEncoder as GLiNERBiEncoder
from gliner.modeling.encoder import Encoder as GLiNEREncoder

from .base import Transformer, hidden_size


def set_text_encoder_dropout(encoder: nn.Module, value: float) -> int:
    """Set text-backbone dropout without changing checkpoint topology."""
    dropout = float(value)
    if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
        raise ValueError("text encoder dropout must be finite and in [0, 1)")

    bert_layer = getattr(encoder, "bert_layer", None)
    backbone = getattr(bert_layer, "model", None)
    if backbone is None:
        return 0

    updated = 0
    for module in backbone.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout
            updated += 1

    backbone_config = getattr(backbone, "config", None)
    if backbone_config is not None:
        for name in (
            "hidden_dropout_prob",
            "attention_probs_dropout_prob",
            "pooler_dropout",
        ):
            if hasattr(backbone_config, name):
                setattr(backbone_config, name, dropout)
    return updated


class InputsEmbedsEncoderMixin:
    """Adds an inputs_embeds entrypoint to the GLiNER encoder variant."""

    @property
    def model_hidden_size(self) -> int:
        return hidden_size(self.bert_layer.model.config)

    def encode_inputs_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        kwargs.pop("packing_config", None)
        kwargs.pop("token_lengths", None)
        model_kwargs = dict(kwargs)
        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask
        token_embeddings = self.bert_layer(
            *args,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )
        if hasattr(self, "projection"):
            token_embeddings = self.projection(token_embeddings)
        return token_embeddings

    def encode_input_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the backbone on token ids, passing its own extra inputs (e.g. image patches) through."""
        token_embeddings = self.bert_layer(input_ids, attention_mask=attention_mask, **kwargs)
        if hasattr(self, "projection"):
            token_embeddings = self.projection(token_embeddings)
        return token_embeddings


class TextTransformer(Transformer):
    """GLiFormer transformer variant exposed from GLiFormer."""

    pass


# Inputs laid out per token or per image that sequence packing would have to
# move with the tokens; a batch carrying any of them is encoded padded.
_PACKING_BLOCKING_KWARGS = (
    "pair_attention_mask", "inputs_embeds", "position_ids", "bbox", "layout_input_mask",
    "page_token_ids", "page_input_mask", "pixel_values", "vision_attention_mask",
    "image_batch_idx", "image_page_ids",
)


class TextEncoder(InputsEmbedsEncoderMixin, GLiNEREncoder):
    """GLiNER Encoder methods backed by GLiFormer's transformer wrapper.

    With ``config.train_sequence_packing`` the rows of a training batch are
    packed into streams no longer than the padded batch (first-fit, block
    diagonal attention), so padding is not encoded. GLiNER only packs at
    inference; the packing itself is index bookkeeping, so gradients flow.
    """

    def _packs_in_training(self, input_ids, attention_mask, args, kwargs) -> bool:
        if not self.training or args:
            return False
        if not getattr(self.bert_layer.config, "train_sequence_packing", False):
            return False
        if not (isinstance(input_ids, torch.Tensor) and isinstance(attention_mask, torch.Tensor)):
            return False
        if input_ids.dim() != 2 or input_ids.shape[0] < 2:
            return False
        if any(kwargs.get(name) is not None for name in _PACKING_BLOCKING_KWARGS):
            return False
        token_type_ids = kwargs.get("token_type_ids")
        # Segment ids would need packing too; all-zero ones (DeBERTa-v3) do not.
        return token_type_ids is None or not bool(token_type_ids.any())

    def encode_text(self, input_ids, attention_mask, *args, **kwargs):
        if not self._packs_in_training(input_ids, attention_mask, args, kwargs):
            return super().encode_text(input_ids, attention_mask, *args, **kwargs)
        from gliner.infer_packing import InferencePackingConfig

        kwargs.pop("token_type_ids", None)
        kwargs.pop("packing_config", None)
        token_lengths = kwargs.pop("token_lengths", None)
        # Streams never exceed the padded batch, so no row is truncated.
        config = InferencePackingConfig(max_length=int(input_ids.shape[1]))
        token_embeddings = self._encode_with_packing(
            input_ids, attention_mask, config, None, token_lengths=token_lengths, **kwargs,
        )
        if hasattr(self, "projection"):
            token_embeddings = self.projection(token_embeddings)
        return token_embeddings

    def __init__(
        self,
        config: Any,
        from_pretrained: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        local_files_only: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        self.bert_layer = TextTransformer(
            config.model_name, config, from_pretrained, cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        embedding_config = getattr(config, "embedding_config", None)
        encoder_dropout = getattr(embedding_config, "encoder_dropout", None)
        if encoder_dropout is not None:
            set_text_encoder_dropout(self, encoder_dropout)
        bert_hidden_size = hidden_size(self.bert_layer.model.config)
        if config.hidden_size != bert_hidden_size:
            self.projection = nn.Linear(bert_hidden_size, config.hidden_size)


class TextBiEncoder(InputsEmbedsEncoderMixin, GLiNERBiEncoder):
    """GLiNER BiEncoder methods backed by GLiFormer's transformer wrapper."""

    def __init__(
        self,
        config: Any,
        from_pretrained: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        local_files_only: bool = False,
    ) -> None:
        TextEncoder.__init__(
            self, config, from_pretrained, cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        if config.labels_encoder is not None:
            self.labels_encoder = TextTransformer(
                config.labels_encoder,
                config,
                from_pretrained,
                True,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            le_hidden_size = hidden_size(self.labels_encoder.model.config)
            if config.hidden_size != le_hidden_size:
                self.labels_projection = nn.Linear(le_hidden_size, config.hidden_size)


Encoder = TextEncoder
BiEncoder = TextBiEncoder
