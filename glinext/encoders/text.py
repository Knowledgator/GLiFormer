from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch import nn

from gliner.modeling.encoder import BiEncoder as GLiNERBiEncoder
from gliner.modeling.encoder import Encoder as GLiNEREncoder

from .base import Transformer, hidden_size


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


class TextTransformer(Transformer):
    """GLiNExT transformer variant exposed from GLiNExT."""

    pass


class TextEncoder(InputsEmbedsEncoderMixin, GLiNEREncoder):
    """GLiNER Encoder methods backed by GLiNExT's transformer wrapper."""

    def __init__(
        self,
        config: Any,
        from_pretrained: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.bert_layer = TextTransformer(config.model_name, config, from_pretrained, cache_dir=cache_dir)
        bert_hidden_size = hidden_size(self.bert_layer.model.config)
        if config.hidden_size != bert_hidden_size:
            self.projection = nn.Linear(bert_hidden_size, config.hidden_size)


class TextBiEncoder(InputsEmbedsEncoderMixin, GLiNERBiEncoder):
    """GLiNER BiEncoder methods backed by GLiNExT's transformer wrapper."""

    def __init__(
        self,
        config: Any,
        from_pretrained: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        TextEncoder.__init__(self, config, from_pretrained, cache_dir=cache_dir)
        if config.labels_encoder is not None:
            self.labels_encoder = TextTransformer(
                config.labels_encoder,
                config,
                from_pretrained,
                True,
                cache_dir=cache_dir,
            )
            le_hidden_size = hidden_size(self.labels_encoder.model.config)
            if config.hidden_size != le_hidden_size:
                self.labels_projection = nn.Linear(le_hidden_size, config.hidden_size)


Encoder = TextEncoder
BiEncoder = TextBiEncoder
