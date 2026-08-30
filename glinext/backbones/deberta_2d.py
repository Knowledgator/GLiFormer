from collections.abc import Sequence
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Attention,
    DebertaV2Config as HFDebertaV2Config,
    DebertaV2Encoder,
    DebertaV2Layer,
    DebertaV2SelfOutput,
    DisentangledSelfAttention,
    LayerNorm,
    scaled_size_sqrt,
)

from ..layers.positional_layer import SpatialEmbeddings


def _prepare_layout_input_mask(
    layout_input_mask: Optional[torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Normalize the per-example flag controlling 2D layout features."""
    if layout_input_mask is None:
        return None
    if not isinstance(layout_input_mask, torch.Tensor):
        layout_input_mask = torch.as_tensor(layout_input_mask)
    if layout_input_mask.shape != (batch_size,):
        raise ValueError(
            "layout_input_mask must have shape (batch,), "
            f"got {tuple(layout_input_mask.shape)} for batch size {batch_size}"
        )
    return layout_input_mask.to(device=device, dtype=torch.bool)


def _relative_position_bucket(
    relative_position: torch.Tensor,
    num_buckets: int = 32,
    max_distance: int = 128,
    bidirectional: bool = True,
) -> torch.Tensor:
    """T5-style relative position buckets for signed pairwise distances."""
    relative_buckets = torch.zeros_like(relative_position, dtype=torch.long)
    if bidirectional:
        num_buckets //= 2
        relative_buckets = relative_buckets + (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

    max_distance = max(max_distance, 2)
    max_exact = max(num_buckets // 2, 1)
    is_small = relative_position < max_exact
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact + 1e-6)
        / torch.log(torch.tensor(max(max_distance / max_exact, 1.0001), device=relative_position.device, dtype=torch.float))
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, num_buckets - 1),
    )
    relative_buckets = relative_buckets + torch.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets


def build_layout_relative_position(
    bbox: torch.Tensor,
    num_buckets: int,
    max_distance: int,
) -> torch.Tensor:
    """Build 2D T5-style relative buckets from box x/y coordinates.

    Returns a tensor shaped ``(batch, query, key, 2)`` containing bucket ids for
    x and y coordinate differences.
    """
    bbox = bbox.long()
    x, y = bbox[..., 0], bbox[..., 1]
    features = (x, y)
    rel_features = []
    for value in features:
        rel = value[:, :, None] - value[:, None, :]
        rel_features.append(_relative_position_bucket(rel, num_buckets, max_distance))
    return torch.stack(rel_features, dim=-1)


class LayoutDebertaConfig(HFDebertaV2Config):
    model_type = "layout-deberta"

    def __init__(
        self,
        max_2d_position_embeddings: int = 1024,
        max_page_embeddings: int = 1024,
        coordinate_size: int = 128,
        shape_size: int = 64,
        layout_embedding_type: str = "absolute",
        layout_relative_attention: Optional[bool] = None,
        layout_position_buckets: int = 32,
        layout_max_relative_positions: int = 1024,
        layout_bias_propagation: str = "all_layers",
        **kwargs,
    ):
        layout_embedding_type = kwargs.pop("spatial_embedding_type", layout_embedding_type)
        super().__init__(**kwargs)
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.max_page_embeddings = max_page_embeddings
        self.coordinate_size = coordinate_size
        self.shape_size = shape_size
        self.layout_embedding_type = layout_embedding_type
        self.spatial_embedding_type = layout_embedding_type
        if layout_relative_attention is None:
            layout_relative_attention = layout_embedding_type in {"relative", "both", "absolute_relative"}
        self.layout_relative_attention = layout_relative_attention
        self.layout_position_buckets = layout_position_buckets
        self.layout_max_relative_positions = layout_max_relative_positions
        self.layout_bias_propagation = layout_bias_propagation


class LayoutDebertaEmbeddings(nn.Module):
    """DeBERTa embeddings with optional absolute 2D layout embeddings."""

    def __init__(self, config):
        super().__init__()
        pad_token_id = getattr(config, "pad_token_id", 0)
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)

        self.position_biased_input = getattr(config, "position_biased_input", True)
        if self.position_biased_input:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)
        else:
            self.position_embeddings = None

        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)
        else:
            self.token_type_embeddings = None

        max_page_embeddings = int(getattr(config, "max_page_embeddings", 0) or 0)
        if max_page_embeddings > 0:
            self.page_embeddings = nn.Embedding(max_page_embeddings, self.embedding_size)
        else:
            self.page_embeddings = None

        if self.embedding_size != config.hidden_size:
            self.embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias=False)
        else:
            self.embed_proj = None

        self.layout_embedding_type = getattr(config, "layout_embedding_type", getattr(config, "spatial_embedding_type", "none"))
        if self.layout_embedding_type in {"absolute", "both", "absolute_relative"}:
            self.spatial_embeddings = SpatialEmbeddings(config)
        else:
            self.spatial_embeddings = None

        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        bbox=None,
        layout_input_mask=None,
        page_token_ids=None,
        mask=None,
        inputs_embeds=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings(position_ids.long())
        if self.token_type_embeddings is not None:
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)
        if self.page_embeddings is not None:
            if page_token_ids is None:
                page_token_ids = torch.zeros(input_shape, dtype=torch.long, device=embeddings.device)
            if page_token_ids.shape != input_shape:
                raise ValueError(
                    f"page_token_ids must have shape {tuple(input_shape)}, got {tuple(page_token_ids.shape)}"
                )
            page_token_ids = page_token_ids.to(device=embeddings.device, dtype=torch.long)
            page_token_ids = torch.clamp(page_token_ids, 0, self.page_embeddings.num_embeddings - 1)
            embeddings = embeddings + self.page_embeddings(page_token_ids)
        if self.spatial_embeddings is not None and bbox is not None:
            spatial_embeddings = self.spatial_embeddings(bbox)
            prepared_layout_mask = _prepare_layout_input_mask(
                layout_input_mask,
                input_shape[0],
                spatial_embeddings.device,
            )
            if prepared_layout_mask is not None:
                spatial_embeddings = spatial_embeddings * prepared_layout_mask[:, None, None].to(
                    dtype=spatial_embeddings.dtype
                )
            embeddings = embeddings + spatial_embeddings

        if self.embed_proj is not None:
            embeddings = self.embed_proj(embeddings)

        embeddings = self.LayerNorm(embeddings)
        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            embeddings = embeddings * mask.to(dtype=embeddings.dtype)
        return self.dropout(embeddings)


class LayoutDisentangledSelfAttention(DisentangledSelfAttention):
    """DeBERTa disentangled attention with an additive 2D relative layout bias."""

    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        layout_attention_bias=None,
    ):
        if query_states is None:
            query_states = hidden_states
        query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)

        rel_att = None
        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        scale = scaled_size_sqrt(query_layer, scale_factor).to(device=query_layer.device)
        attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_attention_bias(
                query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
            )
        if rel_att is not None:
            attention_scores = attention_scores + rel_att
        attention_scores = attention_scores.view(
            -1,
            self.num_attention_heads,
            attention_scores.size(-2),
            attention_scores.size(-1),
        )

        if layout_attention_bias is not None:
            attention_scores = attention_scores + layout_attention_bias.to(dtype=attention_scores.dtype)

        attention_mask = attention_mask.bool()
        attention_scores = attention_scores.masked_fill(~attention_mask, torch.finfo(query_layer.dtype).min)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.bmm(
            attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)),
            value_layer,
        )
        context_layer = (
            context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        context_layer = context_layer.view(context_layer.size()[:-2] + (-1,))
        if output_attentions:
            return context_layer, attention_probs
        return context_layer, None


class LayoutDebertaAttention(DebertaV2Attention):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.self = LayoutDisentangledSelfAttention(config)
        self.output = DebertaV2SelfOutput(config)
        self.config = config

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions: bool = False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        layout_attention_bias=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self_output, att_matrix = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
            layout_attention_bias=layout_attention_bias,
        )
        if query_states is None:
            query_states = hidden_states
        attention_output = self.output(self_output, query_states)
        if output_attentions:
            return attention_output, att_matrix
        return attention_output, None


class LayoutDebertaLayer(DebertaV2Layer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = LayoutDebertaAttention(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        output_attentions: bool = False,
        layout_attention_bias=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attention_output, att_matrix = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
            layout_attention_bias=layout_attention_bias,
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if output_attentions:
            return layer_output, att_matrix
        return layer_output, None


class LayoutDebertaEncoder(DebertaV2Encoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([LayoutDebertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.layout_relative_attention = getattr(config, "layout_relative_attention", False)
        self.layout_bias_propagation = getattr(config, "layout_bias_propagation", "all_layers")
        if self.layout_relative_attention:
            self.layout_position_buckets = getattr(config, "layout_position_buckets", 32)
            self.layout_max_relative_positions = getattr(config, "layout_max_relative_positions", 1024)
            self.layout_relative_x_bias = nn.Embedding(self.layout_position_buckets, config.num_attention_heads)
            self.layout_relative_y_bias = nn.Embedding(self.layout_position_buckets, config.num_attention_heads)

    def get_layout_attention_bias(
        self,
        bbox: Optional[torch.Tensor],
        layout_input_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if not self.layout_relative_attention or bbox is None:
            return None
        buckets = build_layout_relative_position(
            bbox,
            num_buckets=self.layout_position_buckets,
            max_distance=self.layout_max_relative_positions,
        )
        x_bias = self.layout_relative_x_bias(buckets[..., 0])
        y_bias = self.layout_relative_y_bias(buckets[..., 1])
        bias = x_bias + y_bias
        bias = bias.permute(0, 3, 1, 2).contiguous()
        prepared_layout_mask = _prepare_layout_input_mask(
            layout_input_mask,
            bbox.shape[0],
            bias.device,
        )
        if prepared_layout_mask is not None:
            bias = bias * prepared_layout_mask[:, None, None, None].to(dtype=bias.dtype)
        return bias

    def _layout_bias_for_layer(self, layout_attention_bias, layer_idx: int):
        if layout_attention_bias is None:
            return None
        if self.layout_bias_propagation in {"all_layers", "propagate", "all"}:
            return layout_attention_bias
        if self.layout_bias_propagation in {"first_layer", "first", "none"}:
            return layout_attention_bias if layer_idx == 0 else None
        raise ValueError(f"Unknown layout_bias_propagation: {self.layout_bias_propagation!r}")

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        return_dict=True,
        bbox: Optional[torch.Tensor] = None,
        layout_input_mask: Optional[torch.Tensor] = None,
        layout_attention_bias: Optional[torch.Tensor] = None,
    ):
        if attention_mask.dim() <= 2:
            input_mask = attention_mask
        else:
            input_mask = attention_mask.sum(-2) > 0
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)
        if layout_attention_bias is None:
            layout_attention_bias = self.get_layout_attention_bias(bbox, layout_input_mask)

        all_hidden_states: Optional[Tuple[torch.Tensor, ...]] = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        next_kv = hidden_states
        rel_embeddings = self.get_rel_embedding()
        output_states = hidden_states
        for i, layer_module in enumerate(self.layer):
            output_states, attn_weights = layer_module(
                next_kv,
                attention_mask,
                query_states=query_states,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
                output_attentions=output_attentions,
                layout_attention_bias=self._layout_bias_for_layer(layout_attention_bias, i),
            )
            if output_attentions:
                all_attentions = all_attentions + (attn_weights,)

            if i == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states,)

            if query_states is not None:
                query_states = output_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = output_states

        if not return_dict:
            return tuple(v for v in [output_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=output_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class LayoutDebertaPreTrainedModel(PreTrainedModel):
    config_class = LayoutDebertaConfig
    base_model_prefix = "deberta"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LayoutDebertaLayer"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()


class LayoutDebertaModel(LayoutDebertaPreTrainedModel):
    supports_layout_input_mask = True

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = LayoutDebertaEmbeddings(config)
        self.encoder = LayoutDebertaEncoder(config)
        self.z_steps = 0
        self.config = config
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError("The prune function is not implemented in LayoutDeberta.")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        layout_input_mask: Optional[torch.Tensor] = None,
        page_token_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if bbox is not None and bbox.shape[:2] != input_shape:
            raise ValueError(f"bbox must have shape (batch, seq_len, 4), got {tuple(bbox.shape)} for input {tuple(input_shape)}")
        if layout_input_mask is not None and bbox is None:
            raise ValueError("layout_input_mask requires bbox")
        layout_input_mask = _prepare_layout_input_mask(layout_input_mask, input_shape[0], device)
        if page_token_ids is not None and page_token_ids.shape != input_shape:
            raise ValueError(
                f"page_token_ids must have shape {tuple(input_shape)}, got {tuple(page_token_ids.shape)}"
            )

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            bbox=bbox,
            layout_input_mask=layout_input_mask,
            page_token_ids=page_token_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        layout_attention_bias = self.encoder.get_layout_attention_bias(bbox, layout_input_mask)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
            bbox=bbox,
            layout_input_mask=layout_input_mask,
            layout_attention_bias=layout_attention_bias,
        )
        encoded_layers = encoder_outputs[1] if not return_dict else encoder_outputs.hidden_states

        if self.z_steps > 1:
            hidden_states = encoded_layers[-2]
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            query_states = encoded_layers[-1]
            rel_embeddings = self.encoder.get_rel_embedding()
            attention_mask_ext = self.encoder.get_attention_mask(attention_mask)
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            for layer in layers[1:]:
                query_states = layer(
                    hidden_states,
                    attention_mask_ext,
                    output_attentions=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                    layout_attention_bias=layout_attention_bias,
                )[0]
                encoded_layers = encoded_layers + (query_states,)

        sequence_output = encoded_layers[-1]
        if not return_dict:
            return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2):]
        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoded_layers if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
        )


# Backwards-compatible aliases for older imports during the transition.
Deberta2DConfig = LayoutDebertaConfig
Deberta2DModel = LayoutDebertaModel
