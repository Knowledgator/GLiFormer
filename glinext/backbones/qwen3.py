import math
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3Model,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)


RELATIVE_POSITION_TYPES = {"none", "t5", "deberta", "alibi"}


def _config_bool(config: Qwen3Config, *names: str, default: bool = False) -> bool:
    for name in names:
        if hasattr(config, name):
            return bool(getattr(config, name))
    return default


def _relative_position_type(config: Qwen3Config) -> str:
    value = getattr(config, "relative_pos_encoding", None)

    if value not in RELATIVE_POSITION_TYPES:
        raise ValueError(
            f"Unknown Qwen3 relative position encoding {value!r}. "
            f"Expected one of {sorted(RELATIVE_POSITION_TYPES)}."
        )
    return value


def _relative_position_bucket(
    relative_position: torch.Tensor,
    num_buckets: int,
    max_distance: int,
    bidirectional: bool,
) -> torch.Tensor:
    """T5 relative position buckets for signed key-query distances."""
    relative_buckets = torch.zeros_like(relative_position, dtype=torch.long)
    if bidirectional:
        num_buckets //= 2
        relative_buckets = relative_buckets + (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.minimum(relative_position, torch.zeros_like(relative_position))

    max_exact = max(num_buckets // 2, 1)
    is_small = relative_position < max_exact
    max_distance = max(max_distance, max_exact + 1)
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact + 1e-6)
        / torch.log(torch.tensor(max_distance / max_exact, device=relative_position.device, dtype=torch.float))
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.minimum(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, num_buckets - 1),
    )
    return relative_buckets + torch.where(is_small, relative_position, relative_position_if_large)


def _get_alibi_slopes(num_heads: int) -> torch.Tensor:
    def get_power_of_2_slopes(n: int) -> list[float]:
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3.0)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(num_heads).is_integer():
        slopes = get_power_of_2_slopes(num_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        slopes = get_power_of_2_slopes(closest_power_of_2)
        extra = _get_alibi_slopes(2 * closest_power_of_2)[0::2][: num_heads - closest_power_of_2].tolist()
        slopes.extend(extra)
    return torch.tensor(slopes, dtype=torch.float32)


def _get_position_ids(
    batch_size: int,
    query_length: int,
    key_length: int,
    device: torch.device,
    position_ids: Optional[torch.LongTensor],
    cache_position: Optional[torch.LongTensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if position_ids is not None:
        query_positions = position_ids.to(device=device, dtype=torch.long)
        if query_positions.dim() == 1:
            query_positions = query_positions.unsqueeze(0)
        query_positions = query_positions[:, -query_length:]
    elif cache_position is not None:
        query_positions = cache_position.to(device=device, dtype=torch.long).view(1, -1)
        query_positions = query_positions[:, -query_length:]
    else:
        query_positions = torch.arange(query_length, device=device, dtype=torch.long).view(1, -1)

    if query_positions.size(0) == 1 and batch_size != 1:
        query_positions = query_positions.expand(batch_size, -1)

    if key_length == query_length:
        key_positions = query_positions
    else:
        key_positions = torch.arange(key_length, device=device, dtype=torch.long).view(1, -1)
        if batch_size != 1:
            key_positions = key_positions.expand(batch_size, -1)
    return query_positions, key_positions


def _bidirectional_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_embeds: torch.Tensor,
    past_key_values: Optional[Cache],
) -> Optional[torch.Tensor]:
    batch_size, query_length = input_embeds.shape[:2]
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    key_length = past_seen_tokens + query_length

    if attention_mask is None:
        return None
    if attention_mask.dim() == 4:
        return attention_mask
    if attention_mask.dim() != 2:
        raise ValueError(
            "Bidirectional Qwen3 attention_mask must be 2D or 4D, "
            f"got shape {tuple(attention_mask.shape)}."
        )

    if attention_mask.size(-1) < key_length:
        prefix = torch.ones(
            batch_size,
            key_length - attention_mask.size(-1),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat([prefix, attention_mask], dim=-1)
    attention_mask = attention_mask[:, -key_length:]
    min_value = torch.finfo(input_embeds.dtype).min
    additive_mask = torch.zeros(
        batch_size,
        1,
        query_length,
        key_length,
        device=input_embeds.device,
        dtype=input_embeds.dtype,
    )
    padding_mask = attention_mask[:, None, None, :].to(device=input_embeds.device).eq(0)
    return additive_mask.masked_fill(padding_mask, min_value)


def eager_attention_forward_with_position_bias(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    position_bias: Optional[torch.Tensor] = None,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if position_bias is not None:
        attn_weights = attn_weights + position_bias.to(dtype=attn_weights.dtype)
    if attention_mask is not None:
        attention_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class GLiNextQwen3Attention(Qwen3Attention):
    """Qwen3 attention with optional additive relative position encodings."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.bidirectional_attention = _config_bool(
            config,
            "bidirectional_attention",
            "qwen3_bidirectional_attention",
            default=False,
        )
        self.relative_pos_encoding = _relative_position_type(config)
        self.relative_position_buckets = int(
            getattr(config, "relative_attention_num_buckets", getattr(config, "position_buckets", 32))
        )
        self.max_relative_positions = int(
            getattr(config, "relative_attention_max_distance", getattr(config, "max_relative_positions", 128))
        )

        if self.relative_pos_encoding == "t5":
            self.relative_attention_bias = nn.Embedding(self.relative_position_buckets, config.num_attention_heads)
        elif self.relative_pos_encoding == "deberta":
            embedding_size = 2 * self.max_relative_positions - 1
            self.relative_embeddings = nn.Embedding(embedding_size, self.head_dim)
            self.pos_key_proj = nn.Linear(self.head_dim, config.num_attention_heads * self.head_dim, bias=False)
            self.pos_query_proj = nn.Linear(self.head_dim, config.num_attention_heads * self.head_dim, bias=False)
        elif self.relative_pos_encoding == "alibi":
            slopes = _get_alibi_slopes(config.num_attention_heads)
            self.register_buffer("alibi_slopes", slopes.view(1, config.num_attention_heads, 1, 1), persistent=False)

    def _relative_bias(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        cache_position: Optional[torch.LongTensor],
    ) -> Optional[torch.Tensor]:
        if self.relative_pos_encoding == "none":
            return None

        batch_size, _, query_length, _ = query_states.shape
        key_length = key_states.shape[-2]
        query_positions, key_positions = _get_position_ids(
            batch_size,
            query_length,
            key_length,
            query_states.device,
            position_ids,
            cache_position,
        )
        relative_position = key_positions[:, None, :] - query_positions[:, :, None]

        if self.relative_pos_encoding == "t5":
            buckets = _relative_position_bucket(
                relative_position,
                num_buckets=self.relative_position_buckets,
                max_distance=self.max_relative_positions,
                bidirectional=self.bidirectional_attention,
            )
            bias = self.relative_attention_bias(buckets)
            return bias.permute(0, 3, 1, 2).contiguous()

        elif self.relative_pos_encoding == "alibi":
            if self.bidirectional_attention:
                distance = relative_position.abs().neg()
            else:
                distance = relative_position.clamp(max=0)
            return self.alibi_slopes.to(device=query_states.device, dtype=query_states.dtype) * distance[:, None, :, :]
        else:
            clipped_position = (query_positions[:, :, None] - key_positions[:, None, :]).clamp(
                min=-(self.max_relative_positions - 1),
                max=self.max_relative_positions - 1,
            )
            rel_index = clipped_position + self.max_relative_positions - 1
            rel_embeddings = self.relative_embeddings.weight
            pos_key_layer = self.pos_key_proj(rel_embeddings).view(-1, self.config.num_attention_heads, self.head_dim)
            pos_query_layer = self.pos_query_proj(rel_embeddings).view(-1, self.config.num_attention_heads, self.head_dim)

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            c2p_att = torch.einsum("bhqd,rhd->bhqr", query_states, pos_key_layer.to(dtype=query_states.dtype))
            p2c_att = torch.einsum("bhkd,rhd->bhrk", key_states, pos_query_layer.to(dtype=key_states.dtype))
            gather_index = rel_index[:, None, :, :].expand(-1, self.config.num_attention_heads, -1, -1)
            c2p_att = torch.gather(c2p_att, dim=-1, index=gather_index)
            p2c_att = torch.gather(p2c_att, dim=2, index=gather_index)
            return (c2p_att + p2c_att) * self.scaling

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.relative_pos_encoding == "none":
            return super().forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_ids=position_ids,
                **kwargs,
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        position_bias = self._relative_bias(query_states, key_states, position_ids, cache_position)
        attn_output, attn_weights = eager_attention_forward_with_position_bias(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            position_bias=position_bias,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class GLiNextQwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GLiNextQwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class GLiNextQwen3Model(Qwen3PreTrainedModel):
    """AutoModel-compatible Qwen3 backbone with encoder-style attention options.

    Extra config fields:
      - ``bidirectional_attention``: if true, uses padding-only attention masks.
      - ``relative_pos_encoding``: one of ``none``, ``t5``, ``deberta``, ``alibi``.
      - ``relative_attention_num_buckets`` / ``position_buckets`` for T5 buckets.
      - ``relative_attention_max_distance`` / ``max_relative_positions``.
    """

    config_class = Qwen3Config
    _no_split_modules = ["GLiNextQwen3DecoderLayer"]

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.bidirectional_attention = _config_bool(
            config,
            "bidirectional_attention",
            "qwen3_bidirectional_attention",
            default=False,
        )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GLiNextQwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if not self.bidirectional_attention:
            return Qwen3Model.forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        if use_cache:
            raise ValueError("Qwen3 bidirectional_attention=True is an encoder-style mode and does not support use_cache.")
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(mask_mapping := attention_mask, dict):
            bidirectional_mask = _bidirectional_attention_mask(attention_mask, inputs_embeds, past_key_values)
            mask_mapping = {"full_attention": bidirectional_mask, "sliding_attention": bidirectional_mask}

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if kwargs.get("output_hidden_states", False) else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if all_hidden_states is not None:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=False,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states = all_hidden_states + (hidden_states,)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
        )


Qwen3BidirectionalModel = GLiNextQwen3Model

__all__ = [
    "GLiNextQwen3Attention",
    "GLiNextQwen3DecoderLayer",
    "GLiNextQwen3Model",
    "Qwen3BidirectionalModel",
]
