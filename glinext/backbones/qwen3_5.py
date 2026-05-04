from typing import Optional

import torch
from torch import nn
from transformers.modeling_layers import GradientCheckpointingLayer

from .qwen3 import (
    _bidirectional_attention_mask,
    _config_bool,
    _get_position_ids,
    _relative_position_bucket,
    _relative_position_type,
    _get_alibi_slopes,
    eager_attention_forward_with_position_bias,
)

try:
    from transformers.cache_utils import Cache
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5Attention,
        Qwen3_5GatedDeltaNet,
        Qwen3_5MLP,
        Qwen3_5Model,
        Qwen3_5ModelOutputWithPast,
        Qwen3_5PreTrainedModel,
        Qwen3_5RMSNorm,
        Qwen3_5TextModel,
        Qwen3_5TextRotaryEmbedding,
        Qwen3_5VisionModel,
        apply_rotary_pos_emb,
        repeat_kv,
    )

    QWEN3_5_AVAILABLE = True
except ImportError:
    Qwen3_5Config = None
    Qwen3_5TextConfig = None
    QWEN3_5_AVAILABLE = False


if QWEN3_5_AVAILABLE:

    class GLiNextQwen3_5Attention(Qwen3_5Attention):
        """Qwen3.5 full attention with optional additive relative position encodings."""

        def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
            super().__init__(config=config, layer_idx=layer_idx)
            self.bidirectional_attention = _config_bool(
                config,
                "bidirectional_attention",
                "qwen3_5_bidirectional_attention",
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
                None,
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

            if self.relative_pos_encoding == "alibi":
                if self.bidirectional_attention:
                    distance = relative_position.abs().neg()
                else:
                    distance = relative_position.clamp(max=0)
                return self.alibi_slopes.to(device=query_states.device, dtype=query_states.dtype) * distance[:, None, :, :]

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
            position_ids: torch.LongTensor | None = None,
            **kwargs,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            if self.relative_pos_encoding == "none":
                return super().forward(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    **kwargs,
                )

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states, gate = torch.chunk(
                self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
                2,
                dim=-1,
            )
            gate = gate.reshape(*input_shape, -1)
            query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

            attn_output, attn_weights = eager_attention_forward_with_position_bias(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                position_bias=self._relative_bias(query_states, key_states, position_ids),
                **kwargs,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_output * torch.sigmoid(gate)
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights


    class GLiNextQwen3_5DecoderLayer(GradientCheckpointingLayer):
        def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
            super().__init__()
            self.hidden_size = config.hidden_size
            self.layer_type = config.layer_types[layer_idx]
            if self.layer_type == "linear_attention":
                self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
            elif self.layer_type == "full_attention":
                self.self_attn = GLiNextQwen3_5Attention(config, layer_idx)
            else:
                raise ValueError(f"Unknown Qwen3.5 layer type: {self.layer_type!r}")
            self.mlp = Qwen3_5MLP(config, config.intermediate_size)
            self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: Cache | None = None,
            **kwargs,
        ) -> torch.FloatTensor:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            if self.layer_type == "linear_attention":
                hidden_states = self.linear_attn(
                    hidden_states=hidden_states,
                    cache_params=past_key_values,
                    attention_mask=attention_mask,
                )
            else:
                hidden_states, _ = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states


    class GLiNextQwen3_5TextModel(Qwen3_5TextModel):
        config_class = Qwen3_5TextConfig
        _no_split_modules = ["GLiNextQwen3_5DecoderLayer"]

        def __init__(self, config: Qwen3_5TextConfig):
            Qwen3_5PreTrainedModel.__init__(self, config)
            self.bidirectional_attention = _config_bool(
                config,
                "bidirectional_attention",
                "qwen3_5_bidirectional_attention",
                default=False,
            )
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
            self.layers = nn.ModuleList(
                [GLiNextQwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
            )
            self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
            self.gradient_checkpointing = False
            self.post_init()

        def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: Cache | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            use_cache: bool | None = None,
            **kwargs,
        ) -> Qwen3_5ModelOutputWithPast:
            if not self.bidirectional_attention:
                return Qwen3_5TextModel.forward(
                    self,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    **kwargs,
                )
            if use_cache:
                raise ValueError(
                    "Qwen3.5 bidirectional_attention=True is an encoder-style mode and does not support use_cache."
                )
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            if position_ids is None:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
            elif position_ids.ndim == 2:
                position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

            if position_ids.ndim == 3 and position_ids.shape[0] == 4:
                text_position_ids = position_ids[0]
                rope_position_ids = position_ids[1:]
            else:
                text_position_ids = None
                rope_position_ids = position_ids

            full_attention_mask = _bidirectional_attention_mask(attention_mask, inputs_embeds, past_key_values)
            linear_attn_mask = self._update_linear_attn_mask(attention_mask, past_key_values)
            hidden_states = inputs_embeds
            position_embeddings = self.rotary_emb(hidden_states, rope_position_ids)

            for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
                layer_mask = linear_attn_mask if self.config.layer_types[i] == "linear_attention" else full_attention_mask
                hidden_states = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=layer_mask,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    use_cache=False,
                    **kwargs,
                )

            hidden_states = self.norm(hidden_states)
            return Qwen3_5ModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=None,
            )


    class GLiNextQwen3_5Model(Qwen3_5Model):
        """Multimodal Qwen3.5 with GLiNExT text-backbone attention options."""

        config_class = Qwen3_5Config
        _no_split_modules = ["GLiNextQwen3_5DecoderLayer", "Qwen3_5VisionBlock"]

        def __init__(self, config: Qwen3_5Config):
            Qwen3_5PreTrainedModel.__init__(self, config)
            text_config = config.text_config
            for name in (
                "bidirectional_attention",
                "qwen3_5_bidirectional_attention",
                "relative_pos_encoding",
                "qwen3_relative_pos_encoding",
                "qwen3_5_relative_pos_encoding",
                "relative_attention_num_buckets",
                "position_buckets",
                "relative_attention_max_distance",
                "max_relative_positions",
            ):
                if hasattr(config, name) and not hasattr(text_config, name):
                    setattr(text_config, name, getattr(config, name))
            self.visual = Qwen3_5VisionModel._from_config(config.vision_config)
            self.language_model = GLiNextQwen3_5TextModel._from_config(text_config)
            self.rope_deltas = None
            self.post_init()


    Qwen3_5BidirectionalModel = GLiNextQwen3_5Model

else:
    GLiNextQwen3_5Attention = None
    GLiNextQwen3_5DecoderLayer = None
    GLiNextQwen3_5TextModel = None
    GLiNextQwen3_5Model = None
    Qwen3_5BidirectionalModel = None


__all__ = [
    "GLiNextQwen3_5Attention",
    "GLiNextQwen3_5DecoderLayer",
    "GLiNextQwen3_5TextModel",
    "GLiNextQwen3_5Model",
    "Qwen3_5BidirectionalModel",
    "QWEN3_5_AVAILABLE",
]
