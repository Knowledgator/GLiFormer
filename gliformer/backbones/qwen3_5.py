from typing import Optional

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.modeling_layers import GradientCheckpointingLayer

from .qwen3 import (
    _bidirectional_attention_mask,
    _config_bool,
    _get_position_ids,
    _is_flash_attention,
    _relative_position_bucket,
    _relative_position_type,
    _get_alibi_slopes,
    attention_forward_with_position_bias,
)

# The cuDNN attention kernel returns NaN gradients for head_dim 256 under an
# attention mask (seen on torch 2.14 / cuDNN 9.24, B200). Qwen3.5 has head_dim
# 256 and always gets a padding mask, so its attention avoids that kernel.
_SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

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
        Qwen3_5VisionAttention,
        Qwen3_5VisionModel,
        apply_rotary_pos_emb,
        apply_rotary_pos_emb_vision,
        repeat_kv,
    )

    from transformers.vision_utils import (
        get_vision_attention_seqlens,
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )

    try:
        from torch.nn.attention.varlen import varlen_attn
    except ImportError:
        varlen_attn = None
    try:
        from transformers.modeling_flash_attention_utils import is_flash_attention_requested
    except ImportError:
        def is_flash_attention_requested(config):
            return "flash" in str(getattr(config, "_attn_implementation", "") or "")

    try:
        from transformers.masking_utils import create_recurrent_attention_mask
    except ImportError:  # older releases keep this on the text model as `_update_linear_attn_mask`
        create_recurrent_attention_mask = None

    QWEN3_5_AVAILABLE = True

    def _route_cpu_to_torch_kernels():
        """Run the Gated DeltaNet helpers in plain torch for CPU tensors.

        transformers binds these helpers to the ``causal_conv1d`` / ``fla`` kernels
        whenever those packages import, without checking the device, so a CPU
        forward fails with "Expected x.is_cuda()". The torch versions stay
        reachable through ``__wrapped__``.
        """
        import inspect

        from transformers.models.qwen3_5 import modeling_qwen3_5 as stock

        names = (
            "causal_conv1d_fn",
            "causal_conv1d_update",
            "torch_chunk_gated_delta_rule",
            "torch_recurrent_gated_delta_rule",
        )
        for name in names:
            fast = getattr(stock, name, None)
            if fast is None or getattr(fast, "_gliformer_device_routed", False):
                continue
            fallback = inspect.unwrap(fast)
            if fallback is fast:
                continue

            def routed(*args, _fast=fast, _fallback=fallback, **kwargs):
                if args and isinstance(args[0], torch.Tensor) and not args[0].is_cuda:
                    return _fallback(*args, **kwargs)
                return _fast(*args, **kwargs)

            routed._gliformer_device_routed = True
            setattr(stock, name, routed)

    _route_cpu_to_torch_kernels()
except ImportError:
    Qwen3_5Config = None
    Qwen3_5TextConfig = None
    QWEN3_5_AVAILABLE = False


if QWEN3_5_AVAILABLE:

    class GLiFormerQwen3_5Attention(Qwen3_5Attention):
        """Qwen3.5 full attention with optional additive relative position encodings.

        With ``relative_pos_encoding`` unset this is the stock attention, on whichever
        backend ``_attn_implementation`` names (``sdpa``, ``flash_attention_2``, ``eager``).
        """

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
            with sdpa_kernel(_SDPA_BACKENDS):
                return self._attention_forward(
                    hidden_states, position_embeddings, attention_mask, past_key_values, position_ids, **kwargs
                )

        def _attention_forward(
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

            attn_output, attn_weights = attention_forward_with_position_bias(
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


    class GLiFormerQwen3_5DecoderLayer(GradientCheckpointingLayer):
        def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
            super().__init__()
            self.hidden_size = config.hidden_size
            self.layer_type = config.layer_types[layer_idx]
            if self.layer_type == "linear_attention":
                self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
            elif self.layer_type == "full_attention":
                self.self_attn = GLiFormerQwen3_5Attention(config, layer_idx)
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


    class GLiFormerQwen3_5TextModel(Qwen3_5TextModel):
        config_class = Qwen3_5TextConfig
        _no_split_modules = ["GLiFormerQwen3_5DecoderLayer"]

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
                [GLiFormerQwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
            )
            self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
            self.gradient_checkpointing = False
            self.post_init()

        def _linear_attention_mask(
            self,
            attention_mask: torch.Tensor | None,
            inputs_embeds: torch.Tensor,
            past_key_values: Cache | None,
        ) -> torch.Tensor | None:
            """The 2D padding mask for the Gated DeltaNet layers, however this transformers builds it."""
            if create_recurrent_attention_mask is None:
                return self._update_linear_attn_mask(attention_mask, past_key_values)
            if (
                past_key_values is None
                and attention_mask is not None
                and attention_mask.ndim == 2
                and inputs_embeds.shape[1] > 1
            ):
                # The stock helper drops an all-ones mask, which reads it back from the
                # GPU on every forward. The mask only multiplies the hidden states, so
                # keeping it gives the same values without waiting for the GPU.
                return attention_mask[:, -inputs_embeds.shape[1]:].contiguous()
            return create_recurrent_attention_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

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

            full_attention_mask = _bidirectional_attention_mask(
                attention_mask, inputs_embeds, past_key_values, _is_flash_attention(self.config)
            )
            linear_attn_mask = self._linear_attention_mask(attention_mask, inputs_embeds, past_key_values)
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
                    is_causal=False,
                    **kwargs,
                )

            hidden_states = self.norm(hidden_states)
            return Qwen3_5ModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=None,
            )


    def _vision_max_seqlen(cu_seqlens: torch.Tensor) -> int:
        """Longest image of a vision batch, read once and kept on the tensor.

        Every vision block gets the same ``cu_seqlens`` tensor, so the
        device read happens once per forward instead of once per block.
        """

        cached = getattr(cu_seqlens, "_gliformer_max_seqlen", None)
        if cached is None:
            cached = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
            cu_seqlens._gliformer_max_seqlen = cached
        return cached


    class GLiFormerQwen3_5VisionAttention(Qwen3_5VisionAttention):
        """Vision attention over all images in one variable-length kernel.

        Without flash-attn the stock module runs attention once per image in
        every block, reading the image lengths back from the device each
        time. ``torch.nn.attention.varlen.varlen_attn`` does the same
        block-diagonal attention in one call.
        """

        def forward(
            self,
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
            max_seqlen: int | None = None,
            **kwargs,
        ) -> torch.Tensor:
            if (
                varlen_attn is None
                or not hidden_states.is_cuda
                or is_flash_attention_requested(self.config)
                or position_embeddings is None
            ):
                return super().forward(hidden_states, cu_seqlens, position_embeddings, max_seqlen, **kwargs)
            seq_length = hidden_states.shape[0]
            query, key, value = (
                self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
            )
            cos, sin = position_embeddings
            query, key = apply_rotary_pos_emb_vision(query, key, cos, sin)
            if query.dtype not in (torch.float16, torch.bfloat16) or value.dtype != query.dtype:
                return super().forward(hidden_states, cu_seqlens, position_embeddings, max_seqlen, **kwargs)
            if max_seqlen is None:
                max_seqlen = _vision_max_seqlen(cu_seqlens)
            cu = cu_seqlens.to(torch.int32)
            output = varlen_attn(
                query.contiguous(), key.contiguous(), value.contiguous(),
                cu, cu, int(max_seqlen), int(max_seqlen), scale=self.scaling,
            )
            return self.proj(output.reshape(seq_length, -1).contiguous())


    class GLiFormerQwen3_5Model(Qwen3_5Model):
        """Multimodal Qwen3.5 with GLiFormer text-backbone attention options."""

        config_class = Qwen3_5Config
        _no_split_modules = ["GLiFormerQwen3_5DecoderLayer", "Qwen3_5VisionBlock"]

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
            for module in self.visual.modules():
                # Same parameters; only the attention call changes.
                if type(module) is Qwen3_5VisionAttention:
                    module.__class__ = GLiFormerQwen3_5VisionAttention
            self.language_model = GLiFormerQwen3_5TextModel._from_config(text_config)
            self.rope_deltas = None
            self.post_init()

        def get_rope_index(
            self,
            input_ids: torch.LongTensor,
            mm_token_type_ids: torch.IntTensor,
            image_grid_thw: torch.LongTensor | None = None,
            video_grid_thw: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            **kwargs,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """The stock M-RoPE positions, computed on the host.

            The stock version loops over rows with several device reads and
            dozens of tiny kernels per row. The same integer computation on
            CPU copies gives identical positions at a fraction of the cost.
            """

            device = input_ids.device
            if device.type == "cpu":
                return super().get_rope_index(
                    input_ids, mm_token_type_ids, image_grid_thw, video_grid_thw, attention_mask, **kwargs,
                )
            host = [
                None if tensor is None else tensor.to("cpu", non_blocking=True)
                for tensor in (input_ids, mm_token_type_ids, image_grid_thw, video_grid_thw, attention_mask)
            ]
            torch.cuda.current_stream(device).synchronize()
            position_ids, deltas = super().get_rope_index(*host, **kwargs)
            return position_ids.to(device, non_blocking=True), deltas.to(device, non_blocking=True)

        def forward(self, *args, native_image_grid_thw: torch.Tensor | None = None, **kwargs):
            # ``native_image_grid_thw``: a CPU copy of ``image_grid_thw``. With it the
            # vision tower sizes its inputs on the host instead of reading the grid
            # back from the GPU.
            self._host_image_grid_thw = native_image_grid_thw
            try:
                return super().forward(*args, **kwargs)
            finally:
                self._host_image_grid_thw = None
                # Kept by the stock model to continue a generation. An encoder never does,
                # and a later inputs_embeds call would be shifted by this batch's deltas.
                self.rope_deltas = None

        def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
            grid = getattr(self, "_host_image_grid_thw", None)
            if grid is None or image_grid_thw is None or not pixel_values.is_cuda:
                return super().get_image_features(pixel_values, image_grid_thw, **kwargs)
            # The stock method, with the grid-derived inputs of the vision tower
            # computed on the CPU copy (the same helpers it calls on the device grid).
            kwargs.pop("return_dict", None)
            visual = self.visual
            device = pixel_values.device
            grid = grid.to(torch.long)
            interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
                grid,
                num_grid_per_side=visual.num_grid_per_side,
                mode=visual.interpolation_mode,
                align_corners=visual.interpolation_align_corners,
                spatial_merge_size=visual.config.spatial_merge_size,
            )
            position_ids = get_vision_position_ids(grid, visual.spatial_merge_size)
            cu_seqlens, _ = get_vision_attention_seqlens(grid, visual.config)
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())

            def move(tensor):
                return tensor.pin_memory().to(device, non_blocking=True)

            vision_output = visual(
                pixel_values.type(visual.dtype),
                grid_thw=image_grid_thw,
                interp_indices=move(interp_indices),
                interp_weights=move(interp_weights),
                position_ids=move(position_ids),
                cu_seqlens=move(cu_seqlens),
                max_seqlen=max_seqlen,
                return_dict=True,
                **kwargs,
            )
            split_sizes = (grid.prod(-1) // visual.spatial_merge_size**2).tolist()
            vision_output.pooler_output = torch.split(vision_output.pooler_output, split_sizes)
            return vision_output

        def get_placeholder_mask(self, input_ids, inputs_embeds, image_features=None, video_features=None):
            if getattr(self, "_host_image_grid_thw", None) is not None and input_ids is not None:
                # The token count check reads a device sum back; the image blocks were
                # laid out on the host from this same grid, so the counts match.
                return super().get_placeholder_mask(input_ids, inputs_embeds)
            return super().get_placeholder_mask(input_ids, inputs_embeds, image_features, video_features)


    Qwen3_5BidirectionalModel = GLiFormerQwen3_5Model

else:
    GLiFormerQwen3_5VisionAttention = None
    GLiFormerQwen3_5Attention = None
    GLiFormerQwen3_5DecoderLayer = None
    GLiFormerQwen3_5TextModel = None
    GLiFormerQwen3_5Model = None
    Qwen3_5BidirectionalModel = None


__all__ = [
    "GLiFormerQwen3_5Attention",
    "GLiFormerQwen3_5DecoderLayer",
    "GLiFormerQwen3_5TextModel",
    "GLiFormerQwen3_5Model",
    "GLiFormerQwen3_5VisionAttention",
    "Qwen3_5BidirectionalModel",
    "QWEN3_5_AVAILABLE",
]
