import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, DebertaV2Model, T5EncoderModel
from transformers.modeling_outputs import BaseModelOutput

from gliner.modeling.layers import LayersFuser
from gliner.utils import MissedPackageException, is_module_available

from glinext.backbones import BackboneSpec, get_backbone


IS_LLM2VEC = is_module_available("llm2vec")
IS_PEFT = is_module_available("peft")
IS_TURBOT5 = is_module_available("turbot5")
IS_FLASHDEBERTA = is_module_available("flashdeberta")

if IS_LLM2VEC:
    from llm2vec.models import GemmaBiModel, LlamaBiModel, MistralBiModel, Qwen2BiModel

    DECODER_MODEL_MAPPING = {
        "MistralConfig": MistralBiModel,
        "LlamaConfig": LlamaBiModel,
        "GemmaConfig": GemmaBiModel,
        "Qwen2Config": Qwen2BiModel,
    }
else:
    DECODER_MODEL_MAPPING = {}

if IS_TURBOT5:
    from turbot5.model.modeling import T5EncoderModel as FlashT5EncoderModel

if IS_FLASHDEBERTA:
    from flashdeberta import FlashDebertaV2Model

if IS_PEFT:
    from peft import LoraConfig, get_peft_model


def hidden_size(model_config: Any) -> int:
    for name in ("hidden_size", "d_model", "encoder_embed_dim"):
        value = getattr(model_config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not infer hidden size from encoder config")


def _coerce_backbone_config(encoder_config: Any, backbone: Optional[BackboneSpec]) -> Any:
    if backbone is None or backbone.config_class is None:
        return encoder_config
    if isinstance(encoder_config, backbone.config_class):
        return encoder_config
    if isinstance(encoder_config, dict):
        config_dict = dict(encoder_config)
    elif hasattr(encoder_config, "to_dict"):
        config_dict = encoder_config.to_dict()
    else:
        raise TypeError(
            f"Cannot coerce encoder_config of type {type(encoder_config).__name__} "
            f"for backbone_type={backbone.name!r}"
        )
    config_dict.pop("model_type", None)
    return backbone.config_class(**config_dict)


class Transformer(nn.Module):
    """GLiNExT transformer wrapper with explicit custom-backbone support."""

    def __init__(
        self,
        model_name: str,
        config: Any,
        from_pretrained: bool = False,
        labels_encoder: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()
        if labels_encoder:
            encoder_config = config.labels_encoder_config
        else:
            encoder_config = config.encoder_config

        backbone = None if labels_encoder else get_backbone(getattr(config, "backbone_type", "auto"))

        if encoder_config is None:
            encoder_config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
            if config.vocab_size != -1 and not labels_encoder:
                encoder_config.vocab_size = config.vocab_size
        encoder_config = _coerce_backbone_config(encoder_config, backbone)

        if config._attn_implementation is not None and not labels_encoder:
            encoder_config._attn_implementation = config._attn_implementation

        config_name = encoder_config.__class__.__name__
        kwargs: Dict[str, Any] = {}

        if backbone is not None:
            custom = True
            ModelClass = backbone.model_class
        elif config_name in DECODER_MODEL_MAPPING:
            if not IS_LLM2VEC:
                raise MissedPackageException(
                    f"The llm2vec package must be installed to use this decoder model: {config_name}"
                )
            ModelClass = DECODER_MODEL_MAPPING[config_name]
            custom = True
        elif config_name in {"T5Config", "MT5Config"}:
            custom = True
            turbot5_type = os.environ.get("TURBOT5_ATTN_TYPE", "basic")
            if turbot5_type and IS_TURBOT5:
                ModelClass = FlashT5EncoderModel
                kwargs = {"attention_type": turbot5_type}
                encoder_config.attention_type = turbot5_type
            else:
                ModelClass = T5EncoderModel
        elif config_name in {"DebertaV2Config"}:
            custom = True
            if os.environ.get("USE_FLASHDEBERTA", "") and IS_FLASHDEBERTA:
                ModelClass = FlashDebertaV2Model
            else:
                ModelClass = DebertaV2Model
        else:
            custom = False
            ModelClass = AutoModel

        if from_pretrained:
            pretrained_kwargs = dict(kwargs)
            if cache_dir is not None:
                pretrained_kwargs["cache_dir"] = cache_dir
            if backbone is not None:
                pretrained_kwargs["config"] = encoder_config
            self.model = ModelClass.from_pretrained(model_name, **pretrained_kwargs, trust_remote_code=True)
        elif not custom:
            self.model = ModelClass.from_config(encoder_config, trust_remote_code=True)
        else:
            self.model = ModelClass(encoder_config, **kwargs)

        adapter_config_file = Path(model_name) / "adapter_config.json"
        if adapter_config_file.exists():
            if not IS_PEFT:
                warnings.warn(
                    "Adapter configs were detected, if you want to apply them you need to install peft package.",
                    stacklevel=2,
                )
            else:
                adapter_config = LoraConfig.from_pretrained(model_name)
                self.model = get_peft_model(self.model, adapter_config)

        if config.fuse_layers:
            self.layers_fuser = LayersFuser(encoder_config.num_hidden_layers, hidden_size(encoder_config))

        if labels_encoder:
            config.labels_encoder_config = encoder_config
        else:
            config.encoder_config = encoder_config

        self.config = config

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if not getattr(self.model, "supports_layout_input_mask", False):
            kwargs.pop("layout_input_mask", None)
        pair_attention_mask = kwargs.pop("pair_attention_mask", None)
        base_attention_mask = kwargs.pop("attention_mask", None)
        args = list(args)
        input_ids = kwargs.pop("input_ids", None)
        if input_ids is None and args:
            input_ids = args[0]
            args = args[1:]
        args = tuple(args)

        kwargs.setdefault("output_attentions", False)
        kwargs.setdefault("return_dict", True)
        if self.config.fuse_layers:
            kwargs["output_hidden_states"] = True
        else:
            kwargs.setdefault("output_hidden_states", False)

        if pair_attention_mask is not None:
            mask_info = self._prepare_pair_attention_masks(
                pair_attention_mask,
                base_attention_mask,
                input_ids,
                kwargs.get("inputs_embeds"),
            )
            model_kwargs = dict(kwargs)
            model_name = self.model.__class__.__name__

            if model_name in {"DebertaV2Model", "DebertaModel", "FlashDebertaV2Model", "LayoutDebertaModel"}:
                output = self._forward_deberta(
                    input_ids=input_ids,
                    model_kwargs=model_kwargs,
                    mask_info=mask_info,
                )
            elif model_name == "ModernBertModel":
                output = self._forward_modernbert(
                    input_ids=input_ids,
                    model_kwargs=model_kwargs,
                    mask_info=mask_info,
                )
            elif model_name in {"T5EncoderModel", "MT5EncoderModel", "T5Model"}:
                output = self._forward_t5(
                    input_ids=input_ids,
                    model_kwargs=model_kwargs,
                    mask_info=mask_info,
                )
            else:
                model_kwargs.pop("packing_config", None)
                model_kwargs["attention_mask"] = mask_info["extended_mask"]
                output = self.model(*args, **model_kwargs)
        else:
            if base_attention_mask is not None:
                kwargs["attention_mask"] = base_attention_mask
            output = self.model(input_ids, *args, **kwargs)

        if self.config.fuse_layers:
            return self.layers_fuser(output.hidden_states)
        return output[0]

    def _get_model_dtype(self) -> torch.dtype:
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _prepare_pair_attention_masks(
        self,
        pair_attention_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        input_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        device = pair_attention_mask.device
        if input_ids is not None:
            device = input_ids.device
        elif inputs_embeds is not None:
            device = inputs_embeds.device

        pair_mask_bool = pair_attention_mask.to(device=device, dtype=torch.bool)
        token_mask_bool = pair_mask_bool.any(dim=-1)
        if attention_mask is not None:
            token_mask_bool = token_mask_bool & attention_mask.to(device=device, dtype=torch.bool)

        seq_len = pair_mask_bool.size(-1)
        if seq_len:
            identity = torch.eye(seq_len, device=device, dtype=torch.bool).unsqueeze(0)
            pair_mask_bool = pair_mask_bool | (identity & token_mask_bool.unsqueeze(-1))

        active = token_mask_bool.unsqueeze(-1) & token_mask_bool.unsqueeze(-2)
        pair_mask_bool = pair_mask_bool & active

        if attention_mask is not None:
            token_mask = token_mask_bool.to(attention_mask.dtype)
        else:
            token_mask = token_mask_bool.to(dtype=torch.float32)

        mask_dtype = self._get_model_dtype()
        neg_inf = torch.finfo(mask_dtype).min
        extended_mask = (
            torch.zeros(pair_mask_bool.shape, dtype=mask_dtype, device=device)
            .masked_fill(~pair_mask_bool, neg_inf)
            .unsqueeze(1)
        )

        inactive = ~token_mask_bool
        if inactive.any():
            extended_mask = extended_mask.masked_fill(
                inactive.unsqueeze(1).unsqueeze(-1),
                torch.tensor(0.0, dtype=mask_dtype, device=device),
            )

        return {
            "token_mask": token_mask,
            "token_mask_bool": token_mask_bool,
            "extended_mask": extended_mask,
            "block_mask": pair_mask_bool,
        }

    def _forward_deberta(
        self,
        input_ids: Optional[torch.Tensor],
        model_kwargs: Dict[str, Any],
        mask_info: Dict[str, torch.Tensor],
    ) -> BaseModelOutput:
        inputs_embeds = model_kwargs.pop("inputs_embeds", None)
        token_type_ids = model_kwargs.pop("token_type_ids", None)
        position_ids = model_kwargs.pop("position_ids", None)
        bbox = model_kwargs.pop("bbox", None)
        layout_input_mask = model_kwargs.pop("layout_input_mask", None)
        page_token_ids = model_kwargs.pop("page_token_ids", None)
        output_attentions = model_kwargs.pop("output_attentions")
        produce_hidden = model_kwargs.pop("output_hidden_states")
        return_dict = model_kwargs.pop("return_dict")

        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided for packed attention")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot supply both input_ids and inputs_embeds")

        if token_type_ids is None:
            ref = inputs_embeds if inputs_embeds is not None else input_ids
            shape = ref.size()[:-1] if inputs_embeds is not None else ref.size()
            token_type_ids = torch.zeros(shape, dtype=torch.long, device=ref.device)

        embedding_kwargs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "mask": mask_info["token_mask"],
            "inputs_embeds": inputs_embeds,
        }
        if bbox is not None:
            embedding_kwargs["bbox"] = bbox
            if layout_input_mask is not None:
                embedding_kwargs["layout_input_mask"] = layout_input_mask
        if page_token_ids is not None:
            embedding_kwargs["page_token_ids"] = page_token_ids
        embedding_output = self.model.embeddings(**embedding_kwargs)

        encoder_kwargs = {
            "output_hidden_states": True,
            "output_attentions": output_attentions,
            "return_dict": True,
        }
        if bbox is not None:
            encoder_kwargs["bbox"] = bbox
            if layout_input_mask is not None:
                encoder_kwargs["layout_input_mask"] = layout_input_mask
        encoder_outputs = self.model.encoder(
            embedding_output,
            mask_info["block_mask"],
            **encoder_kwargs,
        )

        encoded_layers = list(encoder_outputs.hidden_states)
        if getattr(self.model, "z_steps", 0) > 1:
            hidden_states = encoded_layers[-2]
            layers = [self.model.encoder.layer[-1] for _ in range(self.model.z_steps)]
            query_states = encoded_layers[-1]
            rel_embeddings = self.model.encoder.get_rel_embedding()
            attention_mask = self.model.encoder.get_attention_mask(mask_info["block_mask"])
            rel_pos = self.model.encoder.get_rel_pos(embedding_output)
            layout_attention_bias = None
            if bbox is not None and hasattr(self.model.encoder, "get_layout_attention_bias"):
                layout_attention_bias = self.model.encoder.get_layout_attention_bias(
                    bbox,
                    layout_input_mask,
                )
            for layer in layers[1:]:
                layer_kwargs = {
                    "output_attentions": False,
                    "query_states": query_states,
                    "relative_pos": rel_pos,
                    "rel_embeddings": rel_embeddings,
                }
                if layout_attention_bias is not None:
                    layer_kwargs["layout_attention_bias"] = layout_attention_bias
                query_states = layer(hidden_states, attention_mask, **layer_kwargs)
                if isinstance(query_states, (tuple, list)):
                    query_states = query_states[0]
                encoded_layers.append(query_states)

        sequence_output = encoded_layers[-1]
        hidden_states_tuple = tuple(encoded_layers) if produce_hidden else None
        attentions = encoder_outputs.attentions if output_attentions else None

        if not return_dict:
            result = (sequence_output,)
            if hidden_states_tuple is not None:
                result += (hidden_states_tuple,)
            if attentions is not None:
                result += (attentions,)
            return result

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=hidden_states_tuple,
            attentions=attentions,
        )

    def _forward_modernbert(
        self,
        input_ids: Optional[torch.Tensor],
        model_kwargs: Dict[str, Any],
        mask_info: Dict[str, torch.Tensor],
    ) -> BaseModelOutput:
        inputs_embeds = model_kwargs.pop("inputs_embeds", None)
        position_ids = model_kwargs.pop("position_ids", None)
        cu_seqlens = model_kwargs.pop("cu_seqlens", None)
        max_seqlen = model_kwargs.pop("max_seqlen", None)
        batch_size = model_kwargs.pop("batch_size", None)
        seq_len = model_kwargs.pop("seq_len", None)
        output_attentions = model_kwargs.pop("output_attentions")
        output_hidden_states = model_kwargs.pop("output_hidden_states")
        return_dict = model_kwargs.pop("return_dict")

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("ModernBERT requires exactly one of input_ids or inputs_embeds")

        token_mask_bool = mask_info["token_mask_bool"].to(torch.bool)
        if batch_size is None or seq_len is None:
            ref = inputs_embeds if inputs_embeds is not None else input_ids
            batch_size, seq_len = ref.shape[:2]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        original_impl = self.model.config._attn_implementation
        if original_impl == "flash_attention_2":
            self.model.config._attn_implementation = "eager"

        self.model._maybe_set_compile()
        global_attention_mask, sliding_window_mask = self.model._update_attention_mask(
            token_mask_bool,
            output_attentions=output_attentions,
        )

        block = mask_info["block_mask"].unsqueeze(1)
        neg_inf = torch.finfo(global_attention_mask.dtype).min
        global_attention_mask = global_attention_mask.masked_fill(~block, neg_inf)
        sliding_window_mask = sliding_window_mask.masked_fill(~block, neg_inf)

        hidden_states = self.model.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for encoder_layer in self.model.layers:
            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=global_attention_mask,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions and len(layer_outputs) > 1:
                all_self_attentions = (*all_self_attentions, layer_outputs[1])

        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        hidden_states = self.model.final_norm(hidden_states)
        if original_impl == "flash_attention_2":
            self.model.config._attn_implementation = original_impl

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def _forward_t5(
        self,
        input_ids: Optional[torch.Tensor],
        model_kwargs: Dict[str, Any],
        mask_info: Dict[str, torch.Tensor],
    ) -> BaseModelOutput:
        stack = self.model.encoder
        kw_input_ids = model_kwargs.pop("input_ids", None)
        if input_ids is None or kw_input_ids is not None:
            input_ids = kw_input_ids

        inputs_embeds = model_kwargs.pop("inputs_embeds", None)
        head_mask = model_kwargs.pop("head_mask", None)
        past_key_values = model_kwargs.pop("past_key_values", None)
        use_cache = model_kwargs.pop("use_cache", stack.config.use_cache)
        output_attentions = model_kwargs.pop("output_attentions")
        output_hidden_states = model_kwargs.pop("output_hidden_states")
        return_dict = model_kwargs.pop("return_dict")
        cache_position = model_kwargs.pop("cache_position", None)

        if model_kwargs:
            raise ValueError(f"Unsupported kwargs for T5 forward: {list(model_kwargs.keys())}")

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided")
            inputs_embeds = stack.embed_tokens(input_ids)
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = inputs_embeds.device
        if cache_position is None:
            cache_position = torch.arange(seq_length, device=device)

        block_mask = mask_info["block_mask"].to(device=device, dtype=torch.bool)
        dtype = inputs_embeds.dtype
        neg_inf = torch.finfo(dtype).min
        causal_mask = torch.zeros(block_mask.shape, dtype=dtype, device=device)
        causal_mask = causal_mask.masked_fill(~block_mask, neg_inf).unsqueeze(1)

        head_mask = stack.get_head_mask(head_mask, stack.config.num_layers)
        hidden_states = stack.dropout(inputs_embeds)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        position_bias = None

        for idx, layer_module in enumerate(stack.block):
            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)
            layer_head_mask = head_mask[idx] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=causal_mask,
                position_bias=position_bias,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                encoder_decoder_position_bias=None,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=None,
                past_key_values=None if not use_cache else past_key_values,
                use_cache=False,
                output_attentions=output_attentions,
                return_dict=True,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]
            position_bias = layer_outputs[1]
            if output_attentions:
                all_attentions = (*all_attentions, layer_outputs[2])

        hidden_states = stack.final_layer_norm(hidden_states)
        hidden_states = stack.dropout(hidden_states)
        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        if not return_dict:
            result = (hidden_states,)
            if output_hidden_states:
                result += (all_hidden_states,)
            if output_attentions:
                result += (all_attentions,)
            return result
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


__all__ = ["Transformer", "hidden_size"]
