from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from .audio import AudioEncoder, audio_token_mask
from .media import apply_input_mask, get_config_value as _get_config_value
from .text import TextBiEncoder, TextEncoder
from .vision import VisionEncoder, vision_encoder_kwargs, vision_token_mask


@dataclass
class OmniEncoderOutput:
    embeddings: torch.Tensor
    attention_mask: torch.Tensor
    modality_order: Tuple[str, ...]
    modality_lengths: Dict[str, int]
    modality_spans: Dict[str, Tuple[int, int]]
    text_embeddings: Optional[torch.Tensor] = None
    text_attention_mask: Optional[torch.Tensor] = None
    vision_embeddings: Optional[torch.Tensor] = None
    vision_attention_mask: Optional[torch.Tensor] = None
    vision_spatial_shape: Optional[torch.Tensor] = None
    vision_prefix_tokens: Optional[torch.Tensor] = None
    audio_embeddings: Optional[torch.Tensor] = None
    audio_attention_mask: Optional[torch.Tensor] = None
    labels_embeddings: Optional[torch.Tensor] = None


def _ones_mask(embeddings: torch.Tensor) -> torch.Tensor:
    return torch.ones(
        embeddings.shape[:2],
        dtype=torch.long,
        device=embeddings.device,
    )


def _resize_mask(mask: Optional[torch.Tensor], embeddings: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return _ones_mask(embeddings)
    if mask.shape[-1] == embeddings.shape[1]:
        return mask.to(device=embeddings.device)
    return _ones_mask(embeddings)


_TRANSFORMER_KWARGS = {
    "pair_attention_mask",
    "token_type_ids",
    "position_ids",
    "head_mask",
    "output_attentions",
    "output_hidden_states",
    "return_dict",
    "packing_config",
    "token_lengths",
    "layout_input_mask",
    "page_input_mask",
}


class LayoutEncoder(TextEncoder):
    """Text encoder for document-layout backbones.

    Layout-aware Hugging Face models such as LayoutLMv3 accept ``bbox`` and
    optionally ``pixel_values``. Custom GLiNExT backbones such as
    ``LayoutDebertaModel`` accept the same ``bbox`` argument and ignore
    unsupported image inputs through their ``**kwargs``.
    """

    @classmethod
    def _pop_bbox(cls, kwargs: Dict[str, Any], bbox: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        value = kwargs.pop("bbox", None)
        if bbox is None and value is not None:
            bbox = value
        return bbox

    @staticmethod
    def _text_length(input_ids: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor]) -> Optional[int]:
        if input_ids is not None:
            return int(input_ids.shape[1])
        if attention_mask is not None:
            return int(attention_mask.shape[1])
        return None

    def __init__(
        self,
        config: Any,
        from_pretrained: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__(config, from_pretrained=from_pretrained, cache_dir=cache_dir)
        self.layout_image_tokens = bool(_get_config_value(config, "layout_image_tokens", True))
        self._last_layout_extra_mask: Optional[torch.Tensor] = None
        if self.layout_image_tokens:
            self.vision_encoder = VisionEncoder(
                config,
                from_pretrained=from_pretrained,
                cache_dir=cache_dir,
            )
            output_hidden_size = int(_get_config_value(config, "hidden_size", self.model_hidden_size))
            if output_hidden_size != self.model_hidden_size:
                self.layout_vision_projection = nn.Linear(output_hidden_size, self.model_hidden_size)

    @staticmethod
    def _resize_bbox(
        bbox: torch.Tensor,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        text_length = LayoutEncoder._text_length(input_ids, attention_mask)
        if text_length is None or bbox.shape[1] == text_length:
            return bbox
        bbox = bbox[:, :text_length]
        if bbox.shape[1] < text_length:
            bbox = torch.nn.functional.pad(bbox, (0, 0, 0, text_length - bbox.shape[1]))
        return bbox

    @staticmethod
    def _resize_page_token_ids(
        page_token_ids: torch.Tensor,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        text_length = LayoutEncoder._text_length(input_ids, attention_mask)
        if text_length is None or page_token_ids.shape[1] == text_length:
            return page_token_ids
        page_token_ids = page_token_ids[:, :text_length]
        if page_token_ids.shape[1] < text_length:
            page_token_ids = torch.nn.functional.pad(page_token_ids, (0, text_length - page_token_ids.shape[1]))
        return page_token_ids

    def _supports_page_token_ids(self) -> bool:
        embeddings = getattr(getattr(self.bert_layer, "model", None), "embeddings", None)
        return getattr(embeddings, "page_embeddings", None) is not None

    def _supports_page_input_mask(self) -> bool:
        model = getattr(self.bert_layer, "model", None)
        return bool(getattr(model, "supports_page_input_mask", False))

    def _text_input_embeddings(
        self,
        input_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            return inputs_embeds
        if input_ids is None:
            raise ValueError("input_ids or inputs_embeds are required")
        return self.get_input_embeddings()(input_ids)

    @staticmethod
    def _image_batch_index(
        pixel_values: torch.Tensor,
        image_batch_idx: Optional[torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        if image_batch_idx is None:
            if pixel_values.shape[0] == batch_size:
                return torch.arange(batch_size, device=pixel_values.device, dtype=torch.long)
            raise ValueError(
                "image_batch_idx is required when pixel_values contains a flattened "
                "set of page images whose count differs from the text batch size."
            )
        image_batch_idx = image_batch_idx.to(device=pixel_values.device, dtype=torch.long).view(-1)
        if image_batch_idx.numel() != pixel_values.shape[0]:
            raise ValueError(
                f"image_batch_idx must contain one entry per image, got {image_batch_idx.numel()} "
                f"for {pixel_values.shape[0]} images."
            )
        return image_batch_idx

    @staticmethod
    def _image_page_ids(
        image_page_ids: Optional[torch.Tensor],
        image_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        if image_page_ids is None:
            return torch.zeros(image_count, dtype=torch.long, device=device)
        image_page_ids = image_page_ids.to(device=device, dtype=torch.long).view(-1)
        if image_page_ids.numel() != image_count:
            raise ValueError(
                f"image_page_ids must contain one entry per image, got {image_page_ids.numel()} "
                f"for {image_count} images."
            )
        return image_page_ids

    def _encode_vision_pages(
        self,
        pixel_values: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if pixel_values is None or not hasattr(self, "vision_encoder"):
            return None
        image_embeddings = self.vision_encoder(pixel_values)
        if hasattr(self, "layout_vision_projection"):
            image_embeddings = self.layout_vision_projection(image_embeddings)
        return image_embeddings

    def _combine_text_and_image_inputs(
        self,
        text_inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        bbox: Optional[torch.Tensor],
        page_token_ids: Optional[torch.Tensor],
        pixel_values: Optional[torch.Tensor],
        vision_attention_mask: Optional[torch.Tensor],
        image_batch_idx: Optional[torch.Tensor],
        image_page_ids: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        self._last_layout_extra_mask = None
        image_embeddings = self._encode_vision_pages(pixel_values)
        if image_embeddings is None:
            if attention_mask is None:
                attention_mask = torch.ones(text_inputs.shape[:2], dtype=torch.long, device=text_inputs.device)
            return text_inputs, attention_mask.to(device=text_inputs.device), bbox, page_token_ids

        batch_size, text_len = text_inputs.shape[:2]
        image_embeddings = image_embeddings.to(device=text_inputs.device, dtype=text_inputs.dtype)
        image_batch_idx = self._image_batch_index(pixel_values, image_batch_idx, batch_size).to(device=text_inputs.device)
        image_page_ids = self._image_page_ids(image_page_ids, image_embeddings.shape[0], text_inputs.device)
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, text_len, dtype=torch.long, device=text_inputs.device)
        else:
            attention_mask = attention_mask.to(device=text_inputs.device)

        if vision_attention_mask is not None and vision_attention_mask.shape[-1] == image_embeddings.shape[1]:
            image_masks = vision_attention_mask.to(device=text_inputs.device, dtype=attention_mask.dtype)
        else:
            image_masks = torch.ones(
                image_embeddings.shape[:2],
                dtype=attention_mask.dtype,
                device=text_inputs.device,
            )

        grouped_embeddings: list[torch.Tensor] = []
        grouped_masks: list[torch.Tensor] = []
        grouped_bboxes: list[Optional[torch.Tensor]] = []
        grouped_pages: list[Optional[torch.Tensor]] = []
        max_len = 0
        for batch_idx in range(batch_size):
            selected = torch.where(image_batch_idx == batch_idx)[0]
            image_part = image_embeddings[selected].reshape(-1, image_embeddings.shape[-1])
            image_mask = image_masks[selected].reshape(-1)
            parts = [text_inputs[batch_idx], image_part]
            masks = [attention_mask[batch_idx], image_mask]
            combined = torch.cat(parts, dim=0)
            combined_mask = torch.cat(masks, dim=0)
            grouped_embeddings.append(combined)
            grouped_masks.append(combined_mask)
            max_len = max(max_len, int(combined.shape[0]))

            if bbox is not None:
                image_bbox = torch.zeros(
                    image_part.shape[0],
                    4,
                    dtype=bbox.dtype,
                    device=text_inputs.device,
                )
                grouped_bboxes.append(torch.cat([bbox[batch_idx].to(device=text_inputs.device), image_bbox], dim=0))
            if page_token_ids is not None:
                image_pages = image_page_ids[selected].repeat_interleave(image_embeddings.shape[1])
                grouped_pages.append(torch.cat([page_token_ids[batch_idx].to(device=text_inputs.device), image_pages], dim=0))

        combined_inputs = torch.zeros(
            batch_size,
            max_len,
            text_inputs.shape[-1],
            dtype=text_inputs.dtype,
            device=text_inputs.device,
        )
        combined_mask = torch.zeros(
            batch_size,
            max_len,
            dtype=attention_mask.dtype,
            device=text_inputs.device,
        )
        combined_bbox = None
        if grouped_bboxes:
            combined_bbox = torch.zeros(batch_size, max_len, 4, dtype=bbox.dtype, device=text_inputs.device)
        combined_pages = None
        if grouped_pages:
            combined_pages = torch.zeros(batch_size, max_len, dtype=torch.long, device=text_inputs.device)

        for batch_idx, (embeds, mask) in enumerate(zip(grouped_embeddings, grouped_masks)):
            length = int(embeds.shape[0])
            combined_inputs[batch_idx, :length] = embeds
            combined_mask[batch_idx, :length] = mask
            if combined_bbox is not None:
                combined_bbox[batch_idx, :length] = grouped_bboxes[batch_idx]
            if combined_pages is not None:
                combined_pages[batch_idx, :length] = grouped_pages[batch_idx]

        self._last_layout_extra_mask = combined_mask[:, text_len:]
        return combined_inputs, combined_mask, combined_bbox, combined_pages

    @staticmethod
    def _truncate_to_text_tokens(
        token_embeddings: torch.Tensor,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        text_length = LayoutEncoder._text_length(input_ids, attention_mask)
        if text_length is not None and token_embeddings.shape[1] > text_length:
            return token_embeddings[:, :text_length]
        return token_embeddings

    def _encode_layout_tokens(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        page_token_ids: Optional[torch.Tensor] = None,
        page_input_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        vision_attention_mask: Optional[torch.Tensor] = None,
        image_batch_idx: Optional[torch.Tensor] = None,
        image_page_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        bbox = LayoutEncoder._pop_bbox(kwargs, bbox)
        model_kwargs = dict(kwargs)
        page_token_ids = model_kwargs.pop("page_token_ids", page_token_ids)
        page_input_mask = model_kwargs.pop("page_input_mask", page_input_mask)
        vision_attention_mask = model_kwargs.pop("vision_attention_mask", vision_attention_mask)
        image_batch_idx = model_kwargs.pop("image_batch_idx", image_batch_idx)
        image_page_ids = model_kwargs.pop("image_page_ids", image_page_ids)
        text_inputs_embeds = None
        combined_image_tokens = False
        original_input_ids = input_ids
        original_attention_mask = attention_mask
        if bbox is not None:
            if not isinstance(bbox, torch.Tensor):
                bbox = torch.as_tensor(bbox)
            if input_ids is not None:
                bbox = bbox.to(device=input_ids.device)
            elif attention_mask is not None:
                bbox = bbox.to(device=attention_mask.device)
            bbox = LayoutEncoder._resize_bbox(bbox, input_ids, attention_mask)
            model_kwargs["bbox"] = bbox

        if page_token_ids is not None:
            if not isinstance(page_token_ids, torch.Tensor):
                page_token_ids = torch.as_tensor(page_token_ids)
            if input_ids is not None:
                page_token_ids = page_token_ids.to(device=input_ids.device)
            elif attention_mask is not None:
                page_token_ids = page_token_ids.to(device=attention_mask.device)
            page_token_ids = LayoutEncoder._resize_page_token_ids(page_token_ids, input_ids, attention_mask)
            if self._supports_page_token_ids():
                model_kwargs["page_token_ids"] = page_token_ids
        elif pixel_values is not None and hasattr(self, "vision_encoder") and self._supports_page_token_ids():
            ref = input_ids if input_ids is not None else attention_mask
            if ref is not None:
                page_token_ids = torch.zeros(ref.shape[:2], dtype=torch.long, device=ref.device)

        if page_token_ids is not None and self._supports_page_input_mask():
            if page_input_mask is None:
                page_input_mask = torch.ones(
                    page_token_ids.shape[0],
                    dtype=torch.bool,
                    device=page_token_ids.device,
                )
            elif not isinstance(page_input_mask, torch.Tensor):
                page_input_mask = torch.as_tensor(page_input_mask)
            page_input_mask = page_input_mask.to(device=page_token_ids.device, dtype=torch.bool)
            model_kwargs["page_input_mask"] = page_input_mask

        if pixel_values is not None and hasattr(self, "vision_encoder"):
            text_inputs_embeds = self._text_input_embeddings(input_ids, inputs_embeds)
            combined = self._combine_text_and_image_inputs(
                text_inputs_embeds,
                attention_mask,
                bbox,
                page_token_ids if self._supports_page_token_ids() else None,
                pixel_values,
                vision_attention_mask,
                image_batch_idx,
                image_page_ids,
            )
            inputs_embeds, attention_mask, bbox, page_token_ids = combined
            input_ids = None
            combined_image_tokens = True
            model_kwargs["attention_mask"] = attention_mask
            if bbox is not None:
                model_kwargs["bbox"] = bbox
            else:
                model_kwargs.pop("bbox", None)
            if page_token_ids is not None and self._supports_page_token_ids():
                model_kwargs["page_token_ids"] = page_token_ids
                if page_input_mask is not None and self._supports_page_input_mask():
                    model_kwargs["page_input_mask"] = page_input_mask
            elif "page_token_ids" in model_kwargs:
                model_kwargs.pop("page_token_ids", None)
                model_kwargs.pop("page_input_mask", None)
        elif attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask

        if pixel_values is not None and not hasattr(self, "vision_encoder"):
            model_kwargs["pixel_values"] = pixel_values

        if inputs_embeds is not None:
            model_kwargs["inputs_embeds"] = inputs_embeds

        token_embeddings = self.bert_layer(
            input_ids=input_ids,
            **model_kwargs,
        )
        if not combined_image_tokens:
            token_embeddings = LayoutEncoder._truncate_to_text_tokens(
                token_embeddings,
                original_input_ids,
                original_attention_mask,
            )
        if hasattr(self, "projection"):
            token_embeddings = self.projection(token_embeddings)
        return token_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        page_token_ids: Optional[torch.Tensor] = None,
        page_input_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._encode_layout_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            page_token_ids=page_token_ids,
            page_input_mask=page_input_mask,
            pixel_values=pixel_values,
            **kwargs,
        )

    def encode_inputs_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        kwargs.pop("packing_config", None)
        kwargs.pop("token_lengths", None)
        if args:
            raise TypeError("LayoutEncoder.encode_inputs_embeds only accepts keyword layout arguments")
        return LayoutEncoder._encode_layout_tokens(
            self,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


class LayoutBiEncoder(TextBiEncoder):
    """Layout encoder with optional text-label bi-encoder support."""

    def __init__(
        self,
        config: Any,
        from_pretrained: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__(config, from_pretrained=from_pretrained, cache_dir=cache_dir)
        self.layout_image_tokens = bool(_get_config_value(config, "layout_image_tokens", True))
        self._last_layout_extra_mask: Optional[torch.Tensor] = None
        if self.layout_image_tokens:
            self.vision_encoder = VisionEncoder(
                config,
                from_pretrained=from_pretrained,
                cache_dir=cache_dir,
            )
            output_hidden_size = int(_get_config_value(config, "hidden_size", self.model_hidden_size))
            if output_hidden_size != self.model_hidden_size:
                self.layout_vision_projection = nn.Linear(output_hidden_size, self.model_hidden_size)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels_input_ids: Optional[torch.Tensor] = None,
        labels_attention_mask: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        page_token_ids: Optional[torch.Tensor] = None,
        page_input_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        token_embeddings = LayoutEncoder._encode_layout_tokens(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            page_token_ids=page_token_ids,
            page_input_mask=page_input_mask,
            pixel_values=pixel_values,
            **kwargs,
        )
        if labels_input_ids is None or labels_attention_mask is None:
            return token_embeddings
        labels_embeddings = self.encode_labels(labels_input_ids, labels_attention_mask)
        return token_embeddings, labels_embeddings

    def encode_inputs_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return LayoutEncoder.encode_inputs_embeds(self, inputs_embeds, attention_mask, *args, **kwargs)


class OmniEncoder(nn.Module):
    """Omni encoder that contextualizes all modalities with the GLiNER text encoder.

    Text token ids are converted with the text backbone input embedding table.
    Vision/audio backbones produce modality tokens, those tokens are projected
    into the text transformer's embedding dimension, and the combined sequence is
    encoded through ``inputs_embeds``.
    """

    modalities: Tuple[str, ...] = ("text", "vision", "audio")
    text_encoder_cls = TextEncoder

    def __init__(
        self,
        config: Any,
        from_pretrained: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.text_encoder = self.text_encoder_cls(
            config,
            from_pretrained=from_pretrained,
            cache_dir=cache_dir,
        )
        self.model_hidden_size = self.text_encoder.model_hidden_size
        self.output_hidden_size = int(_get_config_value(config, "hidden_size", self.model_hidden_size))

        enabled_modalities = _get_config_value(config, "omni_modalities", self.modalities)
        if enabled_modalities is None:
            enabled_modalities = self.modalities
        self.enabled_modalities = set(enabled_modalities)

        self.feature_encoders = nn.ModuleDict()
        if "vision" in self.enabled_modalities:
            self.feature_encoders["vision"] = VisionEncoder(
                config,
                from_pretrained=from_pretrained,
                cache_dir=cache_dir,
            )
        if "audio" in self.enabled_modalities:
            self.feature_encoders["audio"] = AudioEncoder(
                config,
                from_pretrained=from_pretrained,
                cache_dir=cache_dir,
            )

        if "text" not in self.enabled_modalities and not self.feature_encoders:
            raise ValueError("At least one omni modality must be enabled")

        self.input_projections = nn.ModuleDict()
        if "vision" in self.feature_encoders and self.output_hidden_size != self.model_hidden_size:
            self.input_projections["vision"] = nn.Linear(self.output_hidden_size, self.model_hidden_size)
        if "audio" in self.feature_encoders and self.output_hidden_size != self.model_hidden_size:
            self.input_projections["audio"] = nn.Linear(self.output_hidden_size, self.model_hidden_size)

        self.modality_embeddings = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(self.model_hidden_size))
                for name in self.enabled_modalities
            }
        )

    def resize_token_embeddings(
        self,
        new_num_tokens: int,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        return self.text_encoder.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.text_encoder.get_input_embeddings()

    def _add_modality_embedding(self, name: str, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings + self.modality_embeddings[name].view(1, 1, -1).to(dtype=embeddings.dtype)

    def _project_inputs(self, name: str, embeddings: torch.Tensor) -> torch.Tensor:
        if name in self.input_projections:
            return self.input_projections[name](embeddings)
        return embeddings

    def _text_inputs(
        self,
        input_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if "text" not in self.enabled_modalities:
            return None, None
        if inputs_embeds is None:
            if input_ids is None:
                return None, None
            inputs_embeds = self.get_input_embeddings()(input_ids)
        return self._add_modality_embedding("text", inputs_embeds), _resize_mask(attention_mask, inputs_embeds)

    def _vision_inputs(
        self,
        pixel_values: Optional[torch.Tensor],
        vision_attention_mask: Optional[torch.Tensor],
        vision_input_mask: Optional[torch.Tensor],
        kwargs: Dict[str, Any],
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        if "vision" not in self.feature_encoders or pixel_values is None:
            return None, None, None, None
        features = self.feature_encoders["vision"].forward_features(
            pixel_values,
            **vision_encoder_kwargs(kwargs),
        )
        embeddings = features.token_embeddings
        mask = vision_token_mask(
            embeddings,
            vision_attention_mask,
            features.prefix_tokens,
            features.spatial_shape,
        )
        mask = apply_input_mask(mask, vision_input_mask)
        embeddings = self._project_inputs("vision", embeddings)
        return (
            self._add_modality_embedding("vision", embeddings),
            mask,
            features.spatial_shape,
            features.prefix_tokens,
        )

    def _audio_inputs(
        self,
        audio_values: Optional[torch.Tensor],
        audio_attention_mask: Optional[torch.Tensor],
        audio_input_mask: Optional[torch.Tensor],
        kwargs: Dict[str, Any],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if "audio" not in self.feature_encoders or audio_values is None:
            return None, None
        embeddings = self.feature_encoders["audio"](
            audio_values,
            attention_mask=audio_attention_mask,
        )
        mask = audio_token_mask(
            self.feature_encoders["audio"],
            embeddings,
            audio_attention_mask,
        )
        mask = apply_input_mask(mask, audio_input_mask)
        embeddings = self._project_inputs("audio", embeddings)
        return self._add_modality_embedding("audio", embeddings), mask

    @staticmethod
    def _modality_spans(order: Tuple[str, ...], lengths: Dict[str, int]) -> Dict[str, Tuple[int, int]]:
        spans: Dict[str, Tuple[int, int]] = {}
        offset = 0
        for name in order:
            length = lengths[name]
            spans[name] = (offset, offset + length)
            offset += length
        return spans

    @staticmethod
    def _split_encoded(
        encoded: torch.Tensor,
        spans: Dict[str, Tuple[int, int]],
    ) -> Dict[str, Optional[torch.Tensor]]:
        chunks: Dict[str, Optional[torch.Tensor]] = {"text": None, "vision": None, "audio": None}
        for name, (start, end) in spans.items():
            chunks[name] = encoded[:, start:end]
        return chunks

    @staticmethod
    def _resize_bbox(bbox: torch.Tensor, length: int) -> torch.Tensor:
        if bbox.shape[1] == length:
            return bbox
        bbox = bbox[:, :length]
        if bbox.shape[1] < length:
            pad = length - bbox.shape[1]
            bbox = torch.nn.functional.pad(bbox, (0, 0, 0, pad))
        return bbox

    @staticmethod
    def _get_layout_bbox(kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
        return kwargs.get("bbox")

    def _combined_bbox(
        self,
        kwargs: Dict[str, Any],
        active_parts: list[tuple[str, torch.Tensor, torch.Tensor]],
        combined_inputs: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        bbox = self._get_layout_bbox(kwargs)
        if bbox is None:
            return None
        if not isinstance(bbox, torch.Tensor):
            bbox = torch.as_tensor(bbox)
        if bbox.dim() != 3 or bbox.shape[-1] != 4:
            raise ValueError(f"bbox must have shape (batch, seq_len, 4), got {tuple(bbox.shape)}")
        if bbox.shape[0] != combined_inputs.shape[0]:
            raise ValueError(
                f"bbox batch size {bbox.shape[0]} does not match omni batch size {combined_inputs.shape[0]}"
            )
        if bbox.shape[1] == combined_inputs.shape[1]:
            return bbox.to(device=combined_inputs.device)

        chunks = []
        for name, embeds, _ in active_parts:
            length = int(embeds.shape[1])
            if name == "text":
                chunk = self._resize_bbox(bbox, length)
                chunk = chunk.to(device=combined_inputs.device)
            else:
                chunk = torch.zeros(
                    bbox.shape[0],
                    length,
                    4,
                    dtype=bbox.dtype,
                    device=combined_inputs.device,
                )
            chunks.append(chunk)
        return torch.cat(chunks, dim=1)

    def encode_omni(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        vision_attention_mask: Optional[torch.Tensor] = None,
        vision_input_mask: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        audio_input_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> OmniEncoderOutput:
        text_inputs, text_mask = self._text_inputs(input_ids, inputs_embeds, attention_mask)
        (
            vision_inputs,
            vision_mask,
            vision_spatial_shape,
            vision_prefix_tokens,
        ) = self._vision_inputs(
            pixel_values, vision_attention_mask, vision_input_mask, kwargs,
        )
        audio_inputs, audio_mask = self._audio_inputs(
            audio_values, audio_attention_mask, audio_input_mask, kwargs,
        )

        parts = [
            ("text", text_inputs, text_mask),
            ("vision", vision_inputs, vision_mask),
            ("audio", audio_inputs, audio_mask),
        ]
        active_parts = [(name, embeds, mask) for name, embeds, mask in parts if embeds is not None]
        if not active_parts:
            raise ValueError("No enabled omni encoder received inputs")

        combined_inputs = torch.cat([embeds for _, embeds, _ in active_parts], dim=1)
        combined_mask = torch.cat([mask for _, _, mask in active_parts], dim=1)
        modality_order = tuple(name for name, _, _ in active_parts)
        modality_lengths = {name: int(embeds.shape[1]) for name, embeds, _ in active_parts}
        modality_spans = self._modality_spans(modality_order, modality_lengths)

        text_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in _TRANSFORMER_KWARGS
        }
        combined_bbox = self._combined_bbox(kwargs, active_parts, combined_inputs)
        if combined_bbox is not None:
            text_kwargs["bbox"] = combined_bbox
        encoded = self.text_encoder.encode_inputs_embeds(
            combined_inputs,
            combined_mask,
            **text_kwargs,
        )
        chunks = self._split_encoded(encoded, modality_spans)

        return OmniEncoderOutput(
            embeddings=encoded,
            attention_mask=combined_mask,
            modality_order=modality_order,
            modality_lengths=modality_lengths,
            modality_spans=modality_spans,
            text_embeddings=chunks["text"],
            text_attention_mask=text_mask,
            vision_embeddings=chunks["vision"],
            vision_attention_mask=vision_mask,
            vision_spatial_shape=vision_spatial_shape,
            vision_prefix_tokens=vision_prefix_tokens,
            audio_embeddings=chunks["audio"],
            audio_attention_mask=audio_mask,
        )

    def forward(
        self,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Union[OmniEncoderOutput, Tuple[torch.Tensor, torch.Tensor]]:
        output = self.encode_omni(**kwargs)
        if not return_dict:
            return output.embeddings, output.attention_mask
        return output


class OmniBiEncoder(OmniEncoder):
    """Omni encoder with GLiNER bi-encoder label support."""

    text_encoder_cls = TextBiEncoder

    def encode_labels(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.text_encoder.encode_labels(input_ids, attention_mask, **kwargs)

    def forward(
        self,
        labels_input_ids: Optional[torch.Tensor] = None,
        labels_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Union[OmniEncoderOutput, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        output = self.encode_omni(**kwargs)
        labels_embeddings = None
        if labels_input_ids is not None and labels_attention_mask is not None:
            labels_embeddings = self.encode_labels(labels_input_ids, labels_attention_mask)
        output.labels_embeddings = labels_embeddings
        if not return_dict:
            return output.embeddings, output.attention_mask, labels_embeddings
        return output


class TextVisionOmniEncoder(OmniEncoder):
    modalities = ("text", "vision")


class TextVisionOmniBiEncoder(OmniBiEncoder):
    modalities = ("text", "vision")


class TextAudioOmniEncoder(OmniEncoder):
    modalities = ("text", "audio")


class TextAudioOmniBiEncoder(OmniBiEncoder):
    modalities = ("text", "audio")


class VisionAudioOmniEncoder(OmniEncoder):
    modalities = ("vision", "audio")


class TriOmniEncoder(OmniEncoder):
    modalities = ("text", "vision", "audio")


class TriOmniBiEncoder(OmniBiEncoder):
    modalities = ("text", "vision", "audio")
