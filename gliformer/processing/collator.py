"""Data collators for GLiFormer processor variants."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

import torch

from .label_augmentation import (
    LABEL_AUGMENTATION_INDEX_KEY,
    LABEL_AUGMENTATION_MARKER_KEY,
    BatchLabelAugmenter,
    LabelAugmentationConfig,
)
from ..prompt_features import (
    PROMPT_MARKER_COUNTS_KEY,
    count_prompt_markers,
    prompt_marker_token_indices,
)
from ..encoders.native_vision import NativeVisionHost, uses_native_vision
from .token_units import is_token_level, unit_pieces_batch
from .processor import (
    BaseGLiFormerProcessor,
    GLiFormerAudioProcessor,
    GLiFormerLayoutProcessor,
    GLiFormerOmniProcessor,
    GLiFormerProcessor,
    GLiFormerTextProcessor,
    GLiFormerVisionProcessor,
)


class BaseGLiFormerDataCollator(ABC):
    """Common two-stage collation for all GLiFormer processors.

    Processors own variant-specific collation/tokenization. Collators keep the
    DataLoader contract stable and attach model/decoder metadata that should
    not be duplicated in every processor.
    """

    def __init__(
        self,
        config,
        data_processor: Optional[BaseGLiFormerProcessor] = None,
        return_tokens: bool = False,
        return_id_to_classes: bool = False,
        return_entities: bool = False,
        prepare_labels: bool = True,
        return_classes_mapping: bool = True,
        label_augmentation=None,
    ):
        self.config = config
        self.data_processor = data_processor
        self.return_tokens = return_tokens
        self.return_id_to_classes = return_id_to_classes
        self.return_entities = return_entities
        self.prepare_labels = prepare_labels
        self.return_classes_mapping = return_classes_mapping
        self.label_augmentation = LabelAugmentationConfig.from_value(
            label_augmentation
        )
        is_active = getattr(
            self.label_augmentation,
            "is_active",
            self.label_augmentation.enabled,
        )
        if callable(is_active):
            is_active = is_active()
        self.label_augmenter = (
            BatchLabelAugmenter(self.label_augmentation)
            if is_active
            else None
        )

    def _prepare_augmentation_batch(self, input_x):
        """Strip training markers and return augmenter-specific kwargs."""

        marked = [
            isinstance(item, Mapping)
            and item.get(LABEL_AUGMENTATION_MARKER_KEY) is True
            for item in input_x
        ]
        if any(marked) and not all(marked):
            raise ValueError(
                "A collator batch cannot mix label-augmentation-marked "
                "training records with unmarked evaluation records."
            )
        if not any(marked):
            return input_x, {}

        clean_batch = []
        batch_ids = []
        for position, item in enumerate(input_x):
            clean_item = dict(item)
            clean_item.pop(LABEL_AUGMENTATION_MARKER_KEY, None)
            batch_ids.append(
                clean_item.pop(LABEL_AUGMENTATION_INDEX_KEY, position)
            )
            clean_batch.append(clean_item)

        if self.label_augmenter is None:
            return clean_batch, {}
        return clean_batch, {
            "label_augmenter": self.label_augmenter,
            "label_augmentation_batch_ids": batch_ids,
        }

    def collate_batch(self, input_x: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        if self.data_processor is None:
            raise ValueError("data_processor must be provided for GLiFormer collation.")
        input_x, augmentation_kwargs = self._prepare_augmentation_batch(input_x)
        return self.data_processor.collate_raw_batch(
            input_x,
            **kwargs,
            **augmentation_kwargs,
        )

    def collate_function(self, raw_batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if self.data_processor is None:
            raise ValueError("data_processor must be provided for GLiFormer collation.")
        prepare_labels = kwargs.pop("prepare_labels", self.prepare_labels)
        return self.data_processor.tokenize_and_prepare_labels(
            raw_batch,
            prepare_labels=prepare_labels,
            **kwargs,
        )

    @staticmethod
    def _add_precomputed_lengths(model_input: Dict[str, Any]) -> None:
        attention_mask = model_input.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor) and "token_lengths" not in model_input:
            model_input["token_lengths"] = attention_mask.sum(dim=-1, dtype=torch.int64).tolist()

        text_lengths = model_input.get("text_lengths")
        if isinstance(text_lengths, torch.Tensor) and "word_lengths" not in model_input:
            model_input["word_lengths"] = text_lengths.view(-1).tolist()

    def _add_prompt_marker_counts(self, model_input: Dict[str, Any]) -> None:
        """Per-row prompt-marker counts as Python lists.

        The model sizes its prompt-feature tensors from these, so it never has
        to read them back from the GPU.
        """

        input_ids = model_input.get("input_ids")
        attention_mask = model_input.get("attention_mask")
        if (
            PROMPT_MARKER_COUNTS_KEY in model_input
            or not isinstance(input_ids, torch.Tensor)
            or not isinstance(attention_mask, torch.Tensor)
            or input_ids.device.type != "cpu"
            or attention_mask.device.type != "cpu"
            or input_ids.dim() != 2
            or attention_mask.shape != input_ids.shape
        ):
            return
        token_indices = prompt_marker_token_indices(self.config)
        if token_indices:
            model_input[PROMPT_MARKER_COUNTS_KEY] = count_prompt_markers(
                input_ids,
                attention_mask,
                token_indices,
            )

    def _add_common_returns(self, model_input: Dict[str, Any], raw_batch: Dict[str, Any]) -> None:
        if self.return_classes_mapping and "classes_mapping" not in model_input:
            model_input["classes_mapping"] = raw_batch.get("classes_mapping")
        if self.return_tokens:
            tokens = raw_batch.get("tokens")
            if is_token_level(self.config):
                tokens = unit_pieces_batch(tokens, raw_batch.get("text"))
            model_input["tokens"] = tokens
        if self.return_id_to_classes:
            model_input["id_to_classes"] = raw_batch.get("id_to_classes")
        if self.return_entities:
            model_input["entities"] = raw_batch.get("entities")

    @staticmethod
    def _filter_none_values(data: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in data.items() if value is not None}

    @abstractmethod
    def __call__(self, input_x: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


class GLiFormerTextDataCollator(BaseGLiFormerDataCollator):
    """Collator for text-only GLiFormer models."""

    data_processor_type = GLiFormerTextProcessor

    def _add_text_fields(self, model_input: Dict[str, Any], raw_batch: Dict[str, Any]) -> None:
        # Modern text processors derive the retained source length after the
        # combined schema-prompt/source sequence is subtokenized and truncated.
        # Keep the raw word count only as a compatibility fallback.
        model_input.setdefault("text_lengths", raw_batch.get("seq_length"))

    def __call__(self, input_x: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        raw_batch = self.collate_batch(input_x, **kwargs)
        model_input = self.collate_function(
            raw_batch,
            prepare_labels=self.prepare_labels,
            **kwargs,
        )
        self._add_text_fields(model_input, raw_batch)
        self._add_common_returns(model_input, raw_batch)
        self._add_precomputed_lengths(model_input)
        self._add_prompt_marker_counts(model_input)
        return self._filter_none_values(model_input)


class GLiFormerLayoutDataCollator(GLiFormerTextDataCollator):
    """Collator for document-layout GLiFormer models.

    Layout processors still use the text path, but may also provide ``bbox`` and
    ``pixel_values`` fields. Those are already added by the processor.
    """

    data_processor_type = GLiFormerLayoutProcessor


class GLiFormerVisionDataCollator(BaseGLiFormerDataCollator):
    """Collator for efficient vision-only bi-encoder models."""

    data_processor_type = GLiFormerVisionProcessor

    def __call__(self, input_x: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        raw_batch = self.collate_batch(input_x, **kwargs)
        model_input = self.collate_function(
            raw_batch,
            prepare_labels=self.prepare_labels,
            **kwargs,
        )
        self._add_common_returns(model_input, raw_batch)
        return self._filter_none_values(model_input)


class GLiFormerAudioDataCollator(BaseGLiFormerDataCollator):
    """Collator for efficient audio-only bi-encoder models."""

    data_processor_type = GLiFormerAudioProcessor

    def __call__(self, input_x: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        raw_batch = self.collate_batch(input_x, **kwargs)
        model_input = self.collate_function(
            raw_batch,
            prepare_labels=self.prepare_labels,
            **kwargs,
        )
        self._add_common_returns(model_input, raw_batch)
        return self._filter_none_values(model_input)


class GLiFormerOmniDataCollator(GLiFormerTextDataCollator):
    """Collator for omni-modal GLiFormer models."""

    data_processor_type = GLiFormerOmniProcessor

    def __call__(self, input_x: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        model_input = super().__call__(input_x, **kwargs)
        self._add_native_vision_host(model_input)
        return model_input

    def _add_native_vision_host(self, model_input: Dict[str, Any]) -> None:
        """CPU copies of the inputs that size a native vision batch, so the model
        lays the image blocks out without reading them back from the GPU."""
        if not uses_native_vision(self.config):
            return
        input_ids = model_input.get("input_ids")
        image_grid_thw = model_input.get("image_grid_thw")
        if not torch.is_tensor(input_ids) or not torch.is_tensor(image_grid_thw):
            return
        model_input["native_vision_host"] = NativeVisionHost(
            input_ids=input_ids,
            attention_mask=model_input.get("attention_mask"),
            image_grid_thw=image_grid_thw,
            vision_input_mask=model_input.get("vision_input_mask"),
        )


class GLiFormerDataCollator(GLiFormerOmniDataCollator):
    """Backward-compatible collator retaining the legacy all-capability path."""

    data_processor_type = GLiFormerProcessor


def resolve_gliformer_collator_class(config):
    """Resolve the collator class matching ``config.model_variant``."""
    variant = getattr(config, "model_variant", None) or "text"
    if variant == "text":
        return GLiFormerTextDataCollator
    if variant == "layout":
        return GLiFormerLayoutDataCollator
    if variant == "vision":
        return GLiFormerVisionDataCollator
    if variant == "audio":
        return GLiFormerAudioDataCollator
    if variant == "omni":
        return GLiFormerOmniDataCollator
    raise ValueError(f"Unknown GLiFormer model_variant: {variant!r}")
