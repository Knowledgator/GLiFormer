"""Data collators for GLiNExT processor variants."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch

from .processor import (
    BaseGLiNextProcessor,
    GLiNextAudioProcessor,
    GLiNextLayoutProcessor,
    GLiNextOmniProcessor,
    GLiNextProcessor,
    GLiNextTextProcessor,
    GLiNextVisionProcessor,
)


class BaseGLiNExTDataCollator(ABC):
    """Common two-stage collation for all GLiNExT processors.

    Processors own variant-specific collation/tokenization. Collators keep the
    DataLoader contract stable and attach model/decoder metadata that should
    not be duplicated in every processor.
    """

    def __init__(
        self,
        config,
        data_processor: Optional[BaseGLiNextProcessor] = None,
        return_tokens: bool = False,
        return_id_to_classes: bool = False,
        return_entities: bool = False,
        prepare_labels: bool = True,
        return_classes_mapping: bool = True,
    ):
        self.config = config
        self.data_processor = data_processor
        self.return_tokens = return_tokens
        self.return_id_to_classes = return_id_to_classes
        self.return_entities = return_entities
        self.prepare_labels = prepare_labels
        self.return_classes_mapping = return_classes_mapping

    def collate_batch(self, input_x: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        if self.data_processor is None:
            raise ValueError("data_processor must be provided for GLiNExT collation.")
        return self.data_processor.collate_raw_batch(input_x, **kwargs)

    def collate_function(self, raw_batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if self.data_processor is None:
            raise ValueError("data_processor must be provided for GLiNExT collation.")
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

    def _add_common_returns(self, model_input: Dict[str, Any], raw_batch: Dict[str, Any]) -> None:
        if self.return_classes_mapping and "classes_mapping" not in model_input:
            model_input["classes_mapping"] = raw_batch.get("classes_mapping")
        if self.return_tokens:
            model_input["tokens"] = raw_batch.get("tokens")
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


class GLiNExTTextDataCollator(BaseGLiNExTDataCollator):
    """Collator for text-only GLiNExT models."""

    data_processor_type = GLiNextTextProcessor

    def _add_text_fields(self, model_input: Dict[str, Any], raw_batch: Dict[str, Any]) -> None:
        model_input["text_lengths"] = raw_batch.get("seq_length")

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
        return self._filter_none_values(model_input)


class GLiNExTLayoutDataCollator(GLiNExTTextDataCollator):
    """Collator for document-layout GLiNExT models.

    Layout processors still use the text path, but may also provide ``bbox`` and
    ``pixel_values`` fields. Those are already added by the processor.
    """

    data_processor_type = GLiNextLayoutProcessor


class GLiNExTVisionDataCollator(BaseGLiNExTDataCollator):
    """Collator for efficient vision-only bi-encoder models."""

    data_processor_type = GLiNextVisionProcessor

    def __call__(self, input_x: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        raw_batch = self.collate_batch(input_x, **kwargs)
        model_input = self.collate_function(
            raw_batch,
            prepare_labels=self.prepare_labels,
            **kwargs,
        )
        self._add_common_returns(model_input, raw_batch)
        return self._filter_none_values(model_input)


class GLiNExTAudioDataCollator(BaseGLiNExTDataCollator):
    """Collator for efficient audio-only bi-encoder models."""

    data_processor_type = GLiNextAudioProcessor

    def __call__(self, input_x: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        raw_batch = self.collate_batch(input_x, **kwargs)
        model_input = self.collate_function(
            raw_batch,
            prepare_labels=self.prepare_labels,
            **kwargs,
        )
        self._add_common_returns(model_input, raw_batch)
        return self._filter_none_values(model_input)


class GLiNExTOmniDataCollator(GLiNExTTextDataCollator):
    """Collator for omni-modal GLiNExT models."""

    data_processor_type = GLiNextOmniProcessor


class GLiNExTDataCollator(GLiNExTOmniDataCollator):
    """Backward-compatible collator retaining the legacy all-capability path."""

    data_processor_type = GLiNextProcessor


def resolve_glinext_collator_class(config):
    """Resolve the collator class matching ``config.model_variant``."""
    variant = getattr(config, "model_variant", None) or "text"
    if variant == "text":
        return GLiNExTTextDataCollator
    if variant == "layout":
        return GLiNExTLayoutDataCollator
    if variant == "vision":
        return GLiNExTVisionDataCollator
    if variant == "audio":
        return GLiNExTAudioDataCollator
    if variant == "omni":
        return GLiNExTOmniDataCollator
    raise ValueError(f"Unknown GLiNExT model_variant: {variant!r}")

