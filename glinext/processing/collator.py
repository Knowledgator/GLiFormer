"""GLiNExTDataCollator — multi-task data collator for training and inference."""

from typing import Any, Dict, List, Optional

from gliner.data_processing.collator import BaseDataCollator

from .processor import GLiNextProcessor


class GLiNExTDataCollator(BaseDataCollator):
    """Data collator for GLiNExT multi-task model.

    Delegates all multi-task complexity to :class:`GLiNextProcessor`.
    Handles both training (with labels) and inference (without labels) modes.

    Args:
        config: GLiNextConfig instance.
        data_processor: GLiNextProcessor that handles tokenization, prompt
            construction, class mapping, and label creation.
        return_tokens: Include original word tokens in output (for decoding).
        return_id_to_classes: Include class ID → name mappings.
        return_entities: Include raw entity annotations.
        prepare_labels: Whether to create training label tensors.
    """

    def __init__(
        self,
        config,
        data_processor: Optional[GLiNextProcessor] = None,
        return_tokens: bool = False,
        return_id_to_classes: bool = False,
        return_entities: bool = False,
        prepare_labels: bool = True,
    ):
        super().__init__(
            config,
            data_processor=data_processor,
            return_tokens=return_tokens,
            return_id_to_classes=return_id_to_classes,
            return_entities=return_entities,
            prepare_labels=prepare_labels,
        )

    def __call__(
        self,
        input_x: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        """Collate a list of examples into a model-ready batch.

        Steps:
            1. ``collate_raw_batch`` — resolve spans, build BatchClassesMapping,
               truncate texts.
            2. ``tokenize_and_prepare_labels`` — tokenize, construct prompts,
               optionally create per-task label tensors.
            3. Attach metadata needed by the model and decoder
               (text_lengths, tokens, classes_mapping).

        Args:
            input_x: List of example dicts. Each has ``tokenized_text`` and
                optional task-specific annotation keys (``extraction``,
                ``classification``, ``open_relex``, ``structuring``,
                ``embedding``).
            **kwargs: Forwarded to the processor.

        Returns:
            Flat dict of tensors and metadata ready for ``GLiNExTModel.forward()``.
        """
        # 1. Raw batch via processor — spans resolved, classes mapped
        raw_batch = self.data_processor.collate_raw_batch(input_x, **kwargs)

        # 2. Tokenize + prepare labels (training) or just tokenize (inference)
        model_input = self.data_processor.tokenize_and_prepare_labels(
            raw_batch, prepare_labels=self.prepare_labels,
        )

        # 3. Attach fields needed by model / decoder
        model_input["text_lengths"] = raw_batch["seq_length"]

        # classes_mapping is needed by the model (for _build_flat_inputs)
        # and by the decoder (for label name resolution)
        if "classes_mapping" not in model_input:
            model_input["classes_mapping"] = raw_batch.get("classes_mapping")

        # 4. Conditional returns
        if self.return_tokens:
            model_input["tokens"] = raw_batch.get("tokens")
        if self.return_id_to_classes:
            model_input["id_to_classes"] = raw_batch.get("id_to_classes")
        if self.return_entities:
            model_input["entities"] = raw_batch.get("entities")

        return self._filter_none_values(model_input)
