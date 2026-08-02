"""GLiNExT — main user-facing class for multi-task information extraction."""

import json
import logging
import inspect
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from huggingface_hub import PyTorchModelHubMixin

from gliner.model import BaseGLiNER
from gliner.data_processing.tokenizer import WordsSplitter

from .config import (
    GLiNextAudioConfig,
    GLiNextConfig,
    GLiNextLayoutConfig,
    GLiNextOmniConfig,
    GLiNextTextConfig,
    GLiNextVisionConfig,
    GLINEXT_MODEL_TYPE_TO_VARIANT,
    GLINEXT_VARIANT_TO_CONFIG_CLASS,
    resolve_glinext_config_class,
)
from .model import (
    GLiNExTAudioModel,
    GLiNExTLayoutModel,
    GLiNExTModel,
    GLiNExTOmniModel,
    GLiNExTTextModel,
    GLiNExTVisionModel,
    resolve_glinext_model_class,
)
from .processing.processor import GLiNextProcessor, resolve_glinext_processor_class
from .processing.pdf import GLiNextPDFProcessor
from .processing.decoder import GLiNExTDecoder
from .processing.collator import (
    GLiNExTAudioDataCollator,
    GLiNExTDataCollator,
    GLiNExTLayoutDataCollator,
    GLiNExTOmniDataCollator,
    GLiNExTTextDataCollator,
    GLiNExTVisionDataCollator,
    resolve_glinext_collator_class,
)
from .processing.schema import GLiNExTSchema

logger = logging.getLogger(__name__)


class BaseGLiNExT(BaseGLiNER):
    """Unified multi-task information extraction model.

    Supports NER, classification, relation extraction (joint & open),
    structuring (JSON schema extraction), embedding, and counting —
    all through a single shared encoder backbone.

    Inherits model loading/saving from :class:`BaseGLiNER` and adds
    multi-task inference with per-task convenience methods.

    Example::

        model = GLiNExT.from_pretrained("knowledgator/glinext-base")

        # Single-task
        entities = model.predict_entities("Apple is a company", ["company", "person"])

        # Multi-task
        results = model.inference(
            ["Apple released the iPhone"],
            entities=["company", "product"],
            classes=["tech_news", "sports_news"],
        )
        # results["ner"] -> List[List[dict]]
        # results["classification"] -> List[List[dict]]

        # Schema-based
        schema = model.create_schema()
        schema.add_entities(["person", "org"])
        schema.add_classes(["positive", "negative"])
        results = model.inference_from_schema(texts, schema)
    """

    config_class = GLiNextConfig
    model_class = None
    data_processor_class = GLiNextProcessor
    data_collator_class = GLiNExTDataCollator
    decoder_class = GLiNExTDecoder

    @classmethod
    def create_training_args(cls, *args, **kwargs):
        """Preserve explicit zero values for non-encoder optimizer settings.

        The installed GLiNER factory selects its fallbacks with boolean ``or``,
        which treats a deliberate ``0.0`` learning rate or weight decay as if it
        were omitted. Keep the upstream construction and defaults, then restore
        only explicitly supplied, non-``None`` values.
        """

        base_factory = super().create_training_args
        supplied = inspect.signature(base_factory).bind_partial(*args, **kwargs)
        training_args = base_factory(*args, **kwargs)
        for name in ("others_lr", "others_weight_decay"):
            value = supplied.arguments.get(name)
            if value is not None:
                setattr(training_args, name, value)
        return training_args

    # ── Setup overrides ───────────────────────────────────────────────

    def _create_model(self, config, backbone_from_pretrained, cache_dir, **kwargs):
        model_cls = self.model_class or resolve_glinext_model_class(config)
        self.model = model_cls(config, from_pretrained=backbone_from_pretrained, cache_dir=cache_dir, **kwargs)
        return self.model

    def _create_data_processor(self, config, cache_dir, tokenizer=None, words_splitter=None, **kwargs):
        """Create processor, loading labels tokenizer for bi-encoder mode."""
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=cache_dir)
            self._set_tokenizer_spec_tokens(tokenizer)

        labels_tokenizer = None
        if config.labels_encoder is not None:
            labels_tokenizer = AutoTokenizer.from_pretrained(config.labels_encoder, cache_dir=cache_dir)
        else:
            variant = getattr(config, "model_variant", "")
            if variant in {"vision", "audio"}:
                labels_tokenizer = tokenizer

        if words_splitter is None:
            words_splitter = WordsSplitter(config.words_splitter_type)

        processor_cls = resolve_glinext_processor_class(config)
        self.data_processor = processor_cls(
            config, tokenizer, words_splitter, labels_tokenizer=labels_tokenizer,
        )
        return self.data_processor

    def _create_data_collator(self, **kwargs):
        collator_cls = self.data_collator_class or resolve_glinext_collator_class(self.config)
        return collator_cls(
            self.config,
            data_processor=self.data_processor,
            prepare_labels=True,
            **kwargs,
        )

    @staticmethod
    def _supported_base_loader_kwargs(loader, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Keep dispatcher kwargs compatible with the installed GLiNER version."""
        signature = inspect.signature(loader)
        parameters = signature.parameters
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_kwargs:
            # Older GLiNER releases accept **model_kwargs and forward unknown
            # loader options into the model constructor. Keep arbitrary caller
            # model kwargs, but only pass GLiNExT dispatcher options when the
            # installed base loader declares support for them.
            dispatcher_only = {"quantize", "dtype"}
            return {
                key: value
                for key, value in kwargs.items()
                if key not in dispatcher_only or key in parameters
            }
        return {
            key: value
            for key, value in kwargs.items()
            if key in parameters
        }

    @classmethod
    def _load_config(cls, config_file: Path, **config_overrides) -> GLiNextConfig:
        """Load GLiNExT config, dispatching factory loads by saved model_type."""
        with open(config_file) as f:
            config_dict = json.load(f)

        for key, value in config_overrides.items():
            if value is not None:
                config_dict[key] = value

        specific_config_classes = (
            GLiNextTextConfig,
            GLiNextLayoutConfig,
            GLiNextVisionConfig,
            GLiNextAudioConfig,
            GLiNextOmniConfig,
        )
        if cls.config_class in specific_config_classes:
            config_cls = cls.config_class
        else:
            config_cls = resolve_glinext_config_class(config_dict)

        model_type = config_dict.pop("model_type", None)
        if model_type in GLINEXT_MODEL_TYPE_TO_VARIANT:
            config_dict["model_variant"] = GLINEXT_MODEL_TYPE_TO_VARIANT[model_type]
        return config_cls(**config_dict)

    def _get_special_tokens(self):
        """Return special tokens to add to the tokenizer.

        Order matters — ``set_class_indices`` looks them up by string after addition.
        """
        # Collect unique parent tokens (per-task may differ from shared)
        parent_tokens = sorted({
            self.config.ner_parent_token,
            self.config.cat_parent_token,
            self.config.open_rel_parent_token,
            self.config.struct_parent_token,
        })

        tokens = [self.config.ent_token, self.config.sep_token] + parent_tokens

        if self.config.classification_config is not None:
            tokens.append(self.config.cat_token)

        if (self.config.joint_relex_config is not None
                or self.config.open_relex_config is not None):
            tokens.append(self.config.rel_token)

        if (
            self.config.structuring_config is not None
            or self.config.set_structuring_config is not None
        ):
            tokens.append(self.config.child_token)
            structuring_configs = (
                self.config.structuring_config,
                self.config.set_structuring_config,
            )
            if any(
                cfg is not None and getattr(cfg, "multi_level", False)
                for cfg in structuring_configs
            ):
                tokens.extend(
                    [
                        self.config.structuring_child_token,
                        self.config.structuring_end_token,
                    ]
                )

        if (self.config.image_classification_config is not None
                or self.config.object_detection_config is not None
                or self.config.segmentation_config is not None
                or self.config.audio_classification_config is not None
                or self.config.audio_segmentation_config is not None):
            tokens.append(self.config.obj_token)

        return tokens

    def set_class_indices(self):
        """Set token indices by looking up each special token in the vocab."""
        tok = self.data_processor.transformer_tokenizer

        def _idx(token_str):
            token_id = tok.convert_tokens_to_ids(token_str)
            if isinstance(token_id, list):
                token_id = token_id[0]
            return token_id

        # Core tokens
        self.config.class_token_index = _idx(self.config.ent_token)
        self.config.parent_token_index = _idx(self.config.parent_token)

        # Per-task parent token indices
        if self.config.ner_config is not None:
            self.config.ner_config.parent_token_index = _idx(self.config.ner_parent_token)
        if self.config.classification_config is not None:
            self.config.classification_config.parent_token_index = _idx(self.config.cat_parent_token)
        if self.config.open_relex_config is not None:
            self.config.open_relex_config.parent_token_index = _idx(self.config.open_rel_parent_token)
        if self.config.structuring_config is not None:
            self.config.structuring_config.parent_token_index = _idx(self.config.struct_parent_token)
        if self.config.set_structuring_config is not None:
            self.config.set_structuring_config.parent_token_index = _idx(
                self.config.struct_parent_token
            )
        for cfg_name in (
            "image_classification_config", "object_detection_config", "segmentation_config",
            "audio_classification_config", "audio_segmentation_config",
        ):
            cfg = getattr(self.config, cfg_name, None)
            if cfg is not None:
                cfg.parent_token_index = _idx(self.config.parent_token)

        # Per-task child token indices
        if self.config.classification_config is not None:
            cat_idx = _idx(self.config.cat_token)
            self.config.classification_config.cat_token_index = cat_idx
            self.config.cat_token_index = cat_idx

        rel_token_added = (self.config.joint_relex_config is not None
                           or self.config.open_relex_config is not None)
        if rel_token_added:
            rel_idx = _idx(self.config.rel_token)
            self.config.rel_token_index = rel_idx
            if self.config.joint_relex_config is not None:
                self.config.joint_relex_config.rel_token_index = rel_idx
            if self.config.open_relex_config is not None:
                self.config.open_relex_config.rel_token_index = rel_idx

        if (
            self.config.structuring_config is not None
            or self.config.set_structuring_config is not None
        ):
            child_idx = _idx(self.config.child_token)
            if self.config.structuring_config is not None:
                self.config.structuring_config.child_token_index = child_idx
            if self.config.set_structuring_config is not None:
                self.config.set_structuring_config.child_token_index = child_idx
            self.config.child_token_index = child_idx

        if (self.config.image_classification_config is not None
                or self.config.object_detection_config is not None
                or self.config.segmentation_config is not None
                or self.config.audio_classification_config is not None
                or self.config.audio_segmentation_config is not None):
            obj_idx = _idx(self.config.obj_token)
            self.config.obj_token_index = obj_idx

    def resize_embeddings(self, set_class_token_index=True):
        """Resize token embeddings to match tokenizer vocabulary."""
        if set_class_token_index:
            self.set_class_indices()

        tokenizer = self.data_processor.transformer_tokenizer
        if len(tokenizer) != self.config.vocab_size:
            new_num_tokens = len(tokenizer)
            token_rep_layer = self.model.token_rep_layer
            if getattr(token_rep_layer, "resizes_labels_encoder_only", False):
                self.config.vocab_size = new_num_tokens
            else:
                model_embeds = token_rep_layer.resize_token_embeddings(new_num_tokens, None)
                self.config.vocab_size = model_embeds.num_embeddings
                if hasattr(self.config, "encoder_config") and self.config.encoder_config is not None:
                    self.config.encoder_config.vocab_size = model_embeds.num_embeddings

    def _build_inference_input(
        self,
        all_tokens: List[List[str]],
        entities=None,
        classes=None,
        relations=None,
        joint_relations=None,
        structures=None,
    ) -> List[Dict[str, Any]]:
        """Build ``input_x`` dicts for the collator from label arguments.

        Each text gets a dict with ``tokenized_text`` plus task-specific
        annotation stubs that tell the processor which tasks are active and
        what their label spaces are.
        """
        input_x = []
        task_processors = self.data_processor.task_processors.values()
        for tokens in all_tokens:
            item: Dict[str, Any] = {"tokenized_text": tokens}
            for processor in task_processors:
                processor.contribute_inference_input(
                    item,
                    entities=entities,
                    classes=classes,
                    relations=relations,
                    joint_relations=joint_relations,
                    structures=structures,
                )
            input_x.append(item)
        return input_x

    def _build_pdf_inference_input(
        self,
        pdf_items: List[Dict[str, Any]],
        entities=None,
        classes=None,
        relations=None,
        joint_relations=None,
        structures=None,
    ) -> List[Dict[str, Any]]:
        input_x = []
        task_processors = self.data_processor.task_processors.values()
        for pdf_item in pdf_items:
            item = dict(pdf_item)
            item["tokenized_text"] = list(pdf_item.get("tokenized_text") or [])
            item.setdefault("text", " ".join(item["tokenized_text"]))
            for processor in task_processors:
                processor.contribute_inference_input(
                    item,
                    entities=entities,
                    classes=classes,
                    relations=relations,
                    joint_relations=joint_relations,
                    structures=structures,
                )
            input_x.append(item)
        return input_x

    @staticmethod
    def _single_or_batch(items, single: bool):
        return items[0] if single else items

    @staticmethod
    def _label_group_count(labels) -> int:
        if isinstance(labels, dict):
            return len(labels)
        return 1 if labels is not None else 0

    @staticmethod
    def _collapse_single_group_results(results: List[Any]) -> List[Any]:
        collapsed = []
        for result in results:
            if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
                collapsed.append(result[0])
            else:
                collapsed.append(result)
        return collapsed

    @staticmethod
    def _normalize_texts(texts: Union[str, List[str]]) -> Tuple[List[str], bool]:
        if isinstance(texts, str):
            return [texts], True
        return list(texts), False

    def prepare_inputs(self, texts: List[str]):
        """Tokenize texts and keep word-token to character-offset mappings."""
        all_tokens = []
        all_start_token_idx_to_text_idx = []
        all_end_token_idx_to_text_idx = []

        for text in texts:
            tokens = []
            start_token_idx_to_text_idx = []
            end_token_idx_to_text_idx = []
            for token, start, end in self.data_processor.words_splitter(text):
                tokens.append(token)
                start_token_idx_to_text_idx.append(start)
                end_token_idx_to_text_idx.append(end)
            all_tokens.append(tokens)
            all_start_token_idx_to_text_idx.append(start_token_idx_to_text_idx)
            all_end_token_idx_to_text_idx.append(end_token_idx_to_text_idx)

        return all_tokens, all_start_token_idx_to_text_idx, all_end_token_idx_to_text_idx

    @staticmethod
    def _pdf_tokens_to_text_and_maps(tokens: List[str]) -> Tuple[str, List[int], List[int]]:
        text_parts = []
        start_map = []
        end_map = []
        offset = 0
        for idx, token in enumerate(tokens):
            token = str(token)
            if idx > 0:
                text_parts.append(" ")
                offset += 1
            start_map.append(offset)
            text_parts.append(token)
            offset += len(token)
            end_map.append(offset)
        return "".join(text_parts), start_map, end_map

    @staticmethod
    def _filter_valid_texts(texts: List[str]) -> Tuple[List[str], List[int]]:
        """Drop empty text inputs while preserving their original indices."""
        valid_texts = []
        valid_to_orig_idx = []
        for idx, text in enumerate(texts):
            if isinstance(text, str) and text.strip():
                valid_texts.append(text)
                valid_to_orig_idx.append(idx)
        return valid_texts, valid_to_orig_idx

    @staticmethod
    def _normalize_media_inputs(inputs: Any, tensor_batch_rank: Optional[int] = None) -> Tuple[List[Any], bool]:
        if isinstance(inputs, (str, Path)):
            return [inputs], True
        if isinstance(inputs, torch.Tensor):
            if tensor_batch_rank is not None and inputs.dim() == tensor_batch_rank:
                return [inputs[i] for i in range(inputs.shape[0])], False
            return [inputs], True
        if isinstance(inputs, (list, tuple)):
            return list(inputs), False
        return [inputs], True

    @staticmethod
    def _normalize_media_label_groups(
        labels: Union[List[str], Dict[str, List[str]]],
    ) -> List[Dict[str, Any]]:
        if isinstance(labels, dict):
            return [
                {
                    "name": name,
                    "all_labels": list(dict.fromkeys(group_labels)),
                    "true_labels": [],
                }
                for name, group_labels in labels.items()
            ]
        if isinstance(labels, list):
            return [{"name": None, "all_labels": list(dict.fromkeys(labels)), "true_labels": []}]
        raise TypeError(f"Expected labels to be a list or dict, got {type(labels)}.")

    @staticmethod
    def _media_item(value: Any, path_key: str, tensor_key: str) -> Dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, (str, Path)):
            return {path_key: value}
        return {tensor_key: value}

    def _require_task_heads(self, *task_names: str):
        missing = [
            task_name for task_name in task_names
            if getattr(self.config, f"{task_name}_config", None) is None
        ]
        if missing:
            configured = [
                name for name in (
                    "ner", "classification", "joint_relex", "open_relex",
                    "structuring", "set_structuring", "image_classification", "object_detection",
                    "segmentation", "audio_classification", "audio_segmentation",
                    "count", "embedding",
                )
                if getattr(self.config, f"{name}_config", None) is not None
            ]
            raise ValueError(
                "Requested task head is not enabled in config: "
                f"{', '.join(missing)}. Enabled heads: {configured or 'none'}."
            )

    def _validate_requested_inference_heads(
        self,
        entities=None,
        classes=None,
        relations=None,
        joint_relations=None,
        structures=None,
    ):
        if entities is not None:
            self._require_task_heads("ner")
        if classes is not None:
            self._require_task_heads("classification")
        if relations is not None:
            self._require_task_heads("open_relex")
        if joint_relations is not None:
            self._require_task_heads("joint_relex")
        if structures is not None:
            if (
                self.config.structuring_config is None
                and self.config.set_structuring_config is None
            ):
                self._require_task_heads("structuring")

    @torch.no_grad()
    def inference(
        self,
        texts: Union[str, List[str]],
        entities: Optional[Union[List[str], Dict[str, List[str]]]] = None,
        classes: Optional[Union[List[str], Dict[str, List[str]]]] = None,
        relations: Optional[Union[List[str], Dict[str, List[str]]]] = None,
        joint_relations: Optional[Dict[str, dict]] = None,
        structures: Optional[
            Union[Dict[str, Union[List[str], dict]], List[dict]]
        ] = None,
        flat_ner: bool = True,
        threshold: float = 0.5,
        multi_label: bool = False,
        batch_size: int = 8,
        manual_structuring_count: Optional[int] = None,
        structuring_dedup: bool = True,
        objectness_threshold: Optional[float] = None,
        preserve_empty_records: bool = False,
        return_anchor_diagnostics: bool = False,
        **kwargs,
    ) -> Dict[str, List]:
        """Run multi-task inference.

        Always returns a dict keyed by task name. Only tasks with provided
        labels are included in the output.

        Args:
            texts: Input text(s).
            entities: NER entity types. ``List[str]`` for single group,
                ``Dict[str, List[str]]`` for named groups.
            classes: Classification labels. Same format as entities.
            relations: Open relation types. Same format as entities.
            joint_relations: Joint NER + relex groups.
                ``{parent_name: {"entities": [...], "relations": [...]}}``.
            structures: Structuring schemas. Flat schemas use
                ``{schema_name: [field1, ...]}``; multi-level schemas may use
                nested dictionaries/lists. A list exemplar requests a raw
                list root; wrap an object exemplar as ``{"$root": {...}}``
                to request a raw object root rather than named schema groups.
            flat_ner: Enforce non-overlapping spans.
            threshold: Confidence threshold.
            objectness_threshold: Optional structuring-anchor objectness
                threshold. When omitted, structuring reuses ``threshold``.
            preserve_empty_records: Keep objectness-selected records that have
                no extracted field evidence. Disabled by default to avoid
                emitting all-null records from unused set-prediction slots.
            return_anchor_diagnostics: Include per-text diagnostics for
                activated physical anchors, raw learned relations, and the
                final hierarchy connections selected by the multi-level
                decoder. Disabled by default.
            multi_label: Allow multiple labels per span.
            batch_size: Batch size for processing.

        Returns:
            Dict mapping task name to per-text predictions::

                {
                    "ner": List[List[dict]],                    # per text, list of entities
                    "classification": List[List[dict]],         # per text, list of labels
                    "open_relex": List[List[dict]],             # per text, list of triples
                    "structuring": List[Union[dict, list]],     # per text; preserves root shape
                }
        """
        self.eval()
        self._validate_requested_inference_heads(
            entities=entities,
            classes=classes,
            relations=relations,
            joint_relations=joint_relations,
            structures=structures,
        )

        # Normalize input
        if isinstance(texts, str):
            texts = [texts]
        num_original = len(texts)

        # Filter empty texts
        valid_texts, valid_to_orig_idx = self._filter_valid_texts(texts)
        if not valid_texts:
            empty_results = self.data_processor.empty_inference_results(
                num_original,
                entities=entities,
                classes=classes,
                relations=relations,
                joint_relations=joint_relations,
                structures=structures,
            )
            if return_anchor_diagnostics:
                for task_name in ("structuring", "set_structuring"):
                    if task_name not in empty_results:
                        continue
                    empty_results[
                        f"{task_name}_anchor_diagnostics"
                    ] = [
                        {
                            "summary": {
                                "schema_group_count": 0,
                                "activated_anchor_count": 0,
                                "logical_anchor_count": 0,
                                "selected_connection_count": 0,
                                "raw_relation_connection_count": 0,
                            },
                            "groups": [],
                        }
                        for _ in range(num_original)
                    ]
            return empty_results

        kwargs = self._select_valid_forward_kwargs(kwargs, valid_to_orig_idx, num_original)

        # Tokenize
        all_tokens, all_start_maps, all_end_maps = self.prepare_inputs(valid_texts)

        # Build input for collator
        input_x = self._build_inference_input(
            all_tokens, entities, classes, relations, joint_relations, structures,
        )

        # Create collator (inference mode: no labels)
        collator_cls = self.data_collator_class or resolve_glinext_collator_class(self.config)
        collator = collator_cls(
            self.config,
            data_processor=self.data_processor,
            return_tokens=True,
            prepare_labels=False,
        )

        data_loader = DataLoader(
            input_x,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        # Process batches
        if manual_structuring_count is not None:
            kwargs["manual_structuring_count"] = manual_structuring_count
        decoder_kwargs = dict(kwargs.pop("decoder_kwargs", None) or {})
        if objectness_threshold is not None:
            decoder_kwargs["objectness_threshold"] = objectness_threshold
        if preserve_empty_records:
            decoder_kwargs["preserve_empty_records"] = True

        all_decoded, all_classes_mappings = self._process_multitask_batches(
            data_loader, threshold, flat_ner, multi_label,
            decoder_kwargs=decoder_kwargs or None,
            **kwargs,
        )

        # Map results back to original text indices
        return self._map_multitask_results(
            all_decoded,
            valid_to_orig_idx,
            all_start_maps,
            all_end_maps,
            valid_texts,
            num_original,
            all_classes_mappings,
            structures=structures,
            structuring_dedup=structuring_dedup,
            return_anchor_diagnostics=return_anchor_diagnostics,
        )

    @torch.no_grad()
    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        words: Optional[List[str]] = None,
        bbox: Optional[List[List[int]]] = None,
        pixel_values: Optional[Any] = None,
        pages: Union[List[int], Tuple[int, ...], None] = None,
        password: Optional[str] = None,
        add_image_token: bool = True,
        return_pixel_values: Optional[bool] = None,
        return_word_bboxes: bool = True,
        return_page_ids: bool = False,
        split_pages: bool = True,
        extract_tables: bool = False,
        table_settings: Optional[Dict[str, Any]] = None,
        entities: Optional[Union[List[str], Dict[str, List[str]]]] = None,
        classes: Optional[Union[List[str], Dict[str, List[str]]]] = None,
        relations: Optional[Union[List[str], Dict[str, List[str]]]] = None,
        joint_relations: Optional[Dict[str, dict]] = None,
        structures: Optional[
            Union[Dict[str, Union[List[str], dict]], List[dict]]
        ] = None,
        flat_ner: bool = True,
        threshold: float = 0.5,
        multi_label: bool = False,
        batch_size: int = 8,
        manual_structuring_count: Optional[int] = None,
        structuring_dedup: bool = True,
        return_pages: bool = False,
        objectness_threshold: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, List]:
        """Run GLiNExT layout/text inference over PDF pages.

        ``words``/``bbox`` may be supplied to skip PDF text extraction. The
        processor emits ``tokenized_text`` plus optional word ``bboxes`` and
        optional page screenshots as ``pixel_values``. Set ``extract_tables``
        to replace detected table regions with markdown-like table tokens.
        """
        self.eval()
        self._validate_requested_inference_heads(
            entities=entities,
            classes=classes,
            relations=relations,
            joint_relations=joint_relations,
            structures=structures,
        )

        pdf_processor = GLiNextPDFProcessor()
        pdf_items = pdf_processor(
            pdf_path,
            words=words,
            bbox=bbox,
            pixel_values=pixel_values,
            pages=pages,
            password=password,
            add_image_token=add_image_token,
            return_pixel_values=return_pixel_values,
            return_word_bboxes=return_word_bboxes,
            return_page_ids=return_page_ids,
            split_pages=split_pages,
            extract_tables=extract_tables,
            table_settings=table_settings,
        )
        num_pages = len(pdf_items)
        if num_pages == 0:
            results = self.data_processor.empty_inference_results(
                0,
                entities=entities,
                classes=classes,
                relations=relations,
                joint_relations=joint_relations,
                structures=structures,
            )
            if return_pages:
                results["pages"] = []
            return results

        valid_items = []
        valid_to_orig_idx = []
        all_start_maps = []
        all_end_maps = []
        valid_texts = []
        for idx, item in enumerate(pdf_items):
            tokens = list(item.get("tokenized_text") or [])
            if not tokens:
                continue
            text, start_map, end_map = self._pdf_tokens_to_text_and_maps(tokens)
            item["text"] = text
            valid_items.append(item)
            valid_to_orig_idx.append(idx)
            all_start_maps.append(start_map)
            all_end_maps.append(end_map)
            valid_texts.append(text)

        if not valid_items:
            results = self.data_processor.empty_inference_results(
                num_pages,
                entities=entities,
                classes=classes,
                relations=relations,
                joint_relations=joint_relations,
                structures=structures,
            )
            if return_pages:
                results["pages"] = [item.get("pages", item.get("page", idx)) for idx, item in enumerate(pdf_items)]
            return results

        input_x = self._build_pdf_inference_input(
            valid_items, entities, classes, relations, joint_relations, structures,
        )

        collator_cls = self.data_collator_class or resolve_glinext_collator_class(self.config)
        collator = collator_cls(
            self.config,
            data_processor=self.data_processor,
            return_tokens=True,
            prepare_labels=False,
        )
        data_loader = DataLoader(
            input_x,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        if manual_structuring_count is not None:
            kwargs["manual_structuring_count"] = manual_structuring_count
        decoder_kwargs = dict(kwargs.pop("decoder_kwargs", None) or {})
        if objectness_threshold is not None:
            decoder_kwargs["objectness_threshold"] = objectness_threshold

        all_decoded, all_classes_mappings = self._process_multitask_batches(
            data_loader,
            threshold,
            flat_ner,
            multi_label,
            decoder_kwargs=decoder_kwargs or None,
            **kwargs,
        )
        results = self._map_multitask_results(
            all_decoded,
            valid_to_orig_idx,
            all_start_maps,
            all_end_maps,
            valid_texts,
            num_pages,
            all_classes_mappings,
            structures=structures,
            structuring_dedup=structuring_dedup,
        )
        if return_pages:
            results["pages"] = [item.get("pages", item.get("page", idx)) for idx, item in enumerate(pdf_items)]
        return results

    @staticmethod
    def _infer_batch_size(batch: Dict[str, Any], model_batch: Dict[str, Any]) -> int:
        tokens = batch.get("tokens")
        if tokens is not None:
            return len(tokens)
        for key in ("input_ids", "pixel_values", "audio_values"):
            value = model_batch.get(key)
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
        return 0

    @staticmethod
    def _select_valid_forward_kwargs(
        kwargs: Dict[str, Any],
        valid_to_orig_idx: List[int],
        num_original: int,
    ) -> Dict[str, Any]:
        result = dict(kwargs)
        for key in (
            "pixel_values", "vision_attention_mask", "vision_input_mask",
            "audio_values", "audio_attention_mask", "audio_input_mask", "bbox",
            "page_token_ids",
        ):
            value = result.get(key)
            if isinstance(value, torch.Tensor) and value.shape[0] == num_original:
                index = torch.tensor(valid_to_orig_idx, dtype=torch.long, device=value.device)
                result[key] = value.index_select(0, index)
        return result

    @staticmethod
    def _batch_forward_kwargs(kwargs: Dict[str, Any], offset: int, batch_size: int, total: int) -> Dict[str, Any]:
        result = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.shape[0] == total:
                result[key] = value[offset:offset + batch_size]
            else:
                result[key] = value
        return result

    def _process_multitask_batches(
        self,
        data_loader: DataLoader,
        threshold: float,
        flat_ner: bool,
        multi_label: Optional[bool],
        decoder_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[Dict[str, list], list]:
        """Run model forward + decode for each batch, accumulating per-task results.

        Returns:
            Tuple of (accumulated decoded results, list of per-batch-item
            classes_mappings for schema name resolution).
        """
        device = self.device
        model_dtype = self._model_floating_dtype()
        accumulated: Dict[str, list] = {}
        all_classes_mappings: list = []
        total_items = len(data_loader.dataset) if hasattr(data_loader, "dataset") else 0
        offset = 0

        for batch in data_loader:
            # Move tensors to device
            model_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    v = v.to(device)
                    if v.is_floating_point() and model_dtype is not None:
                        v = v.to(dtype=model_dtype)
                    model_batch[k] = v
                else:
                    # classes_mapping, tokens, etc. — pass through
                    model_batch[k] = v

            batch_size = self._infer_batch_size(batch, model_batch)
            batch_kwargs = self._batch_forward_kwargs(kwargs, offset, batch_size, total_items)

            # Forward — kwargs (e.g. manual_structuring_count, multimodal tensors) flow through.
            model_output = self.model(**model_batch, threshold=threshold, **batch_kwargs)
            offset += batch_size

            # Decode
            classes_mapping = batch.get("classes_mapping")
            tokens = batch.get("tokens")

            decoded = self.decoder.decode(
                model_output,
                classes_mapping=classes_mapping,
                threshold=threshold,
                flat_ner=flat_ner,
                multi_label=multi_label,
                texts=tokens,
                **(decoder_kwargs or {}),
            )

            # Accumulate per-item classes_mapping for schema name resolution
            if classes_mapping is not None:
                for i in range(batch_size):
                    all_classes_mappings.append((classes_mapping, i))

            # Accumulate decoded results
            for task_name, task_results in decoded.items():
                if task_name not in accumulated:
                    accumulated[task_name] = []
                accumulated[task_name].extend(task_results)

        return accumulated, all_classes_mappings

    def _model_floating_dtype(self) -> Optional[torch.dtype]:
        for parameter in self.model.parameters():
            if parameter.is_floating_point():
                return parameter.dtype
        return None

    @torch.no_grad()
    def _predict_media_task(
        self,
        inputs: Any,
        labels: Union[List[str], Dict[str, List[str]]],
        task_name: str,
        path_key: str,
        tensor_key: str,
        tensor_batch_rank: Optional[int],
        threshold: float = 0.5,
        multi_label: Optional[bool] = True,
        batch_size: int = 8,
        decoder_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self._require_task_heads(task_name)
        self.eval()

        values, single = self._normalize_media_inputs(inputs, tensor_batch_rank=tensor_batch_rank)
        label_groups = self._normalize_media_label_groups(labels)

        input_x = []
        for value in values:
            item = self._media_item(value, path_key=path_key, tensor_key=tensor_key)
            item[task_name] = [dict(group) for group in label_groups]
            input_x.append(item)

        collator_cls = self.data_collator_class or resolve_glinext_collator_class(self.config)
        collator = collator_cls(
            self.config,
            data_processor=self.data_processor,
            return_tokens=False,
            prepare_labels=False,
        )

        data_loader = DataLoader(
            input_x,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        decoded, _ = self._process_multitask_batches(
            data_loader,
            threshold=threshold,
            flat_ner=True,
            multi_label=multi_label,
            decoder_kwargs=decoder_kwargs,
            **kwargs,
        )
        results = self.decoder.map_results(
            decoded,
            valid_to_orig_idx=list(range(len(values))),
            all_start_maps=[[] for _ in values],
            all_end_maps=[[] for _ in values],
            valid_texts=["" for _ in values],
            num_original=len(values),
        )
        task_results = results.get(task_name, [[] for _ in values])
        if len(label_groups) == 1:
            task_results = self._collapse_single_group_results(task_results)
        return self._single_or_batch(task_results, single)

    def _map_multitask_results(
        self,
        decoded: Dict[str, list],
        valid_to_orig_idx: List[int],
        all_start_maps: List[List[int]],
        all_end_maps: List[List[int]],
        valid_texts: List[str],
        num_original: int,
        all_classes_mappings: Optional[list] = None,
        structures: Optional[
            Union[Dict[str, Union[List[str], dict]], List[dict]]
        ] = None,
        structuring_dedup: bool = True,
        return_anchor_diagnostics: bool = False,
    ) -> Dict[str, List]:
        """Map decoded results back to original text indices and char positions."""
        return self.decoder.map_results(
            decoded,
            valid_to_orig_idx=valid_to_orig_idx,
            all_start_maps=all_start_maps,
            all_end_maps=all_end_maps,
            valid_texts=valid_texts,
            num_original=num_original,
            all_classes_mappings=all_classes_mappings,
            structures=structures,
            structuring_dedup=structuring_dedup,
            return_anchor_diagnostics=return_anchor_diagnostics,
        )

    # ── Per-task convenience methods ───────────────────────────────────

    def predict_entities(
        self,
        texts: Union[str, List[str]],
        entities: Union[List[str], Dict[str, List[str]]],
        flat_ner: bool = True,
        threshold: float = 0.5,
        multi_label: bool = False,
        **kwargs,
    ) -> Union[List[Dict], List[List[Dict]]]:
        """Predict entities for one text or a batch of texts.

        Returns:
            Entity dicts for a single input, or a per-input list for a batch.
        """
        text_batch, single = self._normalize_texts(texts)
        results = self.inference(
            text_batch, entities=entities, flat_ner=flat_ner,
            threshold=threshold, multi_label=multi_label, **kwargs,
        )
        return self._single_or_batch(results.get("ner", [[] for _ in text_batch]), single)

    def classify(
        self,
        texts: Union[str, List[str]],
        classes: Union[List[str], Dict[str, List[str]]],
        threshold: float = 0.5,
        **kwargs,
    ) -> Union[List[Dict], List[List[Dict]]]:
        """Classify one text or a batch of texts.

        Returns:
            Label dicts for a single input, or a per-input list for a batch.
        """
        text_batch, single = self._normalize_texts(texts)
        results = self.inference(
            text_batch, classes=classes, threshold=threshold, **kwargs,
        )
        task_results = results.get("classification", [[] for _ in text_batch])
        if self._label_group_count(classes) == 1:
            task_results = self._collapse_single_group_results(task_results)
        return self._single_or_batch(task_results, single)

    def predict_relations(
        self,
        texts: Union[str, List[str]],
        relations: Union[List[str], Dict[str, List[str]]],
        threshold: float = 0.5,
        flat_ner: bool = True,
        **kwargs,
    ) -> Union[List[Dict], List[List[Dict]]]:
        """Extract relations from one text or a batch of texts.

        Returns:
            Relation dicts for a single input, or a per-input list for a batch.
        """
        text_batch, single = self._normalize_texts(texts)
        results = self.inference(
            text_batch, relations=relations, threshold=threshold,
            flat_ner=flat_ner, **kwargs,
        )
        task_results = results.get("open_relex", [[] for _ in text_batch])
        if self._label_group_count(relations) == 1:
            task_results = self._collapse_single_group_results(task_results)
        return self._single_or_batch(task_results, single)

    def structure(
        self,
        texts: Union[str, List[str]],
        structures: Union[Dict[str, Union[List[str], dict]], List[dict]],
        threshold: float = 0.5,
        flat_ner: bool = True,
        objectness_threshold: Optional[float] = None,
        preserve_empty_records: bool = False,
        return_anchor_diagnostics: bool = False,
        **kwargs,
    ) -> Union[dict, list, List[Union[dict, list]]]:
        """Extract structured data from one text or a batch of texts.

        Returns:
            A schema result dict for a single input, or one result per input.
            Multi-level root object/list schemas are returned in their original
            root shape. Flat schemas retain the historical form::

                {"person": [{"name": "John", "age": "30"}, ...]}

            When ``return_anchor_diagnostics`` is true, returns
            ``(structured_result, diagnostics)``. Diagnostics distinguish
            physical model slots from decoder-created logical anchors and
            report the final parent-child connections used to build the JSON.
        """
        text_batch, single = self._normalize_texts(texts)
        results = self.inference(
            text_batch, structures=structures, threshold=threshold,
            objectness_threshold=objectness_threshold,
            preserve_empty_records=preserve_empty_records,
            return_anchor_diagnostics=return_anchor_diagnostics,
            flat_ner=flat_ner,
            **kwargs,
        )
        task_name = "structuring"
        task_results = results.get(task_name)
        if task_results is None:
            task_name = "set_structuring"
            task_results = results.get(
                task_name,
                [{} for _ in text_batch],
            )
        structured_result = self._single_or_batch(task_results, single)
        if not return_anchor_diagnostics:
            return structured_result

        diagnostics = results.get(
            f"{task_name}_anchor_diagnostics",
            [
                {
                    "summary": {
                        "schema_group_count": 0,
                        "activated_anchor_count": 0,
                        "logical_anchor_count": 0,
                        "selected_connection_count": 0,
                        "raw_relation_connection_count": 0,
                    },
                    "groups": [],
                }
                for _ in text_batch
            ],
        )
        return structured_result, self._single_or_batch(diagnostics, single)

    def classify_images(
        self,
        images: Any,
        classes: Union[List[str], Dict[str, List[str]]],
        threshold: float = 0.5,
        multi_label: bool = True,
        batch_size: int = 8,
        **kwargs,
    ):
        """Classify one image or a batch of images."""
        return self._predict_media_task(
            images,
            labels=classes,
            task_name="image_classification",
            path_key="image",
            tensor_key="pixel_values",
            tensor_batch_rank=4,
            threshold=threshold,
            multi_label=multi_label,
            batch_size=batch_size,
            **kwargs,
        )

    def detect_objects(
        self,
        images: Any,
        classes: Union[List[str], Dict[str, List[str]]],
        threshold: float = 0.5,
        multi_label: Optional[bool] = None,
        batch_size: int = 8,
        **kwargs,
    ):
        """Detect objects in one image or a batch of images."""
        return self._predict_media_task(
            images,
            labels=classes,
            task_name="object_detection",
            path_key="image",
            tensor_key="pixel_values",
            tensor_batch_rank=4,
            threshold=threshold,
            multi_label=multi_label,
            batch_size=batch_size,
            **kwargs,
        )

    def segment_instances(
        self,
        images: Any,
        classes: Union[List[str], Dict[str, List[str]]],
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        return_masks: bool = False,
        multi_label: Optional[bool] = None,
        batch_size: int = 8,
        **kwargs,
    ):
        """Segment object instances in one image or a batch of images."""
        return self._predict_media_task(
            images,
            labels=classes,
            task_name="segmentation",
            path_key="image",
            tensor_key="pixel_values",
            tensor_batch_rank=4,
            threshold=threshold,
            multi_label=multi_label,
            batch_size=batch_size,
            decoder_kwargs={
                "mask_threshold": mask_threshold,
                "return_masks": return_masks,
            },
            **kwargs,
        )

    def classify_audio(
        self,
        audio: Any,
        classes: Union[List[str], Dict[str, List[str]]],
        threshold: float = 0.5,
        multi_label: bool = True,
        batch_size: int = 8,
        **kwargs,
    ):
        """Classify one audio input or a batch of audio inputs."""
        return self._predict_media_task(
            audio,
            labels=classes,
            task_name="audio_classification",
            path_key="audio",
            tensor_key="audio_values",
            tensor_batch_rank=None,
            threshold=threshold,
            multi_label=multi_label,
            batch_size=batch_size,
            **kwargs,
        )

    def segment_audio(
        self,
        audio: Any,
        classes: Union[List[str], Dict[str, List[str]]],
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        return_masks: bool = False,
        multi_label: Optional[bool] = None,
        batch_size: int = 8,
        **kwargs,
    ):
        """Segment one audio input or a batch of audio inputs."""
        return self._predict_media_task(
            audio,
            labels=classes,
            task_name="audio_segmentation",
            path_key="audio",
            tensor_key="audio_values",
            tensor_batch_rank=None,
            threshold=threshold,
            multi_label=multi_label,
            batch_size=batch_size,
            decoder_kwargs={
                "mask_threshold": mask_threshold,
                "return_masks": return_masks,
            },
            **kwargs,
        )

    @torch.no_grad()
    def embed_text(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 8,
    ) -> torch.Tensor:
        """Compute text embeddings via the shared encoder.

        Uses the same encoding path as training: texts are tokenized directly
        (without task prompts) and encoded through the shared encoder, then
        pooled using the EmbeddingHead's pooling layer (if available) or
        mean pooling as fallback.

        Args:
            texts: Input text(s).
            batch_size: Batch size for processing.

        Returns:
            Tensor of shape (N, D) with pooled text embeddings.
        """
        self._require_task_heads("embedding")
        self.eval()
        if isinstance(texts, str):
            texts = [texts]

        valid_texts, valid_to_orig_idx = self._filter_valid_texts(texts)
        if not valid_texts:
            return torch.zeros(len(texts), self.config.hidden_size)

        # Tokenize texts directly (no prompts) — matching the training path
        # where embedding pair texts are tokenized via transformer_tokenizer
        all_tokens, _, _ = self.prepare_inputs(valid_texts)

        data_loader = DataLoader(
            all_tokens, batch_size=batch_size, shuffle=False,
            collate_fn=lambda batch: self.data_processor.transformer_tokenizer(
                batch,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                padding="longest",
            ),
        )

        device = self.device
        all_embeddings = []

        # Get EmbeddingHead's pooling if available
        embedding_head = self.model.heads["embedding"] if (hasattr(self.model, "heads") and "embedding" in self.model.heads) else None

        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Encode through shared encoder — same as training
            token_embeds = self.model.encode_embedding_tokens(input_ids, attention_mask)

            # Pool using EmbeddingHead's pooling layer — same as training
            if embedding_head is not None:
                projected = embedding_head._project(token_embeds)
                pooled = embedding_head.pooling(projected, attention_mask)
            else:
                # Fallback: mean pooling
                mask_f = attention_mask.unsqueeze(-1).float()
                pooled = (token_embeds * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)

            all_embeddings.append(pooled.cpu())

        valid_embeddings = torch.cat(all_embeddings, dim=0)

        # Map back to original indices
        result = torch.zeros(len(texts), valid_embeddings.shape[-1])
        for valid_i, orig_i in enumerate(valid_to_orig_idx):
            result[orig_i] = valid_embeddings[valid_i]

        return result

    @torch.no_grad()
    def embed_labels(
        self,
        labels: List[str],
        batch_size: int = 8,
    ) -> torch.Tensor:
        """Compute label embeddings (bi-encoder only).

        Args:
            labels: List of label strings.
            batch_size: Batch size for processing.

        Returns:
            Tensor of shape (N, D) with label embeddings.

        Raises:
            NotImplementedError: If the model is not a bi-encoder.
        """
        if self.config.labels_encoder is None:
            raise NotImplementedError(
                "embed_labels requires a bi-encoder model (set labels_encoder in config)."
            )

        self.eval()
        if not hasattr(self.model.token_rep_layer, "encode_labels"):
            raise NotImplementedError("embed_labels requires a labels-capable token_rep_layer.")

        labels_tokenizer = self.data_processor.labels_tokenizer
        data_loader = DataLoader(labels, batch_size=batch_size, collate_fn=lambda x: x)

        device = self.device
        all_embeddings = []

        for batch_labels in data_loader:
            tokenized = labels_tokenizer(
                batch_labels,
                return_tensors="pt",
                truncation=True,
                padding="longest",
            )
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            embeds = self.model.token_rep_layer.encode_labels(input_ids, attention_mask)
            all_embeddings.append(embeds.cpu())

        return torch.cat(all_embeddings, dim=0)

    # ── Schema-based inference ─────────────────────────────────────────

    def create_schema(self) -> GLiNExTSchema:
        """Create a new schema builder for structured inference."""
        return GLiNExTSchema()

    def inference_from_schema(
        self,
        texts: Union[str, List[str]],
        schema: GLiNExTSchema,
        **kwargs,
    ) -> Dict[str, List]:
        """Run inference using a pre-built schema.

        Args:
            texts: Input text(s).
            schema: A :class:`GLiNExTSchema` instance.
            **kwargs: Additional arguments passed to :meth:`inference`.

        Returns:
            Dict of per-task predictions. If the schema has typed structure
            fields, structuring values are automatically converted.
        """
        inference_kwargs = schema.to_inference_kwargs()
        inference_kwargs.update(kwargs)
        results = self.inference(texts, **inference_kwargs)

        # Apply output formatting for typed structuring schemas
        formatter = schema.build_output_formatter()
        if formatter is not None and "structuring" in results:
            results["structuring"] = formatter.format_batch(results["structuring"])
        if formatter is not None and "set_structuring" in results:
            results["set_structuring"] = formatter.format_batch(
                results["set_structuring"]
            )

        return results

    # ── Training ─────────────────────────────────────────────────────────

    def _get_freezable_components(self):
        """Get dict mapping component names to their modules for freezing."""
        components = {}

        # Shared text encoder
        if (hasattr(self, "model")
                and hasattr(self.model, "token_rep_layer")
                and hasattr(self.model.token_rep_layer, "bert_layer")):
            components["text_encoder"] = self.model.token_rep_layer.bert_layer.model
        elif (hasattr(self, "model")
              and hasattr(self.model, "token_rep_layer")
              and hasattr(self.model.token_rep_layer, "text_encoder")
              and hasattr(self.model.token_rep_layer.text_encoder, "bert_layer")):
            components["text_encoder"] = self.model.token_rep_layer.text_encoder.bert_layer.model

        # Labels encoder (bi-encoder)
        if (self.config.labels_encoder is not None
                and hasattr(self.model, "token_rep_layer")
                and hasattr(self.model.token_rep_layer, "labels_encoder")):
            components["labels_encoder"] = self.model.token_rep_layer.labels_encoder.model
        elif (self.config.labels_encoder is not None
              and hasattr(self.model, "token_rep_layer")
              and hasattr(self.model.token_rep_layer, "text_encoder")
              and hasattr(self.model.token_rep_layer.text_encoder, "labels_encoder")):
            components["labels_encoder"] = self.model.token_rep_layer.text_encoder.labels_encoder.model

        if hasattr(self.model, "vision_encoder"):
            components["vision_encoder"] = self.model.vision_encoder
        if hasattr(self.model, "audio_encoder"):
            components["audio_encoder"] = self.model.audio_encoder
        if hasattr(self.model, "vision_fusion"):
            components["vision_fusion"] = self.model.vision_fusion
        if hasattr(self.model, "audio_fusion"):
            components["audio_fusion"] = self.model.audio_fusion
        if (hasattr(self.model, "token_rep_layer")
                and hasattr(self.model.token_rep_layer, "feature_encoders")):
            for name, module in self.model.token_rep_layer.feature_encoders.items():
                components[f"{name}_encoder"] = module

        # Individual task heads
        if hasattr(self.model, "heads"):
            for head_name, head_module in self.model.heads.items():
                components[head_name] = head_module

        # RNN layer
        if hasattr(self.model, "rnn"):
            components["rnn"] = self.model.rnn

        # Other shared representation/joint layers
        if hasattr(self.model, "cross_fuser"):
            components["cross_fuser"] = self.model.cross_fuser
        if hasattr(self.model, "shared_anchor_modeling"):
            components["shared_anchor_modeling"] = self.model.shared_anchor_modeling
        if hasattr(self.model, "shared_anchor_refine"):
            components["shared_anchor_refine"] = self.model.shared_anchor_refine

        return components

    def train_head_only_parameters(self) -> Dict[str, int]:
        """Freeze shared model parameters and leave only head-owned parameters trainable.

        Shared modules that are referenced by heads, such as shared anchor
        modeling/refinement layers, stay frozen because they are not
        head-specific even though they appear under task head modules.
        """
        if not hasattr(self, "model") or not hasattr(self.model, "heads"):
            raise ValueError("Head-only training requires an initialized GLiNExT model with task heads.")

        self.model.requires_grad_(False)

        shared_param_ids = set()
        for module_name in ("shared_anchor_modeling", "shared_anchor_refine"):
            module = getattr(self.model, module_name, None)
            if module is not None:
                shared_param_ids.update(id(param) for param in module.parameters())

        trainable_param_ids = set()
        frozen_shared_param_ids = set()
        for head in self.model.heads.values():
            for param in head.parameters():
                if id(param) in shared_param_ids:
                    frozen_shared_param_ids.add(id(param))
                    continue
                param.requires_grad_(True)
                trainable_param_ids.add(id(param))

        params_by_id = {id(param): param for param in self.model.parameters()}
        trainable_params = sum(params_by_id[param_id].numel() for param_id in trainable_param_ids)
        frozen_shared_params = sum(
            params_by_id[param_id].numel()
            for param_id in frozen_shared_param_ids
            if param_id in params_by_id
        )
        frozen_params = sum(
            param.numel() for param in self.model.parameters() if not param.requires_grad
        )
        return {
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "frozen_shared_head_params": frozen_shared_params,
        }

    def train_model(
        self,
        train_dataset,
        eval_dataset=None,
        training_args=None,
        freeze_components: Optional[List[str]] = None,
        train_head_only: bool = False,
        compile_model: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        **training_kwargs,
    ):
        """Train the GLiNExT model.

        Uses the HuggingFace Trainer with a custom subclass that handles
        multi-task label keys and differential learning rates.

        Args:
            train_dataset: Training dataset — list of dicts with task-specific
                annotations (``extraction``, ``classification``, ``open_relex``,
                ``structuring``, ``embedding`` keys).
            eval_dataset: Optional evaluation dataset (same format).
            training_args: ``TrainingArguments`` instance from
                ``gliner.training.trainer``. Created from ``training_kwargs``
                if not provided.
            freeze_components: Components to freeze during training. Options:
                ``"text_encoder"``, ``"labels_encoder"``, ``"rnn"``, or any
                task head name (``"ner"``, ``"classification"``, etc.).
            train_head_only: Freeze encoders and shared/joint layers, training
                only parameters owned by task heads.
            compile_model: Whether to compile the model with ``torch.compile``.
            output_dir: Output directory for checkpoints (required if
                ``training_args`` is None).
            **training_kwargs: Passed to ``create_training_args()`` when
                ``training_args`` is not provided.

        Returns:
            Trainer instance with training state.

        Example::

            model = GLiNExT.load_from_config(config, output_dir="./output")

            train_data = [
                {
                    "text": "Apple is a company",
                    "extraction": [{"name": None, "ner": [["Apple", "company"]]}],
                    "classification": [{"all_labels": ["tech"], "true_labels": ["tech"]}],
                },
                ...
            ]

            trainer = model.train_model(
                train_data,
                output_dir="./output",
                max_steps=1000,
                learning_rate=5e-5,
                others_lr=1e-4,
            )
        """
        from .training import GLiNExTTrainer

        if training_args is None:
            if output_dir is None:
                raise ValueError("Either training_args or output_dir must be provided")
            training_args = self.create_training_args(output_dir=output_dir, **training_kwargs)

        if compile_model:
            self.compile()

        if train_head_only:
            freeze_stats = self.train_head_only_parameters()
            logger.info(
                "Head-only training enabled: %s trainable params, %s frozen params",
                freeze_stats["trainable_params"],
                freeze_stats["frozen_params"],
            )

        if freeze_components:
            for component_name in freeze_components:
                self.freeze_component(component_name)

        data_collator = self._create_data_collator()

        # Build trainer kwargs from the installed Trainer API. Development
        # builds such as ``5.0.0.dev0`` compare lower than the final 5.0.0
        # release even though they already use ``processing_class``.
        trainer_kwargs = {
            "model": self,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "data_collator": data_collator,
        }
        trainer_parameters = inspect.signature(
            GLiNExTTrainer.__mro__[1].__init__
        ).parameters
        if "processing_class" in trainer_parameters:
            trainer_kwargs["processing_class"] = self.data_processor.transformer_tokenizer
        else:
            trainer_kwargs["tokenizer"] = self.data_processor.transformer_tokenizer

        trainer = GLiNExTTrainer(**trainer_kwargs)
        trainer.train()

        return trainer

    # ── Evaluation ─────────────────────────────────────────────────────

    def evaluate(self, test_data, flat_ner=False, multi_label=False,
                 threshold=0.5, batch_size=12, **kwargs):
        """Evaluate the model on test data.

        Args:
            test_data: List of annotated examples.
            flat_ner: Use flat NER evaluation.
            multi_label: Allow multi-label evaluation.
            threshold: Detection threshold.
            batch_size: Batch size.

        Returns:
            Tuple of (evaluation_output, f1_score).
        """
        self.eval()
        # TODO: Implement multi-task evaluation
        raise NotImplementedError(
            "Multi-task evaluation is not yet implemented. "
            "Use task-specific evaluation pipelines."
        )


# Backward-compatible spelling used by existing imports/tests.
BaseGLiNeXT = BaseGLiNExT


class GLiNExTText(BaseGLiNExT):
    """User-facing text GLiNExT wrapper."""

    config_class = GLiNextTextConfig
    model_class = GLiNExTTextModel
    data_collator_class = GLiNExTTextDataCollator


class GLiNExTVision(BaseGLiNExT):
    """User-facing vision GLiNExT wrapper."""

    config_class = GLiNextVisionConfig
    model_class = GLiNExTVisionModel
    data_collator_class = GLiNExTVisionDataCollator


class GLiNExTAudio(BaseGLiNExT):
    """User-facing audio GLiNExT wrapper."""

    config_class = GLiNextAudioConfig
    model_class = GLiNExTAudioModel
    data_collator_class = GLiNExTAudioDataCollator


class GLiNExTLayout(BaseGLiNExT):
    """User-facing document-layout GLiNExT wrapper."""

    config_class = GLiNextLayoutConfig
    model_class = GLiNExTLayoutModel
    data_collator_class = GLiNExTLayoutDataCollator


class GLiNExTOmni(BaseGLiNExT):
    """User-facing omni-modal GLiNExT wrapper."""

    config_class = GLiNextOmniConfig
    model_class = GLiNExTOmniModel
    data_collator_class = GLiNExTOmniDataCollator


_GLINEXT_MODEL_TO_WRAPPER = {
    GLiNExTTextModel: GLiNExTText,
    GLiNExTModel: GLiNExTText,
    GLiNExTVisionModel: GLiNExTVision,
    GLiNExTAudioModel: GLiNExTAudio,
    GLiNExTLayoutModel: GLiNExTLayout,
    GLiNExTOmniModel: GLiNExTOmni,
}

class GLiNExT(nn.Module, PyTorchModelHubMixin):
    """Factory class that instantiates the appropriate GLiNExT wrapper.

    Mirrors :class:`gliner.model.GLiNER`: the factory reads the config,
    resolves the concrete GLiNExT type, delegates construction/loading to that
    subclass, then replaces itself with the concrete instance.
    """

    def __init__(self, config: Union[str, Path, GLiNextConfig, dict], **kwargs):
        super().__init__()
        config = self._coerce_config(config)
        glinext_class = self._get_glinext_class(config)
        new_instance = glinext_class(config, **kwargs)
        self.__class__ = type(new_instance)
        self.__dict__ = new_instance.__dict__

    def train_head_only_parameters(self) -> Dict[str, int]:
        """Freeze shared parameters and train only task-head-owned parameters.

        Construction normally turns this factory into a concrete GLiNExT
        wrapper. Keeping the method on the public factory as well preserves the
        API for callers that invoke helpers through :class:`GLiNExT` directly.
        """

        return BaseGLiNExT.train_head_only_parameters(self)

    @staticmethod
    def _config_class_for_variant(config_dict: Dict[str, Any]):
        variant = config_dict.get("model_variant") or "text"
        try:
            return GLINEXT_VARIANT_TO_CONFIG_CLASS[variant]
        except KeyError as exc:
            raise ValueError(
                "model_variant must be one of "
                f"{sorted(GLINEXT_VARIANT_TO_CONFIG_CLASS)}, got {variant!r}."
            ) from exc

    @classmethod
    def _coerce_config_dict(cls, config_dict: Dict[str, Any]) -> GLiNextConfig:
        config_dict = dict(config_dict)
        config_cls = resolve_glinext_config_class(config_dict)
        model_type = config_dict.pop("model_type", None)
        if model_type in GLINEXT_MODEL_TYPE_TO_VARIANT:
            config_dict["model_variant"] = GLINEXT_MODEL_TYPE_TO_VARIANT[model_type]
        return config_cls(**config_dict)

    @classmethod
    def _coerce_config(cls, config: Union[str, Path, GLiNextConfig, dict]) -> GLiNextConfig:
        if isinstance(config, (str, Path)):
            config_path = Path(config)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config}")
            with open(config_path) as f:
                config_dict = json.load(f)
            return cls._coerce_config_dict(config_dict)
        if isinstance(config, dict):
            return cls._coerce_config_dict(config)
        if isinstance(config, GLiNextConfig):
            config_cls = cls._config_class_for_variant(config.to_dict())
            if isinstance(config, config_cls):
                return config
            return cls._coerce_config_dict(config.to_dict())
        raise TypeError(f"config must be a GLiNextConfig object, path to config file, or dict. Got {type(config)}")

    @staticmethod
    def _get_glinext_class(config: GLiNextConfig):
        model_cls = resolve_glinext_model_class(config)
        try:
            return _GLINEXT_MODEL_TO_WRAPPER[model_cls]
        except KeyError as exc:
            raise ValueError(f"No GLiNExT wrapper registered for model class {model_cls}") from exc

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        model_dir: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        force_download: bool = False,
        proxies: Optional[dict] = None,
        resume_download: bool = False,
        local_files_only: bool = False,
        token: Union[str, bool, None] = None,
        map_location: str = "cpu",
        strict: bool = False,
        load_tokenizer: Optional[bool] = None,
        resize_token_embeddings: Optional[bool] = True,
        compile_torch_model: Optional[bool] = False,
        quantize: Optional[str] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        load_onnx_model: Optional[bool] = False,
        onnx_model_file: Optional[str] = "model.onnx",
        session_options=None,
        max_length: Optional[int] = None,
        max_width: Optional[int] = None,
        post_fusion_schema: Optional[str] = None,
        _attn_implementation: Optional[str] = None,
        **model_kwargs,
    ):
        if model_dir is None:
            model_dir = BaseGLiNeXT._download_model(
                model_id, revision, cache_dir, force_download, proxies, resume_download, token, local_files_only
            )
        else:
            model_dir = Path(model_dir)

        config_file = model_dir / "gliner_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"No config file found in {model_dir}")

        config = BaseGLiNeXT._load_config(
            config_file,
            max_len=max_length,
            max_width=max_width,
            post_fusion_schema=post_fusion_schema,
            _attn_implementation=_attn_implementation,
        )
        config = cls._coerce_config(config)
        glinext_class = cls._get_glinext_class(config)
        logger.info("Loading the following GLiNExT type: %s...", glinext_class)
        loader_kwargs = BaseGLiNExT._supported_base_loader_kwargs(
            glinext_class.from_pretrained,
            {
                "model_id": model_id,
                "model_dir": model_dir,
                "revision": revision,
                "cache_dir": cache_dir,
                "force_download": force_download,
                "proxies": proxies,
                "resume_download": resume_download,
                "local_files_only": local_files_only,
                "token": token,
                "map_location": map_location,
                "strict": strict,
                "load_tokenizer": load_tokenizer,
                "resize_token_embeddings": resize_token_embeddings,
                "compile_torch_model": compile_torch_model,
                "quantize": quantize,
                "dtype": dtype,
                "load_onnx_model": load_onnx_model,
                "onnx_model_file": onnx_model_file,
                "session_options": session_options,
                "max_length": max_length,
                "max_width": max_width,
                "post_fusion_schema": post_fusion_schema,
                "_attn_implementation": _attn_implementation,
                **model_kwargs,
            },
        )
        return glinext_class.from_pretrained(**loader_kwargs)

    @classmethod
    def load_from_config(
        cls,
        config: Union[str, Path, GLiNextConfig, dict],
        cache_dir: Optional[Union[str, Path]] = None,
        load_tokenizer: bool = True,
        resize_token_embeddings: bool = True,
        backbone_from_pretrained: bool = True,
        compile_torch_model: bool = False,
        quantize: Optional[str] = None,
        map_location: str = "cpu",
        max_length: Optional[int] = None,
        max_width: Optional[int] = None,
        post_fusion_schema: Optional[str] = None,
        _attn_implementation: Optional[str] = None,
        **model_kwargs,
    ):
        config_instance = cls._coerce_config(config)
        glinext_class = cls._get_glinext_class(config_instance)
        loader_kwargs = BaseGLiNExT._supported_base_loader_kwargs(
            glinext_class.load_from_config,
            {
                "config": config_instance,
                "cache_dir": cache_dir,
                "load_tokenizer": load_tokenizer,
                "resize_token_embeddings": resize_token_embeddings,
                "backbone_from_pretrained": backbone_from_pretrained,
                "compile_torch_model": compile_torch_model,
                "quantize": quantize,
                "map_location": map_location,
                "max_length": max_length,
                "max_width": max_width,
                "post_fusion_schema": post_fusion_schema,
                "_attn_implementation": _attn_implementation,
                **model_kwargs,
            },
        )
        return glinext_class.load_from_config(**loader_kwargs)

    @classmethod
    def from_config(cls, *args, **kwargs):
        return cls.load_from_config(*args, **kwargs)
