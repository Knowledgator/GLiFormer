"""GLiNExT — main user-facing class for multi-task information extraction."""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from gliner.model import BaseEncoderGLiNER
from gliner.data_processing.tokenizer import WordsSplitter

from .config import GLiNextConfig
from .model import GLiNExTModel, resolve_glinext_model_class
from .processing.processor import GLiNextProcessor
from .processing.decoder import GLiNExTDecoder
from .processing.collator import GLiNExTDataCollator
from .processing.schema import GLiNExTSchema

logger = logging.getLogger(__name__)


class GLiNExT(BaseEncoderGLiNER):
    """Unified multi-task information extraction model.

    Supports NER, classification, relation extraction (joint & open),
    structuring (JSON schema extraction), embedding, and counting —
    all through a single shared encoder backbone.

    Inherits model loading/saving from :class:`BaseGLiNER` and adds
    multi-task inference with per-task convenience methods.

    Example::

        model = GLiNExT.from_pretrained("urchade/glinext-base")

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
    model_class = GLiNExTModel
    data_processor_class = GLiNextProcessor
    data_collator_class = GLiNExTDataCollator
    decoder_class = GLiNExTDecoder

    # ── Setup overrides ────────────────────────────────────────���───────

    def _create_model(self, config, backbone_from_pretrained, cache_dir, **kwargs):
        model_cls = resolve_glinext_model_class(config)
        self.model = model_cls(config, from_pretrained=backbone_from_pretrained, cache_dir=cache_dir, **kwargs)
        return self.model


    def _create_data_processor(self, config, cache_dir, tokenizer=None, words_splitter=None, **kwargs):
        """Create processor, loading labels tokenizer for bi-encoder mode."""
        labels_tokenizer = None
        if config.labels_encoder is not None:
            labels_tokenizer = AutoTokenizer.from_pretrained(config.labels_encoder, cache_dir=cache_dir)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=cache_dir)
            self._set_tokenizer_spec_tokens(tokenizer)

        if words_splitter is None:
            words_splitter = WordsSplitter(config.words_splitter_type)

        self.data_processor = GLiNextProcessor(
            config, tokenizer, words_splitter, labels_tokenizer=labels_tokenizer,
        )
        return self.data_processor

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

        if self.config.structuring_config is not None:
            tokens.append(self.config.child_token)

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

        if self.config.structuring_config is not None:
            child_idx = _idx(self.config.child_token)
            self.config.structuring_config.child_token_index = child_idx
            self.config.child_token_index = child_idx

        if (self.config.image_classification_config is not None
                or self.config.object_detection_config is not None
                or self.config.segmentation_config is not None
                or self.config.audio_classification_config is not None
                or self.config.audio_segmentation_config is not None):
            obj_idx = _idx(self.config.obj_token)
            self.config.obj_token_index = obj_idx
            for cfg_name in (
                "image_classification_config", "object_detection_config", "segmentation_config",
                "audio_classification_config", "audio_segmentation_config",
            ):
                cfg = getattr(self.config, cfg_name, None)
                if cfg is not None:
                    cfg.obj_token_index = obj_idx

    def resize_embeddings(self, set_class_token_index=True):
        """Resize token embeddings to match tokenizer vocabulary."""
        if set_class_token_index:
            self.set_class_indices()

        tokenizer = self.data_processor.transformer_tokenizer
        if len(tokenizer) != self.config.vocab_size:
            new_num_tokens = len(tokenizer)
            model_embeds = self.model.token_rep_layer.resize_token_embeddings(new_num_tokens, None)
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

    def _require_task_heads(self, *task_names: str):
        missing = [
            task_name for task_name in task_names
            if getattr(self.config, f"{task_name}_config", None) is None
        ]
        if missing:
            configured = [
                name for name in (
                    "ner", "classification", "joint_relex", "open_relex",
                    "structuring", "image_classification", "object_detection",
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
            self._require_task_heads("structuring")

    @torch.no_grad()
    def inference(
        self,
        texts: Union[str, List[str]],
        entities: Optional[Union[List[str], Dict[str, List[str]]]] = None,
        classes: Optional[Union[List[str], Dict[str, List[str]]]] = None,
        relations: Optional[Union[List[str], Dict[str, List[str]]]] = None,
        joint_relations: Optional[Dict[str, dict]] = None,
        structures: Optional[Dict[str, Union[List[str], dict]]] = None,
        flat_ner: bool = True,
        threshold: float = 0.5,
        multi_label: bool = False,
        batch_size: int = 8,
        manual_structuring_count: Optional[int] = None,
        structuring_dedup: bool = True,
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
            structures: Structuring schemas.
                ``{schema_name: [field1, field2, ...]}`` or
                ``{schema_name: {"fields": [...]}}``.
            flat_ner: Enforce non-overlapping spans.
            threshold: Confidence threshold.
            multi_label: Allow multiple labels per span.
            batch_size: Batch size for processing.

        Returns:
            Dict mapping task name to per-text predictions::

                {
                    "ner": List[List[dict]],                    # per text, list of entities
                    "classification": List[List[dict]],         # per text, list of labels
                    "open_relex": List[List[dict]],             # per text, list of triples
                    "structuring": List[Dict[str, List[dict]]], # per text, {schema: [instances]}
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
            return self.data_processor.empty_inference_results(
                num_original,
                entities=entities,
                classes=classes,
                relations=relations,
                joint_relations=joint_relations,
                structures=structures,
            )

        kwargs = self._select_valid_forward_kwargs(kwargs, valid_to_orig_idx, num_original)

        # Tokenize
        all_tokens, all_start_maps, all_end_maps = self.prepare_inputs(valid_texts)

        # Build input for collator
        input_x = self._build_inference_input(
            all_tokens, entities, classes, relations, joint_relations, structures,
        )

        # Create collator (inference mode: no labels)
        collator = GLiNExTDataCollator(
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

        all_decoded, all_classes_mappings = self._process_multitask_batches(
            data_loader, threshold, flat_ner, multi_label,
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
        )

    @staticmethod
    def _select_valid_forward_kwargs(
        kwargs: Dict[str, Any],
        valid_to_orig_idx: List[int],
        num_original: int,
    ) -> Dict[str, Any]:
        result = dict(kwargs)
        for key in ("pixel_values", "vision_attention_mask", "input_values", "audio_attention_mask", "word_bboxes", "bbox"):
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
        multi_label: bool,
        **kwargs,
    ) -> Tuple[Dict[str, list], list]:
        """Run model forward + decode for each batch, accumulating per-task results.

        Returns:
            Tuple of (accumulated decoded results, list of per-batch-item
            classes_mappings for schema name resolution).
        """
        device = self.device
        accumulated: Dict[str, list] = {}
        all_classes_mappings: list = []
        total_items = len(data_loader.dataset) if hasattr(data_loader, "dataset") else 0
        offset = 0

        for batch in data_loader:
            # Move tensors to device
            model_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    model_batch[k] = v.to(device)
                else:
                    # classes_mapping, tokens, etc. — pass through
                    model_batch[k] = v

            batch_size = len(batch.get("tokens") or model_batch.get("input_ids", []))
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
            )

            # Accumulate per-item classes_mapping for schema name resolution
            if classes_mapping is not None:
                batch_size = len(tokens) if tokens else 0
                for i in range(batch_size):
                    all_classes_mappings.append((classes_mapping, i))

            # Accumulate decoded results
            for task_name, task_results in decoded.items():
                if task_name not in accumulated:
                    accumulated[task_name] = []
                accumulated[task_name].extend(task_results)

        return accumulated, all_classes_mappings

    def _map_multitask_results(
        self,
        decoded: Dict[str, list],
        valid_to_orig_idx: List[int],
        all_start_maps: List[List[int]],
        all_end_maps: List[List[int]],
        valid_texts: List[str],
        num_original: int,
        all_classes_mappings: Optional[list] = None,
        structures: Optional[Dict[str, Union[List[str], dict]]] = None,
        structuring_dedup: bool = True,
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
        )

    # ── Per-task convenience methods ───────────────────────────────────

    def predict_entities(
        self,
        text: str,
        entities: Union[List[str], Dict[str, List[str]]],
        flat_ner: bool = True,
        threshold: float = 0.5,
        multi_label: bool = False,
        **kwargs,
    ) -> List[Dict]:
        """Predict entities for a single text.

        Returns:
            List of entity dicts with start, end, text, label, score.
        """
        results = self.inference(
            [text], entities=entities, flat_ner=flat_ner,
            threshold=threshold, multi_label=multi_label, **kwargs,
        )
        return results.get("ner", [[]])[0]

    def batch_predict_entities(
        self,
        texts: List[str],
        entities: Union[List[str], Dict[str, List[str]]],
        flat_ner: bool = True,
        threshold: float = 0.5,
        multi_label: bool = False,
        **kwargs,
    ) -> List[List[Dict]]:
        """Predict entities for multiple texts.

        Returns:
            List of lists of entity dicts.
        """
        results = self.inference(
            texts, entities=entities, flat_ner=flat_ner,
            threshold=threshold, multi_label=multi_label, **kwargs,
        )
        return results.get("ner", [[] for _ in texts])

    def classify(
        self,
        text: str,
        classes: Union[List[str], Dict[str, List[str]]],
        threshold: float = 0.5,
        **kwargs,
    ) -> List[Dict]:
        """Classify a single text.

        Returns:
            List of predicted label dicts with class_id, score.
        """
        results = self.inference(
            [text], classes=classes, threshold=threshold, **kwargs,
        )
        return results.get("classification", [[]])[0]

    def predict_relations(
        self,
        text: str,
        relations: Union[List[str], Dict[str, List[str]]],
        threshold: float = 0.5,
        flat_ner: bool = True,
        **kwargs,
    ) -> List[Dict]:
        """Extract relations from a single text.

        Returns:
            List of relation triple dicts with head, tail, relation, score.
        """
        results = self.inference(
            [text], relations=relations, threshold=threshold,
            flat_ner=flat_ner, **kwargs,
        )
        return results.get("open_relex", [[]])[0]

    def structure(
        self,
        text: str,
        structures: Dict[str, Union[List[str], dict]],
        threshold: float = 0.5,
        flat_ner: bool = True,
        **kwargs,
    ) -> Dict[str, List[Dict]]:
        """Extract structured data from a single text.

        Returns:
            Dict mapping schema names to lists of extracted instances::

                {"person": [{"name": "John", "age": "30"}, ...]}
        """
        results = self.inference(
            [text], structures=structures, threshold=threshold,
            flat_ner=flat_ner, **kwargs,
        )
        return results.get("structuring", [{}])[0]

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
        import transformers
        from packaging import version

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

        # Build trainer kwargs — handle transformers v4 vs v5 API
        trainer_kwargs = {
            "model": self,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "data_collator": data_collator,
        }
        if version.parse(transformers.__version__) < version.parse("5.0.0"):
            trainer_kwargs["tokenizer"] = self.data_processor.transformer_tokenizer
        else:
            trainer_kwargs["processing_class"] = self.data_processor.transformer_tokenizer

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
