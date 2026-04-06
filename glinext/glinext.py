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
from .model import GLiNExTModel
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
        self.model = GLiNExTModel(config, from_pretrained=backbone_from_pretrained, cache_dir=cache_dir, **kwargs)
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
        tokens = [self.config.ent_token, self.config.sep_token,
                  self.config.parent_token]

        if self.config.classification_config is not None:
            tokens.append(self.config.cat_token)

        if (self.config.joint_relex_config is not None
                or self.config.open_relex_config is not None):
            tokens.append(self.config.rel_token)

        if self.config.structuring_config is not None:
            tokens.append(self.config.child_token)

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

        # Per-task token indices
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

    @staticmethod
    def _normalize_label_groups(labels) -> Dict[str, List[str]]:
        """Normalize ``List[str]`` or ``Dict[str, List[str]]`` to dict form.

        A plain list becomes a single unnamed group ``{None: labels}``.
        """
        if labels is None:
            return {}
        if isinstance(labels, list):
            return {None: list(dict.fromkeys(labels))}
        if isinstance(labels, dict):
            return {k: list(dict.fromkeys(v)) for k, v in labels.items()}
        raise TypeError(f"Expected list or dict for labels, got {type(labels)}")

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
        entity_groups = self._normalize_label_groups(entities)
        class_groups = self._normalize_label_groups(classes)
        relation_groups = self._normalize_label_groups(relations)

        input_x = []
        for tokens in all_tokens:
            item: Dict[str, Any] = {"tokenized_text": tokens}

            # NER / Joint Relex
            if entity_groups or joint_relations:
                extraction = []
                for parent_name, ent_labels in entity_groups.items():
                    entry = {
                        "name": parent_name,
                        "ner": [],
                    }
                    # If this is a joint group, also add relations
                    if joint_relations and parent_name in joint_relations:
                        entry["relations"] = []
                        entry["all_rel_labels"] = joint_relations[parent_name].get("relations", [])
                    entry["all_labels"] = ent_labels
                    extraction.append(entry)

                # Joint-only groups (no NER labels)
                if joint_relations:
                    for parent_name, jconf in joint_relations.items():
                        if parent_name not in entity_groups:
                            extraction.append({
                                "name": parent_name,
                                "ner": [],
                                "relations": [],
                                "all_labels": jconf.get("entities", []),
                                "all_rel_labels": jconf.get("relations", []),
                            })

                item["extraction"] = extraction

            # Classification
            if class_groups:
                classification = []
                for parent_name, cls_labels in class_groups.items():
                    classification.append({
                        "name": parent_name,
                        "all_labels": cls_labels,
                        "true_labels": [],
                    })
                item["classification"] = classification

            # Open Relation Extraction
            if relation_groups:
                open_relex = []
                for parent_name, rel_labels in relation_groups.items():
                    open_relex.append({
                        "name": parent_name,
                        "relations": [],
                        "all_labels": rel_labels,
                    })
                item["open_relex"] = open_relex

            # Structuring
            if structures:
                structuring = {}
                for schema_name, fields in structures.items():
                    field_list = fields if isinstance(fields, list) else fields.get("fields", [])
                    # Dummy instance with all field names so the processor
                    # can build the class mapping during inference
                    structuring[schema_name] = [dict.fromkeys(field_list, "")] if field_list else []
                item["structuring"] = structuring
                item["structuring_schema"] = {
                    schema_name: fields if isinstance(fields, list) else fields.get("fields", [])
                    for schema_name, fields in structures.items()
                }

            input_x.append(item)
        return input_x

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

        # Normalize input
        if isinstance(texts, str):
            texts = [texts]
        num_original = len(texts)

        # Filter empty texts
        valid_texts, valid_to_orig_idx = self._filter_valid_texts(texts)
        if not valid_texts:
            return self._empty_results(num_original, entities, classes, relations, structures)

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
        all_decoded, all_classes_mappings = self._process_multitask_batches(
            data_loader, threshold, flat_ner, multi_label, **kwargs,
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
        )

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

        for batch in data_loader:
            # Move tensors to device
            model_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    model_batch[k] = v.to(device)
                else:
                    # classes_mapping, tokens, etc. — pass through
                    model_batch[k] = v

            # Forward
            model_output = self.model(**model_batch, threshold=threshold)

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
                    all_classes_mappings.append(classes_mapping)

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
    ) -> Dict[str, List]:
        """Map decoded results back to original text indices and char positions."""
        results = {}

        for task_name, task_results in decoded.items():
            if task_name in ("ner", "joint_relex"):
                results[task_name] = self._map_span_results(
                    task_results, valid_to_orig_idx, all_start_maps,
                    all_end_maps, valid_texts, num_original,
                )
            elif task_name in ("open_relex",):
                results[task_name] = self._map_relex_results(
                    task_results, valid_to_orig_idx, all_start_maps,
                    all_end_maps, valid_texts, num_original,
                )
            elif task_name in ("structuring",):
                results[task_name] = self._map_structuring_results(
                    task_results, valid_to_orig_idx, all_start_maps,
                    all_end_maps, valid_texts, num_original,
                    all_classes_mappings,
                )
            else:
                # classification, embedding, count — no span mapping needed
                results[task_name] = self._map_passthrough_results(
                    task_results, valid_to_orig_idx, num_original,
                )

        return results

    def _map_span_results(
        self,
        task_results: list,
        valid_to_orig_idx: List[int],
        all_start_maps: List[List[int]],
        all_end_maps: List[List[int]],
        valid_texts: List[str],
        num_original: int,
    ) -> List[List[Dict]]:
        """Map NER-style span results (token indices → char positions)."""
        output = [[] for _ in range(num_original)]

        for valid_i, per_text_groups in enumerate(task_results):
            orig_i = valid_to_orig_idx[valid_i]
            start_map = all_start_maps[valid_i]
            end_map = all_end_maps[valid_i]
            text = valid_texts[valid_i]

            entities = []
            # per_text_groups is List[List[Span]] (groups of spans)
            groups = per_text_groups if isinstance(per_text_groups, list) else [per_text_groups]
            for group in groups:
                if not isinstance(group, list):
                    group = [group]
                for span in group:
                    if hasattr(span, 'start') and hasattr(span, 'entity_type'):
                        # Span object
                        if span.start < len(start_map) and span.end < len(end_map):
                            start_char = start_map[span.start]
                            end_char = end_map[span.end]
                            entity = {
                                "start": start_char,
                                "end": end_char,
                                "text": text[start_char:end_char],
                                "label": span.entity_type,
                                "score": span.score,
                            }
                            if span.class_probs is not None:
                                entity["class_probs"] = span.class_probs
                            entities.append(entity)
                    elif isinstance(span, dict):
                        entities.append(span)

            output[orig_i] = entities
        return output

    def _map_relex_results(
        self,
        task_results: list,
        valid_to_orig_idx: List[int],
        all_start_maps: List[List[int]],
        all_end_maps: List[List[int]],
        valid_texts: List[str],
        num_original: int,
    ) -> List[List[Dict]]:
        """Map open relex results (triples with token-level head/tail → char)."""
        output = [[] for _ in range(num_original)]

        for valid_i, per_text_groups in enumerate(task_results):
            orig_i = valid_to_orig_idx[valid_i]
            start_map = all_start_maps[valid_i]
            end_map = all_end_maps[valid_i]
            text = valid_texts[valid_i]

            triples = []
            groups = per_text_groups if isinstance(per_text_groups, list) else [per_text_groups]
            for group in groups:
                if isinstance(group, list):
                    for triple in group:
                        triples.append(self._map_triple_chars(triple, start_map, end_map, text))
                elif isinstance(group, dict):
                    triples.append(self._map_triple_chars(group, start_map, end_map, text))

            output[orig_i] = triples
        return output

    @staticmethod
    def _map_triple_chars(triple, start_map, end_map, text):
        """Map a single relation triple from token indices to char positions."""
        mapped = dict(triple)
        for role in ("head", "tail"):
            if role in mapped and isinstance(mapped[role], dict):
                span = mapped[role]
                st = span.get("start", 0)
                ed = span.get("end", 0)
                if st < len(start_map) and ed < len(end_map):
                    start_char = start_map[st]
                    end_char = end_map[ed]
                    mapped[role] = {
                        "start": start_char,
                        "end": end_char,
                        "text": text[start_char:end_char],
                    }
        return mapped

    def _map_structuring_results(
        self,
        task_results: list,
        valid_to_orig_idx: List[int],
        all_start_maps: List[List[int]],
        all_end_maps: List[List[int]],
        valid_texts: List[str],
        num_original: int,
        all_classes_mappings: Optional[list] = None,
    ) -> List[Dict[str, List[Dict]]]:
        """Map structuring results into schema-keyed JSON output.

        The decoder returns per-text groups where each group corresponds to a
        schema (identified via ``classes_mapping.structuring_mapping``). Each
        group contains instances, and each instance is a list of field dicts.

        This method assembles them into::

            {schema_name: [{field1: value1, field2: value2}, ...]}

        with char-level positions for each value.
        """
        output: List[Dict[str, List[Dict]]] = [{} for _ in range(num_original)]

        for valid_i, per_text_groups in enumerate(task_results):
            orig_i = valid_to_orig_idx[valid_i]
            start_map = all_start_maps[valid_i]
            end_map = all_end_maps[valid_i]
            text = valid_texts[valid_i]

            # Resolve schema names from classes_mapping
            schema_names = self._get_structuring_schema_names(
                all_classes_mappings, valid_i,
            )

            result_dict: Dict[str, List[Dict]] = {}
            groups = per_text_groups if isinstance(per_text_groups, list) else [per_text_groups]

            for group_idx, group in enumerate(groups):
                schema_name = (schema_names[group_idx]
                               if group_idx < len(schema_names)
                               else f"schema_{group_idx}")

                if schema_name not in result_dict:
                    result_dict[schema_name] = []

                # group is a list of instances (each instance = list of field dicts)
                instances = group if isinstance(group, list) else [group]
                for instance in instances:
                    if isinstance(instance, list):
                        # Instance is a list of field dicts → assemble into one dict
                        instance_dict = {}
                        for field in instance:
                            mapped = self._map_field_to_value(
                                field, start_map, end_map, text,
                            )
                            if mapped is not None:
                                field_name = mapped["field"]
                                value = mapped["value"]
                                # Multiple values for same field → aggregate into list
                                if field_name in instance_dict:
                                    existing = instance_dict[field_name]
                                    if not isinstance(existing, list):
                                        instance_dict[field_name] = [existing, value]
                                    else:
                                        existing.append(value)
                                else:
                                    instance_dict[field_name] = value
                        if instance_dict:
                            result_dict[schema_name].append(instance_dict)
                    elif isinstance(instance, dict):
                        # Already a flat dict (shouldn't normally happen)
                        result_dict[schema_name].append(instance)

            output[orig_i] = result_dict
        return output

    @staticmethod
    def _map_field_to_value(field, start_map, end_map, text):
        """Map a field dict to {field, value} with char-level text extraction.

        Returns:
            Dict with 'field' (name) and 'value' (extracted text), or None.
        """
        if not isinstance(field, dict) or "field" not in field:
            return None
        st = field.get("start", 0)
        ed = field.get("end", 0)
        if st < len(start_map) and ed < len(end_map):
            start_char = start_map[st]
            end_char = end_map[ed]
            return {
                "field": field["field"],
                "value": text[start_char:end_char],
            }
        # Fallback: use token-level text if available
        return {
            "field": field["field"],
            "value": field.get("text", ""),
        }

    @staticmethod
    def _get_structuring_schema_names(
        all_classes_mappings: Optional[list],
        valid_idx: int,
    ) -> List[str]:
        """Extract schema names for a batch item from accumulated classes_mappings."""
        if not all_classes_mappings or valid_idx >= len(all_classes_mappings):
            return []
        cm = all_classes_mappings[valid_idx]
        if cm is None or not hasattr(cm, "structuring_mapping"):
            return []
        # The valid_idx corresponds to a batch item within its original batch.
        # Since we accumulate one mapping per item, structuring_mapping[0] has
        # the schema items for this text (the mapping is batch-level, but each
        # item in the DataLoader batch shares the same mapping).
        # We return all schema names across all batch items in this mapping.
        names = []
        for sm in cm.structuring_mapping:
            for item in sm.items:
                names.append(getattr(item, "name", None) or f"schema_{len(names)}")
        return names

    @staticmethod
    def _map_passthrough_results(
        task_results: list,
        valid_to_orig_idx: List[int],
        num_original: int,
    ) -> List:
        """Map results that don't need span→char conversion."""
        output = [[] for _ in range(num_original)]
        for valid_i, result in enumerate(task_results):
            orig_i = valid_to_orig_idx[valid_i]
            output[orig_i] = result
        return output

    @staticmethod
    def _empty_results(num_texts, entities, classes, relations, structures):
        """Return empty results dict when there's nothing to process."""
        results = {}
        if entities is not None:
            results["ner"] = [[] for _ in range(num_texts)]
        if classes is not None:
            results["classification"] = [[] for _ in range(num_texts)]
        if relations is not None:
            results["open_relex"] = [[] for _ in range(num_texts)]
        if structures is not None:
            results["structuring"] = [{} for _ in range(num_texts)]
        return results

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

        Args:
            texts: Input text(s).
            batch_size: Batch size for processing.

        Returns:
            Tensor of shape (N, D) with pooled text embeddings.
        """
        self.eval()
        if isinstance(texts, str):
            texts = [texts]

        valid_texts, valid_to_orig_idx = self._filter_valid_texts(texts)
        if not valid_texts:
            return torch.zeros(len(texts), self.config.hidden_size)

        all_tokens, _, _ = self.prepare_inputs(valid_texts)

        # Minimal input — no task labels, just encode text
        input_x = [{"tokenized_text": tk} for tk in all_tokens]

        collator = GLiNExTDataCollator(
            self.config,
            data_processor=self.data_processor,
            prepare_labels=False,
        )

        data_loader = DataLoader(
            input_x, batch_size=batch_size, shuffle=False, collate_fn=collator,
        )

        device = self.device
        all_embeddings = []

        for batch in data_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            output = self.model(**batch)
            # Mean pooling over valid word positions
            words_emb = output.words_embedding  # (B, W, D)
            mask = output.mask  # (B, W)
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (words_emb * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
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

        from gliner.modeling.encoder import BiEncoder

        self.eval()
        if not isinstance(self.model.token_rep_layer, BiEncoder):
            raise NotImplementedError("embed_labels requires BiEncoder token_rep_layer.")

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

        # Labels encoder (bi-encoder)
        if (self.config.labels_encoder is not None
                and hasattr(self.model, "token_rep_layer")
                and hasattr(self.model.token_rep_layer, "labels_encoder")):
            components["labels_encoder"] = self.model.token_rep_layer.labels_encoder.model

        # Individual task heads
        if hasattr(self.model, "heads"):
            for head_name, head_module in self.model.heads.items():
                components[head_name] = head_module

        # RNN layer
        if hasattr(self.model, "rnn"):
            components["rnn"] = self.model.rnn

        return components

    def train_model(
        self,
        train_dataset,
        eval_dataset=None,
        training_args=None,
        freeze_components: Optional[List[str]] = None,
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
                    "tokenized_text": ["Apple", "is", "a", "company"],
                    "extraction": [{"name": None, "ner": [["Apple", 0, 0, "company"]]}],
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
