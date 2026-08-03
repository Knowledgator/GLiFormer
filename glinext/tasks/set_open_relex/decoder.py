"""Decoder for independent entity-first set relation extraction."""

import torch

from ...processing.decoder import unflatten_by_batch_origin
from ..span_decoder import SpanDecoder


class SetOpenRelexDecoder(SpanDecoder):
    """Decode relation slots by assigning one entity to each endpoint role.

    The public relation tensor contains per-slot, per-relation source/target
    confidence logits with shape ``(BN, A, R, 2)``.  A separate assignment
    tensor, ``(BN, A, R, E, 2)``, selects the concrete entity span for each
    role.  Keeping those decisions separate lets the head train unordered
    relation slots while the decoder still emits ordinary directed triples.
    """

    logits_attr = "set_open_rel_logits"
    assignment_logits_attr = "set_open_rel_assignment_logits"
    span_idx_attr = "set_open_rel_span_idx"
    span_mask_attr = "set_open_rel_span_mask"
    anchor_mask_attr = "set_open_rel_anchor_mask"
    objectness_logits_attr = "set_open_rel_objectness_logits"
    batch_origin_attr = "set_open_rel_batch_origin"

    def __init__(self, config):
        super().__init__(config)
        task_config = getattr(config, "set_open_relex_config", None)
        self.objectness_threshold = (
            getattr(task_config, "anchor_objectness_threshold", None)
            if task_config is not None
            else None
        )

    @staticmethod
    def _shape_error(name: str, expected: str, actual: torch.Tensor):
        raise ValueError(
            f"{name} must have shape {expected}, got {tuple(actual.shape)}"
        )

    def _validate_inputs(
        self,
        logits,
        assignment_logits,
        span_idx,
        span_mask,
        anchor_mask,
        objectness_logits,
        batch_origin,
    ):
        if logits.dim() != 4 or logits.shape[-1] != 2:
            self._shape_error(self.logits_attr, "(BN, A, R, 2)", logits)

        batch_groups, anchor_count, relation_count, _ = logits.shape
        if (
            assignment_logits.dim() != 5
            or assignment_logits.shape[:3]
            != (batch_groups, anchor_count, relation_count)
            or assignment_logits.shape[-1] != 2
        ):
            self._shape_error(
                self.assignment_logits_attr,
                "(BN, A, R, E, 2)",
                assignment_logits,
            )

        entity_count = assignment_logits.shape[3]
        if span_idx.shape != (batch_groups, entity_count, 2):
            self._shape_error(
                self.span_idx_attr,
                f"({batch_groups}, {entity_count}, 2)",
                span_idx,
            )
        if span_mask.shape != (batch_groups, entity_count):
            self._shape_error(
                self.span_mask_attr,
                f"({batch_groups}, {entity_count})",
                span_mask,
            )

        expected_anchor_shape = (batch_groups, anchor_count)
        for value, name in (
            (anchor_mask, self.anchor_mask_attr),
            (objectness_logits, self.objectness_logits_attr),
        ):
            if value is not None and value.shape != expected_anchor_shape:
                self._shape_error(
                    name,
                    f"({batch_groups}, {anchor_count})",
                    value,
                )

        if batch_origin is not None and batch_origin.shape != (batch_groups,):
            self._shape_error(
                self.batch_origin_attr,
                f"({batch_groups},)",
                batch_origin,
            )

        return batch_groups, anchor_count, relation_count, entity_count

    @staticmethod
    def _reverse_relation_mapping(item) -> dict[int, str]:
        relation_mapping = getattr(item, "rel_class_to_id", None)
        if relation_mapping is None:
            return {}
        if hasattr(relation_mapping, "get_reverse_mapping"):
            return relation_mapping.get_reverse_mapping()
        class_to_id = getattr(relation_mapping, "class_to_id", None)
        if isinstance(class_to_id, dict):
            return {class_id: name for name, class_id in class_to_id.items()}
        return {}

    def _build_flat_relation_maps(
        self,
        classes_mapping,
        batch_groups: int,
    ) -> list[dict[int, str]]:
        """Return one relation-id mapping for every flattened schema group."""

        if classes_mapping is None:
            return [{} for _ in range(batch_groups)]
        if isinstance(classes_mapping, dict):
            return [dict(classes_mapping) for _ in range(batch_groups)]
        if isinstance(classes_mapping, list):
            return [
                dict(classes_mapping[idx])
                if idx < len(classes_mapping) else {}
                for idx in range(batch_groups)
            ]

        maps = [{} for _ in range(batch_groups)]
        iterator = None
        for iterator_name in (
            "flat_set_open_relex_iter",
            "flat_open_relex_iter",
        ):
            candidate = getattr(classes_mapping, iterator_name, None)
            if candidate is not None:
                iterator = candidate()
                break

        if iterator is not None:
            for flat_idx, _, _, item in iterator:
                if 0 <= flat_idx < batch_groups:
                    maps[flat_idx] = self._reverse_relation_mapping(item)
            return maps

        mapping_list = getattr(
            classes_mapping,
            "set_open_relex_mapping",
            getattr(classes_mapping, "open_relex_mapping", []),
        )
        flat_idx = 0
        for item_mapping in mapping_list:
            for item in getattr(item_mapping, "items", []):
                if flat_idx >= batch_groups:
                    return maps
                maps[flat_idx] = self._reverse_relation_mapping(item)
                flat_idx += 1
        return maps

    @staticmethod
    def _active_anchor_mask(
        logits,
        anchor_mask,
        objectness_logits,
        objectness_threshold,
    ):
        active = torch.ones(
            logits.shape[:2],
            dtype=torch.bool,
            device=logits.device,
        )
        if anchor_mask is not None:
            active &= anchor_mask.to(device=logits.device).bool()
        if objectness_logits is not None:
            active &= (
                torch.sigmoid(objectness_logits.to(device=logits.device))
                > objectness_threshold
            )
        return active

    def decode(
        self,
        model_output,
        classes_mapping=None,
        threshold=None,
        texts=None,
        objectness_threshold=None,
        **kwargs,
    ) -> list[list[dict]]:
        """Decode entity assignments into directed relation triples."""

        del kwargs
        logits = getattr(model_output, self.logits_attr, None)
        if logits is None:
            return []

        assignment_logits = getattr(
            model_output,
            self.assignment_logits_attr,
            None,
        )
        span_idx = getattr(model_output, self.span_idx_attr, None)
        span_mask = getattr(model_output, self.span_mask_attr, None)
        missing = [
            name
            for value, name in (
                (assignment_logits, self.assignment_logits_attr),
                (span_idx, self.span_idx_attr),
                (span_mask, self.span_mask_attr),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "set open relation decoding requires " + ", ".join(missing)
            )

        anchor_mask = getattr(model_output, self.anchor_mask_attr, None)
        objectness_logits = getattr(
            model_output,
            self.objectness_logits_attr,
            None,
        )
        batch_origin = getattr(model_output, self.batch_origin_attr, None)
        (
            batch_groups,
            anchor_count,
            relation_count,
            _,
        ) = self._validate_inputs(
            logits,
            assignment_logits,
            span_idx,
            span_mask,
            anchor_mask,
            objectness_logits,
            batch_origin,
        )

        threshold = self.threshold if threshold is None else threshold
        if objectness_threshold is None:
            objectness_threshold = self.objectness_threshold
        if objectness_threshold is None:
            objectness_threshold = threshold

        if batch_origin is None:
            batch_origin = torch.arange(
                batch_groups,
                dtype=torch.long,
                device=logits.device,
            )
        batch_size = getattr(model_output, "batch_size", None)
        if batch_size is None:
            batch_size = (
                int(batch_origin.max().item()) + 1
                if batch_origin.numel() else 0
            )

        role_probs = torch.sigmoid(logits)
        assignment_probs = torch.sigmoid(assignment_logits)
        active_anchors = self._active_anchor_mask(
            logits,
            anchor_mask,
            objectness_logits,
            objectness_threshold,
        )
        relation_maps = self._build_flat_relation_maps(
            classes_mapping,
            batch_groups,
        )

        flat_results = []
        for batch_idx in range(batch_groups):
            text_idx = int(batch_origin[batch_idx].item())
            relation_map = relation_maps[batch_idx]
            valid_entities = span_mask[batch_idx].bool().clone()
            valid_entities &= span_idx[batch_idx, :, 0] >= 0
            valid_entities &= (
                span_idx[batch_idx, :, 1]
                >= span_idx[batch_idx, :, 0]
            )
            triples = {}

            if valid_entities.any():
                for anchor_idx in range(anchor_count):
                    if not bool(active_anchors[batch_idx, anchor_idx]):
                        continue
                    for relation_idx in range(relation_count):
                        if relation_map and relation_idx not in relation_map:
                            continue
                        public_scores = role_probs[
                            batch_idx,
                            anchor_idx,
                            relation_idx,
                        ]
                        if not bool((public_scores > threshold).all()):
                            continue

                        endpoint_ids = []
                        endpoint_scores = []
                        for role_idx in range(2):
                            candidates = assignment_probs[
                                batch_idx,
                                anchor_idx,
                                relation_idx,
                                :,
                                role_idx,
                            ].masked_fill(~valid_entities, float("-inf"))
                            entity_idx = int(candidates.argmax().item())
                            entity_score = float(candidates[entity_idx].item())
                            if entity_score <= threshold:
                                break
                            endpoint_ids.append(entity_idx)
                            endpoint_scores.append(entity_score)
                        if len(endpoint_ids) != 2:
                            continue

                        head_idx, tail_idx = endpoint_ids
                        head_start, head_end = (
                            int(value)
                            for value in span_idx[
                                batch_idx, head_idx
                            ].tolist()
                        )
                        tail_start, tail_end = (
                            int(value)
                            for value in span_idx[
                                batch_idx, tail_idx
                            ].tolist()
                        )
                        relation_name = relation_map.get(
                            relation_idx,
                            str(relation_idx),
                        )
                        source_confidence = min(
                            float(public_scores[0].item()),
                            endpoint_scores[0],
                        )
                        target_confidence = min(
                            float(public_scores[1].item()),
                            endpoint_scores[1],
                        )
                        score = (source_confidence + target_confidence) / 2.0
                        triple = {
                            "head": {
                                "start": head_start,
                                "end": head_end,
                                "text": self.resolve_span_text(
                                    texts,
                                    text_idx,
                                    head_start,
                                    head_end,
                                ),
                            },
                            "tail": {
                                "start": tail_start,
                                "end": tail_end,
                                "text": self.resolve_span_text(
                                    texts,
                                    text_idx,
                                    tail_start,
                                    tail_end,
                                ),
                            },
                            "relation": relation_name,
                            "score": score,
                        }
                        key = (
                            head_start,
                            head_end,
                            tail_start,
                            tail_end,
                            relation_name,
                        )
                        previous = triples.get(key)
                        if previous is None or score > previous["score"]:
                            triples[key] = triple

            flat_results.append(list(triples.values()))

        return unflatten_by_batch_origin(
            flat_results,
            batch_origin,
            int(batch_size),
        )

    def map_results(
        self,
        task_results: list,
        valid_to_orig_idx: list[int],
        all_start_maps: list[list[int]],
        all_end_maps: list[list[int]],
        valid_texts: list[str],
        num_original: int,
        **kwargs,
    ) -> list[list[dict]]:
        """Map token endpoints to character offsets like open relex."""

        del kwargs
        output = [[] for _ in range(num_original)]
        for valid_idx, per_text_groups in enumerate(task_results):
            original_idx = valid_to_orig_idx[valid_idx]
            start_map = all_start_maps[valid_idx]
            end_map = all_end_maps[valid_idx]
            text = valid_texts[valid_idx]
            triples = []
            groups = (
                per_text_groups
                if isinstance(per_text_groups, list)
                else [per_text_groups]
            )
            for group in groups:
                values = group if isinstance(group, list) else [group]
                for triple in values:
                    if isinstance(triple, dict):
                        triples.append(
                            self._map_triple_chars(
                                triple,
                                start_map,
                                end_map,
                                text,
                            )
                        )
            output[original_idx] = triples
        return output

    @staticmethod
    def _map_triple_chars(triple, start_map, end_map, text):
        mapped = dict(triple)
        for role in ("head", "tail"):
            value = mapped.get(role)
            if not isinstance(value, dict):
                continue
            span = dict(value)
            start = int(span.get("start", -1))
            end = int(span.get("end", -1))
            if 0 <= start < len(start_map) and 0 <= end < len(end_map):
                start_char = start_map[start]
                end_char = end_map[end]
                span.update(
                    {
                        "start": start_char,
                        "end": end_char,
                        "text": text[start_char:end_char],
                    }
                )
            mapped[role] = span
        return mapped


SetOpenRelationExtractionDecoder = SetOpenRelexDecoder


__all__ = ["SetOpenRelexDecoder", "SetOpenRelationExtractionDecoder"]
