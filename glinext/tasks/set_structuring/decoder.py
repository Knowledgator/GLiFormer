"""Decoder for independent entity-first set structuring."""

import torch

from ...processing.decoder import unflatten_by_batch_origin
from ..span_decoder import Span
from ..structuring.decoder import StructuringDecoder
from ..structuring.multilevel_decoder import make_multi_level_group_result


class SetStructuringDecoder(StructuringDecoder):
    """Join NER field decisions with second-stage anchor memberships."""

    config_attr = "set_structuring_config"
    token_logits_attr = "set_structuring_entity_logits"
    field_logits_attr = "set_structuring_field_logits"
    span_logits_attr = "set_structuring_logits"
    batch_origin_attr = "set_structuring_batch_origin"
    anchor_mask_attr = "set_structuring_anchor_mask"
    objectness_logits_attr = "set_structuring_objectness_logits"
    relation_scores_attr = "set_structuring_anchor_relation_scores"
    span_idx_attr = "set_structuring_span_idx"
    span_mask_attr = "set_structuring_span_mask"

    def decode(
        self,
        model_output,
        classes_mapping=None,
        threshold=None,
        flat_ner=True,
        multi_label=False,
        texts=None,
        objectness_threshold=None,
        preserve_empty_records=False,
        **kwargs,
    ):
        """Decode typed NER spans into unordered record-anchor groups.

        Composite checkpoints expose field logits ``(BN, E, C)`` and
        independent anchor-membership logits ``(BN, A, E)``. A compatibility
        branch can still decode previously materialized legacy outputs with
        combined ``(BN, A, E, C)`` logits; it does not restore the old forward
        semantics when a checkpoint is loaded into the refactored head.
        """

        membership_logits = getattr(model_output, self.span_logits_attr, None)
        field_logits = getattr(model_output, self.field_logits_attr, None)
        span_idx = getattr(model_output, self.span_idx_attr, None)
        span_mask = getattr(model_output, self.span_mask_attr, None)
        if membership_logits is None or span_idx is None or span_mask is None:
            return []

        if membership_logits.dim() == 4 and field_logits is None:
            return super().decode(
                model_output,
                classes_mapping=classes_mapping,
                threshold=threshold,
                flat_ner=flat_ner,
                multi_label=multi_label,
                texts=texts,
                objectness_threshold=objectness_threshold,
                preserve_empty_records=preserve_empty_records,
                **kwargs,
            )
        if membership_logits.dim() != 3:
            raise ValueError(
                "set_structuring_logits must have shape (BN, A, E), got "
                f"{tuple(membership_logits.shape)}"
            )
        if field_logits is None or field_logits.dim() != 3:
            raise ValueError(
                "set_structuring_field_logits must have shape (BN, E, C)"
            )

        batch_groups, anchor_count, entity_count = membership_logits.shape
        if field_logits.shape[:2] != (batch_groups, entity_count):
            raise ValueError(
                "set structuring field logits must share the membership "
                f"(BN, E) axes; got {tuple(field_logits.shape[:2])} and "
                f"{(batch_groups, entity_count)}"
            )
        if (
            span_idx.dim() != 3
            or span_idx.shape != (batch_groups, entity_count, 2)
        ):
            raise ValueError(
                "set_structuring_span_idx must have shape (BN, E, 2), got "
                f"{tuple(span_idx.shape)}"
            )
        if span_mask.shape != (batch_groups, entity_count):
            raise ValueError(
                "set_structuring_span_mask must have shape (BN, E), got "
                f"{tuple(span_mask.shape)}"
            )

        raw_anchor_mask = getattr(model_output, self.anchor_mask_attr, None)
        objectness_logits = getattr(
            model_output,
            self.objectness_logits_attr,
            None,
        )
        relation_scores = getattr(
            model_output,
            self.relation_scores_attr,
            None,
        )
        for value, name in (
            (raw_anchor_mask, self.anchor_mask_attr),
            (objectness_logits, self.objectness_logits_attr),
        ):
            if value is not None and value.shape != (
                batch_groups,
                anchor_count,
            ):
                raise ValueError(
                    f"{name} must have shape (BN, A), got "
                    f"{tuple(value.shape)}"
                )
        if relation_scores is not None and (
            relation_scores.dim() != 3
            or relation_scores.shape != (
                batch_groups,
                anchor_count,
                anchor_count,
            )
        ):
            raise ValueError(
                f"{self.relation_scores_attr} must have shape (BN, A, A), "
                f"got {tuple(relation_scores.shape)}"
            )

        if objectness_threshold is None and threshold is not None:
            objectness_threshold = threshold
        if objectness_threshold is None:
            objectness_threshold = self.objectness_threshold
        threshold = self.threshold if threshold is None else threshold
        if objectness_threshold is None:
            objectness_threshold = threshold
        anchor_mask = self._resolve_anchor_mask(
            raw_anchor_mask,
            objectness_logits,
            objectness_threshold,
            relation_scores=relation_scores,
            expected_shape=(batch_groups, anchor_count),
        )
        reliable_presence_mask = None
        if objectness_logits is not None:
            reliable_presence_mask = (
                torch.sigmoid(objectness_logits) > objectness_threshold
            )
            if raw_anchor_mask is not None:
                reliable_presence_mask &= raw_anchor_mask.bool()

        class_count = field_logits.shape[2]
        membership_probs = torch.sigmoid(membership_logits)
        field_probs = torch.sigmoid(field_logits)
        id_to_fields = self._build_field_class_maps(
            classes_mapping,
            batch_groups,
        )
        multi_level_contexts = self._build_multi_level_contexts(
            classes_mapping,
            batch_groups,
        )
        anchor_mask = self._rescue_nested_relation_anchors(
            anchor_mask,
            raw_anchor_mask,
            relation_scores,
            multi_level_contexts,
        )
        batch_origin = getattr(model_output, self.batch_origin_attr, None)
        if batch_origin is None:
            batch_origin = torch.arange(
                batch_groups,
                device=membership_logits.device,
            )
        elif batch_origin.shape != (batch_groups,):
            raise ValueError(
                "set_structuring_batch_origin must have shape (BN,), got "
                f"{tuple(batch_origin.shape)}"
            )
        batch_size = getattr(model_output, "batch_size", None)
        if batch_size is None:
            batch_size = int(batch_origin.max().item()) + 1

        flat_results = []
        for batch_idx in range(batch_groups):
            text_idx = int(batch_origin[batch_idx].item())
            field_id_to_class = (
                id_to_fields[batch_idx]
                if batch_idx < len(id_to_fields) and id_to_fields[batch_idx]
                else {idx: str(idx) for idx in range(class_count)}
            )
            valid_entities = torch.where(
                span_mask[batch_idx, :entity_count].bool()
            )[0]
            instances = []
            nodes = []
            context = (
                multi_level_contexts[batch_idx]
                if batch_idx < len(multi_level_contexts)
                else None
            )
            is_multi_level = bool(
                context and getattr(context["mapping"], "multi_level", False)
            )
            for anchor_idx in range(anchor_count):
                if (
                    anchor_mask is not None
                    and not bool(anchor_mask[batch_idx, anchor_idx])
                ):
                    continue

                spans = []
                for entity_idx in valid_entities.tolist():
                    membership_score = membership_probs[
                        batch_idx,
                        anchor_idx,
                        entity_idx,
                    ]
                    if membership_score <= threshold:
                        continue
                    class_ids = torch.where(
                        field_probs[batch_idx, entity_idx] > threshold
                    )[0]
                    for class_idx in class_ids.tolist():
                        if class_idx not in field_id_to_class:
                            continue
                        spans.append(
                            Span(
                                start=int(
                                    span_idx[batch_idx, entity_idx, 0].item()
                                ),
                                end=int(
                                    span_idx[batch_idx, entity_idx, 1].item()
                                ),
                                entity_type=field_id_to_class[class_idx],
                                score=min(
                                    float(membership_score.item()),
                                    float(
                                        field_probs[
                                            batch_idx,
                                            entity_idx,
                                            class_idx,
                                        ].item()
                                    ),
                                ),
                            )
                        )

                spans = self.greedy_search(spans, flat_ner, multi_label)
                fields = self._spans_to_fields(spans, texts, text_idx)
                if is_multi_level:
                    nodes.append({
                        "anchor_index": anchor_idx,
                        "fields": fields,
                        "presence_is_reliable": bool(
                            reliable_presence_mask is not None
                            and reliable_presence_mask[
                                batch_idx, anchor_idx
                            ]
                        ),
                    })
                elif spans:
                    instances.append(
                        fields
                    )
            if is_multi_level:
                flat_results.append(make_multi_level_group_result(
                    nodes,
                    (
                        relation_scores[batch_idx]
                        if relation_scores is not None else None
                    ),
                    context["mapping"],
                    context["output_mode"],
                    preserve_empty_records=preserve_empty_records,
                ))
            else:
                flat_results.append(instances)

        return unflatten_by_batch_origin(
            flat_results,
            batch_origin,
            batch_size,
        )


__all__ = ["SetStructuringDecoder"]
