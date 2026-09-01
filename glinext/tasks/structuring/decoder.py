"""Decoder for entity-first set-prediction structuring."""

import torch

from ...processing.decoder import unflatten_by_batch_origin
from ...processing.structuring_decoder import StructuringDecoder as _BaseStructuringDecoder
from ..span_decoder import Span


class StructuringDecoder(_BaseStructuringDecoder):
    """Join NER field decisions with second-stage anchor memberships."""

    config_attr = "structuring_config"
    mapping_attr = "structuring_mapping"
    token_logits_attr = "structuring_entity_logits"
    field_logits_attr = "structuring_field_logits"
    span_logits_attr = "structuring_logits"
    batch_origin_attr = "structuring_batch_origin"
    anchor_mask_attr = "structuring_anchor_mask"
    objectness_logits_attr = "structuring_objectness_logits"
    relation_scores_attr = "structuring_anchor_relation_scores"
    span_idx_attr = "structuring_span_idx"
    span_mask_attr = "structuring_span_mask"

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

        Field logits have shape ``(BN, E, C)`` and independent
        anchor-membership logits have shape ``(BN, A, E)``.
        """

        membership_logits = getattr(model_output, self.span_logits_attr, None)
        field_logits = getattr(model_output, self.field_logits_attr, None)
        span_idx = getattr(model_output, self.span_idx_attr, None)
        span_mask = getattr(model_output, self.span_mask_attr, None)
        if membership_logits is None or span_idx is None or span_mask is None:
            return []

        if membership_logits.dim() != 3:
            raise ValueError(
                "structuring_logits must have shape (BN, A, E), got "
                f"{tuple(membership_logits.shape)}"
            )
        if field_logits is None or field_logits.dim() != 3:
            raise ValueError(
                "structuring_field_logits must have shape (BN, E, C)"
            )

        batch_groups, anchor_count, entity_count = membership_logits.shape
        if field_logits.shape[:2] != (batch_groups, entity_count):
            raise ValueError(
                "structuring field logits must share the membership "
                f"(BN, E) axes; got {tuple(field_logits.shape[:2])} and "
                f"{(batch_groups, entity_count)}"
            )
        if (
            span_idx.dim() != 3
            or span_idx.shape != (batch_groups, entity_count, 2)
        ):
            raise ValueError(
                "structuring_span_idx must have shape (BN, E, 2), got "
                f"{tuple(span_idx.shape)}"
            )
        if span_mask.shape != (batch_groups, entity_count):
            raise ValueError(
                "structuring_span_mask must have shape (BN, E), got "
                f"{tuple(span_mask.shape)}"
            )

        class_count = field_logits.shape[2]
        membership_probs = torch.sigmoid(membership_logits)
        field_probs = torch.sigmoid(field_logits)
        context = self._prepare_decode_context(
            model_output,
            classes_mapping,
            batch_groups=batch_groups,
            anchor_count=anchor_count,
            device=membership_logits.device,
            threshold=threshold,
            objectness_threshold=objectness_threshold,
        )

        flat_results = []
        for batch_idx in range(batch_groups):
            text_idx = int(context.batch_origin[batch_idx].item())
            field_id_to_class = (
                context.id_to_fields[batch_idx]
                if batch_idx < len(context.id_to_fields)
                and context.id_to_fields[batch_idx]
                else {idx: str(idx) for idx in range(class_count)}
            )
            valid_entities = torch.where(
                span_mask[batch_idx, :entity_count].bool()
            )[0]
            anchor_entries = []
            group_context = (
                context.multi_level_contexts[batch_idx]
                if batch_idx < len(context.multi_level_contexts)
                else None
            )
            for anchor_idx in range(anchor_count):
                if (
                    context.anchor_mask is not None
                    and not bool(
                        context.anchor_mask[batch_idx, anchor_idx]
                    )
                ):
                    continue

                spans = []
                for entity_idx in valid_entities.tolist():
                    membership_score = membership_probs[
                        batch_idx,
                        anchor_idx,
                        entity_idx,
                    ]
                    if membership_score <= context.threshold:
                        continue
                    class_ids = torch.where(
                        field_probs[batch_idx, entity_idx] > context.threshold
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
                anchor_entries.append({
                    "anchor_index": anchor_idx,
                    "fields": fields,
                    "presence_is_reliable": bool(
                        context.reliable_presence_mask is not None
                        and context.reliable_presence_mask[
                            batch_idx, anchor_idx
                        ]
                    ),
                })
            flat_results.append(
                self._finalize_anchor_group(
                    anchor_entries,
                    relation_scores=(
                        context.relation_scores[batch_idx]
                        if context.relation_scores is not None else None
                    ),
                    context=group_context,
                    preserve_empty_records=preserve_empty_records,
                )
            )

        return unflatten_by_batch_origin(
            flat_results,
            context.batch_origin,
            context.batch_size,
        )


__all__ = ["StructuringDecoder"]
