"""Pair-first set-prediction open relation extraction."""

import torch

from ..classification.scorer import ClassificationScorer
from ..open_relex.model import OpenRelexHead


class SetOpenRelexHead(OpenRelexHead):
    """Extract source/tail pairs before open-vocabulary relation typing.

    Every anchor acts as a relation-pair query. It first predicts independent,
    class-agnostic BIO scores for the source and tail entities. The resulting
    anchor/pair representation is then multiplied by each relation-label
    representation. Broadcasting that class energy over both entity roles
    preserves the regular ``(group, anchor, class, token, role, BIO)`` output.
    """

    name = "set_open_relex"
    dependencies = []

    def __init__(self, config, hidden_size, dropout, shared_layers=None):
        super().__init__(config, hidden_size, dropout, shared_layers)
        open_rel_cfg = config.open_relex_config
        self.cls_head = ClassificationScorer.from_config(
            scorer_type=getattr(open_rel_cfg, "scorer_type", "dot"),
            hidden_size=hidden_size,
        )

    def _compute_relation_scores(self, flat_inputs, open_rel_count, threshold):
        feature_embeddings = flat_inputs.words_embedding
        feature_mask = flat_inputs.mask
        relation_embeddings = flat_inputs.child_embedding

        anchors, anchor_mask = self._generate_anchors(
            flat_inputs.parent_embedding,
            feature_embeddings,
            count=open_rel_count,
            threshold=threshold,
            feature_mask=feature_mask,
        )
        anchors = self._refine_anchors(
            anchors,
            feature_embeddings,
            memory_mask=feature_mask,
            anchor_mask=anchor_mask,
        )

        batch_size, anchor_count, hidden_size = anchors.shape
        class_count = relation_embeddings.shape[1]
        sequence_length = feature_embeddings.shape[1]

        # Extract the two entity roles without conditioning their geometry on
        # a relation class. The refined anchor is the representation of this
        # ordered source/tail pair.
        source_logits = self.head_scorer(
            anchors,
            feature_embeddings,
            word_mask=feature_mask,
        )
        tail_logits = self.tail_scorer(
            anchors,
            feature_embeddings,
            word_mask=feature_mask,
        )

        label_representations = self._model_anchors(
            anchors,
            relation_embeddings,
            anchor_mask=anchor_mask,
            child_mask=flat_inputs.child_mask,
        )
        class_logits = self.cls_head(
            anchors.reshape(batch_size * anchor_count, hidden_size),
            label_representations.reshape(
                batch_size * anchor_count,
                class_count,
                hidden_size,
            ),
        ).reshape(batch_size, anchor_count, class_count)

        class_energy = class_logits[:, :, :, None, None]
        source_class_logits = source_logits[:, :, None, :, :] + class_energy
        tail_class_logits = tail_logits[:, :, None, :, :] + class_energy
        logits = torch.stack(
            [source_class_logits, tail_class_logits],
            dim=-2,
        )
        if feature_mask is not None:
            logits = logits * feature_mask[:, None, None, :, None, None].to(
                dtype=logits.dtype,
            )

        fused_flat = label_representations.reshape(
            batch_size,
            anchor_count * class_count,
            hidden_size,
        )
        predictions = (
            logits,
            anchors,
            anchor_mask,
            fused_flat,
            (
                batch_size,
                anchor_count,
                class_count,
                sequence_length,
            ),
        )
        extra = {
            "pair_representations": anchors,
            "source_logits": source_logits,
            "head_entity_logits": source_logits,
            "tail_entity_logits": tail_logits,
            "class_logits": class_logits,
        }
        return predictions, extra

    def _compute_span_relation_scores(
        self,
        feature_embeddings,
        span_idx,
        anchors,
        fused_flat,
        dims,
        prediction_extra,
    ):
        """Extract source/tail span candidates before relation typing."""

        batch_size, anchor_count, class_count, _ = dims
        span_rep = self.span_rep_layer(feature_embeddings, span_idx)
        source_span_rep = self.head_span_proj(span_rep)
        tail_span_rep = self.tail_span_proj(span_rep)
        source_scores = torch.einsum(
            "BSD,BAD->BSA",
            source_span_rep,
            anchors,
        )
        tail_scores = torch.einsum(
            "BSD,BAD->BSA",
            tail_span_rep,
            anchors,
        )
        class_logits = prediction_extra["class_logits"]
        if class_logits.shape != (batch_size, anchor_count, class_count):
            raise ValueError(
                "Set open-relation class logits do not match span dimensions"
            )
        source_class_scores = (
            source_scores[:, :, :, None]
            + class_logits[:, None, :, :]
        )
        tail_class_scores = (
            tail_scores[:, :, :, None]
            + class_logits[:, None, :, :]
        )
        return torch.stack(
            [source_class_scores, tail_class_scores],
            dim=-1,
        )


SetOpenRelationExtractionHead = SetOpenRelexHead


__all__ = ["SetOpenRelexHead", "SetOpenRelationExtractionHead"]
