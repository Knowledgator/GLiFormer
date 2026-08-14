"""Independent entity-first set-prediction open relation extraction."""

import math
import warnings
from copy import copy

import torch
from gliner.modeling.span_rep import SpanRepLayer
from gliner.modeling.utils import extract_spans_from_tokens
from torch import nn

from ...layers.mlp import create_mlp
from .. import TaskHeadOutput
from ..matcher import (
    batched_masked_assignment,
    gold_anchor_mask,
    matched_anchor_targets,
    matched_objectness_loss,
)
from ..ner.model import NERHead


class SetOpenRelexHead(NERHead):
    """Recognize entities, group directed pairs, then classify relations.

    The first stage is the existing NER pass. Its selected spans are pooled and
    passed to a relation-independent set of pair anchors. Each anchor assigns a
    source and target entity, the two soft endpoint representations are fused
    into a directed pair representation, and that representation is compared
    with every open-vocabulary relation embedding.

    Two score tensors are intentionally exposed:

    * ``logits`` has shape ``(BN, N, R)`` and classifies each pair against the
      ``R`` requested relations.
    * ``extra["assignment_logits"]`` has shape ``(BN, N, E, 2)`` and selects
      the source and target from the ``E`` recognized spans independently of
      the relation vocabulary.

    ``N`` is a permutation-invariant pair-slot axis. Hungarian matching
    supervises pair assignment and relation classification jointly and also
    provides the objectness targets.
    """

    name = "set_open_relex"
    dependencies = []

    def __init__(self, config, hidden_size, dropout, shared_layers=None):
        set_cfg = config.set_open_relex_config
        shared_layers = shared_layers or {}

        # Entity recognition always uses one parent anchor. Relation query
        # settings belong exclusively to the second stage.
        # A shallow task-local copy avoids re-running the set-query config's
        # fixed-slot validation while configuring the internal one-anchor NER
        # stage. No nested component mapping is mutated here.
        entity_cfg = copy(set_cfg)
        entity_cfg.anchor_mode = "parent"
        entity_cfg.anchor_layer = None
        entity_cfg.anchor_normalization = "none"
        entity_cfg.anchor_refine_layers = 0
        entity_cfg.anchor_refinement = "none"
        entity_cfg.anchor_memory_position = "none"
        entity_cfg.anchor_query_position = "none"
        entity_cfg.anchor_self_attention_bias = "none"
        entity_cfg.anchor_cross_attention_bias = "none"
        entity_cfg.represent_spans = False
        super().__init__(
            config,
            hidden_size,
            dropout,
            shared_layers={},
            task_config=entity_cfg,
        )

        self.entity_loss_coef = float(set_cfg.entity_loss_coef)
        self.assignment_loss_coef = float(set_cfg.assignment_loss_coef)
        self.bio_loss_reduction = set_cfg.bio_loss_reduction
        self._relation_capacity_warning_emitted = False

        self.relation_anchor_layer = self._build_anchor_layer(
            set_cfg,
            hidden_size,
            dropout,
        )
        self.relation_anchor_normalizer = self._build_anchor_normalizer(
            set_cfg,
            hidden_size,
        )
        self._relation_uses_fixed_slots = hasattr(
            self.relation_anchor_layer,
            "num_slots",
        )
        groups_layer = getattr(self.relation_anchor_layer, "groups_layer", None)
        self._relation_anchor_capacity = int(
            getattr(
                self.relation_anchor_layer,
                "num_slots",
                getattr(groups_layer, "max_count", set_cfg.max_count),
            )
        )
        refinement = self._build_anchor_refinement(
            set_cfg,
            hidden_size,
            dropout,
            shared_layers,
        )
        if refinement is not None:
            self.relation_anchor_refine = refinement
        self.relation_anchor_refine_positions = None
        self.relation_self_attention_bias = None
        self.relation_cross_attention_bias = None
        if hasattr(self, "relation_anchor_refine"):
            self.relation_anchor_refine_positions = (
                self._build_refinement_positions(set_cfg, hidden_size)
            )
            (
                self.relation_self_attention_bias,
                self.relation_cross_attention_bias,
            ) = self._build_refinement_biases(
                set_cfg,
                num_heads=self.relation_anchor_refine.num_heads,
            )

        self.entity_span_rep_layer = SpanRepLayer(
            span_mode="token_level",
            hidden_size=hidden_size,
            max_width=getattr(config, "max_width", 12),
            dropout=dropout,
        )
        self.endpoint_query = nn.Linear(hidden_size, hidden_size * 2)
        self.pair_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )
        self.logit_scale = math.sqrt(hidden_size)

        self.use_anchor_objectness = bool(set_cfg.anchor_objectness)
        self.anchor_objectness_loss_coef = float(
            set_cfg.anchor_objectness_loss_coef
        )
        if self.use_anchor_objectness:
            self.objectness_head = create_mlp(
                input_dim=hidden_size,
                intermediate_dims=[hidden_size],
                output_dim=1,
                dropout=dropout,
                activation="gelu",
            )

    @classmethod
    def from_config(cls, config, shared_layers=None, **kwargs):
        if config.set_open_relex_config is None:
            return None
        return cls(
            config,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            shared_layers=shared_layers,
        )

    @staticmethod
    def _sanitize_spans(span_idx, span_mask, sequence_length, word_mask=None):
        if span_idx.dim() != 3 or span_idx.shape[-1] != 2:
            raise ValueError("set open relation spans must have shape (BN, E, 2)")
        if span_idx.shape[:2] != span_mask.shape:
            raise ValueError("span indices and mask must share (BN, E)")

        valid = span_mask.bool()
        starts = span_idx[..., 0]
        ends = span_idx[..., 1]
        valid &= (starts >= 0) & (ends >= starts) & (ends < sequence_length)
        if word_mask is not None and sequence_length > 0:
            fitted_mask = word_mask[:, :sequence_length].bool()
            safe_starts = starts.clamp(min=0, max=sequence_length - 1)
            safe_ends = ends.clamp(min=0, max=sequence_length - 1)
            valid &= fitted_mask.gather(1, safe_starts)
            valid &= fitted_mask.gather(1, safe_ends)
        safe_idx = torch.where(
            valid.unsqueeze(-1),
            span_idx,
            torch.zeros_like(span_idx),
        )
        return safe_idx, valid

    @staticmethod
    def _coalesce_spans(span_idx, span_mask, assignment_labels=None):
        """Keep one entity row per boundary and merge its endpoint targets."""

        unique_per_batch = []
        source_groups = []
        for batch_idx in range(span_idx.shape[0]):
            boundary_to_output = {}
            unique = []
            groups = []
            for entity_idx in torch.where(span_mask[batch_idx].bool())[0].tolist():
                boundary = tuple(span_idx[batch_idx, entity_idx].tolist())
                output_idx = boundary_to_output.get(boundary)
                if output_idx is None:
                    output_idx = len(unique)
                    boundary_to_output[boundary] = output_idx
                    unique.append(boundary)
                    groups.append([])
                groups[output_idx].append(entity_idx)
            unique_per_batch.append(unique)
            source_groups.append(groups)

        max_entities = max(
            max((len(spans) for spans in unique_per_batch), default=0),
            1,
        )
        unique_idx = span_idx.new_zeros(span_idx.shape[0], max_entities, 2)
        unique_mask = span_mask.new_zeros(span_idx.shape[0], max_entities)
        for batch_idx, spans in enumerate(unique_per_batch):
            if spans:
                unique_idx[batch_idx, :len(spans)] = span_idx.new_tensor(spans)
                unique_mask[batch_idx, :len(spans)] = True

        if assignment_labels is None:
            return unique_idx, unique_mask, None
        if assignment_labels.dim() != 4:
            raise ValueError(
                "set_open_rel_assignment_labels must have shape (BN, G, E, 2)"
            )
        if assignment_labels.shape[0] != span_idx.shape[0] or (
            assignment_labels.shape[2] != span_idx.shape[1]
        ):
            raise ValueError(
                "set open relation endpoint labels must share the span entity axis"
            )

        merged = assignment_labels.new_zeros(
            assignment_labels.shape[0],
            assignment_labels.shape[1],
            max_entities,
            assignment_labels.shape[3],
        )
        for batch_idx, groups in enumerate(source_groups):
            for output_idx, source_ids in enumerate(groups):
                merged[batch_idx, :, output_idx] = assignment_labels[
                    batch_idx, :, source_ids
                ].amax(dim=1)
        return unique_idx, unique_mask, merged

    def _pool_entity_spans(self, words_embedding, span_idx, span_mask):
        if words_embedding.shape[1] == 0:
            return words_embedding.new_zeros(
                words_embedding.shape[0], span_idx.shape[1], words_embedding.shape[-1]
            )
        safe_idx = span_idx * span_mask.unsqueeze(-1).long()
        representations = self.entity_span_rep_layer(words_embedding, safe_idx)
        return representations * span_mask.unsqueeze(-1).to(representations.dtype)

    def _relation_anchors(
        self,
        flat_inputs,
        entity_representations,
        entity_mask,
        batch,
    ):
        # Always retain a valid parent memory cell. This keeps transformer
        # query generation/refinement finite when NER extracts no entities.
        memory = torch.cat(
            [flat_inputs.parent_embedding.unsqueeze(1), entity_representations],
            dim=1,
        )
        memory_mask = torch.cat(
            [
                torch.ones(
                    entity_mask.shape[0],
                    1,
                    dtype=torch.bool,
                    device=entity_mask.device,
                ),
                entity_mask.bool(),
            ],
            dim=1,
        )
        count = None
        if not self._relation_uses_fixed_slots:
            # Count/query-driven layers must see the same capacity in training
            # and inference. Passing set_open_rel_count here would leak the
            # gold number of pairs during training and disappear at inference.
            count = flat_inputs.parent_embedding.new_full(
                (flat_inputs.parent_embedding.shape[0],),
                self._relation_anchor_capacity,
                dtype=torch.long,
            )

        anchors, anchor_mask = self.relation_anchor_layer(
            flat_inputs.parent_embedding,
            memory,
            count=count,
            threshold=batch.get("threshold", 0.5),
            feature_mask=memory_mask,
        )
        anchors = self.relation_anchor_normalizer(
            anchors,
            anchor_mask,
            source_embeddings=memory,
            source_mask=memory_mask,
        )
        if hasattr(self, "relation_anchor_refine"):
            anchors = self.relation_anchor_refine(
                anchors,
                memory,
                token_mask=memory_mask,
                query_mask=anchor_mask,
                position_encoding=self.relation_anchor_refine_positions,
                self_attention_bias_module=self.relation_self_attention_bias,
                cross_attention_bias_module=self.relation_cross_attention_bias,
            )
        return anchors, anchor_mask

    def _score_pairs_and_relations(
        self,
        anchors,
        anchor_mask,
        relation_embeddings,
        relation_mask,
        entity_representations,
        entity_mask,
    ):
        batch_size, anchor_count, hidden_size = anchors.shape
        endpoint_queries = self.endpoint_query(anchors).reshape(
            batch_size,
            anchor_count,
            2,
            hidden_size,
        )
        assignment_logits = torch.einsum(
            "BNKD,BED->BNEK",
            endpoint_queries,
            entity_representations,
        ) / self.logit_scale

        assignment_valid = (
            anchor_mask.bool().unsqueeze(2)
            & entity_mask.bool().unsqueeze(1)
        )
        masked_assignment_logits = assignment_logits.masked_fill(
            ~assignment_valid.unsqueeze(-1),
            torch.finfo(assignment_logits.dtype).min,
        )
        endpoint_weights = torch.softmax(masked_assignment_logits, dim=2)
        endpoint_weights = endpoint_weights * assignment_valid.unsqueeze(-1).to(
            endpoint_weights.dtype
        )
        endpoint_weights = endpoint_weights / endpoint_weights.sum(
            dim=2,
            keepdim=True,
        ).clamp(min=torch.finfo(endpoint_weights.dtype).eps)

        endpoint_representations = torch.einsum(
            "BNEK,BED->BNKD",
            endpoint_weights,
            entity_representations,
        )
        directed_endpoints = torch.cat(
            [
                endpoint_representations[:, :, 0],
                endpoint_representations[:, :, 1],
            ],
            dim=-1,
        )
        pair_representations = self.pair_fusion(directed_endpoints)
        pair_representations = pair_representations * anchor_mask.unsqueeze(-1).to(
            pair_representations.dtype
        )
        relation_logits = torch.einsum(
            "BND,BRD->BNR",
            pair_representations,
            relation_embeddings,
        ) / self.logit_scale

        relation_valid = (
            anchor_mask.bool().unsqueeze(2)
            & relation_mask.bool().unsqueeze(1)
        )
        relation_logits = relation_logits * relation_valid.to(relation_logits.dtype)
        assignment_logits = assignment_logits * assignment_valid.unsqueeze(-1).to(
            assignment_logits.dtype
        )
        return (
            relation_logits,
            assignment_logits,
            endpoint_weights,
            endpoint_representations,
            pair_representations,
        )

    @staticmethod
    def _gold_anchor_mask(labels, label_count):
        return gold_anchor_mask(labels, label_count)

    def _match_anchors(
        self,
        relation_logits,
        assignment_logits,
        relation_labels,
        assignment_labels,
        anchor_mask,
        gold_mask,
        relation_mask,
        entity_mask,
        base_loss_fn,
    ):
        def pair_cost(batch_idx, prediction_ids, gold_ids):
            relation_pred = relation_logits[batch_idx, prediction_ids]
            endpoint_pred = assignment_logits[batch_idx, prediction_ids]
            relation_gold = relation_labels[batch_idx, gold_ids]
            endpoint_gold = assignment_labels[batch_idx, gold_ids]
            relation_cost_mask = relation_mask[batch_idx].to(
                relation_pred.dtype
            )
            endpoint_cost_mask = entity_mask[batch_idx].to(
                endpoint_pred.dtype
            )[:, None]

            paired_relation_pred = relation_pred[:, None].expand(
                -1, relation_gold.shape[0], -1
            )
            paired_relation_gold = relation_gold[None].expand(
                relation_pred.shape[0], -1, -1
            )
            paired_endpoint_pred = endpoint_pred[:, None].expand(
                -1, endpoint_gold.shape[0], -1, -1
            )
            paired_endpoint_gold = endpoint_gold[None].expand(
                endpoint_pred.shape[0], -1, -1, -1
            )

            relation_cost = (
                base_loss_fn(paired_relation_pred, paired_relation_gold)
                * relation_cost_mask[None, None, :]
            ).sum(dim=-1)
            endpoint_cost = (
                base_loss_fn(paired_endpoint_pred, paired_endpoint_gold)
                * endpoint_cost_mask[None, None, :, :]
            ).sum(dim=(-1, -2))
            cost = relation_cost + endpoint_cost

            if prediction_ids.numel() > gold_ids.numel():
                relation_zero = (
                    base_loss_fn(relation_pred, torch.zeros_like(relation_pred))
                    * relation_cost_mask[None, :]
                ).sum(dim=-1)
                endpoint_zero = (
                    base_loss_fn(endpoint_pred, torch.zeros_like(endpoint_pred))
                    * endpoint_cost_mask[None, :, :]
                ).sum(dim=(-1, -2))
                cost = cost - (relation_zero + endpoint_zero)[:, None]
            return cost

        return batched_masked_assignment(anchor_mask, gold_mask, pair_cost)

    def _assignment_loss(
        self,
        relation_logits,
        assignment_logits,
        relation_labels,
        assignment_labels,
        anchor_mask,
        relation_mask,
        entity_mask,
        label_count,
        base_loss_fn,
    ):
        if relation_labels.dim() != 3:
            raise ValueError(
                "set_open_rel_labels must have shape (BN, G, R)"
            )
        if assignment_labels.dim() != 4 or assignment_labels.shape[-1] != 2:
            raise ValueError(
                "set_open_rel_assignment_labels must have shape (BN, G, E, 2)"
            )
        if assignment_labels.shape[:2] != relation_labels.shape[:2]:
            raise ValueError("set open relation targets must share (BN, G)")
        if relation_labels.shape[0] != relation_logits.shape[0] or (
            assignment_labels.shape[0] != assignment_logits.shape[0]
        ):
            raise ValueError("set open relation targets and logits must share BN")
        relation_count = min(relation_logits.shape[2], relation_labels.shape[2])
        entity_count = min(assignment_logits.shape[2], assignment_labels.shape[2])
        public_pred = relation_logits[:, :, :relation_count]
        endpoint_pred = assignment_logits[:, :, :entity_count]
        public_gold = relation_labels[:, :, :relation_count]
        endpoint_gold = assignment_labels[:, :, :entity_count]
        gold_mask = self._gold_anchor_mask(public_gold, label_count)
        slot_capacity = relation_logits.shape[1]
        gold_counts = gold_mask.sum(dim=1)
        active_slot_counts = anchor_mask[:, :slot_capacity].bool().sum(dim=1)
        overflow = gold_counts > active_slot_counts
        if overflow.any() and not self._relation_capacity_warning_emitted:
            overflow_idx = (gold_counts - active_slot_counts).argmax()
            gold_count = int(gold_counts[overflow_idx].item())
            active_slots = int(active_slot_counts[overflow_idx].item())
            warnings.warn(
                "Set open relation gold count exceeds the active "
                f"relation-slot capacity ({gold_count} gold pairs, "
                f"{active_slots} slots active out of {slot_capacity}). "
                "Hungarian matching will supervise at most the active slot "
                "count per group; unmatched gold pairs are ignored. Increase "
                "anchor_layer.params.num_slots or choose an anchor layer that "
                "activates enough slots to cover every pair.",
                UserWarning,
                stacklevel=2,
            )
            self._relation_capacity_warning_emitted = True

        matches = self._match_anchors(
            public_pred,
            endpoint_pred,
            public_gold,
            endpoint_gold,
            anchor_mask,
            gold_mask,
            relation_mask[:, :relation_count],
            entity_mask[:, :entity_count],
            base_loss_fn,
        )
        public_targets = matched_anchor_targets(
            public_pred,
            public_gold,
            matches,
        )
        endpoint_targets = matched_anchor_targets(
            endpoint_pred,
            endpoint_gold,
            matches,
        )

        public_losses = base_loss_fn(public_pred, public_targets)
        endpoint_losses = base_loss_fn(endpoint_pred, endpoint_targets)
        public_mask = (
            anchor_mask.to(public_losses.dtype)[:, :, None]
            * relation_mask[:, None, :relation_count].to(public_losses.dtype)
        )
        endpoint_mask = (
            anchor_mask.to(endpoint_losses.dtype)[:, :, None, None]
            * entity_mask[:, None, :entity_count, None].to(endpoint_losses.dtype)
        )
        public_total = (public_losses * public_mask).sum()
        endpoint_total = (endpoint_losses * endpoint_mask).sum()
        if self.bio_loss_reduction == "mean":
            public_denominator = public_mask.expand_as(public_losses).sum().clamp(min=1)
            endpoint_denominator = endpoint_mask.expand_as(endpoint_losses).sum().clamp(min=1)
            public_total = public_total / public_denominator
            endpoint_total = endpoint_total / endpoint_denominator
        return public_total + endpoint_total, matches

    @staticmethod
    def _objectness_loss(logits, matches, anchor_mask, base_loss_fn=None):
        return matched_objectness_loss(
            logits,
            matches,
            anchor_mask,
            loss_fn=base_loss_fn,
        )

    def _reduce_entity_loss(self, loss, word_mask, relation_mask):
        if loss is None or self.bio_loss_reduction == "sum":
            return loss
        per_row_elements = (
            word_mask.to(loss.dtype).sum(dim=1)
            * relation_mask.to(loss.dtype).sum(dim=1)
            * 3
        )
        return loss / per_row_elements.sum().clamp(min=1)

    def forward(
        self,
        shared,
        dependency_outputs,
        flat_inputs=None,
        base_loss_fn=None,
        **batch,
    ):
        entity_labels = batch.get("set_open_rel_entity_labels")
        span_idx = batch.get("set_open_rel_span_idx")
        span_mask = batch.get("set_open_rel_span_mask")
        relation_labels = batch.get("set_open_rel_labels")
        assignment_labels = batch.get("set_open_rel_assignment_labels")

        entity_output = super().forward(
            shared,
            dependency_outputs,
            flat_inputs=flat_inputs,
            base_loss_fn=base_loss_fn,
            ner_labels=entity_labels,
            threshold=batch.get("threshold", 0.5),
        )
        if entity_output.logits is None:
            return entity_output

        if span_idx is None or span_mask is None:
            if entity_labels is not None:
                batch_size = entity_output.logits.shape[0]
                span_idx = torch.zeros(
                    batch_size,
                    1,
                    2,
                    dtype=torch.long,
                    device=entity_output.logits.device,
                )
                span_mask = torch.zeros(
                    batch_size,
                    1,
                    dtype=torch.bool,
                    device=entity_output.logits.device,
                )
            else:
                span_idx, span_mask = extract_spans_from_tokens(
                    entity_output.logits,
                    labels=None,
                    threshold=batch.get("threshold", 0.5),
                )

        words_embedding = entity_output.extra.get(
            "words_embedding", flat_inputs.words_embedding
        )
        span_idx, span_mask = self._sanitize_spans(
            span_idx,
            span_mask,
            words_embedding.shape[1],
            word_mask=entity_output.extra.get("mask"),
        )
        span_idx, span_mask, assignment_labels = self._coalesce_spans(
            span_idx,
            span_mask,
            assignment_labels,
        )
        entity_representations = self._pool_entity_spans(
            words_embedding,
            span_idx,
            span_mask,
        )
        anchors, anchor_mask = self._relation_anchors(
            flat_inputs,
            entity_representations,
            span_mask,
            batch,
        )
        (
            relation_logits,
            assignment_logits,
            endpoint_weights,
            endpoint_representations,
            pair_representations,
        ) = (
            self._score_pairs_and_relations(
                anchors,
                anchor_mask,
                flat_inputs.child_embedding,
                flat_inputs.child_mask,
                entity_representations,
                span_mask,
            )
        )

        objectness_logits = None
        if self.use_anchor_objectness:
            objectness_logits = self.objectness_head(anchors).squeeze(-1)

        entity_loss = self._reduce_entity_loss(
            entity_output.loss,
            entity_output.extra.get("mask", flat_inputs.mask),
            flat_inputs.child_mask,
        )
        combined_loss = None
        if entity_loss is not None:
            combined_loss = self.entity_loss_coef * entity_loss
        assignment_loss = None
        objectness_loss = None
        matches = None
        if (
            relation_labels is not None
            and assignment_labels is not None
            and base_loss_fn is not None
        ):
            assignment_loss, matches = self._assignment_loss(
                relation_logits,
                assignment_logits,
                relation_labels,
                assignment_labels,
                anchor_mask,
                flat_inputs.child_mask,
                span_mask,
                batch.get("set_open_rel_count"),
                base_loss_fn,
            )
            weighted_assignment = self.assignment_loss_coef * assignment_loss
            combined_loss = (
                weighted_assignment
                if combined_loss is None
                else combined_loss + weighted_assignment
            )
            if objectness_logits is not None:
                objectness_loss = self._objectness_loss(
                    objectness_logits,
                    matches,
                    anchor_mask,
                    base_loss_fn=base_loss_fn,
                )
                combined_loss = combined_loss + (
                    self.anchor_objectness_loss_coef * objectness_loss
                )

        extra = {
            "entity_logits": entity_output.logits,
            "entity_spans": span_idx,
            "entity_mask": span_mask,
            "entity_representations": entity_representations,
            "anchors": anchors,
            "modeled_anchors": pair_representations,
            "pair_representations": pair_representations,
            "anchor_mask": anchor_mask,
            "objectness_logits": objectness_logits,
            "assignment_logits": assignment_logits,
            "endpoint_weights": endpoint_weights,
            "endpoint_representations": endpoint_representations,
            "span_logits": assignment_logits,
            "span_idx": span_idx,
            "span_mask": span_mask,
            "entity_loss": (
                entity_loss.detach() if entity_loss is not None else None
            ),
            "assignment_loss": (
                assignment_loss.detach() if assignment_loss is not None else None
            ),
            "objectness_loss": (
                objectness_loss.detach() if objectness_loss is not None else None
            ),
            "anchor_matches": matches,
        }
        return TaskHeadOutput(
            loss=combined_loss,
            logits=relation_logits,
            extra=extra,
        )


SetOpenRelationExtractionHead = SetOpenRelexHead


__all__ = ["SetOpenRelexHead", "SetOpenRelationExtractionHead"]
