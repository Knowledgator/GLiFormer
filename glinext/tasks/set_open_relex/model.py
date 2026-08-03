"""Independent entity-first set-prediction open relation extraction."""

import math
from copy import copy

import torch
from gliner.modeling.span_rep import SpanRepLayer
from gliner.modeling.utils import extract_spans_from_tokens
from torch import nn

from ...layers import AnchorModeling
from ...layers.mlp import create_mlp
from .. import TaskHeadOutput
from ..losses import binary_focal_or_bce
from ..matcher import minimum_cost_assignment
from ..ner.model import NERHead


class SetOpenRelexHead(NERHead):
    """Recognize entities first, then predict an unordered set of relations.

    The first stage is a complete relation-aware NER pass.  Its selected spans
    are pooled before relation-instance queries are generated, refined, and
    modeled with the open-vocabulary relation representations.

    Two score tensors are intentionally exposed:

    * ``logits`` has shape ``(BN, A, R, 2)``.  It predicts whether each query
      expresses each relation with a valid source and target role.
    * ``extra["assignment_logits"]`` has shape ``(BN, A, R, E, 2)``.  The
      additional entity axis is mathematically necessary to identify which of
      the ``E`` recognized spans fills each role.

    ``A`` is a permutation-invariant relation-slot axis.  Hungarian matching
    supervises both tensors jointly and provides the objectness targets.
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
        self.relation_anchor_modeling = shared_layers.get(
            "anchor_modeling"
        ) or AnchorModeling.from_config(
            getattr(set_cfg, "anchor_modeling", "linear"),
            hidden_size,
            dropout=dropout,
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
        self.relation_role_head = nn.Linear(hidden_size, 2)
        self.endpoint_query = nn.Linear(hidden_size, hidden_size * 2)
        self.endpoint_scale = math.sqrt(hidden_size)

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
        if assignment_labels.dim() != 5:
            raise ValueError(
                "set_open_rel_assignment_labels must have shape (BN, G, R, E, 2)"
            )
        if assignment_labels.shape[0] != span_idx.shape[0] or (
            assignment_labels.shape[3] != span_idx.shape[1]
        ):
            raise ValueError(
                "set open relation endpoint labels must share the span entity axis"
            )

        merged = assignment_labels.new_zeros(
            assignment_labels.shape[0],
            assignment_labels.shape[1],
            assignment_labels.shape[2],
            max_entities,
            assignment_labels.shape[4],
        )
        for batch_idx, groups in enumerate(source_groups):
            for output_idx, source_ids in enumerate(groups):
                merged[batch_idx, :, :, output_idx] = assignment_labels[
                    batch_idx, :, :, source_ids
                ].amax(dim=2)
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
            count = batch.get("set_open_rel_count")

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

    def _score_relations_and_endpoints(
        self,
        anchors,
        anchor_mask,
        relation_embeddings,
        relation_mask,
        entity_representations,
        entity_mask,
    ):
        modeled = self.relation_anchor_modeling(
            anchors,
            relation_embeddings,
            anchor_mask=anchor_mask,
            child_mask=relation_mask,
        )
        relation_logits = self.relation_role_head(modeled)
        batch_size, anchor_count, relation_count, hidden_size = modeled.shape
        endpoint_queries = self.endpoint_query(modeled).reshape(
            batch_size,
            anchor_count,
            relation_count,
            2,
            hidden_size,
        )
        assignment_logits = torch.einsum(
            "BARKD,BED->BAREK",
            endpoint_queries,
            entity_representations,
        ) / self.endpoint_scale
        # The public role energy is also a prior on every entity pointer.
        assignment_logits = assignment_logits + relation_logits.unsqueeze(3)

        public_valid = anchor_mask.bool().unsqueeze(2) & relation_mask.bool().unsqueeze(1)
        assignment_valid = public_valid.unsqueeze(3) & entity_mask.bool()[:, None, None, :]
        relation_logits = relation_logits * public_valid.unsqueeze(-1).to(
            relation_logits.dtype
        )
        assignment_logits = assignment_logits * assignment_valid.unsqueeze(-1).to(
            assignment_logits.dtype
        )
        return relation_logits, assignment_logits, modeled

    @staticmethod
    def _gold_anchor_mask(labels, label_count):
        batch_size, gold_count = labels.shape[:2]
        if label_count is None:
            return labels.detach().abs().flatten(start_dim=2).sum(dim=-1) > 0
        if not torch.is_tensor(label_count):
            label_count = torch.as_tensor(label_count, device=labels.device)
        if label_count.dim() == 0:
            label_count = label_count.unsqueeze(0).expand(batch_size)
        label_count = label_count.to(labels.device).long().clamp(min=0, max=gold_count)
        return torch.arange(gold_count, device=labels.device)[None] < label_count[:, None]

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
        matches = [[] for _ in range(relation_logits.shape[0])]
        with torch.no_grad():
            for batch_idx in range(relation_logits.shape[0]):
                prediction_ids = torch.where(anchor_mask[batch_idx].bool())[0]
                gold_ids = torch.where(gold_mask[batch_idx].bool())[0]
                if prediction_ids.numel() == 0 or gold_ids.numel() == 0:
                    continue

                public_pred = relation_logits[batch_idx, prediction_ids]
                endpoint_pred = assignment_logits[batch_idx, prediction_ids]
                public_gold = relation_labels[batch_idx, gold_ids]
                endpoint_gold = assignment_labels[batch_idx, gold_ids]
                public_mask = relation_mask[batch_idx].to(public_pred.dtype)[None, None, :, None]
                endpoint_mask = (
                    relation_mask[batch_idx].bool()[:, None]
                    & entity_mask[batch_idx].bool()[None, :]
                ).to(endpoint_pred.dtype)[None, None, :, :, None]

                paired_public_pred = public_pred[:, None].expand(
                    -1, public_gold.shape[0], -1, -1
                )
                paired_public_gold = public_gold[None].expand(
                    public_pred.shape[0], -1, -1, -1
                )
                paired_endpoint_pred = endpoint_pred[:, None].expand(
                    -1, endpoint_gold.shape[0], -1, -1, -1
                )
                paired_endpoint_gold = endpoint_gold[None].expand(
                    endpoint_pred.shape[0], -1, -1, -1, -1
                )

                public_cost = (
                    base_loss_fn(
                        paired_public_pred,
                        paired_public_gold,
                    )
                    * public_mask
                ).sum(dim=(-1, -2))
                endpoint_cost = (
                    base_loss_fn(
                        paired_endpoint_pred,
                        paired_endpoint_gold,
                    )
                    * endpoint_mask
                ).sum(dim=(-1, -2, -3))
                pair_cost = public_cost + endpoint_cost

                if prediction_ids.numel() > gold_ids.numel():
                    public_zero = (
                        base_loss_fn(public_pred, torch.zeros_like(public_pred))
                        * public_mask.squeeze(0)
                    ).sum(dim=(-1, -2))
                    endpoint_zero = (
                        base_loss_fn(endpoint_pred, torch.zeros_like(endpoint_pred))
                        * endpoint_mask.squeeze(0)
                    ).sum(dim=(-1, -2, -3))
                    pair_cost -= (public_zero + endpoint_zero)[:, None]

                for predicted_idx, gold_idx in minimum_cost_assignment(pair_cost):
                    matches[batch_idx].append(
                        (
                            int(prediction_ids[predicted_idx].item()),
                            int(gold_ids[gold_idx].item()),
                        )
                    )
        return matches

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
        if relation_labels.dim() != 4 or relation_labels.shape[-1] != 2:
            raise ValueError(
                "set_open_rel_labels must have shape (BN, G, R, 2)"
            )
        if assignment_labels.dim() != 5 or assignment_labels.shape[-1] != 2:
            raise ValueError(
                "set_open_rel_assignment_labels must have shape (BN, G, R, E, 2)"
            )
        if assignment_labels.shape[:3] != relation_labels.shape[:3]:
            raise ValueError("set open relation targets must share (BN, G, R)")
        if relation_labels.shape[1] > relation_logits.shape[1]:
            raise ValueError(
                "Set open relation gold count exceeds the configured relation-slot capacity"
            )

        relation_count = min(relation_logits.shape[2], relation_labels.shape[2])
        entity_count = min(assignment_logits.shape[3], assignment_labels.shape[3])
        public_pred = relation_logits[:, :, :relation_count]
        endpoint_pred = assignment_logits[:, :, :relation_count, :entity_count]
        public_gold = relation_labels[:, :, :relation_count]
        endpoint_gold = assignment_labels[:, :, :relation_count, :entity_count]
        gold_mask = self._gold_anchor_mask(public_gold, label_count)

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
        public_targets = torch.zeros_like(public_pred)
        endpoint_targets = torch.zeros_like(endpoint_pred)
        for batch_idx, batch_matches in enumerate(matches):
            for predicted_anchor, gold_anchor in batch_matches:
                public_targets[batch_idx, predicted_anchor] = public_gold[
                    batch_idx, gold_anchor
                ]
                endpoint_targets[batch_idx, predicted_anchor] = endpoint_gold[
                    batch_idx, gold_anchor
                ]

        public_losses = base_loss_fn(public_pred, public_targets)
        endpoint_losses = base_loss_fn(endpoint_pred, endpoint_targets)
        public_mask = (
            anchor_mask.to(public_losses.dtype)[:, :, None, None]
            * relation_mask[:, None, :relation_count, None].to(public_losses.dtype)
        )
        endpoint_mask = (
            public_mask.unsqueeze(3)
            * entity_mask[:, None, None, :entity_count, None].to(endpoint_losses.dtype)
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
        targets = torch.zeros_like(logits)
        for batch_idx, batch_matches in enumerate(matches):
            for predicted_anchor, _ in batch_matches:
                targets[batch_idx, predicted_anchor] = 1.0
        loss_fn = base_loss_fn or binary_focal_or_bce
        losses = loss_fn(logits.float(), targets.float())
        mask = anchor_mask.to(losses.dtype)
        return (losses * mask).sum() / mask.sum().clamp(min=1)

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
        relation_logits, assignment_logits, modeled_anchors = (
            self._score_relations_and_endpoints(
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
            "modeled_anchors": modeled_anchors,
            "anchor_mask": anchor_mask,
            "objectness_logits": objectness_logits,
            "assignment_logits": assignment_logits,
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
