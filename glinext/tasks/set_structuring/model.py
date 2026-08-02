"""Independent entity-first set structuring head."""

from dataclasses import replace

import torch
from gliner.modeling.multitask.relations_layers import RelationsRepLayer
from gliner.modeling.span_rep import SpanRepLayer
from gliner.modeling.utils import extract_spans_from_tokens

from ...layers.mlp import create_mlp
from .. import TaskHeadOutput
from ..losses import binary_focal_or_bce
from ..matcher import minimum_cost_assignment
from ..ner.model import NERHead
from ..structuring.anchor_relations import anchor_relation_loss
from ..structuring.model import validate_structuring_anchor_capacity


class SetStructuringHead(NERHead):
    """Compose classical field NER with span-to-record classification.

    This follows the same composition used by :class:`JointRelexHead`: the
    inherited NER forward pass is a complete first stage, concrete entity
    spans are selected from its output, and only those pooled entities enter
    the task-specific second stage.

    The public score tensors intentionally remain separate:

    * ``output.logits`` is ``(BN, L, C, 3)`` entity-extraction BIO logits.
    * ``output.extra["entity_field_logits"]`` is ``(BN, E, C)`` and is
      derived directly from the first-stage BIO logits for the selected spans.
    * ``output.extra["structuring_logits"]`` is ``(BN, A, E)`` span-to-record
      membership logits computed from pooled spans and refined record anchors.

    Here ``A`` is the record-slot count, ``E`` the extracted entity count,
    and ``C`` the field count. Record anchors never participate in entity
    extraction or field classification.
    """

    name = "set_structuring"
    dependencies = []

    def __init__(self, config, hidden_size, dropout, shared_layers=None):
        set_cfg = config.set_structuring_config
        shared_layers = shared_layers or {}

        # NER must always have exactly one parent anchor.  Record-query
        # settings belong exclusively to the second stage.
        entity_cfg = replace(
            set_cfg,
            anchor_mode="parent",
            anchor_layer=None,
            anchor_normalization="none",
            anchor_refine_layers=0,
            anchor_refinement="none",
            anchor_memory_position="none",
            anchor_query_position="none",
            anchor_self_attention_bias="none",
            anchor_cross_attention_bias="none",
            represent_spans=False,
        )
        entity_shared_layers = {
            name: layer
            for name, layer in shared_layers.items()
            if name == "anchor_modeling"
        }
        super().__init__(
            config,
            hidden_size,
            dropout,
            shared_layers=entity_shared_layers,
            task_config=entity_cfg,
        )

        self.entity_loss_coef = set_cfg.entity_loss_coef
        self.assignment_loss_coef = set_cfg.span_loss_coef
        self.bio_loss_reduction = set_cfg.bio_loss_reduction

        self.record_anchor_layer = self._build_anchor_layer(
            set_cfg,
            hidden_size,
            dropout,
        )
        self.record_anchor_normalizer = self._build_anchor_normalizer(
            set_cfg,
            hidden_size,
        )
        self._record_uses_fixed_slots = hasattr(
            self.record_anchor_layer,
            "num_slots",
        )

        refinement = self._build_anchor_refinement(
            set_cfg,
            hidden_size,
            dropout,
            shared_layers,
        )
        if refinement is not None:
            self.record_anchor_refine = refinement
        self.record_anchor_refine_positions = None
        self.record_self_attention_bias = None
        self.record_cross_attention_bias = None
        if hasattr(self, "record_anchor_refine"):
            self.record_anchor_refine_positions = (
                self._build_refinement_positions(
                    set_cfg,
                    hidden_size,
                )
            )
            (
                self.record_self_attention_bias,
                self.record_cross_attention_bias,
            ) = self._build_refinement_biases(
                set_cfg,
                num_heads=self.record_anchor_refine.num_heads,
            )

        # As in joint relation extraction, the second stage always needs a
        # token-level representation for each selected entity regardless of
        # whether the generic NER auxiliary span loss is enabled.
        self.entity_span_rep_layer = SpanRepLayer(
            span_mode="token_level",
            hidden_size=hidden_size,
            max_width=getattr(config, "max_width", 12),
            dropout=dropout,
        )

        self.use_anchor_objectness = set_cfg.anchor_objectness
        self.anchor_objectness_loss_coef = (
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

        self.multi_level = bool(getattr(set_cfg, "multi_level", False))
        self.anchor_relations_loss_coef = getattr(
            set_cfg, "anchor_relations_loss_coef", 1.0,
        )
        if self.multi_level:
            self.anchor_relations_rep_layer = RelationsRepLayer(
                in_dim=hidden_size,
                relation_mode=set_cfg.anchor_relations_layer,
                hidden_dim=hidden_size,
            )

    @classmethod
    def from_config(cls, config, shared_layers=None, **kwargs):
        if config.set_structuring_config is None:
            return None
        return cls(
            config,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            shared_layers=shared_layers,
        )

    @staticmethod
    def _entity_token_labels(structuring_labels):
        if structuring_labels is None:
            return None
        # A field mention is an entity regardless of which gold record owns
        # it.  Collapse only the record axis, preserving field and BIO axes.
        return structuring_labels.amax(dim=1)

    def _record_anchors(self, flat_inputs, batch):
        feature_embeddings = flat_inputs.words_embedding
        feature_mask = self._fit_feature_mask(
            flat_inputs.mask,
            feature_embeddings.shape[1],
        )
        count = None
        if not self._record_uses_fixed_slots:
            count = batch.get("structuring_count")
            if count is None:
                count = batch.get("count_val")

        anchors, anchor_mask = self.record_anchor_layer(
            flat_inputs.parent_embedding,
            feature_embeddings,
            count=count,
            threshold=batch.get("threshold", 0.5),
            feature_mask=feature_mask,
        )
        anchors = self.record_anchor_normalizer(
            anchors,
            anchor_mask,
            source_embeddings=feature_embeddings,
            source_mask=feature_mask,
        )
        if hasattr(self, "record_anchor_refine"):
            anchors = self.record_anchor_refine(
                anchors,
                feature_embeddings,
                token_mask=feature_mask,
                query_mask=anchor_mask,
                position_encoding=self.record_anchor_refine_positions,
                self_attention_bias_module=self.record_self_attention_bias,
                cross_attention_bias_module=self.record_cross_attention_bias,
            )
        return anchors, anchor_mask

    def _pool_entity_spans(self, words_embedding, span_idx, span_mask):
        """Represent selected entities through the relex token span layer."""

        if words_embedding.shape[1] == 0:
            return words_embedding.new_zeros(
                words_embedding.shape[0],
                span_idx.shape[1],
                words_embedding.shape[-1],
            )
        safe_span_idx = span_idx * span_mask.unsqueeze(-1).long()
        representations = self.entity_span_rep_layer(
            words_embedding,
            safe_span_idx,
        )
        return representations * span_mask.unsqueeze(-1).to(
            representations.dtype
        )

    @staticmethod
    def _coalesce_spans(span_idx, span_mask, span_labels=None):
        """Keep one entity row per boundary and merge all of its targets."""

        if span_idx.shape[:2] != span_mask.shape:
            raise ValueError("span indices and mask must share (BN, E)")
        if (
            span_labels is not None
            and span_labels.shape[:2] != span_idx.shape[:2]
        ):
            raise ValueError("span labels and indices must share (BN, E)")

        unique_per_batch = []
        source_groups = []
        for batch_idx in range(span_idx.shape[0]):
            boundary_to_output = {}
            unique = []
            groups = []
            for entity_idx in torch.where(span_mask[batch_idx].bool())[0].tolist():
                span = tuple(span_idx[batch_idx, entity_idx].tolist())
                output_idx = boundary_to_output.get(span)
                if output_idx is None:
                    output_idx = len(unique)
                    boundary_to_output[span] = output_idx
                    unique.append(span)
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
            if not spans:
                continue
            count = len(spans)
            unique_idx[batch_idx, :count] = span_idx.new_tensor(spans)
            unique_mask[batch_idx, :count] = True
        if span_labels is None:
            return unique_idx, unique_mask, None

        unique_labels = span_labels.new_zeros(
            (span_labels.shape[0], max_entities, *span_labels.shape[2:])
        )
        for batch_idx, groups in enumerate(source_groups):
            for output_idx, source_ids in enumerate(groups):
                unique_labels[batch_idx, output_idx] = span_labels[
                    batch_idx,
                    source_ids,
                ].amax(dim=0)
        return unique_idx, unique_mask, unique_labels

    @classmethod
    def _deduplicate_spans(cls, span_idx, span_mask):
        """Compatibility wrapper for inference-only span deduplication."""

        unique_idx, unique_mask, _ = cls._coalesce_spans(
            span_idx,
            span_mask,
        )
        return unique_idx, unique_mask

    @staticmethod
    def _sanitize_spans(span_idx, span_mask, sequence_length, word_mask=None):
        """Mask invalid span rows before a third-party span gather."""

        if span_idx.shape[:2] != span_mask.shape:
            raise ValueError("span indices and mask must share (BN, E)")
        valid = span_mask.bool()
        starts = span_idx[..., 0]
        ends = span_idx[..., 1]
        valid = (
            valid
            & (starts >= 0)
            & (ends >= starts)
            & (ends < int(sequence_length))
        )
        if word_mask is not None and int(sequence_length) > 0:
            fitted_word_mask = word_mask[:, :sequence_length].bool()
            safe_starts = starts.clamp(min=0, max=sequence_length - 1)
            safe_ends = ends.clamp(min=0, max=sequence_length - 1)
            valid = (
                valid
                & fitted_word_mask.gather(1, safe_starts)
                & fitted_word_mask.gather(1, safe_ends)
            )
        safe_idx = torch.where(
            valid.unsqueeze(-1),
            span_idx,
            torch.zeros_like(span_idx),
        )
        return safe_idx, valid

    @staticmethod
    @torch.no_grad()
    def _span_field_logits(
        entity_logits,
        span_idx,
        span_mask,
        word_mask=None,
    ):
        """Project classical NER BIO logits onto selected entity spans.

        The minimum raw start/end/inside logit is used because sigmoid is
        monotonic. These are exactly the decisions that make a class a valid
        span proposal in ``extract_spans_from_tokens``. Immediate outside
        tokens are useful for ranking overlapping NER spans, but are not an
        additional class-acceptance threshold and therefore stay out of this
        field decision.
        """

        batch_size, sequence_length, class_count, _ = entity_logits.shape
        entity_count = span_idx.shape[1]
        field_logits = entity_logits.new_full(
            (batch_size, entity_count, class_count),
            float("-inf"),
        )
        for batch_idx in range(batch_size):
            for entity_idx in torch.where(span_mask[batch_idx].bool())[0].tolist():
                start = int(span_idx[batch_idx, entity_idx, 0].item())
                end = int(span_idx[batch_idx, entity_idx, 1].item())
                if start < 0 or end < start or end >= sequence_length:
                    continue
                if (
                    word_mask is not None
                    and not word_mask[batch_idx, start:end + 1].bool().all()
                ):
                    continue
                score_parts = [
                    entity_logits[batch_idx, start, :, 0],
                    entity_logits[batch_idx, end, :, 1],
                    entity_logits[batch_idx, start:end + 1, :, 2].amin(dim=0),
                ]
                field_logits[batch_idx, entity_idx] = torch.stack(
                    score_parts,
                    dim=0,
                ).amin(dim=0)
        return field_logits

    @staticmethod
    def _score_anchor_membership(
        entity_representations,
        entity_mask,
        record_anchors,
        anchor_mask,
    ):
        """Classify each pooled entity into each refined record anchor."""

        logits = torch.einsum(
            "BED,BAD->BAE",
            entity_representations,
            record_anchors,
        )
        valid = (
            anchor_mask.bool().unsqueeze(2)
            & entity_mask.bool().unsqueeze(1)
        )
        return logits * valid.to(logits.dtype)

    def _score_anchor_relations(self, record_anchors, anchor_mask):
        """Score only active anchors, then restore the public padded axes."""

        batch_size, anchor_count, hidden_size = record_anchors.shape
        active_rows = torch.where(anchor_mask.bool().any(dim=1))[0]
        if active_rows.numel() == 0:
            return record_anchors.new_zeros(
                batch_size,
                anchor_count,
                anchor_count,
            )

        active_mask = anchor_mask[active_rows].bool()
        compact_count = int(active_mask.sum(dim=1).max().item())
        compact_indices = torch.zeros(
            active_rows.numel(),
            compact_count,
            dtype=torch.long,
            device=record_anchors.device,
        )
        compact_mask = torch.zeros_like(compact_indices, dtype=torch.bool)
        for compact_batch_idx, source_batch_idx in enumerate(
            active_rows.tolist()
        ):
            indices = torch.where(anchor_mask[source_batch_idx].bool())[0]
            compact_indices[compact_batch_idx, :indices.numel()] = indices
            compact_mask[compact_batch_idx, :indices.numel()] = True

        compact_anchors = record_anchors[active_rows].gather(
            1,
            compact_indices.unsqueeze(-1).expand(-1, -1, hidden_size),
        )
        compact_scores = self.anchor_relations_rep_layer(
            compact_anchors,
            compact_mask,
        )
        compact_scores = compact_scores * (
            compact_mask.unsqueeze(1) & compact_mask.unsqueeze(2)
        ).to(compact_scores.dtype)

        # Scatter both compact relation axes back to the stable record-slot
        # indices expected by matching and decoding.
        column_scattered = compact_scores.new_zeros(
            active_rows.numel(),
            compact_count,
            anchor_count,
        ).scatter_add(
            2,
            compact_indices.unsqueeze(1).expand(-1, compact_count, -1),
            compact_scores,
        )
        active_scores = compact_scores.new_zeros(
            active_rows.numel(),
            anchor_count,
            anchor_count,
        ).scatter_add(
            1,
            compact_indices.unsqueeze(-1).expand(-1, -1, anchor_count),
            column_scattered,
        )
        return compact_scores.new_zeros(
            batch_size,
            anchor_count,
            anchor_count,
        ).index_copy(0, active_rows, active_scores)

    @staticmethod
    def _gold_anchor_mask(labels, label_count):
        batch_size, anchor_count = labels.shape[:2]
        if label_count is None:
            return labels.detach().abs().reshape(
                batch_size,
                anchor_count,
                -1,
            ).sum(dim=-1) > 0
        if not torch.is_tensor(label_count):
            label_count = torch.as_tensor(
                label_count,
                device=labels.device,
            )
        if label_count.dim() == 0:
            label_count = label_count.unsqueeze(0).expand(batch_size)
        label_count = label_count.to(labels.device).long().clamp(
            min=0,
            max=anchor_count,
        )
        return torch.arange(
            anchor_count,
            device=labels.device,
        ).unsqueeze(0) < label_count.unsqueeze(1)

    def _match_record_anchors(
        self,
        predictions,
        labels,
        prediction_mask,
        gold_mask,
        entity_mask,
        base_loss_fn,
    ):
        matches = [[] for _ in range(predictions.shape[0])]
        with torch.no_grad():
            for batch_idx in range(predictions.shape[0]):
                prediction_ids = torch.where(prediction_mask[batch_idx])[0]
                gold_ids = torch.where(gold_mask[batch_idx])[0]
                if prediction_ids.numel() == 0 or gold_ids.numel() == 0:
                    continue

                predicted = predictions[batch_idx, prediction_ids]
                gold = labels[batch_idx, gold_ids]
                pair_predictions = predicted[:, None].expand(
                    -1,
                    gold.shape[0],
                    -1,
                )
                pair_gold = gold[None].expand(
                    predicted.shape[0],
                    -1,
                    -1,
                )
                pair_losses = base_loss_fn(pair_predictions, pair_gold)
                pair_cost = (
                    pair_losses * entity_mask[batch_idx][None, None]
                ).sum(dim=-1)

                if prediction_ids.numel() > gold_ids.numel():
                    zero_losses = base_loss_fn(
                        predicted,
                        torch.zeros_like(predicted),
                    )
                    zero_cost = (
                        zero_losses
                        * entity_mask[batch_idx].unsqueeze(0)
                    ).sum(dim=-1)
                    pair_cost = pair_cost - zero_cost[:, None]

                assignment = minimum_cost_assignment(pair_cost)
                pairs = (
                    (prediction_ids[pred], gold_ids[gold])
                    for pred, gold in assignment
                )
                matches[batch_idx].extend(
                    (int(pred.item()), int(gold.item()))
                    for pred, gold in pairs
                )
        return matches

    def _assignment_loss(
        self,
        logits,
        labels,
        anchor_mask,
        entity_mask,
        label_count,
        base_loss_fn,
    ):
        # Predictions: (BN, A_pred, E). Supervision retains its processor-level
        # field axis as (BN, E, A_gold, C); field identity belongs to NER, so
        # collapse only that axis for the membership objective.
        anchor_count = logits.shape[1]
        validate_structuring_anchor_capacity(
            labels,
            label_count,
            anchor_count,
            anchor_dim=2,
            task_name="Set structuring",
        )
        entity_count = min(logits.shape[2], labels.shape[1])
        gold_anchor_count = labels.shape[2]
        predictions = logits[:, :, :entity_count]
        gold = labels[:, :entity_count].amax(dim=-1).permute(0, 2, 1)
        prediction_mask = anchor_mask[:, :anchor_count].bool()
        gold_mask = self._gold_anchor_mask(gold, label_count)
        supervised_entity_mask = entity_mask[:, :entity_count].to(
            predictions.dtype
        )

        matches = self._match_record_anchors(
            predictions,
            gold,
            prediction_mask,
            gold_mask[:, :gold_anchor_count],
            supervised_entity_mask,
            base_loss_fn,
        )
        targets = torch.zeros_like(predictions)
        for batch_idx, batch_matches in enumerate(matches):
            for predicted_anchor, gold_anchor in batch_matches:
                if predicted_anchor < anchor_count:
                    targets[batch_idx, predicted_anchor] = gold[
                        batch_idx,
                        gold_anchor,
                    ]

        losses = base_loss_fn(predictions, targets)
        loss_mask = (
            prediction_mask.to(losses.dtype).unsqueeze(-1)
            * supervised_entity_mask.unsqueeze(1)
        )
        if self.bio_loss_reduction == "mean":
            assignment_loss = (losses * loss_mask).sum() / loss_mask.sum().clamp(
                min=1.0
            )
        else:
            assignment_loss = (losses * loss_mask).sum()
        return assignment_loss, matches, prediction_mask

    def _objectness_loss(
        self,
        logits,
        matches,
        anchor_mask,
        base_loss_fn=None,
    ):
        targets = torch.zeros_like(logits)
        for batch_idx, batch_matches in enumerate(matches):
            for predicted_anchor, _ in batch_matches:
                targets[batch_idx, predicted_anchor] = 1.0
        loss_fn = base_loss_fn or binary_focal_or_bce
        losses = loss_fn(
            logits.float(),
            targets.float(),
        )
        mask = anchor_mask.to(losses.dtype)
        return (losses * mask).sum() / mask.sum().clamp(min=1.0)

    def forward(
        self,
        shared,
        dependency_outputs,
        flat_inputs=None,
        base_loss_fn=None,
        **batch,
    ):
        structuring_labels = batch.get("structuring_labels")
        structuring_span_idx = batch.get("structuring_span_idx")
        structuring_span_mask = batch.get("structuring_span_mask")
        structuring_span_labels = batch.get("structuring_span_labels")

        entity_labels = self._entity_token_labels(structuring_labels)

        # Stage 1: a complete classical NER forward pass. Structuring spans are
        # intentionally not passed into NER; they are teacher-forced candidates
        # for stage 2 in the same way rel_span_idx is used by joint relex.
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

        # Stage 2 starts from explicit entities.  Gold spans preserve target
        # alignment during training; inference mirrors joint relation
        # extraction and obtains them directly from the NER logits.
        if structuring_span_idx is not None and structuring_span_mask is not None:
            entity_span_idx = structuring_span_idx
            entity_mask = structuring_span_mask
        elif structuring_labels is not None:
            # A supervised row can legitimately have no visible entity after
            # prompt/token truncation. Falling back to thresholded predictions
            # here can create O(W^2*C) random span candidates during training.
            batch_size = entity_output.logits.shape[0]
            entity_span_idx = torch.zeros(
                batch_size,
                1,
                2,
                dtype=torch.long,
                device=entity_output.logits.device,
            )
            entity_mask = torch.zeros(
                batch_size,
                1,
                dtype=torch.bool,
                device=entity_output.logits.device,
            )
        else:
            entity_span_idx, entity_mask = extract_spans_from_tokens(
                entity_output.logits,
                labels=None,
                threshold=batch.get("threshold", 0.5),
            )

        words_embedding = entity_output.extra.get(
            "words_embedding",
            flat_inputs.words_embedding,
        )
        entity_span_idx, entity_mask = self._sanitize_spans(
            entity_span_idx,
            entity_mask,
            words_embedding.shape[1],
            word_mask=entity_output.extra.get("mask"),
        )
        (
            entity_span_idx,
            entity_mask,
            structuring_span_labels,
        ) = self._coalesce_spans(
            entity_span_idx,
            entity_mask,
            structuring_span_labels,
        )

        entity_representations = self._pool_entity_spans(
            words_embedding,
            entity_span_idx,
            entity_mask,
        )
        record_anchors, anchor_mask = self._record_anchors(
            flat_inputs,
            batch,
        )
        anchor_relation_scores = None
        if self.multi_level:
            anchor_relation_scores = self._score_anchor_relations(
                record_anchors,
                anchor_mask,
            )
        entity_field_logits = self._span_field_logits(
            entity_output.logits,
            entity_span_idx,
            entity_mask,
            word_mask=entity_output.extra.get("mask"),
        )
        structuring_logits = self._score_anchor_membership(
            entity_representations,
            entity_mask,
            record_anchors,
            anchor_mask,
        )

        objectness_logits = None
        if self.use_anchor_objectness:
            objectness_logits = self.objectness_head(record_anchors).squeeze(-1)

        assignment_loss = None
        objectness_loss = None
        anchor_matches = None
        supervised_anchor_mask = anchor_mask.bool()
        relation_loss = None
        combined_loss = None
        if entity_output.loss is not None:
            combined_loss = self.entity_loss_coef * entity_output.loss

        if structuring_span_labels is not None and base_loss_fn is not None:
            assignment_loss, anchor_matches, supervised_anchor_mask = (
                self._assignment_loss(
                    structuring_logits,
                    structuring_span_labels,
                    anchor_mask,
                    entity_mask,
                    batch.get("structuring_count"),
                    base_loss_fn,
                )
            )
            weighted_assignment_loss = (
                self.assignment_loss_coef * assignment_loss
            )
            combined_loss = (
                weighted_assignment_loss
                if combined_loss is None
                else combined_loss + weighted_assignment_loss
            )
            if objectness_logits is not None:
                objectness_loss = self._objectness_loss(
                    objectness_logits,
                    anchor_matches,
                    supervised_anchor_mask,
                    base_loss_fn=base_loss_fn,
                )
                weighted_objectness_loss = (
                    self.anchor_objectness_loss_coef * objectness_loss
                )
                combined_loss = combined_loss + weighted_objectness_loss

        relation_labels = batch.get("structuring_relation_labels")
        if (
            anchor_relation_scores is not None
            and relation_labels is not None
            and base_loss_fn is not None
        ):
            relation_loss = anchor_relation_loss(
                anchor_relation_scores,
                relation_labels,
                supervised_anchor_mask,
                base_loss_fn=base_loss_fn,
                relation_group_mask=batch.get(
                    "structuring_relation_group_mask"
                ),
                anchor_matches=anchor_matches,
                label_count=batch.get("structuring_count"),
            )
            weighted_relation_loss = (
                self.anchor_relations_loss_coef * relation_loss
            )
            combined_loss = (
                weighted_relation_loss
                if combined_loss is None
                else combined_loss + weighted_relation_loss
            )

        extra = {
            **entity_output.extra,
            "entity_logits": entity_output.logits,
            "entity_spans": entity_span_idx,
            "entity_mask": entity_mask,
            "entity_representations": entity_representations,
            "entity_field_logits": entity_field_logits,
            "entity_anchor_logits": structuring_logits.permute(0, 2, 1),
            "entity_assignment_logits": structuring_logits.permute(0, 2, 1),
            "assignment_logits": structuring_logits,
            "structuring_logits": structuring_logits,
            # Span indices/masks are shared with the composite decoder. The
            # span logits themselves are anchor-membership logits, not field
            # classification logits from a second span classifier.
            "span_logits": structuring_logits,
            "span_idx": entity_span_idx,
            "span_mask": entity_mask,
            "groups_output": record_anchors,
            "anchor_mask": anchor_mask,
            "objectness_logits": objectness_logits,
            "entity_loss": (
                entity_output.loss.detach()
                if entity_output.loss is not None else None
            ),
            "assignment_loss": (
                assignment_loss.detach()
                if assignment_loss is not None else None
            ),
            "objectness_loss": (
                objectness_loss.detach()
                if objectness_loss is not None else None
            ),
            "anchor_matches": anchor_matches,
            "anchor_relation_scores": anchor_relation_scores,
            "anchor_relation_loss": (
                relation_loss.detach() if relation_loss is not None else None
            ),
        }
        return TaskHeadOutput(
            loss=combined_loss,
            logits=entity_output.logits,
            extra=extra,
        )


__all__ = ["SetStructuringHead"]
