"""Independent entity-first set structuring head."""

from dataclasses import replace

import torch
from gliner.modeling.span_rep import SpanRepLayer
from gliner.modeling.utils import extract_spans_from_tokens
from torch import nn

from ...layers.mlp import create_mlp
from .. import TaskHeadOutput
from ..ner.model import NERHead
from ..structuring.model import validate_structuring_anchor_capacity


class SetStructuringHead(NERHead):
    """Extract field entities, then score every entity against record slots.

    This follows the same composition used by :class:`JointRelexHead`: the
    inherited NER forward pass is a complete first stage, concrete entity
    spans are selected from its output, and only those pooled entities enter
    the task-specific second stage.

    The two public score tensors intentionally remain separate:

    * ``output.logits`` is ``(BN, L, C, 3)`` entity-extraction BIO logits.
    * ``output.extra["structuring_logits"]`` is ``(BN, A, E, C)``
      per-entity record/field logits.

    Here ``A`` is the record-slot count, ``E`` the extracted entity count,
    and ``C`` the field count.  Record anchors never participate in the NER
    stage.
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

    @staticmethod
    def _entity_span_labels(structuring_span_labels):
        if structuring_span_labels is None:
            return None
        # (BN, E, A, C) -> (BN, E, C), for the inherited NER span loss.
        return structuring_span_labels.amax(dim=2)

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

    def _pool_entities(self, words_embedding, span_idx, span_mask):
        safe_span_idx = span_idx * span_mask.unsqueeze(-1).long()
        representations = self.entity_span_rep_layer(
            words_embedding,
            safe_span_idx,
        )
        return representations * span_mask.unsqueeze(-1).to(
            representations.dtype
        )

    @staticmethod
    def _score_entities(
        entity_representations,
        entity_mask,
        field_representations,
        field_mask,
        record_anchors,
        anchor_mask,
    ):
        """Return per-entity field, record, and combined logits."""

        entity_field_logits = torch.einsum(
            "BED,BCD->BEC",
            entity_representations,
            field_representations,
        )
        entity_anchor_logits = torch.einsum(
            "BED,BAD->BEA",
            entity_representations,
            record_anchors,
        )
        structuring_logits = (
            entity_anchor_logits.permute(0, 2, 1).unsqueeze(-1)
            + entity_field_logits.unsqueeze(1)
        )

        entity_valid = entity_mask.bool()
        entity_field_logits = entity_field_logits * (
            entity_valid.unsqueeze(-1) & field_mask.bool().unsqueeze(1)
        ).to(entity_field_logits.dtype)
        entity_anchor_logits = entity_anchor_logits * (
            entity_valid.unsqueeze(-1) & anchor_mask.bool().unsqueeze(1)
        ).to(entity_anchor_logits.dtype)
        structuring_logits = structuring_logits * (
            anchor_mask.bool().unsqueeze(2).unsqueeze(3)
            & entity_valid.unsqueeze(1).unsqueeze(3)
            & field_mask.bool().unsqueeze(1).unsqueeze(1)
        ).to(structuring_logits.dtype)
        return (
            entity_field_logits,
            entity_anchor_logits,
            structuring_logits,
        )

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

    @staticmethod
    def _hungarian_rows_to_cols(cost):
        """Minimum-cost assignment for a matrix with rows <= columns."""

        row_count = len(cost)
        column_count = len(cost[0]) if row_count else 0
        if row_count == 0 or column_count == 0:
            return []
        if row_count > column_count:
            raise ValueError("Hungarian assignment requires rows <= columns")

        row_potential = [0.0] * (row_count + 1)
        column_potential = [0.0] * (column_count + 1)
        matching = [0] * (column_count + 1)
        previous = [0] * (column_count + 1)

        for row in range(1, row_count + 1):
            matching[0] = row
            column = 0
            minimum = [float("inf")] * (column_count + 1)
            used = [False] * (column_count + 1)
            while True:
                used[column] = True
                current_row = matching[column]
                delta = float("inf")
                next_column = 0
                for candidate in range(1, column_count + 1):
                    if used[candidate]:
                        continue
                    current = (
                        cost[current_row - 1][candidate - 1]
                        - row_potential[current_row]
                        - column_potential[candidate]
                    )
                    if current < minimum[candidate]:
                        minimum[candidate] = current
                        previous[candidate] = column
                    if minimum[candidate] < delta:
                        delta = minimum[candidate]
                        next_column = candidate
                for candidate in range(column_count + 1):
                    if used[candidate]:
                        row_potential[matching[candidate]] += delta
                        column_potential[candidate] -= delta
                    else:
                        minimum[candidate] -= delta
                column = next_column
                if matching[column] == 0:
                    break

            while True:
                next_column = previous[column]
                matching[column] = matching[next_column]
                column = next_column
                if column == 0:
                    break

        return [
            (matching[column] - 1, column - 1)
            for column in range(1, column_count + 1)
            if matching[column] != 0
        ]

    def _match_record_anchors(
        self,
        predictions,
        labels,
        prediction_mask,
        gold_mask,
        entity_field_mask,
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
                    -1,
                )
                pair_gold = gold[None].expand(
                    predicted.shape[0],
                    -1,
                    -1,
                    -1,
                )
                pair_losses = base_loss_fn(pair_predictions, pair_gold)
                pair_cost = (
                    pair_losses * entity_field_mask[batch_idx][None, None]
                ).sum(dim=(-2, -1))

                if prediction_ids.numel() > gold_ids.numel():
                    zero_losses = base_loss_fn(
                        predicted,
                        torch.zeros_like(predicted),
                    )
                    zero_cost = (
                        zero_losses
                        * entity_field_mask[batch_idx].unsqueeze(0)
                    ).sum(dim=(-2, -1))
                    pair_cost = pair_cost - zero_cost[:, None]

                if prediction_ids.numel() <= gold_ids.numel():
                    assignment = self._hungarian_rows_to_cols(
                        pair_cost.cpu().tolist()
                    )
                    pairs = (
                        (prediction_ids[pred], gold_ids[gold])
                        for pred, gold in assignment
                    )
                else:
                    assignment = self._hungarian_rows_to_cols(
                        pair_cost.t().contiguous().cpu().tolist()
                    )
                    pairs = (
                        (prediction_ids[pred], gold_ids[gold])
                        for gold, pred in assignment
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
        field_mask,
        label_count,
        base_loss_fn,
    ):
        # Predictions: (BN, A_pred, E, C); labels: (BN, E, A_gold, C).
        anchor_count = logits.shape[1]
        validate_structuring_anchor_capacity(
            labels,
            label_count,
            anchor_count,
            anchor_dim=2,
            task_name="Set structuring",
        )
        entity_count = min(logits.shape[2], labels.shape[1])
        class_count = min(logits.shape[3], labels.shape[3])
        gold_anchor_count = labels.shape[2]
        predictions = logits[:, :, :entity_count, :class_count]
        gold = labels[:, :entity_count, :, :class_count].permute(0, 2, 1, 3)
        prediction_mask = anchor_mask[:, :anchor_count].bool()
        gold_mask = self._gold_anchor_mask(gold, label_count)
        entity_field_mask = (
            entity_mask[:, :entity_count].bool().unsqueeze(-1)
            & field_mask[:, :class_count].bool().unsqueeze(1)
        ).to(predictions.dtype)

        matches = self._match_record_anchors(
            predictions,
            gold,
            prediction_mask,
            gold_mask[:, :gold_anchor_count],
            entity_field_mask,
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
            prediction_mask.to(losses.dtype).unsqueeze(-1).unsqueeze(-1)
            * entity_field_mask.unsqueeze(1)
        )
        if self.bio_loss_reduction == "mean":
            assignment_loss = (losses * loss_mask).sum() / loss_mask.sum().clamp(
                min=1.0
            )
        else:
            assignment_loss = (losses * loss_mask).sum()
        return assignment_loss, matches, prediction_mask

    def _objectness_loss(self, logits, matches, anchor_mask):
        targets = torch.zeros_like(logits)
        for batch_idx, batch_matches in enumerate(matches):
            for predicted_anchor, _ in batch_matches:
                targets[batch_idx, predicted_anchor] = 1.0
        losses = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
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
        entity_span_labels = self._entity_span_labels(
            structuring_span_labels
        )

        # Stage 1: a complete, anchor-independent NER forward pass.
        entity_output = super().forward(
            shared,
            dependency_outputs,
            flat_inputs=flat_inputs,
            base_loss_fn=base_loss_fn,
            ner_labels=entity_labels,
            span_idx=structuring_span_idx,
            span_mask=structuring_span_mask,
            span_labels=entity_span_labels,
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
        entity_representations = self._pool_entities(
            words_embedding,
            entity_span_idx,
            entity_mask,
        )
        record_anchors, anchor_mask = self._record_anchors(
            flat_inputs,
            batch,
        )
        (
            entity_field_logits,
            entity_anchor_logits,
            structuring_logits,
        ) = self._score_entities(
            entity_representations,
            entity_mask,
            flat_inputs.child_embedding,
            flat_inputs.child_mask,
            record_anchors,
            anchor_mask,
        )

        objectness_logits = None
        if self.use_anchor_objectness:
            objectness_logits = self.objectness_head(record_anchors).squeeze(-1)

        assignment_loss = None
        objectness_loss = None
        anchor_matches = None
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
                    flat_inputs.child_mask,
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
                )
                weighted_objectness_loss = (
                    self.anchor_objectness_loss_coef * objectness_loss
                )
                combined_loss = combined_loss + weighted_objectness_loss

        extra = {
            **entity_output.extra,
            "entity_logits": entity_output.logits,
            "entity_spans": entity_span_idx,
            "entity_mask": entity_mask,
            "entity_representations": entity_representations,
            "entity_field_logits": entity_field_logits,
            "entity_anchor_logits": entity_anchor_logits,
            "entity_assignment_logits": entity_anchor_logits,
            "assignment_logits": structuring_logits,
            "structuring_logits": structuring_logits,
            # Compatibility names used by span-based structuring consumers.
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
        }
        return TaskHeadOutput(
            loss=combined_loss,
            logits=entity_output.logits,
            extra=extra,
        )


__all__ = ["SetStructuringHead"]
