"""Independent entity-first set structuring head."""

from dataclasses import replace

import torch
import torch.nn.functional as F
from gliner.modeling.span_rep import SpanRepLayer
from gliner.modeling.utils import extract_spans_from_tokens

from ...layers.mlp import create_mlp
from ...layers.structuring_relations import (
    initialize_anchor_relations,
    maybe_anchor_relation_loss,
    score_anchor_relations,
    validate_structuring_anchor_capacity,
)
from ...processing.structuring_compat import legacy_task_value
from .. import SetStructuringTaskHeadOutput
from ..losses import binary_focal_or_bce, binary_loss_with_focal_overrides
from ..matcher import (
    batched_masked_assignment,
    gold_anchor_mask,
    matched_anchor_targets,
    matched_objectness_loss,
)
from ..ner.model import NERHead


class SetStructuringHead(NERHead):
    """Compose classical field NER with span-to-record classification.

    This follows the same composition used by :class:`JointRelexHead`: the
    inherited NER forward pass is a complete first stage, concrete entity
    spans are selected from its output, and only those pooled entities enter
    the task-specific second stage.  When ``reuse_ner_head`` is enabled, the
    first stage uses the standalone NER module's parameters on structuring's
    own field prompts.  It cannot consume the standalone NER output because
    extraction groups and structuring schemas have independent class axes.

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

    def __init__(
        self,
        config,
        hidden_size,
        dropout,
        shared_layers=None,
        ner_head=None,
    ):
        set_cfg = config.set_structuring_config
        shared_layers = shared_layers or {}
        self._owns_ner_head = not set_cfg.reuse_ner_head

        if self._owns_ner_head:
            # NER must always have exactly one parent anchor. Record-query
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
        else:
            if ner_head is None:
                raise ValueError(
                    "Set Structuring NER reuse requires an initialized "
                    "standalone NER head"
                )
            # Keep one registration/state-dict/optimizer owner for the shared
            # module. The model already owns it under heads.ner.
            torch.nn.Module.__init__(self)
            self.config = config
            self.loss_coef = set_cfg.loss_coef
            self.__dict__["_reused_ner_head"] = ner_head

        self.entity_loss_coef = set_cfg.entity_loss_coef
        self.assignment_loss_coef = set_cfg.assignment_loss_coef
        self.bio_loss_reduction = set_cfg.bio_loss_reduction
        self.matcher_membership_cost = set_cfg.matcher_membership_cost
        self.matcher_dice_cost = set_cfg.matcher_dice_cost
        self.matcher_objectness_cost = set_cfg.matcher_objectness_cost
        self.matcher_membership_temperature = (
            set_cfg.matcher_membership_temperature
        )
        self.matcher_objectness_temperature = (
            set_cfg.matcher_objectness_temperature
        )
        self.anchor_objectness_threshold = (
            set_cfg.anchor_objectness_threshold
        )
        for component in ("ner", "matching", "objectness"):
            for suffix in ("alpha", "gamma", "prob_margin"):
                name = f"{component}_focal_loss_{suffix}"
                setattr(self, name, getattr(set_cfg, name))

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

        initialize_anchor_relations(self, set_cfg, hidden_size)

    @classmethod
    def from_config(
        cls,
        config,
        shared_layers=None,
        ner_head=None,
        **kwargs,
    ):
        if config.set_structuring_config is None:
            return None
        return cls(
            config,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            shared_layers=shared_layers,
            ner_head=ner_head,
        )

    def _forward_entity_ner(
        self,
        shared,
        flat_inputs,
        base_loss_fn,
        entity_labels,
        threshold,
        focal_loss_alpha=None,
        focal_loss_gamma=None,
        focal_loss_prob_margin=None,
    ):
        """Run the private or shared NER module on structuring field inputs."""

        base_loss_fn = self._component_loss_fn(
            base_loss_fn,
            "ner",
            focal_loss_alpha=focal_loss_alpha,
            focal_loss_gamma=focal_loss_gamma,
            focal_loss_prob_margin=focal_loss_prob_margin,
        )

        ner_kwargs = {
            "flat_inputs": flat_inputs,
            "base_loss_fn": base_loss_fn,
            "ner_labels": entity_labels,
            "threshold": threshold,
        }
        if self._owns_ner_head:
            return NERHead.forward(
                self,
                shared,
                dependency_outputs={},
                **ner_kwargs,
            )

        reused_ner_head = self.__dict__.get("_reused_ner_head")
        if reused_ner_head is None:
            raise RuntimeError("the reused NER head is no longer available")
        return reused_ner_head(
            shared,
            dependency_outputs={},
            **ner_kwargs,
        )

    def _component_loss_fn(
        self,
        base_loss_fn,
        component,
        *,
        focal_loss_alpha=None,
        focal_loss_gamma=None,
        focal_loss_prob_margin=None,
    ):
        """Bind one set-structuring stage's focal-loss controls."""

        if base_loss_fn is None:
            return None
        resolved = {}
        for suffix, explicit in (
            ("alpha", focal_loss_alpha),
            ("gamma", focal_loss_gamma),
            ("prob_margin", focal_loss_prob_margin),
        ):
            resolved[f"focal_loss_{suffix}"] = (
                getattr(self, f"{component}_focal_loss_{suffix}")
                if explicit is None
                else explicit
            )

        def component_loss_fn(logits, targets):
            return binary_loss_with_focal_overrides(
                base_loss_fn,
                logits,
                targets,
                **resolved,
            )

        return component_loss_fn

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
            count = legacy_task_value(
                batch,
                "set_structuring",
                "count",
                fallback_prefix="structuring",
            )
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

        return score_anchor_relations(
            getattr(self, "anchor_relations_rep_layer", None),
            record_anchors,
            anchor_mask,
            compact=True,
        )

    @staticmethod
    def _gold_anchor_mask(labels, label_count):
        return gold_anchor_mask(labels, label_count)

    @staticmethod
    def _threshold_aware_probability(
        logits,
        threshold,
        temperature,
        *,
        eps=1e-6,
    ):
        """Calibrate logits so the inference threshold maps to probability .5."""

        logits = logits.float()
        threshold = torch.as_tensor(
            threshold,
            dtype=logits.dtype,
            device=logits.device,
        )
        threshold_logit = torch.logit(threshold, eps=eps)
        adjusted_logits = (logits - threshold_logit) / float(temperature)
        return adjusted_logits, adjusted_logits.sigmoid()

    @staticmethod
    def _membership_matching_cost(
        logits,
        labels,
        entity_mask,
        *,
        threshold=0.5,
        eps=1e-6,
    ):
        """Return threshold-centered signed BCE over supervised entities."""

        logits = logits.float()
        pair_logits = logits[:, None].expand(
            -1,
            labels.shape[0],
            -1,
        )
        pair_labels = labels[None].expand(
            logits.shape[0],
            -1,
            -1,
        ).to(pair_logits.dtype)
        bce = F.binary_cross_entropy_with_logits(
            pair_logits,
            pair_labels,
            reduction="none",
        )
        signed_bce = 1.0 - 2.0 * torch.exp(-bce)
        threshold = torch.as_tensor(
            threshold,
            dtype=signed_bce.dtype,
            device=signed_bce.device,
        ).clamp(min=0.0, max=1.0)
        # The raw signed BCE is zero at probability 0.5. Re-scale each side
        # of the configured inference boundary independently so that the
        # boundary remains zero while the excellent/poor endpoints stay -1/1.
        boundary = (2.0 * pair_labels - 1.0) * (1.0 - 2.0 * threshold)
        centered_bce = signed_bce - boundary
        signed_bce = torch.where(
            centered_bce < 0,
            centered_bce / (boundary + 1.0).clamp(min=eps),
            centered_bce / (1.0 - boundary).clamp(min=eps),
        ).clamp(min=-1.0, max=1.0)
        mask = entity_mask.to(signed_bce.dtype)[None, None]
        return (signed_bce * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(
            min=1.0
        )

    @staticmethod
    def _dice_matching_cost(
        probabilities,
        labels,
        entity_mask,
        *,
        eps=1e-6,
    ):
        """Return pairwise soft-Dice cost over supervised entities only."""

        mask = entity_mask.to(probabilities.dtype)
        predicted = probabilities[:, None] * mask[None, None]
        gold = labels[None].to(probabilities.dtype) * mask[None, None]
        intersection = (predicted * gold).sum(dim=-1)
        predicted_sum = predicted.sum(dim=-1)
        gold_sum = gold.sum(dim=-1)
        dice = (2.0 * intersection + eps) / (
            predicted_sum + gold_sum + eps
        )
        return 1.0 - 2.0 * dice

    @staticmethod
    def _objectness_matching_cost(probabilities, gold_count):
        """Broadcast bounded anchor-presence cost across gold records."""

        return (1.0 - 2.0 * probabilities)[:, None].expand(-1, gold_count)

    def _record_anchor_pair_cost(
        self,
        predicted_logits,
        labels,
        entity_mask,
        *,
        objectness_logits=None,
        membership_threshold=0.5,
        objectness_threshold=None,
        eps=1e-6,
    ):
        """Build a bounded train/inference-aligned record matching cost."""

        _, membership_probabilities = (
            self._threshold_aware_probability(
                predicted_logits,
                membership_threshold,
                self.matcher_membership_temperature,
                eps=eps,
            )
        )
        membership_cost = self._membership_matching_cost(
            predicted_logits,
            labels,
            entity_mask,
            threshold=membership_threshold,
            eps=eps,
        )
        dice_cost = self._dice_matching_cost(
            membership_probabilities,
            labels,
            entity_mask,
            eps=eps,
        )

        total_cost = (
            self.matcher_membership_cost * membership_cost
            + self.matcher_dice_cost * dice_cost
        )
        total_weight = self.matcher_membership_cost + self.matcher_dice_cost

        if self.use_anchor_objectness and objectness_logits is not None:
            if objectness_threshold is None:
                objectness_threshold = (
                    self.anchor_objectness_threshold
                    if self.anchor_objectness_threshold is not None
                    else membership_threshold
                )
            _, objectness_probabilities = self._threshold_aware_probability(
                objectness_logits,
                objectness_threshold,
                self.matcher_objectness_temperature,
                eps=eps,
            )
            objectness_cost = self._objectness_matching_cost(
                objectness_probabilities,
                labels.shape[0],
            )
            total_cost = (
                total_cost
                + self.matcher_objectness_cost * objectness_cost
            )
            total_weight += self.matcher_objectness_cost

        return total_cost / max(total_weight, eps)

    def _match_record_anchors(
        self,
        predictions,
        labels,
        prediction_mask,
        gold_mask,
        entity_mask,
        base_loss_fn,
        objectness_logits=None,
        membership_threshold=0.5,
        objectness_threshold=None,
    ):
        def pair_cost(batch_idx, prediction_ids, gold_ids):
            predicted = predictions[batch_idx, prediction_ids]
            gold = labels[batch_idx, gold_ids]
            predicted_objectness = (
                objectness_logits[batch_idx, prediction_ids]
                if objectness_logits is not None
                else None
            )
            return self._record_anchor_pair_cost(
                predicted,
                gold,
                entity_mask[batch_idx],
                objectness_logits=predicted_objectness,
                membership_threshold=membership_threshold,
                objectness_threshold=objectness_threshold,
            )

        return batched_masked_assignment(
            prediction_mask,
            gold_mask,
            pair_cost,
        )

    def _assignment_loss(
        self,
        logits,
        labels,
        anchor_mask,
        entity_mask,
        label_count,
        base_loss_fn,
        objectness_logits=None,
        membership_threshold=0.5,
        objectness_threshold=None,
        focal_loss_alpha=None,
        focal_loss_gamma=None,
        focal_loss_prob_margin=None,
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
        matching_loss_fn = self._component_loss_fn(
            base_loss_fn,
            "matching",
            focal_loss_alpha=focal_loss_alpha,
            focal_loss_gamma=focal_loss_gamma,
            focal_loss_prob_margin=focal_loss_prob_margin,
        )

        matches = self._match_record_anchors(
            predictions,
            gold,
            prediction_mask,
            gold_mask[:, :gold_anchor_count],
            supervised_entity_mask,
            matching_loss_fn,
            objectness_logits=objectness_logits,
            membership_threshold=membership_threshold,
            objectness_threshold=objectness_threshold,
        )
        targets = matched_anchor_targets(predictions, gold, matches)

        losses = matching_loss_fn(predictions, targets)
        # Membership is defined only for matched records. Unmatched active
        # slots remain negative examples for the separate objectness loss.
        matched_anchor_mask = torch.zeros_like(prediction_mask)
        for batch_idx, batch_matches in enumerate(matches):
            for prediction_idx, _ in batch_matches:
                matched_anchor_mask[batch_idx, prediction_idx] = True
        loss_mask = (
            matched_anchor_mask.to(losses.dtype).unsqueeze(-1)
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
        focal_loss_alpha=None,
        focal_loss_gamma=None,
        focal_loss_prob_margin=None,
    ):
        objectness_loss_fn = self._component_loss_fn(
            base_loss_fn or binary_focal_or_bce,
            "objectness",
            focal_loss_alpha=focal_loss_alpha,
            focal_loss_gamma=focal_loss_gamma,
            focal_loss_prob_margin=focal_loss_prob_margin,
        )
        return matched_objectness_loss(
            logits,
            matches,
            anchor_mask,
            loss_fn=objectness_loss_fn,
        )

    def _reduce_entity_loss(self, loss, entity_mask, child_mask):
        """Normalize NER by supervised entities and active field classes."""

        if loss is None or self.bio_loss_reduction == "sum":
            return loss
        entity_count = entity_mask.to(loss.dtype).sum(dim=1)
        class_count = child_mask.to(loss.dtype).sum(dim=1)
        return loss / (entity_count * class_count).sum().clamp(min=1.0)

    def forward(
        self,
        shared,
        dependency_outputs,
        flat_inputs=None,
        base_loss_fn=None,
        **batch,
    ):
        def target(suffix):
            return legacy_task_value(
                batch,
                "set_structuring",
                suffix,
                fallback_prefix="structuring",
            )

        structuring_labels = target("labels")
        structuring_count = target("count")
        structuring_span_idx = target("span_idx")
        structuring_span_mask = target("span_mask")
        structuring_span_labels = target("span_labels")

        entity_labels = self._entity_token_labels(structuring_labels)
        configured_membership_threshold = batch.get("threshold")
        membership_threshold = (
            0.5
            if configured_membership_threshold is None
            else configured_membership_threshold
        )
        objectness_threshold = batch.get("objectness_threshold")
        if objectness_threshold is None:
            if configured_membership_threshold is not None:
                objectness_threshold = configured_membership_threshold
            elif self.anchor_objectness_threshold is not None:
                objectness_threshold = self.anchor_objectness_threshold
            else:
                objectness_threshold = membership_threshold

        # Stage 1: a complete classical NER forward pass. Structuring spans are
        # intentionally not passed into NER; they are teacher-forced candidates
        # for stage 2 in the same way rel_span_idx is used by joint relex.
        entity_output = self._forward_entity_ner(
            shared,
            flat_inputs,
            base_loss_fn,
            entity_labels,
            membership_threshold,
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
        entity_loss = self._reduce_entity_loss(
            entity_output.loss,
            entity_mask,
            flat_inputs.child_mask,
        )
        if entity_loss is not None:
            combined_loss = self.entity_loss_coef * entity_loss

        if structuring_span_labels is not None and base_loss_fn is not None:
            assignment_loss, anchor_matches, supervised_anchor_mask = (
                self._assignment_loss(
                    structuring_logits,
                    structuring_span_labels,
                    anchor_mask,
                    entity_mask,
                    structuring_count,
                    base_loss_fn,
                    objectness_logits=objectness_logits,
                    membership_threshold=membership_threshold,
                    objectness_threshold=objectness_threshold,
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

        relation_loss, weighted_relation_loss = maybe_anchor_relation_loss(
            anchor_relation_scores,
            target("relation_labels"),
            supervised_anchor_mask,
            base_loss_fn=base_loss_fn,
            relation_group_mask=target("relation_group_mask"),
            anchor_matches=anchor_matches,
            label_count=structuring_count,
            loss_coef=self.anchor_relations_loss_coef,
            focal_loss_alpha=self.anchor_relations_focal_loss_alpha,
            focal_loss_gamma=self.anchor_relations_focal_loss_gamma,
            focal_loss_prob_margin=(
                self.anchor_relations_focal_loss_prob_margin
            ),
        )
        if weighted_relation_loss is not None:
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
            # Canonical internal record-membership layout is (BN, A, E).
            "membership_logits": structuring_logits,
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
                entity_loss.detach() if entity_loss is not None else None
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
        return SetStructuringTaskHeadOutput(
            loss=combined_loss,
            logits=entity_output.logits,
            extra=extra,
        )


__all__ = ["SetStructuringHead"]
