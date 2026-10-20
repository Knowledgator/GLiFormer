"""Entity-first set-prediction structuring head."""

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
from .. import StructuringTaskHeadOutput
from ..losses import binary_focal_or_bce, binary_loss_with_focal_overrides
from ..matcher import (
    batched_masked_assignment,
    gold_anchor_mask,
    matched_anchor_targets,
    matched_objectness_loss,
)
from ..ner.model import NERHead
from .typed import (
    ORIGIN_ANCHOR,
    ORIGIN_REFERENCE,
    TypedFieldModule,
    gold_anchor_entities,
    gold_record_anchors,
    select_inference_anchors,
)

_TYPED_LABEL_KEYS = (
    "structuring_typed_class_targets",
    "structuring_typed_quantity_presence",
)


class StructuringHead(NERHead):
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

    name = "structuring"
    dependencies = []

    def __init__(
        self,
        config,
        hidden_size,
        dropout,
        shared_layers=None,
        ner_head=None,
    ):
        task_cfg = config.structuring_config
        shared_layers = shared_layers or {}
        self._owns_ner_head = not task_cfg.reuse_ner_head

        if self._owns_ner_head:
            # NER must always have exactly one parent anchor. Record-query
            # settings belong exclusively to the second stage.
            entity_cfg = replace(
                task_cfg,
                anchor_mode="parent",
                anchor_layer=None,
                anchor_normalization="none",
                anchor_refine_layers=0,
                anchor_refinement="none",
                anchor_memory_position="none",
                anchor_query_position="none",
                anchor_self_attention_bias="none",
                anchor_cross_attention_bias="none",
                position_bucket_normalization="none",
                position_bucket_attention_bias_type="none",
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
                    "Structuring NER reuse requires an initialized "
                    "standalone NER head"
                )
            # Keep one registration/state-dict/optimizer owner for the shared
            # module. The model already owns it under heads.ner.
            torch.nn.Module.__init__(self)
            self.config = config
            self.loss_coef = task_cfg.loss_coef
            self.__dict__["_reused_ner_head"] = ner_head

        self.entity_loss_coef = task_cfg.entity_loss_coef
        self.assignment_loss_coef = task_cfg.assignment_loss_coef
        self.bio_loss_reduction = task_cfg.bio_loss_reduction
        self.matcher_membership_cost = task_cfg.matcher_membership_cost
        self.matcher_dice_cost = task_cfg.matcher_dice_cost
        self.matcher_objectness_cost = task_cfg.matcher_objectness_cost
        self.matcher_membership_temperature = (
            task_cfg.matcher_membership_temperature
        )
        self.matcher_objectness_temperature = (
            task_cfg.matcher_objectness_temperature
        )
        self.anchor_objectness_threshold = (
            task_cfg.anchor_objectness_threshold
        )
        for component in ("ner", "matching", "objectness"):
            for suffix in ("alpha", "gamma", "prob_margin"):
                name = f"{component}_focal_loss_{suffix}"
                setattr(self, name, getattr(task_cfg, name))

        self.record_anchor_layer = self._build_anchor_layer(
            task_cfg,
            hidden_size,
            dropout,
        )
        self.record_anchor_normalizer = self._build_anchor_normalizer(
            task_cfg,
            hidden_size,
        )
        self._record_uses_fixed_slots = hasattr(
            self.record_anchor_layer,
            "num_slots",
        )

        refinement = self._build_anchor_refinement(
            task_cfg,
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
                    task_cfg,
                    hidden_size,
                )
            )
            (
                self.record_self_attention_bias,
                self.record_cross_attention_bias,
            ) = self._build_refinement_biases(
                task_cfg,
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

        self.use_anchor_objectness = task_cfg.anchor_objectness
        self.anchor_objectness_loss_coef = (
            task_cfg.anchor_objectness_loss_coef
        )
        if self.use_anchor_objectness:
            self.objectness_head = create_mlp(
                input_dim=hidden_size,
                intermediate_dims=[hidden_size],
                output_dim=1,
                dropout=dropout,
                activation="gelu",
            )

        initialize_anchor_relations(self, task_cfg, hidden_size)
        self._initialize_typed_fields(task_cfg, hidden_size, dropout)

    def _initialize_typed_fields(self, task_cfg, hidden_size, dropout):
        self.typed_fields = bool(getattr(task_cfg, "typed_fields", False))
        if not self.typed_fields:
            return
        self.typed_head = TypedFieldModule(
            hidden_size,
            dropout,
            num_bins=task_cfg.typed_num_bins,
            num_heads=task_cfg.typed_readout_heads,
            max_horizon=task_cfg.typed_max_horizon,
        )
        # Token ids are assigned after the tokenizer grows, i.e. after this
        # head is built, so they are read from the config at forward time.
        self._typed_task_config = task_cfg
        self.embed_option_token = bool(task_cfg.embed_option_token)
        self.typed_class_loss_coef = task_cfg.typed_class_loss_coef
        self.typed_quantity_loss_coef = task_cfg.typed_quantity_loss_coef
        self.typed_point_loss_coef = task_cfg.typed_point_loss_coef
        self.typed_presence_loss_coef = task_cfg.typed_presence_loss_coef
        self.typed_class_focal_loss_alpha = task_cfg.typed_class_focal_loss_alpha
        self.typed_class_focal_loss_gamma = task_cfg.typed_class_focal_loss_gamma
        self.typed_class_focal_loss_prob_margin = None

    @classmethod
    def from_config(
        cls,
        config,
        shared_layers=None,
        ner_head=None,
        **kwargs,
    ):
        if config.structuring_config is None:
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
        field_loss_mask=None,
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
        if field_loss_mask is not None:
            ner_kwargs["ner_field_loss_mask"] = field_loss_mask
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
        """Bind one structuring stage's focal-loss controls."""

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
            task_name="Structuring",
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

    def forward(
        self,
        shared,
        dependency_outputs,
        flat_inputs=None,
        base_loss_fn=None,
        **batch,
    ):
        def target(suffix):
            return batch.get(f"structuring_{suffix}")

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
            field_loss_mask=(
                batch.get("structuring_typed_bio_field_mask")
                if entity_labels is not None
                else None
            ),
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
        # Stage 1 is a complete NER pass, and ``NERHead._bio_loss`` already
        # normalizes it over the active (word x field) cells -- for the reused
        # standalone head too. Reducing again here would divide the entity
        # signal a second time and shrink it away from the stage-2 terms.
        entity_loss = entity_output.loss
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

        typed_supervised = self.typed_fields and any(
            batch.get(key) is not None for key in _TYPED_LABEL_KEYS
        )
        if (
            typed_supervised
            and anchor_matches is None
            and base_loss_fn is not None
            and structuring_count is not None
        ):
            # No span targets in this batch, so Hungarian matching did not
            # run. Typed-only records keep their order on the slot axis.
            anchor_matches = [
                [
                    (index, index)
                    for index in range(
                        min(int(structuring_count[group].item()), anchor_mask.shape[1])
                    )
                    if bool(anchor_mask[group, index])
                ]
                for group in range(anchor_mask.shape[0])
            ]
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
                combined_loss = (
                    weighted_objectness_loss
                    if combined_loss is None
                    else combined_loss + weighted_objectness_loss
                )

        typed_extra = {}
        if self.typed_fields:
            typed_loss, typed_extra = self._typed_forward(
                shared,
                flat_inputs,
                batch,
                words_embedding=words_embedding,
                word_mask=entity_output.extra.get("mask"),
                record_anchors=record_anchors,
                anchor_mask=anchor_mask,
                anchor_matches=anchor_matches,
                gold_count=structuring_count,
                num_gold=(
                    structuring_labels.shape[1]
                    if structuring_labels is not None else 0
                ),
                entity_representations=entity_representations,
                entity_field_logits=entity_field_logits,
                membership_logits=structuring_logits,
                entity_mask=entity_mask,
                span_labels=structuring_span_labels,
                base_loss_fn=base_loss_fn,
                threshold=membership_threshold,
                supervised=typed_supervised,
            )
            if typed_loss is not None:
                combined_loss = (
                    typed_loss
                    if combined_loss is None
                    else combined_loss + typed_loss
                )
            if self.training and combined_loss is not None:
                # Batches without some typed kinds leave those parameters
                # without gradients, which DDP rejects unless it searches for
                # unused parameters. A zero-weighted term keeps them attached.
                combined_loss = combined_loss + 0.0 * sum(
                    parameter.sum() for parameter in self.typed_head.parameters()
                )

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
            **typed_extra,
        }
        return StructuringTaskHeadOutput(
            loss=combined_loss,
            logits=entity_output.logits,
            extra=extra,
        )


    # -- typed fields ---------------------------------------------------------

    def _option_stream(self, shared):
        """Every ``[OPTION]`` representation per batch row, in prompt order.

        Returns ``(B, N, D)`` embeddings and ``(B,)`` counts.  With
        ``embed_option_token`` the marker itself is used (bidirectional
        encoders); otherwise the last token of the option label, which is the
        only position a causal decoder lets see the whole label.
        """

        token_embeds = shared.token_embeds
        input_ids = shared.input_ids
        if token_embeds is None or input_ids is None or self.option_token_index < 0:
            return None, None
        attention = (
            shared.attention_mask.bool()
            if shared.attention_mask is not None
            else torch.ones_like(input_ids, dtype=torch.bool)
        )
        markers = (input_ids == self.option_token_index) & attention
        counts = markers.sum(dim=1)
        batch, length, dim = token_embeds.shape
        width = max(int(counts.max().item()) if counts.numel() else 0, 1)
        stream = token_embeds.new_zeros(batch, width, dim)
        if not self.embed_option_token:
            boundary = markers.clone()
            for index in (self.child_token_index_value(), *self.typed_boundary_token_indices):
                if index is not None and index >= 0:
                    boundary |= input_ids == index
            boundary |= ~attention
        for row in range(batch):
            positions = torch.nonzero(markers[row], as_tuple=False).squeeze(-1)
            if positions.numel() == 0:
                continue
            if not self.embed_option_token:
                following = []
                row_boundary = boundary[row]
                for position in positions.tolist():
                    end = position + 1
                    while end < length and not bool(row_boundary[end]):
                        end += 1
                    following.append(max(position, end - 1))
                positions = torch.tensor(following, device=input_ids.device)
            stream[row, : positions.numel()] = token_embeds[row, positions]
        return stream, counts

    def child_token_index_value(self):
        return int(getattr(self.config, "child_token_index", -1) or -1)

    def _typed_config(self):
        struct_cfg = getattr(self.config, "structuring_config", None)
        return struct_cfg if struct_cfg is not None else self._typed_task_config

    @property
    def option_token_index(self):
        return int(getattr(self._typed_config(), "option_token_index", -1))

    @property
    def typed_boundary_token_indices(self):
        return tuple(
            int(index)
            for index in (
                getattr(self._typed_config(), "typed_boundary_token_indices", None) or ()
            )
        )

    def _group_option_embeddings(self, shared, batch, batch_origin):
        option_index = batch.get("structuring_typed_option_index")
        option_mask = batch.get("structuring_typed_option_mask")
        if option_index is None or option_mask is None:
            return None, None
        stream, counts = self._option_stream(shared)
        if stream is None:
            return None, None
        rows = batch_origin.to(stream.device).clamp(0, stream.shape[0] - 1)
        index = option_index.to(stream.device)
        safe = index.clamp(0, stream.shape[1] - 1)
        embeddings = stream[rows[:, None, None], safe]
        # Options cut off by prompt truncation are not supervised or scored.
        visible = index < counts[rows][:, None, None]
        return embeddings, option_mask.bool().to(stream.device) & visible

    @staticmethod
    def _gather_fields(child_embedding, field_idx):
        width = child_embedding.shape[1]
        if width == 0:
            return child_embedding.new_zeros(
                *field_idx.shape, child_embedding.shape[-1]
            )
        index = field_idx.clamp(0, width - 1).unsqueeze(-1).expand(
            -1, -1, child_embedding.shape[-1]
        )
        return child_embedding.gather(1, index)

    @staticmethod
    def _gather_entities(entity_representations, entity_index):
        groups, records, fields = entity_index.shape
        dim = entity_representations.shape[-1]
        if entity_representations.shape[1] == 0:
            return entity_representations.new_zeros(groups, records, fields, dim)
        flat = entity_index.clamp(0, entity_representations.shape[1] - 1).reshape(
            groups, records * fields, 1
        ).expand(-1, -1, dim)
        return entity_representations.gather(1, flat).reshape(
            groups, records, fields, dim
        )

    def _typed_forward(
        self,
        shared,
        flat_inputs,
        batch,
        *,
        words_embedding,
        word_mask,
        record_anchors,
        anchor_mask,
        anchor_matches,
        gold_count,
        num_gold,
        entity_representations,
        entity_field_logits,
        membership_logits,
        entity_mask,
        span_labels,
        base_loss_fn,
        threshold,
        supervised,
    ):
        """Class/quantity predictions and, with targets, their losses."""

        class_field_idx = batch.get("structuring_typed_class_field_idx")
        quantity_field_idx = batch.get("structuring_typed_quantity_field_idx")
        if class_field_idx is None and quantity_field_idx is None:
            return None, {}
        device = record_anchors.device
        if word_mask is None:
            word_mask = flat_inputs.mask
        word_mask = self._fit_feature_mask(word_mask, words_embedding.shape[1]).bool()
        child = flat_inputs.child_embedding
        extra = {}
        total = None

        def add(value, coef):
            nonlocal total
            if value is None:
                return
            weighted = coef * value
            total = weighted if total is None else total + weighted

        if supervised:
            if num_gold == 0:
                return None, {}
            records, record_mask = gold_record_anchors(
                record_anchors, anchor_mask, anchor_matches, num_gold, gold_count,
            )
        else:
            records, record_mask = record_anchors, anchor_mask.bool()

        if class_field_idx is not None:
            class_field_idx = class_field_idx.to(device)
            class_field_mask = batch["structuring_typed_class_field_mask"].to(device).bool()
            fields = self._gather_fields(child, class_field_idx)
            vectors = self.typed_head.record_field_vectors(
                records, fields, words_embedding, word_mask,
            )
            options, option_mask = self._group_option_embeddings(
                shared, batch, flat_inputs.batch_origin,
            )
            if options is not None:
                logits, none_logits = self.typed_head.class_logits(vectors, options)
                field_ok = class_field_mask & option_mask.any(-1)
                if supervised and batch.get("structuring_typed_class_targets") is not None:
                    targets = batch["structuring_typed_class_targets"].to(device)
                    targets = targets[:, :num_gold, : logits.shape[2], : logits.shape[3]]
                    unit = (
                        batch["structuring_typed_class_unit_mask"].to(device).bool()[:, :num_gold]
                        & record_mask[:, :, None]
                        & field_ok[:, None, :]
                    )
                    loss_fn = self._component_loss_fn(
                        base_loss_fn
                        or (lambda x, y: F.binary_cross_entropy_with_logits(
                            x, y, reduction="none"
                        )),
                        "typed_class",
                    )
                    class_loss = self.typed_head.class_loss(
                        logits,
                        none_logits,
                        targets.to(logits.dtype),
                        option_mask,
                        unit,
                        batch["structuring_typed_class_multi"].to(device),
                        loss_fn,
                    )
                    add(class_loss, self.typed_class_loss_coef)
                    extra["typed_class_loss"] = class_loss.detach()
                else:
                    extra["typed_class_logits"] = logits.masked_fill(
                        ~option_mask[:, None], float("-inf")
                    )
                    extra["typed_class_none_logits"] = none_logits
                    extra["typed_class_field_mask"] = field_ok

        if quantity_field_idx is not None:
            quantity_field_idx = quantity_field_idx.to(device)
            quantity_mask = batch["structuring_typed_quantity_field_mask"].to(device).bool()
            bounded = batch["structuring_typed_quantity_bounded"].to(device).bool()
            horizon = batch["structuring_typed_quantity_horizon"].to(device).long()
            fields = self._gather_fields(child, quantity_field_idx)
            vectors = self.typed_head.record_field_vectors(
                records, fields, words_embedding, word_mask,
            )
            if supervised and batch.get("structuring_typed_quantity_presence") is not None:
                origin = batch["structuring_typed_quantity_origin"].to(device)[:, :num_gold]
                anchor_index, anchor_found = gold_anchor_entities(
                    span_labels, quantity_field_idx, num_gold,
                )
                wants_anchor = origin.eq(ORIGIN_ANCHOR)
                anchor_present = anchor_found & wants_anchor & ~bounded[:, None, :]
                anchor_repr = self._gather_entities(entity_representations, anchor_index)
                levels = batch["structuring_typed_quantity_levels"].to(device)[:, :num_gold]
                steps = levels.shape[3]
                logits, offsets, presence_logits = self.typed_head.quantity_outputs(
                    vectors, anchor_repr, anchor_present, origin, steps,
                )
                step_mask = (
                    torch.arange(steps, device=device)[None, None, None, :]
                    < horizon[:, None, :, None]
                ).expand(-1, num_gold, -1, -1)
                unit = (
                    batch["structuring_typed_quantity_presence_mask"].to(device).bool()[:, :num_gold]
                    & record_mask[:, :, None]
                    & quantity_mask[:, None, :]
                    # The target is relative to an anchor the batch lost.
                    & ~(wants_anchor & ~anchor_present)
                )
                dist_loss, point_loss, presence_loss = self.typed_head.quantity_loss(
                    logits,
                    offsets,
                    presence_logits,
                    levels=levels,
                    values=batch["structuring_typed_quantity_values"].to(device)[:, :num_gold],
                    point_mask=batch["structuring_typed_quantity_point_mask"].to(device)[:, :num_gold],
                    point=batch["structuring_typed_quantity_point"].to(device)[:, :num_gold],
                    point_value_mask=batch[
                        "structuring_typed_quantity_point_value_mask"
                    ].to(device)[:, :num_gold],
                    presence=batch["structuring_typed_quantity_presence"].to(device)[:, :num_gold],
                    unit_mask=unit,
                    step_mask=step_mask,
                )
                add(dist_loss, self.typed_quantity_loss_coef)
                add(point_loss, self.typed_point_loss_coef)
                add(presence_loss, self.typed_presence_loss_coef)
                extra.update({
                    "typed_quantity_loss": dist_loss.detach(),
                    "typed_point_loss": point_loss.detach(),
                    "typed_presence_loss": presence_loss.detach(),
                })
            elif not supervised:
                reference = batch["structuring_typed_quantity_reference"].to(device).bool()
                anchor_index, anchor_present = select_inference_anchors(
                    entity_field_logits,
                    membership_logits,
                    entity_mask,
                    quantity_field_idx,
                    float(threshold),
                )
                anchor_present = anchor_present & ~bounded[:, None, :]
                origin = torch.where(
                    anchor_present,
                    torch.full_like(anchor_index, ORIGIN_ANCHOR),
                    torch.where(
                        reference[:, None, :].expand_as(anchor_index),
                        torch.full_like(anchor_index, ORIGIN_REFERENCE),
                        torch.zeros_like(anchor_index),
                    ),
                )
                anchor_repr = self._gather_entities(
                    entity_representations, anchor_index.clamp(min=0),
                )
                steps = int(horizon.max().item()) if horizon.numel() else 1
                logits, offsets, presence_logits = self.typed_head.quantity_outputs(
                    vectors, anchor_repr, anchor_present, origin, steps,
                )
                extra.update({
                    "typed_quantity_probs": torch.softmax(logits.float(), dim=-1),
                    "typed_quantity_offsets": offsets.float(),
                    "typed_quantity_presence_logits": presence_logits,
                    "typed_quantity_anchor_entity": torch.where(
                        anchor_present, anchor_index, torch.full_like(anchor_index, -1)
                    ),
                    "typed_quantity_origin": origin,
                    "typed_quantity_field_mask": quantity_mask,
                })
        return total, extra


__all__ = ["StructuringHead"]
