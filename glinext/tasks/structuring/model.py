"""Structuring task head."""

import logging

import torch

from ...layers.mlp import create_mlp
from ...layers.structuring_relations import (
    initialize_anchor_relations,
    maybe_anchor_relation_loss,
    score_anchor_relations,
    validate_structuring_anchor_capacity,
)
from .. import StructuringTaskHeadOutput, TaskHeadOutput
from ..anchored_extraction import AnchoredSpanExtractionHead
from ..matcher import (
    batched_masked_assignment,
    gold_anchor_mask,
    matched_objectness_loss,
)

logger = logging.getLogger(__name__)

class StructuringHead(AnchoredSpanExtractionHead):
    """Structuring via anchor-based span extraction.

    Generalizes :class:`NERHead` to A >= 1 anchors per instance. Shares the
    :class:`AnchoredSpanExtractionHead` pipeline: flattening the (A, C)
    dimensions produces the same scoring path as NER.

    Loss is permutation-invariant over the anchor dimension when
    ``use_anchor_matching=True`` (default): a Hungarian assignment matches
    each predicted anchor slot to its closest gold instance per sample.
    """

    name = "structuring"
    dependencies = []

    def __init__(self, config, hidden_size, dropout, shared_layers=None):
        struct_cfg = config.structuring_config
        super().__init__(struct_cfg, config, hidden_size, dropout, shared_layers)
        self.child_token_index = struct_cfg.child_token_index
        self.embed_child_token = struct_cfg.embed_child_token
        self.max_count = struct_cfg.max_count
        self.use_anchor_matching = struct_cfg.use_anchor_matching
        # Fixed-slot anchor layers always emit `num_slots` anchors. Passing
        # `count` would mask all but the first N slots, removing the negative
        # supervision Hungarian needs for unmatched slots — so we keep all
        # slots active and let the matcher decide which fire.
        self._anchor_uses_fixed_slots = hasattr(self.anchor_layer, "num_slots")

        # Loss-shaping config
        self.bio_loss_reduction = getattr(struct_cfg, "bio_loss_reduction", "sum")
        self.negatives = getattr(struct_cfg, "negatives", 1.0)
        self.masking_mode = getattr(struct_cfg, "masking", "none")

        # Diagnostic logging
        self.log_loss_stats = getattr(struct_cfg, "log_loss_stats", False)
        self.log_loss_stats_every = max(1, getattr(struct_cfg, "log_loss_stats_every", 50))
        self.register_buffer(
            "_log_step", torch.zeros(1, dtype=torch.long), persistent=False,
        )

        # Anchor objectness head
        self.use_anchor_objectness = getattr(struct_cfg, "anchor_objectness", False)
        self.anchor_objectness_loss_coef = getattr(
            struct_cfg, "anchor_objectness_loss_coef", 1.0,
        )
        self.anchor_objectness_threshold = getattr(
            struct_cfg, "anchor_objectness_threshold", 0.5,
        )
        if self.use_anchor_objectness:
            self.objectness_head = create_mlp(
                input_dim=hidden_size,
                intermediate_dims=[hidden_size],
                output_dim=1,
                dropout=dropout,
                activation="gelu",
            )

        initialize_anchor_relations(self, struct_cfg, hidden_size)

    @classmethod
    def from_config(cls, config, shared_layers=None, **kwargs):
        struct_cfg = config.structuring_config
        if struct_cfg is None:
            return None
        if getattr(struct_cfg, "head_type", "structuring") != cls.name:
            return None
        return cls(config, hidden_size=config.hidden_size, dropout=config.dropout,
                   shared_layers=shared_layers)

    def _compute_structuring_scores(self, flat_inputs, batch):
        """Return the public BIO scores plus implementation-specific values."""

        return self._compute_bio_scores(flat_inputs, batch), {}

    def _span_proposal_tensors(
        self,
        scores,
        structuring_labels,
        prediction_extra,
        anchor_count,
    ):
        """Return the BIO tensors used to propose direct span candidates.

        The regular head proposes spans from its anchor-conditioned scores.
        Alternative implementations can expose an earlier extraction stage
        without changing the public structuring forward contract.
        """

        return scores, None, anchor_count

    def _compute_structuring_span_logits(
        self,
        feature_embeddings,
        span_idx,
        anchors,
        fused_flat,
        dims,
        prediction_extra,
    ):
        """Score supplied spans through the regular fused-anchor path."""

        batch_size, anchor_count, class_count, _ = dims
        return self._span_logits_from_fused(
            feature_embeddings,
            fused_flat,
            batch_size,
            anchor_count,
            class_count,
            span_idx,
        )

    def _anchor_kwargs(self, batch):
        """Pass structuring/count-head-predicted anchor counts when available.

        For fixed-slot anchor layers we deliberately do not pass a count so
        all ``num_slots`` slots stay active. ``structuring_count`` is still
        consumed by the Hungarian matcher's gold-anchor mask.
        """
        kwargs = super()._anchor_kwargs(batch)
        if self._anchor_uses_fixed_slots:
            return kwargs
        count = batch.get("structuring_count")
        if count is None:
            count = batch.get("count_val")
        if count is not None:
            kwargs["count"] = count
        return kwargs

    def forward(self, shared, dependency_outputs, flat_inputs=None,
                base_loss_fn=None, **batch):
        structuring_labels = batch.get("structuring_labels")
        threshold = batch.get("threshold", 0.5)

        span_idx = batch.get("structuring_span_idx")
        span_mask = batch.get("structuring_span_mask")
        span_labels = batch.get("structuring_span_labels")

        if flat_inputs.child_embedding.shape[1] == 0:
            return TaskHeadOutput()

        if (
            structuring_labels is None
            and batch.get("structuring_count") is None
            and batch.get("count_val") is None
            and not self.training
            and not self._anchor_uses_fixed_slots
        ):
            batch["structuring_count"] = flat_inputs.parent_embedding.new_full(
                (flat_inputs.parent_embedding.shape[0],),
                self.max_count,
                dtype=torch.long,
            )

        (
            (scores, anchors, anchor_mask, fused_flat, (B, A, C, L)),
            prediction_extra,
        ) = self._compute_structuring_scores(flat_inputs, batch)
        label_count = batch.get("structuring_count")
        if label_count is None:
            label_count = batch.get("count_val")

        # Anchor-objectness logits (one sigmoid score per anchor slot).
        objectness_logits = None
        if self.use_anchor_objectness:
            objectness_logits = self.objectness_head(anchors).squeeze(-1)  # (BN, A)

        anchor_relation_scores = score_anchor_relations(
            getattr(self, "anchor_relations_rep_layer", None),
            anchors,
            anchor_mask,
        )

        # Optional span-level rescoring. It remains training-only for the
        # regular head; implementations whose primary semantic unit is an
        # extracted entity span may opt into inference via ``span_inference``.
        span_logits_out = None
        if (
            self.represent_spans and hasattr(self, "span_rep_layer")
            and (
                structuring_labels is not None
                or getattr(self, "span_inference", False)
            )
        ):
            proposal_scores, proposal_labels, proposal_anchor_count = (
                self._span_proposal_tensors(
                    scores,
                    structuring_labels,
                    prediction_extra,
                    A,
                )
            )
            span_idx, span_mask = self._maybe_extract_spans(
                proposal_scores,
                A=proposal_anchor_count,
                span_idx=span_idx,
                span_mask=span_mask,
                threshold=threshold,
                labels=proposal_labels,
            )
            if span_idx is not None:
                span_logits_out = self._compute_structuring_span_logits(
                    flat_inputs.words_embedding,
                    span_idx,
                    anchors,
                    fused_flat,
                    (B, A, C, L),
                    prediction_extra,
                )

        loss = None
        loss_stats = None
        anchor_matches = None
        supervised_anchor_mask = anchor_mask.bool()
        relation_loss = None
        if structuring_labels is not None and base_loss_fn is not None:
            if self.use_anchor_matching:
                loss, anchor_matches, supervised_anchor_mask, loss_stats = self._anchor_matched_bio_loss(
                    scores=scores, labels=structuring_labels,
                    anchor_mask=anchor_mask,
                    word_mask=flat_inputs.mask,
                    child_mask=flat_inputs.child_mask,
                    base_loss_fn=base_loss_fn,
                    label_count=label_count,
                )
                if span_labels is not None and span_logits_out is not None:
                    span_loss = self._anchor_matched_span_loss(
                        span_logits=span_logits_out, span_labels=span_labels,
                        anchor_matches=anchor_matches,
                        supervised_anchor_mask=supervised_anchor_mask,
                        span_mask=span_mask,
                        child_mask=flat_inputs.child_mask, base_loss_fn=base_loss_fn,
                    )
                    loss = loss + self.span_loss_coef * span_loss
                if objectness_logits is not None:
                    obj_loss = self._objectness_loss(
                        objectness_logits=objectness_logits,
                        anchor_matches=anchor_matches,
                        supervised_anchor_mask=supervised_anchor_mask,
                        base_loss_fn=base_loss_fn,
                    )
                    loss = loss + self.anchor_objectness_loss_coef * obj_loss
                    if loss_stats is not None:
                        loss_stats["objectness_loss"] = float(obj_loss.detach().item())
            else:
                loss, loss_stats = self._bio_loss_with_stats(
                    scores=scores, labels=structuring_labels,
                    anchor_mask=anchor_mask,
                    word_mask=flat_inputs.mask,
                    child_mask=flat_inputs.child_mask,
                    base_loss_fn=base_loss_fn,
                    label_count=label_count,
                )
                if span_labels is not None and span_logits_out is not None:
                    span_loss = self._span_loss(
                        span_logits=span_logits_out, span_labels=span_labels,
                        anchor_mask=anchor_mask, span_mask=span_mask,
                        child_mask=flat_inputs.child_mask, base_loss_fn=base_loss_fn,
                    )
                    loss = loss + self.span_loss_coef * span_loss
                if objectness_logits is not None:
                    # Without matching, supervise objectness against the
                    # heuristic gold-anchor mask (slots < label_count).
                    if label_count is not None:
                        gold_mask = self._label_anchor_mask(
                            structuring_labels, label_count,
                        )
                        obj_loss = self._objectness_loss(
                            objectness_logits=objectness_logits,
                            anchor_matches=None,
                            supervised_anchor_mask=anchor_mask.bool(),
                            gold_mask=gold_mask,
                            base_loss_fn=base_loss_fn,
                        )
                        loss = loss + self.anchor_objectness_loss_coef * obj_loss
                        if loss_stats is not None:
                            loss_stats["objectness_loss"] = float(obj_loss.detach().item())

        relation_loss, weighted_relation_loss = maybe_anchor_relation_loss(
            anchor_relation_scores,
            batch.get("structuring_relation_labels"),
            anchor_mask.bool(),
            base_loss_fn=base_loss_fn,
            relation_group_mask=batch.get(
                "structuring_relation_group_mask"
            ),
            anchor_matches=anchor_matches,
            label_count=label_count,
            loss_coef=self.anchor_relations_loss_coef,
        )
        if weighted_relation_loss is not None:
            loss = (
                weighted_relation_loss
                if loss is None
                else loss + weighted_relation_loss
            )
            if loss_stats is not None:
                loss_stats["anchor_relation_loss"] = float(
                    relation_loss.detach().item()
                )

        if loss_stats is not None and self.log_loss_stats and self.training:
            self._maybe_log_stats(loss_stats)

        output_extra = {
            "groups_output": anchors,
            "anchor_mask": anchor_mask,
            "objectness_logits": objectness_logits,
            "span_logits": span_logits_out,
            "span_idx": span_idx,
            "span_mask": span_mask,
            "loss_stats": loss_stats,
            "anchor_relation_scores": anchor_relation_scores,
            "anchor_relation_loss": (
                relation_loss.detach() if relation_loss is not None else None
            ),
            "anchor_matches": anchor_matches,
        }
        output_extra.update(prediction_extra)

        return StructuringTaskHeadOutput(
            loss=loss,
            logits=scores,
            extra=output_extra,
        )

    # ── Loss reduction & negative masking ────────────────────────────

    def _build_negative_mask(self, labels):
        """GLiNER-style negative sampling mask, adapted to BIO/span shapes.

        Positives (``labels > 0``) are always kept (mask=1). Negatives are
        sampled or weighted by ``self.negatives`` according to
        ``self.masking_mode``:

        - ``"none"``    → mask is 1 everywhere (no sampling).
        - ``"global"``  → element-wise Bernoulli over labels==0.
        - ``"global_weighted"`` → retain every negative with weight
                          ``self.negatives`` instead of stochastic sampling.
        - ``"label"``   → drop negatives only at (anchor, field) cells whose
                          labels sum to 0 across the (sequence, BIO) dims.
        - ``"span"``    → drop negatives only at (anchor, token) positions
                          whose labels sum to 0 across the (field, BIO) dims.
        - ``"anchor"``  → drop negatives only for anchor slots that have no
                          positives anywhere — the structuring use-case.

        Always returns a tensor with the same shape as ``labels`` (or
        ``None`` when no sampling is needed, which keeps the fast path).
        """
        if self.masking_mode == "none" or self.negatives >= 1.0:
            return None

        keep = float(self.negatives)
        labels_pos = labels > 0  # treat any non-zero as positive
        if self.masking_mode == "global_weighted":
            return torch.where(
                labels_pos,
                torch.ones_like(labels),
                torch.full_like(labels, keep),
            )

        rand = torch.rand_like(labels)
        sampled = (rand < keep).to(labels.dtype)

        if self.masking_mode == "global":
            return torch.where(labels_pos, torch.ones_like(labels), sampled)

        if self.masking_mode == "anchor":
            # labels: (BN, A, L, C, 3) for BIO, (BN, A, S, C) for span-level.
            # Pure-negative anchor: no positives across (L, C, 3) / (S, C).
            collapse_dims = tuple(range(2, labels.dim()))
            pure_neg_anchor = labels_pos.float().sum(dim=collapse_dims) == 0  # (BN, A)
            shape = [1] * labels.dim()
            shape[0] = labels.shape[0]
            shape[1] = labels.shape[1]
            pure_neg_anchor = pure_neg_anchor.view(shape).expand_as(labels)
            return torch.where(pure_neg_anchor, sampled, torch.ones_like(labels))

        if self.masking_mode == "label":
            # (BN, A, L, C, 3): collapse over (L, BIO) → keep (BN, A, C).
            if labels.dim() == 5:
                pure_neg = labels_pos.float().sum(dim=(2, 4)) == 0  # (BN, A, C)
                shape = [labels.shape[0], labels.shape[1], 1, labels.shape[3], 1]
                pure_neg = pure_neg.view(shape).expand_as(labels)
            elif labels.dim() == 4:
                pure_neg = labels_pos.float().sum(dim=2) == 0       # (BN, A, C)
                shape = [labels.shape[0], labels.shape[1], 1, labels.shape[3]]
                pure_neg = pure_neg.view(shape).expand_as(labels)
            else:
                return None
            return torch.where(pure_neg, sampled, torch.ones_like(labels))

        if self.masking_mode == "span":
            # (BN, A, L, C, 3): collapse over (C, BIO) → keep (BN, A, L).
            if labels.dim() == 5:
                pure_neg = labels_pos.float().sum(dim=(3, 4)) == 0  # (BN, A, L)
                shape = [labels.shape[0], labels.shape[1], labels.shape[2], 1, 1]
                pure_neg = pure_neg.view(shape).expand_as(labels)
            elif labels.dim() == 4:
                pure_neg = labels_pos.float().sum(dim=3) == 0       # (BN, A, S)
                shape = [labels.shape[0], labels.shape[1], labels.shape[2], 1]
                pure_neg = pure_neg.view(shape).expand_as(labels)
            else:
                return None
            return torch.where(pure_neg, sampled, torch.ones_like(labels))

        return None

    def _reduce(self, weighted_losses, full_mask):
        """Apply configured reduction over masked element-wise losses.

        ``weighted_losses`` is the element-wise focal loss. ``full_mask``
        combines structural validity with any negative sampling/weighting, so
        the mean denominator represents the same effective cells as the sum.
        """
        if self.bio_loss_reduction == "mean":
            denom = full_mask.sum().clamp(min=1.0)
            return (weighted_losses * full_mask).sum() / denom
        return (weighted_losses * full_mask).sum()

    @staticmethod
    def _apply_negative_mask(full_mask, negative_mask):
        """Fold sampling into the loss mask, including the mean denominator."""

        if negative_mask is None:
            return full_mask
        return full_mask * negative_mask.to(dtype=full_mask.dtype)

    # ── Anchor-matched losses ────────────────────────────────────────

    def _anchor_matched_bio_loss(
        self,
        scores,
        labels,
        anchor_mask,
        word_mask,
        child_mask,
        base_loss_fn,
        label_count=None,
    ):
        """Permutation-invariant BIO loss over structuring instance anchors.

        For each sample, assigns predicted anchors to gold instances via a
        Hungarian matcher minimising the per-pair masked BIO loss. Unmatched
        predictions are supervised against an all-zero target (negative
        anchors). When predictions exceed gold instances, matching uses each
        prediction's incremental cost relative to its all-zero target.
        """
        validate_structuring_anchor_capacity(
            labels,
            label_count,
            scores.shape[1],
            anchor_dim=1,
        )
        min_A = min(scores.shape[1], labels.shape[1])
        min_L = min(scores.shape[2], labels.shape[2])
        min_C = min(scores.shape[3], labels.shape[3])

        pred = scores[:, :min_A, :min_L, :min_C, :]
        lbl = labels[:, :min_A, :min_L, :min_C, :]

        pred_anchor_mask = anchor_mask[:, :min_A].bool()
        label_anchor_mask = self._label_anchor_mask(lbl, label_count)
        word_mask_f = word_mask[:, :min_L].float()
        child_mask_f = child_mask[:, :min_C].float()

        target = torch.zeros_like(pred)
        matches = self._compute_anchor_matches(
            pred=pred, lbl=lbl,
            pred_anchor_mask=pred_anchor_mask,
            label_anchor_mask=label_anchor_mask,
            word_mask_f=word_mask_f,
            child_mask_f=child_mask_f,
            base_loss_fn=base_loss_fn,
        )

        for b, batch_matches in enumerate(matches):
            for pred_anchor, label_anchor in batch_matches:
                target[b, pred_anchor] = lbl[b, label_anchor]

        losses = base_loss_fn(pred, target)
        neg_mask = self._build_negative_mask(target)
        full_mask = (
            pred_anchor_mask.float()[:, :, None, None, None]
            * word_mask_f[:, None, :, None, None]
            * child_mask_f[:, None, None, :, None]
        )
        loss_mask = self._apply_negative_mask(full_mask, neg_mask)
        loss_stats = self._compute_loss_stats(
            losses=losses, target=target, full_mask=loss_mask,
            pred_anchor_mask=pred_anchor_mask, matches=matches,
        )
        return self._reduce(losses, loss_mask), matches, pred_anchor_mask, loss_stats

    def _anchor_matched_span_loss(
        self,
        span_logits,
        span_labels,
        anchor_matches,
        supervised_anchor_mask,
        span_mask,
        child_mask,
        base_loss_fn,
    ):
        """Span loss using the same anchor assignment as the BIO loss.

        ``span_logits`` has shape ``(BN, A, S, C)``; ``span_labels`` is laid
        out as ``(BN, S, A, C)`` so we index the gold anchor on the third
        dim and copy into the matched predicted-anchor slot.
        """
        min_A = min(span_logits.shape[1], span_labels.shape[2])
        min_S = min(span_logits.shape[2], span_labels.shape[1])
        min_C = min(span_logits.shape[3], span_labels.shape[3])

        pred = span_logits[:, :min_A, :min_S, :min_C]
        labels = span_labels[:, :min_S, :min_A, :min_C]
        target = torch.zeros_like(pred)

        for b, batch_matches in enumerate(anchor_matches):
            for pred_anchor, label_anchor in batch_matches:
                if pred_anchor < min_A and label_anchor < min_A:
                    target[b, pred_anchor] = labels[b, :, label_anchor, :]

        losses = base_loss_fn(pred, target)
        neg_mask = self._build_negative_mask(target)
        full_mask = (
            supervised_anchor_mask[:, :min_A].float()[:, :, None, None]
            * span_mask[:, :min_S].float()[:, None, :, None]
            * child_mask[:, :min_C].float()[:, None, None, :]
        )
        loss_mask = self._apply_negative_mask(full_mask, neg_mask)
        return self._reduce(losses, loss_mask)

    # ── Positional (non-matched) losses with stats ───────────────────

    def _bio_loss_with_stats(self, scores, labels, anchor_mask, word_mask,
                             child_mask, base_loss_fn, label_count=None):
        """Positional BIO loss + diagnostic stats (no Hungarian matching)."""
        validate_structuring_anchor_capacity(
            labels,
            label_count,
            scores.shape[1],
            anchor_dim=1,
        )
        min_A = min(scores.shape[1], labels.shape[1])
        min_L = min(scores.shape[2], labels.shape[2])
        min_C = min(scores.shape[3], labels.shape[3])

        pred = scores[:, :min_A, :min_L, :min_C, :]
        lbl = labels[:, :min_A, :min_L, :min_C, :]

        losses = base_loss_fn(pred, lbl)
        neg_mask = self._build_negative_mask(lbl)
        full_mask = (
            anchor_mask[:, :min_A].float()[:, :, None, None, None]
            * word_mask[:, :min_L].float()[:, None, :, None, None]
            * child_mask[:, :min_C].float()[:, None, None, :, None]
        )
        loss_mask = self._apply_negative_mask(full_mask, neg_mask)
        loss_stats = self._compute_loss_stats(
            losses=losses, target=lbl, full_mask=loss_mask,
            pred_anchor_mask=anchor_mask[:, :min_A].bool(), matches=None,
        )
        return self._reduce(losses, loss_mask), loss_stats

    # ── Anchor objectness ────────────────────────────────────────────

    def _objectness_loss(self, objectness_logits, anchor_matches,
                        supervised_anchor_mask, gold_mask=None,
                        base_loss_fn=None):
        """Focal loss for the per-anchor "is this slot used?" head.

        When ``anchor_matches`` is provided, the target is built from the
        Hungarian assignment: matched predicted slots → 1, others → 0. When
        ``gold_mask`` is provided instead (no matching), it is used directly.
        Loss is averaged over the supervised-anchor mask so it stays
        comparable across batches with varying valid-anchor counts.
        """
        return matched_objectness_loss(
            objectness_logits,
            anchor_matches,
            supervised_anchor_mask,
            gold_mask=gold_mask,
            loss_fn=base_loss_fn,
        )

    # ── Diagnostic loss-stat helpers ─────────────────────────────────

    def _compute_loss_stats(self, losses, target, full_mask, pred_anchor_mask,
                            matches=None):
        """Per-batch positive/negative loss totals split by anchor partition.

        Returned dict contains floats (detached from the graph). Computed
        unconditionally because the cost is negligible relative to the
        forward pass; logging cadence is controlled by ``log_loss_stats``.
        """
        with torch.no_grad():
            pos_elem = (target > 0).to(losses.dtype) * full_mask
            neg_elem = (target == 0).to(losses.dtype) * full_mask

            if matches is not None:
                # Build a (B, A) mask: 1 where a slot was matched to a gold instance.
                matched = torch.zeros(
                    losses.shape[0], losses.shape[1],
                    device=losses.device, dtype=losses.dtype,
                )
                for b, batch_matches in enumerate(matches):
                    for pred_anchor, _ in batch_matches:
                        matched[b, pred_anchor] = 1.0
                unmatched = (pred_anchor_mask.to(losses.dtype) - matched).clamp(min=0)
            else:
                matched = pred_anchor_mask.to(losses.dtype)
                unmatched = torch.zeros_like(matched)

            # Broadcast (B, A) anchor partition to losses shape.
            shape = [losses.shape[0], losses.shape[1]] + [1] * (losses.dim() - 2)
            matched_b = matched.view(shape)
            unmatched_b = unmatched.view(shape)

            pos_matched = (losses * pos_elem * matched_b).sum().item()
            neg_matched = (losses * neg_elem * matched_b).sum().item()
            pos_unmatched = (losses * pos_elem * unmatched_b).sum().item()
            neg_unmatched = (losses * neg_elem * unmatched_b).sum().item()

            pos_count = pos_elem.sum().item()
            neg_count = neg_elem.sum().item()
            matched_anchors = matched.sum().item()
            unmatched_anchors = unmatched.sum().item()

        return {
            "pos_loss_matched": pos_matched,
            "neg_loss_matched": neg_matched,
            "pos_loss_unmatched": pos_unmatched,
            "neg_loss_unmatched": neg_unmatched,
            "pos_count": pos_count,
            "neg_count": neg_count,
            "matched_anchors": matched_anchors,
            "unmatched_anchors": unmatched_anchors,
        }

    def _maybe_log_stats(self, stats):
        """Log diagnostic stats every ``log_loss_stats_every`` training calls."""
        step = int(self._log_step.item())
        self._log_step += 1
        if step % self.log_loss_stats_every != 0:
            return

        pos_total = stats["pos_loss_matched"] + stats["pos_loss_unmatched"]
        neg_total = stats["neg_loss_matched"] + stats["neg_loss_unmatched"]
        pos_count = max(stats["pos_count"], 1.0)
        neg_count = max(stats["neg_count"], 1.0)
        ratio = neg_total / max(pos_total, 1e-9)
        per_pos = pos_total / pos_count
        per_neg = neg_total / neg_count

        logger.info(
            "[structuring/loss-stats step=%d] "
            "pos=%.3f (matched=%.3f, unmatched=%.3f, count=%.0f, per-elem=%.4g) | "
            "neg=%.3f (matched=%.3f, unmatched=%.3f, count=%.0f, per-elem=%.4g) | "
            "neg/pos=%.2fx | anchors matched=%.0f unmatched=%.0f",
            step,
            pos_total, stats["pos_loss_matched"], stats["pos_loss_unmatched"],
            pos_count, per_pos,
            neg_total, stats["neg_loss_matched"], stats["neg_loss_unmatched"],
            neg_count, per_neg,
            ratio,
            stats["matched_anchors"], stats["unmatched_anchors"],
        )

    # ── Matching helpers ─────────────────────────────────────────────

    def _compute_anchor_matches(
        self, pred, lbl, pred_anchor_mask, label_anchor_mask,
        word_mask_f, child_mask_f, base_loss_fn,
    ):
        """Hungarian matching of predicted anchors → gold instances per sample.

        Returns ``matches[b]`` = list of ``(pred_anchor, label_anchor)`` pairs.
        Cost is the masked BIO loss per (pred, gold) pair. When predictions
        outnumber gold instances, ``zero_cost`` (loss vs all-zero target) is
        subtracted per prediction so assignment compares the incremental cost
        of owning a record instead of each prediction's absolute loss scale.
        """
        token_child_mask = (
            word_mask_f[:, :, None] * child_mask_f[:, None, :]
        ).unsqueeze(-1)  # (B, L, C, 1)

        def pair_cost(batch_idx, pred_idx, label_idx):
            pred_b = pred[batch_idx, pred_idx]
            lbl_b = lbl[batch_idx, label_idx]
            mask_b = token_child_mask[batch_idx]
            pair_losses = base_loss_fn(
                pred_b[:, None].expand(
                    -1, label_idx.numel(), -1, -1, -1
                ),
                lbl_b[None].expand(
                    pred_idx.numel(), -1, -1, -1, -1
                ),
            )
            cost = (pair_losses * mask_b[None, None]).sum(
                dim=(-3, -2, -1)
            )
            if pred_idx.numel() > label_idx.numel():
                zero_losses = base_loss_fn(
                    pred_b,
                    torch.zeros_like(pred_b),
                )
                zero_cost = (
                    zero_losses * mask_b.unsqueeze(0)
                ).sum(dim=(-3, -2, -1))
                cost = cost - zero_cost[:, None]
            return cost

        return batched_masked_assignment(
            pred_anchor_mask,
            label_anchor_mask,
            pair_cost,
        )

    @staticmethod
    def _label_anchor_mask(labels, label_count):
        """Per-sample valid gold-anchor mask of shape (B, A).

        Uses ``label_count`` (number of gold instances per sample) when
        provided. Falls back to "any non-zero label entry" — useful for
        direct head tests where no count is supplied.
        """
        return gold_anchor_mask(labels, label_count)
