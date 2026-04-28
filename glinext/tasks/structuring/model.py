"""Structuring task head."""

import logging

import torch
from torch import nn

from .. import TaskHeadOutput
from ..anchored_extraction import AnchoredSpanExtractionHead

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
        super().__init__(config.structuring_config, config, hidden_size, dropout, shared_layers)
        struct_cfg = config.structuring_config
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
            self.objectness_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )

    @classmethod
    def from_config(cls, config, shared_layers=None, **kwargs):
        if config.structuring_config is None:
            return None
        return cls(config, hidden_size=config.hidden_size, dropout=config.dropout,
                   shared_layers=shared_layers)

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

        scores, anchors, anchor_mask, fused_flat, (B, A, C, L) = self._compute_bio_scores(
            flat_inputs, batch,
        )

        # Anchor-objectness logits (one sigmoid score per anchor slot).
        objectness_logits = None
        if self.use_anchor_objectness:
            objectness_logits = self.objectness_head(anchors).squeeze(-1)  # (BN, A)

        # Optional span-level rescoring — auxiliary training signal only.
        span_logits_out = None
        if (
            self.represent_spans and hasattr(self, "span_rep_layer")
            and structuring_labels is not None
        ):
            span_idx, span_mask = self._maybe_extract_spans(
                scores, A=A, span_idx=span_idx, span_mask=span_mask, threshold=threshold,
            )
            if span_idx is not None:
                span_logits_out = self._span_logits_from_fused(
                    flat_inputs.words_embedding, fused_flat, B, A, C, span_idx,
                )

        loss = None
        loss_stats = None
        if structuring_labels is not None and base_loss_fn is not None:
            if self.use_anchor_matching:
                label_count = batch.get("structuring_count")
                if label_count is None:
                    label_count = batch.get("count_val")
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
                    label_count = batch.get("structuring_count")
                    if label_count is None:
                        label_count = batch.get("count_val")
                    if label_count is not None:
                        gold_mask = self._label_anchor_mask(
                            structuring_labels, label_count,
                        )
                        obj_loss = self._objectness_loss(
                            objectness_logits=objectness_logits,
                            anchor_matches=None,
                            supervised_anchor_mask=anchor_mask.bool(),
                            gold_mask=gold_mask,
                        )
                        loss = loss + self.anchor_objectness_loss_coef * obj_loss
                        if loss_stats is not None:
                            loss_stats["objectness_loss"] = float(obj_loss.detach().item())

            if loss_stats is not None and self.log_loss_stats and self.training:
                self._maybe_log_stats(loss_stats)

        return TaskHeadOutput(
            loss=loss,
            logits=scores,
            extra={
                "groups_output": anchors,
                "anchor_mask": anchor_mask,
                "objectness_logits": objectness_logits,
                "span_logits": span_logits_out,
                "span_idx": span_idx,
                "span_mask": span_mask,
                "loss_stats": loss_stats,
            },
        )

    # ── Loss reduction & negative masking ────────────────────────────

    def _build_negative_mask(self, labels):
        """GLiNER-style negative sampling mask, adapted to BIO/span shapes.

        Positives (``labels > 0``) are always kept (mask=1). Negatives are
        kept independently with probability ``self.negatives`` according to
        ``self.masking_mode``:

        - ``"none"``    → mask is 1 everywhere (no sampling).
        - ``"global"``  → element-wise Bernoulli over labels==0.
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

        ``weighted_losses`` is the focal loss already multiplied by any
        per-element weights (negative-sampling, etc.). ``full_mask`` is the
        structural mask (anchor / token / field validity).
        """
        if self.bio_loss_reduction == "mean":
            denom = full_mask.sum().clamp(min=1.0)
            return (weighted_losses * full_mask).sum() / denom
        return (weighted_losses * full_mask).sum()

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
        anchors). When predictions exceed gold instances, the matcher only
        claims a prediction for a gold instance if doing so beats the
        per-prediction "train as negative" cost.
        """
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
        if neg_mask is not None:
            losses = losses * neg_mask
        full_mask = (
            pred_anchor_mask.float()[:, :, None, None, None]
            * word_mask_f[:, None, :, None, None]
            * child_mask_f[:, None, None, :, None]
        )
        loss_stats = self._compute_loss_stats(
            losses=losses, target=target, full_mask=full_mask,
            pred_anchor_mask=pred_anchor_mask, matches=matches,
        )
        return self._reduce(losses, full_mask), matches, pred_anchor_mask, loss_stats

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
        if neg_mask is not None:
            losses = losses * neg_mask
        full_mask = (
            supervised_anchor_mask[:, :min_A].float()[:, :, None, None]
            * span_mask[:, :min_S].float()[:, None, :, None]
            * child_mask[:, :min_C].float()[:, None, None, :]
        )
        return self._reduce(losses, full_mask)

    # ── Positional (non-matched) losses with stats ───────────────────

    def _bio_loss_with_stats(self, scores, labels, anchor_mask, word_mask,
                             child_mask, base_loss_fn):
        """Positional BIO loss + diagnostic stats (no Hungarian matching)."""
        min_A = min(scores.shape[1], labels.shape[1])
        min_L = min(scores.shape[2], labels.shape[2])
        min_C = min(scores.shape[3], labels.shape[3])

        pred = scores[:, :min_A, :min_L, :min_C, :]
        lbl = labels[:, :min_A, :min_L, :min_C, :]

        losses = base_loss_fn(pred, lbl)
        neg_mask = self._build_negative_mask(lbl)
        if neg_mask is not None:
            losses = losses * neg_mask
        full_mask = (
            anchor_mask[:, :min_A].float()[:, :, None, None, None]
            * word_mask[:, :min_L].float()[:, None, :, None, None]
            * child_mask[:, :min_C].float()[:, None, None, :, None]
        )
        loss_stats = self._compute_loss_stats(
            losses=losses, target=lbl, full_mask=full_mask,
            pred_anchor_mask=anchor_mask[:, :min_A].bool(), matches=None,
        )
        return self._reduce(losses, full_mask), loss_stats

    # ── Anchor objectness ────────────────────────────────────────────

    def _objectness_loss(self, objectness_logits, anchor_matches,
                        supervised_anchor_mask, gold_mask=None):
        """BCE loss for the per-anchor "is this slot used?" head.

        When ``anchor_matches`` is provided, the target is built from the
        Hungarian assignment: matched predicted slots → 1, others → 0. When
        ``gold_mask`` is provided instead (no matching), it is used directly.
        Loss is averaged over the supervised-anchor mask so it stays
        comparable across batches with varying valid-anchor counts.
        """
        target = torch.zeros_like(objectness_logits)
        if anchor_matches is not None:
            for b, batch_matches in enumerate(anchor_matches):
                for pred_anchor, _ in batch_matches:
                    if pred_anchor < target.shape[1]:
                        target[b, pred_anchor] = 1.0
        elif gold_mask is not None:
            min_A = min(target.shape[1], gold_mask.shape[1])
            target[:, :min_A] = gold_mask[:, :min_A].to(target.dtype)

        losses = nn.functional.binary_cross_entropy_with_logits(
            objectness_logits, target, reduction="none",
        )
        mask = supervised_anchor_mask.float()
        denom = mask.sum().clamp(min=1.0)
        return (losses * mask).sum() / denom

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
        subtracted per prediction so the matcher only claims predictions
        that benefit from a real label more than from being trained as a
        negative.
        """
        B = pred.shape[0]
        matches = [[] for _ in range(B)]
        token_child_mask = (
            word_mask_f[:, :, None] * child_mask_f[:, None, :]
        ).unsqueeze(-1)  # (B, L, C, 1)

        with torch.no_grad():
            for b in range(B):
                pred_idx = torch.nonzero(pred_anchor_mask[b], as_tuple=False).flatten()
                label_idx = torch.nonzero(label_anchor_mask[b], as_tuple=False).flatten()
                n_pred, n_label = pred_idx.numel(), label_idx.numel()
                if n_pred == 0 or n_label == 0:
                    continue

                pred_b = pred[b, pred_idx]                 # (n_pred, L, C, 3)
                lbl_b = lbl[b, label_idx]                  # (n_label, L, C, 3)
                mask_b = token_child_mask[b]               # (L, C, 1)

                pair_pred = pred_b[:, None].expand(-1, n_label, -1, -1, -1)
                pair_lbl = lbl_b[None].expand(n_pred, -1, -1, -1, -1)
                pair_losses = base_loss_fn(pair_pred, pair_lbl)
                pair_cost = (pair_losses * mask_b[None, None]).sum(dim=(-3, -2, -1))
                # pair_cost: (n_pred, n_label)

                if n_pred > n_label:
                    # Subtract per-prediction "train as negative" cost so the
                    # matcher only assigns a gold instance to predictions that
                    # gain from it. Constant per row, so safe to subtract.
                    zero_losses = base_loss_fn(pred_b, torch.zeros_like(pred_b))
                    zero_cost = (zero_losses * mask_b.unsqueeze(0)).sum(dim=(-3, -2, -1))
                    pair_cost = pair_cost - zero_cost[:, None]

                # Hungarian implementation requires rows <= cols.
                if n_pred <= n_label:
                    assignment = self._hungarian_rows_to_cols(pair_cost.cpu().tolist())
                    for pred_pos, label_pos in assignment:
                        matches[b].append((
                            int(pred_idx[pred_pos].item()),
                            int(label_idx[label_pos].item()),
                        ))
                else:
                    assignment = self._hungarian_rows_to_cols(
                        pair_cost.t().contiguous().cpu().tolist()
                    )
                    for label_pos, pred_pos in assignment:
                        matches[b].append((
                            int(pred_idx[pred_pos].item()),
                            int(label_idx[label_pos].item()),
                        ))

        return matches

    @staticmethod
    def _label_anchor_mask(labels, label_count):
        """Per-sample valid gold-anchor mask of shape (B, A).

        Uses ``label_count`` (number of gold instances per sample) when
        provided. Falls back to "any non-zero label entry" — useful for
        direct head tests where no count is supplied.
        """
        B, A = labels.shape[:2]
        if label_count is not None:
            if not torch.is_tensor(label_count):
                label_count = torch.as_tensor(label_count, device=labels.device)
            if label_count.dim() == 0:
                label_count = label_count.unsqueeze(0).expand(B)
            label_count = label_count.to(device=labels.device).long().clamp(min=0, max=A)
            return torch.arange(A, device=labels.device).unsqueeze(0) < label_count.unsqueeze(1)

        flat = labels.detach().abs().reshape(B, A, -1)
        return flat.sum(dim=-1) > 0

    @staticmethod
    def _hungarian_rows_to_cols(cost):
        """Min-cost rows-to-distinct-cols assignment (Jonker-Volgenant).

        Pure-Python so we don't pull scipy in. Requires ``n_rows <= n_cols``.
        Returns ``[(row, col), ...]`` covering every row.
        """
        n_rows = len(cost)
        n_cols = len(cost[0]) if n_rows else 0
        if n_rows == 0 or n_cols == 0:
            return []
        if n_rows > n_cols:
            raise ValueError("Hungarian assignment requires rows <= columns")

        u = [0.0] * (n_rows + 1)
        v = [0.0] * (n_cols + 1)
        p = [0] * (n_cols + 1)
        way = [0] * (n_cols + 1)

        for i in range(1, n_rows + 1):
            p[0] = i
            j0 = 0
            minv = [float("inf")] * (n_cols + 1)
            used = [False] * (n_cols + 1)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = float("inf")
                j1 = 0
                for j in range(1, n_cols + 1):
                    if used[j]:
                        continue
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
                for j in range(n_cols + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break

            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        assignment = []
        for j in range(1, n_cols + 1):
            if p[j] != 0:
                assignment.append((p[j] - 1, j - 1))
        return assignment
