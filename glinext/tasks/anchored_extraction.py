"""Base head for anchor-based BIO span extraction.

Shared by NER and Structuring — both score (anchor × child) pairs against
sequence features to produce per-token BIO logits.

Structuring is a generalization of NER where the anchor dimension A can be
greater than one; flattening (A, C) yields the same scoring pipeline as NER.
"""

import torch
from gliner.modeling.utils import extract_spans_from_tokens

from . import TaskHead, TaskHeadOutput
from ..layers import AnchoredSpanScorer


class AnchoredSpanExtractionHead(TaskHead):
    """Anchor + child → BIO scores over text tokens.

    Produces raw scores of shape ``(BN, A, L, C, 3)`` from
    ``anchor_modeling(anchor_layer(parent), child)`` fused against word
    features via :class:`AnchoredSpanScorer`. Subclasses reshape, mask and
    decode the output per task.
    """

    def __init__(self, task_cfg, config, hidden_size, dropout, shared_layers=None):
        super().__init__()
        self.config = config
        self.loss_coef = task_cfg.loss_coef
        if shared_layers is None:
            shared_layers = {}
        self._init_anchor_pipeline(task_cfg, config, hidden_size, dropout, shared_layers)
        self.scorer = AnchoredSpanScorer(hidden_size, dropout=dropout)

    # ── Extension hooks ──────────────────────────────────────────────

    def _anchor_kwargs(self, batch):
        """Extra kwargs forwarded to ``self.anchor_layer``. Default: threshold only."""
        return {"threshold": batch.get("threshold", 0.5)}

    # ── Shared pipeline ──────────────────────────────────────────────

    def _compute_bio_scores(self, flat_inputs, batch):
        """Anchor + child fusion, scored against word tokens.

        Returns:
            scores:       (BN, A, L, C, 3) BIO logits
            anchors:      (BN, A, D) anchor representations (post-refine)
            anchor_mask:  (BN, A) valid anchor mask
            fused_flat:   (BN, A*C, D) fused reps (for span-level reuse)
            dims:         (B, A, C, L)
        """
        feature_embeddings = flat_inputs.words_embedding
        feature_mask = flat_inputs.mask
        child_embedding = flat_inputs.child_embedding
        parent_embedding = flat_inputs.parent_embedding

        anchors, anchor_mask = self.anchor_layer(
            parent_embedding, feature_embeddings, feature_mask=feature_mask, **self._anchor_kwargs(batch),
        )
        if hasattr(self, "anchor_refine"):
            anchors = self.anchor_refine(anchors, feature_embeddings, token_mask=feature_mask)

        fused = self.anchor_modeling(anchors, child_embedding)   # (BN, A, C, D)
        B, A, C, D = fused.shape
        L = feature_embeddings.shape[1]
        fused_flat = fused.reshape(B, A * C, D)
        scores_flat = self.scorer(fused_flat, feature_embeddings, word_mask=feature_mask)
        # (BN, A*C, L, 3) → (BN, A, C, L, 3) → (BN, A, L, C, 3)
        scores = scores_flat.reshape(B, A, C, L, 3).permute(0, 1, 3, 2, 4)
        return scores, anchors, anchor_mask, fused_flat, (B, A, C, L)

    def _bio_loss(self, scores, labels, anchor_mask, word_mask, child_mask, base_loss_fn):
        """Masked BIO loss over (BN, A, L, C, 3) predictions and labels."""
        min_A = min(scores.shape[1], labels.shape[1])
        min_L = min(scores.shape[2], labels.shape[2])
        min_C = min(scores.shape[3], labels.shape[3])

        pred = scores[:, :min_A, :min_L, :min_C, :]
        lbl = labels[:, :min_A, :min_L, :min_C, :]

        losses = base_loss_fn(pred, lbl)

        full_mask = (
            anchor_mask[:, :min_A].float()[:, :, None, None, None]
            * word_mask[:, :min_L].float()[:, None, :, None, None]
            * child_mask[:, :min_C].float()[:, None, None, :, None]
        )
        return (losses * full_mask).sum()

    def _span_loss(self, span_logits, span_labels, anchor_mask, span_mask,
                   child_mask, base_loss_fn):
        """Masked span-level loss.

        ``span_logits`` has shape (BN, A, S, C), ``span_labels`` (BN, S, A, C).
        """
        min_A = min(span_logits.shape[1], span_labels.shape[2])
        min_S = min(span_logits.shape[2], span_labels.shape[1])
        min_C = min(span_logits.shape[3], span_labels.shape[3])

        pred = span_logits[:, :min_A, :min_S, :min_C]
        lbl = span_labels[:, :min_S, :min_A, :min_C].permute(0, 2, 1, 3)

        losses = base_loss_fn(pred, lbl)
        full_mask = (
            anchor_mask[:, :min_A].float()[:, :, None, None]
            * span_mask[:, :min_S].float()[:, None, :, None]
            * child_mask[:, :min_C].float()[:, None, None, :]
        )
        return (losses * full_mask).sum()

    def _maybe_extract_spans(self, scores, A, span_idx, span_mask, threshold, labels=None):
        """Extract (span_idx, span_mask) from BIO scores when not supplied.

        Collapses the anchor dim by taking the max so span proposals don't miss
        spans that fire on some anchor.
        """
        if span_idx is not None:
            return span_idx, span_mask
        source = scores.squeeze(1) if A == 1 else scores.max(dim=1).values
        span_idx, span_mask = extract_spans_from_tokens(source, labels, threshold)
        span_idx = span_idx * span_mask.unsqueeze(-1).long()
        return span_idx, span_mask

    def _span_logits_from_fused(self, feature_embeddings, fused_flat, B, A, C, span_idx):
        """Default span-level scoring: span_rep · fused — shape (BN, A, S, C)."""
        span_rep = self.span_rep_layer(feature_embeddings, span_idx)          # (B, S, D)
        span_logits_flat = torch.einsum("BSD,BND->BSN", span_rep, fused_flat)  # (B, S, A*C)
        S = span_rep.shape[1]
        return span_logits_flat.reshape(B, S, A, C).permute(0, 2, 1, 3)
