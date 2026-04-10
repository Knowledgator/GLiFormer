"""Structuring task head."""

import torch
from torch import nn

from .. import TaskHead, TaskHeadOutput
from ...layers import AnchoredSpanScorer


class StructuringHead(TaskHead):
    """Structuring via anchor-based span extraction using configurable anchor layer.

    Supports optional span representation (represent_spans) for direct span-level
    scoring alongside token-level BIO scoring.
    """

    name = "structuring"
    dependencies = []

    def __init__(self, config, hidden_size, dropout, shared_layers=None):
        super().__init__()
        struct_cfg = config.structuring_config
        self.loss_coef = struct_cfg.loss_coef
        self.child_token_index = struct_cfg.child_token_index
        self.embed_child_token = struct_cfg.embed_child_token
        if shared_layers is None:
            shared_layers = {}

        self._init_anchor_pipeline(struct_cfg, config, hidden_size, dropout, shared_layers)

        self.anchored_scorer = AnchoredSpanScorer(hidden_size, dropout=dropout)

    @classmethod
    def from_config(cls, config, shared_layers=None, **kwargs):
        if config.structuring_config is None:
            return None
        return cls(config, hidden_size=config.hidden_size, dropout=config.dropout,
                   shared_layers=shared_layers)

    def forward(self, shared, dependency_outputs, flat_inputs=None,
                base_loss_fn=None, **batch):
        count_val = batch.get("count_val")
        structuring_labels = batch.get("structuring_labels")
        structuring_count = batch.get("structuring_count")
        threshold = batch.get("threshold", 0.5)

        # Span representation inputs
        span_idx = batch.get("structuring_span_idx")
        span_mask = batch.get("structuring_span_mask")
        span_labels = batch.get("structuring_span_labels")

        words_embedding = flat_inputs.words_embedding       # (BN, W, D)
        mask = flat_inputs.mask                              # (BN, W)
        child_embedding = flat_inputs.child_embedding        # (BN, max_C, D)
        child_embedding_mask = flat_inputs.child_mask        # (BN, max_C)
        parent_embedding = flat_inputs.parent_embedding      # (BN, D)

        if child_embedding.shape[1] == 0:
            return TaskHeadOutput()

        count_for_groups = structuring_count if structuring_count is not None else count_val
        anchors, anchor_mask = self.anchor_layer(
            parent_embedding, words_embedding,
            count=count_for_groups, threshold=threshold,
        )

        if hasattr(self, "anchor_refine"):
            anchors = self.anchor_refine(anchors, words_embedding, token_mask=mask)

        # Fuse anchors + children: (B, X, C, D) then score against words
        fused = self.anchor_modeling(anchors, child_embedding)
        B, X, C, D = fused.shape
        fused_flat = fused.view(B, X * C, D)
        # Score fused reps against word embeddings: (B, X*C, L, 3)
        structuring_logits_flat = self.anchored_scorer(
            fused_flat, words_embedding, word_mask=mask,
        )
        # Reshape and permute to (B, X, L, C, 3) to match labels and decoder
        structuring_logits = structuring_logits_flat.view(B, X, C, -1, 3).permute(0, 1, 3, 2, 4)

        # Optional span representation
        span_logits_out = None
        if self.represent_spans and hasattr(self, "span_rep_layer"):
            if span_idx is not None:
                span_rep = self.span_rep_layer(words_embedding, span_idx)  # (B, S, D)
                # Score spans against fused (anchor+child) reps: (B, S, X*C)
                span_logits_flat = torch.einsum("BSD,BND->BSN", span_rep, fused_flat)
                # Reshape to (B, X, S, C) for per-anchor, per-field span scores
                S = span_rep.shape[1]
                span_logits_out = span_logits_flat.view(B, S, X, C).permute(0, 2, 1, 3)

        loss = None
        if structuring_labels is not None and base_loss_fn is not None:
            min_X = min(structuring_logits.shape[1], structuring_labels.shape[1])
            min_L = min(structuring_logits.shape[2], structuring_labels.shape[2])
            min_C = min(structuring_logits.shape[3], structuring_labels.shape[3])

            pred = structuring_logits[:, :min_X, :min_L, :min_C, :]
            labels = structuring_labels[:, :min_X, :min_L, :min_C, :]

            all_losses = base_loss_fn(pred, labels)

            inst_mask = anchor_mask[:, :min_X].float()
            word_mask_f = mask[:, :min_L].float()
            child_mask_f = child_embedding_mask[:, :min_C].float()

            full_mask = (
                inst_mask[:, :, None, None, None]
                * word_mask_f[:, None, :, None, None]
                * child_mask_f[:, None, None, :, None]
            )

            loss = (all_losses * full_mask).sum()

            # Span-level loss
            if span_labels is not None and span_logits_out is not None:
                min_X_s = min(span_logits_out.shape[1], span_labels.shape[2])
                min_S = min(span_logits_out.shape[2], span_labels.shape[1])
                min_C_s = min(span_logits_out.shape[3], span_labels.shape[3])

                # span_logits_out: (B, X, S, C), span_labels: (B, S, X, C)
                span_pred = span_logits_out[:, :min_X_s, :min_S, :min_C_s]
                s_labels = span_labels[:, :min_S, :min_X_s, :min_C_s].permute(0, 2, 1, 3)

                span_losses = base_loss_fn(span_pred, s_labels)
                s_inst_mask = anchor_mask[:, :min_X_s].float()
                s_span_mask = span_mask[:, :min_S].float()
                s_child_mask = child_embedding_mask[:, :min_C_s].float()

                s_full_mask = (
                    s_inst_mask[:, :, None, None]
                    * s_span_mask[:, None, :, None]
                    * s_child_mask[:, None, None, :]
                )
                span_loss = (span_losses * s_full_mask).sum()
                loss = loss + self.span_loss_coef * span_loss

        return TaskHeadOutput(
            loss=loss,
            logits=structuring_logits,
            extra={
                "groups_output": anchors,
                "anchor_mask": anchor_mask,
                "span_logits": span_logits_out,
                "span_idx": span_idx,
                "span_mask": span_mask,
            },
        )
