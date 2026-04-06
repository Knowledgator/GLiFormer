"""Open relation extraction head (GLiNER2 style).

Standalone anchor-based head — no NER dependency. Uses configurable anchor
layers to generate relation instance slots, fuses with [REL] type embeddings,
and extracts head/tail spans via dual AnchoredSpanScorers.

Supports optional span representation (represent_spans) for direct span-level
scoring alongside token-level BIO scoring.
"""

import torch
from torch import nn

from gliner.modeling.utils import extract_prompt_features
from gliner.modeling.span_rep import SpanRepLayer

from .. import TaskHead, TaskHeadOutput, TaskFlatInputs, SharedRepresentations
from ...layers import AnchoredSpanScorer, AnchorLayer, AnchorModeling, AnchorCrossAttentionLayer


class OpenRelexHead(TaskHead):
    """Anchor-based relation extraction via dual head/tail span scoring.

    For each (anchor, rel_type) pair, extracts both a head entity span
    and a tail entity span in the text.

    Output logits shape: (B, X, C, L, 2, 3)
        X = anchors, C = rel classes, L = seq len, 2 = [head, tail], 3 = [start, inside, end]
    """

    name = "open_relex"
    dependencies = []

    def __init__(self, config, hidden_size, dropout, shared_layers=None):
        super().__init__()
        cfg = config.open_relex_config
        self.loss_coef = cfg.loss_coef
        self.rel_token_index = cfg.rel_token_index
        self.embed_rel_token = cfg.embed_rel_token
        self.represent_spans = getattr(cfg, 'represent_spans', False)
        self.span_loss_coef = getattr(cfg, 'span_loss_coef', 1.0)
        if shared_layers is None:
            shared_layers = {}

        # Anchor layer (configurable strategy)
        anchor_mode = cfg.anchor_mode
        if anchor_mode == "lstm":
            anchor_mode = "rotary"

        self.anchor_layer = AnchorLayer.from_config(
            anchor_mode, hidden_size,
            max_count=cfg.max_count,
            num_slots=cfg.num_fixed_slots,
            num_heads=cfg.groups_num_heads,
            num_layers=cfg.groups_num_layers,
            dropout=dropout,
        )

        # Anchor-child fusion
        if "anchor_modeling" in shared_layers:
            self.anchor_modeling = shared_layers["anchor_modeling"]
        else:
            self.anchor_modeling = AnchorModeling.from_config(
                cfg.anchor_modeling, hidden_size, dropout=dropout,
            )

        if "anchor_refine" in shared_layers:
            self.anchor_refine = shared_layers["anchor_refine"]
        else:
            refine_layers = getattr(cfg, "anchor_refine_layers", 0)
            if refine_layers > 0:
                refine_heads = getattr(cfg, "anchor_refine_heads", 8)
                self.anchor_refine = AnchorCrossAttentionLayer(
                    hidden_size, num_heads=refine_heads, num_layers=refine_layers, dropout=dropout,
                )

        # Dual scorers: one for head spans, one for tail spans
        self.head_scorer = AnchoredSpanScorer(hidden_size, dropout=dropout)
        self.tail_scorer = AnchoredSpanScorer(hidden_size, dropout=dropout)

        if self.represent_spans:
            self.span_rep_layer = SpanRepLayer(
                span_mode="token_level",
                hidden_size=hidden_size,
                max_width=getattr(config, "max_width", 12),
                dropout=dropout,
            )
            # Separate projections for head/tail span scoring
            self.head_span_proj = nn.Linear(hidden_size, hidden_size)
            self.tail_span_proj = nn.Linear(hidden_size, hidden_size)

    @classmethod
    def from_config(cls, config, shared_layers=None, **kwargs):
        if config.open_relex_config is None:
            return None
        return cls(config, hidden_size=config.hidden_size, dropout=config.dropout,
                   shared_layers=shared_layers)

    def forward(self, shared, dependency_outputs, flat_inputs=None,
                open_rel_label_embeds=None, base_loss_fn=None, **batch):
        open_rel_labels = batch.get("open_rel_labels")
        open_rel_count = batch.get("open_rel_count")
        threshold = batch.get("threshold", 0.5)

        # Span representation inputs
        span_idx = batch.get("open_rel_span_idx")
        span_mask = batch.get("open_rel_span_mask")
        span_labels = batch.get("open_rel_span_labels")

        # Use flat_inputs (BN-indexed) when available
        if flat_inputs is not None:
            words_embedding = flat_inputs.words_embedding
            mask = flat_inputs.mask
            rel_embedding = flat_inputs.child_embedding
            rel_embedding_mask = flat_inputs.child_mask
            parent_embedding = flat_inputs.parent_embedding.unsqueeze(1)  # (BN, 1, D)
        else:
            token_embeds = shared.token_embeds
            input_ids = shared.input_ids
            attention_mask = shared.attention_mask
            words_embedding = shared.words_embedding
            mask = shared.mask
            parent_embedding = shared.prompts_embedding

            batch_size, _, embed_dim = token_embeds.shape

            # 1. Get [REL] type embeddings
            if open_rel_label_embeds is not None:
                rel_embedding = open_rel_label_embeds
                rel_embedding_mask = torch.ones(
                    rel_embedding.shape[:-1], dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            else:
                rel_embedding, rel_embedding_mask = extract_prompt_features(
                    self.rel_token_index, token_embeds, input_ids, attention_mask,
                    batch_size, embed_dim, self.embed_rel_token,
                )

        if rel_embedding.shape[1] == 0:
            return TaskHeadOutput()

        # 2. Generate anchors
        anchors, anchor_mask = self.anchor_layer(
            parent_embedding, words_embedding,
            count=open_rel_count, threshold=threshold,
        )

        if hasattr(self, "anchor_refine"):
            anchors = self.anchor_refine(anchors, words_embedding, token_mask=mask)

        # 3. Fuse anchors + rel types: (B, X, C, D)
        fused = self.anchor_modeling(anchors, rel_embedding)
        B, X, C, D = fused.shape
        L = words_embedding.shape[1]
        fused_flat = fused.view(B, X * C, D)

        # 4. Score head and tail spans separately
        head_logits_flat = self.head_scorer(fused_flat, words_embedding, word_mask=mask)  # (B, X*C, L, 3)
        tail_logits_flat = self.tail_scorer(fused_flat, words_embedding, word_mask=mask)  # (B, X*C, L, 3)

        # 5. Reshape and stack: (B, X, C, L, 2, 3)
        head_logits = head_logits_flat.view(B, X, C, L, 3)
        tail_logits = tail_logits_flat.view(B, X, C, L, 3)
        logits = torch.stack([head_logits, tail_logits], dim=-2)  # (B, X, C, L, 2, 3)

        # 6. Optional span representation
        span_logits_out = None
        if self.represent_spans and hasattr(self, "span_rep_layer"):
            if span_idx is not None:
                span_rep = self.span_rep_layer(words_embedding, span_idx)  # (B, S, D)
                S = span_rep.shape[1]
                # Separate projections for head/tail roles
                head_span_rep = self.head_span_proj(span_rep)  # (B, S, D)
                tail_span_rep = self.tail_span_proj(span_rep)  # (B, S, D)
                # Score: (B, S, X*C) each
                head_span_scores = torch.einsum("BSD,BND->BSN", head_span_rep, fused_flat)
                tail_span_scores = torch.einsum("BSD,BND->BSN", tail_span_rep, fused_flat)
                # Reshape to (B, X, C, S) and stack head/tail: (B, S, X, C, 2)
                head_span = head_span_scores.view(B, S, X, C)
                tail_span = tail_span_scores.view(B, S, X, C)
                span_logits_out = torch.stack([head_span, tail_span], dim=-1)  # (B, S, X, C, 2)

        # 7. Loss computation
        loss = None
        if open_rel_labels is not None and base_loss_fn is not None:
            X_pred, X_label = logits.shape[1], open_rel_labels.shape[1]
            C_pred, C_label = logits.shape[2], open_rel_labels.shape[2]
            L_pred, L_label = logits.shape[3], open_rel_labels.shape[3]

            min_X = min(X_pred, X_label)
            min_C = min(C_pred, C_label)
            min_L = min(L_pred, L_label)

            pred = logits[:, :min_X, :min_C, :min_L, :, :]
            labels = open_rel_labels[:, :min_X, :min_C, :min_L, :, :]

            all_losses = base_loss_fn(pred, labels)

            # Build masks: anchor × rel_class × word
            inst_mask = anchor_mask[:, :min_X].float()
            word_mask_f = mask[:, :min_L].float()
            rel_mask_f = rel_embedding_mask[:, :min_C].float()

            full_mask = (
                inst_mask[:, :, None, None, None, None]
                * rel_mask_f[:, None, :, None, None, None]
                * word_mask_f[:, None, None, :, None, None]
            )

            loss = (all_losses * full_mask).sum()

            # Span-level loss
            if span_labels is not None and span_logits_out is not None:
                # span_logits_out: (B, S, X, C, 2)
                # span_labels:     (B, S, X, C, 2)
                min_S = min(span_logits_out.shape[1], span_labels.shape[1])
                min_X_s = min(span_logits_out.shape[2], span_labels.shape[2])
                min_C_s = min(span_logits_out.shape[3], span_labels.shape[3])

                span_pred = span_logits_out[:, :min_S, :min_X_s, :min_C_s, :]
                s_labels = span_labels[:, :min_S, :min_X_s, :min_C_s, :]

                span_losses = base_loss_fn(span_pred, s_labels)
                s_span_mask = span_mask[:, :min_S].float()
                s_inst_mask = anchor_mask[:, :min_X_s].float()
                s_rel_mask = rel_embedding_mask[:, :min_C_s].float()

                s_full_mask = (
                    s_span_mask[:, :, None, None, None]
                    * s_inst_mask[:, None, :, None, None]
                    * s_rel_mask[:, None, None, :, None]
                )
                span_loss = (span_losses * s_full_mask).sum()
                loss = loss + self.span_loss_coef * span_loss

        return TaskHeadOutput(
            loss=loss,
            logits=logits,
            extra={
                "anchors": anchors,
                "anchor_mask": anchor_mask,
                "span_logits": span_logits_out,
                "span_idx": span_idx,
                "span_mask": span_mask,
            },
        )
