"""Open relation extraction head (GLiNER2 style).

Standalone anchor-based head — no NER dependency. Uses configurable anchor
layers to generate relation instance slots, fuses with [RELATION] type embeddings,
and extracts head/tail spans via dual AnchoredSpanScorers.

Supports optional span representation (represent_spans) for direct span-level
scoring alongside token-level BIO scoring.
"""

import torch
from torch import nn

from ...layers import AnchoredSpanScorer
from .. import TaskHead, TaskHeadOutput


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
        if shared_layers is None:
            shared_layers = {}

        self._init_anchor_components(cfg, config, hidden_size, dropout, shared_layers)

        # Dual scorers: one for head spans, one for tail spans
        self.head_scorer = AnchoredSpanScorer(hidden_size, dropout=dropout)
        self.tail_scorer = AnchoredSpanScorer(hidden_size, dropout=dropout)

        if self.represent_spans:
            self.head_span_proj = nn.Linear(hidden_size, hidden_size)
            self.tail_span_proj = nn.Linear(hidden_size, hidden_size)

    @classmethod
    def from_config(cls, config, shared_layers=None, **kwargs):
        open_rel_cfg = config.open_relex_config
        if open_rel_cfg is None:
            return None
        if getattr(open_rel_cfg, "head_type", "open_relex") != cls.name:
            return None
        return cls(config, hidden_size=config.hidden_size, dropout=config.dropout,
                   shared_layers=shared_layers)

    def _compute_relation_scores(self, flat_inputs, open_rel_count, threshold):
        """Compute canonical open-relation logits and reusable components."""

        feature_embeddings = flat_inputs.words_embedding
        feature_mask = flat_inputs.mask
        rel_embedding = flat_inputs.child_embedding
        parent_embedding = flat_inputs.parent_embedding

        anchors, anchor_mask = self._generate_anchors(
            parent_embedding,
            feature_embeddings,
            # Gold counts do not exist at inference. Masking padded slots only
            # during training leaves them unsupervised and then activates all
            # of them at inference. Train and infer with the same query set;
            # padded labels below make unused slots explicit background.
            count=None,
            threshold=threshold,
            feature_mask=feature_mask,
        )
        anchors = self._refine_anchors(
            anchors,
            feature_embeddings,
            memory_mask=feature_mask,
            anchor_mask=anchor_mask,
        )

        fused = self._model_anchors(
            anchors,
            rel_embedding,
            anchor_mask=anchor_mask,
            child_mask=flat_inputs.child_mask,
        )
        batch_size, anchor_count, class_count, hidden_size = fused.shape
        sequence_length = feature_embeddings.shape[1]
        fused_flat = fused.reshape(
            batch_size,
            anchor_count * class_count,
            hidden_size,
        )

        head_logits_flat = self.head_scorer(
            fused_flat,
            feature_embeddings,
            word_mask=feature_mask,
        )
        tail_logits_flat = self.tail_scorer(
            fused_flat,
            feature_embeddings,
            word_mask=feature_mask,
        )
        head_logits = head_logits_flat.reshape(
            batch_size,
            anchor_count,
            class_count,
            sequence_length,
            3,
        )
        tail_logits = tail_logits_flat.reshape(
            batch_size,
            anchor_count,
            class_count,
            sequence_length,
            3,
        )
        logits = torch.stack([head_logits, tail_logits], dim=-2)
        predictions = (
            logits,
            anchors,
            anchor_mask,
            fused_flat,
            (
                batch_size,
                anchor_count,
                class_count,
                sequence_length,
            ),
        )
        return predictions, {}

    def _compute_span_relation_scores(
        self,
        feature_embeddings,
        span_idx,
        anchors,
        fused_flat,
        dims,
        prediction_extra,
    ):
        """Score supplied span candidates with the regular fused hypotheses."""

        batch_size, anchor_count, class_count, _ = dims
        span_rep = self.span_rep_layer(feature_embeddings, span_idx)
        span_count = span_rep.shape[1]
        head_span_rep = self.head_span_proj(span_rep)
        tail_span_rep = self.tail_span_proj(span_rep)
        head_span_scores = torch.einsum(
            "BSD,BND->BSN",
            head_span_rep,
            fused_flat,
        )
        tail_span_scores = torch.einsum(
            "BSD,BND->BSN",
            tail_span_rep,
            fused_flat,
        )
        head_span = head_span_scores.reshape(
            batch_size,
            span_count,
            anchor_count,
            class_count,
        )
        tail_span = tail_span_scores.reshape(
            batch_size,
            span_count,
            anchor_count,
            class_count,
        )
        return torch.stack([head_span, tail_span], dim=-1)

    def forward(self, shared, dependency_outputs, flat_inputs=None,
                base_loss_fn=None, **batch):
        open_rel_labels = batch.get("open_rel_labels")
        open_rel_count = batch.get("open_rel_count")
        threshold = batch.get("threshold", 0.5)

        # Span representation inputs
        span_idx = batch.get("open_rel_span_idx")
        span_mask = batch.get("open_rel_span_mask")
        span_labels = batch.get("open_rel_span_labels")

        feature_embeddings = flat_inputs.words_embedding     # (BN, W, D)
        feature_mask = flat_inputs.mask                      # (BN, W)
        rel_embedding = flat_inputs.child_embedding          # (BN, max_C, D)
        rel_embedding_mask = flat_inputs.child_mask          # (BN, max_C)

        if rel_embedding.shape[1] == 0:
            return TaskHeadOutput()

        (
            (logits, anchors, anchor_mask, fused_flat, (B, X, C, L)),
            prediction_extra,
        ) = self._compute_relation_scores(
            flat_inputs,
            open_rel_count,
            threshold,
        )

        # 6. Optional span representation
        span_logits_out = None
        if self.represent_spans and hasattr(self, "span_rep_layer"):
            if span_idx is not None:
                span_logits_out = self._compute_span_relation_scores(
                    feature_embeddings,
                    span_idx,
                    anchors,
                    fused_flat,
                    (B, X, C, L),
                    prediction_extra,
                )

        # 7. Loss computation
        loss = None
        if open_rel_labels is not None and base_loss_fn is not None:
            X_pred, X_label = logits.shape[1], open_rel_labels.shape[1]
            C_pred, C_label = logits.shape[2], open_rel_labels.shape[2]
            L_pred, L_label = logits.shape[3], open_rel_labels.shape[3]

            min_C = min(C_pred, C_label)
            min_L = min(L_pred, L_label)

            pred = logits[:, :, :min_C, :min_L, :, :]
            labels = pred.new_zeros(pred.shape)
            copy_X = min(X_pred, X_label)
            labels[:, :copy_X] = open_rel_labels[
                :, :copy_X, :min_C, :min_L, :, :
            ]

            all_losses = base_loss_fn(pred, labels)

            # Build masks: anchor × rel_class × word
            inst_mask = anchor_mask[:, :X_pred].float()
            word_mask_f = feature_mask[:, :min_L].float()
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
                min_C_s = min(span_logits_out.shape[3], span_labels.shape[3])

                X_span_pred = span_logits_out.shape[2]
                X_span_label = span_labels.shape[2]
                span_pred = span_logits_out[:, :min_S, :, :min_C_s, :]
                s_labels = span_pred.new_zeros(span_pred.shape)
                copy_X_s = min(X_span_pred, X_span_label)
                s_labels[:, :, :copy_X_s] = span_labels[
                    :, :min_S, :copy_X_s, :min_C_s, :
                ]

                span_losses = base_loss_fn(span_pred, s_labels)
                s_span_mask = span_mask[:, :min_S].float()
                s_inst_mask = anchor_mask[:, :X_span_pred].float()
                s_rel_mask = rel_embedding_mask[:, :min_C_s].float()

                s_full_mask = (
                    s_span_mask[:, :, None, None, None]
                    * s_inst_mask[:, None, :, None, None]
                    * s_rel_mask[:, None, None, :, None]
                )
                span_loss = (span_losses * s_full_mask).sum()
                loss = loss + self.span_loss_coef * span_loss

        output_extra = {
            "anchors": anchors,
            "anchor_mask": anchor_mask,
            "span_logits": span_logits_out,
            "span_idx": span_idx,
            "span_mask": span_mask,
        }
        output_extra.update(prediction_extra)

        return TaskHeadOutput(
            loss=loss,
            logits=logits,
            extra=output_extra,
        )
