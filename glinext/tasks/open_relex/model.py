"""Open relation extraction head (GLiNER2 style).

Standalone anchor-based head — no NER dependency. Uses configurable anchor
layers to generate relation instance slots, fuses with [REL] type embeddings,
and extracts head/tail spans via dual AnchoredSpanScorers.
"""

import torch

from gliner.modeling.utils import extract_prompt_features

from .. import TaskHead, TaskHeadOutput, SharedRepresentations
from ...layers import AnchoredSpanScorer, AnchorLayer, AnchorModeling


class OpenRelexHead(TaskHead):
    """Anchor-based relation extraction via dual head/tail span scoring.

    For each (anchor, rel_type) pair, extracts both a head entity span
    and a tail entity span in the text.

    Output logits shape: (B, X, C, L, 2, 3)
        X = anchors, C = rel classes, L = seq len, 2 = [head, tail], 3 = [start, inside, end]
    """

    name = "open_relex"
    dependencies = []

    def __init__(self, config, hidden_size, dropout):
        super().__init__()
        cfg = config.open_relex_config
        self.loss_coef = cfg.loss_coef
        self.rel_token_index = cfg.rel_token_index
        self.embed_rel_token = cfg.embed_rel_token

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
        self.anchor_modeling = AnchorModeling.from_config(
            cfg.anchor_modeling, hidden_size, dropout=dropout,
        )

        # Dual scorers: one for head spans, one for tail spans
        self.head_scorer = AnchoredSpanScorer(hidden_size, dropout=dropout)
        self.tail_scorer = AnchoredSpanScorer(hidden_size, dropout=dropout)

    @classmethod
    def from_config(cls, config, **kwargs):
        if config.open_relex_config is None:
            return None
        return cls(config, hidden_size=config.hidden_size, dropout=config.dropout)

    def forward(self, shared, dependency_outputs,
                open_rel_label_embeds=None, base_loss_fn=None, **batch):
        token_embeds = shared.token_embeds
        input_ids = shared.input_ids
        attention_mask = shared.attention_mask
        words_embedding = shared.words_embedding
        mask = shared.mask
        prompts_embedding = shared.prompts_embedding

        open_rel_labels = batch.get("open_rel_labels")
        open_rel_count = batch.get("open_rel_count")
        threshold = batch.get("threshold", 0.5)

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
            prompts_embedding, words_embedding,
            count=open_rel_count, threshold=threshold,
        )

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

        # 6. Loss computation
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

        return TaskHeadOutput(
            loss=loss,
            logits=logits,
            extra={
                "anchors": anchors,
                "anchor_mask": anchor_mask,
            },
        )
