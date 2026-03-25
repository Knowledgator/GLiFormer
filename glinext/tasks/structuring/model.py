"""Structuring task head."""

import torch

from gliner.modeling.utils import extract_prompt_features

from .. import TaskHead, TaskHeadOutput, SharedRepresentations
from ...layers import AnchoredSpanScorer, AnchorLayer, AnchorModeling


class StructuringHead(TaskHead):
    """Structuring via anchor-based span extraction using configurable anchor layer."""

    name = "structuring"
    dependencies = []

    def __init__(self, config, hidden_size, dropout):
        super().__init__()
        struct_cfg = config.structuring_config
        self.loss_coef = struct_cfg.loss_coef
        self.child_token_index = struct_cfg.child_token_index
        self.embed_child_token = struct_cfg.embed_child_token

        # Map groups_layer config to anchor_mode for AnchorLayer factory
        anchor_mode = struct_cfg.groups_layer
        if anchor_mode == "lstm":
            anchor_mode = "rotary"

        self.anchor_layer = AnchorLayer.from_config(
            anchor_mode, hidden_size,
            max_count=struct_cfg.max_count,
            num_slots=getattr(struct_cfg, "num_fixed_slots", 10),
            num_heads=struct_cfg.groups_num_heads,
            num_layers=struct_cfg.groups_num_layers,
            dropout=dropout,
        )

        anchor_modeling_type = getattr(struct_cfg, "anchor_modeling", "linear")
        self.anchor_modeling = AnchorModeling.from_config(
            anchor_modeling_type, hidden_size, dropout=dropout,
        )

        self.anchored_scorer = AnchoredSpanScorer(hidden_size, dropout=dropout)

    @classmethod
    def from_config(cls, config, **kwargs):
        if config.structuring_config is None:
            return None
        return cls(config, hidden_size=config.hidden_size, dropout=config.dropout)

    def forward(self, shared, dependency_outputs, child_label_embeds=None,
                base_loss_fn=None, **batch):
        token_embeds = shared.token_embeds
        input_ids = shared.input_ids
        attention_mask = shared.attention_mask
        words_embedding = shared.words_embedding
        mask = shared.mask
        prompts_embedding = shared.prompts_embedding
        prompts_embedding_mask = shared.prompts_embedding_mask

        gold_count_val = batch.get("gold_count_val")
        structuring_labels = batch.get("structuring_labels")
        structuring_count = batch.get("structuring_count")
        threshold = batch.get("threshold", 0.5)

        batch_size, _, embed_dim = token_embeds.shape

        if child_label_embeds is not None:
            child_embedding = child_label_embeds
            child_embedding_mask = torch.ones(
                child_label_embeds.shape[:-1], dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
        else:
            child_embedding, child_embedding_mask = extract_prompt_features(
                self.child_token_index, token_embeds, input_ids, attention_mask,
                batch_size, embed_dim, self.embed_child_token,
            )

        if child_embedding.shape[1] == 0:
            return TaskHeadOutput()

        count_for_groups = structuring_count if structuring_count is not None else gold_count_val
        anchors, anchor_mask = self.anchor_layer(
            prompts_embedding, words_embedding,
            count=count_for_groups, threshold=threshold,
        )

        # Fuse anchors + children: (B, X, C, D) then score against words
        fused = self.anchor_modeling(anchors, child_embedding)
        B, X, C, D = fused.shape
        fused_flat = fused.view(B, X * C, D)
        # Score fused reps against word embeddings: (B, X*C, L, 3)
        structuring_logits_flat = self.anchored_scorer(
            fused_flat, words_embedding, word_mask=mask,
        )
        structuring_logits = structuring_logits_flat.view(B, X, C, -1, 3)

        loss = None
        if structuring_labels is not None and base_loss_fn is not None:
            X_pred = structuring_logits.shape[1]
            X_label = structuring_labels.shape[1]
            L_pred = structuring_logits.shape[2]
            L_label = structuring_labels.shape[2]
            C_pred = structuring_logits.shape[3]
            C_label = structuring_labels.shape[3]

            min_X = min(X_pred, X_label)
            min_L = min(L_pred, L_label)
            min_C = min(C_pred, C_label)

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

        return TaskHeadOutput(
            loss=loss,
            logits=structuring_logits,
            extra={
                "groups_output": anchors,
                "anchor_mask": anchor_mask,
            },
        )
