"""Classification task head — anchor paradigm."""

import torch
from torch import nn

from gliner.modeling.loss_functions import focal_loss_with_logits
from gliner.modeling.utils import extract_prompt_features

from .. import TaskHead, TaskHeadOutput, TaskFlatInputs, SharedRepresentations
from ...layers import Pooling


class ClassificationHead(TaskHead):
    """Classification head using the unified anchor paradigm.

    Flow: anchor_layer(parent_embedding) → anchor_modeling(anchor, cat_embedding)
    → squeeze(1) → dot product with pooled text → scores (B, C).
    """

    name = "classification"
    dependencies = []

    def __init__(self, config, hidden_size, dropout, shared_layers=None):
        super().__init__()
        cat_cfg = config.classification_config
        self.loss_coef = cat_cfg.loss_coef
        self.cat_token_index = cat_cfg.cat_token_index
        self.embed_cat_token = cat_cfg.embed_cat_token
        if shared_layers is None:
            shared_layers = {}

        self._init_anchor_pipeline(cat_cfg, config, hidden_size, dropout, shared_layers)

        self.pooling = Pooling.from_config(
            pooling_type=getattr(cat_cfg, "pooling_type", "mean"),
            hidden_size=hidden_size,
        )

    @classmethod
    def from_config(cls, config, shared_layers=None, **kwargs):
        if config.classification_config is None:
            return None
        return cls(config, hidden_size=config.hidden_size, dropout=config.dropout,
                   shared_layers=shared_layers)

    def forward(self, shared, dependency_outputs, flat_inputs=None, cat_label_embeds=None, **batch):
        cat_labels = batch.get("cat_labels")

        # Use flat_inputs (BN-indexed) when available
        if flat_inputs is not None:
            cat_embedding = flat_inputs.child_embedding
            cat_embedding_mask = flat_inputs.child_mask
            text_rep = self.pooling(flat_inputs.words_embedding, flat_inputs.mask)
            context = flat_inputs.parent_embedding  # (BN, D)
        else:
            token_embeds = shared.token_embeds
            input_ids = shared.input_ids
            attention_mask = shared.attention_mask
            batch_size, _, embed_dim = token_embeds.shape

            if cat_label_embeds is not None:
                cat_embedding = cat_label_embeds
                cat_embedding_mask = torch.ones(
                    cat_embedding.shape[:-1], dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            else:
                cat_embedding, cat_embedding_mask = extract_prompt_features(
                    self.cat_token_index, token_embeds, input_ids, attention_mask,
                    batch_size, embed_dim, self.embed_cat_token,
                )

            text_rep = self.pooling(token_embeds, attention_mask)
            context = cat_embedding.mean(dim=1)  # (B, D)

        # Anchor paradigm: anchor_layer → anchor_modeling → dot product
        anchor_rep, anchor_mask = self.anchor_layer(context)
        if hasattr(self, "anchor_refine"):
            anchor_rep = self.anchor_refine(anchor_rep, cat_embedding)
        fused = self.anchor_modeling(anchor_rep, cat_embedding)  # (B, 1, C, D)
        fused = fused.squeeze(1)  # (B, C, D) — parent mode always A=1
        scores = torch.einsum("bd,bcd->bc", text_rep, fused)

        loss = None
        if cat_labels is not None:
            all_losses = focal_loss_with_logits(scores, cat_labels)
            valid_mask = cat_embedding_mask
            all_losses = all_losses * valid_mask
            loss = all_losses.sum()

        return TaskHeadOutput(loss=loss, logits=scores)
