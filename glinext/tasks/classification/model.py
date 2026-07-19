"""Classification task head — anchor paradigm."""

import torch
from torch import nn

from gliner.modeling.loss_functions import focal_loss_with_logits

from .. import TaskHead, TaskHeadOutput
from ...layers import Pooling
from .scorer import ClassificationScorer


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

        self.scorer = ClassificationScorer.from_config(
            scorer_type=getattr(cat_cfg, "scorer_type", "dot"),
            hidden_size=hidden_size,
        )

    @classmethod
    def from_config(cls, config, shared_layers=None, **kwargs):
        if config.classification_config is None:
            return None
        return cls(config, hidden_size=config.hidden_size, dropout=config.dropout,
                   shared_layers=shared_layers)

    def forward(self, shared, dependency_outputs, flat_inputs=None, **batch):
        cat_labels = batch.get("cat_labels")
        base_loss_fn = batch.get("base_loss_fn")

        cat_embedding = flat_inputs.child_embedding      # (BN, max_C, D)
        cat_embedding_mask = flat_inputs.child_mask       # (BN, max_C)
        feature_embeddings = flat_inputs.words_embedding  # (BN, W, D)
        feature_mask = flat_inputs.mask                   # (BN, W)
        text_rep = self.pooling(feature_embeddings, feature_mask)
        context = flat_inputs.parent_embedding            # (BN, D)

        # Anchor paradigm: anchor_layer → anchor_modeling → dot product
        anchor_rep, anchor_mask = self.anchor_layer(context, feature_embeddings, feature_mask=feature_mask)
        if hasattr(self, "anchor_refine"):
            anchor_rep = self.anchor_refine(anchor_rep, feature_embeddings, token_mask=feature_mask)
        fused = self._reduce_fused_anchors(
            self.anchor_modeling(anchor_rep, cat_embedding),
            anchor_mask,
        )
        scores = self.scorer(text_rep, fused)

        loss = None
        if cat_labels is not None:
            loss_fn = base_loss_fn or focal_loss_with_logits
            all_losses = loss_fn(scores, cat_labels)
            valid_mask = cat_embedding_mask
            all_losses = all_losses * valid_mask
            loss = all_losses.sum()

        return TaskHeadOutput(loss=loss, logits=scores)
