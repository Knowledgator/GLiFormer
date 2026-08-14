"""Classification task head — anchor paradigm."""


import torch

from ...layers import Pooling
from .. import TaskHead, TaskHeadOutput
from ..losses import binary_focal_or_bce
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

        self._init_anchor_components(cat_cfg, config, hidden_size, dropout, shared_layers)

        self.pooling_type = getattr(cat_cfg, "pooling_type", "mean")
        self.pooling = Pooling.from_config(
            pooling_type=self.pooling_type,
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

    def _pool_text(self, shared, flat_inputs):
        """Pool one source representation for every classification group.

        ``flat_inputs.words_embedding`` contains source *words* only, so its
        first position is not the transformer CLS token.  CLS pooling must use
        the raw encoder sequence and expand it through ``batch_origin`` for
        items that contain multiple classification groups.  This also remains
        defined when a long schema prompt consumes the source-word budget.
        """

        if self.pooling_type == "cls" and shared.token_embeds.shape[1] > 0:
            batch_origin = flat_inputs.batch_origin.to(
                device=shared.token_embeds.device,
                dtype=torch.long,
            )
            return shared.token_embeds.index_select(0, batch_origin)[:, 0]
        return self.pooling(
            flat_inputs.words_embedding,
            flat_inputs.mask,
        )

    def forward(self, shared, dependency_outputs, flat_inputs=None, **batch):
        cat_labels = batch.get("cat_labels")
        base_loss_fn = batch.get("base_loss_fn")

        cat_embedding = flat_inputs.child_embedding      # (BN, max_C, D)
        cat_embedding_mask = flat_inputs.child_mask       # (BN, max_C)
        feature_embeddings = flat_inputs.words_embedding  # (BN, W, D)
        feature_mask = flat_inputs.mask                   # (BN, W)
        text_rep = self._pool_text(shared, flat_inputs)
        context = flat_inputs.parent_embedding            # (BN, D)

        # Anchor paradigm: anchor_layer → anchor_modeling → dot product
        anchor_rep, anchor_mask = self._generate_anchors(
            context,
            feature_embeddings,
            feature_mask=feature_mask,
        )
        anchor_rep = self._refine_anchors(
            anchor_rep,
            feature_embeddings,
            memory_mask=feature_mask,
            anchor_mask=anchor_mask,
        )
        fused = self._reduce_fused_anchors(
            self._model_anchors(
                anchor_rep,
                cat_embedding,
                anchor_mask=anchor_mask,
                child_mask=cat_embedding_mask,
            ),
            anchor_mask,
        )
        scores = self.scorer(text_rep, fused)

        loss = None
        if cat_labels is not None:
            loss_fn = base_loss_fn or binary_focal_or_bce
            all_losses = loss_fn(scores, cat_labels)
            valid_mask = cat_embedding_mask
            all_losses = all_losses * valid_mask
            loss = all_losses.sum()

        return TaskHeadOutput(loss=loss, logits=scores)
