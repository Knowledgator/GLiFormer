"""Classification task head."""

import torch
from torch import nn

from gliner.modeling.loss_functions import focal_loss_with_logits
from gliner.modeling.utils import extract_prompt_features

from .. import TaskHead, TaskHeadOutput, SharedRepresentations
from ...layers import FeaturesProjector, Pooling


class ClassificationScorer(nn.Module):
    """Scores text against class label embeddings."""

    def __init__(self, hidden_size: int, scorer_type: str = "dot", dropout: float = 0.1):
        super().__init__()
        self.scorer_type = scorer_type

        if scorer_type == "weighted-dot":
            self.proj_text = nn.Linear(hidden_size, hidden_size * 2)
            self.proj_label = nn.Linear(hidden_size, hidden_size * 2)
            self.out_mlp = nn.Sequential(
                nn.Linear(hidden_size * 3, hidden_size * 4),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(hidden_size * 4, 1),
            )
        elif scorer_type == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(hidden_size * 4, 1),
            )

    def forward(self, text_rep: torch.Tensor, label_rep: torch.Tensor) -> torch.Tensor:
        if self.scorer_type == "weighted-dot":
            B, D = text_rep.shape
            C = label_rep.shape[1]
            text_proj = self.proj_text(text_rep).view(B, 1, 2, D)
            label_proj = self.proj_label(label_rep).view(B, C, 2, D)
            text_proj = text_proj.expand(-1, C, -1, -1)
            cat = torch.cat([text_proj[:, :, 0], label_proj[:, :, 0],
                             text_proj[:, :, 1] * label_proj[:, :, 1]], dim=-1)
            return self.out_mlp(cat).squeeze(-1)
        elif self.scorer_type == "dot":
            return torch.einsum("bd,bcd->bc", text_rep, label_rep)
        elif self.scorer_type == "mlp":
            return self.mlp(label_rep).squeeze(-1)


class ClassificationHead(TaskHead):
    """Classification head: text-vs-label scorer."""

    name = "classification"
    dependencies = []

    def __init__(self, config, hidden_size, dropout):
        super().__init__()
        cat_cfg = config.classification_config
        self.loss_coef = cat_cfg.loss_coef
        self.cat_token_index = cat_cfg.cat_token_index
        self.embed_cat_token = cat_cfg.embed_cat_token

        self.pooling = Pooling.from_config(
            pooling_type=getattr(cat_cfg, "pooling_type", "mean"),
            hidden_size=hidden_size,
        )
        self.cat_scorer = ClassificationScorer(
            hidden_size, scorer_type=cat_cfg.layer_type, dropout=dropout,
        )
        self.cat_projector = FeaturesProjector(config)

    @classmethod
    def from_config(cls, config, **kwargs):
        if config.classification_config is None:
            return None
        return cls(config, hidden_size=config.hidden_size, dropout=config.dropout)

    def forward(self, shared, dependency_outputs, cat_label_embeds=None, **batch):
        token_embeds = shared.token_embeds
        input_ids = shared.input_ids
        attention_mask = shared.attention_mask
        cat_labels = batch.get("cat_labels")

        batch_size, _, embed_dim = token_embeds.shape

        if cat_label_embeds is not None:
            cat_embedding = self.cat_projector(cat_label_embeds)
            cat_embedding_mask = torch.ones(
                cat_embedding.shape[:-1], dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
        else:
            cat_embedding, cat_embedding_mask = extract_prompt_features(
                self.cat_token_index, token_embeds, input_ids, attention_mask,
                batch_size, embed_dim, self.embed_cat_token,
            )
            cat_embedding = self.cat_projector(cat_embedding)

        text_rep = self.pooling(token_embeds, attention_mask)

        scores = self.cat_scorer(text_rep, cat_embedding)

        loss = None
        if cat_labels is not None:
            all_losses = focal_loss_with_logits(scores, cat_labels)
            valid_mask = cat_embedding_mask
            all_losses = all_losses * valid_mask
            loss = all_losses.sum()

        return TaskHeadOutput(loss=loss, logits=scores)
