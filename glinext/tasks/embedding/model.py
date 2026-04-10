"""Embedding similarity task head."""

import torch
from torch import nn
from torch.nn import functional as F

from .. import TaskHead, TaskHeadOutput
from ...layers import Pooling
from ...layers.mlp import create_mlp


class EmbeddingLoss(nn.Module):
    """Base class for embedding loss functions. Use `EmbeddingLoss.from_config()` to construct."""

    _registry = {}

    def __init_subclass__(cls, loss_fn: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if loss_fn is not None:
            EmbeddingLoss._registry[loss_fn] = cls

    @staticmethod
    def from_config(loss_fn: str = "mse", **kwargs) -> "EmbeddingLoss":
        cls = EmbeddingLoss._registry.get(loss_fn)
        if cls is None:
            raise ValueError(f"Unknown loss function: {loss_fn!r}. "
                             f"Available: {list(EmbeddingLoss._registry)}")
        return cls(**kwargs)

    def forward(self, similarities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MSELoss(EmbeddingLoss, loss_fn="mse"):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, similarities, labels):
        return F.mse_loss(similarities, labels.float())


class ContrastiveLoss(EmbeddingLoss, loss_fn="contrastive"):

    def __init__(self, margin: float = 1.0, **kwargs):
        super().__init__()
        self.margin = margin

    def forward(self, similarities, labels):
        positive = labels.float() * (1 - similarities).pow(2)
        negative = (1 - labels.float()) * F.relu(similarities - self.margin).pow(2)
        return (positive + negative).mean()


class TripletLoss(EmbeddingLoss, loss_fn="triplet"):
    """Falls back to MSE without explicit triplets."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, similarities, labels):
        return F.mse_loss(similarities, labels.float())


class EmbeddingHead(TaskHead):
    """Semantic similarity via configurable pooling, similarity, and loss functions.

    During training, pair texts are encoded as a separate batch through the
    shared encoder.  The head receives pre-encoded token embeddings
    (``embedding_encodings``) and corresponding attention mask, pools them,
    and computes pairwise similarity + loss.

    During inference, the head uses the shared encoder output
    (``shared.words_embedding``) to produce pooled text embeddings.
    """

    name = "embedding"
    dependencies = []

    def __init__(self, config):
        super().__init__()
        emb_cfg = config.embedding_config
        self.loss_coef = emb_cfg.loss_coef
        self.similarity_fn = getattr(emb_cfg, "similarity_fn", "cosine")

        projection_dim = getattr(emb_cfg, "projection_dim", None)
        if projection_dim is not None:
            self.projection = create_mlp(config.hidden_size, [config.hidden_size], projection_dim)
        else:
            self.projection = None

        pooling_hidden_size = projection_dim if projection_dim is not None else config.hidden_size
        self.pooling = Pooling.from_config(
            pooling_type=getattr(emb_cfg, "pooling_type", "mean"),
            hidden_size=pooling_hidden_size,
        )
        self.loss = EmbeddingLoss.from_config(
            loss_fn=getattr(emb_cfg, "loss_fn", "mse"),
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        if config.embedding_config is None:
            return None
        return cls(config)

    def _project(self, embeddings):
        if self.projection is not None:
            return self.projection(embeddings)
        return embeddings

    def _similarity(self, emb_a, emb_b):
        if self.similarity_fn == "dot":
            return (emb_a * emb_b).sum(dim=-1)
        elif self.similarity_fn == "l2":
            return -torch.norm(emb_a - emb_b, p=2, dim=-1)
        else:  # "cosine"
            emb_a = F.normalize(emb_a, p=2, dim=-1)
            emb_b = F.normalize(emb_b, p=2, dim=-1)
            return (emb_a * emb_b).sum(dim=-1)

    def forward(self, shared, dependency_outputs, **batch):
        embedding_labels = batch.get("embedding_labels")
        embedding_pair_idx = batch.get("embedding_pair_idx")
        embedding_encodings = batch.get("embedding_encodings")
        embedding_encoding_mask = batch.get("embedding_encoding_mask")

        # Training path: pair texts encoded as a separate batch
        if embedding_encodings is not None and embedding_pair_idx is not None:
            mask = embedding_encoding_mask.bool() if embedding_encoding_mask is not None else None
            pooled = self.pooling(self._project(embedding_encodings), mask)
            if self.similarity_fn == "cosine":
                pooled = F.normalize(pooled, p=2, dim=-1)

            idx_a = embedding_pair_idx[:, 0]
            idx_b = embedding_pair_idx[:, 1]
            emb_a = pooled[idx_a]
            emb_b = pooled[idx_b]

            similarities = self._similarity(emb_a, emb_b)

            loss = None
            if embedding_labels is not None:
                loss = self.loss(similarities, embedding_labels)

            return TaskHeadOutput(loss=loss, logits=similarities)

        # Inference / fallback path: use shared encoder output
        if embedding_pair_idx is not None:
            words_embedding = shared.words_embedding
            mask = shared.mask

            pooled = self.pooling(self._project(words_embedding), mask)
            if self.similarity_fn == "cosine":
                pooled = F.normalize(pooled, p=2, dim=-1)

            idx_a = embedding_pair_idx[:, 0]
            idx_b = embedding_pair_idx[:, 1]
            emb_a = pooled[idx_a]
            emb_b = pooled[idx_b]

            similarities = self._similarity(emb_a, emb_b)

            loss = None
            if embedding_labels is not None:
                loss = self.loss(similarities, embedding_labels)

            return TaskHeadOutput(loss=loss, logits=similarities)

        # No embedding data — return pooled embeddings for downstream use
        return TaskHeadOutput()