"""Count task head."""

import torch
from torch import nn
from torch.nn import functional as F

from .. import TaskHead, TaskHeadOutput


class CountModule(nn.Module):
    """Predicts entity count per sample — regression or classification."""

    def __init__(self, hidden_size: int, mode: str = "regression", max_count: int = 20):
        super().__init__()
        self.mode = mode
        self.max_count = max_count

        if mode == "classification":
            self.head = nn.Linear(hidden_size, max_count + 1)
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.mode == "classification":
            targets_cls = targets.long().clamp(0, self.max_count)
            return F.cross_entropy(logits, targets_cls)
        else:
            return F.mse_loss(logits.squeeze(-1), targets.float())


class CountHead(TaskHead):
    """Count prediction head from mean-pooled prompt embeddings."""

    name = "count"
    dependencies = []

    def __init__(self, config, hidden_size):
        super().__init__()
        count_cfg = config.count_config
        self.loss_coef = count_cfg.loss_coef
        self.count_head = CountModule(
            hidden_size, mode=count_cfg.mode, max_count=count_cfg.max_count,
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        if config.count_config is None:
            return None
        return cls(config, hidden_size=config.hidden_size)

    def forward(self, shared, dependency_outputs, flat_inputs=None, **batch):
        count_targets = batch.get("count_targets")

        pooled = flat_inputs.parent_embedding  # (BN, D)

        logits = self.count_head(pooled)

        loss = None
        if count_targets is not None:
            loss = self.count_head.loss(logits, count_targets)

        return TaskHeadOutput(loss=loss, logits=logits)
