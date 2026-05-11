"""Configurable sequence pooling layers."""

import torch
from torch import nn


class Pooling(nn.Module):
    """Base class for sequence pooling. Use `Pooling.from_config()` to construct."""

    _registry = {}

    def __init_subclass__(cls, pooling_type: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if pooling_type is not None:
            Pooling._registry[pooling_type] = cls

    @staticmethod
    def from_config(pooling_type: str = "mean", hidden_size: int = 0) -> "Pooling":
        cls = Pooling._registry.get(pooling_type)
        if cls is None:
            raise ValueError(f"Unknown pooling type: {pooling_type!r}. "
                             f"Available: {list(Pooling._registry)}")
        return cls(hidden_size=hidden_size)

    def forward(self, token_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MeanPooling(Pooling, pooling_type="mean"):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, token_embeds, attention_mask):
        mask_f = attention_mask.unsqueeze(-1).to(dtype=token_embeds.dtype)
        return (token_embeds * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)


class CLSPooling(Pooling, pooling_type="cls"):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, token_embeds, attention_mask):
        return token_embeds[:, 0]


class MaxPooling(Pooling, pooling_type="max"):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, token_embeds, attention_mask):
        filled = token_embeds.masked_fill(~attention_mask.bool().unsqueeze(-1), -1e9)
        return filled.max(dim=1).values


class WeightedPooling(Pooling, pooling_type="weighted"):

    def __init__(self, hidden_size: int = 0, **kwargs):
        super().__init__()
        self.pool_weights = nn.Linear(hidden_size, 1)

    def forward(self, token_embeds, attention_mask):
        weights = self.pool_weights(token_embeds).squeeze(-1)
        weights = weights.masked_fill(~attention_mask.bool(), -1e9)
        weights = torch.softmax(weights, dim=1).unsqueeze(-1)
        return (token_embeds * weights).sum(dim=1)
