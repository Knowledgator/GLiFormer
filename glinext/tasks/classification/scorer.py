"""Classification scorers — registry-based factory for text × label scoring."""

import torch
from torch import nn


class ClassificationScorer(nn.Module):
    """Base class for classification scorers. Use `ClassificationScorer.from_config()` to construct."""

    _registry = {}

    def __init_subclass__(cls, scorer_type: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if scorer_type is not None:
            ClassificationScorer._registry[scorer_type] = cls

    @staticmethod
    def from_config(scorer_type: str = "dot", hidden_size: int = 0) -> "ClassificationScorer":
        cls = ClassificationScorer._registry.get(scorer_type)
        if cls is None:
            raise ValueError(f"Unknown scorer type: {scorer_type!r}. "
                             f"Available: {list(ClassificationScorer._registry)}")
        return cls(hidden_size=hidden_size)

    def forward(self, text_rep: torch.Tensor, label_rep: torch.Tensor) -> torch.Tensor:
        """Score text against label representations.

        Args:
            text_rep: (B, D) pooled text representation.
            label_rep: (B, C, D) label representations.

        Returns:
            scores: (B, C) classification scores.
        """
        raise NotImplementedError


class DotScorer(ClassificationScorer, scorer_type="dot"):
    """Simple dot product scoring."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, text_rep, label_rep):
        return torch.einsum("bd,bcd->bc", text_rep, label_rep)


class WeightedDotScorer(ClassificationScorer, scorer_type="weighted-dot"):
    """Learnable weighted dot product with projection and MLP fusion."""

    def __init__(self, hidden_size: int = 0, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.proj_text = nn.Linear(hidden_size, hidden_size * 2)
        self.proj_label = nn.Linear(hidden_size, hidden_size * 2)
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 4),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, 1),
        )

    def forward(self, text_rep, label_rep):
        batch_size, hidden_size = text_rep.shape
        num_classes = label_rep.shape[1]

        text_rep = self.proj_text(text_rep).view(batch_size, 1, 1, 2, hidden_size)
        label_rep = self.proj_label(label_rep).view(batch_size, 1, num_classes, 2, hidden_size)

        text_rep = text_rep.expand(-1, -1, num_classes, -1, -1).permute(3, 0, 1, 2, 4)
        label_rep = label_rep.expand(-1, 1, -1, -1, -1).permute(3, 0, 1, 2, 4)

        cat = torch.cat([text_rep[0], label_rep[0], text_rep[1] * label_rep[1]], dim=-1)
        return self.out_mlp(cat).view(batch_size, num_classes)


class MLPScorer(ClassificationScorer, scorer_type="mlp"):
    """MLP-based scoring on concatenated text + label representations."""

    def __init__(self, hidden_size: int = 0, **kwargs):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, text_rep, label_rep):
        batch_size, num_labels, dim = label_rep.shape
        text_exp = text_rep.unsqueeze(1).expand(batch_size, num_labels, dim)
        combined = torch.cat([text_exp, label_rep], dim=-1)
        return self.mlp(combined).squeeze(-1)


class HopfieldScorer(ClassificationScorer, scorer_type="hopfield"):
    """Hopfield-style attention-based iterative scoring."""

    def __init__(self, hidden_size: int = 0, beta: float = 4.0, num_iterations: int = 1, **kwargs):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.beta = beta
        self.num_iterations = num_iterations

    def forward(self, text_rep, label_rep):
        for _ in range(self.num_iterations):
            text_exp = text_rep.unsqueeze(1)  # (B, 1, D)
            query = self.q_proj(label_rep)    # (B, C, D)
            key = self.k_proj(text_exp)       # (B, 1, D)
            value = self.v_proj(text_exp)     # (B, 1, D)

            attn = torch.bmm(query, key.transpose(1, 2))  # (B, C, 1)
            attn = (attn * self.beta).softmax(dim=1)
            context = attn * value  # (B, C, D)
            label_rep = label_rep + context

        return self.mlp(label_rep).squeeze(-1)  # (B, C)
