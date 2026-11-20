"""Prompt schema as extra cross-attention memory for anchor refinement.

With ``anchor_memory_schema`` a head's anchors (record slots, detection
queries, relation slots, calculation slots, ...) cross-attend to the prompt
schema as well as to their main memory (text words, image or audio tokens,
entities). The schema is the parent token (structure / task marker) and its
child tokens (fields, labels, relation types), appended after the main
memory. Schema tokens have no position: a learned embedding takes its place,
and a positional attention bias on the main memory is padded with 0 (or a
learned per-head bias) for them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .groups import MemorySegment

SCHEMA_TOKENS = ("all", "parent", "children")
SCHEMA_NORMS = ("none", "layernorm")
_DEFAULTS = {"tokens": "all", "norm": "none", "head_bias": False}


def parse_schema_memory_spec(spec: Any, *, component_name: str = "anchor_memory_schema"):
    """Normalize ``false | true | {tokens, norm, head_bias}``; None when off.

    ``tokens``: which schema tokens to append ("all" = parent + children).
    ``norm``: "layernorm" normalizes schema tokens before appending (useful
    when the main memory is normalized differently, e.g. image tokens).
    ``head_bias``: learn one additive attention bias per head for schema keys,
    so few schema tokens are not drowned out by a long main memory.
    """

    if spec is None or spec is False:
        return None
    if spec is True:
        return dict(_DEFAULTS)
    if not isinstance(spec, Mapping):
        raise TypeError(f"{component_name} must be a boolean or a mapping")
    options = dict(spec)
    enabled = options.pop("enabled", True)
    if not isinstance(enabled, bool):
        raise TypeError(f"{component_name}.enabled must be a boolean")
    unknown = set(options) - set(_DEFAULTS)
    if unknown:
        raise ValueError(
            f"Unsupported {component_name} options: {sorted(unknown)}. "
            f"Available: {sorted(_DEFAULTS) + ['enabled']}"
        )
    if not enabled:
        return None
    parsed = {**_DEFAULTS, **options}
    if parsed["tokens"] not in SCHEMA_TOKENS:
        raise ValueError(f"{component_name}.tokens must be one of {list(SCHEMA_TOKENS)}")
    if parsed["norm"] not in SCHEMA_NORMS:
        raise ValueError(f"{component_name}.norm must be one of {list(SCHEMA_NORMS)}")
    if not isinstance(parsed["head_bias"], bool):
        raise TypeError(f"{component_name}.head_bias must be a boolean")
    return parsed


class SchemaMemory(nn.Module):
    """Builds the schema :class:`MemorySegment` for one refinement decoder."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        tokens: str = "all",
        norm: str = "none",
        head_bias: bool = False,
    ):
        super().__init__()
        self.tokens = tokens
        # Added to schema keys (and values, as main-memory positions are) in
        # place of a position, so queries can tell schema from main memory.
        self.embedding = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.norm = nn.LayerNorm(hidden_size) if norm == "layernorm" else None
        self.head_bias = nn.Parameter(torch.zeros(num_heads)) if head_bias else None

    @classmethod
    def from_config(cls, spec, hidden_size: int, num_heads: int) -> SchemaMemory | None:
        parsed = parse_schema_memory_spec(spec)
        if parsed is None:
            return None
        return cls(hidden_size, num_heads, **parsed)

    def forward(
        self,
        parent: torch.Tensor | None,
        children: torch.Tensor | None = None,
        child_mask: torch.Tensor | None = None,
    ) -> MemorySegment:
        """``parent`` is ``(B, D)``; ``children`` ``(B, C, D)`` with ``(B, C)`` mask."""

        parts, masks = [], []
        if self.tokens in ("all", "parent") and parent is not None:
            parts.append(parent.unsqueeze(1))
            masks.append(torch.ones(parent.shape[0], 1, dtype=torch.bool, device=parent.device))
        if self.tokens in ("all", "children") and children is not None:
            parts.append(children)
            masks.append(
                child_mask.to(device=children.device).bool()
                if child_mask is not None
                else torch.ones(children.shape[:2], dtype=torch.bool, device=children.device)
            )
        if not parts:
            raise ValueError("anchor_memory_schema found no schema tokens to append")
        dtype = parts[0].dtype
        tokens = torch.cat([part.to(dtype) for part in parts], dim=1)
        if self.norm is not None:
            tokens = self.norm(tokens)
        return MemorySegment(
            tokens=tokens,
            mask=torch.cat(masks, dim=1),
            embedding=self.embedding,
            bias=self.head_bias,
        )


__all__ = ["SchemaMemory", "parse_schema_memory_spec"]
