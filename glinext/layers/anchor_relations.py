"""Anchor-slot selection of directed entity pairs."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn


class AnchorPairRelationsOutput(NamedTuple):
    """Pair-selection tensors emitted by :class:`AnchorPairRelationsLayer`."""

    assignment_logits: torch.Tensor
    pair_idx: torch.Tensor
    pair_mask: torch.Tensor
    anchors: torch.Tensor
    anchor_mask: torch.Tensor


class AnchorPairRelationsLayer(nn.Module):
    """Map a bounded set of anchors to source/target entity indices.

    Anchor acquisition and optional refinement produce ``A`` relation slots.
    The configured anchor-modeling strategy conditions each slot on two
    learned endpoint roles (source and target).  The resulting endpoint
    queries are multiplied by the entity representations, producing
    ``(B, A, E, 2)`` assignment logits.  Hard entity indices are exposed as
    ``(B, A, 2)`` for the downstream MLP or triples scorer.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        anchor_layer: nn.Module,
        anchor_normalizer: nn.Module,
        anchor_modeling: nn.Module,
        max_anchors: int,
        anchor_refinement: nn.Module | None = None,
        refinement_positions: nn.Module | None = None,
        self_attention_bias: nn.Module | None = None,
        cross_attention_bias: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.anchor_layer = anchor_layer
        self.anchor_normalizer = anchor_normalizer
        self.anchor_modeling = anchor_modeling
        self.anchor_refinement = anchor_refinement
        self.refinement_positions = refinement_positions
        self.self_attention_bias = self_attention_bias
        self.cross_attention_bias = cross_attention_bias
        self.max_anchors = int(max_anchors)
        if self.max_anchors <= 0:
            raise ValueError("max_anchors must be positive")

        self.endpoint_roles = nn.Parameter(
            torch.empty(2, self.hidden_size)
        )
        nn.init.orthogonal_(self.endpoint_roles)
        self.assignment_scale = math.sqrt(self.hidden_size)

    @staticmethod
    def _deduplicate_pairs(
        pair_idx: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask repeated directed pairs while retaining their first slot."""

        pair_mask = pair_mask.clone()
        for batch_idx in range(pair_idx.shape[0]):
            seen: set[tuple[int, int]] = set()
            for anchor_idx in torch.where(pair_mask[batch_idx])[0].tolist():
                pair = tuple(pair_idx[batch_idx, anchor_idx].tolist())
                if pair in seen:
                    pair_mask[batch_idx, anchor_idx] = False
                else:
                    seen.add(pair)
        pair_idx = pair_idx.masked_fill(~pair_mask.unsqueeze(-1), -1)
        return pair_idx, pair_mask

    def _anchor_count(self, context_embedding: torch.Tensor) -> torch.Tensor | None:
        # Fixed-width layers already expose their full slot table when count is
        # absent. Count-driven layers must also use the configured capacity in
        # both training and inference; gold pair counts must never leak here.
        if hasattr(self.anchor_layer, "num_slots"):
            return None
        return torch.full(
            (context_embedding.shape[0],),
            self.max_anchors,
            dtype=torch.long,
            device=context_embedding.device,
        )

    def _generate_anchors(
        self,
        context_embedding: torch.Tensor,
        entity_representations: torch.Tensor,
        entity_mask: torch.Tensor,
        threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = context_embedding.shape[0]
        context_mask = torch.ones(
            batch_size,
            1,
            dtype=torch.bool,
            device=context_embedding.device,
        )
        memory = torch.cat(
            [context_embedding.unsqueeze(1), entity_representations],
            dim=1,
        )
        memory_mask = torch.cat(
            [context_mask, entity_mask.to(device=context_embedding.device).bool()],
            dim=1,
        )
        anchors, anchor_mask = self.anchor_layer(
            context_embedding,
            memory,
            count=self._anchor_count(context_embedding),
            threshold=threshold,
            feature_mask=memory_mask,
        )
        # ``A`` is a capacity, including for feature-derived anchor strategies.
        anchors = anchors[:, : self.max_anchors]
        anchor_mask = anchor_mask[:, : self.max_anchors].bool()
        anchors = self.anchor_normalizer(
            anchors,
            anchor_mask,
            source_embeddings=memory,
            source_mask=memory_mask,
        )
        if self.anchor_refinement is not None:
            anchors = self.anchor_refinement(
                anchors,
                memory,
                token_mask=memory_mask,
                query_mask=anchor_mask,
                position_encoding=self.refinement_positions,
                self_attention_bias_module=self.self_attention_bias,
                cross_attention_bias_module=self.cross_attention_bias,
            )
        return anchors, anchor_mask

    def forward(
        self,
        context_embedding: torch.Tensor,
        entity_representations: torch.Tensor,
        entity_mask: torch.Tensor,
        *,
        threshold: float = 0.5,
        deduplicate: bool = True,
    ) -> AnchorPairRelationsOutput:
        if entity_representations.dim() != 3:
            raise ValueError(
                "entity_representations must have shape (B, E, D)"
            )
        if entity_mask.shape != entity_representations.shape[:2]:
            raise ValueError("entity_mask must have shape (B, E)")
        if context_embedding.shape != (
            entity_representations.shape[0],
            entity_representations.shape[2],
        ):
            raise ValueError("context embedding must have shape (B, D)")

        anchors, anchor_mask = self._generate_anchors(
            context_embedding,
            entity_representations,
            entity_mask,
            threshold,
        )
        batch_size, anchor_count, hidden_size = anchors.shape
        entity_count = entity_representations.shape[1]
        role_embeddings = self.endpoint_roles.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        role_mask = torch.ones(
            batch_size,
            2,
            dtype=torch.bool,
            device=anchors.device,
        )
        endpoint_queries = self.anchor_modeling(
            anchors,
            role_embeddings,
            anchor_mask=anchor_mask,
            child_mask=role_mask,
        )
        if endpoint_queries.shape != (
            batch_size,
            anchor_count,
            2,
            hidden_size,
        ):
            raise ValueError(
                "anchor modeling must produce endpoint queries with shape "
                "(B, A, 2, D)"
            )

        assignment_logits = torch.einsum(
            "BAKD,BED->BAEK",
            endpoint_queries,
            entity_representations,
        ) / self.assignment_scale
        valid_entities = entity_mask.to(device=anchors.device).bool()
        if entity_count == 0:
            pair_idx = torch.full(
                (batch_size, anchor_count, 2),
                -1,
                dtype=torch.long,
                device=anchors.device,
            )
            pair_mask = torch.zeros(
                batch_size,
                anchor_count,
                dtype=torch.bool,
                device=anchors.device,
            )
            return AnchorPairRelationsOutput(
                assignment_logits,
                pair_idx,
                pair_mask,
                anchors,
                anchor_mask,
            )

        assignment_logits = assignment_logits.masked_fill(
            ~valid_entities[:, None, :, None],
            torch.finfo(assignment_logits.dtype).min,
        )
        pair_idx = assignment_logits.argmax(dim=2)
        selected_valid = valid_entities.gather(
            1,
            pair_idx.reshape(batch_size, -1),
        ).reshape(batch_size, anchor_count, 2)
        pair_mask = (
            anchor_mask
            & selected_valid.all(dim=-1)
            & (pair_idx[..., 0] != pair_idx[..., 1])
        )
        pair_idx = pair_idx.masked_fill(~pair_mask.unsqueeze(-1), -1)
        if deduplicate:
            pair_idx, pair_mask = self._deduplicate_pairs(
                pair_idx,
                pair_mask,
            )
        return AnchorPairRelationsOutput(
            assignment_logits,
            pair_idx,
            pair_mask,
            anchors,
            anchor_mask,
        )


__all__ = ["AnchorPairRelationsLayer", "AnchorPairRelationsOutput"]
