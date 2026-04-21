"""Joint NER + Relation Extraction head (GLiNER-relex style).

Inherits NER scoring from NERHead and adds adjacency-based entity pair
scoring against [REL] type embeddings. Runs NER first, selects entity
spans, builds an adjacency matrix, and scores entity pairs for relations.
"""

from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from gliner.modeling.loss_functions import focal_loss_with_logits
from gliner.modeling.multitask.relations_layers import RelationsRepLayer
from gliner.modeling.multitask.triples_layers import TriplesScoreLayer
from gliner.modeling.utils import (
    build_entity_pairs, extract_prompt_features, extract_spans_from_tokens,
)

from .. import TaskHeadOutput, SharedRepresentations
from ..ner.model import NERHead
from ...layers import PairRepLayer


class JointRelexHead(NERHead):
    """Joint NER + relation extraction via adjacency-based entity pair scoring.

    Inherits NER forward pass from NERHead, then:
    1. Selects entity spans from NER scores
    2. Builds adjacency matrix between entities
    3. Scores entity pairs against [REL] type embeddings
    """

    name = "joint_relex"
    dependencies = []  # NER is built-in, not an external dependency

    def __init__(self, config, hidden_size, dropout, shared_layers=None):
        super().__init__(config, hidden_size, dropout, shared_layers=shared_layers)
        rel_cfg = config.joint_relex_config
        self.rel_loss_coef = rel_cfg.loss_coef
        self.adjacency_loss_coef = rel_cfg.adjacency_loss_coef
        self.rel_token_index = rel_cfg.rel_token_index
        self.embed_rel_token = rel_cfg.embed_rel_token

        # Relation-specific layers (adjacency mode)
        self.relations_rep_layer = RelationsRepLayer(
            in_dim=hidden_size, relation_mode=rel_cfg.layer_type,
        )
        if rel_cfg.triples_layer is not None:
            self.triples_score_layer = TriplesScoreLayer(rel_cfg.triples_layer)
        else:
            self.pair_rep_layer = PairRepLayer(
                hidden_size, pair_rep_type=rel_cfg.pair_rep_type, dropout=dropout,
            )

    @classmethod
    def from_config(cls, config, shared_layers=None, **kwargs):
        if config.joint_relex_config is None:
            return None
        return cls(config, hidden_size=config.hidden_size, dropout=config.dropout,
                   shared_layers=shared_layers)

    @staticmethod
    def _pool_entity_spans(words_embedding, span_idx, span_mask):
        """Mean-pool token embeddings over each [start, end] span (end inclusive)."""
        B, E, _ = span_idx.shape
        W = words_embedding.shape[1]
        device = words_embedding.device

        start = span_idx[..., 0].clamp(min=0, max=max(W - 1, 0)).unsqueeze(-1)
        end = span_idx[..., 1].clamp(min=0, max=max(W - 1, 0)).unsqueeze(-1)
        positions = torch.arange(W, device=device).view(1, 1, W)
        span_pos_mask = (positions >= start) & (positions <= end)
        span_pos_mask = span_pos_mask & span_mask.unsqueeze(-1).bool()
        span_pos_mask_f = span_pos_mask.to(words_embedding.dtype)

        summed = torch.einsum("BEW,BWD->BED", span_pos_mask_f, words_embedding)
        counts = span_pos_mask_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        pooled = summed / counts
        pooled = pooled * span_mask.unsqueeze(-1).to(pooled.dtype)
        return pooled, span_mask.to(torch.long)

    def _get_rel_prompts(self, shared, rel_label_embeds, flat_rel_prompts, flat_rel_prompts_mask):
        """Return per-group (BN, C_rel, D) [REL] embeddings.

        Prefers processor/model-provided flat tensors (already split per
        extraction group). Falls back to batch-level extraction when those
        aren't available (e.g. unit tests calling the head directly).
        """
        if flat_rel_prompts is not None:
            return flat_rel_prompts, flat_rel_prompts_mask
        if rel_label_embeds is not None:
            rel_prompts = rel_label_embeds
            rel_mask = torch.ones(
                rel_label_embeds.shape[:-1], dtype=shared.attention_mask.dtype,
                device=shared.attention_mask.device,
            )
            return rel_prompts, rel_mask
        batch_size = shared.token_embeds.shape[0]
        embed_dim = shared.token_embeds.shape[2]
        rel_prompts, rel_mask = extract_prompt_features(
            self.rel_token_index, shared.token_embeds, shared.input_ids,
            shared.attention_mask, batch_size, embed_dim, self.embed_rel_token,
        )
        return rel_prompts, rel_mask

    def _forward_adjacency(self, shared, target_span_rep, target_span_mask,
                            rel_labels, adjacency_threshold, rel_label_embeds,
                            flat_rel_prompts=None, flat_rel_prompts_mask=None):
        """Adjacency matrix + entity pair relation scoring."""
        B, E_ent, D = target_span_rep.shape
        pred_adj_matrix = self.relations_rep_layer(target_span_rep, target_span_mask)

        rel_prompts, rel_prompts_mask = self._get_rel_prompts(
            shared, rel_label_embeds, flat_rel_prompts, flat_rel_prompts_mask,
        )
        C_rel = rel_prompts.size(1)

        if rel_labels is not None:
            adj_matrix = (rel_labels.sum(dim=-1) > 0).float()
            adj_for_selection = adj_matrix
        else:
            adj_matrix = None
            adj_for_selection = pred_adj_matrix

        pair_idx, pair_mask, head_rep, tail_rep = build_entity_pairs(
            adj_for_selection, target_span_rep, threshold=adjacency_threshold,
        )
        N = head_rep.size(1)
        pair_scores = None

        if hasattr(self, "pair_rep_layer"):
            pair_rep = self.pair_rep_layer(head_rep, tail_rep)
            pair_scores = torch.einsum("BND,BCD->BNC", pair_rep, rel_prompts)
        elif hasattr(self, "triples_score_layer"):
            h = head_rep.unsqueeze(2).expand(B, N, C_rel, D)
            t = tail_rep.unsqueeze(2).expand(B, N, C_rel, D)
            r = rel_prompts.unsqueeze(1).expand(B, N, C_rel, D)
            pair_scores = self.triples_score_layer(
                h.reshape(-1, D), r.reshape(-1, D), t.reshape(-1, D),
            ).view(B, N, C_rel)

        loss = None
        if rel_labels is not None and pair_scores is not None:
            adj_mask = target_span_mask.float().unsqueeze(1) * target_span_mask.float().unsqueeze(2)
            adj_logits = pred_adj_matrix.unsqueeze(-1).view(B, -1, 1)
            adj_labels = adj_matrix.unsqueeze(-1).view(B, -1, 1)
            adj_loss = (focal_loss_with_logits(adj_logits, adj_labels) * adj_mask.unsqueeze(-1).view(B, -1, 1)).sum()

            head_indices = pair_idx[..., 0].clamp(min=0)
            tail_indices = pair_idx[..., 1].clamp(min=0)
            batch_idx_t = torch.arange(B, device=rel_labels.device).unsqueeze(1)
            rel_matrix = rel_labels[batch_idx_t, head_indices, tail_indices]

            if rel_matrix.size(-1) < C_rel:
                rel_matrix = F.pad(rel_matrix, (0, C_rel - rel_matrix.size(-1)))
            elif rel_matrix.size(-1) > C_rel:
                rel_matrix = rel_matrix[..., :C_rel]

            rel_mask_expanded = pair_mask.unsqueeze(-1).expand(B, N, C_rel)
            class_mask = rel_prompts_mask.unsqueeze(1).expand(B, N, C_rel)
            combined = rel_mask_expanded * class_mask
            rel_loss = (focal_loss_with_logits(pair_scores, rel_matrix) * combined).sum()

            loss = adj_loss * self.adjacency_loss_coef + rel_loss * self.rel_loss_coef

        return TaskHeadOutput(
            loss=loss,
            logits=pair_scores,
            extra={"rel_idx": pair_idx, "rel_mask": pair_mask},
        )

    def forward(self, shared, dependency_outputs, flat_inputs=None, base_loss_fn=None,
                rel_label_embeds=None, flat_rel_prompts=None, flat_rel_prompts_mask=None,
                **batch):
        # 1. Run NER forward (inherited) — passes flat_inputs through
        ner_output = super().forward(
            shared, dependency_outputs, flat_inputs=flat_inputs, base_loss_fn=base_loss_fn, **batch,
        )

        # 2. Select entity spans.
        #    Training: use processor-provided per-entity spans so target_span_rep
        #    is aligned with rel_labels' entity_id axis.
        #    Inference: fall back to NER-extracted spans.
        ner_scores = ner_output.logits
        words_embedding = ner_output.extra.get("words_embedding", shared.words_embedding)

        rel_span_idx = batch.get("rel_span_idx")
        rel_span_mask = batch.get("rel_span_mask")

        if rel_span_idx is not None and rel_span_mask is not None:
            target_span_rep, target_span_mask = self._pool_entity_spans(
                words_embedding, rel_span_idx, rel_span_mask,
            )
        else:
            span_idx = ner_output.extra.get("span_idx")
            span_mask = ner_output.extra.get("span_mask")
            if span_idx is None or span_mask is None:
                span_idx, span_mask = extract_spans_from_tokens(
                    ner_scores, labels=None, threshold=batch.get("threshold", 0.5),
                )
            target_span_rep, target_span_mask = self._pool_entity_spans(
                words_embedding, span_idx, span_mask,
            )

        # 3. Build adjacency + score relation types
        rel_output = self._forward_adjacency(
            shared, target_span_rep, target_span_mask,
            batch.get("rel_labels"), batch.get("adjacency_threshold", 0.5),
            rel_label_embeds,
            flat_rel_prompts=flat_rel_prompts,
            flat_rel_prompts_mask=flat_rel_prompts_mask,
        )

        # 4. Combine losses
        combined_loss = None
        if ner_output.loss is not None or rel_output.loss is not None:
            ner_loss = ner_output.loss if ner_output.loss is not None else 0.0
            rel_loss = rel_output.loss if rel_output.loss is not None else 0.0
            combined_loss = ner_loss + rel_loss

        # 5. Return merged output
        return TaskHeadOutput(
            loss=combined_loss,
            logits=ner_output.logits,
            extra={
                **ner_output.extra,
                "rel_logits": rel_output.logits,
                "rel_idx": rel_output.extra.get("rel_idx"),
                "rel_mask": rel_output.extra.get("rel_mask"),
            },
        )
