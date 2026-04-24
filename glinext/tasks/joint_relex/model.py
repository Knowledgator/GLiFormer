"""Joint NER + Relation Extraction head (GLiNER-relex style).

Inherits NER scoring from NERHead and adds relation scoring against [REL]
type embeddings. When enabled, the adjacency layer filters candidate pairs;
otherwise the head falls back to scoring all directed entity pairs, matching
the original GLiNER behavior.
"""

from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from gliner.modeling.loss_functions import focal_loss_with_logits
from gliner.modeling.multitask.relations_layers import RelationsRepLayer
from gliner.modeling.multitask.triples_layers import TriplesScoreLayer
from gliner.modeling.span_rep import SpanRepLayer
from gliner.modeling.utils import (
    build_all_entity_pairs,
    build_entity_pairs,
    extract_prompt_features,
    extract_spans_from_tokens,
)

from .. import TaskHeadOutput, SharedRepresentations
from ..ner.model import NERHead
from ...layers import PairRepLayer


class JointRelexHead(NERHead):
    """Joint NER + relation extraction with optional adjacency pair selection.

    Inherits NER forward pass from NERHead, then:
    1. Selects entity spans from NER scores
    2. Optionally builds an adjacency matrix between entities
    3. Scores candidate entity pairs against [REL] type embeddings
    """

    name = "joint_relex"
    dependencies = []  # NER is built-in, not an external dependency

    def __init__(self, config, hidden_size, dropout, shared_layers=None):
        super().__init__(config, hidden_size, dropout, shared_layers=shared_layers)
        rel_cfg = config.joint_relex_config
        self.ner_loss_coef = getattr(config.ner_config, "loss_coef", 1.0)
        self.rel_loss_coef = rel_cfg.loss_coef
        self.adjacency_loss_coef = rel_cfg.adjacency_loss_coef
        self.rel_token_index = rel_cfg.rel_token_index
        self.embed_rel_token = rel_cfg.embed_rel_token
        # This head combines its own NER and relation losses internally. Keep
        # the orchestrator from multiplying the combined loss a second time.
        self.loss_coef = 1.0
        self.rel_span_rep_layer = SpanRepLayer(
            span_mode="token_level",
            hidden_size=hidden_size,
            max_width=getattr(config, "max_width", 12),
            dropout=dropout,
        )

        # ``layer_type='none'`` keeps the relation head active but skips
        # adjacency prediction so all directed entity pairs are scored.
        if rel_cfg.layer_type not in (None, "none"):
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

    def _pool_entity_spans(self, words_embedding, span_idx, span_mask):
        """Represent entity spans with GLiNER-relex's token-level SpanRepLayer."""
        span_idx = span_idx * span_mask.unsqueeze(-1).long()
        span_rep = self.rel_span_rep_layer(words_embedding, span_idx)
        span_rep = span_rep * span_mask.unsqueeze(-1).to(span_rep.dtype)
        return span_rep, span_mask.to(torch.long)

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

    def _forward_relations(self, shared, target_span_rep, target_span_mask,
                           rel_labels, adjacency_threshold, rel_label_embeds,
                           flat_rel_prompts=None, flat_rel_prompts_mask=None,
                           rel_pair_mask=None, base_loss_fn=None, loss_kwargs=None):
        """Relation scoring with optional adjacency-based pair selection."""
        B, E_ent, D = target_span_rep.shape
        use_adjacency = hasattr(self, "relations_rep_layer")
        pred_adj_matrix = None
        if use_adjacency:
            pred_adj_matrix = self.relations_rep_layer(target_span_rep, target_span_mask)

        rel_prompts, rel_prompts_mask = self._get_rel_prompts(
            shared, rel_label_embeds, flat_rel_prompts, flat_rel_prompts_mask,
        )
        C_rel = rel_prompts.size(1)

        if rel_labels is not None:
            positive_adj_matrix = (rel_labels.sum(dim=-1) > 0).float()
            if rel_pair_mask is not None:
                adj_matrix = rel_pair_mask.float()
            else:
                adj_matrix = positive_adj_matrix
        else:
            adj_matrix = None

        if use_adjacency:
            adj_for_selection = adj_matrix if adj_matrix is not None else pred_adj_matrix
            pair_idx, pair_mask, head_rep, tail_rep = build_entity_pairs(
                adj_for_selection, target_span_rep, threshold=adjacency_threshold,
            )
        else:
            pair_idx, pair_mask, head_rep, tail_rep = build_all_entity_pairs(
                target_span_rep, target_span_mask,
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
            if loss_kwargs is None:
                loss_kwargs = {}
            rel_losses = self._call_elementwise_loss(
                base_loss_fn, pair_scores, rel_matrix, normalize_prob=True, **loss_kwargs,
            )
            rel_loss = (rel_losses * combined).sum()

            if use_adjacency and pred_adj_matrix is not None and adj_matrix is not None:
                adj_mask = target_span_mask.float().unsqueeze(1) * target_span_mask.float().unsqueeze(2)
                adj_logits = pred_adj_matrix.unsqueeze(-1).view(B, -1, 1)
                adj_labels = adj_matrix.unsqueeze(-1).view(B, -1, 1)
                adj_losses = self._call_elementwise_loss(
                    base_loss_fn, adj_logits, adj_labels, normalize_prob=False, **loss_kwargs,
                )
                adj_loss = (adj_losses * adj_mask.unsqueeze(-1).view(B, -1, 1)).sum()
                loss = adj_loss * self.adjacency_loss_coef + rel_loss * self.rel_loss_coef
            else:
                loss = rel_loss * self.rel_loss_coef

        return TaskHeadOutput(
            loss=loss,
            logits=pair_scores,
            extra={"rel_idx": pair_idx, "rel_mask": pair_mask},
        )

    @staticmethod
    def _relation_loss_kwargs(batch):
        """Mirror GLiNER-relex relation-loss kwargs, including rel_* overrides."""
        kwargs = {}
        for out_key, sources in (
            ("alpha", ("rel_alpha", "alpha")),
            ("gamma", ("rel_gamma", "gamma")),
            ("prob_margin", ("rel_prob_margin", "prob_margin")),
            ("label_smoothing", ("rel_label_smoothing", "label_smoothing")),
            ("negatives", ("rel_negatives", "negatives")),
            ("masking", ("rel_masking", "masking")),
        ):
            for source in sources:
                value = batch.get(source)
                if value is not None:
                    kwargs[out_key] = value
                    break
        return kwargs

    @staticmethod
    def _call_elementwise_loss(loss_fn, logits, labels, normalize_prob=True, **kwargs):
        """Call BaseModel._loss when available, falling back to focal loss.

        Unit tests sometimes pass focal_loss_with_logits directly; production
        GLiNExTModel passes BaseModel._loss, which supports negative sampling.
        """
        if loss_fn is not None:
            try:
                return loss_fn(logits, labels, normalize_prob=normalize_prob, **kwargs)
            except TypeError:
                supported = {
                    "alpha", "gamma", "prob_margin", "label_smoothing",
                }
                fallback_kwargs = {k: v for k, v in kwargs.items() if k in supported}
                return loss_fn(logits, labels, normalize_prob=normalize_prob, **fallback_kwargs)
        supported = {
            "alpha", "gamma", "prob_margin", "label_smoothing",
        }
        fallback_kwargs = {k: v for k, v in kwargs.items() if k in supported}
        return focal_loss_with_logits(
            logits, labels, normalize_prob=normalize_prob, **fallback_kwargs,
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

        # 3. Build candidate pairs and score relation types
        rel_output = self._forward_relations(
            shared, target_span_rep, target_span_mask,
            batch.get("rel_labels"), batch.get("adjacency_threshold", 0.5),
            rel_label_embeds,
            flat_rel_prompts=flat_rel_prompts,
            flat_rel_prompts_mask=flat_rel_prompts_mask,
            rel_pair_mask=batch.get("rel_pair_mask"),
            base_loss_fn=base_loss_fn,
            loss_kwargs=self._relation_loss_kwargs(batch),
        )

        # 4. Combine losses
        combined_loss = None
        if ner_output.loss is not None or rel_output.loss is not None:
            ner_loss = ner_output.loss * self.ner_loss_coef if ner_output.loss is not None else 0.0
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
                "rel_entity_spans": rel_span_idx if rel_span_idx is not None else span_idx,
            },
        )
