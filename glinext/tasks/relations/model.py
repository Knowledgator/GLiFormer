"""Relations task head."""

from typing import Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from gliner.modeling.loss_functions import focal_loss_with_logits
from gliner.modeling.multitask.relations_layers import RelationsRepLayer
from gliner.modeling.multitask.triples_layers import TriplesScoreLayer
from gliner.modeling.utils import build_entity_pairs, extract_prompt_features

from .. import TaskHead, TaskHeadOutput, SharedRepresentations
from ...layers import PairRepLayer, PromptRelationExtractor


class RelationsHead(TaskHead):
    """Relation extraction head: adjacency + relation type scoring."""

    name = "relations"
    dependencies = ["ner"]

    def __init__(self, config, hidden_size, dropout):
        super().__init__()
        rel_cfg = config.relations_config
        self.loss_coef = rel_cfg.loss_coef
        self.adjacency_loss_coef = rel_cfg.adjacency_loss_coef
        self.rel_loss_coef = rel_cfg.loss_coef
        self.rel_mode = rel_cfg.rel_mode
        self.rel_token_index = rel_cfg.rel_token_index
        self.embed_rel_token = rel_cfg.embed_rel_token

        if rel_cfg.rel_mode == "adjacency":
            self.relations_rep_layer = RelationsRepLayer(
                in_dim=hidden_size, relation_mode=rel_cfg.layer_type,
            )
            if rel_cfg.triples_layer is not None:
                self.triples_score_layer = TriplesScoreLayer(rel_cfg.triples_layer)
            else:
                self.pair_rep_layer = PairRepLayer(
                    hidden_size, pair_rep_type=rel_cfg.pair_rep_type, dropout=dropout,
                )
        elif rel_cfg.rel_mode == "prompt":
            self.prompt_rel_extractor = PromptRelationExtractor(
                hidden_size, dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown rel_mode: {rel_cfg.rel_mode}")

    @classmethod
    def from_config(cls, config, **kwargs):
        if config.relations_config is None:
            return None
        return cls(config, hidden_size=config.hidden_size, dropout=config.dropout)

    def _select_entity_spans(self, scores, words_embedding, ner_labels=None,
                              threshold=0.5, top_k=None):
        B, W, _, _ = scores.shape
        D = words_embedding.shape[-1]
        token_confidence = torch.sigmoid(scores).max(dim=-1).values.max(dim=-1).values

        if ner_labels is not None:
            keep = (ner_labels.sum(dim=(-1, -2)) > 0) if ner_labels.dim() == 4 else (ner_labels.sum(dim=-1) > 0)
        else:
            keep = token_confidence > threshold

        if top_k is not None:
            sel_scores = token_confidence.masked_fill(~keep, -1.0)
            top_idx = sel_scores.topk(k=min(top_k, W), dim=1).indices
            keep = torch.zeros_like(keep)
            keep.scatter_(1, top_idx, True)

        rep_mask = keep.long()
        lengths = rep_mask.sum(dim=-1)
        max_len = max(lengths.max().item(), 1)

        target_rep = words_embedding.new_zeros(B, max_len, D)
        target_mask = rep_mask.new_zeros(B, max_len)

        if rep_mask.any():
            new_col_idx = (rep_mask.cumsum(dim=1) - 1)
            batch_idx, old_col_idx = torch.where(rep_mask.bool())
            new_col = new_col_idx[rep_mask.bool()]
            target_rep[batch_idx, new_col] = words_embedding[batch_idx, old_col_idx]
            target_mask[batch_idx, new_col] = 1

        return target_rep, target_mask

    def forward(self, shared, dependency_outputs, rel_label_embeds=None, **batch):
        ner_output = dependency_outputs.get("ner")
        if ner_output is None:
            return TaskHeadOutput()

        ner_scores = ner_output.logits
        words_embedding = ner_output.extra.get("words_embedding", shared.words_embedding)
        ner_labels = batch.get("ner_labels")
        rel_labels = batch.get("rel_labels")
        threshold = batch.get("threshold", 0.5)
        adjacency_threshold = batch.get("adjacency_threshold", 0.5)

        target_span_rep, target_span_mask = self._select_entity_spans(
            ner_scores, words_embedding, ner_labels, threshold,
        )

        if self.rel_mode == "adjacency":
            return self._forward_adjacency(
                shared, target_span_rep, target_span_mask,
                rel_labels, adjacency_threshold, rel_label_embeds,
            )
        else:
            return self._forward_prompt(
                shared, target_span_rep, target_span_mask,
                rel_labels, adjacency_threshold, rel_label_embeds,
            )

    def _get_rel_prompts(self, shared, rel_label_embeds):
        if rel_label_embeds is not None:
            rel_prompts = rel_label_embeds
            rel_mask = torch.ones(
                rel_label_embeds.shape[:-1], dtype=shared.attention_mask.dtype,
                device=shared.attention_mask.device,
            )
        else:
            batch_size = shared.token_embeds.shape[0]
            embed_dim = shared.token_embeds.shape[2]
            rel_prompts, rel_mask = extract_prompt_features(
                self.rel_token_index, shared.token_embeds, shared.input_ids,
                shared.attention_mask, batch_size, embed_dim, self.embed_rel_token,
            )
        return rel_prompts, rel_mask

    def _forward_adjacency(self, shared, target_span_rep, target_span_mask,
                            rel_labels, adjacency_threshold, rel_label_embeds):
        if not hasattr(self, "relations_rep_layer"):
            return TaskHeadOutput()

        B, E_ent, D = target_span_rep.shape
        pred_adj_matrix = self.relations_rep_layer(target_span_rep, target_span_mask)

        rel_prompts, rel_prompts_mask = self._get_rel_prompts(shared, rel_label_embeds)
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

    def _forward_prompt(self, shared, target_span_rep, target_span_mask,
                         rel_labels, adjacency_threshold, rel_label_embeds):
        if not hasattr(self, "prompt_rel_extractor"):
            return TaskHeadOutput()

        B, E_ent, D = target_span_rep.shape
        rel_prompts, rel_prompts_mask = self._get_rel_prompts(shared, rel_label_embeds)
        C_rel = rel_prompts.size(1)

        full_scores = self.prompt_rel_extractor(
            target_span_rep, rel_prompts, target_span_mask,
        )

        loss = None
        if rel_labels is not None:
            C_label = rel_labels.size(-1)
            E_label = rel_labels.size(1)

            scores_for_loss = full_scores[:, :E_label, :E_label, :C_rel]
            labels_for_loss = rel_labels[..., :C_rel]

            if scores_for_loss.size(-1) > labels_for_loss.size(-1):
                labels_for_loss = F.pad(labels_for_loss, (0, scores_for_loss.size(-1) - labels_for_loss.size(-1)))
            elif labels_for_loss.size(-1) > scores_for_loss.size(-1):
                labels_for_loss = labels_for_loss[..., :scores_for_loss.size(-1)]

            entity_mask = target_span_mask[:, :E_label].float()
            pair_valid = entity_mask.unsqueeze(2) * entity_mask.unsqueeze(1)
            class_valid = rel_prompts_mask[:, :scores_for_loss.size(-1)]
            full_mask = pair_valid.unsqueeze(-1) * class_valid[:, None, None, :]

            loss = (focal_loss_with_logits(scores_for_loss, labels_for_loss) * full_mask).sum()
            loss = loss * self.rel_loss_coef

        with torch.no_grad():
            pseudo_adj = torch.sigmoid(full_scores).max(dim=-1).values
            pseudo_adj = pseudo_adj * (
                ~torch.eye(E_ent, device=pseudo_adj.device, dtype=torch.bool)
            ).unsqueeze(0).float()

        pair_idx, pair_mask, _, _ = build_entity_pairs(
            pseudo_adj, target_span_rep, threshold=adjacency_threshold,
        )
        N = pair_idx.size(1)

        head_indices = pair_idx[..., 0].clamp(min=0)
        tail_indices = pair_idx[..., 1].clamp(min=0)
        batch_idx_t = torch.arange(B, device=full_scores.device).unsqueeze(1)
        pair_scores = full_scores[batch_idx_t, head_indices, tail_indices]

        if pair_scores.size(-1) < C_rel:
            pair_scores = F.pad(pair_scores, (0, C_rel - pair_scores.size(-1)))
        elif pair_scores.size(-1) > C_rel:
            pair_scores = pair_scores[..., :C_rel]

        return TaskHeadOutput(
            loss=loss,
            logits=pair_scores,
            extra={"rel_idx": pair_idx, "rel_mask": pair_mask},
        )
