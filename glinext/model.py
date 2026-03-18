"""GLiNExT: unified multi-task model for NER, classification, relations, structuring, counting, and decoding."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers.utils import ModelOutput

from gliner.modeling.base import BaseModel
from gliner.modeling.encoder import Encoder, BiEncoder
from gliner.modeling.decoder import Decoder
from gliner.modeling.layers import CrossFuser, LstmSeq2SeqEncoder, create_projection_layer
from gliner.modeling.scorers import Scorer
from gliner.modeling.span_rep import SpanRepLayer
from gliner.modeling.loss_functions import cross_entropy_loss, focal_loss_with_logits
from gliner.modeling.multitask.relations_layers import RelationsRepLayer
from gliner.modeling.multitask.triples_layers import TriplesScoreLayer
from gliner.modeling.utils import (
    build_entity_pairs,
    extract_prompt_features,
    extract_word_embeddings,
    extract_spans_from_tokens,
    extract_prompt_features_and_word_embeddings,
)

from .config import GLiNextConfig
from .layers import (
    AnchoredSpanScorer,
    FeaturesProjector,
    PairRepLayer,
    PromptRelationExtractor,
    RotaryGroupLSTM,
    QueryGroupLSTM,
    QueryGroupTransformer,
)


@dataclass
class GLiNExTOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    # NER
    ner_logits: Optional[torch.FloatTensor] = None
    span_logits: Optional[torch.FloatTensor] = None
    span_idx: Optional[torch.LongTensor] = None
    span_mask: Optional[torch.Tensor] = None
    # Classification
    cat_logits: Optional[torch.FloatTensor] = None
    # Relations
    rel_logits: Optional[torch.FloatTensor] = None
    rel_idx: Optional[torch.LongTensor] = None
    rel_mask: Optional[torch.Tensor] = None
    # Count
    count_logits: Optional[torch.FloatTensor] = None
    # Groups / Structuring
    groups_output: Optional[torch.FloatTensor] = None
    groups_mask: Optional[torch.Tensor] = None
    # Structuring (anchor-based span extraction)
    structuring_logits: Optional[torch.FloatTensor] = None
    structuring_anchor_mask: Optional[torch.Tensor] = None
    # Embedding similarity
    embedding_logits: Optional[torch.FloatTensor] = None
    # Embeddings (for downstream use)
    words_embedding: Optional[torch.FloatTensor] = None
    mask: Optional[torch.LongTensor] = None
    prompts_embedding: Optional[torch.FloatTensor] = None
    prompts_embedding_mask: Optional[torch.LongTensor] = None


class ClassificationScorer(nn.Module):
    """Scores text against class label embeddings.

    Supports:
      - 'dot': simple dot product
      - 'weighted-dot': projected bilinear interaction
    """
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
        """
        Args:
            text_rep: (B, D) pooled text representation.
            label_rep: (B, C, D) class label embeddings.
        Returns:
            scores: (B, C)
        """
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

class CountHead(nn.Module):
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
        """
        Args:
            x: (B, D) or (N, D) pooled representation.
        Returns:
            regression: (B, 1) predicted counts
            classification: (B, max_count+1) logits
        """
        return self.head(x)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.mode == "classification":
            targets_cls = targets.long().clamp(0, self.max_count)
            return F.cross_entropy(logits, targets_cls)
        else:
            return F.mse_loss(logits.squeeze(-1), targets.float())

class GLiNExTModel(BaseModel):
    """Unified multi-task model composing optional heads:

    - **NER**: Token-level scorer (start/end/inside) — always present.
    - **Classification**: Text-vs-label scorer — enabled by ``classifier_layer``.
    - **Relations**: Adjacency + relation scoring — enabled by ``relations_layer``.
    - **Decoder**: Autoregressive label generation — enabled by ``decoder_model``.
    - **Groups**: Count-aware grouping via GRU/Transformer — enabled by ``groups_layer``.
    - **Count**: Regression or classification count head — enabled by ``count_layer``.
    """

    def __init__(
        self,
        config: GLiNextConfig,
        from_pretrained: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        super().__init__(config, from_pretrained, cache_dir)

        if config.labels_encoder is not None:
            self.token_rep_layer = BiEncoder(config, from_pretrained, cache_dir=cache_dir)
        else:
            self.token_rep_layer = Encoder(config, from_pretrained, cache_dir=cache_dir)

        if config.num_rnn_layers > 0:
            self.rnn = LstmSeq2SeqEncoder(config, num_layers=config.num_rnn_layers)

        if config.post_fusion_schema:
            self.cross_fuser = CrossFuser(
                config.hidden_size,
                config.hidden_size,
                num_heads=self.token_rep_layer.bert_layer.model.config.num_attention_heads,
                num_layers=config.num_post_fusion_layers,
                dropout=config.dropout,
                schema=config.post_fusion_schema,
            )

        self.scorer = Scorer(config.hidden_size, config.dropout)

        if getattr(config, "represent_spans", False):
            self.span_rep_layer = SpanRepLayer(
                span_mode="token_level",
                hidden_size=config.hidden_size,
                max_width=getattr(config, "max_width", 12),
                dropout=config.dropout,
            )

        if config.classifier_layer is not None:
            self.cat_scorer = ClassificationScorer(
                config.hidden_size,
                scorer_type=config.classifier_layer,
                dropout=config.dropout,
            )
            self.cat_projector = FeaturesProjector(config)

        if config.relations_layer is not None:
            if config.rel_mode == "adjacency":
                self.relations_rep_layer = RelationsRepLayer(
                    in_dim=config.hidden_size, relation_mode=config.relations_layer,
                )
                if config.triples_layer is not None:
                    self.triples_score_layer = TriplesScoreLayer(config.triples_layer)
                else:
                    self.pair_rep_layer = PairRepLayer(
                        config.hidden_size,
                        pair_rep_type=config.pair_rep_type,
                        dropout=config.dropout,
                    )
            elif config.rel_mode == "prompt":
                self.prompt_rel_extractor = PromptRelationExtractor(
                    config.hidden_size, dropout=config.dropout,
                )
            else:
                raise ValueError(f"Unknown rel_mode: {config.rel_mode}")

        # ── Decoder head ─────────────────────────────────────────────────
        if config.decoder_model is not None:
            self.decoder = Decoder(config, from_pretrained, cache_dir=cache_dir)
            if config.hidden_size != self.decoder.decoder_hidden_size:
                self._enc2dec_proj = create_projection_layer(
                    config.hidden_size, config.dropout, self.decoder.decoder_hidden_size,
                )

        # ── Count head ───────────────────────────────────────────────────
        if config.count_layer is not None:
            self.count_head = CountHead(
                config.hidden_size,
                mode=config.count_mode,
                max_count=config.max_count,
            )

        # ── Groups head ──────────────────────────────────────────────────
        if config.groups_layer is not None:
            if config.groups_layer == "lstm":
                self.groups_layer = RotaryGroupLSTM(
                    hidden_size=config.hidden_size,
                    max_count=config.max_count,
                )
            elif config.groups_layer == "query_lstm":
                self.groups_layer = QueryGroupLSTM(
                    hidden_size=config.hidden_size,
                    max_count=config.max_count,
                )
            elif config.groups_layer == "query_transformer":
                self.groups_layer = QueryGroupTransformer(
                    hidden_size=config.hidden_size,
                    num_heads=config.groups_num_heads,
                    num_layers=config.groups_num_layers,
                    dropout=config.dropout,
                    max_count=config.max_count,
                )
            else:
                raise ValueError(f"Unknown groups_layer: {config.groups_layer}")

            # Anchored span scorer for structuring (anchor + child → spans)
            self.anchored_scorer = AnchoredSpanScorer(
                config.hidden_size, dropout=config.dropout,
            )

    def get_representations(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text_lengths: torch.Tensor,
        words_mask: torch.Tensor,
        labels_input_ids: Optional[torch.Tensor] = None,
        labels_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run encoder and extract prompt + word embeddings.

        Returns:
            token_embeds, prompts_embedding, prompts_embedding_mask, words_embedding, mask
        """
        encoder_kwargs = {k: kwargs[k] for k in ("packing_config", "pair_attention_mask") if k in kwargs}

        if isinstance(self.token_rep_layer, BiEncoder) and labels_input_ids is not None:
            token_embeds, labels_embeds = self.token_rep_layer(
                input_ids, attention_mask, labels_input_ids, labels_attention_mask, **encoder_kwargs,
            )
            batch_size, _, embed_dim = token_embeds.shape
            max_text_length = text_lengths.max()
            words_embedding, mask = extract_word_embeddings(
                token_embeds, words_mask, attention_mask,
                batch_size, max_text_length, embed_dim, text_lengths,
            )
            labels_embeds = labels_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            labels_mask = torch.ones(labels_embeds.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device)
            if hasattr(self, "cross_fuser"):
                labels_embeds, words_embedding = self.cross_fuser(labels_embeds, words_embedding, labels_mask, mask)
            if hasattr(self, "rnn"):
                words_embedding = self.rnn(words_embedding, mask)
            return token_embeds, labels_embeds, labels_mask, words_embedding, mask
        else:
            token_embeds = self.token_rep_layer(input_ids, attention_mask, **encoder_kwargs)
            prompts_embedding, prompts_embedding_mask, words_embedding, mask = (
                extract_prompt_features_and_word_embeddings(
                    self.config.class_token_index, token_embeds, input_ids, attention_mask,
                    text_lengths, words_mask, self.config.embed_ent_token,
                )
            )
            if hasattr(self, "rnn"):
                words_embedding = self.rnn(words_embedding, mask)
            return token_embeds, prompts_embedding, prompts_embedding_mask, words_embedding, mask

    def _forward_ner(
        self,
        words_embedding: torch.Tensor,
        mask: torch.Tensor,
        prompts_embedding: torch.Tensor,
        prompts_embedding_mask: torch.Tensor,
        ner_labels: Optional[torch.Tensor] = None,
        span_idx: Optional[torch.Tensor] = None,
        span_mask: Optional[torch.Tensor] = None,
        span_labels: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Token-level NER scoring + optional span scoring.

        Returns:
            (loss, ner_scores, span_logits, span_idx, span_mask)
        """
        if ner_labels is not None:
            target_W = ner_labels.shape[1]
            words_embedding, mask = self._fit_length(words_embedding, mask, target_W)
            target_C = max(prompts_embedding.size(1), ner_labels.size(-2))
            prompts_embedding, prompts_embedding_mask = self._fit_length(
                prompts_embedding, prompts_embedding_mask, target_C,
            )

        # (B, W, C, 3) — start, end, inside
        scores = self.scorer(words_embedding, prompts_embedding)

        # Optional span representation
        span_logits_out = None
        if getattr(self.config, "represent_spans", False) and hasattr(self, "span_rep_layer"):
            if span_idx is None:
                span_idx, span_mask = extract_spans_from_tokens(scores, ner_labels, threshold)
                span_idx = span_idx * span_mask.unsqueeze(-1).long()
            span_rep = self.span_rep_layer(words_embedding, span_idx)
            span_logits_out = torch.einsum("BND,BCD->BNC", span_rep, prompts_embedding)

        loss = None
        if ner_labels is not None:
            loss = self._ner_loss(scores, ner_labels, prompts_embedding_mask, mask)
            if span_labels is not None and span_logits_out is not None:
                span_loss = self._ner_loss(span_logits_out, span_labels, prompts_embedding_mask, span_mask)
                loss = self.config.token_loss_coef * loss + self.config.span_loss_coef * span_loss

        return loss, scores, span_logits_out, span_idx, span_mask

    def _ner_loss(self, scores, labels, prompts_embedding_mask, word_mask):
        all_losses = self._loss(scores, labels)
        mask = word_mask.unsqueeze(-1) * prompts_embedding_mask.unsqueeze(1)
        if all_losses.dim() == 4:
            mask = mask.unsqueeze(-1)
        all_losses = all_losses * mask
        return all_losses.sum()

    def _forward_cat(
        self,
        token_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cat_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Classification scoring.

        Extracts [CAT] token embeddings from the shared encoder output and
        scores them against pooled text representations.

        Returns:
            (loss, logits)  — logits shape: (total_cat_groups, C)
        """
        if not hasattr(self, "cat_scorer"):
            return None, None

        batch_size, _, embed_dim = token_embeds.shape

        # Extract class embeddings at [CAT] token positions
        cat_embedding, cat_embedding_mask = extract_prompt_features(
            self.config.cat_token_index, token_embeds, input_ids, attention_mask,
            batch_size, embed_dim, self.config.embed_cat_token,
        )
        cat_embedding = self.cat_projector(cat_embedding)

        # Pool text representation — use mean of token_embeds (masked)
        text_rep = (token_embeds * attention_mask.unsqueeze(-1).float()).sum(dim=1)
        text_rep = text_rep / attention_mask.float().sum(dim=1, keepdim=True).clamp(min=1)

        scores = self.cat_scorer(text_rep, cat_embedding)  # (B, C)

        loss = None
        if cat_labels is not None:
            # cat_labels: (total_cat_groups, C) from processor
            # Need to handle the fact that cat groups may be flattened differently than batch
            all_losses = focal_loss_with_logits(scores, cat_labels)
            valid_mask = cat_embedding_mask  # (B, C)
            all_losses = all_losses * valid_mask
            loss = all_losses.sum()

        return loss, scores

    def _forward_relations(
        self,
        token_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_span_rep: Optional[torch.Tensor] = None,
        target_span_mask: Optional[torch.Tensor] = None,
        rel_labels: Optional[torch.Tensor] = None,
        adjacency_threshold: float = 0.5,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Relation extraction from entity span representations.

        Dispatches to adjacency-based or prompt-based extraction depending on
        ``config.rel_mode``.

        Args:
            token_embeds: (B_enc, S, D) encoder outputs.
            input_ids: (B_enc, S) token ids (for extracting [REL] prompts).
            attention_mask: (B_enc, S).
            target_span_rep: (B, E, D) selected entity representations.
            target_span_mask: (B, E) mask for valid entities.
            rel_labels: (B, E, E, C) ground-truth relation labels.
            adjacency_threshold: threshold for pair selection.

        Returns:
            (loss, pair_scores, pair_idx, pair_mask)
        """
        if target_span_rep is None:
            return None, None, None, None

        if self.config.rel_mode == "adjacency":
            return self._forward_relations_adjacency(
                token_embeds, input_ids, attention_mask,
                target_span_rep, target_span_mask,
                rel_labels, adjacency_threshold,
            )
        elif self.config.rel_mode == "prompt":
            return self._forward_relations_prompt(
                token_embeds, input_ids, attention_mask,
                target_span_rep, target_span_mask,
                rel_labels, adjacency_threshold,
            )
        else:
            raise ValueError(f"Unknown rel_mode: {self.config.rel_mode}")

    def _forward_relations_adjacency(
        self,
        token_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_span_rep: torch.Tensor,
        target_span_mask: torch.Tensor,
        rel_labels: Optional[torch.Tensor],
        adjacency_threshold: float,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Adjacency-based relation extraction (Type 1).

        1. Predict adjacency matrix (which entities are related).
        2. Select entity pairs from adjacency.
        3. Score pairs against relation type embeddings via pair_rep_layer or triples_score_layer.
        """
        if not hasattr(self, "relations_rep_layer"):
            return None, None, None, None

        batch_size, _, embed_dim = token_embeds.shape
        B, E_ent, D = target_span_rep.shape

        pred_adj_matrix = self.relations_rep_layer(target_span_rep, target_span_mask)

        # Extract relation type embeddings
        rel_prompts_embedding, rel_prompts_embedding_mask = extract_prompt_features(
            self.config.rel_token_index, token_embeds, input_ids, attention_mask,
            batch_size, embed_dim, self.config.embed_rel_token,
        )
        C_rel = rel_prompts_embedding.size(1)

        # Derive ground-truth adjacency from rel_labels for pair selection during training
        if rel_labels is not None:
            adj_matrix = (rel_labels.sum(dim=-1) > 0).float()  # (B, E, E)
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
            pair_scores = torch.einsum("BND,BCD->BNC", pair_rep, rel_prompts_embedding)
        elif hasattr(self, "triples_score_layer"):
            h = head_rep.unsqueeze(2).expand(B, N, C_rel, D)
            t = tail_rep.unsqueeze(2).expand(B, N, C_rel, D)
            r = rel_prompts_embedding.unsqueeze(1).expand(B, N, C_rel, D)
            pair_scores = self.triples_score_layer(
                h.reshape(-1, D), r.reshape(-1, D), t.reshape(-1, D),
            ).view(B, N, C_rel)

        loss = None
        if rel_labels is not None and pair_scores is not None:
            # Adjacency loss
            adj_mask = target_span_mask.float().unsqueeze(1) * target_span_mask.float().unsqueeze(2)
            adj_logits = pred_adj_matrix.unsqueeze(-1).view(B, -1, 1)
            adj_labels = adj_matrix.unsqueeze(-1).view(B, -1, 1)
            adj_loss = (focal_loss_with_logits(adj_logits, adj_labels) * adj_mask.unsqueeze(-1).view(B, -1, 1)).sum()

            # Build per-pair relation targets from rel_labels (B, E, E, C)
            head_indices = pair_idx[..., 0].clamp(min=0)  # (B, N)
            tail_indices = pair_idx[..., 1].clamp(min=0)  # (B, N)
            batch_idx = torch.arange(B, device=rel_labels.device).unsqueeze(1)
            rel_matrix = rel_labels[batch_idx, head_indices, tail_indices]  # (B, N, C)

            # Pad/trim relation classes to match prompt count
            if rel_matrix.size(-1) < C_rel:
                rel_matrix = F.pad(rel_matrix, (0, C_rel - rel_matrix.size(-1)))
            elif rel_matrix.size(-1) > C_rel:
                rel_matrix = rel_matrix[..., :C_rel]

            # Relation loss
            rel_mask_expanded = pair_mask.unsqueeze(-1).expand(B, N, C_rel)
            class_mask = rel_prompts_embedding_mask.unsqueeze(1).expand(B, N, C_rel)
            combined = rel_mask_expanded * class_mask
            rel_loss = (focal_loss_with_logits(pair_scores, rel_matrix) * combined).sum()

            loss = adj_loss * self.config.adjacency_loss_coef + rel_loss * self.config.rel_loss_coef

        return loss, pair_scores, pair_idx, pair_mask

    def _forward_relations_prompt(
        self,
        token_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_span_rep: torch.Tensor,
        target_span_mask: torch.Tensor,
        rel_labels: Optional[torch.Tensor],
        adjacency_threshold: float,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prompt-based relation extraction (Type 2).

        1. Relation prompt embeddings guide source entity selection.
        2. Source + relation representation selects target entities.
        3. Dense (B, E, E, C) scores are computed, then pairs are selected.
        """
        if not hasattr(self, "prompt_rel_extractor"):
            return None, None, None, None

        batch_size, _, embed_dim = token_embeds.shape
        B, E_ent, D = target_span_rep.shape

        # Extract relation type embeddings
        rel_prompts_embedding, rel_prompts_embedding_mask = extract_prompt_features(
            self.config.rel_token_index, token_embeds, input_ids, attention_mask,
            batch_size, embed_dim, self.config.embed_rel_token,
        )
        C_rel = rel_prompts_embedding.size(1)

        # Compute dense scores: (B, E, E, C)
        full_scores = self.prompt_rel_extractor(
            target_span_rep, rel_prompts_embedding, target_span_mask,
        )

        # Loss on the full dense matrix
        loss = None
        if rel_labels is not None:
            # Align class dimension
            C_label = rel_labels.size(-1)
            E_label = rel_labels.size(1)

            scores_for_loss = full_scores[:, :E_label, :E_label, :C_rel]
            labels_for_loss = rel_labels[..., :C_rel]

            if scores_for_loss.size(-1) > labels_for_loss.size(-1):
                labels_for_loss = F.pad(labels_for_loss, (0, scores_for_loss.size(-1) - labels_for_loss.size(-1)))
            elif labels_for_loss.size(-1) > scores_for_loss.size(-1):
                labels_for_loss = labels_for_loss[..., :scores_for_loss.size(-1)]

            # Entity-pair mask
            entity_mask = target_span_mask[:, :E_label].float()
            pair_valid = entity_mask.unsqueeze(2) * entity_mask.unsqueeze(1)  # (B, E, E)
            class_valid = rel_prompts_embedding_mask[:, :scores_for_loss.size(-1)]  # (B, C)
            full_mask = pair_valid.unsqueeze(-1) * class_valid[:, None, None, :]  # (B, E, E, C)

            loss = (focal_loss_with_logits(scores_for_loss, labels_for_loss) * full_mask).sum()
            loss = loss * self.config.rel_loss_coef

        # Select pairs for output (same format as adjacency mode)
        with torch.no_grad():
            pseudo_adj = torch.sigmoid(full_scores).max(dim=-1).values  # (B, E, E)
            pseudo_adj = pseudo_adj * (
                ~torch.eye(E_ent, device=pseudo_adj.device, dtype=torch.bool)
            ).unsqueeze(0).float()

        pair_idx, pair_mask, _, _ = build_entity_pairs(
            pseudo_adj, target_span_rep, threshold=adjacency_threshold,
        )
        N = pair_idx.size(1)

        # Gather per-pair scores
        head_indices = pair_idx[..., 0].clamp(min=0)
        tail_indices = pair_idx[..., 1].clamp(min=0)
        batch_idx = torch.arange(B, device=full_scores.device).unsqueeze(1)
        pair_scores = full_scores[batch_idx, head_indices, tail_indices]  # (B, N, C)

        # Align to C_rel
        if pair_scores.size(-1) < C_rel:
            pair_scores = F.pad(pair_scores, (0, C_rel - pair_scores.size(-1)))
        elif pair_scores.size(-1) > C_rel:
            pair_scores = pair_scores[..., :C_rel]

        return loss, pair_scores, pair_idx, pair_mask

    def _forward_decoder(
        self,
        decoder_embedding: Optional[torch.Tensor] = None,
        decoder_embedding_mask: Optional[torch.Tensor] = None,
        decoder_labels_ids: Optional[torch.Tensor] = None,
        decoder_labels_mask: Optional[torch.Tensor] = None,
        decoder_labels: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Decoder forward pass — returns loss or None."""
        if not hasattr(self, "decoder") or decoder_embedding is None:
            return None
        if decoder_labels_ids is None:
            return None

        # Get raw decoder inputs: flatten valid spans
        B, S, T, D = decoder_embedding.shape
        valid_spans = decoder_embedding_mask.any(-1)
        keep_idx = valid_spans.view(-1).nonzero(as_tuple=False).squeeze(1)

        if keep_idx.numel() == 0:
            return None

        span_tokens = decoder_embedding.view(B * S, T, D)[keep_idx]
        span_tokens_mask = decoder_embedding_mask.view(B * S, T)[keep_idx]

        if hasattr(self, "_enc2dec_proj"):
            span_tokens = self._enc2dec_proj(span_tokens)

        label_embeds = self.decoder.ids_to_embeds(decoder_labels_ids)
        decoder_inputs = torch.cat([span_tokens, label_embeds[:, :-1, :]], dim=1)
        attn_inputs = torch.cat([span_tokens_mask.to(decoder_labels_mask.dtype), decoder_labels_mask[:, :-1]], dim=1)

        decoder_outputs = self.decoder(inputs_embeds=decoder_inputs, attention_mask=attn_inputs)

        blank = torch.full(
            (decoder_labels.size(0), span_tokens.size(1)), -100,
            dtype=decoder_labels.dtype, device=decoder_labels.device,
        )
        targets = torch.cat([blank, decoder_labels], dim=1)
        loss = cross_entropy_loss(decoder_outputs, targets[:, 1:])
        return loss

    def _forward_count(
        self,
        prompts_embedding: torch.Tensor,
        prompts_embedding_mask: torch.Tensor,
        count_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Count prediction from mean-pooled prompt embeddings.

        Returns:
            (loss, logits)
        """
        if not hasattr(self, "count_head"):
            return None, None

        # Mean-pool over valid prompt positions: (B, D)
        mask_f = prompts_embedding_mask.float().unsqueeze(-1)
        pooled = (prompts_embedding * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        logits = self.count_head(pooled)

        loss = None
        if count_targets is not None:
            loss = self.count_head.loss(logits, count_targets)

        return loss, logits


    def _forward_groups(
        self,
        prompts_embedding: torch.Tensor,
        words_embedding: Optional[torch.Tensor] = None,
        gold_count_val: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Group structure generation.

        For RotaryGroupLSTM: pc_emb is (M, D), no token_emb needed.
        For QueryGroupLSTM / QueryGroupTransformer: uses token_emb (B, L, D).

        Returns:
            (groups_output, groups_mask)
        """
        if not hasattr(self, "groups_layer"):
            return None, None

        if isinstance(self.groups_layer, RotaryGroupLSTM):
            # RotaryGroupLSTM expects 2D pc_emb (M, D) and scalar count
            # Process per-sample if needed
            if prompts_embedding.dim() == 3:
                # prompts_embedding: (B, M, D) — process first sample or iterate
                outputs = []
                for b in range(prompts_embedding.size(0)):
                    count = gold_count_val[b].item() if gold_count_val is not None else 1
                    count = max(int(count), 1)
                    out = self.groups_layer(prompts_embedding[b], count)  # (count, M, D)
                    outputs.append(out)
                return outputs, None
            else:
                count = gold_count_val.item() if gold_count_val is not None else 1
                count = max(int(count), 1)
                return self.groups_layer(prompts_embedding, count), None
        else:
            # QueryGroupLSTM or QueryGroupTransformer
            if words_embedding is None:
                return None, None
            return self.groups_layer(
                prompts_embedding.mean(dim=0) if prompts_embedding.dim() == 3 else prompts_embedding,
                words_embedding,
                gold_count_val=gold_count_val,
                threshold=threshold,
            )

    def _forward_structuring(
        self,
        token_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        words_embedding: torch.Tensor,
        mask: torch.Tensor,
        prompts_embedding: torch.Tensor,
        prompts_embedding_mask: torch.Tensor,
        gold_count_val: Optional[torch.Tensor] = None,
        structuring_labels: Optional[torch.Tensor] = None,
        structuring_count: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Structuring via anchor-based span extraction.

        Unified anchor paradigm:
          1. Groups layer generates instance anchors from parent embeddings
          2. [CHILD] embeddings provide field type representations
          3. AnchoredSpanScorer: (anchor, field) → scores text positions → spans

        Args:
            token_embeds: (B_enc, S, D) raw encoder output (for extracting [CHILD] tokens).
            input_ids: (B_enc, S) for token index extraction.
            attention_mask: (B_enc, S).
            words_embedding: (B, W, D) text token embeddings.
            mask: (B, W) valid text token mask.
            prompts_embedding: (B, C_ent, D) parent/entity prompt embeddings.
            prompts_embedding_mask: (B, C_ent).
            gold_count_val: (B,) ground-truth instance counts (training).
            structuring_labels: (B, X, L, C_field, 3) target span labels.
            structuring_count: (B,) number of instances per schema.
            threshold: Score threshold for inference.

        Returns:
            (loss, structuring_logits, groups_output, anchor_mask)
        """
        if not hasattr(self, "groups_layer") or not hasattr(self, "anchored_scorer"):
            return None, None, None, None

        batch_size, _, embed_dim = token_embeds.shape

        # Extract [CHILD] token embeddings for field types
        child_embedding, child_embedding_mask = extract_prompt_features(
            self.config.child_token_index, token_embeds, input_ids, attention_mask,
            batch_size, embed_dim, self.config.embed_child_token,
        )

        if child_embedding.shape[1] == 0:
            return None, None, None, None

        # Step 1: Generate instance anchors via groups layer
        count_for_groups = structuring_count if structuring_count is not None else gold_count_val
        groups_output, groups_mask_out = self._forward_groups(
            prompts_embedding, words_embedding, count_for_groups, threshold,
        )

        if groups_output is None:
            return None, None, None, None

        # Normalize groups_output to (B, X, D) anchor representations
        if isinstance(groups_output, list):
            # RotaryGroupLSTM: list of (count, M, D) per batch item
            max_instances = max(o.shape[0] for o in groups_output)
            D = groups_output[0].shape[-1]
            anchors = torch.zeros(len(groups_output), max_instances, D,
                                  device=words_embedding.device)
            anchor_mask = torch.zeros(len(groups_output), max_instances,
                                      dtype=torch.bool, device=words_embedding.device)
            for b, out in enumerate(groups_output):
                n = out.shape[0]
                anchors[b, :n] = out.mean(dim=1)  # mean over M fields → (count, D)
                anchor_mask[b, :n] = True
        else:
            anchors = groups_output  # (B, k, D)
            anchor_mask = groups_mask_out if groups_mask_out is not None else torch.ones(
                anchors.shape[:2], dtype=torch.bool, device=anchors.device
            )

        # Step 2: Anchored span scoring
        # anchors: (B, X, D), child_embedding: (B, C, D), words_embedding: (B, L, D)
        structuring_logits = self.anchored_scorer(
            anchors, child_embedding, words_embedding,
            child_mask=child_embedding_mask, word_mask=mask,
        )  # (B, X, L, C, 3)

        # Step 3: Loss computation
        loss = None
        if structuring_labels is not None:
            X_pred = structuring_logits.shape[1]
            X_label = structuring_labels.shape[1]
            L_pred = structuring_logits.shape[2]
            L_label = structuring_labels.shape[2]
            C_pred = structuring_logits.shape[3]
            C_label = structuring_labels.shape[3]

            min_X = min(X_pred, X_label)
            min_L = min(L_pred, L_label)
            min_C = min(C_pred, C_label)

            pred = structuring_logits[:, :min_X, :min_L, :min_C, :]
            labels = structuring_labels[:, :min_X, :min_L, :min_C, :]

            all_losses = self._loss(pred, labels)

            # Mask: valid instances × valid positions × valid fields
            inst_mask = anchor_mask[:, :min_X].float()
            word_mask_f = mask[:, :min_L].float()
            child_mask_f = child_embedding_mask[:, :min_C].float()

            full_mask = (
                inst_mask[:, :, None, None, None]
                * word_mask_f[:, None, :, None, None]
                * child_mask_f[:, None, None, :, None]
            )

            loss = (all_losses * full_mask).sum()

        return loss, structuring_logits, groups_output, anchor_mask

    def _forward_embedding(
        self,
        words_embedding: torch.Tensor,
        mask: torch.Tensor,
        embedding_labels: Optional[torch.Tensor] = None,
        embedding_pair_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Semantic similarity via cosine distance between pooled text pairs.

        Args:
            words_embedding: (B, W, D) word-level representations.
            mask: (B, W) valid token mask.
            embedding_labels: (N,) target similarity scores in [-1, 1].
            embedding_pair_idx: (N, 2) indices into the batch dimension
                specifying which items form each pair.

        Returns:
            (loss, similarities) — similarities shape: (N,)
        """
        if embedding_pair_idx is None:
            return None, None

        # Mean-pool over valid tokens: (B, D)
        mask_f = mask.float().unsqueeze(-1)  # (B, W, 1)
        pooled = (words_embedding * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        # L2-normalize for cosine similarity
        pooled = F.normalize(pooled, p=2, dim=-1)

        # Gather pair embeddings
        idx_a = embedding_pair_idx[:, 0]  # (N,)
        idx_b = embedding_pair_idx[:, 1]  # (N,)
        emb_a = pooled[idx_a]  # (N, D)
        emb_b = pooled[idx_b]  # (N, D)

        # Cosine similarity (already normalized)
        similarities = (emb_a * emb_b).sum(dim=-1)  # (N,)

        loss = None
        if embedding_labels is not None:
            loss = F.mse_loss(similarities, embedding_labels.float())

        return loss, similarities

    def _select_entity_spans(
        self,
        scores: torch.Tensor,
        words_embedding: torch.Tensor,
        ner_labels: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select high-confidence entity spans for downstream relation/decoder heads.

        Returns:
            (target_rep, target_mask) — (B, E, D), (B, E)
        """
        B, W, _, _ = scores.shape
        D = words_embedding.shape[-1]

        # Max over class and 3 channels → confidence per token
        token_confidence = torch.sigmoid(scores).max(dim=-1).values.max(dim=-1).values  # (B, W)

        if ner_labels is not None:
            # Use ground truth: any token with positive label
            keep = (ner_labels.sum(dim=(-1, -2)) > 0) if ner_labels.dim() == 4 else (ner_labels.sum(dim=-1) > 0)
        else:
            keep = token_confidence > threshold

        if top_k is not None:
            sel_scores = token_confidence.masked_fill(~keep, -1.0)
            top_idx = sel_scores.topk(k=min(top_k, W), dim=1).indices
            keep = torch.zeros_like(keep)
            keep.scatter_(1, top_idx, True)

        # Pack valid tokens
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

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        words_mask: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
        # NER
        ner_labels: Optional[torch.Tensor] = None,
        span_idx: Optional[torch.Tensor] = None,
        span_mask: Optional[torch.Tensor] = None,
        span_labels: Optional[torch.Tensor] = None,
        # Classification
        cat_labels: Optional[torch.Tensor] = None,
        # Relations
        rel_labels: Optional[torch.Tensor] = None,
        # Decoder
        decoder_labels_ids: Optional[torch.Tensor] = None,
        decoder_labels_mask: Optional[torch.Tensor] = None,
        decoder_labels: Optional[torch.Tensor] = None,
        # Count
        count_targets: Optional[torch.Tensor] = None,
        # Groups / Structuring
        gold_count_val: Optional[torch.Tensor] = None,
        structuring_labels: Optional[torch.Tensor] = None,
        structuring_count: Optional[torch.Tensor] = None,
        # Embedding similarity
        embedding_labels: Optional[torch.Tensor] = None,
        embedding_pair_idx: Optional[torch.Tensor] = None,
        # Labels encoder (bi-encoder)
        labels_input_ids: Optional[torch.Tensor] = None,
        labels_attention_mask: Optional[torch.Tensor] = None,
        # Misc
        threshold: float = 0.5,
        adjacency_threshold: float = 0.5,
        **kwargs,
    ) -> GLiNExTOutput:

        # ── 1. Encode ────────────────────────────────────────────────────
        token_embeds, prompts_embedding, prompts_embedding_mask, words_embedding, mask = (
            self.get_representations(
                input_ids, attention_mask, text_lengths, words_mask,
                labels_input_ids=labels_input_ids,
                labels_attention_mask=labels_attention_mask,
                **kwargs,
            )
        )

        total_loss = torch.tensor(0.0, device=words_embedding.device)

        # ── 2. NER ───────────────────────────────────────────────────────
        ner_loss, ner_scores, span_logits, span_idx, span_mask = self._forward_ner(
            words_embedding, mask, prompts_embedding, prompts_embedding_mask,
            ner_labels=ner_labels, span_idx=span_idx, span_mask=span_mask,
            span_labels=span_labels, threshold=threshold,
        )
        if ner_loss is not None:
            total_loss = total_loss + self.config.ner_loss_coef * ner_loss

        # ── 3. Classification ────────────────────────────────────────────
        cat_loss, cat_logits = self._forward_cat(
            token_embeds, input_ids, attention_mask, cat_labels=cat_labels,
        )
        if cat_loss is not None:
            total_loss = total_loss + self.config.cat_loss_coef * cat_loss

        # ── 4. Relations ─────────────────────────────────────────────────
        rel_loss, rel_logits, rel_idx, rel_mask = None, None, None, None
        has_rel_head = hasattr(self, "relations_rep_layer") or hasattr(self, "prompt_rel_extractor")
        if has_rel_head:
            target_span_rep, target_span_mask = self._select_entity_spans(
                ner_scores, words_embedding, ner_labels, threshold,
            )
            rel_loss, rel_logits, rel_idx, rel_mask = self._forward_relations(
                token_embeds, input_ids, attention_mask,
                target_span_rep=target_span_rep, target_span_mask=target_span_mask,
                rel_labels=rel_labels,
                adjacency_threshold=adjacency_threshold,
            )
        if rel_loss is not None:
            total_loss = total_loss + rel_loss  # already weighted internally

        # ── 5. Count ─────────────────────────────────────────────────────
        count_loss, count_logits = self._forward_count(
            prompts_embedding, prompts_embedding_mask, count_targets,
        )
        if count_loss is not None:
            total_loss = total_loss + self.config.count_loss_coef * count_loss

        # ── 6. Decoder ───────────────────────────────────────────────────
        decoder_loss = None
        if hasattr(self, "decoder") and decoder_labels_ids is not None:
            # Build decoder embeddings from span representations
            if hasattr(self, "span_rep_layer") and span_idx is not None:
                decoder_emb = self.span_rep_layer(words_embedding, span_idx)
                decoder_emb_mask = span_mask if span_mask is not None else torch.ones(
                    decoder_emb.shape[:-1], dtype=torch.long, device=decoder_emb.device,
                )
                decoder_loss = self._forward_decoder(
                    decoder_emb, decoder_emb_mask,
                    decoder_labels_ids, decoder_labels_mask, decoder_labels,
                )
        if decoder_loss is not None:
            total_loss = total_loss + self.config.decoder_loss_coef * decoder_loss

        # ── 7. Structuring (anchor-based span extraction via groups) ────
        structuring_loss, structuring_logits, groups_output, structuring_anchor_mask = (
            self._forward_structuring(
                token_embeds, input_ids, attention_mask,
                words_embedding, mask,
                prompts_embedding, prompts_embedding_mask,
                gold_count_val=gold_count_val,
                structuring_labels=structuring_labels,
                structuring_count=structuring_count,
                threshold=threshold,
            )
        )
        groups_mask = structuring_anchor_mask
        if structuring_loss is not None:
            total_loss = total_loss + self.config.structuring_loss_coef * structuring_loss

        # ── 8. Embedding similarity ─────────────────────────────────────
        embedding_loss, embedding_logits = self._forward_embedding(
            words_embedding, mask, embedding_labels, embedding_pair_idx,
        )
        if embedding_loss is not None:
            total_loss = total_loss + self.config.embedding_loss_coef * embedding_loss

        # ── Collect losses ───────────────────────────────────────────────
        has_any_loss = any(l is not None for l in [ner_loss, cat_loss, rel_loss, decoder_loss, count_loss, embedding_loss, structuring_loss])
        final_loss = total_loss if has_any_loss else None

        return GLiNExTOutput(
            loss=final_loss,
            ner_logits=ner_scores,
            span_logits=span_logits,
            span_idx=span_idx,
            span_mask=span_mask,
            cat_logits=cat_logits,
            rel_logits=rel_logits,
            rel_idx=rel_idx,
            rel_mask=rel_mask,
            count_logits=count_logits,
            groups_output=groups_output,
            groups_mask=groups_mask,
            structuring_logits=structuring_logits,
            structuring_anchor_mask=structuring_anchor_mask,
            embedding_logits=embedding_logits,
            words_embedding=words_embedding,
            mask=mask,
            prompts_embedding=prompts_embedding,
            prompts_embedding_mask=prompts_embedding_mask,
        )

    def loss(self, *args, **kwargs):
        """Compute loss via forward pass."""
        output = self.forward(*args, **kwargs)
        return output.loss
