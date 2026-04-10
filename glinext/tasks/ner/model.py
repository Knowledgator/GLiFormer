"""NER task head."""

import torch
from torch import nn

from gliner.modeling.utils import extract_spans_from_tokens

from .. import TaskHead, TaskHeadOutput
from ...layers import AnchoredSpanScorer


class NERHead(TaskHead):
    """Token-level NER scorer (start/end/inside) with anchor paradigm.

    Always uses AnchoredSpanScorer with configurable anchor layer.
    Default: parent anchor mode (single anchor = context embedding).
    """

    name = "ner"
    dependencies = []

    def __init__(self, config, hidden_size, dropout, shared_layers=None):
        super().__init__()
        self.config = config
        ner_cfg = config.ner_config
        self.loss_coef = ner_cfg.loss_coef
        if shared_layers is None:
            shared_layers = {}

        self._init_anchor_pipeline(ner_cfg, config, hidden_size, dropout, shared_layers)
        self.scorer = AnchoredSpanScorer(hidden_size, dropout=dropout)

    @classmethod
    def from_config(cls, config, shared_layers=None, **kwargs):
        if config.ner_config is None:
            return None
        return cls(
            config,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            shared_layers=shared_layers,
        )

    def _ner_loss(self, scores, labels, prompts_embedding_mask, word_mask, base_loss_fn):
        all_losses = base_loss_fn(scores, labels)
        mask = word_mask.unsqueeze(-1) * prompts_embedding_mask.unsqueeze(1)
        if all_losses.dim() == 4:
            mask = mask.unsqueeze(-1)
        all_losses = all_losses * mask
        return all_losses.sum()

    def _fit_length(self, tensor, mask, target_length):
        """Pad or trim tensor and mask to target_length along dim=1."""
        current = tensor.shape[1]
        if current == target_length:
            return tensor, mask
        if current < target_length:
            pad_size = target_length - current
            tensor = torch.nn.functional.pad(tensor, [0] * (2 * (tensor.dim() - 2)) + [0, pad_size])
            mask = torch.nn.functional.pad(mask, [0, pad_size])
        else:
            tensor = tensor[:, :target_length]
            mask = mask[:, :target_length]
        return tensor, mask

    def forward(self, shared, dependency_outputs, flat_inputs=None, base_loss_fn=None, **batch):
        # Use flat_inputs (BN-indexed) when available, else fall back to shared (B-indexed)
        if flat_inputs is not None:
            words_embedding = flat_inputs.words_embedding
            mask = flat_inputs.mask
            prompts_embedding = flat_inputs.child_embedding
            prompts_embedding_mask = flat_inputs.child_mask
        else:
            words_embedding = shared.words_embedding
            mask = shared.mask
            prompts_embedding = shared.prompts_embedding
            prompts_embedding_mask = shared.prompts_embedding_mask

        ner_labels = batch.get("ner_labels")
        span_idx = batch.get("span_idx")
        span_mask = batch.get("span_mask")
        span_labels = batch.get("span_labels")
        threshold = batch.get("threshold", 0.5)

        if ner_labels is not None:
            target_W = ner_labels.shape[1]
            words_embedding, mask = self._fit_length(words_embedding, mask, target_W)
            target_C = max(prompts_embedding.size(1), ner_labels.size(-2))
            prompts_embedding, prompts_embedding_mask = self._fit_length(
                prompts_embedding, prompts_embedding_mask, target_C,
            )

        # Anchor paradigm: anchor_layer → anchor_refine → anchor_modeling → scorer
        if flat_inputs is not None:
            context = flat_inputs.parent_embedding  # (BN, D)
        else:
            context = prompts_embedding.mean(dim=1)  # (B, D)
        anchor_rep, anchor_mask = self.anchor_layer(
            context, words_embedding,
        )
        if hasattr(self, "anchor_refine"):
            anchor_rep = self.anchor_refine(anchor_rep, words_embedding, token_mask=mask)
        fused = self.anchor_modeling(anchor_rep, prompts_embedding)
        B_a, A, C, D = fused.shape
        L = words_embedding.shape[1]
        fused_flat = fused.view(B_a, A * C, D)
        scores_flat = self.scorer(fused_flat, words_embedding, word_mask=mask)
        scores = scores_flat.view(B_a, A, C, L, 3).permute(0, 1, 3, 2, 4)
        # Squeeze anchor dim for parent mode (A=1) to maintain (B, W, C, 3) shape
        if A == 1:
            scores = scores.squeeze(1)

        # Optional span representation
        span_logits_out = None
        if self.represent_spans and hasattr(self, "span_rep_layer"):
            if span_idx is None:
                span_idx, span_mask = extract_spans_from_tokens(scores, ner_labels, threshold)
                span_idx = span_idx * span_mask.unsqueeze(-1).long()
            span_rep = self.span_rep_layer(words_embedding, span_idx)
            span_logits_out = torch.einsum("BND,BCD->BNC", span_rep, prompts_embedding)

        loss = None
        if ner_labels is not None and base_loss_fn is not None:
            loss = self._ner_loss(scores, ner_labels, prompts_embedding_mask, mask, base_loss_fn)
            if span_labels is not None and span_logits_out is not None:
                span_loss = self._ner_loss(span_logits_out, span_labels, prompts_embedding_mask, span_mask, base_loss_fn)
                token_loss_coef = getattr(self.config, 'token_loss_coef', 1.0)
                loss = token_loss_coef * loss + self.span_loss_coef * span_loss

        return TaskHeadOutput(
            loss=loss,
            logits=scores,
            extra={
                "span_logits": span_logits_out,
                "span_idx": span_idx,
                "span_mask": span_mask,
                "words_embedding": words_embedding,
                "mask": mask,
            },
        )
