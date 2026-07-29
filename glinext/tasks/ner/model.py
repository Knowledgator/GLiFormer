"""NER task head."""

import torch

from .. import TaskHeadOutput
from ..anchored_extraction import AnchoredSpanExtractionHead


class NERHead(AnchoredSpanExtractionHead):
    """Token-level NER scorer (start/end/inside) with anchor paradigm.

    Uses the shared :class:`AnchoredSpanExtractionHead` pipeline. NER operates
    with a single parent anchor (A=1), so the anchor dim is squeezed on output
    to keep the canonical NER shape ``(BN, W, C, 3)``.
    """

    name = "ner"
    dependencies = []

    def __init__(
        self,
        config,
        hidden_size,
        dropout,
        shared_layers=None,
        task_config=None,
    ):
        # Composite entity-first heads can reuse the exact NER pipeline with
        # their own task-local loss/span settings.  Ordinary NER continues to
        # use ``ner_config``.
        task_config = task_config or config.ner_config
        super().__init__(
            task_config,
            config,
            hidden_size,
            dropout,
            shared_layers,
        )

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

    @staticmethod
    def _fit_length(tensor, mask, target_length):
        """Pad or trim ``tensor`` and ``mask`` to ``target_length`` along dim=1."""
        tensor_length = tensor.shape[1]
        if tensor_length < target_length:
            pad_size = target_length - tensor_length
            tensor = torch.nn.functional.pad(tensor, [0] * (2 * (tensor.dim() - 2)) + [0, pad_size])
        elif tensor_length > target_length:
            tensor = tensor[:, :target_length]

        mask_length = mask.shape[1]
        if mask_length < target_length:
            mask = torch.nn.functional.pad(mask, [0, target_length - mask_length])
        elif mask_length > target_length:
            mask = mask[:, :target_length]
        return tensor, mask

    def forward(self, shared, dependency_outputs, flat_inputs=None, base_loss_fn=None, **batch):
        ner_labels = batch.get("ner_labels")
        span_idx = batch.get("span_idx")
        span_mask = batch.get("span_mask")
        span_labels = batch.get("span_labels")
        threshold = batch.get("threshold", 0.5)

        # Align word/prompt tensors to label dims (padding covers truncated inputs).
        if ner_labels is not None:
            target_W = ner_labels.shape[1]
            flat_inputs.words_embedding, flat_inputs.mask = self._fit_length(
                flat_inputs.words_embedding, flat_inputs.mask, target_W,
            )
            target_C = max(flat_inputs.child_embedding.size(1), ner_labels.size(-2))
            flat_inputs.child_embedding, flat_inputs.child_mask = self._fit_length(
                flat_inputs.child_embedding, flat_inputs.child_mask, target_C,
            )

        scores, _anchors, anchor_mask, _fused_flat, (B, A, C, L) = self._compute_bio_scores(
            flat_inputs, batch,
        )
        # Parent anchor → A=1. Squeeze to (BN, L, C, 3) for backward compat.
        logits = scores.squeeze(1) if A == 1 else scores

        words_embedding = flat_inputs.words_embedding
        word_mask = flat_inputs.mask
        child_mask = flat_inputs.child_mask

        # Optional span-level rescoring — auxiliary training signal only.
        # NER scores spans directly against child embeddings (no anchor fusion)
        # to preserve historical behavior.
        span_logits_out = None
        if (
            self.represent_spans and hasattr(self, "span_rep_layer")
            and ner_labels is not None
            and logits.shape[1] > 0 and logits.shape[-2] > 0
        ):
            span_idx, span_mask = self._maybe_extract_spans(
                logits, A=1, span_idx=span_idx, span_mask=span_mask,
                threshold=threshold, labels=ner_labels,
            )
            span_rep = self.span_rep_layer(words_embedding, span_idx)
            span_logits_out = torch.einsum("BND,BCD->BNC", span_rep, flat_inputs.child_embedding)

        loss = None
        if ner_labels is not None and base_loss_fn is not None:
            # NER has A=1 with all-ones anchor_mask → equivalent to word × child masking.
            loss = self._bio_loss(
                scores=scores, labels=ner_labels.unsqueeze(1),
                anchor_mask=anchor_mask, word_mask=word_mask, child_mask=child_mask,
                base_loss_fn=base_loss_fn,
            )
            if span_labels is not None and span_logits_out is not None:
                span_losses = base_loss_fn(span_logits_out, span_labels)
                span_loss_mask = span_mask.unsqueeze(-1) * child_mask.unsqueeze(1)
                if span_losses.dim() == 4:
                    span_loss_mask = span_loss_mask.unsqueeze(-1)
                span_loss = (span_losses * span_loss_mask).sum()
                token_loss_coef = getattr(self.config, "token_loss_coef", 1.0)
                loss = token_loss_coef * loss + self.span_loss_coef * span_loss

        return TaskHeadOutput(
            loss=loss,
            logits=logits,
            extra={
                "span_logits": span_logits_out,
                "span_idx": span_idx,
                "span_mask": span_mask,
                "words_embedding": words_embedding,
                "mask": word_mask,
            },
        )
