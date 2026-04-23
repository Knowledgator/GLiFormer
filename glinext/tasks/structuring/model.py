"""Structuring task head."""

from .. import TaskHeadOutput
from ..anchored_extraction import AnchoredSpanExtractionHead


class StructuringHead(AnchoredSpanExtractionHead):
    """Structuring via anchor-based span extraction.

    Generalizes :class:`NERHead` to A >= 1 anchors per instance. Shares the
    :class:`AnchoredSpanExtractionHead` pipeline: flattening the (A, C)
    dimensions produces the same scoring path as NER.
    """

    name = "structuring"
    dependencies = []

    def __init__(self, config, hidden_size, dropout, shared_layers=None):
        super().__init__(config.structuring_config, config, hidden_size, dropout, shared_layers)
        struct_cfg = config.structuring_config
        self.child_token_index = struct_cfg.child_token_index
        self.embed_child_token = struct_cfg.embed_child_token

    @classmethod
    def from_config(cls, config, shared_layers=None, **kwargs):
        if config.structuring_config is None:
            return None
        return cls(config, hidden_size=config.hidden_size, dropout=config.dropout,
                   shared_layers=shared_layers)

    def _anchor_kwargs(self, batch):
        """Pass structuring/count-head-predicted anchor counts when available."""
        kwargs = super()._anchor_kwargs(batch)
        count = batch.get("structuring_count")
        if count is None:
            count = batch.get("count_val")
        if count is not None:
            kwargs["count"] = count
        return kwargs

    def forward(self, shared, dependency_outputs, flat_inputs=None,
                base_loss_fn=None, **batch):
        structuring_labels = batch.get("structuring_labels")
        threshold = batch.get("threshold", 0.5)

        span_idx = batch.get("structuring_span_idx")
        span_mask = batch.get("structuring_span_mask")
        span_labels = batch.get("structuring_span_labels")

        if flat_inputs.child_embedding.shape[1] == 0:
            return TaskHeadOutput()

        scores, anchors, anchor_mask, fused_flat, (B, A, C, L) = self._compute_bio_scores(
            flat_inputs, batch,
        )

        # Optional span-level rescoring — auxiliary training signal only.
        span_logits_out = None
        if (
            self.represent_spans and hasattr(self, "span_rep_layer")
            and structuring_labels is not None
        ):
            span_idx, span_mask = self._maybe_extract_spans(
                scores, A=A, span_idx=span_idx, span_mask=span_mask, threshold=threshold,
            )
            if span_idx is not None:
                span_logits_out = self._span_logits_from_fused(
                    flat_inputs.words_embedding, fused_flat, B, A, C, span_idx,
                )

        loss = None
        if structuring_labels is not None and base_loss_fn is not None:
            loss = self._bio_loss(
                scores=scores, labels=structuring_labels,
                anchor_mask=anchor_mask,
                word_mask=flat_inputs.mask,
                child_mask=flat_inputs.child_mask,
                base_loss_fn=base_loss_fn,
            )
            if span_labels is not None and span_logits_out is not None:
                span_loss = self._span_loss(
                    span_logits=span_logits_out, span_labels=span_labels,
                    anchor_mask=anchor_mask, span_mask=span_mask,
                    child_mask=flat_inputs.child_mask, base_loss_fn=base_loss_fn,
                )
                loss = loss + self.span_loss_coef * span_loss

        return TaskHeadOutput(
            loss=loss,
            logits=scores,
            extra={
                "groups_output": anchors,
                "anchor_mask": anchor_mask,
                "span_logits": span_logits_out,
                "span_idx": span_idx,
                "span_mask": span_mask,
            },
        )
