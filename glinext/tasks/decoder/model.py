"""Decoder task head."""

import torch
from torch import nn

from gliner.modeling.decoder import Decoder
from gliner.modeling.layers import create_projection_layer
from gliner.modeling.loss_functions import cross_entropy_loss
from gliner.modeling.span_rep import SpanRepLayer

from .. import TaskHead, TaskHeadOutput, SharedRepresentations


class DecoderHead(TaskHead):
    """Autoregressive decoder head for label generation.

    Can receive span embeddings either:
    1. From batch kwargs (decoder_embedding) — pre-computed externally
    2. From NER dependency output — using span_idx to extract span reps from word embeddings
    """

    name = "decoder"
    dependencies = ["ner"]

    def __init__(self, config, from_pretrained=False, cache_dir=None):
        super().__init__()
        dec_cfg = config.decoder_head_config
        self.loss_coef = dec_cfg.loss_coef
        self.full_decoder_context = dec_cfg.full_decoder_context
        self.decoder = Decoder(config, from_pretrained, cache_dir=cache_dir)

        if config.hidden_size != self.decoder.decoder_hidden_size:
            self._enc2dec_proj = create_projection_layer(
                config.hidden_size, config.dropout, self.decoder.decoder_hidden_size,
            )

        # Span rep layer for extracting span embeddings from NER output
        self.span_rep_layer = SpanRepLayer(
            span_mode="token_level",
            hidden_size=config.hidden_size,
            max_width=getattr(config, "max_width", 12),
            dropout=config.dropout,
        )

    @classmethod
    def from_config(cls, config, from_pretrained=False, cache_dir=None, **kwargs):
        if config.decoder_head_config is None:
            return None
        return cls(config, from_pretrained=from_pretrained, cache_dir=cache_dir)

    def _get_span_embeddings(self, shared, dependency_outputs, batch):
        """Get span embeddings from NER output or batch kwargs."""
        # Priority 1: pre-computed decoder_embedding from batch
        decoder_embedding = batch.get("decoder_embedding")
        decoder_embedding_mask = batch.get("decoder_embedding_mask")
        if decoder_embedding is not None and decoder_embedding_mask is not None:
            return decoder_embedding, decoder_embedding_mask

        # Priority 2: extract from NER span representations
        ner_output = dependency_outputs.get("ner")
        if ner_output is None:
            return None, None

        span_idx = ner_output.extra.get("span_idx")
        words_embedding = ner_output.extra.get("words_embedding", shared.words_embedding)

        if span_idx is None or words_embedding is None:
            return None, None

        # Extract span representations: (B, S, D) or (B*N, S, D)
        span_rep = self.span_rep_layer(words_embedding, span_idx)

        span_mask = ner_output.extra.get("span_mask")
        if span_mask is None:
            span_mask = torch.ones(span_rep.shape[:2], dtype=torch.bool, device=span_rep.device)

        # Reshape to (B, S, 1, D) to match expected decoder_embedding format (B, S, T, D)
        # where T=1 since each span is a single representation
        span_rep = span_rep.unsqueeze(2)
        span_mask = span_mask.unsqueeze(2)

        return span_rep, span_mask

    def forward(self, shared, dependency_outputs, **batch):
        decoder_labels_ids = batch.get("decoder_labels_ids")
        decoder_labels_mask = batch.get("decoder_labels_mask")
        decoder_labels = batch.get("decoder_labels")

        decoder_embedding, decoder_embedding_mask = self._get_span_embeddings(
            shared, dependency_outputs, batch,
        )

        if decoder_embedding is None or decoder_labels_ids is None:
            return TaskHeadOutput()

        B, S, T, D = decoder_embedding.shape
        valid_spans = decoder_embedding_mask.any(-1)
        keep_idx = valid_spans.view(-1).nonzero(as_tuple=False).squeeze(1)

        if keep_idx.numel() == 0:
            return TaskHeadOutput()

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

        return TaskHeadOutput(loss=loss)
