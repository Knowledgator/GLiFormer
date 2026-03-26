"""GLiNExT: unified multi-task model — thin orchestrator over modular task heads."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
from pathlib import Path

import torch
from torch import nn
from transformers.utils import ModelOutput

from gliner.modeling.base import BaseModel
from gliner.modeling.encoder import Encoder, BiEncoder
from gliner.modeling.layers import CrossFuser, LstmSeq2SeqEncoder
from gliner.modeling.utils import (
    extract_word_embeddings,
    extract_prompt_features_and_word_embeddings,
)

from .config import GLiNextConfig
from .tasks import SharedRepresentations, TaskHeadOutput
from .tasks.ner.model import NERHead
from .tasks.classification.model import ClassificationHead
from .tasks.joint_relex.model import JointRelexHead
from .tasks.open_relex.model import OpenRelexHead
from .tasks.count.model import CountHead
from .tasks.structuring.model import StructuringHead
from .tasks.decoder.model import DecoderHead
from .tasks.embedding.model import EmbeddingHead


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
    # Joint Relex (NER + relation extraction)
    joint_rel_logits: Optional[torch.FloatTensor] = None
    joint_rel_idx: Optional[torch.LongTensor] = None
    joint_rel_mask: Optional[torch.Tensor] = None
    # Open Relex (anchor-based relation extraction)
    open_rel_logits: Optional[torch.FloatTensor] = None
    open_rel_anchor_mask: Optional[torch.Tensor] = None
    open_rel_span_logits: Optional[torch.FloatTensor] = None
    open_rel_span_idx: Optional[torch.LongTensor] = None
    open_rel_span_mask: Optional[torch.Tensor] = None
    # Count
    count_logits: Optional[torch.FloatTensor] = None
    # Groups / Structuring
    groups_output: Optional[torch.FloatTensor] = None
    groups_mask: Optional[torch.Tensor] = None
    # Structuring (anchor-based span extraction)
    structuring_logits: Optional[torch.FloatTensor] = None
    structuring_anchor_mask: Optional[torch.Tensor] = None
    structuring_span_logits: Optional[torch.FloatTensor] = None
    structuring_span_idx: Optional[torch.LongTensor] = None
    structuring_span_mask: Optional[torch.Tensor] = None
    # Embedding similarity
    embedding_logits: Optional[torch.FloatTensor] = None
    # Embeddings (for downstream use)
    words_embedding: Optional[torch.FloatTensor] = None
    mask: Optional[torch.LongTensor] = None
    prompts_embedding: Optional[torch.FloatTensor] = None
    prompts_embedding_mask: Optional[torch.LongTensor] = None


# Fixed execution order respecting dependencies
_EXECUTION_ORDER = ["ner", "classification", "count", "joint_relex", "open_relex",
                    "decoder", "structuring", "embedding"]

# Head classes in registration order
_HEAD_CLASSES = [NERHead, ClassificationHead, CountHead, JointRelexHead,
                 OpenRelexHead, StructuringHead, DecoderHead, EmbeddingHead]


class GLiNExTModel(BaseModel):
    """Unified multi-task model composing optional task heads.

    Each task is a standalone TaskHead subclass. The model registers enabled heads
    via nn.ModuleDict and executes them in dependency order during forward().
    """

    def __init__(
        self,
        config: GLiNextConfig,
        from_pretrained: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        super().__init__(config, from_pretrained, cache_dir)

        # ── Shared encoder ──────────────────────────────────────────────
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

        # ── Register task heads ─────────────────────────────────────────
        self.heads = nn.ModuleDict()
        for HeadClass in _HEAD_CLASSES:
            head = HeadClass.from_config(
                config,
                from_pretrained=from_pretrained,
                cache_dir=cache_dir,
            )
            if head is not None:
                self.heads[head.name] = head

    def _encode_label_type(
        self,
        label_input_ids: Optional[torch.Tensor],
        label_attention_mask: Optional[torch.Tensor],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if label_input_ids is None or not isinstance(self.token_rep_layer, BiEncoder):
            return None
        labels_embeds = self.token_rep_layer.encode_labels(label_input_ids, label_attention_mask)
        return labels_embeds.unsqueeze(0).expand(batch_size, -1, -1)

    def _encode_all_labels_batched(
        self,
        batch_size: int,
        cat_labels_input_ids: Optional[torch.Tensor] = None,
        cat_labels_attention_mask: Optional[torch.Tensor] = None,
        rel_labels_input_ids: Optional[torch.Tensor] = None,
        rel_labels_attention_mask: Optional[torch.Tensor] = None,
        child_labels_input_ids: Optional[torch.Tensor] = None,
        child_labels_attention_mask: Optional[torch.Tensor] = None,
        open_rel_labels_input_ids: Optional[torch.Tensor] = None,
        open_rel_labels_attention_mask: Optional[torch.Tensor] = None,
    ):
        """Batch all task label inputs into a single BiEncoder pass, then split results."""
        if not isinstance(self.token_rep_layer, BiEncoder):
            return None, None, None, None

        # Collect all non-None label inputs with their sizes
        parts = []
        sizes = []
        for ids, mask in [
            (cat_labels_input_ids, cat_labels_attention_mask),
            (rel_labels_input_ids, rel_labels_attention_mask),
            (child_labels_input_ids, child_labels_attention_mask),
            (open_rel_labels_input_ids, open_rel_labels_attention_mask),
        ]:
            if ids is not None:
                parts.append((ids, mask))
                sizes.append(ids.shape[0])
            else:
                parts.append(None)
                sizes.append(0)

        total = sum(sizes)
        if total == 0:
            return None, None, None, None

        # Concatenate all label inputs along batch dim
        all_ids = []
        all_masks = []
        for part in parts:
            if part is not None:
                all_ids.append(part[0])
                all_masks.append(part[1])

        if not all_ids:
            return None, None, None, None

        # Pad to same seq length and concatenate
        max_len = max(ids.shape[1] for ids in all_ids)
        padded_ids = []
        padded_masks = []
        for ids, m in zip(all_ids, all_masks):
            if ids.shape[1] < max_len:
                pad_size = max_len - ids.shape[1]
                ids = torch.nn.functional.pad(ids, (0, pad_size), value=0)
                m = torch.nn.functional.pad(m, (0, pad_size), value=0)
            padded_ids.append(ids)
            padded_masks.append(m)

        batched_ids = torch.cat(padded_ids, dim=0)
        batched_masks = torch.cat(padded_masks, dim=0)

        # Single encoder pass
        all_embeds = self.token_rep_layer.encode_labels(batched_ids, batched_masks)

        # Split back and expand to batch size
        results = []
        offset = 0
        for size in sizes:
            if size > 0:
                embeds = all_embeds[offset:offset + size]
                embeds = embeds.unsqueeze(0).expand(batch_size, -1, -1)
                results.append(embeds)
                offset += size
            else:
                results.append(None)

        return results[0], results[1], results[2], results[3]

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
        """Run encoder and extract prompt + word embeddings."""
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
        # Joint Relex
        rel_labels: Optional[torch.Tensor] = None,
        # Open Relex
        open_rel_labels: Optional[torch.Tensor] = None,
        open_rel_count: Optional[torch.Tensor] = None,
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
        # Labels encoder (bi-encoder) — NER labels
        labels_input_ids: Optional[torch.Tensor] = None,
        labels_attention_mask: Optional[torch.Tensor] = None,
        # Labels encoder — classification labels
        cat_labels_input_ids: Optional[torch.Tensor] = None,
        cat_labels_attention_mask: Optional[torch.Tensor] = None,
        # Labels encoder — relation labels
        rel_labels_input_ids: Optional[torch.Tensor] = None,
        rel_labels_attention_mask: Optional[torch.Tensor] = None,
        # Labels encoder — structuring/child labels
        child_labels_input_ids: Optional[torch.Tensor] = None,
        child_labels_attention_mask: Optional[torch.Tensor] = None,
        # Labels encoder — open relex labels
        open_rel_labels_input_ids: Optional[torch.Tensor] = None,
        open_rel_labels_attention_mask: Optional[torch.Tensor] = None,
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

        shared = SharedRepresentations(
            token_embeds=token_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            words_embedding=words_embedding,
            mask=mask,
            prompts_embedding=prompts_embedding,
            prompts_embedding_mask=prompts_embedding_mask,
        )

        batch_size = words_embedding.shape[0]
        total_loss = torch.tensor(0.0, device=words_embedding.device)

        # ── 1b. Encode task-specific labels via labels encoder ──────────
        cat_label_embeds, rel_label_embeds, child_label_embeds, open_rel_label_embeds = (
            self._encode_all_labels_batched(
                batch_size,
                cat_labels_input_ids, cat_labels_attention_mask,
                rel_labels_input_ids, rel_labels_attention_mask,
                child_labels_input_ids, child_labels_attention_mask,
                open_rel_labels_input_ids, open_rel_labels_attention_mask,
            )
        )

        # Collect all batch kwargs for heads
        batch_kwargs = dict(
            ner_labels=ner_labels, span_idx=span_idx, span_mask=span_mask,
            span_labels=span_labels, cat_labels=cat_labels, rel_labels=rel_labels,
            open_rel_labels=open_rel_labels, open_rel_count=open_rel_count,
            decoder_labels_ids=decoder_labels_ids, decoder_labels_mask=decoder_labels_mask,
            decoder_labels=decoder_labels, count_targets=count_targets,
            gold_count_val=gold_count_val, structuring_labels=structuring_labels,
            structuring_count=structuring_count, embedding_labels=embedding_labels,
            embedding_pair_idx=embedding_pair_idx,
            threshold=threshold, adjacency_threshold=adjacency_threshold,
        )

        # ── 2. Execute heads in order ───────────────────────────────────
        head_outputs = {}
        for name in _EXECUTION_ORDER:
            if name not in self.heads:
                continue
            head = self.heads[name]

            dep_outputs = {d: head_outputs[d] for d in head.dependencies if d in head_outputs}

            # Pass task-specific label embeds
            extra_kwargs = {}
            if name == "classification" and cat_label_embeds is not None:
                extra_kwargs["cat_label_embeds"] = cat_label_embeds
            elif name == "joint_relex" and rel_label_embeds is not None:
                extra_kwargs["rel_label_embeds"] = rel_label_embeds
            elif name == "open_relex" and open_rel_label_embeds is not None:
                extra_kwargs["open_rel_label_embeds"] = open_rel_label_embeds
            elif name == "structuring" and child_label_embeds is not None:
                extra_kwargs["child_label_embeds"] = child_label_embeds

            # Pass base_loss_fn for heads that need it
            if name in ("ner", "joint_relex", "open_relex", "structuring"):
                extra_kwargs["base_loss_fn"] = self._loss

            output = head(shared, dependency_outputs=dep_outputs, **extra_kwargs, **batch_kwargs)
            head_outputs[name] = output

            if output.loss is not None:
                total_loss = total_loss + head.loss_coef * output.loss

        # ── 3. Collect outputs ──────────────────────────────────────────
        has_any_loss = any(ho.loss is not None for ho in head_outputs.values())
        final_loss = total_loss if has_any_loss else None

        ner_out = head_outputs.get("ner", TaskHeadOutput())
        cat_out = head_outputs.get("classification", TaskHeadOutput())
        joint_rel_out = head_outputs.get("joint_relex", TaskHeadOutput())
        open_rel_out = head_outputs.get("open_relex", TaskHeadOutput())
        count_out = head_outputs.get("count", TaskHeadOutput())
        struct_out = head_outputs.get("structuring", TaskHeadOutput())
        emb_out = head_outputs.get("embedding", TaskHeadOutput())

        # joint_relex NER logits override standalone NER if joint_relex is active
        effective_ner_logits = joint_rel_out.logits if joint_rel_out.logits is not None else ner_out.logits
        effective_ner_extra = joint_rel_out.extra if joint_rel_out.logits is not None else ner_out.extra

        return GLiNExTOutput(
            loss=final_loss,
            ner_logits=effective_ner_logits,
            span_logits=effective_ner_extra.get("span_logits"),
            span_idx=effective_ner_extra.get("span_idx"),
            span_mask=effective_ner_extra.get("span_mask"),
            cat_logits=cat_out.logits,
            joint_rel_logits=joint_rel_out.extra.get("rel_logits"),
            joint_rel_idx=joint_rel_out.extra.get("rel_idx"),
            joint_rel_mask=joint_rel_out.extra.get("rel_mask"),
            open_rel_logits=open_rel_out.logits,
            open_rel_anchor_mask=open_rel_out.extra.get("anchor_mask"),
            open_rel_span_logits=open_rel_out.extra.get("span_logits"),
            open_rel_span_idx=open_rel_out.extra.get("span_idx"),
            open_rel_span_mask=open_rel_out.extra.get("span_mask"),
            count_logits=count_out.logits,
            groups_output=struct_out.extra.get("groups_output"),
            groups_mask=struct_out.extra.get("anchor_mask"),
            structuring_logits=struct_out.logits,
            structuring_anchor_mask=struct_out.extra.get("anchor_mask"),
            structuring_span_logits=struct_out.extra.get("span_logits"),
            structuring_span_idx=struct_out.extra.get("span_idx"),
            structuring_span_mask=struct_out.extra.get("span_mask"),
            embedding_logits=emb_out.logits,
            words_embedding=words_embedding,
            mask=mask,
            prompts_embedding=prompts_embedding,
            prompts_embedding_mask=prompts_embedding_mask,
        )

    def loss(self, *args, **kwargs):
        """Compute loss via forward pass."""
        output = self.forward(*args, **kwargs)
        return output.loss
