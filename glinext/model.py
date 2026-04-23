"""GLiNExT: unified multi-task model — thin orchestrator over modular task heads."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
from torch import nn
from transformers.utils import ModelOutput

from gliner.modeling.base import BaseModel
from gliner.modeling.encoder import Encoder, BiEncoder
from gliner.modeling.layers import CrossFuser, LstmSeq2SeqEncoder
from gliner.modeling.utils import (
    extract_word_embeddings,
    extract_prompt_features,
    extract_prompt_features_and_word_embeddings,
)

from .config import GLiNextConfig
from .tasks import SharedRepresentations, TaskFlatInputs, TaskHeadOutput
from .layers import AnchorModeling, AnchorCrossAttentionLayer
from .tasks.ner.model import NERHead
from .tasks.classification.model import ClassificationHead
from .tasks.joint_relex.model import JointRelexHead
from .tasks.open_relex.model import OpenRelexHead
from .tasks.count.model import CountHead
from .tasks.structuring.model import StructuringHead
from .tasks.embedding.model import EmbeddingHead


@dataclass
class GLiNExTOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    # Original batch size (B) for unflattening BN-indexed outputs
    batch_size: Optional[int] = None
    # NER
    ner_logits: Optional[torch.FloatTensor] = None
    ner_batch_origin: Optional[torch.LongTensor] = None
    span_logits: Optional[torch.FloatTensor] = None
    span_idx: Optional[torch.LongTensor] = None
    span_mask: Optional[torch.Tensor] = None
    # Classification
    cat_logits: Optional[torch.FloatTensor] = None
    cat_batch_origin: Optional[torch.LongTensor] = None
    # Joint Relex (NER + relation extraction)
    joint_rel_logits: Optional[torch.FloatTensor] = None
    joint_rel_batch_origin: Optional[torch.LongTensor] = None
    joint_rel_idx: Optional[torch.LongTensor] = None
    joint_rel_mask: Optional[torch.Tensor] = None
    # Open Relex (anchor-based relation extraction)
    open_rel_logits: Optional[torch.FloatTensor] = None
    open_rel_batch_origin: Optional[torch.LongTensor] = None
    open_rel_anchor_mask: Optional[torch.Tensor] = None
    open_rel_span_logits: Optional[torch.FloatTensor] = None
    open_rel_span_idx: Optional[torch.LongTensor] = None
    open_rel_span_mask: Optional[torch.Tensor] = None
    # Count
    count_logits: Optional[torch.FloatTensor] = None
    count_batch_origin: Optional[torch.LongTensor] = None
    # Groups / Structuring
    groups_output: Optional[torch.FloatTensor] = None
    groups_mask: Optional[torch.Tensor] = None
    # Structuring (anchor-based span extraction)
    structuring_logits: Optional[torch.FloatTensor] = None
    structuring_batch_origin: Optional[torch.LongTensor] = None
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
                    "structuring", "embedding"]

# Head classes in registration order
_HEAD_CLASSES = [NERHead, ClassificationHead, CountHead, JointRelexHead,
                 OpenRelexHead, StructuringHead, EmbeddingHead]


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

        # ── Shared layers (optional) ───────────────────────────────────
        shared_layers = {}
        if config.shared_anchor_modeling is not None:
            self.shared_anchor_modeling = AnchorModeling.from_config(
                config.shared_anchor_modeling, config.hidden_size, dropout=config.dropout,
            )
            shared_layers["anchor_modeling"] = self.shared_anchor_modeling

        if config.shared_anchor_refine_layers > 0:
            self.shared_anchor_refine = AnchorCrossAttentionLayer(
                config.hidden_size,
                num_heads=config.shared_anchor_refine_heads,
                num_layers=config.shared_anchor_refine_layers,
                dropout=config.dropout,
            )
            shared_layers["anchor_refine"] = self.shared_anchor_refine

        # ── Register task heads ─────────────────────────────────────────
        self.heads = nn.ModuleDict()
        for HeadClass in _HEAD_CLASSES:
            head = HeadClass.from_config(
                config,
                from_pretrained=from_pretrained,
                cache_dir=cache_dir,
                shared_layers=shared_layers,
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

    def _predict_structuring_counts_from_count_head(
        self,
        count_logits: Optional[torch.Tensor],
        flat_inputs_map: Dict[str, TaskFlatInputs],
    ) -> Optional[torch.Tensor]:
        """Project count-head outputs onto structuring groups during inference.

        The count head is built over a concatenated flat order of:
        classification groups, extraction groups, then structuring groups.
        Structuring counts therefore live in the final contiguous block.
        """
        if count_logits is None or "count" not in flat_inputs_map or "structuring" not in flat_inputs_map:
            return None

        struct_bn = flat_inputs_map["structuring"].batch_origin.shape[0]
        if struct_bn == 0:
            return None

        count_cfg = self.config.count_config
        if count_cfg and count_cfg.mode == "classification":
            predicted = count_logits.argmax(dim=-1)
        else:
            predicted = count_logits.squeeze(-1).round().long()

        predicted = predicted.clamp(min=0)
        if predicted.shape[0] < struct_bn:
            return None

        return predicted[-struct_bn:]

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

    # ── Prompt order for parent offset computation ────────────────
    # prepare_inputs adds groups in this order: classification → ner → open_relex → structuring
    _PROMPT_TASK_ORDER = ["classification", "ner", "open_relex", "structuring"]

    def _parent_offset_for_item(self, classes_mapping, task_name, batch_idx):
        """Compute the parent token offset for `task_name` within batch item `batch_idx`.

        Parents appear in prompt order: classification → ner → open_relex → structuring.
        Returns the index of the first parent for this task within the item's parent list.
        """
        offset = 0
        task_group_counts = {
            "classification": lambda i: len(classes_mapping.cat_mapping[i].cat_class_to_id),
            "ner": lambda i: len(classes_mapping.extraction_mapping[i].items),
            "joint_relex": lambda i: len(classes_mapping.extraction_mapping[i].items),
            "open_relex": lambda i: len(classes_mapping.open_relex_mapping[i].items)
                if hasattr(classes_mapping, 'open_relex_mapping') and i < len(classes_mapping.open_relex_mapping) else 0,
            "structuring": lambda i: len(classes_mapping.structuring_mapping[i].items)
                if hasattr(classes_mapping, 'structuring_mapping') and i < len(classes_mapping.structuring_mapping) else 0,
        }
        # Map joint_relex to same prompt position as ner
        effective_task = "ner" if task_name == "joint_relex" else task_name
        for t in self._PROMPT_TASK_ORDER:
            if t == effective_task:
                break
            offset += task_group_counts.get(t, lambda i: 0)(batch_idx)
        return offset

    def _child_sizes_for_task(self, classes_mapping, task_name, batch_idx, group_idx):
        """Number of child tokens for a specific group."""
        if task_name in ("ner", "joint_relex"):
            return len(classes_mapping.extraction_mapping[batch_idx].items[group_idx].ner_class_to_id.class_to_id)
        elif task_name == "classification":
            return len(classes_mapping.cat_mapping[batch_idx].cat_class_to_id[group_idx].class_to_id)
        elif task_name == "structuring":
            return len(classes_mapping.structuring_mapping[batch_idx].items[group_idx].field_class_to_id.class_to_id)
        elif task_name == "open_relex":
            return len(classes_mapping.open_relex_mapping[batch_idx].items[group_idx].rel_class_to_id.class_to_id)
        return 0

    def _get_flat_iter(self, classes_mapping, task_name):
        """Return the appropriate flat iterator for a task."""
        iters = {
            "ner": classes_mapping.flat_extraction_iter,
            "joint_relex": classes_mapping.flat_extraction_iter,
            "classification": classes_mapping.flat_cat_iter,
            "structuring": classes_mapping.flat_structuring_iter,
            "open_relex": classes_mapping.flat_open_relex_iter,
        }
        return iters[task_name]

    def _build_flat_rel_prompts(
        self,
        rel_prompts_batch: torch.Tensor,
        rel_prompts_batch_mask: torch.Tensor,
        classes_mapping,
        embed_dim: int,
        device,
        dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Split batch-level [REL] prompts into per-extraction-group tensors.

        The prompt for each batch item concatenates [REL] tokens across its
        extraction groups in group order, so we slice by per-group rel-class
        counts to recover (BN, max_C_rel, D) aligned with the joint_relex
        flat_inputs batch axis.
        """
        slices: List[Tuple[int, int, int]] = []
        item_rel_offset: dict = {}
        for _, batch_idx, group_idx, ext_mapping in classes_mapping.flat_extraction_iter():
            rel_map = ext_mapping.rel_class_to_id
            n_rel = len(rel_map.class_to_id) if rel_map is not None else 0
            start = item_rel_offset.get(batch_idx, 0)
            slices.append((batch_idx, start, start + n_rel))
            item_rel_offset[batch_idx] = start + n_rel

        BN = len(slices)
        if BN == 0:
            return None, None

        max_C = max((ce - cs for _, cs, ce in slices), default=0)
        flat = torch.zeros(BN, max_C, embed_dim, device=device, dtype=dtype)
        flat_mask = torch.zeros(BN, max_C, device=device, dtype=rel_prompts_batch_mask.dtype)
        for idx, (bi, cs, ce) in enumerate(slices):
            n = ce - cs
            if n > 0 and ce <= rel_prompts_batch.shape[1]:
                flat[idx, :n] = rel_prompts_batch[bi, cs:ce]
                flat_mask[idx, :n] = rel_prompts_batch_mask[bi, cs:ce]
        return flat, flat_mask

    def _build_flat_rel_prompts_from_label_embeds(
        self,
        rel_label_embeds: torch.Tensor,
        label_group_sizes: Optional[torch.Tensor],
        classes_mapping,
        embed_dim: int,
        device,
        dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Split labels-encoder rel embeds into per-group tensors via label_group_sizes."""
        if label_group_sizes is None:
            return None, None
        BN = int(label_group_sizes.shape[0])
        if BN == 0:
            return None, None

        batch_origins: List[int] = []
        for _, batch_idx, _, _ in classes_mapping.flat_extraction_iter():
            batch_origins.append(batch_idx)

        cumsum = torch.cumsum(label_group_sizes, 0)
        max_C = int(label_group_sizes.max().item()) if BN > 0 else 0
        flat = torch.zeros(BN, max_C, embed_dim, device=device, dtype=dtype)
        flat_mask = torch.zeros(BN, max_C, device=device, dtype=torch.long)
        for g in range(BN):
            c_start = 0 if g == 0 else int(cumsum[g - 1].item())
            c_end = int(cumsum[g].item())
            n = c_end - c_start
            if n > 0 and c_end <= rel_label_embeds.shape[1]:
                flat[g, :n] = rel_label_embeds[batch_origins[g], c_start:c_end]
                flat_mask[g, :n] = 1
        return flat, flat_mask

    def _build_flat_inputs(
        self,
        parent_embeds: torch.Tensor,
        parent_mask: torch.Tensor,
        child_embeds: torch.Tensor,
        child_mask: Optional[torch.Tensor],
        words_embedding: torch.Tensor,
        word_mask: torch.Tensor,
        classes_mapping,
        task_name: str,
        label_group_sizes: Optional[torch.Tensor] = None,
        per_task_parents: bool = False,
    ) -> Optional[TaskFlatInputs]:
        """Build TaskFlatInputs for a specific task by flattening B-indexed tensors to BN.

        Args:
            parent_embeds: (B, max_P, D) parent token embeddings
            parent_mask: (B, max_P) parent mask
            child_embeds: (B, max_C, D) task-specific child embeddings
            child_mask: (B, max_C) or None
            words_embedding: (B, W, D)
            word_mask: (B, W)
            classes_mapping: BatchClassesMapping
            task_name: "ner", "classification", "structuring", "open_relex", "joint_relex"
            label_group_sizes: (BN,) for labels encoder path — number of children per flat group
            per_task_parents: when True, parent_embeds contains only this task's parents
                (no offset needed); when False, uses shared parent tensor with offset computation
        """
        device = words_embedding.device
        D = words_embedding.shape[-1]

        flat_iter = self._get_flat_iter(classes_mapping, task_name)

        # Collect group descriptors
        batch_origins: List[int] = []
        parent_positions: List[Tuple[int, int]] = []  # (batch_idx, parent_pos)
        child_slices: List[Tuple[int, int, int]] = []  # (batch_idx, start, end)

        # Track per-item child offset for prompt-based splitting
        item_child_offset: dict = {}

        for flat_idx, batch_idx, group_idx, _ in flat_iter():
            batch_origins.append(batch_idx)

            # Parent position within the parent tensor
            if per_task_parents:
                # Per-task parents: positions are task-local (no offset)
                parent_positions.append((batch_idx, group_idx))
            else:
                # Shared parents: offset by preceding tasks in prompt order
                p_offset = self._parent_offset_for_item(classes_mapping, task_name, batch_idx)
                parent_positions.append((batch_idx, p_offset + group_idx))

            if label_group_sizes is None:
                # Prompt path: children within batch item, accumulated by group
                if batch_idx not in item_child_offset:
                    item_child_offset[batch_idx] = 0
                c_start = item_child_offset[batch_idx]
                c_size = self._child_sizes_for_task(classes_mapping, task_name, batch_idx, group_idx)
                child_slices.append((batch_idx, c_start, c_start + c_size))
                item_child_offset[batch_idx] = c_start + c_size

        BN = len(batch_origins)
        if BN == 0:
            return None

        batch_origin = torch.tensor(batch_origins, dtype=torch.long, device=device)

        # Gather words (BN, W, D) and mask (BN, W)
        flat_words = words_embedding[batch_origin]
        flat_word_mask = word_mask[batch_origin]

        # Gather parent embeddings (BN, D)
        flat_parent = torch.zeros(BN, D, device=device, dtype=parent_embeds.dtype)
        for idx, (bi, pp) in enumerate(parent_positions):
            if pp < parent_embeds.shape[1]:
                flat_parent[idx] = parent_embeds[bi, pp]

        # Gather and pad child embeddings (BN, max_C_per_group, D)
        if label_group_sizes is not None:
            # Labels encoder path: children indexed flat across all groups
            cumsum = torch.cumsum(label_group_sizes, 0)
            max_C = int(label_group_sizes.max().item()) if BN > 0 else 0
            flat_children = torch.zeros(BN, max_C, D, device=device, dtype=child_embeds.dtype)
            flat_child_mask = torch.zeros(BN, max_C, device=device, dtype=torch.float)
            for g in range(BN):
                c_start = 0 if g == 0 else int(cumsum[g - 1].item())
                c_end = int(cumsum[g].item())
                n = c_end - c_start
                if n > 0 and c_end <= child_embeds.shape[1]:
                    flat_children[g, :n] = child_embeds[batch_origins[g], c_start:c_end]
                    flat_child_mask[g, :n] = 1.0
        else:
            # Prompt path: children per batch item, split by group sizes
            max_C = max((ce - cs for _, cs, ce in child_slices), default=0)
            flat_children = torch.zeros(BN, max_C, D, device=device, dtype=child_embeds.dtype)
            flat_child_mask = torch.zeros(BN, max_C, device=device, dtype=torch.float)
            for idx, (bi, cs, ce) in enumerate(child_slices):
                n = ce - cs
                if n > 0 and ce <= child_embeds.shape[1]:
                    flat_children[idx, :n] = child_embeds[bi, cs:ce]
                    if child_mask is not None:
                        flat_child_mask[idx, :n] = child_mask[bi, cs:ce].float()
                    else:
                        flat_child_mask[idx, :n] = 1.0

        return TaskFlatInputs(
            words_embedding=flat_words,
            mask=flat_word_mask,
            parent_embedding=flat_parent,
            child_embedding=flat_children,
            child_mask=flat_child_mask,
            batch_origin=batch_origin,
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
        rel_span_idx: Optional[torch.Tensor] = None,
        rel_span_mask: Optional[torch.Tensor] = None,
        # Open Relex
        open_rel_labels: Optional[torch.Tensor] = None,
        open_rel_count: Optional[torch.Tensor] = None,
        # Count
        count_targets: Optional[torch.Tensor] = None,
        # Groups / Structuring
        count_val: Optional[torch.Tensor] = None,
        structuring_labels: Optional[torch.Tensor] = None,
        structuring_count: Optional[torch.Tensor] = None,
        # Embedding similarity
        embedding_labels: Optional[torch.Tensor] = None,
        embedding_pair_idx: Optional[torch.Tensor] = None,
        embedding_input_ids: Optional[torch.Tensor] = None,
        embedding_attention_mask: Optional[torch.Tensor] = None,
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

        classes_mapping = kwargs.get("classes_mapping")

        # ── 1. Encode ────────────────────────────────────────────────────
        token_embeds, prompts_embedding, prompts_embedding_mask, words_embedding, mask = (
            self.get_representations(
                input_ids, attention_mask, text_lengths, words_mask,
                labels_input_ids=labels_input_ids,
                labels_attention_mask=labels_attention_mask,
                **{k: kwargs[k] for k in ("packing_config", "pair_attention_mask") if k in kwargs},
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
        embed_dim = words_embedding.shape[-1]
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

        # ── 1c. Extract parent embeddings ───────────────────────────────
        parent_embeds = None
        parent_mask_t = None
        per_task_parent_embeds: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        if classes_mapping is not None:
            if self.config.uses_per_task_parents:
                # Per-task parent tokens: extract each task's parents independently
                _task_parent_cfgs = {
                    "ner": self.config.ner_config,
                    "classification": self.config.classification_config,
                    "open_relex": self.config.open_relex_config,
                    "structuring": self.config.structuring_config,
                }
                for t_name, t_cfg in _task_parent_cfgs.items():
                    if t_cfg is not None and getattr(t_cfg, 'parent_token_index', -1) > 0:
                        p_e, p_m = extract_prompt_features(
                            t_cfg.parent_token_index, token_embeds, input_ids, attention_mask,
                            batch_size, embed_dim, getattr(t_cfg, 'embed_parent_token', True),
                        )
                        per_task_parent_embeds[t_name] = (p_e, p_m)
                # Joint relex shares NER's parent token
                if "ner" in per_task_parent_embeds:
                    per_task_parent_embeds["joint_relex"] = per_task_parent_embeds["ner"]
            elif self.config.parent_token_index > 0:
                # Legacy: single shared parent token for all tasks
                parent_embeds, parent_mask_t = extract_prompt_features(
                    self.config.parent_token_index, token_embeds, input_ids, attention_mask,
                    batch_size, embed_dim, self.config.embed_parent_token,
                )

        # ── 1d. Extract task-specific child embeddings (prompt path) ────
        # NER children = prompts_embedding (already extracted via class_token_index)
        ner_child_embeds = prompts_embedding
        ner_child_mask = prompts_embedding_mask

        # Classification children (if active and not using labels encoder)
        cat_child_embeds, cat_child_mask = None, None
        if "classification" in self.heads and cat_label_embeds is None:
            cat_cfg = self.config.classification_config
            cat_child_embeds, cat_child_mask = extract_prompt_features(
                cat_cfg.cat_token_index, token_embeds, input_ids, attention_mask,
                batch_size, embed_dim, cat_cfg.embed_cat_token,
            )
        elif cat_label_embeds is not None:
            cat_child_embeds = cat_label_embeds
            cat_child_mask = torch.ones(
                cat_label_embeds.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device,
            )

        # Joint relex: per-group rel prompts (split per extraction group)
        joint_rel_flat_prompts, joint_rel_flat_mask = None, None
        if "joint_relex" in self.heads and rel_label_embeds is None and classes_mapping is not None:
            jr_cfg = self.config.joint_relex_config
            rel_prompts_batch, rel_prompts_batch_mask = extract_prompt_features(
                jr_cfg.rel_token_index, token_embeds, input_ids, attention_mask,
                batch_size, embed_dim, jr_cfg.embed_rel_token,
            )
            joint_rel_flat_prompts, joint_rel_flat_mask = self._build_flat_rel_prompts(
                rel_prompts_batch, rel_prompts_batch_mask, classes_mapping, embed_dim,
                device=words_embedding.device, dtype=rel_prompts_batch.dtype,
            )
        elif "joint_relex" in self.heads and rel_label_embeds is not None and classes_mapping is not None:
            joint_rel_flat_prompts, joint_rel_flat_mask = self._build_flat_rel_prompts_from_label_embeds(
                rel_label_embeds, kwargs.get("rel_labels_group_size"), classes_mapping,
                embed_dim, device=words_embedding.device, dtype=rel_label_embeds.dtype,
            )

        # Open relex children
        open_rel_child_embeds, open_rel_child_mask = None, None
        if "open_relex" in self.heads and open_rel_label_embeds is None:
            or_cfg = self.config.open_relex_config
            open_rel_child_embeds, open_rel_child_mask = extract_prompt_features(
                or_cfg.rel_token_index, token_embeds, input_ids, attention_mask,
                batch_size, embed_dim, or_cfg.embed_rel_token,
            )
        elif open_rel_label_embeds is not None:
            open_rel_child_embeds = open_rel_label_embeds
            open_rel_child_mask = torch.ones(
                open_rel_label_embeds.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device,
            )

        # Structuring children
        struct_child_embeds, struct_child_mask = None, None
        if "structuring" in self.heads and child_label_embeds is None:
            s_cfg = self.config.structuring_config
            struct_child_embeds, struct_child_mask = extract_prompt_features(
                s_cfg.child_token_index, token_embeds, input_ids, attention_mask,
                batch_size, embed_dim, s_cfg.embed_child_token,
            )
        elif child_label_embeds is not None:
            struct_child_embeds = child_label_embeds
            struct_child_mask = torch.ones(
                child_label_embeds.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device,
            )

        # ── 1e. Build TaskFlatInputs per task ───────────────────────────
        flat_inputs_map = {}
        _use_per_task = bool(per_task_parent_embeds)
        _has_parents = parent_embeds is not None or _use_per_task

        if classes_mapping is not None and _has_parents:
            label_group_sizes_map = {
                "ner": kwargs.get("ner_labels_group_size"),
                "classification": kwargs.get("cat_labels_group_size"),
                "structuring": kwargs.get("child_labels_group_size"),
                "open_relex": kwargs.get("open_rel_labels_group_size"),
            }

            task_child_map = {
                "ner": (ner_child_embeds, ner_child_mask),
                "joint_relex": (ner_child_embeds, ner_child_mask),
                "classification": (cat_child_embeds, cat_child_mask),
                "structuring": (struct_child_embeds, struct_child_mask),
                "open_relex": (open_rel_child_embeds, open_rel_child_mask),
            }

            for task_name in ("ner", "joint_relex", "classification", "structuring", "open_relex"):
                if task_name not in self.heads:
                    continue
                child_e, child_m = task_child_map.get(task_name, (None, None))
                if child_e is None:
                    continue

                # Select parent embeddings: per-task or shared
                if _use_per_task:
                    if task_name not in per_task_parent_embeds:
                        continue
                    t_parent_e, t_parent_m = per_task_parent_embeds[task_name]
                else:
                    t_parent_e, t_parent_m = parent_embeds, parent_mask_t

                lgs = label_group_sizes_map.get(
                    "ner" if task_name == "joint_relex" else task_name
                )
                fi = self._build_flat_inputs(
                    t_parent_e, t_parent_m,
                    child_e, child_m,
                    words_embedding, mask,
                    classes_mapping, task_name,
                    label_group_sizes=lgs,
                    per_task_parents=_use_per_task,
                )
                if fi is not None:
                    flat_inputs_map[task_name] = fi

            # Count uses ALL parents (cat + ner + struct), so build a combined flat input
            if "count" in self.heads:
                count_batch_origins = []
                count_parent_embeds_list = []

                _count_tasks = [
                    ("classification", classes_mapping.flat_cat_iter),
                    ("ner", classes_mapping.flat_extraction_iter),
                    ("structuring", classes_mapping.flat_structuring_iter),
                ]

                for c_task, c_iter in _count_tasks:
                    if _use_per_task:
                        t_pe = per_task_parent_embeds.get(c_task)
                        if t_pe is None:
                            continue
                        t_parent_e, _ = t_pe
                        for _, bi, gi, _ in c_iter():
                            count_batch_origins.append(bi)
                            if gi < t_parent_e.shape[1]:
                                count_parent_embeds_list.append(t_parent_e[bi, gi])
                            else:
                                count_parent_embeds_list.append(
                                    torch.zeros(embed_dim, device=words_embedding.device, dtype=t_parent_e.dtype)
                                )
                    else:
                        for _, bi, gi, _ in c_iter():
                            count_batch_origins.append(bi)
                            p_off = self._parent_offset_for_item(classes_mapping, c_task, bi)
                            pp = p_off + gi
                            if pp < parent_embeds.shape[1]:
                                count_parent_embeds_list.append(parent_embeds[bi, pp])
                            else:
                                count_parent_embeds_list.append(
                                    torch.zeros(embed_dim, device=words_embedding.device, dtype=parent_embeds.dtype)
                                )

                BN_count = len(count_batch_origins)
                if BN_count > 0:
                    bo = torch.tensor(count_batch_origins, dtype=torch.long, device=words_embedding.device)
                    flat_parent = torch.stack(count_parent_embeds_list)
                    flat_inputs_map["count"] = TaskFlatInputs(
                        words_embedding=words_embedding[bo],
                        mask=mask[bo],
                        parent_embedding=flat_parent,
                        child_embedding=torch.zeros(BN_count, 0, embed_dim, device=words_embedding.device),
                        child_mask=torch.zeros(BN_count, 0, device=words_embedding.device),
                        batch_origin=bo,
                    )

        # ── 1e. Encode embedding pair texts (separate batch) ───────────
        embedding_encodings = None
        embedding_encoding_mask = None
        if embedding_input_ids is not None and embedding_pair_idx is not None:
            emb_token_embeds = self.token_rep_layer(embedding_input_ids, embedding_attention_mask)
            embedding_encodings = emb_token_embeds
            embedding_encoding_mask = embedding_attention_mask

        # Collect all batch kwargs for heads
        batch_kwargs = dict(
            ner_labels=ner_labels, span_idx=span_idx, span_mask=span_mask,
            span_labels=span_labels, cat_labels=cat_labels, rel_labels=rel_labels,
            rel_span_idx=rel_span_idx, rel_span_mask=rel_span_mask,
            open_rel_labels=open_rel_labels, open_rel_count=open_rel_count,
            count_targets=count_targets,
            count_val=count_val, structuring_labels=structuring_labels,
            structuring_count=structuring_count, embedding_labels=embedding_labels,
            embedding_pair_idx=embedding_pair_idx,
            embedding_encodings=embedding_encodings,
            embedding_encoding_mask=embedding_encoding_mask,
            threshold=threshold, adjacency_threshold=adjacency_threshold,
        )

        # ── 2. Execute heads in order ───────────────────────────────────
        head_outputs = {}
        for name in _EXECUTION_ORDER:
            if name not in self.heads:
                continue

            # Skip anchor-based heads that have no flat_inputs — they require
            # flattened parent/child embeddings to operate.
            if name in ("ner", "classification", "count", "joint_relex", "open_relex", "structuring"):
                if name not in flat_inputs_map:
                    continue

            head = self.heads[name]

            dep_outputs = {d: head_outputs[d] for d in head.dependencies if d in head_outputs}

            extra_kwargs = {}

            # Pass flat_inputs if available
            if name in flat_inputs_map:
                extra_kwargs["flat_inputs"] = flat_inputs_map[name]

            if name == "structuring" and structuring_count is None:
                predicted_structuring_count = self._predict_structuring_counts_from_count_head(
                    head_outputs.get("count", TaskHeadOutput()).logits,
                    flat_inputs_map,
                )
                if predicted_structuring_count is not None:
                    extra_kwargs["structuring_count"] = predicted_structuring_count

            # Pass task-specific label embeds for heads that use them directly
            if name == "joint_relex" and rel_label_embeds is not None:
                extra_kwargs["rel_label_embeds"] = rel_label_embeds
            if name == "joint_relex":
                extra_kwargs["flat_rel_prompts"] = joint_rel_flat_prompts
                extra_kwargs["flat_rel_prompts_mask"] = joint_rel_flat_mask

            # Pass base_loss_fn for heads that need it
            if name in ("ner", "joint_relex", "open_relex", "structuring"):
                extra_kwargs["base_loss_fn"] = self._loss

            call_kwargs = dict(batch_kwargs)
            call_kwargs.update(extra_kwargs)
            output = head(shared, dependency_outputs=dep_outputs, **call_kwargs)
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

        # Prefer the standalone NER head for exported NER outputs.
        # Fall back to joint_relex only when NER is not active.
        effective_ner_logits = ner_out.logits if ner_out.logits is not None else joint_rel_out.logits
        effective_ner_extra = ner_out.extra if ner_out.logits is not None else joint_rel_out.extra
        effective_ner_origin = (
            flat_inputs_map["ner"].batch_origin if "ner" in flat_inputs_map
            else flat_inputs_map["joint_relex"].batch_origin if "joint_relex" in flat_inputs_map
            else flat_inputs_map.get("ner", TaskFlatInputs(
                words_embedding=words_embedding, mask=mask,
                parent_embedding=torch.empty(0), child_embedding=torch.empty(0),
                child_mask=torch.empty(0), batch_origin=torch.arange(batch_size, device=words_embedding.device),
            )).batch_origin
        )

        return GLiNExTOutput(
            loss=final_loss,
            batch_size=batch_size,
            ner_logits=effective_ner_logits,
            ner_batch_origin=effective_ner_origin,
            span_logits=effective_ner_extra.get("span_logits"),
            span_idx=effective_ner_extra.get("span_idx"),
            span_mask=effective_ner_extra.get("span_mask"),
            cat_logits=cat_out.logits,
            cat_batch_origin=flat_inputs_map["classification"].batch_origin if "classification" in flat_inputs_map else None,
            joint_rel_logits=joint_rel_out.extra.get("rel_logits"),
            joint_rel_batch_origin=flat_inputs_map["joint_relex"].batch_origin if "joint_relex" in flat_inputs_map else None,
            joint_rel_idx=joint_rel_out.extra.get("rel_idx"),
            joint_rel_mask=joint_rel_out.extra.get("rel_mask"),
            open_rel_logits=open_rel_out.logits,
            open_rel_batch_origin=flat_inputs_map["open_relex"].batch_origin if "open_relex" in flat_inputs_map else None,
            open_rel_anchor_mask=open_rel_out.extra.get("anchor_mask"),
            open_rel_span_logits=open_rel_out.extra.get("span_logits"),
            open_rel_span_idx=open_rel_out.extra.get("span_idx"),
            open_rel_span_mask=open_rel_out.extra.get("span_mask"),
            count_logits=count_out.logits,
            count_batch_origin=flat_inputs_map["count"].batch_origin if "count" in flat_inputs_map else None,
            groups_output=struct_out.extra.get("groups_output"),
            groups_mask=struct_out.extra.get("anchor_mask"),
            structuring_logits=struct_out.logits,
            structuring_batch_origin=flat_inputs_map["structuring"].batch_origin if "structuring" in flat_inputs_map else None,
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
