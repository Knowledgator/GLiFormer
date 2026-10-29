"""Host-sync-free replacements for gliner's prompt/pair extraction helpers.

``gliner.modeling.utils.extract_prompt_features`` and
``build_all_entity_pairs`` read device values on the host (``max()`` widths,
``torch.where`` and per-row ``.item()``), and every read stalls the CUDA
stream. The versions here return identical tensors but take the per-row
counts that set the output widths from the host: callers pass counts the data
collator computed (or that one batched read produced), and only fall back to
a single ``.tolist()`` when no counts are available.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

PROMPT_MARKER_COUNTS_KEY = "prompt_marker_counts"


def _marker_mask(token_index: int, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    return input_ids.eq(token_index) & attention_mask.gt(0)


def prompt_marker_token_indices(config) -> Tuple[int, ...]:
    """Every prompt-marker token id the forward pass extracts features for."""

    indices = set()

    def add(owner, name):
        value = getattr(owner, name, None) if owner is not None else None
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            indices.add(int(value))

    for name in (
        "class_token_index",
        "parent_token_index",
        "cat_token_index",
        "rel_token_index",
        "child_token_index",
        "obj_token_index",
    ):
        add(config, name)
    for cfg_name in (
        "ner_config",
        "classification_config",
        "joint_relex_config",
        "open_relex_config",
        "structuring_config",
        "image_classification_config",
        "object_detection_config",
        "segmentation_config",
        "audio_classification_config",
        "audio_segmentation_config",
    ):
        cfg = getattr(config, cfg_name, None)
        for name in ("parent_token_index", "cat_token_index", "rel_token_index", "child_token_index"):
            add(cfg, name)
    return tuple(sorted(indices))


def count_prompt_markers(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_indices: Iterable[int],
) -> Dict[int, List[int]]:
    """Per-row marker counts for several token ids with one host read."""

    token_indices = tuple(int(index) for index in token_indices)
    if not token_indices:
        return {}
    counts = torch.stack(
        [_marker_mask(index, input_ids, attention_mask).sum(dim=-1) for index in token_indices]
    )
    host = counts.cpu().tolist()
    return {index: row for index, row in zip(token_indices, host)}


class PromptMarkerCounts:
    """Per-forward host cache of marker counts for one ``input_ids`` tensor.

    Counts come from the collator (``prompt_marker_counts``) when it supplied
    them for these exact ids; any missing token ids are counted together with
    one host read the first time one of them is requested.
    """

    def __init__(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        token_indices: Sequence[int] = (),
        provided: Optional[Dict[int, Sequence[int]]] = None,
        text_lengths: Optional[torch.Tensor] = None,
        word_lengths: Optional[Sequence[int]] = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.text_lengths = text_lengths
        self.max_word_length: Optional[int] = None
        if (
            text_lengths is not None
            and word_lengths is not None
            and len(word_lengths) == text_lengths.numel()
            and len(word_lengths) > 0
        ):
            self.max_word_length = max(int(value) for value in word_lengths)
        self.token_indices = tuple(int(index) for index in token_indices)
        self._counts: Dict[int, List[int]] = {}
        batch = None if input_ids is None or input_ids.dim() != 2 else input_ids.shape[0]
        if provided and batch is not None:
            for index, row in provided.items():
                row = list(row)
                if len(row) == batch:
                    self._counts[int(index)] = [int(value) for value in row]

    def matches(self, input_ids, attention_mask) -> bool:
        return input_ids is self.input_ids and attention_mask is self.attention_mask

    def max_text_length(self, text_lengths) -> Optional[int]:
        """Host ``text_lengths.max()`` when the collator sent ``word_lengths``."""

        if text_lengths is self.text_lengths:
            return self.max_word_length
        return None

    def get(self, token_index: int) -> List[int]:
        token_index = int(token_index)
        if token_index not in self._counts:
            missing = [index for index in self.token_indices if index not in self._counts]
            if token_index not in missing:
                missing.append(token_index)
            self._counts.update(count_prompt_markers(self.input_ids, self.attention_mask, missing))
        return self._counts[token_index]


def extract_prompt_features(
    class_token_index: int,
    token_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int,
    embed_dim: int,
    embed_ent_token: bool = True,
    counts: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same outputs as gliner's ``extract_prompt_features`` without host reads.

    ``counts`` holds each row's number of ``class_token_index`` markers (under
    the attention mask) as Python ints; without it they are read once.
    """

    class_token_mask = _marker_mask(class_token_index, input_ids, attention_mask)
    num_class_tokens = class_token_mask.sum(dim=-1, keepdim=True)
    if counts is None:
        counts = num_class_tokens.view(-1).tolist()
    width = max((int(count) for count in counts), default=0)

    arange = torch.arange(width, dtype=attention_mask.dtype, device=token_embeds.device).expand(batch_size, -1)
    valid = arange < num_class_tokens
    prompts_embedding_mask = valid.to(attention_mask.dtype)

    # Stable sort puts each row's markers first, in position order.
    order = torch.argsort(class_token_mask.to(torch.int8), dim=1, descending=True, stable=True)[:, :width]
    if not embed_ent_token:
        order = order + 1
    order = order.masked_fill(~valid, 0)
    gathered = token_embeds.gather(1, order.unsqueeze(-1).expand(-1, -1, token_embeds.shape[-1]))
    prompts_embedding = gathered.masked_fill(~valid.unsqueeze(-1), 0)
    return prompts_embedding, prompts_embedding_mask


def all_entity_pair_index(counts: Sequence[int]) -> np.ndarray:
    """``(B, N, 2)`` host index of every ordered pair ``i != j`` per row (-1 padded)."""

    pair_lists = []
    for n in counts:
        n = int(n)
        if n < 2:
            pair_lists.append(np.zeros((0, 2), dtype=np.int64))
            continue
        idx = np.arange(n, dtype=np.int64)
        row = np.repeat(idx, n - 1)
        col = np.tile(idx, (n, 1))[~np.eye(n, dtype=bool)]
        pair_lists.append(np.stack([row, col], axis=-1))
    width = max((pairs.shape[0] for pairs in pair_lists), default=0)
    pair_idx = np.full((len(pair_lists), width, 2), -1, dtype=np.int64)
    for b, pairs in enumerate(pair_lists):
        pair_idx[b, : pairs.shape[0]] = pairs
    return pair_idx


def build_all_entity_pairs(
    span_rep: torch.Tensor,
    span_mask: torch.Tensor,
    counts: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same outputs as gliner's ``build_all_entity_pairs`` with one host read.

    The output width depends on the valid-entity counts, so they are read once
    (or taken from ``counts``) and the pair index is built on the host.
    """

    B, _, D = span_rep.shape
    device = span_rep.device
    if counts is None:
        counts = span_mask.long().sum(dim=1).tolist()
    pair_idx_host = all_entity_pair_index(counts)

    if pair_idx_host.shape[1] == 0:
        pair_idx = torch.full((B, 1, 2), -1, dtype=torch.long, device=device)
        pair_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)
        head_rep = tail_rep = torch.zeros((B, 1, D), dtype=span_rep.dtype, device=device)
        return pair_idx, pair_mask, head_rep, tail_rep

    from .tasks.matcher import host_to_device  # deferred: tasks import this module

    pair_idx = host_to_device(pair_idx_host, device)
    pair_mask = pair_idx[..., 0] >= 0
    batch_idx = torch.arange(B, device=device).unsqueeze(1)
    head_rep = span_rep[batch_idx, pair_idx[..., 0].clamp_min(0)]
    tail_rep = span_rep[batch_idx, pair_idx[..., 1].clamp_min(0)]
    return pair_idx, pair_mask, head_rep, tail_rep
