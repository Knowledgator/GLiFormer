"""Sequence editing head.

For every flat group (one ``[EDIT] <instruction>`` prompt section):

1. units = ``[BOS] + text units`` (words or subwords); each is fused with the
   group's ``[EDIT]`` marker vector and gets an op: KEEP, DELETE, REPLACE or
   APPEND (BOS: KEEP or APPEND only);
2. every REPLACE/APPEND position gets ``M`` slot queries (unit state + slot
   position + op embedding). A small non-autoregressive decoder refines them
   (self-attention between the slots of one edit, cross-attention to the
   row's encoder tokens, prompt included);
3. each slot scores the backbone's input embedding matrix (tied output, no
   vocabulary-sized layer) and, with ``copy``, points at input tokens; a gate
   mixes both distributions. A length head picks how many slots are used and
   a space head whether whitespace precedes the content.

Content logits are computed only at gold content slots (training) or at
predicted edits (inference), never for every unit, so the full softmax over
the vocabulary stays cheap.

``content_mode: tags`` replaces steps 2-3 with token classification
(GECToR-style): every REPLACE/APPEND unit picks one string of a fixed
``tag_vocab`` (the most frequent training contents, space bit included)
with a single MLP over the unit state plus the op embedding.
"""

from __future__ import annotations

import weakref

import torch
from torch import nn
from torch.nn import functional as F

from .. import TaskHead, TaskHeadOutput, text_token_states
from ..calculation.model import marker_positions
from .alignment import APPEND, DELETE, KEEP, NUM_OPS, REPLACE

IGNORE = -100
# Rows of the vocabulary softmax computed at once at inference.
_VOCAB_CHUNK = 512


def _loss_stats(terms) -> dict:
    """Detached loss terms; kept on the device so logging them costs no sync."""

    return {name: value.detach() for name, value in terms.items() if torch.is_tensor(value)}


def _mlp(hidden: int, out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out),
    )


class _RowPacking:
    """Packs per-edit slots ``(E, M, D)`` into per-row blocks ``(R, P, D)``.

    Cross-attention then runs once per batch row instead of once per edit.
    """

    def __init__(self, edit_rows: torch.Tensor, num_slots: int):
        self.rows, local = torch.unique(edit_rows, return_inverse=True)
        num_rows = int(self.rows.numel())
        order = torch.argsort(local, stable=True)
        counts = torch.bincount(local, minlength=num_rows)
        starts = torch.cumsum(counts, 0) - counts
        rank = torch.empty_like(local)
        rank[order] = torch.arange(local.numel(), device=local.device) - starts[local[order]]
        self.local = local
        self.width = int(counts.max().item()) * num_slots if local.numel() else num_slots
        self.num_slots = num_slots
        slot = torch.arange(num_slots, device=local.device)
        self.flat = (local[:, None] * self.width + rank[:, None] * num_slots + slot[None]).reshape(-1)
        self.num_rows = num_rows

    def pack(self, x: torch.Tensor) -> torch.Tensor:
        dim = x.shape[-1]
        out = x.new_zeros(self.num_rows * self.width, dim)
        out[self.flat] = x.reshape(-1, dim)
        return out.view(self.num_rows, self.width, dim)

    def unpack(self, y: torch.Tensor) -> torch.Tensor:
        dim = y.shape[-1]
        return y.reshape(-1, dim)[self.flat].view(-1, self.num_slots, dim)


class SlotDecoderLayer(nn.Module):
    """Pre-norm: slot self-attention, cross-attention to tokens, FFN.

    With ``local`` tokens (the edited unit's subwords and its neighbours'),
    the slots attend to them together with each other.
    """

    def __init__(self, hidden: int, heads: int, dropout: float):
        super().__init__()
        self.norm_self = nn.LayerNorm(hidden)
        self.norm_local = nn.LayerNorm(hidden)
        self.self_attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(hidden)
        self.cross_attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * hidden, hidden),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, slots, packing: _RowPacking, memory, memory_pad, local=None, local_pad=None):
        h = self.norm_self(slots)
        if local is not None:
            keys = torch.cat([self.norm_local(local), h], dim=1)
            pad = torch.cat([local_pad, local_pad.new_zeros(h.shape[:2])], dim=1)
            attended = self.self_attn(h, keys, keys, key_padding_mask=pad, need_weights=False)[0]
        else:
            attended = self.self_attn(h, h, h, need_weights=False)[0]
        slots = slots + self.dropout(attended)
        h = packing.pack(self.norm_cross(slots))
        attended = self.cross_attn(h, memory, memory, key_padding_mask=memory_pad, need_weights=False)[0]
        slots = slots + self.dropout(packing.unpack(attended))
        return slots + self.dropout(self.ffn(self.norm_ffn(slots)))


def input_embedding_module(token_rep_layer):
    getter = getattr(token_rep_layer, "get_input_embeddings", None)
    if getter is not None:
        return getter()
    return token_rep_layer.bert_layer.model.get_input_embeddings()


class SequenceEditingHead(TaskHead):
    """Per-unit edit ops plus tied-vocabulary/copy content slots."""

    name = "sequence_editing"
    dependencies = []

    def __init__(self, config, hidden_size: int, dropout: float, embedding_dim: int | None = None):
        super().__init__()
        cfg = config.sequence_editing_config
        self.config = config
        self.cfg = cfg
        self.loss_coef = float(cfg.loss_coef)
        self.hidden_size = hidden_size
        self.max_content = int(cfg.max_content_len)
        embedding_dim = int(embedding_dim or hidden_size)

        self.bos = nn.Parameter(torch.zeros(hidden_size))
        self.bos_proj = nn.Linear(hidden_size, hidden_size)
        self.fuse = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.fuse_norm = nn.LayerNorm(hidden_size)
        self.context = nn.ModuleList(
            nn.TransformerEncoderLayer(
                hidden_size, cfg.decoder_heads, 4 * hidden_size, dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(cfg.context_layers)
        )
        self.op_head = _mlp(hidden_size, NUM_OPS, dropout)
        self.op_embed = nn.Embedding(NUM_OPS, hidden_size)
        self.tag_mode = cfg.content_mode == "tags"
        self.__dict__["_vocab_source"] = None
        if self.tag_mode:
            self.tag_head = _mlp(hidden_size, len(cfg.tag_vocab), dropout)
            self.local_window = 0
            return
        self.length_head = _mlp(hidden_size, self.max_content, dropout)
        self.space_head = _mlp(hidden_size, 1, dropout)

        # Unit scale, like the LayerNorm'd unit states it is added to: a small
        # init leaves the slots nearly identical and they copy each other.
        self.slot_pos = nn.Parameter(torch.randn(self.max_content, hidden_size))
        self.decoder = nn.ModuleList(
            SlotDecoderLayer(hidden_size, cfg.decoder_heads, dropout) for _ in range(cfg.decoder_layers)
        )
        self.out_norm = nn.LayerNorm(hidden_size)
        self.local_window = int(cfg.local_window)
        if self.local_window > 0:
            # Relative unit position (-w..w) of every local token.
            self.local_type = nn.Embedding(2 * self.local_window + 1, hidden_size)
        self.vocab_proj = nn.Linear(hidden_size, embedding_dim)
        if cfg.copy:
            self.pointer_query = nn.Linear(hidden_size, hidden_size)
            self.pointer_key = nn.Linear(hidden_size, hidden_size)
            self.copy_gate = nn.Linear(hidden_size, 1)
        # Set by the model: returns the backbone input embedding matrix.
        self.__dict__["_vocab_source"] = None

    @classmethod
    def from_config(cls, config, **kwargs):
        if getattr(config, "sequence_editing_config", None) is None:
            return None
        return cls(
            config,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            embedding_dim=kwargs.get("embedding_dim"),
        )

    def attach_vocab(self, model) -> None:
        """Read the output vocabulary from ``model.token_rep_layer`` (weakly held)."""

        reference = weakref.ref(model)
        self.__dict__["_vocab_source"] = lambda: input_embedding_module(reference().token_rep_layer).weight

    @property
    def edit_token_index(self) -> int:
        return int(getattr(self.cfg, "edit_token_index", -1))

    # -- pieces ----------------------------------------------------------------

    def _vocab_weight(self):
        source = self.__dict__.get("_vocab_source")
        if source is None:
            raise RuntimeError("SequenceEditingHead has no vocabulary; call attach_vocab(model)")
        weight = source()
        return weight.detach() if self.cfg.detach_vocab else weight

    def _vocab_logits(self, hidden):
        weight = self._vocab_weight()
        # No per-token bias: ``vocab_proj``'s bias gives a prior through the
        # embeddings, and the head stays independent of the vocabulary size.
        return (self.vocab_proj(hidden).to(weight.dtype) @ weight.t()).float()

    def _unit_states(self, units, unit_mask, query):
        """``[BOS] + units`` fused with the group query: ``(G, W + 1, D)``."""

        bos = (self.bos_proj(query) + self.bos).unsqueeze(1)
        seq = torch.cat([bos, units], dim=1)
        mask = torch.cat([unit_mask.new_ones(unit_mask.shape[0], 1), unit_mask], dim=1)
        q = query.unsqueeze(1).expand_as(seq)
        states = self.fuse_norm(seq + self.fuse(torch.cat([seq, q, seq * q], dim=-1)))
        for layer in self.context:
            states = layer(states, src_key_padding_mask=~mask)
        return states, mask

    def _op_logits(self, states):
        logits = self.op_head(states).float()
        # BOS can only keep (nothing) or append.
        blocked = torch.zeros(NUM_OPS, dtype=torch.bool, device=logits.device)
        blocked[DELETE] = True
        blocked[REPLACE] = True
        logits[:, 0] = logits[:, 0].masked_fill(blocked, -1e4)
        return logits

    def _local_context(self, shared, batch, edit_rows, edit_slots):
        """Subword states of units ``slot - w .. slot + w`` per edit: ``(E, K, D)``, pad mask.

        ``span_token_words`` numbers units 1..n at their (first) subwords;
        forward-filling it assigns every later subword to its unit. Slot ``s``
        is unit ``s`` (slot 0 = BOS sees the first units).
        """

        window = self.local_window
        token_units = batch.get("span_token_words")
        tokens = text_token_states(shared)
        if window <= 0 or token_units is None or token_units.shape != tokens.shape[:2]:
            return None, None
        device = tokens.device
        units = torch.cummax(token_units.to(device).long()[edit_rows], dim=1).values
        length = tokens.shape[1]
        position = torch.arange(length, device=device)
        if shared.attention_mask is not None:
            attended = shared.attention_mask.to(device)[edit_rows].bool()
            # The last attended token closes the sequence (special token).
            last = attended.long().sum(dim=1) - 1
            valid = attended & (position[None] < last[:, None])
        else:
            valid = torch.ones_like(units, dtype=torch.bool)
        relative = units - edit_slots[:, None]
        inside = valid & (units >= 1) & (relative.abs() <= window)
        width = min(int(self.cfg.local_max_tokens), length)
        order = torch.where(inside, position[None], position[None] + length).argsort(dim=1)[:, :width]
        keep = inside.gather(1, order)
        local = tokens[edit_rows[:, None], order]
        local = local + self.local_type(relative.gather(1, order).clamp(-window, window) + window).to(local.dtype)
        return local, ~keep

    def _decode_slots(self, edit_states, edit_ops, edit_rows, shared, local=None, local_pad=None):
        """Refined slot states ``(E, M, D)`` and the row packing used."""

        slots = (
            edit_states[:, None, :]
            + self.slot_pos[None].to(edit_states.dtype)
            + self.op_embed(edit_ops)[:, None, :]
        )
        packing = _RowPacking(edit_rows, self.max_content)
        memory = text_token_states(shared)[packing.rows]
        if shared.attention_mask is not None:
            memory_pad = ~shared.attention_mask[packing.rows].bool()
        else:
            memory_pad = torch.zeros(memory.shape[:2], dtype=torch.bool, device=memory.device)
        for layer in self.decoder:
            slots = layer(slots, packing, memory, memory_pad, local, local_pad)
        return self.out_norm(slots), packing, memory, memory_pad

    def _pointer_logits(self, slots, packing, memory, memory_pad):
        """``(E, M, S)`` copy logits over the row's input tokens (masked)."""

        query = packing.pack(self.pointer_query(slots))
        keys = self.pointer_key(memory)
        scores = query @ keys.transpose(1, 2) / (self.hidden_size ** 0.5)
        scores = scores.float().masked_fill(memory_pad[:, None, :], float("-inf"))
        return packing.unpack(scores)

    # -- forward -----------------------------------------------------------------

    def _groups(self, shared, batch):
        group_row = batch.get("seq_edit_group_row")
        marker = batch.get("seq_edit_marker")
        if group_row is None or marker is None or group_row.numel() == 0:
            return None
        device = shared.words_embedding.device
        group_row = group_row.to(device)
        marker = marker.to(device)
        tokens, order, counts = marker_positions(shared, self.edit_token_index)
        if tokens is None:
            return None
        valid = marker < counts[group_row]
        # Same lookup as ``marker_stream(...)[0][group_row, marker.clamp(max=W - 1)]``
        # with ``W = max(counts.max(), 1)``, computed on the device: slots past a
        # row's markers read zeros.
        width = counts.max().clamp(min=1) if counts.numel() else counts.new_ones(())
        slot = torch.minimum(marker, width - 1)
        present = slot < counts[group_row]
        position = order[group_row, slot.clamp(max=order.shape[1] - 1)]
        query = tokens[group_row, position].masked_fill(~present[:, None], 0)
        return group_row, query, valid

    def _unused(self, reference):
        return sum(parameter.sum() for parameter in self.parameters()) * 0.0 + reference.sum() * 0.0

    def forward(self, shared, dependency_outputs, flat_inputs=None, **batch):
        words = shared.words_embedding
        supervised = batch.get("seq_edit_op") is not None
        groups = self._groups(shared, batch)
        if groups is None:
            return TaskHeadOutput(
                loss=self._unused(words) if (self.training and supervised) else None, extra={},
            )
        group_row, query, valid = groups
        units = words[group_row]
        unit_mask = shared.mask[group_row].bool() & valid[:, None]
        states, slot_mask = self._unit_states(units, unit_mask, query)
        op_logits = self._op_logits(states)
        if supervised and self.tag_mode:
            loss, stats = self._tag_losses(states, op_logits, slot_mask, batch)
            return TaskHeadOutput(loss=loss, logits=op_logits, extra={"loss_stats": stats})
        if supervised:
            loss, stats = self._losses(states, op_logits, slot_mask, group_row, shared, batch)
            return TaskHeadOutput(loss=loss, logits=op_logits, extra={"loss_stats": stats})
        decode = self._predict(states, op_logits, slot_mask, group_row, shared, batch)
        return TaskHeadOutput(logits=op_logits, extra={"decode": decode})

    # -- training ----------------------------------------------------------------

    def _op_targets(self, op_logits, slot_mask, batch):
        labels = batch["seq_edit_op"].to(op_logits.device)
        width = min(labels.shape[1], op_logits.shape[1])
        labels = labels[:, :width].masked_fill(~slot_mask[:, :width], IGNORE)
        weights = torch.ones(NUM_OPS, device=op_logits.device)
        weights[KEEP] = self.cfg.keep_weight
        op_loss = F.cross_entropy(
            op_logits[:, :width].reshape(-1, NUM_OPS), labels.reshape(-1),
            weight=weights, ignore_index=IGNORE, label_smoothing=self.cfg.op_label_smoothing,
        )
        if not torch.isfinite(op_loss):
            op_loss = op_logits.sum() * 0.0
        return labels, width, op_loss

    def _tag_logits(self, edit_states, edit_ops):
        return self.tag_head(edit_states + self.op_embed(edit_ops)).float()

    def _tag_losses(self, states, op_logits, slot_mask, batch):
        cfg = self.cfg
        device = op_logits.device
        labels, width, op_loss = self._op_targets(op_logits, slot_mask, batch)
        terms = {"op": op_loss}
        positions = batch.get("seq_edit_pos")
        tags = batch.get("seq_edit_tag")
        index = None
        if positions is not None and tags is not None and positions.numel():
            positions = positions.to(device)
            tags = tags.to(device)
            inside = positions[:, 1] < width
            keep = inside & (labels[positions[:, 0], positions[:, 1].clamp(max=width - 1)] != IGNORE)
            index = keep.nonzero(as_tuple=False).squeeze(-1)
        if index is None or index.numel() == 0:
            terms["content"] = self._unused(states)
        else:
            positions, tags = positions[index], tags[index]
            edit_states = states[positions[:, 0], positions[:, 1]]
            edit_ops = labels[positions[:, 0], positions[:, 1]]
            terms["tag"] = F.cross_entropy(self._tag_logits(edit_states, edit_ops), tags)
        loss = cfg.op_loss_coef * terms["op"] + cfg.tag_loss_coef * terms.get("tag", 0.0) + terms.get("content", 0.0)
        return loss, _loss_stats(terms)

    def _losses(self, states, op_logits, slot_mask, group_row, shared, batch):
        cfg = self.cfg
        device = op_logits.device
        labels, width, op_loss = self._op_targets(op_logits, slot_mask, batch)
        terms = {"op": op_loss}

        positions = batch.get("seq_edit_pos")
        if positions is not None and positions.numel():
            positions = positions.to(device)
            keep = (positions[:, 1] < width) & (labels[positions[:, 0], positions[:, 1].clamp(max=width - 1)] != IGNORE)
            index = keep.nonzero(as_tuple=False).squeeze(-1)
        else:
            index = None
        if index is None or index.numel() == 0:
            terms["content"] = self._unused(states)
        else:
            positions = positions[index]
            lengths = batch["seq_edit_length"].to(device)[index]
            tokens = batch["seq_edit_tokens"].to(device)[index]
            spaces = batch["seq_edit_space"].to(device)[index]
            edit_states = states[positions[:, 0], positions[:, 1]]
            edit_ops = labels[positions[:, 0], positions[:, 1]]
            edit_rows = group_row[positions[:, 0]]

            length_logits = self.length_head(edit_states).float()
            if (lengths != IGNORE).any():
                terms["length"] = F.cross_entropy(length_logits, lengths, ignore_index=IGNORE)
            space_logit = self.space_head(edit_states).float().squeeze(-1)
            has_space = spaces != IGNORE
            if has_space.any():
                terms["space"] = F.binary_cross_entropy_with_logits(
                    space_logit[has_space], spaces[has_space].float(),
                )

            local, local_pad = self._local_context(shared, batch, edit_rows, positions[:, 1])
            slots, packing, memory, memory_pad = self._decode_slots(
                edit_states, edit_ops, edit_rows, shared, local, local_pad,
            )
            gold = tokens != IGNORE
            target = tokens[gold]
            hidden = slots[gold]
            log_vocab = F.log_softmax(self._vocab_logits(hidden), dim=-1)
            log_prob = log_vocab.gather(1, target[:, None]).squeeze(1)
            if self.cfg.copy:
                pointer = self._pointer_logits(slots, packing, memory, memory_pad)[gold]
                log_pointer = F.log_softmax(pointer, dim=-1)
                memory_ids = shared.input_ids[packing.rows][packing.local]  # (E, S)
                slot_edit = torch.arange(slots.shape[0], device=device)[:, None].expand_as(gold)[gold]
                match = (memory_ids[slot_edit] == target[:, None]) & torch.isfinite(log_pointer)
                # A logsumexp over all -inf has a NaN gradient, so mask with a finite
                # floor and set tokens with no copy source to -inf afterwards.
                floor = torch.finfo(log_pointer.dtype).min
                log_copy = torch.logsumexp(log_pointer.masked_fill(~match, floor), dim=-1)
                log_copy = log_copy.masked_fill(~match.any(dim=-1), float("-inf"))
                gate = self.copy_gate(hidden).float().squeeze(-1)
                log_prob = torch.logaddexp(F.logsigmoid(gate) + log_prob, F.logsigmoid(-gate) + log_copy)
            terms["token"] = -log_prob.mean()

        loss = (
            cfg.op_loss_coef * terms["op"]
            + cfg.length_loss_coef * terms.get("length", 0.0)
            + cfg.token_loss_coef * terms.get("token", 0.0)
            + cfg.space_loss_coef * terms.get("space", 0.0)
            + terms.get("content", 0.0)
        )
        return loss, _loss_stats(terms)

    # -- inference ---------------------------------------------------------------

    def _predict(self, states, op_logits, slot_mask, group_row, shared, batch):
        cfg = self.cfg
        keep_bias = batch.get("seq_edit_keep_bias")
        keep_bias = cfg.keep_bias if keep_bias is None else float(keep_bias)
        min_edit_prob = batch.get("seq_edit_min_edit_prob")
        min_edit_prob = cfg.min_edit_prob if min_edit_prob is None else float(min_edit_prob)
        logits = op_logits.clone()
        logits[..., KEEP] += keep_bias
        probs = logits.softmax(-1)
        ops = probs.argmax(-1).masked_fill(~slot_mask, KEEP)
        edit_prob = (1.0 - probs[..., KEEP]).masked_fill(~slot_mask, 0.0)
        if min_edit_prob > 0:
            weak = edit_prob.amax(dim=1) < min_edit_prob
            ops[weak] = KEEP
        decode = {
            "group_row": group_row,
            "ops": ops,
            "op_probs": probs,
            "slot_mask": slot_mask,
        }
        positions = ((ops == REPLACE) | (ops == APPEND)).nonzero(as_tuple=False)
        if positions.numel() == 0:
            decode["edit_pos"] = positions
            return decode
        edit_states = states[positions[:, 0], positions[:, 1]]
        edit_ops = ops[positions[:, 0], positions[:, 1]]
        edit_rows = group_row[positions[:, 0]]
        if self.tag_mode:
            tag_probs = self._tag_logits(edit_states, edit_ops).softmax(-1)
            best, tag = tag_probs.max(-1)
            decode.update(edit_pos=positions, edit_tag=tag, edit_tag_probs=best)
            return decode
        lengths = self.length_head(edit_states).float().argmax(-1) + 1
        space = torch.sigmoid(self.space_head(edit_states).float().squeeze(-1))
        local, local_pad = self._local_context(shared, batch, edit_rows, positions[:, 1])
        slots, packing, memory, memory_pad = self._decode_slots(
            edit_states, edit_ops, edit_rows, shared, local, local_pad,
        )
        tokens, token_probs = self._argmax_tokens(slots, packing, memory, memory_pad, shared, batch)
        decode.update(
            edit_pos=positions,
            edit_length=lengths,
            edit_tokens=tokens,
            edit_token_probs=token_probs,
            edit_space=space,
        )
        return decode

    def _argmax_tokens(self, slots, packing, memory, memory_pad, shared, batch):
        num_edits, num_slots, dim = slots.shape
        hidden = slots.reshape(-1, dim)
        pointer = None
        if self.cfg.copy:
            pointer = self._pointer_logits(slots, packing, memory, memory_pad).softmax(-1).reshape(num_edits * num_slots, -1)
            memory_ids = shared.input_ids[packing.rows][packing.local]
            memory_ids = memory_ids[:, None, :].expand(-1, num_slots, -1).reshape(num_edits * num_slots, -1)
        blocked = batch.get("seq_edit_blocked_tokens")
        tokens, probs = [], []
        for start in range(0, hidden.shape[0], _VOCAB_CHUNK):
            stop = start + _VOCAB_CHUNK
            dist = self._vocab_logits(hidden[start:stop]).softmax(-1)
            if pointer is not None:
                gate = torch.sigmoid(self.copy_gate(hidden[start:stop]).float())
                dist = gate * dist
                dist.scatter_add_(1, memory_ids[start:stop], (1.0 - gate) * pointer[start:stop])
            if blocked is not None and blocked.numel():
                dist[:, blocked.to(dist.device).clamp(max=dist.shape[1] - 1)] = 0.0
            best, index = dist.max(-1)
            tokens.append(index)
            probs.append(best)
        return (
            torch.cat(tokens).view(num_edits, num_slots),
            torch.cat(probs).view(num_edits, num_slots),
        )


__all__ = ["SequenceEditingHead", "SlotDecoderLayer", "input_embedding_module"]
