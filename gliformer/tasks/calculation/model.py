"""Calculation head: programs over text spans for ``$calculate`` fields.

For every (structuring record anchor, calc field) pair -- a *query* -- the
head:

1. builds a query vector from the field's ``[CALC]`` marker, the record anchor
   and the schema parent vector with the configured anchor modeling
   (``query_modeling``; the default ``readout`` also reads the text);
2. extracts operand spans with it (BIO, the shared
   :class:`~gliformer.layers.AnchoredSpanScorer`);
3. fills ``num_slots`` calculation slots: fixed slot embeddings conditioned on
   the query, refined over [query, operand spans, text] (the shared anchor
   refinement layer);
4. per slot predicts an operation, whether it is active, and two operand sets,
   each a sigmoid over the pool (operand spans | earlier slots | constants).

Slot ``i`` may only reference slots ``j < i``, so the program is acyclic. The
active slots form a prefix and the last one is the result. The decoder runs
the program on the numbers the normalization head reads for the operand spans.

Training queries come from the labels (gold record -> matched slot through the
structuring head's ``anchor_matches``). At inference every active record slot
(objectness above the threshold, or the most confident slot of a group) gets a
query per calc field.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from gliner.modeling.utils import extract_spans_from_tokens
from torch import nn

from ...layers import AnchoredSpanScorer, AnchorModeling
from ...layers.anchor_normalization import AnchorNormalizer
from ...layers.anchor_modeling import rename_state_dict_keys
from ...layers.mlp import create_mlp
from ...layers.span_pooling import SpanTokenSource, build_span_pooler
from .. import TaskHead, TaskHeadOutput, text_token_states
from ..losses import binary_focal_or_bce
from .program import CONSTANTS, NUM_PORTS, OP_NAMES


def marker_stream(shared, token_index: int):
    """Every marker's encoder state per row, in prompt order: ``(B, M, D)``, ``(B,)``."""

    tokens, input_ids = shared.token_embeds, shared.input_ids
    if tokens is None or input_ids is None or token_index < 0:
        return None, None
    attention = (
        shared.attention_mask.bool()
        if shared.attention_mask is not None
        else torch.ones_like(input_ids, dtype=torch.bool)
    )
    markers = (input_ids == token_index) & attention
    counts = markers.sum(dim=1)
    width = max(int(counts.max().item()) if counts.numel() else 0, 1)
    stream = tokens.new_zeros(tokens.shape[0], width, tokens.shape[-1])
    for row in range(tokens.shape[0]):
        positions = torch.nonzero(markers[row], as_tuple=False).squeeze(-1)
        if positions.numel():
            stream[row, : positions.numel()] = tokens[row, positions]
    return stream, counts


def marker_positions(shared, token_index: int):
    """Like :func:`marker_stream` but without a host sync or a per-row loop.

    Returns ``(tokens, order, counts)``: ``order[b, k]`` is the token position
    of row ``b``'s ``k``-th marker (valid for ``k < counts[b]``), so
    ``tokens[b, order[b, k]]`` equals ``marker_stream(...)[0][b, k]``.
    """

    tokens, input_ids = shared.token_embeds, shared.input_ids
    if tokens is None or input_ids is None or token_index < 0:
        return None, None, None
    attention = (
        shared.attention_mask.bool()
        if shared.attention_mask is not None
        else torch.ones_like(input_ids, dtype=torch.bool)
    )
    markers = (input_ids == token_index) & attention
    counts = markers.sum(dim=1)
    # Stable sort puts the markers first, in position order.
    order = torch.argsort(markers.to(torch.int8), dim=1, descending=True, stable=True)
    return tokens, order, counts


def span_fits(spans, word_mask, max_width: int):
    """``(Q, N)``: span no wider than ``max_width`` and all its units unmasked.

    Vectorized ``end - start + 1 <= max_width and word_mask[q, start:end + 1].all()``
    for in-range spans (``0 <= start <= end < L``); other spans get an
    arbitrary value.
    """

    length = word_mask.shape[1]
    holes = F.pad((~word_mask.bool()).long().cumsum(dim=1), (1, 0))  # (Q, L + 1)
    start, end = spans[..., 0], spans[..., 1]
    inside = holes.gather(1, (end + 1).clamp(0, length)) - holes.gather(1, start.clamp(0, length))
    return (end - start + 1 <= max_width) & (inside == 0)


class CalculationHead(TaskHead):
    """Predicts a calculation program per (record, ``$calculate`` field)."""

    name = "calculation"
    dependencies = ["structuring"]

    def __init__(self, config, hidden_size: int, dropout: float):
        super().__init__()
        cfg = config.calculation_config
        self.config = config
        self.cfg = cfg
        self.loss_coef = float(cfg.loss_coef)
        self.hidden_size = hidden_size
        self.num_slots = cfg.num_slots
        self.num_ops = len(OP_NAMES)
        self.num_consts = len(CONSTANTS)

        self.parent_proj = nn.Linear(hidden_size, hidden_size)
        self.query_modeling = AnchorModeling.from_config(
            cfg.query_modeling, hidden_size, dropout=dropout, num_heads=cfg.readout_heads,
        )
        # Checkpoints before anchor modeling stored the readout as ``query_readout``.
        self._register_load_state_dict_pre_hook(self._migrate_state_dict)
        self.scorer = AnchoredSpanScorer(hidden_size, dropout=dropout)
        self.span_pooler = build_span_pooler(cfg, hidden_size, dropout, max_width=cfg.max_span_width)

        self.slot_layer = TaskHead._build_anchor_layer(
            SimpleNamespace(
                anchor_mode="fixed",
                num_fixed_slots=cfg.num_slots,
                anchor_context_gate_init=1.0,
                anchor_context_gate_trainable=True,
            ),
            hidden_size,
            dropout,
        )
        self.slot_refine = None
        if cfg.slot_refine_layers > 0:
            self.slot_refine = TaskHead._build_anchor_refinement(
                SimpleNamespace(
                    anchor_refine_heads=cfg.slot_heads,
                    anchor_refine_layers=cfg.slot_refine_layers,
                    anchor_refine_norm="pre_norm",
                    anchor_refine_layer_scale_init=None,
                ),
                hidden_size,
                dropout,
                {},
            )
        # Slot identity at the scale of the (LayerNorm'd) query: the fixed
        # slot table rows have norm 1, so on their own every slot would look
        # like the query and slots could not learn their step position.
        self.slot_position = nn.Embedding(cfg.num_slots, hidden_size)
        nn.init.normal_(self.slot_position.weight, std=1.0)
        self.slot_norm = AnchorNormalizer.from_config(cfg.slot_normalization, hidden_size)
        # Memory segments: 0 query, 1 operand span, 2 text unit.
        self.memory_type = nn.Embedding(3, hidden_size)
        self.port_role = nn.Embedding(NUM_PORTS, hidden_size)
        self.port_query = create_mlp(
            input_dim=hidden_size, intermediate_dims=[hidden_size], output_dim=hidden_size,
            dropout=dropout, activation="gelu",
        )
        self.span_key = nn.Linear(hidden_size, hidden_size)
        self.slot_key = nn.Linear(hidden_size, hidden_size)
        self.const_embedding = nn.Embedding(self.num_consts, hidden_size)
        self.const_key = nn.Linear(hidden_size, hidden_size)
        # Per pool segment (spans, slots, constants): sets start mostly empty.
        self.port_bias = nn.Parameter(torch.full((3,), -2.0))
        self.op_head = create_mlp(
            input_dim=hidden_size, intermediate_dims=[hidden_size], output_dim=self.num_ops,
            dropout=dropout, activation="gelu",
        )
        self.active_head = create_mlp(
            input_dim=hidden_size, intermediate_dims=[hidden_size], output_dim=1,
            dropout=dropout, activation="gelu",
        )
        # (length, device) -> flattened (start, end, width) of every span
        # ``_ranked_spans`` scores, offset-major.
        self._span_grids = {}

    @staticmethod
    def _migrate_state_dict(state_dict, prefix, *args, **kwargs):
        rename_state_dict_keys(state_dict, prefix, {"query_readout.": "query_modeling."})

    @classmethod
    def from_config(cls, config, **kwargs):
        if getattr(config, "calculation_config", None) is None:
            return None
        if getattr(config, "structuring_config", None) is None:
            raise ValueError("The calculation head needs a structuring head")
        return cls(config, hidden_size=config.hidden_size, dropout=config.dropout)

    # -- queries -------------------------------------------------------------

    @property
    def calc_token_index(self) -> int:
        return int(getattr(self.config.calculation_config, "calc_token_index", -1))

    def _objectness_threshold(self, batch) -> float:
        value = batch.get("objectness_threshold")
        if value is None:
            struct_cfg = self.config.structuring_config
            value = getattr(struct_cfg, "anchor_objectness_threshold", None)
        if value is None:
            value = batch.get("threshold")
        return 0.5 if value is None or isinstance(value, (list, tuple)) else float(value)

    def _training_queries(self, batch, anchor_mask, matches):
        """``(group, anchor, marker slot, label row)`` of supervised queries."""

        listed = batch.get("calculation_query_list")
        if listed is not None:
            # Python list from the collator: no device round trip.
            triples = [(int(g), int(r), int(m)) for g, r, m in listed]
        else:
            triples = list(zip(
                batch["calculation_query_group"].tolist(),
                batch["calculation_query_record"].tolist(),
                batch["calculation_query_marker"].tolist(),
            ))
        host_anchor_mask = None
        queries = []
        for row, (group, record, marker) in enumerate(triples):
            if group >= anchor_mask.shape[0]:
                continue
            anchor = None
            if matches is not None and group < len(matches):
                for prediction, gold in matches[group]:
                    if int(gold) == record:
                        anchor = int(prediction)
                        break
            elif record < anchor_mask.shape[1]:
                if host_anchor_mask is None:
                    host_anchor_mask = anchor_mask.bool().tolist()
                if host_anchor_mask[group][record]:
                    anchor = record
            if anchor is not None:
                queries.append((group, anchor, marker, row))
        return queries

    @staticmethod
    def _marker_ordinals(batch, marker_index, marker_mask):
        """Per group, the ``[CALC]`` ordinal of every calc field slot (``None`` = no marker)."""

        listed = batch.get("calculation_marker_list")
        if listed is not None and len(listed) == marker_index.shape[0]:
            return [list(row) for row in listed]
        return [
            [ordinal if keep else None for ordinal, keep in zip(index_row, mask_row)]
            for index_row, mask_row in zip(marker_index.tolist(), marker_mask.bool().tolist())
        ]

    @staticmethod
    def _keep_marked(queries, ordinals, counts, origin):
        """Drop queries whose marker was cut off by truncation; ``(query, ordinal)``."""

        kept = []
        for query in queries:
            group, _anchor, slot, _row = query
            if group >= len(ordinals) or slot >= len(ordinals[group]):
                continue
            ordinal = ordinals[group][slot]
            if ordinal is not None and ordinal < counts[origin[group]]:
                kept.append((query, ordinal))
        return kept

    def _inference_queries(self, batch, anchor_mask, objectness, marker_mask):
        threshold = self._objectness_threshold(batch)
        queries = []
        for group in range(min(anchor_mask.shape[0], marker_mask.shape[0])):
            valid = anchor_mask[group].bool()
            if not valid.any() or not marker_mask[group].any():
                continue
            if objectness is not None:
                scores = torch.sigmoid(objectness[group].float()).masked_fill(~valid, -1.0)
                active = (scores > threshold).nonzero(as_tuple=False).flatten().tolist()
                if not active:
                    active = [int(scores.argmax())]
            else:
                active = valid.nonzero(as_tuple=False).flatten().tolist()
            for anchor in active:
                for marker in marker_mask[group].nonzero(as_tuple=False).flatten().tolist():
                    queries.append((group, anchor, marker, -1))
        return queries

    # -- candidates ----------------------------------------------------------

    def _candidates(self, bio_logits, word_mask, gold_idx=None, gold_mask=None, gold_spans=None):
        """Operand candidates per query: gold spans first, then predicted ones.

        ``gold_spans`` (per query, a list of ``(start, end)``) or the padded
        ``gold_idx`` / ``gold_mask`` give the gold spans. Predicted spans and
        their validity are computed on the device and read back with a
        single copy; the greedy selection then runs on the host.
        """

        cfg = self.cfg
        topk = cfg.candidate_mode == "topk"
        word_mask = word_mask.bool()
        width = word_mask.shape[1]
        if gold_spans is None and gold_idx is not None:
            gold_spans = [
                [(start, end) for (start, end), keep in zip(idx_row, mask_row) if keep]
                for idx_row, mask_row in zip(gold_idx.tolist(), gold_mask.bool().tolist())
            ]
        if topk:
            predicted, predicted_mask = self._ranked_spans(bio_logits, word_mask)
            fits = span_fits(predicted, word_mask, cfg.max_span_width)
            host = torch.cat(
                [predicted, predicted_mask.unsqueeze(-1).long(), fits.unsqueeze(-1).long()], dim=-1,
            ).cpu()
            predicted, keeps, fits = host[..., :2].tolist(), host[..., 2].tolist(), host[..., 3].tolist()
        else:
            cells = torch.sigmoid(bio_logits.detach().float()) > cfg.operand_threshold  # (Q, L, 3)
            host = torch.cat([cells, word_mask.unsqueeze(-1)], dim=-1).cpu()
            cells = host[..., :3].float().unsqueeze(2)
            predicted, predicted_mask = extract_spans_from_tokens(cells, labels=cells)
            fits = span_fits(predicted, host[..., 3], cfg.max_span_width)
            predicted, keeps, fits = predicted.tolist(), predicted_mask.tolist(), fits.tolist()
        rows = []
        for query in range(bio_logits.shape[0]):
            spans = [tuple(span) for span in gold_spans[query]] if gold_spans is not None else []
            seen = set(spans)
            covered = set()
            if topk:
                for start, end in spans:
                    covered.update(range(start, end + 1))
            for (start, end), keep, fit in zip(predicted[query], keeps[query], fits[query]):
                if len(spans) >= cfg.max_operand_spans:
                    break
                if not keep or (start, end) in seen or start < 0 or end < start or end >= width:
                    continue
                if topk:
                    # Ranked spans: skip those overlapping a better one.
                    if any(unit in covered for unit in range(start, end + 1)):
                        continue
                    covered.update(range(start, end + 1))
                if not fit:
                    continue
                spans.append((start, end))
                seen.add((start, end))
            rows.append(spans)
        count = max(max((len(spans) for spans in rows), default=0), 1)
        padded = [
            [[start, end, 1] for start, end in spans] + [[0, 0, 0]] * (count - len(spans))
            for spans in rows
        ]
        table = torch.tensor(padded, dtype=torch.long).reshape(len(rows), count, 3)
        table = table.to(bio_logits.device, non_blocking=True)
        return table[..., :2].contiguous(), table[..., 2].bool()

    def _span_grid(self, length: int, device):
        """Cached offset-major ``(starts, ends, width scale)`` of ``_ranked_spans``."""

        key = (length, str(device))
        grid = self._span_grids.get(key)
        if grid is None:
            offsets = range(min(self.cfg.max_span_width, length))
            starts = torch.cat([torch.arange(length - offset) for offset in offsets])
            widths = torch.cat([torch.full((length - offset,), offset + 1) for offset in offsets])
            if torch.device(device).type == "cuda":
                # CUDA divides by a Python scalar as a multiply by its float32
                # reciprocal; reproduce that exactly.
                scale = 1.0 / widths.float()
            else:
                scale = widths.float()
            grid = (starts.to(device), (starts + widths - 1).to(device), scale.to(device))
            self._span_grids[key] = grid
        return grid

    def _ranked_spans(self, bio_logits, word_mask):
        """Spans ranked by the geometric mean of start, end and inside probabilities.

        Returns ``(Q, N, 2)`` spans sorted best first and their mask; at most
        ``4 * max_operand_spans`` per query, all above ``candidate_threshold``.
        All widths are scored at once over a cached span grid.
        """

        cfg = self.cfg
        logp = F.logsigmoid(bio_logits.detach().float())  # (Q, L, 3)
        count, length = logp.shape[:2]
        valid = word_mask.bool()
        inside = torch.nn.functional.pad(logp[..., 2].masked_fill(~valid, -1e4).cumsum(dim=1), (1, 0))
        starts, ends, scale = self._span_grid(length, logp.device)
        inside_sum = inside[:, ends + 1] - inside[:, starts]
        if logp.device.type == "cuda":
            mean_inside = inside_sum * scale
        else:
            mean_inside = inside_sum / scale
        scores = (logp[:, starts, 0] + logp[:, ends, 1] + mean_inside) / 3.0
        ok = valid[:, starts] & valid[:, ends]
        scores = scores.masked_fill(~ok, float("-inf"))
        keep = min(scores.shape[1], 4 * cfg.max_operand_spans)
        best, index = scores.topk(keep, dim=1)
        span_idx = torch.stack([starts[index], ends[index]], dim=-1)
        span_mask = best > math.log(cfg.candidate_threshold)
        return span_idx, span_mask

    def _token_source(self, shared, rows, batch):
        if not self.span_pooler.uses_tokens:
            return None
        tokens = text_token_states(shared) if getattr(shared, "token_embeds", None) is not None else None
        token_words = batch.get("span_token_words")
        if tokens is None or token_words is None or token_words.shape != tokens.shape[:2]:
            return None
        return SpanTokenSource(tokens, token_words.to(tokens.device), rows=rows)

    # -- losses --------------------------------------------------------------

    def _unused(self, reference):
        """A zero loss tied to the graph.

        Under DDP every parameter must receive a gradient, so it also touches
        all of them (one reduction kernel each); otherwise that is skipped.
        """

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return sum(parameter.sum() for parameter in self.parameters()) * 0.0 + reference.sum() * 0.0
        return reference.sum() * 0.0

    def _losses(self, outputs, batch, label_rows, word_mask, extra):
        cfg = self.cfg
        device = outputs["op_logits"].device
        if not torch.is_tensor(label_rows):
            label_rows = torch.tensor(label_rows, dtype=torch.long, device=device)

        def take(name):
            return batch[name].to(device)[label_rows]

        terms = {}
        # Operand spans (BIO): focal over text cells.
        bio_labels = take("calculation_bio_labels")
        bio_logits = outputs["bio_logits"].float()
        length = min(bio_logits.shape[1], bio_labels.shape[1])
        bio_cells = binary_focal_or_bce(
            bio_logits[:, :length],
            bio_labels[:, :length].to(bio_logits.dtype),
            focal_loss_alpha=-1.0 if cfg.bio_focal_loss_alpha is None else cfg.bio_focal_loss_alpha,
            focal_loss_gamma=0.0 if cfg.bio_focal_loss_gamma is None else cfg.bio_focal_loss_gamma,
        )
        bio_mask = word_mask[:, :length].unsqueeze(-1).to(bio_cells.dtype).expand_as(bio_cells)
        terms["bio"] = (bio_cells * bio_mask).sum() / bio_mask.sum().clamp(min=1.0)

        active = take("calculation_active").float()
        terms["active"] = F.binary_cross_entropy_with_logits(outputs["active_logits"].float(), active)

        # Mean over labelled steps without a host check for "any labelled":
        # with none the term is 0 (``mean`` would give NaN).
        op = take("calculation_op").reshape(-1)
        terms["op"] = F.cross_entropy(
            outputs["op_logits"].float().reshape(-1, self.num_ops), op, ignore_index=-100, reduction="sum",
        ) / (op != -100).sum().clamp(min=1)

        # Operand sets: BCE per pool element on program steps; normalized per
        # (step, port) so a port's few positives are not drowned out.
        gold_spans = take("calculation_port_spans").float()
        span_logits = outputs["port_span_logits"].float()
        target_spans = torch.zeros_like(span_logits)
        width = min(gold_spans.shape[-1], target_spans.shape[-1])
        target_spans[..., :width] = gold_spans[..., :width]
        ref_logits = outputs["port_ref_logits"].float()
        const_logits = outputs["port_const_logits"].float()
        port_alpha = -1.0 if cfg.port_focal_loss_alpha is None else cfg.port_focal_loss_alpha
        port_gamma = 0.0 if cfg.port_focal_loss_gamma is None else cfg.port_focal_loss_gamma

        def cells(logits, targets):
            return binary_focal_or_bce(
                logits, targets, focal_loss_alpha=port_alpha, focal_loss_gamma=port_gamma,
            )

        step_mask = active.unsqueeze(-1).unsqueeze(-1)  # (Q, K, 1, 1)
        span_valid = outputs["operand_mask"].float()[:, None, None, :]
        ref_valid = outputs["ref_mask"].float()[None, :, None, :]
        total = (
            (cells(span_logits, target_spans) * span_valid * step_mask).sum()
            + (cells(ref_logits, take("calculation_port_refs").float()) * ref_valid * step_mask).sum()
            + (cells(const_logits, take("calculation_port_consts").float()) * step_mask).sum()
        )
        rows = active.sum() * NUM_PORTS
        terms["port"] = total / rows.clamp(min=1.0)

        coefs = {
            "bio": cfg.bio_loss_coef, "active": cfg.active_loss_coef,
            "op": cfg.op_loss_coef, "port": cfg.port_loss_coef,
        }
        loss = None
        for name, value in terms.items():
            extra[f"{name}_loss"] = value.detach()
            weighted = coefs[name] * value
            loss = weighted if loss is None else loss + weighted
        return loss

    # -- forward -------------------------------------------------------------

    def forward(self, shared, dependency_outputs, flat_inputs=None, **batch):
        words = shared.words_embedding
        word_mask_all = shared.mask
        supervised = batch.get("calculation_query_group") is not None

        def empty():
            return TaskHeadOutput(
                loss=self._unused(words) if (self.training and supervised) else None,
                extra={},
            )

        struct = (dependency_outputs or {}).get("structuring")
        marker_index = batch.get("calculation_marker_index")
        marker_mask = batch.get("calculation_marker_mask")
        if struct is None or struct.extra is None or marker_index is None:
            return empty()
        extra_in = struct.extra
        anchors = extra_in.get("groups_output")
        anchor_mask = extra_in.get("anchor_mask")
        origin = extra_in.get("batch_origin")
        parent = extra_in.get("parent_embedding")
        if anchors is None or anchor_mask is None or origin is None or parent is None:
            return empty()
        device = words.device
        field_idx = batch.get("calculation_field_idx")

        if supervised:
            queries = self._training_queries(batch, anchor_mask, extra_in.get("anchor_matches"))
        else:
            queries = self._inference_queries(
                batch, anchor_mask, extra_in.get("objectness_logits"), marker_mask.to(device).bool(),
            )
        tokens, marker_order, counts = marker_positions(shared, self.calc_token_index)
        if tokens is None or not queries:
            return empty()
        # A marker cut off by truncation has no state. One host copy brings
        # the per-row marker counts and the group -> row map.
        batch_size = counts.shape[0]
        host = torch.cat([counts.reshape(-1), origin.reshape(-1).to(counts.device, torch.long)]).tolist()
        kept = self._keep_marked(
            queries, self._marker_ordinals(batch, marker_index, marker_mask),
            host[:batch_size], host[batch_size:],
        )
        if not kept:
            return empty()

        # (group, anchor, marker slot, marker ordinal, label row) per query, one copy.
        index = torch.tensor(
            [[q[0], q[1], q[2], ordinal, max(q[3], 0)] for q, ordinal in kept], dtype=torch.long,
        ).to(device, non_blocking=True)
        group, anchor, marker, ordinal, label_index = index.unbind(dim=1)
        rows = origin.to(device)[group]

        anchor_vectors = anchors[group, anchor]
        if self.cfg.detach_record_anchors:
            anchor_vectors = anchor_vectors.detach()
        calc_vectors = tokens[rows, marker_order[rows, ordinal]]
        record = anchor_vectors + self.parent_proj(parent[group])
        query_words = words[rows]
        query_word_mask = word_mask_all[rows].bool()
        query = self.query_modeling.fuse(
            record.unsqueeze(1), calc_vectors.unsqueeze(1),
            memory=query_words, memory_mask=query_word_mask,
        )[:, 0, 0]

        bio_logits = self.scorer(query.unsqueeze(1), query_words, word_mask=query_word_mask)[:, 0]
        if supervised:
            span_lists = batch.get("calculation_span_list")
            if span_lists is not None:
                operand_spans, operand_mask = self._candidates(
                    bio_logits, query_word_mask, gold_spans=[span_lists[q[3]] for q, _ in kept],
                )
            else:
                operand_spans, operand_mask = self._candidates(
                    bio_logits, query_word_mask,
                    batch["calculation_span_idx"].to(device)[label_index],
                    batch["calculation_span_mask"].to(device)[label_index].bool(),
                )
        else:
            operand_spans, operand_mask = self._candidates(bio_logits, query_word_mask)
        safe_spans = operand_spans * operand_mask.unsqueeze(-1).long()
        span_vectors = self._pool_spans(
            self.span_pooler, query_words, safe_spans, self._token_source(shared, rows, batch),
        )
        span_vectors = span_vectors * operand_mask.unsqueeze(-1).to(span_vectors.dtype)

        slots, slot_mask = self.slot_layer(query)
        slots = slots + self.slot_position.weight.unsqueeze(0)
        slots = self.slot_norm(slots, slot_mask)
        memory = torch.cat([
            (query + self.memory_type.weight[0]).unsqueeze(1),
            span_vectors + self.memory_type.weight[1],
            query_words + self.memory_type.weight[2],
        ], dim=1)
        memory_mask = torch.cat([
            torch.ones_like(operand_mask[:, :1]), operand_mask, query_word_mask,
        ], dim=1)
        if self.slot_refine is not None:
            slots = self.slot_refine(slots, memory, token_mask=memory_mask, query_mask=slot_mask)

        scale = 1.0 / math.sqrt(self.hidden_size)
        port_queries = self.port_query(slots.unsqueeze(2) + self.port_role.weight[None, None])
        span_logits = torch.einsum("qkpd,qsd->qkps", port_queries, self.span_key(span_vectors)) * scale
        span_logits = span_logits + self.port_bias[0]
        span_logits = span_logits.masked_fill(~operand_mask[:, None, None, :], -1e4)
        ref_logits = torch.einsum("qkpd,qjd->qkpj", port_queries, self.slot_key(slots)) * scale
        ref_mask = torch.tril(
            torch.ones(self.num_slots, self.num_slots, dtype=torch.bool, device=device), diagonal=-1,
        )
        ref_logits = (ref_logits + self.port_bias[1]).masked_fill(~ref_mask[None, :, None, :], -1e4)
        const_keys = self.const_key(self.const_embedding.weight)
        const_logits = torch.einsum("qkpd,cd->qkpc", port_queries, const_keys) * scale + self.port_bias[2]
        op_logits = self.op_head(slots)
        active_logits = self.active_head(slots).squeeze(-1)

        outputs = {
            "bio_logits": bio_logits,
            "operand_mask": operand_mask,
            "ref_mask": ref_mask,
            "port_span_logits": span_logits,
            "port_ref_logits": ref_logits,
            "port_const_logits": const_logits,
            "op_logits": op_logits,
            "active_logits": active_logits,
        }
        extra = {
            # Normalization reads these spans (rows = text rows).
            "operand_spans": operand_spans,
            "operand_mask": operand_mask,
            "batch_origin": rows,
        }
        loss = None
        if supervised:
            loss = self._losses(outputs, batch, label_index, query_word_mask, extra)
            if self.training:
                loss = loss + self._unused(query)
        if not self.training:
            extra["decode"] = {
                "query_group": group,
                "query_anchor": anchor,
                "query_marker": marker,
                "query_row": rows,
                "query_field": (
                    field_idx.to(device)[group, marker] if field_idx is not None else marker
                ),
                "operand_spans": operand_spans,
                "operand_mask": operand_mask,
                "operand_probs": None,
                "op_probs": torch.softmax(op_logits.float(), dim=-1),
                "active_probs": torch.sigmoid(active_logits.float()),
                "port_span_probs": torch.sigmoid(span_logits.float()),
                "port_ref_probs": torch.sigmoid(ref_logits.float()),
                "port_const_probs": torch.sigmoid(const_logits.float()),
            }
        return TaskHeadOutput(loss=loss, extra=extra)


__all__ = ["CalculationHead", "marker_positions", "marker_stream", "span_fits"]
