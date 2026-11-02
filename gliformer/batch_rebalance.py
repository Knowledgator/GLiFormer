"""Length-balanced micro-batches: ``train_sampling_strategy: batch_rebalance``.

The same scheme as GLiClass's ``train.py`` (``--train_sampling_strategy
batch_rebalance``): each optimizer step's samples are sorted by estimated
length and split into variable-size micro-batches of balanced padded cost, so
short rows travel together in large micro-batches and long rows in small ones.

The sampler is ``transformers.trainer_pt_utils.BatchRebalanceSampler``
(transformers >= 5.17). Older releases get the copy below, taken unchanged from
transformers v5.17.0 (Copyright 2020-present the HuggingFace Inc. team,
Apache License 2.0).

Lengths are estimated, not measured: encoding a structuring row through the
collator takes tens of milliseconds, too slow for millions of rows. A small
sample of each row kind (its set of task fields) is encoded once, and a linear
fit on character counts maps every row to an encoded length.
"""

import json
import math
import multiprocessing
import random
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Sampler

try:  # transformers >= 5.17
    from transformers.trainer_pt_utils import BatchRebalanceSampler
except ImportError:  # pragma: no cover - exercised on transformers < 5.17

    class BatchRebalanceSampler(Sampler):
        r"""
        Sampler that balances sequence length cost across DP ranks while minimizing padding waste within each
        micro-batch. Designed for distributed training with variable-length sequences (e.g., LLM fine-tuning).

        Uses a cost-aware rebalance algorithm ("Poor Man's Batch Rebalance") that:
        1. For each optimizer step, collects `effective_batch_size` samples (shuffled, same set on all ranks
           via shared seed — no cross-rank communication needed).
        2. Sorts samples by length descending.
        3. Partitions into `dp_size * grad_accum` groups using an iterative cost-balancing algorithm that
           adjusts group sizes (not sample assignments) until costs converge.
        4. Sorts groups by cost descending and assigns the top `dp_size` groups to the first micro-batch
           (slot 0), the next `dp_size` to slot 1, etc. This ensures each micro-batch slot has balanced
           cost across ranks.
        5. Each rank yields its `grad_accum` micro-batches (one from each slot).

        Cost model (fixed; captures padding waste plus attention's quadratic cost):

            cost(bs, max_len) = bs * max_len + QUADRATIC_COST_COEF * bs * max_len^2

        where `bs` is the number of samples in the micro-batch and `max_len` is the length of the longest
        sample (all samples are padded to this length). The linear term captures linear-layer compute and
        padding waste; the quadratic term captures attention's O(L^2) cost.

        Key properties:
            - **Variable batch size**: long-sequence groups get fewer samples, short-sequence groups get
              more, keeping per-group cost balanced.
            - **Fixed effective batch size**: total samples per optimizer step is always
              `effective_batch_size`.
            - **Zero communication**: all ranks run the same deterministic algorithm and yield only their
              portion — no all-gather or broadcast needed.
            - **Padding-aware**: the cost model explicitly accounts for padding waste, unlike
              `LengthGroupedSampler` which only sorts by length.

        Args:
            lengths (`list[int]`):
                Token lengths of all samples in the dataset.
            effective_batch_size (`int`):
                Total samples per optimizer step (`dp_size * grad_accum * per-group batch size`).
            dp_size (`int`):
                Data parallel size (number of ranks).
            grad_accum (`int`):
                Gradient accumulation steps.
            shuffle (`bool`, *optional*, defaults to `True`):
                Whether to shuffle samples each epoch.
            seed (`int`, *optional*, defaults to 42):
                Random seed for shuffling.
            rank (`int`, *optional*, defaults to 0):
                Current rank index.
            drop_last (`bool`, *optional*, defaults to `True`):
                Drop incomplete last batch.

        Example:

        ```python
        >>> from transformers.trainer_pt_utils import BatchRebalanceSampler
        >>> sampler = BatchRebalanceSampler(
        ...     lengths=[100, 200, 300, ...],
        ...     effective_batch_size=16,
        ...     dp_size=4,
        ...     grad_accum=2,
        ...     rank=0,
        ... )
        >>> for batch_indices in sampler:
        ...     # batch_indices is a list of sample indices for this rank's micro-batch
        ...     pass
        ```
        """

        # Weight of the attention's O(L^2) term in the cost model.
        QUADRATIC_COST_COEF = 0.001

        def __init__(
            self,
            lengths: list[int],
            effective_batch_size: int,
            dp_size: int,
            grad_accum: int,
            shuffle: bool = True,
            seed: int = 42,
            rank: int = 0,
            drop_last: bool = True,
        ):
            assert dp_size >= 1
            assert grad_accum >= 1
            assert effective_batch_size >= dp_size * grad_accum

            self.lengths = list(lengths)
            self.effective_batch_size = effective_batch_size
            self.dp_size = dp_size
            self.grad_accum = grad_accum
            self.shuffle = shuffle
            self.seed = seed
            self.rank = rank
            self.drop_last = drop_last
            self.epoch = 0

            self.num_full_batches = len(self.lengths) // effective_batch_size
            self.remainder = len(self.lengths) % effective_batch_size
            # When drop_last=False, the trailing partial batch is also yielded (padded up to the
            # effective batch size when too small to fill every (rank, slot) group), mirroring how
            # PyTorch's DistributedSampler pads the tail. Otherwise trailing samples are silently
            # dropped even though the user asked to keep them.
            self.has_tail = (not drop_last) and self.remainder > 0
            self.num_global_batches = self.num_full_batches + (1 if self.has_tail else 0)

        def set_epoch(self, epoch: int):
            self.epoch = epoch

        def __iter__(self):
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            n = len(self.lengths)

            if self.shuffle:
                order = torch.randperm(n, generator=g).tolist()
            else:
                order = list(range(n))

            for gb in range(self.num_global_batches):
                batch_indices = order[gb * self.effective_batch_size : (gb + 1) * self.effective_batch_size]
                # The trailing partial batch (drop_last=False) may be smaller than the number of
                # (rank, slot) groups (K = dp_size * grad_accum); pad it by repeating samples from
                # the start of the epoch order (same strategy as torch.utils.data.DistributedSampler)
                # so every group still has >= 1 sample and the all-reduce stays consistent.
                if (
                    self.has_tail
                    and gb == self.num_global_batches - 1
                    and len(batch_indices) < self.dp_size * self.grad_accum
                ):
                    pad = self.dp_size * self.grad_accum - len(batch_indices)
                    # Wrap `order` enough times to fill `pad` indices (handles len(order) < pad).
                    padding = (order * math.ceil(pad / len(order)))[:pad]
                    batch_indices = batch_indices + padding
                yield from self._yield_global_batch(batch_indices)

        def _yield_global_batch(self, batch_indices):
            batch_lengths = [self.lengths[i] for i in batch_indices]
            rank_mbs = self._assign(batch_indices, batch_lengths)
            for ga in range(self.grad_accum):
                yield rank_mbs[self.rank][ga]

        def __len__(self):
            return self.num_global_batches * self.grad_accum

        def _cost(self, bs: int, max_len: int) -> float:
            return bs * max_len + self.QUADRATIC_COST_COEF * bs * max_len * max_len

        def _assign(self, indices, lengths):
            # Snake assignment: after balancing, sort the K groups by cost descending and deal them
            # round-wise — slot k receives groups ranked [k*dp_size, (k+1)*dp_size). The highest-cost
            # groups land in slot 0, the next-highest in slot 1, etc., so every slot has the same
            # cost profile across ranks (each rank's slot-k micro-batches cost roughly the same) and
            # the totals even out across grad-accumulation steps.
            sorted_pairs = sorted(zip(indices, lengths), key=lambda x: x[1], reverse=True)

            all_groups = self._balance_groups(sorted_pairs)

            group_costs = [(i, self._group_cost(g)) for i, g in enumerate(all_groups)]
            group_costs.sort(key=lambda x: x[1], reverse=True)

            rank_mbs = [[[] for _ in range(self.grad_accum)] for _ in range(self.dp_size)]

            for slot_idx in range(self.grad_accum):
                slot_start = slot_idx * self.dp_size
                for r in range(self.dp_size):
                    g_idx = group_costs[slot_start + r][0]
                    rank_mbs[r][slot_idx] = [idx for idx, _ in all_groups[g_idx]]

            return rank_mbs

        def _balance_groups(self, sorted_pairs):
            """
            Split `sorted_pairs` (sorted by length descending) into `K = dp_size * grad_accum`
            contiguous groups with near-equal cost, by iterative hill climbing: each step moves one
            sample from the most expensive group to the cheapest one that can receive it.

            The search stops at a local optimum: `seen` detects cycles (no improving move left),
            and `best_counts` snapshots the lowest-spread state seen along the way so exhausting
            the iteration budget only degrades the result gracefully (no samples are dropped).

            When the most expensive group is already at `min_per_group` samples it can no longer
            give samples away, so spread cannot shrink further through it: its whole slot is
            frozen (see `_relieve_and_freeze`) and the search continues on the remaining groups.
            """
            n = len(sorted_pairs)
            K = self.dp_size * self.grad_accum
            min_per_group = 1

            base = n // K
            remainder = n % K
            if n < K * min_per_group:
                raise ValueError(
                    f"Not enough samples ({n}) to fill {K} groups with at least {min_per_group} sample(s) each. "
                    f"Reduce effective_batch_size or increase the dataset size."
                )
            counts = [base + (1 if i < remainder else 0) for i in range(K)]

            def compute_groups_costs(cts):
                # Materialize contiguous groups from per-group counts and price each one.
                groups = []
                idx = 0
                for c in cts:
                    groups.append(sorted_pairs[idx : idx + c])
                    idx += c
                costs = [self._group_cost(g) for g in groups]
                return groups, costs

            frozen = set()
            seen = set()
            best_spread = float("inf")
            best_counts = list(counts)

            # Each iteration moves one sample between groups, so worst-case convergence is O(n).
            # `effective_batch_size` (= n) covers the move-dominated regime (large batch, small K) that a
            # fixed `K * 30` cap could cut off; `K * 30` stays a floor for the detection-dominated regime
            # (tiny batch, n ~= K), where convergence comes from `spread == 0` / `seen` instead. Hitting
            # the cap only degrades the result: `best_counts` is tracked below, so nothing is dropped.
            for _ in range(max(K * 30, self.effective_batch_size)):
                key = tuple(counts)
                if key in seen:
                    break
                seen.add(key)

                groups, costs = compute_groups_costs(counts)

                active = [i for i in range(K) if i not in frozen]
                if len(active) <= 1:
                    break

                active_costs = [costs[i] for i in active]
                spread = max(active_costs) - min(active_costs)
                if spread < best_spread:
                    best_spread = spread
                    best_counts = list(counts)

                hi_idx = max(active, key=lambda i: costs[i])

                if costs[hi_idx] == costs[min(active, key=lambda i: costs[i])]:
                    break

                lo_candidates = sorted(active, key=lambda i: costs[i])
                transferred = False
                for lo_idx in lo_candidates:
                    if lo_idx == hi_idx or costs[lo_idx] == costs[hi_idx]:
                        continue
                    if counts[hi_idx] > min_per_group:
                        counts[hi_idx] -= 1
                        counts[lo_idx] += 1
                        transferred = True
                        break
                if not transferred:
                    if counts[hi_idx] <= min_per_group:
                        cap_cost = costs[hi_idx]
                        slot_members = self._slot_groups(costs, hi_idx)
                        self._relieve_and_freeze(counts, slot_members, hi_idx, frozen, sorted_pairs, cap_cost)
                    else:
                        break

            groups, _ = compute_groups_costs(best_counts)
            return groups

        def _slot_groups(self, costs, pivot_idx):
            """Return the groups that share `pivot_idx`'s slot."""
            K = len(costs)
            dp_size = self.dp_size
            cost_ranking = sorted(range(K), key=lambda i: costs[i], reverse=True)
            pivot_rank = cost_ranking.index(pivot_idx)
            slot_num = pivot_rank // dp_size
            slot_start = slot_num * dp_size
            slot_end = min(slot_start + dp_size, K)
            return set(cost_ranking[slot_start:slot_end])

        def _relieve_and_freeze(self, counts, slot_members, pivot_idx, frozen, sorted_pairs, cap_cost):
            """Freeze `slot_members` and exclude them from further balancing.

            Called when the slot's most expensive group (`pivot_idx`) is down to `min_per_group`
            samples and can no longer give samples away, so its cost cannot shrink anymore.
            Before freezing, top up the slot's other groups from cheaper outside groups, never
            letting a receiver's cost exceed `cap_cost` (the slot's peak). This moves samples
            out of the outside groups, which helps balance the ones still active."""
            K = len(counts)
            for gi in slot_members:
                if gi in frozen or gi == pivot_idx:
                    continue
                while True:
                    _, costs_now = self._groups_costs_from(counts, sorted_pairs)
                    idx_start = sum(counts[:gi])
                    gi_max_len = sorted_pairs[idx_start][1] if counts[gi] > 0 else 0
                    projected = self._cost(counts[gi] + 1, gi_max_len)
                    if projected > cap_cost:
                        break
                    donors = [i for i in range(K) if i not in frozen and i not in slot_members and counts[i] > 1]
                    if not donors:
                        break
                    donor = min(donors, key=lambda i: costs_now[i])
                    counts[gi] += 1
                    counts[donor] -= 1
            frozen.update(slot_members)

        def _groups_costs_from(self, counts, sorted_pairs):
            groups = []
            idx = 0
            for c in counts:
                groups.append(sorted_pairs[idx : idx + c])
                idx += c
            costs = [self._group_cost(g) for g in groups]
            return groups, costs

        def _group_cost(self, group):
            if not group:
                return 0.0
            max_len = group[0][1]
            bs = len(group)
            return self._cost(bs, max_len)



# Fields that never reach the encoder input, or carry no information about its length.
_NON_TASK_KEYS = frozenset({"id", "source", "meta", "text"})


def _row_kind(row: Dict[str, Any]) -> Tuple[str, ...]:
    """Which task fields a row carries; rows of one kind share their prompt layout."""
    kind = sorted(key for key in row if key not in _NON_TASK_KEYS)
    extraction = row.get("extraction")
    if isinstance(extraction, list) and any(isinstance(e, dict) and e.get("relations") for e in extraction):
        kind.append("+relations")
    return tuple(kind)


def _row_features(row: Dict[str, Any]) -> Tuple[float, float]:
    """Characters of the text, and of every other field (labels, schemas, prompts, targets)."""
    text = row.get("text")
    rest = {key: value for key, value in row.items() if key not in _NON_TASK_KEYS}
    return float(len(text) if isinstance(text, str) else 0), float(len(json.dumps(rest, ensure_ascii=False)))


_POOL_RECORDS: Optional[Sequence[Dict[str, Any]]] = None


def _features_chunk(bounds: Tuple[int, int]):
    start, end = bounds
    rows = _POOL_RECORDS[start:end]
    return [_row_kind(row) for row in rows], [_row_features(row) for row in rows]


def _all_features(records: Sequence[Dict[str, Any]], num_workers: int):
    global _POOL_RECORDS
    n = len(records)
    if num_workers <= 1 or n < 200_000 or "fork" not in multiprocessing.get_all_start_methods():
        return [_row_kind(row) for row in records], [_row_features(row) for row in records]
    chunk = math.ceil(n / (num_workers * 4))
    _POOL_RECORDS = records  # inherited by the forked workers, not pickled
    try:
        with multiprocessing.get_context("fork").Pool(num_workers) as pool:
            parts = pool.map(_features_chunk, [(i, min(i + chunk, n)) for i in range(0, n, chunk)])
    finally:
        _POOL_RECORDS = None
    kinds, features = [], []
    for part_kinds, part_features in parts:
        kinds.extend(part_kinds)
        features.extend(part_features)
    return kinds, features


def estimate_lengths(
    records: Sequence[Dict[str, Any]],
    encoded_length: Callable[[Dict[str, Any]], Optional[Tuple[int, int]]],
    max_text_length: int,
    samples_per_kind: int = 64,
    seed: int = 0,
    num_workers: int = 8,
) -> Tuple[List[int], Dict[str, Any]]:
    """Estimated encoded length (text tokens + image tokens) of every record.

    ``encoded_length(row)`` encodes one row and returns ``(text_tokens,
    image_tokens)``, or None if the row cannot be encoded. It runs on up to
    ``samples_per_kind`` rows of each row kind; per kind, text tokens are fit
    as ``a * text_chars + b * other_chars + c`` on the sampled rows that were
    not truncated, and image tokens are taken as the kind's mean. Kinds too
    rare to fit use a fit over all sampled rows.
    """
    kinds, features = _all_features(records, num_workers)
    by_kind: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for index, kind in enumerate(kinds):
        by_kind[kind].append(index)

    rng = random.Random(seed)
    samples: Dict[Tuple[str, ...], List[Tuple[int, int, int]]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # truncation and alignment warnings of the processors
        for kind, indices in by_kind.items():
            chosen = indices if len(indices) <= samples_per_kind else rng.sample(indices, samples_per_kind)
            measured = []
            for index in chosen:
                try:
                    result = encoded_length(records[index])
                except Exception:  # an unencodable row is skipped by the collator too
                    result = None
                if result is not None:
                    measured.append((index, int(result[0]), int(result[1])))
            samples[kind] = measured

    def fit(rows):
        unclipped = [r for r in rows if r[1] < max_text_length]
        rows_for_fit = unclipped if len(unclipped) >= 8 else rows
        if not rows_for_fit:
            return None
        x = np.array([[features[i][0], features[i][1], 1.0] for i, _, _ in rows_for_fit])
        y = np.array([t for _, t, _ in rows_for_fit], dtype=float)
        weights, *_ = np.linalg.lstsq(x, y, rcond=None)
        return weights

    pooled = [row for rows in samples.values() for row in rows]
    global_weights = fit(pooled)
    global_image = float(np.mean([img for _, _, img in pooled])) if pooled else 0.0
    if global_weights is None:
        global_weights = np.array([0.25, 0.25, 0.0])  # ~4 characters per token

    lengths = [0] * len(records)
    report = {}
    for kind, indices in by_kind.items():
        measured = samples.get(kind, [])
        weights = fit(measured) if len(measured) >= 8 else None
        weights = global_weights if weights is None else weights
        image = float(np.mean([img for _, _, img in measured])) if measured else global_image
        x = np.array([[features[i][0], features[i][1], 1.0] for i in indices])
        text = np.clip(x @ weights, 1, max_text_length)
        for index, value in zip(indices, text):
            lengths[index] = int(round(value + image))
        report[" ".join(kind)] = {
            "rows": len(indices),
            "sampled": len(measured),
            "mean_estimate": float(np.mean(text) + image),
            "mean_measured": float(np.mean([t + img for _, t, img in measured])) if measured else None,
        }
    return lengths, report


# Raw-row fields that select a task head; a row's task group is the set it carries.
TASK_FIELDS = (
    "classification",
    "extraction",
    "structuring",
    "embedding",
    "calculation",
    "sequence_editing",
    "object_detection",
    "image_classification",
    "segmentation",
    "open_relex",
    "count",
    "audio_classification",
    "audio_segmentation",
)


def row_task(row: Dict[str, Any]) -> str:
    """The task group of a row: its task fields, whether it has relations, and its media.

    Relations select the joint relex head and media the vision or audio
    tower, so rows that differ there run different parts of the model.
    """
    parts = [field for field in TASK_FIELDS if row.get(field) is not None]
    extraction = row.get("extraction")
    if isinstance(extraction, list) and any(isinstance(e, dict) and e.get("relations") for e in extraction):
        parts.append("+relations")
    if row.get("image") is not None or row.get("images") or row.get("pixel_values") is not None:
        parts.append("+image")
    if row.get("audio") is not None or row.get("audio_values") is not None:
        parts.append("+audio")
    return " ".join(parts) or "none"


class TaskGroupedBatchRebalanceSampler(BatchRebalanceSampler):
    """``BatchRebalanceSampler`` whose micro-batches keep task groups together.

    Each optimizer step still takes the same shuffled ``effective_batch_size``
    samples, so the task mix of a step is unchanged; only their split differs.
    The parent sorts the step by length and cuts it into micro-batches of
    balanced padded cost. Here the step is ordered by task group instead (the
    groups with the longest rows first, each group's rows by length, the
    direction alternating so neighbouring groups meet at similar lengths)
    before the same balancing cuts it into contiguous micro-batches. Most
    micro-batches then hold one task and run only its heads; a micro-batch
    spans two groups only where a cut falls between them.

    Giving every task its own micro-batches is not possible with this model's
    eight micro-batches per step and nine task groups: the largest task would
    take several times the padded tokens of an even split.
    """

    def __init__(self, lengths: Sequence[int], tasks: Sequence[str], **kwargs):
        super().__init__(lengths=lengths, **kwargs)
        if len(tasks) != len(self.lengths):
            raise ValueError(f"Got {len(tasks)} task labels for {len(self.lengths)} samples.")
        self.tasks = list(tasks)

    def _group_cost(self, group):
        # Rows are not sorted across task boundaries, so the longest row is not always first.
        if not group:
            return 0.0
        return self._cost(len(group), max(length for _, length in group))

    def _balance_groups(self, sorted_pairs):
        by_task: Dict[str, List[Tuple[int, int]]] = {}
        for pair in sorted_pairs:  # longest first within each group
            by_task.setdefault(self.tasks[pair[0]], []).append(pair)
        ordered = sorted(by_task.values(), key=lambda group: group[0][1], reverse=True)
        sequence: List[Tuple[int, int]] = []
        for position, group in enumerate(ordered):
            sequence.extend(group if position % 2 == 0 else reversed(group))
        return super()._balance_groups(sequence)
