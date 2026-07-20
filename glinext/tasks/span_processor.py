"""Shared span resolution utilities for extraction-based task processors.

Provides SpanProcessor base class with:
- Character-to-token index mapping
- Text mention → token span resolution (single and batch)
- Text tokenization helper
- Span index preparation with negative sampling
"""

import re
import random
from typing import Dict, List, Optional, Set, Tuple

import torch

from . import TaskProcessor


class SpanProcessor(TaskProcessor):
    """Base class for processors that resolve text mentions to token spans.

    Used by NER, open relex, and structuring processors.
    """

    def __init__(self, config, tokenizer=None, words_splitter=None, parent_token=None, **kwargs):
        super().__init__(config, tokenizer, words_splitter, **kwargs)
        self.words_splitter = words_splitter
        self.parent_token = parent_token or config.parent_token
        self.sep_token = config.sep_token

    @staticmethod
    def _build_token_char_maps(tokens_with_spans):
        """Build char-start→token-idx and char-end→token-idx mappings."""
        s2t = {s: idx for idx, (_, s, _) in enumerate(tokens_with_spans)}
        e2t = {e: idx for idx, (_, _, e) in enumerate(tokens_with_spans)}
        return s2t, e2t

    @staticmethod
    def _match_to_token_range(s_char, e_char, tokens_with_spans, s2t, e2t):
        """Snap a (s_char, e_char) regex match to a token range.

        Strict alignment first (both endpoints land on token boundaries);
        on miss, fall back to the smallest token range that fully contains
        the match. Returns ``None`` when the match falls in a gap between
        tokens. The fallback rescues short numeric values lodged inside
        hyphenated tokens (``'13'`` ↔ ``'13-4'``), word-prefix matches
        (``'researcher'`` ↔ ``'researchers'``), and multi-word values that
        straddle a token whose char-end disagrees with the regex.
        """
        if s_char in s2t and e_char in e2t:
            return s2t[s_char], e2t[e_char]
        start_tok = end_tok = None
        for tok_idx, (_, ts, te) in enumerate(tokens_with_spans):
            if start_tok is None and ts <= s_char < te:
                start_tok = tok_idx
            if ts < e_char <= te:
                end_tok = tok_idx
                break
        if start_tok is None or end_tok is None or end_tok < start_tok:
            return None
        return start_tok, end_tok

    @classmethod
    def _resolve_text_span(cls, text, tokens_with_spans, mention_text):
        """Resolve a single text mention to token start/end indices.

        Returns:
            (start_token, end_token) or (-1, -1) if not found.
        """
        s2t = {s: idx for idx, (_, s, _) in enumerate(tokens_with_spans)}
        e2t = {e: idx for idx, (_, _, e) in enumerate(tokens_with_spans)}
        try:
            for match in re.finditer(re.escape(mention_text), text, re.IGNORECASE):
                rng = cls._match_to_token_range(
                    match.start(), match.end(), tokens_with_spans, s2t, e2t,
                )
                if rng is not None:
                    return rng
        except (ValueError, re.error):
            pass
        return -1, -1

    @classmethod
    def _char_span_to_token_range(cls, text, tokens_with_spans, start, end, mention_text=None):
        """Convert character offsets to an inclusive token span.

        Public training data uses character offsets.  Python-style exclusive
        ``end`` is preferred, but inclusive ``end`` is accepted when it is the
        one that matches ``mention_text`` or when exclusive matching fails.
        """
        if start is None or end is None:
            return -1, -1
        try:
            start = int(start)
            end = int(end)
        except (TypeError, ValueError):
            return -1, -1
        if start < 0 or end < start:
            return -1, -1

        candidates = []
        if text and mention_text is not None:
            mention_text = str(mention_text)
            if text[start:end] == mention_text:
                candidates.append(end)
            if end < len(text) and text[start:end + 1] == mention_text:
                candidates.append(end + 1)
            if not candidates:
                return -1, -1
        candidates.extend([end, end + 1])

        s2t, e2t = cls._build_token_char_maps(tokens_with_spans)
        seen = set()
        for char_end in candidates:
            if char_end in seen:
                continue
            seen.add(char_end)
            if char_end < start:
                continue
            rng = cls._match_to_token_range(
                start, char_end, tokens_with_spans, s2t, e2t,
            )
            if rng is not None:
                return rng
        return -1, -1

    @classmethod
    def _resolve_labeled_span(cls, text, tokens_with_spans, value, label=None, first_only=True):
        """Normalize one annotated value into token spans.

        Accepted input forms:
          - ``{"text": "John", "label": "person", "start": 0, "end": 4}``
          - ``["John", "person"]``
          - ``[0, 4, "person"]`` where offsets are character offsets
          - ``["John", 0, 4, "person"]``

        Returns a list of ``[token_start, token_end, label]`` entries.
        """
        mention_text = None

        if isinstance(value, dict):
            label = (
                label if label is not None else
                value.get('label', value.get('type', value.get('entity_type')))
            )
            mention_text = value.get('text')
            if 'start' in value and 'end' in value:
                start, end = cls._char_span_to_token_range(
                    text, tokens_with_spans, value.get('start'), value.get('end'), mention_text,
                )
                if start >= 0 and end >= 0 and label is not None:
                    return [[start, end, label]]
                if mention_text is None or label is None:
                    return []
            elif mention_text is None or label is None:
                return []

        elif isinstance(value, (list, tuple)):
            if len(value) >= 4 and isinstance(value[0], str) and isinstance(value[1], int) and isinstance(value[2], int):
                mention_text = value[0]
                label = value[-1] if label is None else label
                start, end = cls._char_span_to_token_range(
                    text, tokens_with_spans, value[1], value[2], mention_text,
                )
                if start >= 0 and end >= 0 and label is not None:
                    return [[start, end, label]]
            elif len(value) >= 3 and isinstance(value[0], int) and isinstance(value[1], int):
                label = value[-1] if label is None else label
                start, end = cls._char_span_to_token_range(
                    text, tokens_with_spans, value[0], value[1], None,
                )
                if start >= 0 and end >= 0 and label is not None:
                    return [[start, end, label]]
            elif value:
                mention_text = value[0]
                label = value[-1] if label is None else label
        else:
            mention_text = value

        if mention_text is None or label is None:
            return []

        resolved = cls._resolve_entity_spans(
            text, tokens_with_spans, [[str(mention_text), label]]
        )
        return resolved[:1] if first_only else resolved

    @classmethod
    def _resolve_entity_spans(cls, text, tokens_with_spans, ner):
        """Resolve a list of entity mentions to token indices.

        Handles text mentions, dict values, and character-offset spans.  The
        legacy ``[token_start, token_end, label]`` form is already resolved and
        is therefore passed through unchanged.  Character-offset inputs used
        by task processors are normalized through ``_resolve_labeled_span``;
        keeping the pass-through here preserves the public helper's original
        contract without making the offset interpretation ambiguous.

        Returns:
            List of [start_token, end_token, label].
        """
        if not ner:
            return []
        resolved = []
        for ent in ner:
            if (
                isinstance(ent, (list, tuple))
                and len(ent) == 3
                and isinstance(ent[0], int)
                and isinstance(ent[1], int)
            ):
                resolved.append(list(ent))
            elif isinstance(ent, dict) or (
                isinstance(ent, (list, tuple)) and len(ent) >= 4 and isinstance(ent[0], str)
                and isinstance(ent[1], int) and isinstance(ent[2], int)
            ):
                resolved.extend(cls._resolve_labeled_span(text, tokens_with_spans, ent))
            else:
                ent_text, label = ent[0], ent[-1]
                # Append every occurrence so structuring can supervise all
                # valid spans. NER/joint-relex callers pick ``resolved[0]``
                # per entity, so their 1:1 input→output contract (and
                # relation head_id/tail_id indices) is preserved.
                try:
                    s2t, e2t = cls._build_token_char_maps(tokens_with_spans)
                    for match in re.finditer(re.escape(ent_text), text, re.IGNORECASE):
                        rng = cls._match_to_token_range(
                            match.start(), match.end(), tokens_with_spans, s2t, e2t,
                        )
                        if rng is not None:
                            resolved.append([rng[0], rng[1], label])
                except (ValueError, re.error):
                    continue
        return resolved

    def _tokenize_text(self, item):
        """Tokenize item text via words_splitter, caching result.

        When the item already carries a pre-computed ``tokenized_text``,
        align *those* tokens to character offsets in ``text`` rather than
        re-tokenizing. ``preprocess_example`` feeds the provided
        ``tokenized_text`` to the model, so span indices must reference the
        same boundaries — re-tokenizing here produced indices that pointed
        into a different token sequence (e.g. ``WhitespaceTokenSplitter``
        keeps ``"387-word"`` as one token, while pre-tokenized data may
        split it into ``["387", "-", "word"]``), silently shifting every
        downstream label.

        Returns:
            (tokens_with_spans, tokens) or (None, None) if no text.
        """
        text = item.get('text', '')
        if not text:
            return None, None
        provided = item.get('tokenized_text')
        if provided:
            tokens = list(provided)
            tokens_with_spans = self._align_tokens_to_text(tokens, text)
            # ``tokenized_text`` historically also acted as a write-once
            # cache.  A stale/cache-only value may not describe ``text`` at
            # all; in that case retain it on the item but use the configured
            # splitter for this call, matching the original helper contract.
            cache_is_aligned = all(
                (not token and start == end) or end > start
                for (token, start, end) in tokens_with_spans
            )
            if not cache_is_aligned:
                tokens_with_spans = list(self.words_splitter(text))
                tokens = [tok for tok, _, _ in tokens_with_spans]
        else:
            tokens_with_spans = list(self.words_splitter(text))
            tokens = [tok for tok, _, _ in tokens_with_spans]
            item['tokenized_text'] = tokens
        return tokens_with_spans, tokens

    @staticmethod
    def _align_tokens_to_text(tokens, text):
        """Locate each token's char offsets in ``text`` by sequential search.

        Tokens are matched left-to-right, starting from the cursor that
        followed the previous match — so duplicate tokens snap to the
        right occurrence. A token that cannot be located after the cursor
        gets a zero-width span at the cursor position and is skipped over.
        """
        aligned = []
        cursor = 0
        text_len = len(text)
        for tok in tokens:
            if not tok:
                aligned.append((tok, cursor, cursor))
                continue
            idx = text.find(tok, cursor)
            if idx < 0:
                aligned.append((tok, cursor, cursor))
                continue
            end = idx + len(tok)
            aligned.append((tok, idx, end))
            cursor = min(end, text_len)
        return aligned

    def _generate_negative_spans(
        self, positive_spans: Set[Tuple[int, int]], num_tokens: int,
        num_negatives: int, max_width: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """Generate random spans that don't overlap with positive entities."""
        if max_width is None:
            max_width = getattr(self.config, "max_width", 10)
        negative_spans = []
        attempts = 0
        max_attempts = num_negatives * 20
        while len(negative_spans) < num_negatives and attempts < max_attempts:
            attempts += 1
            start = random.randint(0, num_tokens - 1)
            width = random.randint(1, min(max_width, num_tokens - start))
            end = start + width - 1
            span = (start, end)
            if span in positive_spans:
                continue
            overlaps = False
            for pos_start, pos_end in positive_spans:
                if not (end < pos_start or start > pos_end):
                    overlaps = True
                    break
            if not overlaps and span not in negative_spans:
                negative_spans.append(span)
        return negative_spans

    def _collect_span_candidates(
        self, positive_spans: List[Tuple[int, int]], num_tokens: int,
        neg_ratio: float,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Set[Tuple[int, int]]]:
        """Collect positive + negative span candidates as tensors.

        Args:
            positive_spans: list of (start, end) positive spans.
            num_tokens: total tokens in sequence.
            neg_ratio: ratio of negatives to positives.

        Returns:
            (span_idx, span_mask, positive_set) or (None, None, set()) if empty.
        """
        if not positive_spans or num_tokens == 0:
            return None, None, set()

        positive_set = set(positive_spans)
        all_spans = list(positive_spans)

        neg_count = int(len(all_spans) * neg_ratio)
        if neg_count > 0:
            negatives = self._generate_negative_spans(positive_set, num_tokens, neg_count)
            all_spans.extend(negatives)

        if not all_spans:
            return None, None, positive_set

        span_idx = torch.LongTensor(all_spans)  # (S, 2)
        span_mask = torch.ones(len(all_spans), dtype=torch.bool)
        return span_idx, span_mask, positive_set
