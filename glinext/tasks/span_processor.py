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

    def __init__(self, config, tokenizer=None, words_splitter=None, **kwargs):
        super().__init__(config, tokenizer, words_splitter, **kwargs)
        self.words_splitter = words_splitter
        self.parent_token = config.parent_token
        self.sep_token = config.sep_token

    @staticmethod
    def _build_token_char_maps(tokens_with_spans):
        """Build char-start→token-idx and char-end→token-idx mappings."""
        s2t = {s: idx for idx, (_, s, _) in enumerate(tokens_with_spans)}
        e2t = {e: idx for idx, (_, _, e) in enumerate(tokens_with_spans)}
        return s2t, e2t

    @staticmethod
    def _resolve_text_span(text, tokens_with_spans, mention_text):
        """Resolve a single text mention to token start/end indices.

        Returns:
            (start_token, end_token) or (-1, -1) if not found.
        """
        s2t = {s: idx for idx, (_, s, _) in enumerate(tokens_with_spans)}
        e2t = {e: idx for idx, (_, _, e) in enumerate(tokens_with_spans)}
        try:
            for match in re.finditer(re.escape(mention_text), text, re.IGNORECASE):
                s, e = match.start(), match.end()
                if s in s2t and e in e2t:
                    return s2t[s], e2t[e]
        except (ValueError, re.error):
            pass
        return -1, -1

    @staticmethod
    def _resolve_entity_spans(text, tokens_with_spans, ner):
        """Resolve a list of entity mentions to token indices.

        Handles both text-based [text, label] and pre-resolved [start, end, label] formats.

        Returns:
            List of [start_token, end_token, label].
        """
        if not ner:
            return []
        s2t = {s: idx for idx, (_, s, _) in enumerate(tokens_with_spans)}
        e2t = {e: idx for idx, (_, _, e) in enumerate(tokens_with_spans)}
        resolved = []
        for ent in ner:
            if len(ent) == 3 and isinstance(ent[0], int):
                resolved.append(list(ent))
            else:
                ent_text, label = ent[0], ent[-1]
                try:
                    for match in re.finditer(re.escape(ent_text), text, re.IGNORECASE):
                        s, e = match.start(), match.end()
                        if s in s2t and e in e2t:
                            resolved.append([s2t[s], e2t[e], label])
                except (ValueError, re.error):
                    continue
        return resolved

    def _tokenize_text(self, item):
        """Tokenize item text via words_splitter, caching result.

        Returns:
            (tokens_with_spans, tokens) or (None, None) if no text.
        """
        text = item.get('text', '')
        if not text:
            return None, None
        tokens_with_spans = list(self.words_splitter(text))
        tokens = [tok for tok, _, _ in tokens_with_spans]
        if 'tokenized_text' not in item:
            item['tokenized_text'] = tokens
        return tokens_with_spans, tokens

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