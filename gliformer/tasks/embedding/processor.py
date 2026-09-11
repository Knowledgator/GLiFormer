"""Embedding similarity task processor."""

import torch

from .. import TaskProcessor


class EmbeddingProcessor(TaskProcessor):
    """Processor for embedding similarity task.

    Each training item uses top-level ``text`` as the root/anchor text and
    carries related candidates in its ``embedding`` field.  Preferred forms:

      - ``{"positive": ["similar text"], "negative": ["different text"]}``
      - ``[{"text": "candidate", "label": 1}, {"text": "candidate", "label": 0}]``
      - ``[["candidate", 1], ["candidate", 0]]``

    The legacy explicit pair form ``[text_a, text_b, score]`` is still
    accepted for compatibility.

    The processor collects all pairs from the batch, and the collator
    tokenizes them as a separate batch that the model encodes independently
    through the shared encoder.
    """

    def __init__(self, config, **kwargs):
        super().__init__(config)

    def get_classes_mapping(self, batch_list, **kwargs):
        return None

    @staticmethod
    def _normalize_text(value):
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return " ".join(str(part) for part in value)
        return str(value)

    @staticmethod
    def _score_from_key(key):
        if key in ("positive", "positives", "pos"):
            return 1.0
        if key in ("negative", "negatives", "neg"):
            return 0.0
        return None

    def _iter_root_pairs(self, item):
        root_text = item.get('text')
        embedding = item.get('embedding', [])

        if isinstance(embedding, dict):
            for key, values in embedding.items():
                score = self._score_from_key(key)
                if score is None:
                    continue
                if isinstance(values, (str, dict)) or not isinstance(values, (list, tuple)):
                    values = [values]
                for value in values:
                    if isinstance(value, dict):
                        candidate = value.get('text', value.get('candidate', value.get('value')))
                        value_score = value.get('score', value.get('label', score))
                    else:
                        candidate = value
                        value_score = score
                    if candidate is not None and root_text:
                        yield root_text, candidate, value_score
            return

        for entry in embedding:
            if isinstance(entry, dict):
                candidate = entry.get('text', entry.get('candidate', entry.get('value')))
                score = entry.get('score', entry.get('label'))
                relation = entry.get('relation', entry.get('type'))
                if score is None and isinstance(relation, str):
                    score = self._score_from_key(relation)
                if candidate is not None and score is not None and root_text:
                    yield root_text, candidate, score
                continue

            if isinstance(entry, (list, tuple)):
                if len(entry) == 2 and root_text:
                    yield root_text, entry[0], entry[1]
                elif len(entry) >= 3:
                    yield entry[0], entry[1], entry[2]

    def create_labels(self, batch_list, classes_mapping, **kwargs):
        embedding_texts = []
        pair_indices = []
        scores = []

        for item in batch_list:
            item_pairs = list(self._iter_root_pairs(item))
            for text_a, text_b, score in item_pairs:
                pair_indices.append([len(embedding_texts), len(embedding_texts) + 1])
                embedding_texts.extend([
                    self._normalize_text(text_a),
                    self._normalize_text(text_b),
                ])
                scores.append(float(score))

            # Legacy pair indices counted one encoder row for a batch item
            # without embedding pairs.  Preserve that offset in the separate
            # embedding text batch so the indices remain both compatible and
            # valid for the newer independently encoded representation.
            if not item_pairs:
                embedding_texts.append(self._normalize_text(item.get("text", "")))

        if not scores:
            return None

        return {
            'embedding_texts': embedding_texts,
            'embedding_pair_idx': torch.tensor(pair_indices, dtype=torch.long),
            'embedding_labels': torch.tensor(scores, dtype=torch.float),
        }
