"""Embedding similarity task processor."""

import torch

from .. import TaskProcessor


class EmbeddingProcessor(TaskProcessor):
    """Processor for embedding similarity task.

    Each training item may carry embedding pairs in its ``embedding`` field.
    Each pair is ``[tokenized_text_a, tokenized_text_b, score]`` where
    the texts are lists of word tokens and score is the similarity target.

    The processor collects all pairs from the batch, and the collator
    tokenizes them as a separate batch that the model encodes independently
    through the shared encoder.
    """

    def __init__(self, config, **kwargs):
        super().__init__(config)

    def get_classes_mapping(self, batch_list, **kwargs):
        return None

    def create_labels(self, batch_list, classes_mapping, **kwargs):
        texts_a = []
        texts_b = []
        scores = []

        for item in batch_list:
            for pair in item.get('embedding', []):
                texts_a.append(pair[0])
                texts_b.append(pair[1])
                scores.append(float(pair[2]))

        if not scores:
            return None

        n = len(scores)
        pair_idx = torch.stack([
            torch.arange(n),
            torch.arange(n, 2 * n),
        ], dim=1)

        return {
            'embedding_texts': texts_a + texts_b,
            'embedding_pair_idx': pair_idx,
            'embedding_labels': torch.tensor(scores, dtype=torch.float),
        }
