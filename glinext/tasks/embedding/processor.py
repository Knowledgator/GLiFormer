"""Embedding similarity task processor."""

import torch

from .. import TaskProcessor


class EmbeddingProcessor(TaskProcessor):
    """Processor for embedding similarity task."""

    def __init__(self, config, **kwargs):
        super().__init__(config)

    def get_classes_mapping(self, batch_list, **kwargs):
        return None

    def create_labels(self, batch_list, classes_mapping, **kwargs):
        pair_indices = []
        scores = []
        offset = 0

        for item in batch_list:
            embedding_pairs = item.get('embedding', [])
            for pair in embedding_pairs:
                pair_indices.append([offset, offset + 1])
                scores.append(float(pair[2]))
                offset += 2
            if not embedding_pairs:
                offset += 1

        if not pair_indices:
            return None

        return {
            'embedding_pair_idx': torch.tensor(pair_indices, dtype=torch.long),
            'embedding_labels': torch.tensor(scores, dtype=torch.float),
        }
