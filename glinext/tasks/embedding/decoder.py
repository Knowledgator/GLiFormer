"""Embedding task decoder — returns similarity scores."""

from typing import Dict, List, Optional

import torch


class EmbeddingDecoder:
    """Returns similarity scores from embedding logits."""

    def __init__(self, config):
        self.config = config

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def decode(self, model_output, classes_mapping=None, **kwargs) -> List[float]:
        """Decode embedding logits into similarity scores.

        Args:
            model_output: GLiNExTOutput with embedding_logits.

        Returns:
            List of similarity scores.
        """
        if model_output.embedding_logits is None:
            return []

        return model_output.embedding_logits.tolist()
