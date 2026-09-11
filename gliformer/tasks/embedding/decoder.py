"""Embedding task decoder — returns similarity scores."""

from typing import List

from .. import TaskDecoder


class EmbeddingDecoder(TaskDecoder):
    """Returns similarity scores from embedding logits."""

    def decode(self, model_output, classes_mapping=None, **kwargs) -> List[float]:
        """Decode embedding logits into similarity scores.

        Args:
            model_output: GLiFormerOutput with embedding_logits.

        Returns:
            List of similarity scores.
        """
        if model_output.embedding_logits is None:
            return []

        return model_output.embedding_logits.tolist()
