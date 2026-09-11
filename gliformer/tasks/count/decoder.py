"""Count task decoder — converts count logits to integer predictions."""

from typing import List

from .. import TaskDecoder
from ...processing.decoder import unflatten_by_batch_origin


class CountDecoder(TaskDecoder):
    """Decodes count logits into integer predictions."""

    def decode(self, model_output, classes_mapping=None, **kwargs) -> List[int]:
        """Decode count logits into integer counts.

        Args:
            model_output: GLiFormerOutput with count_logits.

        Returns:
            List[List[int]] — per batch item, per group count predictions.
        """
        if model_output.count_logits is None:
            return []

        count_cfg = self.config.count_config
        if count_cfg and count_cfg.mode == "classification":
            flat_results = model_output.count_logits.argmax(dim=-1).tolist()
        else:
            flat_results = model_output.count_logits.squeeze(-1).round().clamp(min=0).long().tolist()

        # Unflatten BN → B
        return unflatten_by_batch_origin(
            flat_results, model_output.count_batch_origin, model_output.batch_size,
        )
