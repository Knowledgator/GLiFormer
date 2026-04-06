"""Count task decoder — converts count logits to integer predictions."""

from typing import List

from .. import TaskDecoder


class CountDecoder(TaskDecoder):
    """Decodes count logits into integer predictions."""

    def decode(self, model_output, classes_mapping=None, **kwargs) -> List[int]:
        """Decode count logits into integer counts.

        Args:
            model_output: GLiNExTOutput with count_logits.

        Returns:
            List of integer count predictions, or List[List[int]] when batch_origin available.
        """
        if model_output.count_logits is None:
            return []

        count_cfg = self.config.count_config
        if count_cfg and count_cfg.mode == "classification":
            flat_results = model_output.count_logits.argmax(dim=-1).tolist()
        else:
            flat_results = model_output.count_logits.squeeze(-1).round().clamp(min=0).long().tolist()

        batch_origin = getattr(model_output, 'count_batch_origin', None)
        batch_size = getattr(model_output, 'batch_size', None)
        if batch_origin is not None and batch_size is not None:
            from ...processing.decoder import unflatten_by_batch_origin
            return unflatten_by_batch_origin(flat_results, batch_origin, batch_size)

        return flat_results
