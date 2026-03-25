"""Count task decoder — converts count logits to integer predictions."""

from typing import Dict, List, Optional

import torch


class CountDecoder:
    """Decodes count logits into integer predictions."""

    def __init__(self, config):
        self.config = config

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def decode(self, model_output, classes_mapping=None, **kwargs) -> List[int]:
        """Decode count logits into integer counts.

        Args:
            model_output: GLiNExTOutput with count_logits.

        Returns:
            List of integer count predictions.
        """
        if model_output.count_logits is None:
            return []

        count_cfg = self.config.count_config
        if count_cfg and count_cfg.mode == "classification":
            return model_output.count_logits.argmax(dim=-1).tolist()
        else:
            return model_output.count_logits.squeeze(-1).round().clamp(min=0).long().tolist()
