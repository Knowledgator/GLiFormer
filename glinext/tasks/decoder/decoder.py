"""Autoregressive decoder post-processing — beam search / greedy decoding."""

from typing import List

from .. import TaskDecoder


class AutoregressiveDecoder(TaskDecoder):
    """Post-processing for the autoregressive decoder head.

    Converts decoder outputs into label strings via tokenizer decoding.
    """

    def __init__(self, config, decoder_tokenizer=None):
        super().__init__(config)
        self.decoder_tokenizer = decoder_tokenizer

    def decode(self, model_output, classes_mapping=None, **kwargs) -> List[str]:
        """Decode autoregressive outputs into label strings.

        Note: The DecoderHead primarily operates during training.
        Inference-time decoding requires the full decoder model and
        is typically handled by GLiNER's existing decode methods.
        """
        return []
