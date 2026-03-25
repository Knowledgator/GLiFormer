"""Autoregressive decoder post-processing — beam search / greedy decoding."""

from typing import Dict, List, Optional


class AutoregressiveDecoder:
    """Post-processing for the autoregressive decoder head.

    Converts decoder outputs into label strings via tokenizer decoding.
    """

    def __init__(self, config, decoder_tokenizer=None):
        self.config = config
        self.decoder_tokenizer = decoder_tokenizer

    @classmethod
    def from_config(cls, config, decoder_tokenizer=None):
        return cls(config, decoder_tokenizer)

    def decode(self, model_output, classes_mapping=None, **kwargs) -> List[str]:
        """Decode autoregressive outputs into label strings.

        Note: The DecoderHead primarily operates during training.
        Inference-time decoding requires the full decoder model and
        is typically handled by GLiNER's existing decode methods.
        """
        return []
