"""GLiNExT decoder — factory that assembles per-task decoders based on config."""

from typing import Dict, Optional

from .config import GLiNextConfig


class GLiNExTDecoder:
    """General decoder that delegates to per-task decoders for post-processing.

    Each task's decoder converts raw model logits into structured predictions
    (entity spans, classification labels, relation triples, etc.).
    """

    def __init__(self, config: GLiNextConfig):
        self.config = config
        self.task_decoders: Dict[str, object] = {}

        if config.ner_config is not None:
            from .tasks.ner.decoder import NERDecoder
            if hasattr(NERDecoder, 'from_config'):
                self.task_decoders["ner"] = NERDecoder.from_config(config)

        if config.classification_config is not None:
            from .tasks.classification.decoder import ClassificationDecoder
            if hasattr(ClassificationDecoder, 'from_config'):
                self.task_decoders["classification"] = ClassificationDecoder.from_config(config)

        if config.joint_relex_config is not None:
            from .tasks.joint_relex.decoder import JointRelexDecoder
            if hasattr(JointRelexDecoder, 'from_config'):
                self.task_decoders["joint_relex"] = JointRelexDecoder.from_config(config)

        if config.open_relex_config is not None:
            from .tasks.open_relex.decoder import OpenRelexDecoder
            if hasattr(OpenRelexDecoder, 'from_config'):
                self.task_decoders["open_relex"] = OpenRelexDecoder.from_config(config)

        if config.structuring_config is not None:
            from .tasks.structuring.decoder import StructuringDecoder
            if hasattr(StructuringDecoder, 'from_config'):
                self.task_decoders["structuring"] = StructuringDecoder.from_config(config)

        if config.embedding_config is not None:
            from .tasks.embedding.decoder import EmbeddingDecoder
            if hasattr(EmbeddingDecoder, 'from_config'):
                self.task_decoders["embedding"] = EmbeddingDecoder.from_config(config)

        if config.count_config is not None:
            from .tasks.count.decoder import CountDecoder
            if hasattr(CountDecoder, 'from_config'):
                self.task_decoders["count"] = CountDecoder.from_config(config)

    def decode(self, model_output, classes_mapping=None, **kwargs) -> Dict:
        """Decode model output into structured predictions per task.

        Args:
            model_output: GLiNExTOutput from model forward pass.
            classes_mapping: BatchClassesMapping for label resolution.

        Returns:
            Dict mapping task name to decoded predictions.
        """
        results = {}
        for name, decoder in self.task_decoders.items():
            results[name] = decoder.decode(model_output, classes_mapping=classes_mapping, **kwargs)
        return results
