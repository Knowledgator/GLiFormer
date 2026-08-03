"""GLiNExT decoder — factory that assembles per-task decoders based on config."""

from typing import Dict, List, Optional

import torch

from ..config import GLiNextConfig


def unflatten_by_batch_origin(results: list, batch_origin: torch.Tensor, batch_size: int) -> List[list]:
    """Group BN-indexed results back to B-indexed list of lists.

    Args:
        results: List of length BN with per-group results.
        batch_origin: (BN,) tensor mapping flat idx → batch idx.
        batch_size: Original batch size B.

    Returns:
        List of length B, where each element collects results from its groups.
    """
    output = [[] for _ in range(batch_size)]
    for flat_idx, group_result in enumerate(results):
        bi = batch_origin[flat_idx].item()
        if bi < batch_size:
            output[bi].append(group_result)
    return output


class GLiNExTDecoder:
    """General decoder that delegates to per-task decoders for post-processing.

    Each task's decoder converts raw model logits into structured predictions
    (entity spans, classification labels, relation triples, etc.).
    """

    def __init__(self, config: GLiNextConfig):
        self.config = config
        self.task_decoders: Dict[str, object] = {}

        if config.ner_config is not None:
            from ..tasks.ner.decoder import NERDecoder
            if hasattr(NERDecoder, 'from_config'):
                self.task_decoders["ner"] = NERDecoder.from_config(config)

        if config.classification_config is not None:
            from ..tasks.classification.decoder import ClassificationDecoder
            if hasattr(ClassificationDecoder, 'from_config'):
                self.task_decoders["classification"] = ClassificationDecoder.from_config(config)

        if config.joint_relex_config is not None:
            from ..tasks.joint_relex.decoder import JointRelexDecoder
            if hasattr(JointRelexDecoder, 'from_config'):
                self.task_decoders["joint_relex"] = JointRelexDecoder.from_config(config)

        if config.open_relex_config is not None:
            from ..tasks.open_relex.decoder import OpenRelexDecoder
            if hasattr(OpenRelexDecoder, 'from_config'):
                self.task_decoders["open_relex"] = OpenRelexDecoder.from_config(config)

        if getattr(config, "set_open_relex_config", None) is not None:
            from ..tasks.set_open_relex.decoder import SetOpenRelexDecoder
            if hasattr(SetOpenRelexDecoder, 'from_config'):
                self.task_decoders["set_open_relex"] = (
                    SetOpenRelexDecoder.from_config(config)
                )

        if config.structuring_config is not None:
            from ..tasks.structuring.decoder import StructuringDecoder
            if hasattr(StructuringDecoder, 'from_config'):
                self.task_decoders["structuring"] = StructuringDecoder.from_config(config)

        if config.set_structuring_config is not None:
            from ..tasks.set_structuring.decoder import SetStructuringDecoder
            if hasattr(SetStructuringDecoder, 'from_config'):
                self.task_decoders["set_structuring"] = SetStructuringDecoder.from_config(config)

        if config.embedding_config is not None:
            from ..tasks.embedding.decoder import EmbeddingDecoder
            if hasattr(EmbeddingDecoder, 'from_config'):
                self.task_decoders["embedding"] = EmbeddingDecoder.from_config(config)

        if config.count_config is not None:
            from ..tasks.count.decoder import CountDecoder
            if hasattr(CountDecoder, 'from_config'):
                self.task_decoders["count"] = CountDecoder.from_config(config)

        if config.image_classification_config is not None:
            from ..tasks.vision.decoder import ImageClassificationDecoder
            self.task_decoders["image_classification"] = ImageClassificationDecoder.from_config(config)

        if config.audio_classification_config is not None:
            from ..tasks.audio.decoder import AudioClassificationDecoder
            self.task_decoders["audio_classification"] = AudioClassificationDecoder.from_config(config)

        if config.object_detection_config is not None:
            from ..tasks.vision.decoder import ObjectDetectionDecoder
            self.task_decoders["object_detection"] = ObjectDetectionDecoder.from_config(config)

        if config.segmentation_config is not None:
            from ..tasks.vision.decoder import SegmentationDecoder
            self.task_decoders["segmentation"] = SegmentationDecoder.from_config(config)

        if config.audio_segmentation_config is not None:
            from ..tasks.audio.decoder import AudioSegmentationDecoder
            self.task_decoders["audio_segmentation"] = AudioSegmentationDecoder.from_config(config)

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
            decoded = decoder.decode(model_output, classes_mapping=classes_mapping, **kwargs)
            if decoded is not None and decoded != []:
                results[name] = decoded
        return results

    def map_results(self, decoded: Dict[str, list], **kwargs) -> Dict[str, List]:
        """Map decoded task results back through task-specific decoders."""
        return_anchor_diagnostics = bool(
            kwargs.pop("return_anchor_diagnostics", False)
        )
        results = {}
        for name, task_results in decoded.items():
            decoder = self.task_decoders.get(name)
            if decoder is None:
                continue
            task_kwargs = dict(kwargs)
            diagnostics = None
            if return_anchor_diagnostics and name in {
                "structuring", "set_structuring",
            }:
                diagnostics = []
                task_kwargs["anchor_diagnostics_output"] = diagnostics
            results[name] = decoder.map_results(
                task_results,
                **task_kwargs,
            )
            if diagnostics is not None:
                results[f"{name}_anchor_diagnostics"] = diagnostics
        return results
