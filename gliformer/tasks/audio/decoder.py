"""Decoders for audio task heads."""

import torch

from ...processing.decoder import unflatten_by_batch_origin
from ..media import MediaClassificationDecoder, MediaSetPredictionDecoder


class AudioClassificationDecoder(MediaClassificationDecoder):
    logits_name = "audio_classification_logits"
    origin_name = "audio_classification_batch_origin"
    mapping_name = "audio_classification_mapping"


class AudioSegmentationDecoder(MediaSetPredictionDecoder):
    logits_name = "audio_segmentation_logits"
    origin_name = "audio_segmentation_batch_origin"
    geometry_name = "audio_segmentation_segments"
    geometry_key = "segment"
    objectness_name = "audio_segmentation_objectness_logits"
    anchor_mask_name = "audio_segmentation_anchor_mask"
    mapping_name = "audio_segmentation_mapping"
    config_attribute = "audio_segmentation_config"

    def decode(
        self,
        model_output,
        classes_mapping=None,
        threshold=0.5,
        mask_threshold=0.5,
        return_masks=False,
        **kwargs,
    ):
        decoded = super().decode(
            model_output,
            classes_mapping,
            threshold=threshold,
            multi_label=kwargs.pop("multi_label", None),
            **kwargs,
        )
        mask_logits = getattr(model_output, "audio_segmentation_mask_logits", None)
        origin = getattr(model_output, self.origin_name, None)
        if not return_masks or mask_logits is None or origin is None:
            return decoded

        mask_probs = torch.sigmoid(mask_logits.float())
        grouped_masks = unflatten_by_batch_origin(
            [mask_probs[b] for b in range(mask_probs.shape[0])],
            origin,
            model_output.batch_size,
        )
        for batch_idx, groups in enumerate(decoded):
            for group_idx, preds in enumerate(groups):
                if batch_idx >= len(grouped_masks) or group_idx >= len(grouped_masks[batch_idx]):
                    continue
                masks = grouped_masks[batch_idx][group_idx]
                for pred in preds:
                    anchor = pred.get("anchor")
                    if anchor is not None and anchor < masks.shape[0]:
                        pred["mask"] = (masks[anchor] > mask_threshold).detach().cpu()
        return decoded
