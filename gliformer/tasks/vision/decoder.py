"""Decoders for vision task heads."""

import torch

from ..media import MediaClassificationDecoder, MediaSetPredictionDecoder
from ...processing.decoder import unflatten_by_batch_origin


class ImageClassificationDecoder(MediaClassificationDecoder):
    logits_name = "image_classification_logits"
    origin_name = "image_classification_batch_origin"
    mapping_name = "image_classification_mapping"


class ObjectDetectionDecoder(MediaSetPredictionDecoder):
    task_name = "object_detection"
    logits_name = "object_detection_logits"
    origin_name = "object_detection_batch_origin"
    geometry_name = "object_detection_boxes"
    geometry_key = "bbox"
    objectness_name = "object_detection_objectness_logits"
    anchor_mask_name = "object_detection_anchor_mask"
    mapping_name = "object_detection_mapping"
    config_attribute = "object_detection_config"

    def _prepare_geometry(self, geometry):
        # Training keeps edge-crossing corners differentiable. Clip only the
        # public inference result to normalized image space.
        geometry = super()._prepare_geometry(geometry)
        if geometry is None:
            return None
        geometry = geometry.clamp(0.0, 1.0)
        if geometry[2] <= geometry[0] or geometry[3] <= geometry[1]:
            return None
        return geometry


class SegmentationDecoder(ObjectDetectionDecoder):
    task_name = "segmentation"
    logits_name = "segmentation_logits"
    origin_name = "segmentation_batch_origin"
    geometry_name = "segmentation_boxes"
    objectness_name = "segmentation_objectness_logits"
    anchor_mask_name = "segmentation_anchor_mask"
    mapping_name = "segmentation_mapping"
    config_attribute = "segmentation_config"

    def decode(self, model_output, classes_mapping=None, threshold=0.5,
               mask_threshold=0.5, return_masks=False, **kwargs):
        decoded = super().decode(model_output, classes_mapping, threshold, **kwargs)
        mask_logits = getattr(model_output, "segmentation_mask_logits", None)
        mask_validity = getattr(model_output, "segmentation_mask_validity", None)
        origin = getattr(model_output, self.origin_name, None)
        if not return_masks or mask_logits is None or origin is None or not decoded:
            return decoded

        mask_probs = torch.sigmoid(mask_logits.float())
        flat_masks = []
        for b in range(mask_probs.shape[0]):
            flat_masks.append(mask_probs[b])
        grouped_masks = unflatten_by_batch_origin(flat_masks, origin, model_output.batch_size)
        grouped_validity = None
        if mask_validity is not None:
            grouped_validity = unflatten_by_batch_origin(
                [mask_validity[b].bool() for b in range(mask_validity.shape[0])],
                origin,
                model_output.batch_size,
            )
        for batch_idx, groups in enumerate(decoded):
            for group_idx, preds in enumerate(groups):
                if batch_idx >= len(grouped_masks) or group_idx >= len(grouped_masks[batch_idx]):
                    continue
                masks = grouped_masks[batch_idx][group_idx]
                valid_pixels = None
                if grouped_validity is not None:
                    valid_pixels = grouped_validity[batch_idx][group_idx]
                for pred in preds:
                    anchor = pred.get("anchor")
                    if anchor is not None and anchor < masks.shape[0]:
                        decoded_mask = masks[anchor] > mask_threshold
                        if valid_pixels is not None:
                            decoded_mask = decoded_mask & valid_pixels
                        pred["mask"] = decoded_mask.detach().cpu()
        return decoded
