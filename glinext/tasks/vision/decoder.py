"""Decoders for vision task heads."""

from typing import List

import torch

from .. import TaskDecoder
from ...processing.decoder import unflatten_by_batch_origin


def _id_maps(classes_mapping, attr):
    maps = []
    if classes_mapping is None:
        return maps
    for cm in getattr(classes_mapping, attr, []):
        for item in cm.items:
            maps.append(item.class_to_id.get_reverse_mapping())
    return maps


class ImageClassificationDecoder(TaskDecoder):
    def decode(self, model_output, classes_mapping=None, threshold=0.5,
               multi_label=True, **kwargs) -> List[List[dict]]:
        logits = getattr(model_output, "image_classification_logits", None)
        origin = getattr(model_output, "image_classification_batch_origin", None)
        if logits is None or origin is None:
            return []
        probs = torch.sigmoid(logits)
        id_to_class_maps = _id_maps(classes_mapping, "image_classification_mapping")
        flat_results = []
        for b in range(probs.shape[0]):
            id_to_class = id_to_class_maps[b] if b < len(id_to_class_maps) else {}
            num_classes = len(id_to_class) if id_to_class else probs.shape[1]
            if multi_label:
                flat_results.append([
                    {"label": id_to_class.get(c, str(c)), "score": probs[b, c].item()}
                    for c in range(num_classes)
                    if probs[b, c].item() > threshold
                ])
            else:
                valid = probs[b, :num_classes]
                idx = int(valid.argmax().item())
                score = valid[idx].item()
                flat_results.append([
                    {"label": id_to_class.get(idx, str(idx)), "score": score}
                ] if score > threshold else [])
        return unflatten_by_batch_origin(flat_results, origin, model_output.batch_size)


class ObjectDetectionDecoder(TaskDecoder):
    task_name = "object_detection"
    logits_name = "object_detection_logits"
    origin_name = "object_detection_batch_origin"
    bbox_name = "object_detection_boxes"
    objectness_name = "object_detection_objectness_logits"
    anchor_mask_name = "object_detection_anchor_mask"
    mapping_name = "object_detection_mapping"

    def decode(self, model_output, classes_mapping=None, threshold=0.5, **kwargs):
        logits = getattr(model_output, self.logits_name, None)
        boxes = getattr(model_output, self.bbox_name, None)
        origin = getattr(model_output, self.origin_name, None)
        objectness = getattr(model_output, self.objectness_name, None)
        anchor_mask = getattr(model_output, self.anchor_mask_name, None)
        if logits is None or boxes is None or origin is None:
            return []

        class_probs = torch.sigmoid(logits)
        obj_probs = torch.sigmoid(objectness) if objectness is not None else torch.ones_like(class_probs[..., 0])
        id_to_class_maps = _id_maps(classes_mapping, self.mapping_name)
        flat_results = []
        for b in range(class_probs.shape[0]):
            id_to_class = id_to_class_maps[b] if b < len(id_to_class_maps) else {}
            num_classes = len(id_to_class) if id_to_class else class_probs.shape[-1]
            predictions = []
            for a in range(class_probs.shape[1]):
                if anchor_mask is not None and not bool(anchor_mask[b, a].item()):
                    continue
                scores = class_probs[b, a, :num_classes] * obj_probs[b, a]
                best_idx = int(scores.argmax().item())
                score = float(scores[best_idx].item())
                if score <= threshold:
                    continue
                predictions.append({
                    "label": id_to_class.get(best_idx, str(best_idx)),
                    "score": score,
                    "bbox": boxes[b, a].detach().cpu().tolist(),
                    "anchor": a,
                })
            flat_results.append(predictions)
        return unflatten_by_batch_origin(flat_results, origin, model_output.batch_size)


class SegmentationDecoder(ObjectDetectionDecoder):
    task_name = "segmentation"
    logits_name = "segmentation_logits"
    origin_name = "segmentation_batch_origin"
    bbox_name = "segmentation_boxes"
    objectness_name = "segmentation_objectness_logits"
    anchor_mask_name = "segmentation_anchor_mask"
    mapping_name = "segmentation_mapping"

    def decode(self, model_output, classes_mapping=None, threshold=0.5,
               mask_threshold=0.5, return_masks=False, **kwargs):
        decoded = super().decode(model_output, classes_mapping, threshold, **kwargs)
        mask_logits = getattr(model_output, "segmentation_mask_logits", None)
        origin = getattr(model_output, self.origin_name, None)
        if not return_masks or mask_logits is None or origin is None or not decoded:
            return decoded

        mask_probs = torch.sigmoid(mask_logits)
        flat_masks = []
        for b in range(mask_probs.shape[0]):
            flat_masks.append(mask_probs[b])
        grouped_masks = unflatten_by_batch_origin(flat_masks, origin, model_output.batch_size)
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
