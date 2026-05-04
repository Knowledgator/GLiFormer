"""Decoders for audio task heads."""

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


class AudioClassificationDecoder(TaskDecoder):
    def decode(self, model_output, classes_mapping=None, threshold=0.5,
               multi_label=True, **kwargs) -> List[List[dict]]:
        logits = getattr(model_output, "audio_classification_logits", None)
        origin = getattr(model_output, "audio_classification_batch_origin", None)
        if logits is None or origin is None:
            return []
        probs = torch.sigmoid(logits)
        id_to_class_maps = _id_maps(classes_mapping, "audio_classification_mapping")
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


class AudioSegmentationDecoder(TaskDecoder):
    def decode(self, model_output, classes_mapping=None, threshold=0.5,
               mask_threshold=0.5, return_masks=False, **kwargs):
        logits = getattr(model_output, "audio_segmentation_logits", None)
        segments = getattr(model_output, "audio_segmentation_segments", None)
        origin = getattr(model_output, "audio_segmentation_batch_origin", None)
        objectness = getattr(model_output, "audio_segmentation_objectness_logits", None)
        anchor_mask = getattr(model_output, "audio_segmentation_anchor_mask", None)
        if logits is None or segments is None or origin is None:
            return []

        class_probs = torch.sigmoid(logits)
        obj_probs = torch.sigmoid(objectness) if objectness is not None else torch.ones_like(class_probs[..., 0])
        id_to_class_maps = _id_maps(classes_mapping, "audio_segmentation_mapping")
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
                    "segment": segments[b, a].detach().cpu().tolist(),
                    "anchor": a,
                })
            flat_results.append(predictions)

        decoded = unflatten_by_batch_origin(flat_results, origin, model_output.batch_size)
        mask_logits = getattr(model_output, "audio_segmentation_mask_logits", None)
        if not return_masks or mask_logits is None:
            return decoded

        mask_probs = torch.sigmoid(mask_logits)
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
