"""Processors for image classification, detection, and segmentation tasks."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..media_processor import MediaTaskProcessor
from ..media_processor import unique_labels as _unique


class VisionProcessor(MediaTaskProcessor):
    """Shared processor for image-level and object-level supervision.

    The expected training example is::

        {
            "image": "/path/to/image.jpg",
            "objects": [
                {"label": "cat", "bbox": [x1, y1, x2, y2], "mask": ...},
            ],
        }

    Bounding boxes are stored as normalized ``xyxy`` targets. Absolute boxes are
    detected automatically when any coordinate is larger than one.
    """

    skip_item_flag = "_skip_vision_tasks"

    def __init__(self, config, task_name: str, **kwargs):
        super().__init__(config, task_name, **kwargs)
        cfg = getattr(config, f"{task_name}_config", None)
        self.mask_size = int(getattr(cfg, "mask_size", 128))

    @classmethod
    def _instances(cls, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        return list(item.get("objects", []))

    def _group_for_mapping(self, item: Dict[str, Any], group_idx: int) -> Dict[str, Any]:
        groups = self._task_label_groups(item)
        if groups is not None and group_idx < len(groups):
            return groups[group_idx]
        return item

    def _mapping_groups(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        task_groups = self._task_label_groups(item)
        if task_groups is not None:
            return task_groups

        labels = self.labels_from_item(item)
        if not labels and self.task_name == "image_classification":
            derived_groups = []
            for group in self._groups_for_task(item, "object_detection") or []:
                group_labels = (
                    group.get("all_labels")
                    or group.get("labels")
                    or group.get("classes")
                )
                group_labels = _unique(group_labels or [])
                if group_labels:
                    derived_groups.append({
                        "name": group.get("name", item.get("name", self.task_name)),
                        "all_labels": group_labels,
                        "true_labels": self.true_labels_from_item(group),
                    })
            if derived_groups:
                return derived_groups

        return [self._fallback_label_group(item)]

    def create_labels(self, batch_list, classes_mapping, **kwargs):
        if self.task_name == "image_classification":
            return self._create_image_classification_labels(batch_list, classes_mapping)
        if self.task_name == "object_detection":
            return self._create_detection_labels(batch_list, classes_mapping, include_masks=False)
        if self.task_name == "segmentation":
            return self._create_detection_labels(batch_list, classes_mapping, include_masks=True)
        return None

    def _classification_true_labels(self, item: Dict[str, Any], group_idx: int):
        groups = self._task_label_groups(item)
        if groups is not None and group_idx < len(groups):
            group = groups[group_idx]
        else:
            detection_groups = self._groups_for_task(item, "object_detection") or []
            group = detection_groups[group_idx] if group_idx < len(detection_groups) else item
        return self.true_labels_from_item(group)

    def _create_image_classification_labels(self, batch_list, classes_mapping):
        return self._create_classification_labels(
            batch_list,
            classes_mapping,
            "image_classification_labels",
        )

    @staticmethod
    def _normalize_bbox(bbox, image_size: Optional[Tuple[int, int]]):
        box = torch.as_tensor(bbox, dtype=torch.float)
        if box.numel() != 4:
            raise ValueError(f"bbox must contain 4 coordinates, got {bbox!r}")
        if box.max() > 1.0 and image_size is not None:
            h, w = image_size
            scale = box.new_tensor([w, h, w, h]).clamp(min=1)
            box = box / scale
        x1 = torch.minimum(box[0], box[2]).clamp(0, 1)
        y1 = torch.minimum(box[1], box[3]).clamp(0, 1)
        x2 = torch.maximum(box[0], box[2]).clamp(0, 1)
        y2 = torch.maximum(box[1], box[3]).clamp(0, 1)
        return torch.stack([x1, y1, x2, y2])

    def _prepare_detection_objects(self, objects, image_size, label_to_id):
        """Normalize valid labeled targets while preserving source order."""
        candidates = []
        for obj in objects or []:
            if not isinstance(obj, dict):
                continue
            label = obj.get("label")
            bbox = obj.get("bbox")
            if label not in label_to_id or bbox is None:
                continue
            norm_bbox = self._normalize_bbox(bbox, image_size)
            if not torch.isfinite(norm_bbox).all():
                continue
            if norm_bbox[2] <= norm_bbox[0] or norm_bbox[3] <= norm_bbox[1]:
                continue
            candidates.append({
                "object": obj,
                "label": label,
                "bbox": norm_bbox,
            })
        return candidates

    def _object_mask(self, obj, bbox, image_size):
        mask = obj.get("mask")
        if mask is None:
            out = torch.zeros(self.mask_size, self.mask_size, dtype=torch.float)
            x1, y1, x2, y2 = bbox
            x1 = int((x1 * self.mask_size).floor().clamp(0, self.mask_size - 1).item())
            y1 = int((y1 * self.mask_size).floor().clamp(0, self.mask_size - 1).item())
            x2 = int((x2 * self.mask_size).ceil().clamp(1, self.mask_size).item())
            y2 = int((y2 * self.mask_size).ceil().clamp(1, self.mask_size).item())
            out[y1:y2, x1:x2] = 1.0
            return out

        if isinstance(mask, (str, Path)):
            arr = np.array(Image.open(mask).convert("L"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr)
        else:
            tensor = torch.as_tensor(mask, dtype=torch.float)
            if tensor.dim() == 3:
                tensor = tensor.squeeze(0)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(self.mask_size, self.mask_size), mode="nearest")
        return (tensor.squeeze(0).squeeze(0) > 0.5).float()

    def _create_detection_labels(self, batch_list, classes_mapping, include_masks: bool):
        total = self._total_groups(classes_mapping)
        if total == 0:
            return None

        image_sizes = [item.get("_image_size") for item in batch_list]
        grouped_objects = []
        for _, batch_idx, group_idx, mapping in self._mapping_iter(classes_mapping):
            source_idx, _ = self._mapping_group_entry(
                batch_list[batch_idx],
                group_idx,
            )
            group = self._group_for_mapping(
                batch_list[batch_idx],
                source_idx,
            )
            grouped_objects.append(self._prepare_detection_objects(
                group.get("objects", []),
                image_sizes[batch_idx],
                mapping.class_to_id.class_to_id,
            ))
        max_objects = max(
            (len(objects) for objects in grouped_objects),
            default=0,
        )
        max_objects = max(max_objects, 1)
        class_labels = torch.full((total, max_objects), -1, dtype=torch.long)
        bbox_labels = torch.zeros(total, max_objects, 4, dtype=torch.float)
        object_mask = torch.zeros(total, max_objects, dtype=torch.float)
        mask_labels = None
        if include_masks:
            mask_labels = torch.zeros(
                total,
                max_objects,
                self.mask_size,
                self.mask_size,
                dtype=torch.float,
            )

        for flat_idx, batch_idx, group_idx, mapping in self._mapping_iter(classes_mapping):
            label_to_id = mapping.class_to_id.class_to_id
            image_size = image_sizes[batch_idx]
            for obj_idx, candidate in enumerate(grouped_objects[flat_idx]):
                obj = candidate["object"]
                label = candidate["label"]
                norm_bbox = candidate["bbox"]
                class_labels[flat_idx, obj_idx] = label_to_id[label]
                bbox_labels[flat_idx, obj_idx] = norm_bbox
                object_mask[flat_idx, obj_idx] = 1.0
                if include_masks:
                    mask_labels[flat_idx, obj_idx] = self._object_mask(obj, norm_bbox, image_size)

        prefix = "segmentation" if include_masks else "object_detection"
        result = {
            f"{prefix}_class_labels": class_labels,
            f"{prefix}_bbox_labels": bbox_labels,
            f"{prefix}_object_mask": object_mask,
        }
        if include_masks:
            result["segmentation_mask_labels"] = mask_labels
        return result
