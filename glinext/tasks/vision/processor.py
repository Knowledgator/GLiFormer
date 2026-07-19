"""Processors for image classification, detection, and segmentation tasks."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .. import TaskProcessor
from ...processing.mappings import (
    BaseClassMapping,
    BatchClassesMapping,
    VisionClassMapping,
    VisionItemMapping,
)


def _unique(items: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(str(item) for item in items if item is not None))


class VisionProcessor(TaskProcessor):
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

    def __init__(self, config, task_name: str, **kwargs):
        super().__init__(config)
        self.task_name = task_name
        self.obj_token = config.obj_token
        self.parent_token = config.parent_token
        self.sep_token = config.sep_token
        self.image_size = int(getattr(config, "image_size", 224))
        cfg = getattr(config, f"{task_name}_config", None)
        self.max_count = int(getattr(cfg, "max_count", 100))
        self.mask_size = int(getattr(cfg, "mask_size", 128))
        self.min_bbox_side_pixels = float(getattr(cfg, "min_bbox_side_pixels", 0.0))
        self.bbox_dedup_iou_threshold = getattr(cfg, "bbox_dedup_iou_threshold", None)
        if self.bbox_dedup_iou_threshold is not None:
            self.bbox_dedup_iou_threshold = float(self.bbox_dedup_iou_threshold)
        self.object_selection_strategy = str(
            getattr(cfg, "object_selection_strategy", "first")
        ).lower()
        if self.min_bbox_side_pixels < 0:
            raise ValueError("min_bbox_side_pixels must be non-negative")
        if (
            self.bbox_dedup_iou_threshold is not None
            and not 0.0 < self.bbox_dedup_iou_threshold <= 1.0
        ):
            raise ValueError("bbox_dedup_iou_threshold must be in (0, 1]")
        allowed_selection = {"first", "largest", "class_balanced_largest"}
        if self.object_selection_strategy not in allowed_selection:
            raise ValueError(
                "object_selection_strategy must be one of "
                f"{sorted(allowed_selection)}, got {self.object_selection_strategy!r}"
            )

    def _task_label_groups(self, item: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        groups = item.get(self.task_name)
        if groups is None:
            return None
        if isinstance(groups, dict):
            groups = [groups]
        elif isinstance(groups, (list, tuple)) and (
            not groups or not isinstance(groups[0], dict)
        ):
            groups = [{"all_labels": list(groups)}]
        return list(groups)

    @staticmethod
    def _groups_for_task(item: Dict[str, Any], task_name: str) -> Optional[List[Dict[str, Any]]]:
        groups = item.get(task_name)
        if groups is None:
            return None
        if isinstance(groups, dict):
            return [groups]
        if isinstance(groups, (list, tuple)) and groups and isinstance(groups[0], dict):
            return list(groups)
        return None

    def _group_for_mapping(self, item: Dict[str, Any], group_idx: int) -> Dict[str, Any]:
        groups = self._task_label_groups(item)
        if groups is not None and group_idx < len(groups):
            return groups[group_idx]
        return item

    @staticmethod
    def labels_from_item(item: Dict[str, Any]) -> List[str]:
        labels = item.get("labels") or item.get("classes") or item.get("all_labels")
        if labels is not None:
            return _unique(labels)
        return _unique(obj.get("label") for obj in item.get("objects", []))

    @staticmethod
    def true_labels_from_item(item: Dict[str, Any]) -> List[str]:
        labels = item.get("true_labels")
        if labels is not None:
            return _unique(labels)
        return _unique(obj.get("label") for obj in item.get("objects", []))

    def get_classes_mapping(self, batch_list, **kwargs):
        mappings = []
        for item in batch_list:
            if item.get("_skip_vision_tasks"):
                mappings.append(VisionClassMapping())
                continue

            task_groups = self._task_label_groups(item)
            if task_groups is None:
                labels = self.labels_from_item(item)
                if not labels and self.task_name == "image_classification":
                    detection_groups = self._groups_for_task(item, "object_detection") or []
                    task_groups = []
                    for group in detection_groups:
                        group_labels = group.get("all_labels") or group.get("labels") or group.get("classes")
                        group_labels = _unique(group_labels or [])
                        if group_labels:
                            task_groups.append({
                                "name": group.get("name", item.get("name", self.task_name)),
                                "all_labels": group_labels,
                                "true_labels": self.true_labels_from_item(group),
                            })
                    if not task_groups:
                        task_groups = None
                if task_groups is None:
                    task_groups = [{
                        "name": item.get("name", self.task_name),
                        "all_labels": labels,
                    }]

            item_mappings = []
            for group in task_groups:
                labels = group.get("all_labels") or group.get("labels") or group.get("classes")
                labels = _unique(labels or [])
                if not labels:
                    continue
                item_mappings.append(
                    VisionItemMapping(
                        class_to_id=BaseClassMapping(
                            class_to_id={label: idx for idx, label in enumerate(labels)},
                            name=group.get("name", item.get("name", self.task_name)),
                        ),
                        name=group.get("name", item.get("name", self.task_name)),
                    )
                )

            if not item_mappings:
                mappings.append(VisionClassMapping())
                continue
            mappings.append(VisionClassMapping(items=item_mappings))
        return mappings

    def contribute_prompt(self, classes_mapping: BatchClassesMapping, batch_idx, use_labels_encoder=False):
        mapping_list = getattr(classes_mapping, f"{self.task_name}_mapping")
        prompt = []
        if batch_idx >= len(mapping_list):
            return prompt
        for mapping in mapping_list[batch_idx].items:
            prompt.append(self.parent_token)
            if mapping.name:
                prompt.append(mapping.name)
            if not use_labels_encoder:
                for label in mapping.class_to_id.class_to_id:
                    prompt.append(f"{self.obj_token} {label}")
            prompt.append(self.sep_token)
        return prompt

    def create_labels(self, batch_list, classes_mapping, **kwargs):
        if self.task_name == "image_classification":
            return self._create_image_classification_labels(batch_list, classes_mapping)
        if self.task_name == "object_detection":
            return self._create_detection_labels(batch_list, classes_mapping, include_masks=False)
        if self.task_name == "segmentation":
            return self._create_detection_labels(batch_list, classes_mapping, include_masks=True)
        return None

    def _mapping_iter(self, classes_mapping):
        return getattr(classes_mapping, f"flat_{self.task_name}_iter")()

    def _total_groups(self, classes_mapping):
        return getattr(classes_mapping, f"total_{self.task_name}_groups")()

    def _create_image_classification_labels(self, batch_list, classes_mapping):
        total = self._total_groups(classes_mapping)
        if total == 0:
            return None
        max_classes = max(len(m.class_to_id.class_to_id) for _, _, _, m in self._mapping_iter(classes_mapping))
        labels = torch.zeros(total, max_classes, dtype=torch.float)
        for flat_idx, batch_idx, group_idx, mapping in self._mapping_iter(classes_mapping):
            label_to_id = mapping.class_to_id.class_to_id
            item = batch_list[batch_idx]
            groups = self._task_label_groups(item)
            if groups is not None and group_idx < len(groups):
                group = groups[group_idx]
            else:
                detection_groups = self._groups_for_task(item, "object_detection") or []
                group = detection_groups[group_idx] if group_idx < len(detection_groups) else item
            for label in self.true_labels_from_item(group):
                if label in label_to_id:
                    labels[flat_idx, label_to_id[label]] = 1.0
        return {"image_classification_labels": labels}

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

    @staticmethod
    def _bbox_area(bbox: torch.Tensor) -> float:
        width = (bbox[2] - bbox[0]).clamp(min=0)
        height = (bbox[3] - bbox[1]).clamp(min=0)
        return float((width * height).item())

    @staticmethod
    def _bbox_iou(first: torch.Tensor, second: torch.Tensor) -> float:
        left = torch.maximum(first[0], second[0])
        top = torch.maximum(first[1], second[1])
        right = torch.minimum(first[2], second[2])
        bottom = torch.minimum(first[3], second[3])
        intersection = (right - left).clamp(min=0) * (bottom - top).clamp(min=0)
        first_area = (first[2] - first[0]).clamp(min=0) * (first[3] - first[1]).clamp(min=0)
        second_area = (second[2] - second[0]).clamp(min=0) * (second[3] - second[1]).clamp(min=0)
        union = first_area + second_area - intersection
        return float((intersection / union.clamp(min=1e-12)).item())

    def _prepare_detection_objects(self, objects, image_size, label_to_id):
        """Normalize, clean, and deterministically select detection targets.

        Web-COCO is converted from segmentation polygons, so its raw object list
        contains many sub-patch contour fragments and near-identical boxes. A
        dense detector should not spend query slots on targets that the visual
        token grid cannot resolve. Selection happens after cleanup so crowded
        images do not systematically discard labels appearing late in the file.
        """
        candidates = []
        for source_index, obj in enumerate(objects or []):
            if not isinstance(obj, dict):
                continue
            label = obj.get("label")
            bbox = obj.get("bbox")
            if label not in label_to_id or bbox is None:
                continue
            norm_bbox = self._normalize_bbox(bbox, image_size)
            width = float((norm_bbox[2] - norm_bbox[0]).clamp(min=0).item())
            height = float((norm_bbox[3] - norm_bbox[1]).clamp(min=0).item())
            if width <= 0.0 or height <= 0.0:
                continue
            if min(width, height) * self.image_size < self.min_bbox_side_pixels:
                continue
            candidates.append({
                "object": obj,
                "label": label,
                "bbox": norm_bbox,
                "area": width * height,
                "source_index": source_index,
            })

        if self.bbox_dedup_iou_threshold is not None:
            deduplicated = []
            # Consider the largest boxes first so a noisy inner contour cannot
            # displace the more complete same-class object box.
            for candidate in sorted(
                candidates,
                key=lambda item: (-item["area"], item["source_index"]),
            ):
                duplicate = any(
                    kept["label"] == candidate["label"]
                    and self._bbox_iou(kept["bbox"], candidate["bbox"])
                    >= self.bbox_dedup_iou_threshold
                    for kept in deduplicated
                )
                if not duplicate:
                    deduplicated.append(candidate)
            candidates = deduplicated

        if self.object_selection_strategy == "largest":
            candidates.sort(key=lambda item: (-item["area"], item["source_index"]))
        elif self.object_selection_strategy == "class_balanced_largest":
            buckets = {label: [] for label in label_to_id}
            for candidate in candidates:
                buckets[candidate["label"]].append(candidate)
            for bucket in buckets.values():
                bucket.sort(key=lambda item: (-item["area"], item["source_index"]))

            balanced = []
            round_index = 0
            while len(balanced) < self.max_count:
                added = False
                for label in label_to_id:
                    bucket = buckets[label]
                    if round_index < len(bucket):
                        balanced.append(bucket[round_index])
                        added = True
                        if len(balanced) == self.max_count:
                            break
                if not added:
                    break
                round_index += 1
            candidates = balanced
        else:
            candidates.sort(key=lambda item: item["source_index"])

        return candidates[:self.max_count]

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
            group = self._group_for_mapping(batch_list[batch_idx], group_idx)
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
        mask_labels = torch.zeros(total, max_objects, self.mask_size, self.mask_size, dtype=torch.float)

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

    def prepare_label_encoder_inputs(self, classes_mapping, labels_tokenizer):
        if labels_tokenizer is None:
            return None
        all_labels = []
        group_sizes = []
        for _, _, _, mapping in self._mapping_iter(classes_mapping):
            labels = list(mapping.class_to_id.class_to_id)
            all_labels.extend(labels)
            group_sizes.append(len(labels))
        if not all_labels:
            return None
        tokenized = labels_tokenizer(
            all_labels, return_tensors="pt", truncation=True,
            padding="longest", add_special_tokens=True,
        )
        return {
            f"{self.task_name}_labels_input_ids": tokenized["input_ids"],
            f"{self.task_name}_labels_attention_mask": tokenized["attention_mask"],
            f"{self.task_name}_labels_group_size": torch.LongTensor(group_sizes),
        }

    @staticmethod
    def load_image(item: Dict[str, Any], image_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if "pixel_values" in item:
            tensor = torch.as_tensor(item["pixel_values"], dtype=torch.float)
            if tensor.dim() == 3 and tensor.shape[0] not in (1, 3):
                tensor = tensor.permute(2, 0, 1)
            h, w = int(tensor.shape[-2]), int(tensor.shape[-1])
            if tensor.max() > 1:
                tensor = tensor / 255.0
            return tensor, (h, w)

        path = item.get("image")
        if path is None:
            raise ValueError("Vision tasks require each item to contain 'image' or 'pixel_values'.")
        image = Image.open(path).convert("RGB")
        orig_size = (image.height, image.width)
        image = image.resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return tensor, orig_size
