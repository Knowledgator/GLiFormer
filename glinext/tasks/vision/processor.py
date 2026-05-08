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
            labels = self.labels_from_item(item)
            if not labels:
                mappings.append(VisionClassMapping())
                continue
            mappings.append(
                VisionClassMapping(
                    items=[
                        VisionItemMapping(
                            class_to_id=BaseClassMapping(
                                class_to_id={label: idx for idx, label in enumerate(labels)},
                                name=item.get("name", self.task_name),
                            ),
                            name=item.get("name", self.task_name),
                        )
                    ]
                )
            )
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
            for label in self.true_labels_from_item(batch_list[batch_idx]):
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

        max_objects = min(
            self.max_count,
            max((len(item.get("objects", [])) for item in batch_list), default=0),
        )
        max_objects = max(max_objects, 1)
        class_labels = torch.full((total, max_objects), -1, dtype=torch.long)
        bbox_labels = torch.zeros(total, max_objects, 4, dtype=torch.float)
        object_mask = torch.zeros(total, max_objects, dtype=torch.float)
        mask_labels = torch.zeros(total, max_objects, self.mask_size, self.mask_size, dtype=torch.float)

        image_sizes = [item.get("_image_size") for item in batch_list]
        for flat_idx, batch_idx, group_idx, mapping in self._mapping_iter(classes_mapping):
            label_to_id = mapping.class_to_id.class_to_id
            objects = batch_list[batch_idx].get("objects", [])[:max_objects]
            image_size = image_sizes[batch_idx]
            for obj_idx, obj in enumerate(objects):
                label = obj.get("label")
                if label not in label_to_id:
                    continue
                bbox = obj.get("bbox")
                if bbox is None:
                    continue
                norm_bbox = self._normalize_bbox(bbox, image_size)
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
