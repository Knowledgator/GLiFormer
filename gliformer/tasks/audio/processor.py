"""Processors for audio classification and temporal segmentation tasks."""

import math
from typing import Any, Dict, List, Optional

import torch

from ..media_processor import MediaTaskProcessor
from ..media_processor import unique_labels as _unique


class AudioProcessor(MediaTaskProcessor):
    """Shared processor for audio-level and temporal-segmentation supervision."""

    skip_item_flag = "_skip_audio_tasks"

    def __init__(self, config, task_name: str, **kwargs):
        super().__init__(config, task_name, **kwargs)
        cfg = getattr(config, f"{task_name}_config", None)
        self.mask_size = int(getattr(cfg, "mask_size", 256))

    @staticmethod
    def _segments(item: Dict[str, Any]) -> List[Dict[str, Any]]:
        return list(item.get("segments") or item.get("audio_segments") or [])

    @classmethod
    def _instances(cls, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        return cls._segments(item)

    def _fallback_label_group(self, item: Dict[str, Any]) -> Dict[str, Any]:
        labels = self.labels_from_item(item)
        group: Dict[str, Any] = {
            "name": item.get("name", self.task_name),
            "all_labels": labels,
        }
        if "true_labels" in item:
            group["true_labels"] = item.get("true_labels")
        segments = self._segments(item)
        if segments:
            group["segments"] = segments
        return group

    def _mapping_labels(self, group: Dict[str, Any]) -> List[str]:
        labels = (
            group.get("all_labels")
            or group.get("labels")
            or group.get("classes")
            or group.get("true_labels")
            or _unique(segment.get("label") for segment in group.get("segments", []))
            or _unique(segment.get("label") for segment in group.get("audio_segments", []))
        )
        return _unique(labels or [])

    def create_labels(self, batch_list, classes_mapping, **kwargs):
        if self.task_name == "audio_classification":
            return self._create_audio_classification_labels(batch_list, classes_mapping)
        if self.task_name == "audio_segmentation":
            return self._create_audio_segmentation_labels(batch_list, classes_mapping)
        return None

    def _classification_true_labels(self, item: Dict[str, Any], group_idx: int):
        groups = self._label_groups_for_item(item)
        group = groups[group_idx] if group_idx < len(groups) else {}
        true_labels = group.get("true_labels")
        if true_labels is None and self._task_label_groups(item) is None:
            true_labels = self.true_labels_from_item(item)
        return true_labels or []

    def _create_audio_classification_labels(self, batch_list, classes_mapping):
        return self._create_classification_labels(
            batch_list,
            classes_mapping,
            "audio_classification_labels",
        )

    @staticmethod
    def _duration(item: Dict[str, Any]) -> Optional[float]:
        for key in ("audio_duration", "duration", "_audio_duration_seconds"):
            duration = item.get(key)
            if duration is None:
                continue
            try:
                duration = float(duration)
            except (OverflowError, TypeError, ValueError):
                continue
            if math.isfinite(duration) and duration > 0:
                return duration
        return None

    @classmethod
    def _normalize_segment(cls, segment: Dict[str, Any], item: Dict[str, Any]) -> torch.Tensor:
        if "segment" in segment:
            start, end = segment["segment"]
        else:
            start, end = segment.get("start", 0.0), segment.get("end", 0.0)
        values = torch.tensor([float(start), float(end)], dtype=torch.float)
        duration = cls._duration(item)
        if values.max() > 1.0 and duration is not None and duration > 0:
            values = values / float(duration)
        start = torch.minimum(values[0], values[1]).clamp(0, 1)
        end = torch.maximum(values[0], values[1]).clamp(0, 1)
        return torch.stack([start, end])

    def _segment_mask(self, segment: Dict[str, Any], norm_segment: torch.Tensor) -> torch.Tensor:
        mask = segment.get("mask")
        if mask is not None:
            tensor = torch.as_tensor(mask, dtype=torch.float).flatten().view(1, 1, -1)
            tensor = torch.nn.functional.interpolate(tensor, size=self.mask_size, mode="nearest")
            return (tensor.view(-1) > 0.5).float()

        out = torch.zeros(self.mask_size, dtype=torch.float)
        start, end = norm_segment
        s = int((start * self.mask_size).floor().clamp(0, self.mask_size - 1).item())
        e = int((end * self.mask_size).ceil().clamp(1, self.mask_size).item())
        out[s:e] = 1.0
        return out

    def _create_audio_segmentation_labels(self, batch_list, classes_mapping):
        total = self._total_groups(classes_mapping)
        if total == 0:
            return None

        group_segments = []
        for _, batch_idx, group_idx, _ in self._mapping_iter(classes_mapping):
            _, group = self._mapping_group_entry(
                batch_list[batch_idx],
                group_idx,
            )
            segments = group.get("segments") or group.get("audio_segments")
            if segments is None and self._task_label_groups(batch_list[batch_idx]) is None:
                segments = self._segments(batch_list[batch_idx])
            group_segments.append(list(segments or []))

        max_segments = max(
            (len(segments) for segments in group_segments),
            default=0,
        )
        max_segments = max(max_segments, 1)
        class_labels = torch.full((total, max_segments), -1, dtype=torch.long)
        segment_labels = torch.zeros(total, max_segments, 2, dtype=torch.float)
        segment_mask = torch.zeros(total, max_segments, dtype=torch.float)
        mask_labels = torch.zeros(total, max_segments, self.mask_size, dtype=torch.float)

        for flat_idx, batch_idx, group_idx, mapping in self._mapping_iter(classes_mapping):
            label_to_id = mapping.class_to_id.class_to_id
            for seg_idx, segment in enumerate(group_segments[flat_idx]):
                label = segment.get("label")
                if label not in label_to_id:
                    continue
                norm_segment = self._normalize_segment(segment, batch_list[batch_idx])
                class_labels[flat_idx, seg_idx] = label_to_id[label]
                segment_labels[flat_idx, seg_idx] = norm_segment
                segment_mask[flat_idx, seg_idx] = 1.0
                mask_labels[flat_idx, seg_idx] = self._segment_mask(segment, norm_segment)

        return {
            "audio_segmentation_class_labels": class_labels,
            "audio_segmentation_segment_labels": segment_labels,
            "audio_segmentation_object_mask": segment_mask,
            "audio_segmentation_mask_labels": mask_labels,
        }
