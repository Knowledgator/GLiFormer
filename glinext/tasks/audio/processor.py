"""Processors for audio classification and temporal segmentation tasks."""

import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from .. import TaskProcessor
from ...processing.mappings import (
    BaseClassMapping,
    BatchClassesMapping,
    VisionClassMapping,
    VisionItemMapping,
)


def _unique(items: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(str(item) for item in items if item is not None))


class AudioProcessor(TaskProcessor):
    """Shared processor for audio-level and temporal-segmentation supervision."""

    def __init__(self, config, task_name: str, **kwargs):
        super().__init__(config)
        self.task_name = task_name
        self.obj_token = config.obj_token
        self.parent_token = config.parent_token
        self.sep_token = config.sep_token
        cfg = getattr(config, f"{task_name}_config", None)
        self.max_count = int(getattr(cfg, "max_count", 100))
        self.mask_size = int(getattr(cfg, "mask_size", 256))
        self.feature_encoder_types = {"mel", "spectrogram", "conv2d", "mel_conv", "spectrogram_conv", "conv_2d"}

    @staticmethod
    def _segments(item: Dict[str, Any]) -> List[Dict[str, Any]]:
        return list(item.get("segments") or item.get("audio_segments") or [])

    @staticmethod
    def labels_from_item(item: Dict[str, Any]) -> List[str]:
        labels = item.get("labels") or item.get("classes") or item.get("all_labels")
        if labels is not None:
            return _unique(labels)
        return _unique(seg.get("label") for seg in AudioProcessor._segments(item))

    @staticmethod
    def true_labels_from_item(item: Dict[str, Any]) -> List[str]:
        labels = item.get("true_labels")
        if labels is not None:
            return _unique(labels)
        return _unique(seg.get("label") for seg in AudioProcessor._segments(item))

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
        if self.task_name == "audio_classification":
            return self._create_audio_classification_labels(batch_list, classes_mapping)
        if self.task_name == "audio_segmentation":
            return self._create_audio_segmentation_labels(batch_list, classes_mapping)
        return None

    def _mapping_iter(self, classes_mapping):
        return getattr(classes_mapping, f"flat_{self.task_name}_iter")()

    def _total_groups(self, classes_mapping):
        return getattr(classes_mapping, f"total_{self.task_name}_groups")()

    def _create_audio_classification_labels(self, batch_list, classes_mapping):
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
        return {"audio_classification_labels": labels}

    @staticmethod
    def _duration(item: Dict[str, Any]) -> Optional[float]:
        duration = item.get("audio_duration") or item.get("duration")
        if duration is not None:
            return float(duration)
        sample_rate = item.get("sample_rate")
        num_samples = item.get("_audio_num_samples")
        if sample_rate and num_samples:
            return float(num_samples) / float(sample_rate)
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

        max_segments = min(
            self.max_count,
            max((len(self._segments(item)) for item in batch_list), default=0),
        )
        max_segments = max(max_segments, 1)
        class_labels = torch.full((total, max_segments), -1, dtype=torch.long)
        segment_labels = torch.zeros(total, max_segments, 2, dtype=torch.float)
        segment_mask = torch.zeros(total, max_segments, dtype=torch.float)
        mask_labels = torch.zeros(total, max_segments, self.mask_size, dtype=torch.float)

        for flat_idx, batch_idx, group_idx, mapping in self._mapping_iter(classes_mapping):
            label_to_id = mapping.class_to_id.class_to_id
            for seg_idx, segment in enumerate(self._segments(batch_list[batch_idx])[:max_segments]):
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

    def _expects_feature_input(self, item: Dict[str, Any]) -> bool:
        audio_format = str(item.get("audio_format") or item.get("audio_input_format") or "").lower().replace("-", "_")
        encoder_type = str(getattr(self.config, "audio_encoder_type", "") or "").lower().replace("-", "_")
        return (
            encoder_type in self.feature_encoder_types
            or audio_format in self.feature_encoder_types
            or any(key in item for key in ("mel_values", "mel_spectrogram", "audio_features"))
        )

    def _time_length(self, tensor: torch.Tensor) -> int:
        if tensor.dim() <= 1:
            return int(tensor.numel())
        input_format = str(getattr(self.config, "audio_input_format", "freq_first") or "freq_first")
        if input_format == "time_first":
            return int(tensor.shape[-2])
        return int(tensor.shape[-1])

    def load_audio(self, item: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
        feature_key = next(
            (key for key in ("mel_values", "mel_spectrogram", "audio_features") if key in item),
            None,
        )
        if feature_key is not None or "audio_values" in item:
            tensor = torch.as_tensor(
                item[feature_key] if feature_key is not None else item["audio_values"],
                dtype=torch.float,
            )
            if tensor.dim() > 1 and not self._expects_feature_input(item):
                tensor = tensor.reshape(-1)
            return tensor, self._time_length(tensor)

        path = item.get("audio")
        if path is None:
            raise ValueError("Audio tasks require each item to contain 'audio' or 'audio_values'.")
        path = Path(path)
        if path.suffix.lower() != ".wav":
            raise ValueError("Only WAV files are supported without an external audio backend.")

        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())
            sample_rate = handle.getframerate()
        dtype = np.int16 if sample_width == 2 else np.uint8
        audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        if sample_width == 2:
            audio = audio / 32768.0
        else:
            audio = (audio - 128.0) / 128.0
        item.setdefault("sample_rate", sample_rate)
        tensor = torch.from_numpy(audio.copy())
        return tensor, int(tensor.numel())
