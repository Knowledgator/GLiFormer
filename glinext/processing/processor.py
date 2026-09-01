"""GLiNExT processors.

The processor layer is split into a task orchestrator plus modality mixins.
Task processors own prompts and labels; modality mixins own how raw text,
images, audio, and layout boxes become canonical model inputs.
"""

import math
import warnings
import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from gliner.data_processing import BaseProcessor
from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor

from ..tasks.audio.processor import AudioProcessor
from ..tasks.classification.processor import ClassificationProcessor
from ..tasks.count.processor import CountProcessor
from ..tasks.embedding.processor import EmbeddingProcessor
from ..tasks.joint_relex.processor import JointRelexProcessor
from ..tasks.ner.processor import NERProcessor
from ..tasks.open_relex.processor import OpenRelexProcessor
from ..tasks.set_open_relex.processor import SetOpenRelexProcessor
from ..tasks.set_structuring.processor import SetStructuringProcessor
from ..tasks.vision.processor import VisionProcessor
from ..utils import pair_2d
from .mappings import (
    BatchClassesMapping,
    CatClassMapping,
    ExtractionClassMapping,
    OpenRelexClassMapping,
    StructuringClassMapping,
    VisionClassMapping,
)
from .structuring_processor import (
    MULTI_LEVEL_META_KEY,
    SET_MULTI_LEVEL_META_KEY,
    StructuringProcessor,
)

_VISION_TASKS = ("image_classification", "object_detection", "segmentation")
_AUDIO_TASKS = ("audio_classification", "audio_segmentation")
_TEXT_TASKS = (
    "ner",
    "classification",
    "joint_relex",
    "open_relex",
    "set_open_relex",
    "count",
    "structuring",
    "set_structuring",
    "embedding",
)
_ALL_TASKS = (*_TEXT_TASKS, *_VISION_TASKS, *_AUDIO_TASKS)


def _normalize_variant(value: Optional[str]) -> str:
    return value or "text"


def _as_tensor_image(value: Any) -> Tuple[torch.Tensor, Tuple[int, int]]:
    tensor = torch.as_tensor(value, dtype=torch.float)
    if tensor.dim() == 3 and tensor.shape[0] not in (1, 3):
        tensor = tensor.permute(2, 0, 1)
    h, w = int(tensor.shape[-2]), int(tensor.shape[-1])
    if tensor.numel() and tensor.max() > 1:
        tensor = tensor / 255.0
    return tensor.contiguous(), (h, w)


class TorchVisionImageProcessor:
    """Configurable image processor backed by torchvision transforms."""

    _INTERPOLATION_ALIASES = {
        "nearest": "NEAREST",
        "bilinear": "BILINEAR",
        "bicubic": "BICUBIC",
        "lanczos": "LANCZOS",
        "box": "BOX",
        "hamming": "HAMMING",
    }

    def __init__(
        self,
        image_size: Any = 224,
        resize_size: Optional[Any] = None,
        center_crop_size: Optional[Any] = None,
        interpolation: str = "bilinear",
        do_rescale: bool = True,
        do_normalize: bool = False,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
    ):
        try:
            from torchvision import transforms
            from torchvision.transforms import InterpolationMode
        except ImportError as exc:  # pragma: no cover - depends on optional env.
            raise ImportError(
                "TorchVisionImageProcessor requires torchvision. "
                "Install torchvision or set vision_processor_type='auto'."
            ) from exc

        self.image_size = pair_2d(image_size)
        self.resize_size = pair_2d(resize_size if resize_size is not None else image_size)
        self.center_crop_size = pair_2d(center_crop_size) if center_crop_size is not None else None
        self.do_rescale = bool(do_rescale)
        self.do_normalize = bool(do_normalize)
        self.image_mean = image_mean if image_mean is not None else [0.485, 0.456, 0.406]
        self.image_std = image_std if image_std is not None else [0.229, 0.224, 0.225]

        interpolation_name = self._INTERPOLATION_ALIASES.get(str(interpolation).lower(), str(interpolation).upper())
        interpolation_mode = getattr(InterpolationMode, interpolation_name)

        steps = [transforms.Resize(self.resize_size, interpolation=interpolation_mode)]
        if self.center_crop_size is not None:
            steps.append(transforms.CenterCrop(self.center_crop_size))
        if self.do_rescale:
            steps.append(transforms.ToTensor())
        else:
            steps.append(transforms.PILToTensor())
            steps.append(transforms.ConvertImageDtype(torch.float))
        if self.do_normalize:
            steps.append(transforms.Normalize(mean=self.image_mean, std=self.image_std))
        self.transform = transforms.Compose(steps)

    @classmethod
    def from_config(cls, config):
        return cls(
            image_size=getattr(config, "image_size", 224),
            resize_size=getattr(config, "vision_resize_size", None),
            center_crop_size=getattr(config, "vision_center_crop_size", None),
            interpolation=getattr(config, "vision_interpolation", "bilinear"),
            do_rescale=getattr(config, "vision_do_rescale", True),
            do_normalize=getattr(config, "vision_do_normalize", False),
            image_mean=getattr(config, "vision_image_mean", None),
            image_std=getattr(config, "vision_image_std", None),
        )

    def __call__(self, images, return_tensors: Optional[str] = None, **kwargs):
        if isinstance(images, Image.Image):
            batch = [self.transform(images.convert("RGB"))]
        elif isinstance(images, (list, tuple)):
            batch = [
                self.transform(image.convert("RGB") if isinstance(image, Image.Image) else image)
                for image in images
            ]
        else:
            tensor = torch.as_tensor(images, dtype=torch.float)
            if tensor.dim() == 3 and tensor.shape[0] not in (1, 3):
                tensor = tensor.permute(2, 0, 1)
            if self.do_rescale and tensor.numel() and tensor.max() > 1:
                tensor = tensor / 255.0
            if self.do_normalize:
                mean = tensor.new_tensor(self.image_mean).view(-1, 1, 1)
                std = tensor.new_tensor(self.image_std).view(-1, 1, 1)
                tensor = (tensor - mean) / std
            batch = [tensor.contiguous()]

        pixel_values = torch.stack(batch)
        return {"pixel_values": pixel_values} if return_tensors == "pt" else pixel_values


class TorchAudioProcessor:
    """Configurable audio processor backed by torchaudio transforms."""

    _RAW_FORMAT = "raw"
    _SPECTROGRAM_FORMAT = "spectrogram"
    _MEL_FORMAT = "mel_spectrogram"
    _LOG_MEL_FORMAT = "log_mel_spectrogram"
    _OUTPUT_FORMATS = {
        _RAW_FORMAT,
        _SPECTROGRAM_FORMAT,
        _MEL_FORMAT,
        _LOG_MEL_FORMAT,
    }

    def __init__(
        self,
        sampling_rate: Optional[int] = None,
        do_resample: bool = True,
        output_format: str = "raw",
        normalize: bool = False,
        n_fft: int = 400,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        n_mels: int = 80,
        power: float = 2.0,
    ):
        self.sampling_rate = int(sampling_rate) if sampling_rate is not None else None
        self.do_resample = bool(do_resample)
        self.output_format = str(output_format or self._RAW_FORMAT).lower().replace("-", "_")
        if self.output_format not in self._OUTPUT_FORMATS:
            raise ValueError(
                "audio_processor_output_format must be one of "
                f"{sorted(self._OUTPUT_FORMATS)}, got {self.output_format!r}."
            )
        self.normalize = bool(normalize)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length) if hop_length is not None else None
        self.win_length = int(win_length) if win_length is not None else None
        self.n_mels = int(n_mels)
        self.power = float(power)
        self.log_mel = self.output_format == self._LOG_MEL_FORMAT

        self._torchaudio = None
        self._feature_transform = None
        needs_torchaudio = (
            self.sampling_rate is not None and self.do_resample
        ) or self.output_format in {self._SPECTROGRAM_FORMAT, self._MEL_FORMAT, self._LOG_MEL_FORMAT}
        if needs_torchaudio:
            try:
                import torchaudio
            except ImportError as exc:  # pragma: no cover - depends on optional env.
                raise ImportError(
                    "TorchAudioProcessor requires torchaudio for resampling or feature extraction. "
                    "Install torchaudio or use audio_processor_type='auto'."
                ) from exc
            self._torchaudio = torchaudio

    @classmethod
    def from_config(cls, config):
        return cls(
            sampling_rate=getattr(config, "audio_sampling_rate", None),
            do_resample=getattr(config, "audio_do_resample", True),
            output_format=getattr(config, "audio_processor_output_format", "raw"),
            normalize=getattr(config, "audio_do_normalize", False),
            n_fft=getattr(config, "audio_n_fft", 400),
            hop_length=getattr(config, "audio_hop_length", None),
            win_length=getattr(config, "audio_win_length", None),
            n_mels=getattr(config, "audio_n_mels", 80),
            power=getattr(config, "audio_power", 2.0),
        )

    @staticmethod
    def _to_mono_waveform(audio) -> torch.Tensor:
        tensor = torch.as_tensor(audio, dtype=torch.float)
        if tensor.dim() == 0:
            tensor = tensor.view(1)
        if tensor.dim() == 2:
            # Accept either (channels, time) or (time, channels).
            if tensor.shape[0] <= tensor.shape[1]:
                tensor = tensor.mean(dim=0)
            else:
                tensor = tensor.mean(dim=1)
        elif tensor.dim() > 2:
            tensor = tensor.reshape(-1)
        return tensor.contiguous()

    def _maybe_resample(self, waveform: torch.Tensor, sampling_rate: Optional[int]) -> Tuple[torch.Tensor, Optional[int]]:
        if self.sampling_rate is None:
            return waveform, sampling_rate
        if sampling_rate is None or int(sampling_rate) == self.sampling_rate:
            return waveform, self.sampling_rate
        if not self.do_resample:
            return waveform, int(sampling_rate)
        waveform = self._torchaudio.functional.resample(
            waveform,
            orig_freq=int(sampling_rate),
            new_freq=self.sampling_rate,
        )
        return waveform, self.sampling_rate

    def _maybe_normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.normalize or waveform.numel() == 0:
            return waveform
        mean = waveform.mean()
        std = waveform.std().clamp_min(1e-6)
        return (waveform - mean) / std

    def _feature_transform_for(self, sampling_rate: Optional[int]):
        if self.output_format == self._RAW_FORMAT:
            return None
        if self._feature_transform is not None:
            return self._feature_transform
        sample_rate = int(sampling_rate or self.sampling_rate or 16000)
        if self.output_format == self._SPECTROGRAM_FORMAT:
            self._feature_transform = self._torchaudio.transforms.Spectrogram(
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                power=self.power,
            )
        elif self.output_format in {self._MEL_FORMAT, self._LOG_MEL_FORMAT}:
            self._feature_transform = self._torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                power=self.power,
            )
        return self._feature_transform

    def __call__(self, audio, sampling_rate: Optional[int] = None, return_tensors: Optional[str] = None, **kwargs):
        waveform = self._to_mono_waveform(audio)
        waveform, effective_rate = self._maybe_resample(waveform, sampling_rate)
        waveform = self._maybe_normalize(waveform)

        transform = self._feature_transform_for(effective_rate)
        values = waveform if transform is None else transform(waveform)
        if transform is not None and self.log_mel:
            values = torch.log(values.clamp_min(1e-10))

        if return_tensors == "pt":
            return {"audio_values": values.unsqueeze(0)}
        return values


class TextProcessingMixin:
    """Text tokenization and word-level preprocessing."""

    @staticmethod
    def _as_tokenized_mapping(tokenized_inputs):
        """Return a mutable mapping for tokenizer outputs.

        Hugging Face tokenizers normally return ``BatchEncoding``, but small
        tokenizer adapters may expose only the mapping protocol methods used
        by the processor.  Normalize those adapters before task labels and
        modality tensors are attached so collators never depend on an
        object's concrete ``items()`` implementation.
        """
        if isinstance(tokenized_inputs, Mapping):
            return tokenized_inputs

        try:
            normalized = dict(tokenized_inputs)
        except (TypeError, ValueError):
            normalized = {}

        # Some lightweight adapters implement subscription and containment
        # without implementing iteration.  These are the canonical tokenizer
        # fields that can be recovered from such an adapter.
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            if key in normalized:
                continue
            try:
                if key in tokenized_inputs:
                    normalized[key] = tokenized_inputs[key]
            except (KeyError, TypeError):
                continue
        return normalized

    def _preprocess_batch_text(self, batch_list, classes_mapping):
        extraction_processor = self._extraction_task_processor()
        if extraction_processor is not None:
            return [
                extraction_processor.preprocess_example(
                    item, classes_mapping.extraction_mapping[i],
                )
                for i, item in enumerate(batch_list)
            ]
        return [self._preprocess_text_only(item) for item in batch_list]

    def tokenize_inputs(self, texts, classes_mapping, **kwargs):
        input_texts, prompt_lengths = self.prepare_inputs(texts, classes_mapping, **kwargs)

        # ``max_len`` is also the encoder's sequence budget.  Capping the
        # word list alone is insufficient: a single word can expand to many
        # subtokens, and some saved tokenizers advertise an effectively
        # unlimited ``model_max_length``.  In that case ``truncation=True``
        # without an explicit limit leaves the transformer input unbounded.
        tokenizer_output = self.transformer_tokenizer(
            input_texts,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=int(self.config.max_len),
            padding="longest",
        )
        words_masks = self.prepare_word_mask(texts, tokenizer_output, prompt_lengths)
        tokenized_inputs = self._as_tokenized_mapping(tokenizer_output)
        words_mask = torch.tensor(words_masks)
        tokenized_inputs["words_mask"] = words_mask
        # Prompt and source share the transformer's token budget. The largest
        # 1-based source-word id is therefore the exact number of source words
        # that survived subtoken truncation (and is zero for prompt-only rows).
        tokenized_inputs["text_lengths"] = (
            words_mask.amax(dim=-1, keepdim=True).long()
            if words_mask.shape[-1] > 0
            else torch.zeros(
                words_mask.shape[0],
                1,
                dtype=torch.long,
                device=words_mask.device,
            )
        )

        label_enc = self.prepare_all_label_encoder_inputs(classes_mapping)
        tokenized_inputs.update(label_enc)
        return tokenized_inputs


class VisionProcessingMixin:
    """Image loading and configurable image processor integration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_vision_processor()

    def _init_vision_processor(self):
        self.vision_input_processor = getattr(self.config, "vision_processor", None)
        self._vision_processor_initialized = self.vision_input_processor is not None
        if self._vision_processor_initialized or not self._has_vision_inputs():
            return
        self._build_vision_processor()

    def _build_vision_processor(self):
        processor_name = getattr(self.config, "vision_processor_name", None)
        processor_mode = str(getattr(self.config, "vision_processor_type", "custom") or "custom").lower()
        if self.vision_input_processor is None and processor_mode == "auto":
            source = processor_name or getattr(self.config, "vision_model_name", None)
            if source is None:
                raise ValueError("vision_processor_type='auto' requires vision_processor_name or vision_model_name.")
            try:
                self.vision_input_processor = AutoProcessor.from_pretrained(source)
            except ValueError:
                self.vision_input_processor = AutoImageProcessor.from_pretrained(source)
        elif self.vision_input_processor is None:
            self.vision_input_processor = TorchVisionImageProcessor.from_config(self.config)
        self._vision_processor_initialized = True

    def _ensure_vision_processor(self):
        if not getattr(self, "_vision_processor_initialized", False):
            self._build_vision_processor()

    def _has_vision_inputs(self) -> bool:
        return any(name in self.task_processors for name in _VISION_TASKS)

    def _open_image(self, item: Dict[str, Any]) -> Tuple[Image.Image, Tuple[int, int]]:
        path = item.get("image")
        if path is None:
            raise ValueError("Vision tasks require each item to contain 'image' or 'pixel_values'.")
        image = Image.open(path).convert("RGB")
        return image, (image.height, image.width)

    def _apply_vision_processor(self, image: Image.Image, orig_size: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        self._ensure_vision_processor()
        processor = getattr(self, "vision_input_processor", None)
        if callable(processor):
            try:
                output = processor(images=image, return_tensors="pt")
            except TypeError:
                output = processor(image)
        else:
            output = processor

        if isinstance(output, Mapping) or hasattr(output, "get"):
            pixel_values = output.get("pixel_values")
            image_sizes = output.get("image_sizes")
            if image_sizes is None:
                image_sizes = output.get("original_sizes")
        else:
            pixel_values = output
            image_sizes = None
        tensor = torch.as_tensor(pixel_values, dtype=torch.float)
        if tensor.dim() == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 3 and tensor.shape[0] not in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        if image_sizes is not None:
            image_size = torch.as_tensor(image_sizes).view(-1).tolist()
            if len(image_size) >= 2:
                orig_size = (int(image_size[0]), int(image_size[1]))
        return tensor.contiguous(), orig_size

    def load_image(self, item: Dict[str, Any], image_size: Optional[int] = None) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if "pixel_values" in item:
            return _as_tensor_image(item["pixel_values"])
        image, orig_size = self._open_image(item)
        return self._apply_vision_processor(image, orig_size)

    def _prepare_vision_batch(self, batch_list):
        if not self._has_vision_inputs():
            return {}
        image_tensors = []
        image_sizes = []
        for item in batch_list:
            tensor, image_size = self.load_image(item, getattr(self.config, "image_size", None))
            item["_image_size"] = image_size
            image_tensors.append(tensor)
            image_sizes.append(image_size)
        shapes = {tuple(tensor.shape) for tensor in image_tensors}
        if len(shapes) > 1:
            raise ValueError(
                "Vision batches require one processed tensor shape. Configure "
                "vision resizing instead of padding differently sized images, "
                "which would misalign normalized detection coordinates."
            )
        return {
            "pixel_values": torch.stack(image_tensors),
            "image_sizes": torch.tensor(image_sizes, dtype=torch.long),
        }

    def _augment_label_item(self, item, batch, batch_idx):
        super()._augment_label_item(item, batch, batch_idx)
        if "image" in batch and batch_idx < len(batch["image"]):
            item["image"] = batch["image"][batch_idx]
        if batch.get("image_sizes") is not None:
            item["_image_size"] = tuple(int(v) for v in batch["image_sizes"][batch_idx].tolist())


class AudioProcessingMixin:
    """Audio loading, feature handling, and batch padding."""

    feature_encoder_types = {"mel", "spectrogram", "conv2d", "mel_conv", "spectrogram_conv", "conv_2d"}
    duration_metadata_key = "_audio_duration_seconds"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_audio_processor()

    def _init_audio_processor(self):
        self.audio_input_processor = getattr(self.config, "audio_processor", None)
        processor_name = getattr(self.config, "audio_processor_name", None)
        processor_mode = str(getattr(self.config, "audio_processor_type", "custom") or "custom").lower()
        if self.audio_input_processor is None and processor_mode == "auto":
            source = processor_name or getattr(self.config, "audio_model_name", None)
            if source is None:
                raise ValueError("audio_processor_type='auto' requires audio_processor_name or audio_model_name.")
            self.audio_input_processor = AutoProcessor.from_pretrained(source)
        elif self.audio_input_processor is None:
            self.audio_input_processor = TorchAudioProcessor.from_config(self.config)

    def _has_audio_inputs(self) -> bool:
        return any(name in self.task_processors for name in _AUDIO_TASKS)

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

    @staticmethod
    def _positive_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            value = float(value)
        except (OverflowError, TypeError, ValueError):
            return None
        return value if math.isfinite(value) and value > 0 else None

    def _metadata_value(self, item: Dict[str, Any], *keys: str) -> Optional[float]:
        for key in keys:
            value = self._positive_float(item.get(key))
            if value is not None:
                return value
        return None

    def _preserve_audio_duration(
        self,
        item: Dict[str, Any],
        *,
        time_length: Optional[int] = None,
        is_feature: bool = False,
        source_num_samples: Optional[int] = None,
        source_sample_rate: Optional[int] = None,
    ) -> Optional[float]:
        """Store duration before audio processing changes the time axis."""
        duration = self._metadata_value(
            item,
            "audio_duration",
            "duration",
            self.duration_metadata_key,
        )
        sample_rate = self._positive_float(source_sample_rate) or self._metadata_value(
            item,
            "sample_rate",
            "sampling_rate",
            "audio_sample_rate",
        ) or self._positive_float(getattr(self.config, "audio_sampling_rate", None))

        if duration is None:
            num_samples = self._positive_float(source_num_samples) or self._metadata_value(
                item,
                "audio_num_samples",
                "num_samples",
            )
            if num_samples is not None and sample_rate is not None:
                duration = num_samples / sample_rate

        if duration is None and time_length is not None:
            if not is_feature and sample_rate is not None:
                duration = float(time_length) / sample_rate
            elif is_feature:
                frame_rate = self._metadata_value(
                    item,
                    "audio_frame_rate",
                    "feature_frame_rate",
                    "frame_rate",
                    "frames_per_second",
                )
                if frame_rate is not None:
                    duration = float(time_length) / frame_rate
                else:
                    hop_length = self._metadata_value(
                        item,
                        "audio_hop_length",
                        "feature_hop_length",
                        "hop_length",
                    ) or self._positive_float(getattr(self.config, "audio_hop_length", None))
                    if hop_length is not None and sample_rate is not None:
                        duration = float(time_length) * hop_length / sample_rate

        duration = self._positive_float(duration)
        if duration is not None:
            item[self.duration_metadata_key] = duration
        return duration

    @staticmethod
    def _read_wav(path: Path) -> Tuple[np.ndarray, int]:
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
        return audio, sample_rate

    def _apply_audio_processor(self, audio: np.ndarray, sample_rate: int) -> torch.Tensor:
        processor = getattr(self, "audio_input_processor", None)
        if processor is None:
            return torch.from_numpy(audio.copy())
        try:
            output = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        except TypeError:
            output = processor(audio)
        if isinstance(output, dict):
            values = output.get("audio_values")
            if values is None:
                values = output.get("input_values")
            if values is None:
                values = output.get("input_features")
            if values is None:
                values = output.get("features")
        else:
            values = output
        tensor = torch.as_tensor(values, dtype=torch.float)
        if tensor.dim() > 1 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return tensor

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
            is_feature = self._expects_feature_input(item)
            source_time_length = self._time_length(tensor)
            self._preserve_audio_duration(
                item,
                time_length=source_time_length,
                is_feature=is_feature,
            )
            if tensor.dim() > 1 and not is_feature:
                tensor = tensor.reshape(-1)
            return tensor, self._time_length(tensor)

        path = item.get("audio")
        if path is None:
            raise ValueError("Audio tasks require each item to contain 'audio' or 'audio_values'.")
        path = Path(path)
        if path.suffix.lower() != ".wav":
            raise ValueError("Only WAV files are supported without an external audio backend.")
        audio, sample_rate = self._read_wav(path)
        item.setdefault("sample_rate", sample_rate)
        self._preserve_audio_duration(
            item,
            source_num_samples=int(audio.size),
            source_sample_rate=sample_rate,
        )
        tensor = self._apply_audio_processor(audio, sample_rate)
        return tensor, self._time_length(tensor)

    def _prepare_audio_batch(self, batch_list):
        if not self._has_audio_inputs():
            return {}
        audio_tensors = []
        audio_durations = []
        max_audio_shape = None
        max_audio_len = 0
        for item in batch_list:
            tensor, time_length = self.load_audio(item)
            item["_audio_num_samples"] = time_length
            audio_tensors.append(tensor)
            audio_durations.append(item.get(self.duration_metadata_key))
            shape = tuple(tensor.shape)
            if max_audio_shape is None:
                max_audio_shape = list(shape)
            elif len(shape) != len(max_audio_shape):
                raise ValueError("All audio tensors in a batch must have the same rank.")
            else:
                max_audio_shape = [max(current, int(dim)) for current, dim in zip(max_audio_shape, shape)]
            max_audio_len = max(max_audio_len, time_length)

        audio_values = torch.zeros((len(audio_tensors), *max_audio_shape), dtype=torch.float)
        audio_attention_mask = torch.zeros(len(audio_tensors), max_audio_len, dtype=torch.long)
        for i, tensor in enumerate(audio_tensors):
            slices = (i, *[slice(0, int(dim)) for dim in tensor.shape])
            audio_values[slices] = tensor
            audio_attention_mask[i, :self._time_length(tensor)] = 1
        return {
            "audio_values": audio_values,
            "audio_attention_mask": audio_attention_mask,
            self.duration_metadata_key: audio_durations,
        }

    def _augment_label_item(self, item, batch, batch_idx):
        super()._augment_label_item(item, batch, batch_idx)
        if batch.get("audio_values") is not None:
            item["audio_values"] = batch["audio_values"][batch_idx]
        if batch.get("audio_attention_mask") is not None:
            item["_audio_num_samples"] = int(batch["audio_attention_mask"][batch_idx].sum().item())
        durations = batch.get(self.duration_metadata_key)
        if (
            durations is not None
            and batch_idx < len(durations)
            and durations[batch_idx] is not None
        ):
            item[self.duration_metadata_key] = durations[batch_idx]


class LayoutProcessingMixin:
    """Layout word extraction and bbox alignment to tokenizer word IDs."""

    @staticmethod
    def _normalize_bbox_value(bbox) -> List[int]:
        values = torch.as_tensor(bbox, dtype=torch.long).view(-1)
        if values.numel() != 4:
            raise ValueError(f"layout bbox must contain 4 coordinates, got {bbox!r}")
        return [int(v) for v in values.tolist()]

    @staticmethod
    def _normalize_page_id_value(page_id) -> int:
        return int(torch.as_tensor(page_id, dtype=torch.long).view(-1)[0].item())

    @classmethod
    def _page_offsets_to_page_ids(cls, page_offsets, length: int) -> Optional[List[int]]:
        if page_offsets is None:
            return None
        entries = []
        for idx, entry in enumerate(page_offsets):
            if isinstance(entry, dict):
                offset = entry.get("offset", entry.get("start", entry.get("token_start")))
                page_id = entry.get("page_id", entry.get("page", idx))
            elif isinstance(entry, (list, tuple)):
                if len(entry) == 1:
                    offset = entry[0]
                    page_id = idx
                elif len(entry) >= 2:
                    page_id, offset = entry[0], entry[1]
                else:
                    continue
            else:
                offset = entry
                page_id = idx
            if offset is None:
                raise ValueError(f"Layout page_offsets entry is missing an offset: {entry!r}")
            offset = int(offset)
            if offset < 0 or offset > length:
                raise ValueError(
                    f"Layout page offset {offset} is outside tokenized_text length {length}."
                )
            entries.append((offset, cls._normalize_page_id_value(page_id)))
        if not entries:
            return None

        entries = sorted(entries, key=lambda item: item[0])
        if entries[0][0] != 0:
            entries.insert(0, (0, 0))

        page_ids = [0 for _ in range(length)]
        for idx, (start, page_id) in enumerate(entries):
            end = entries[idx + 1][0] if idx + 1 < len(entries) else length
            if end < start:
                raise ValueError(f"Layout page_offsets must be sorted by token offset, got {page_offsets!r}.")
            for token_idx in range(start, end):
                page_ids[token_idx] = page_id
        return page_ids

    def _default_word_page_ids(self, item: Dict[str, Any], length: int) -> Optional[List[int]]:
        page_ids = (
            item.get("word_page_ids")
            or item.get("token_page_ids")
            or item.get("page_ids")
        )
        if page_ids is not None:
            values = [self._normalize_page_id_value(value) for value in page_ids]
            if len(values) != length:
                raise ValueError(
                    f"Layout page_ids must match words length, got {len(values)} and {length}."
                )
            return values
        page_offsets = item.get("page_offsets")
        if page_offsets is not None:
            return self._page_offsets_to_page_ids(page_offsets, length)
        if item.get("page") is not None:
            return [self._normalize_page_id_value(item["page"]) for _ in range(length)]
        return None

    def _extract_layout_words_bboxes_and_pages(
        self,
        item: Dict[str, Any],
    ) -> Tuple[Optional[List[str]], Optional[List[List[int]]], Optional[List[int]]]:
        layout = item.get("layout")
        if isinstance(layout, list):
            words = []
            bboxes = []
            page_ids = []
            has_page_ids = False
            for entry in layout:
                if isinstance(entry, dict):
                    word = entry.get("word", entry.get("text", entry.get("token")))
                    bbox = entry.get("bbox", entry.get("box"))
                    page_id = entry.get("page_id", entry.get("page", entry.get("page_number")))
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    word, bbox = entry[0], entry[1]
                    page_id = entry[2] if len(entry) >= 3 else None
                else:
                    continue
                if word is None or bbox is None:
                    continue
                words.append(str(word))
                bboxes.append(self._normalize_bbox_value(bbox))
                if page_id is not None:
                    has_page_ids = True
                page_ids.append(self._normalize_page_id_value(page_id if page_id is not None else item.get("page", 0)))
            if not words:
                return None, None, None
            return words, bboxes, page_ids if has_page_ids or item.get("page") is not None else None

        words = item.get("tokenized_text")
        bboxes = item.get("bboxes") or item.get("word_bboxes")
        if bboxes is None:
            raw_bbox = item.get("bbox")
            if isinstance(raw_bbox, (list, tuple)) and raw_bbox and not isinstance(raw_bbox[0], (int, float)):
                bboxes = raw_bbox
        if words is not None and bboxes is not None:
            words = list(map(str, words))
            return (
                words,
                [self._normalize_bbox_value(b) for b in bboxes],
                self._default_word_page_ids(item, len(words)),
            )
        return None, None, None

    def _prepare_layout_batch(self, batch_list):
        word_bboxes = []
        word_page_ids = []
        layout_input_mask = []
        page_input_mask = []
        has_layout = False
        has_page_ids = False
        for item in batch_list:
            words, bboxes, page_ids = self._extract_layout_words_bboxes_and_pages(item)
            if words is None or bboxes is None:
                word_bboxes.append(None)
                word_page_ids.append(None)
                layout_input_mask.append(False)
                page_input_mask.append(False)
                continue
            if len(words) != len(bboxes):
                raise ValueError(
                    f"Layout words and bboxes must have the same length, got {len(words)} and {len(bboxes)}."
                )
            if page_ids is not None and len(words) != len(page_ids):
                raise ValueError(
                    f"Layout words and page_ids must have the same length, got {len(words)} and {len(page_ids)}."
                )
            item.setdefault("tokenized_text", words)
            item.setdefault("text", " ".join(words))
            word_bboxes.append(bboxes)
            word_page_ids.append(page_ids)
            layout_input_mask.append(True)
            page_input_mask.append(page_ids is not None)
            has_layout = True
            has_page_ids = has_page_ids or page_ids is not None
        fields = (
            {
                "word_bboxes": word_bboxes,
                "layout_input_mask": torch.tensor(layout_input_mask, dtype=torch.bool),
            }
            if has_layout
            else {}
        )
        if has_page_ids:
            fields["word_page_ids"] = word_page_ids
            fields["page_input_mask"] = torch.tensor(page_input_mask, dtype=torch.bool)
        return fields

    def _augment_label_item(self, item, batch, batch_idx):
        if batch.get("image_batch_idx") is not None:
            super(VisionProcessingMixin, self)._augment_label_item(item, batch, batch_idx)
        else:
            super()._augment_label_item(item, batch, batch_idx)
        if batch.get("word_bboxes") is not None:
            item["word_bboxes"] = batch["word_bboxes"][batch_idx]
        if batch.get("word_page_ids") is not None:
            item["word_page_ids"] = batch["word_page_ids"][batch_idx]

    def tokenize_inputs(self, texts, classes_mapping, **kwargs):
        tokenized_inputs = super().tokenize_inputs(texts, classes_mapping, **kwargs)
        word_bboxes = kwargs.get("word_bboxes")
        word_page_ids = kwargs.get("word_page_ids")
        layout_input_mask = kwargs.get("layout_input_mask")
        page_input_mask = kwargs.get("page_input_mask")
        prompt_lengths = kwargs.get("prompt_lengths")
        if word_bboxes is None and word_page_ids is None:
            return tokenized_inputs

        input_ids = tokenized_inputs["input_ids"]
        bbox = torch.zeros((*input_ids.shape, 4), dtype=torch.long) if word_bboxes is not None else None
        page_token_ids = torch.zeros(input_ids.shape, dtype=torch.long) if word_page_ids is not None else None
        if prompt_lengths is None:
            _, prompt_lengths = self.prepare_inputs(texts, classes_mapping)

        for batch_idx in range(input_ids.shape[0]):
            boxes = word_bboxes[batch_idx] if word_bboxes is not None else None
            page_ids = word_page_ids[batch_idx] if word_page_ids is not None else None
            if not boxes and not page_ids:
                continue
            try:
                word_ids = tokenized_inputs.word_ids(batch_index=batch_idx)
            except Exception:
                word_ids = list(range(input_ids.shape[1]))
            prompt_len = int(prompt_lengths[batch_idx])
            for token_idx, word_id in enumerate(word_ids[: input_ids.shape[1]]):
                if word_id is None or word_id < prompt_len:
                    continue
                source_idx = int(word_id) - prompt_len
                if bbox is not None and boxes is not None and 0 <= source_idx < len(boxes):
                    bbox[batch_idx, token_idx] = torch.tensor(boxes[source_idx], dtype=torch.long)
                if page_token_ids is not None and page_ids is not None and 0 <= source_idx < len(page_ids):
                    page_token_ids[batch_idx, token_idx] = int(page_ids[source_idx])
        if bbox is not None:
            tokenized_inputs["bbox"] = bbox
            if layout_input_mask is None:
                layout_input_mask = [boxes is not None for boxes in word_bboxes]
            layout_input_mask = torch.as_tensor(layout_input_mask, dtype=torch.bool)
            if layout_input_mask.shape != (input_ids.shape[0],):
                raise ValueError(
                    "layout_input_mask must have shape (batch,), "
                    f"got {tuple(layout_input_mask.shape)} for batch size {input_ids.shape[0]}"
                )
            tokenized_inputs["layout_input_mask"] = layout_input_mask
        if page_token_ids is not None:
            tokenized_inputs["page_token_ids"] = page_token_ids
            if page_input_mask is None:
                page_input_mask = [page_ids is not None for page_ids in word_page_ids]
            page_input_mask = torch.as_tensor(page_input_mask, dtype=torch.bool)
            if page_input_mask.shape != (input_ids.shape[0],):
                raise ValueError(
                    "page_input_mask must have shape (batch,), "
                    f"got {tuple(page_input_mask.shape)} for batch size {input_ids.shape[0]}"
                )
            tokenized_inputs["page_input_mask"] = page_input_mask
        return tokenized_inputs


class BaseGLiNextProcessor(TextProcessingMixin, BaseProcessor):
    processor_task_names = _ALL_TASKS

    def __init__(self, config, tokenizer, words_splitter,
                 labels_tokenizer: Optional[object] = None):
        super().__init__(config, tokenizer, words_splitter)
        self.labels_tokenizer = labels_tokenizer

        self.seq_token = config.seq_token
        self.cat_token = config.cat_token
        self.ent_token = config.ent_token
        self.sep_token = config.sep_token
        self.rel_token = config.rel_token
        self.parent_token = config.parent_token
        self.child_token = config.child_token

        # ── Register per-task processors ────────────────────────────────
        self.task_processors: Dict[str, object] = {}
        allowed_tasks = set(self.processor_task_names)

        if "ner" in allowed_tasks and config.ner_config is not None:
            self.task_processors["ner"] = NERProcessor(config, tokenizer, words_splitter)
        if "classification" in allowed_tasks and config.classification_config is not None:
            self.task_processors["classification"] = ClassificationProcessor(config)
        if "joint_relex" in allowed_tasks and config.joint_relex_config is not None:
            self.task_processors["joint_relex"] = JointRelexProcessor(config, tokenizer, words_splitter)
        if "open_relex" in allowed_tasks and config.open_relex_config is not None:
            self.task_processors["open_relex"] = OpenRelexProcessor(config, tokenizer, words_splitter)
        if (
            "set_open_relex" in allowed_tasks
            and getattr(config, "set_open_relex_config", None) is not None
        ):
            self.task_processors["set_open_relex"] = SetOpenRelexProcessor(
                config, tokenizer, words_splitter,
            )
        if "count" in allowed_tasks and config.count_config is not None:
            self.task_processors["count"] = CountProcessor(config)
        if "structuring" in allowed_tasks and config.structuring_config is not None:
            self.task_processors["structuring"] = StructuringProcessor(config, tokenizer, words_splitter)
        if (
            "set_structuring" in allowed_tasks
            and config.set_structuring_config is not None
        ):
            self.task_processors["set_structuring"] = SetStructuringProcessor(
                config, tokenizer, words_splitter,
            )
        if "embedding" in allowed_tasks and config.embedding_config is not None:
            self.task_processors["embedding"] = EmbeddingProcessor(config)
        if "image_classification" in allowed_tasks and config.image_classification_config is not None:
            self.task_processors["image_classification"] = VisionProcessor(config, "image_classification")
        if "object_detection" in allowed_tasks and config.object_detection_config is not None:
            self.task_processors["object_detection"] = VisionProcessor(config, "object_detection")
        if "segmentation" in allowed_tasks and config.segmentation_config is not None:
            self.task_processors["segmentation"] = VisionProcessor(config, "segmentation")
        if "audio_classification" in allowed_tasks and config.audio_classification_config is not None:
            self.task_processors["audio_classification"] = AudioProcessor(config, "audio_classification")
        if "audio_segmentation" in allowed_tasks and config.audio_segmentation_config is not None:
            self.task_processors["audio_segmentation"] = AudioProcessor(config, "audio_segmentation")

    def _extraction_task_processor(self):
        """Return the single processor responsible for joint NER prompts."""
        return self.task_processors.get("ner") or self.task_processors.get(
            "joint_relex"
        )

    # ── Class mappings ──────────────────────────────────────────────────

    def batch_generate_class_mappings(self, batch_list, **kwargs):
        cat_mapping = []
        extraction_mapping = []
        structuring_mapping = []
        set_structuring_mapping = []
        open_relex_mapping = []
        set_open_relex_mapping = []
        image_classification_mapping = []
        audio_classification_mapping = []
        object_detection_mapping = []
        segmentation_mapping = []
        audio_segmentation_mapping = []

        if "classification" in self.task_processors:
            cat_mapping = self.task_processors["classification"].get_classes_mapping(
                batch_list, **kwargs
            )
        else:
            cat_mapping = [CatClassMapping(cat_class_to_id=[]) for _ in batch_list]

        extraction_processor = self._extraction_task_processor()
        if extraction_processor is not None:
            extraction_mapping = extraction_processor.get_classes_mapping(
                batch_list, **kwargs
            )
        else:
            extraction_mapping = [ExtractionClassMapping() for _ in batch_list]

        if "structuring" in self.task_processors:
            structuring_mapping = self.task_processors["structuring"].get_classes_mapping(
                batch_list, **kwargs
            )
        else:
            structuring_mapping = [StructuringClassMapping() for _ in batch_list]

        if "set_structuring" in self.task_processors:
            set_structuring_mapping = self.task_processors[
                "set_structuring"
            ].get_classes_mapping(batch_list, **kwargs)
        else:
            set_structuring_mapping = [
                StructuringClassMapping() for _ in batch_list
            ]

        if "open_relex" in self.task_processors:
            open_relex_mapping = self.task_processors["open_relex"].get_classes_mapping(
                batch_list, **kwargs
            )
        else:
            open_relex_mapping = [OpenRelexClassMapping() for _ in batch_list]

        if "set_open_relex" in self.task_processors:
            set_open_relex_mapping = self.task_processors[
                "set_open_relex"
            ].get_classes_mapping(batch_list, **kwargs)
        else:
            set_open_relex_mapping = [
                OpenRelexClassMapping() for _ in batch_list
            ]

        if "image_classification" in self.task_processors:
            image_classification_mapping = self.task_processors["image_classification"].get_classes_mapping(
                batch_list, **kwargs
            )
        else:
            image_classification_mapping = [VisionClassMapping() for _ in batch_list]

        if "audio_classification" in self.task_processors:
            audio_classification_mapping = self.task_processors["audio_classification"].get_classes_mapping(
                batch_list, **kwargs
            )
        else:
            audio_classification_mapping = [VisionClassMapping() for _ in batch_list]

        if "object_detection" in self.task_processors:
            object_detection_mapping = self.task_processors["object_detection"].get_classes_mapping(
                batch_list, **kwargs
            )
        else:
            object_detection_mapping = [VisionClassMapping() for _ in batch_list]

        if "segmentation" in self.task_processors:
            segmentation_mapping = self.task_processors["segmentation"].get_classes_mapping(
                batch_list, **kwargs
            )
        else:
            segmentation_mapping = [VisionClassMapping() for _ in batch_list]

        if "audio_segmentation" in self.task_processors:
            audio_segmentation_mapping = self.task_processors["audio_segmentation"].get_classes_mapping(
                batch_list, **kwargs
            )
        else:
            audio_segmentation_mapping = [VisionClassMapping() for _ in batch_list]

        classes_mapping = BatchClassesMapping(
            cat_mapping=cat_mapping,
            extraction_mapping=extraction_mapping,
            structuring_mapping=structuring_mapping,
            set_structuring_mapping=set_structuring_mapping,
            open_relex_mapping=open_relex_mapping,
            set_open_relex_mapping=set_open_relex_mapping,
            image_classification_mapping=image_classification_mapping,
            audio_classification_mapping=audio_classification_mapping,
            object_detection_mapping=object_detection_mapping,
            segmentation_mapping=segmentation_mapping,
            audio_segmentation_mapping=audio_segmentation_mapping,
        )

        # Label-schema augmentation is deliberately applied to the ephemeral
        # mappings, not to dataset annotations.  It must happen before text
        # preprocessing because NER preprocessing materializes class ids from
        # these mappings for span-level supervision.
        label_augmenter = kwargs.get("label_augmenter")
        if label_augmenter is not None:
            augmentable_groups = []
            for task_processor in self.task_processors.values():
                augmentable_groups.extend(
                    task_processor.get_augmentable_label_groups(
                        batch_list,
                        classes_mapping,
                    )
                )
            self.last_label_augmentation_stats = label_augmenter.augment(
                augmentable_groups,
                batch_ids=kwargs.get("label_augmentation_batch_ids"),
            )

        return classes_mapping

    # ── Prompt construction ─────────────────────────────────────────────

    def prepare_inputs(self, texts, classes_mapping, blank=None, add_entities=True, **kwargs):
        use_labels_encoder = self.labels_tokenizer is not None
        input_texts = []
        prompt_lengths = []

        # Fixed ordering: classification -> extraction (NER+REL) -> structuring
        for i, text in enumerate(texts):
            prompt = [self.seq_token]

            if "classification" in self.task_processors:
                prompt.extend(self.task_processors["classification"].contribute_prompt(
                    classes_mapping, i, use_labels_encoder,
                ))

            extraction_processor = self._extraction_task_processor()
            if extraction_processor is not None:
                prompt.extend(extraction_processor.contribute_prompt(
                    classes_mapping, i, use_labels_encoder,
                ))

            if "open_relex" in self.task_processors:
                prompt.extend(self.task_processors["open_relex"].contribute_prompt(
                    classes_mapping, i, use_labels_encoder,
                ))

            if "set_open_relex" in self.task_processors:
                prompt.extend(self.task_processors[
                    "set_open_relex"
                ].contribute_prompt(
                    classes_mapping, i, use_labels_encoder,
                ))

            if "structuring" in self.task_processors:
                prompt.extend(self.task_processors["structuring"].contribute_prompt(
                    classes_mapping, i, use_labels_encoder,
                ))

            if "set_structuring" in self.task_processors:
                prompt.extend(self.task_processors[
                    "set_structuring"
                ].contribute_prompt(
                    classes_mapping, i, use_labels_encoder,
                ))

            for name in (
                "image_classification", "object_detection", "segmentation",
                "audio_classification", "audio_segmentation",
            ):
                if name in self.task_processors:
                    prompt.extend(self.task_processors[name].contribute_prompt(
                        classes_mapping, i, use_labels_encoder,
                    ))

            prompt.append(self.sep_token)
            prompt_lengths.append(len(prompt))
            input_texts.append(prompt + list(text))

        return input_texts, prompt_lengths

    # ── Labels encoder inputs ───────────────────────────────────────────

    def prepare_all_label_encoder_inputs(self, classes_mapping):
        """Collect label encoder inputs from all task processors."""
        if self.labels_tokenizer is None:
            return {}

        result = {}
        for name, proc in self.task_processors.items():
            enc = proc.prepare_label_encoder_inputs(classes_mapping, self.labels_tokenizer)
            if enc is not None:
                result.update(enc)
        return result

    def prepare_task_label_encoder_inputs(self, classes_mapping, task_names):
        """Collect label encoder inputs for selected tasks only."""
        labels_tokenizer = self.labels_tokenizer or self.transformer_tokenizer
        result = {}
        for name in task_names:
            proc = self.task_processors.get(name)
            if proc is None:
                continue
            enc = proc.prepare_label_encoder_inputs(classes_mapping, labels_tokenizer)
            if enc is not None:
                result.update(enc)
        return result

    def create_task_labels(self, batch_list, classes_mapping, task_names, max_seq_len=0):
        """Create labels for selected tasks only."""
        labels = {}
        for name in task_names:
            proc = self.task_processors.get(name)
            if proc is None:
                continue
            if (
                name == "joint_relex"
                and "ner" not in self.task_processors
                and hasattr(proc, "create_ner_labels")
            ):
                ner_result = proc.create_ner_labels(
                    batch_list,
                    classes_mapping,
                    max_seq_len=max_seq_len,
                )
                if ner_result is not None:
                    labels.update(ner_result)
            result = proc.create_labels(
                batch_list,
                classes_mapping,
                max_seq_len=max_seq_len,
            )
            if result is not None:
                labels.update(result)
        return labels

    def _augment_label_item(self, item, batch, batch_idx):
        """Hook for processors that need to add per-item label context."""
        return None

    def empty_inference_results(self, num_texts: int, **kwargs):
        results = {}
        for proc in self.task_processors.values():
            empty = proc.empty_inference_result(num_texts, **kwargs)
            if empty:
                results.update(empty)
        return results

    # ── Tokenization ────────────────────────────────────────────────────

    def tokenize_inputs(self, texts, classes_mapping, **kwargs):
        return TextProcessingMixin.tokenize_inputs(self, texts, classes_mapping, **kwargs)

    # ── Span resolution (delegates to task processors) ──────────────────

    def resolve_extraction_spans(self, item):
        extraction_processor = self._extraction_task_processor()
        if extraction_processor is not None:
            extraction_processor.resolve_spans(item)
        return item

    def resolve_structuring_spans(self, item):
        if "structuring" in self.task_processors:
            self.task_processors["structuring"].resolve_spans(item)
        return item

    def resolve_set_structuring_spans(self, item):
        if "set_structuring" in self.task_processors:
            self.task_processors["set_structuring"].resolve_spans(item)
        return item

    def resolve_open_relex_spans(self, item):
        if "open_relex" in self.task_processors:
            self.task_processors["open_relex"].resolve_spans(item)
        return item

    def resolve_set_open_relex_spans(self, item):
        if "set_open_relex" in self.task_processors:
            self.task_processors["set_open_relex"].resolve_spans(item)
        return item

    # ── Generic label creation ────────────────────────────────────────────

    def create_labels(self, batch_list, classes_mapping=None, max_seq_len=0):
        if classes_mapping is None:
            raise ValueError("classes_mapping is required; use create_all_labels for raw batches.")
        return self.create_all_labels(
            batch_list, classes_mapping, max_seq_len=max_seq_len,
        )

    def create_all_labels(self, batch_list, classes_mapping, max_seq_len=0):
        """Create labels from all task processors in a single call.

        Returns a flat dict of all label tensors from all active tasks.
        """
        for item in batch_list:
            for proc in self.task_processors.values():
                if hasattr(proc, "resolve_spans"):
                    proc.resolve_spans(item)

        return self.create_task_labels(
            batch_list,
            classes_mapping,
            tuple(self.task_processors),
            max_seq_len=max_seq_len,
        )

    # ── Label creation (delegates to task processors) ───────────────────

    def create_cat_labels(self, batch_list, classes_mapping):
        if "classification" in self.task_processors:
            result = self.task_processors["classification"].create_labels(batch_list, classes_mapping)
            if result is not None:
                return result["cat_labels"], result["cat_batch_idx"]
        return None

    def create_ner_labels(self, batch_list, classes_mapping, max_seq_len):
        for item in batch_list:
            self.resolve_extraction_spans(item)
        extraction_processor = self._extraction_task_processor()
        if extraction_processor is not None:
            if hasattr(extraction_processor, "create_ner_labels"):
                result = extraction_processor.create_ner_labels(
                    batch_list,
                    classes_mapping,
                    max_seq_len=max_seq_len,
                )
            else:
                result = extraction_processor.create_labels(
                    batch_list,
                    classes_mapping,
                    max_seq_len=max_seq_len,
                )
            if result is not None:
                return result["ner_labels"], result["ner_batch_idx"]
        return None

    def create_rel_labels(
        self,
        batch_list,
        classes_mapping,
        max_seq_len=0,
        sequence_lengths=None,
    ):
        return self.create_joint_rel_labels(
            batch_list,
            classes_mapping,
            max_seq_len=max_seq_len,
            sequence_lengths=sequence_lengths,
        )

    def create_joint_rel_labels(
        self,
        batch_list,
        classes_mapping,
        max_seq_len=0,
        sequence_lengths=None,
    ):
        for item in batch_list:
            self.resolve_extraction_spans(item)
        if "joint_relex" in self.task_processors:
            result = self.task_processors["joint_relex"].create_labels(
                batch_list,
                classes_mapping,
                max_seq_len=max_seq_len,
                sequence_lengths=sequence_lengths,
            )
            if result is not None:
                return result
        return None

    def create_open_rel_labels(self, batch_list, classes_mapping, max_seq_len):
        for item in batch_list:
            self.resolve_open_relex_spans(item)
        if "open_relex" in self.task_processors:
            return self.task_processors["open_relex"].create_labels(
                batch_list, classes_mapping, max_seq_len=max_seq_len,
            )
        return None

    def create_set_open_rel_labels(
        self,
        batch_list,
        classes_mapping,
        max_seq_len,
    ):
        for item in batch_list:
            self.resolve_set_open_relex_spans(item)
        if "set_open_relex" in self.task_processors:
            return self.task_processors["set_open_relex"].create_labels(
                batch_list,
                classes_mapping,
                max_seq_len=max_seq_len,
            )
        return None

    def create_count_labels(self, batch_list, classes_mapping):
        if "count" in self.task_processors:
            result = self.task_processors["count"].create_labels(batch_list, classes_mapping)
            if result is not None:
                return result["count_targets"], result["count_targets"]
        return None

    def create_embedding_labels(self, batch_list):
        if "embedding" in self.task_processors:
            return self.task_processors["embedding"].create_labels(batch_list, None)
        return None

    def create_structuring_labels(
        self,
        batch_list,
        classes_mapping,
        max_seq_len,
        *,
        return_dict=False,
        sequence_lengths=None,
        source_sequence_lengths=None,
    ):
        for item in batch_list:
            self.resolve_structuring_spans(item)
        if "structuring" in self.task_processors:
            result = self.task_processors["structuring"].create_labels(
                batch_list,
                classes_mapping,
                max_seq_len=max_seq_len,
                sequence_lengths=sequence_lengths,
                source_sequence_lengths=source_sequence_lengths,
            )
            if result is not None:
                if return_dict:
                    return result
                return (result["structuring_labels"], result["structuring_mask"],
                        result["structuring_batch_idx"], result["structuring_count"])
        return None

    def create_set_structuring_labels(
        self,
        batch_list,
        classes_mapping,
        max_seq_len,
        *,
        sequence_lengths=None,
        source_sequence_lengths=None,
    ):
        for item in batch_list:
            self.resolve_set_structuring_spans(item)
        processor = self.task_processors.get("set_structuring")
        if processor is None:
            return None
        return processor.create_labels(
            batch_list,
            classes_mapping,
            max_seq_len=max_seq_len,
            sequence_lengths=sequence_lengths,
            source_sequence_lengths=source_sequence_lengths,
        )

    # ── Preprocessing ───────────────────────────────────────────────────

    def preprocess_example(self, item, extraction_mapping=None):
        extraction_processor = self._extraction_task_processor()
        if extraction_processor is not None and extraction_mapping is not None:
            return extraction_processor.preprocess_example(
                item, extraction_mapping,
            )
        return self._preprocess_text_only(item)

    def _preprocess_text_only(self, item):
        text = item.get("text", "")
        if "tokenized_text" in item:
            tokens = list(item["tokenized_text"])
        else:
            raw_tokens = list(self.words_splitter(text))
            if raw_tokens and isinstance(raw_tokens[0], (list, tuple)):
                tokens = [tok[0] for tok in raw_tokens]
            else:
                tokens = raw_tokens
        if len(tokens) == 0:
            tokens = ["[PAD]"]
        max_len = self.config.max_len
        if len(tokens) > max_len:
            warnings.warn(
                f"Sentence of length {len(tokens)} has been truncated to {max_len}",
                stacklevel=2
            )
            tokens = tokens[:max_len]

        return {
            "tokens": tokens,
            "seq_length": len(tokens),
        }

    # ── Batch dict creation ─────────────────────────────────────────────

    def create_batch_dict(self, batch, classes_mapping):
        batch_size = len(batch["tokens"])
        set_open_relex = batch.get("set_open_relex")
        if set_open_relex is None:
            set_processor = self.task_processors.get("set_open_relex")
            if (
                set_processor is not None
                and set_processor.allow_legacy_data
            ):
                set_open_relex = batch.get("open_relex")
        if set_open_relex is None:
            set_open_relex = [[] for _ in range(batch_size)]

        batch_dict = {
            "text": batch.get("text", ["" for _ in range(batch_size)]),
            "tokens": batch["tokens"],
            "seq_length": batch["seq_length"],
            "classes_mapping": classes_mapping,
            "classification": batch.get("classification", [[] for _ in range(batch_size)]),
            "extraction": batch.get("extraction", [[] for _ in range(batch_size)]),
            "embedding": batch.get("embedding", [[] for _ in range(batch_size)]),
            "structuring": batch.get("structuring", [{} for _ in range(batch_size)]),
            "structuring_schema": batch.get(
                "structuring_schema", [{} for _ in range(batch_size)]
            ),
            MULTI_LEVEL_META_KEY: batch.get(
                MULTI_LEVEL_META_KEY, [None for _ in range(batch_size)]
            ),
            "set_structuring": batch.get(
                "set_structuring", [{} for _ in range(batch_size)]
            ),
            "set_structuring_schema": batch.get(
                "set_structuring_schema", [{} for _ in range(batch_size)]
            ),
            SET_MULTI_LEVEL_META_KEY: batch.get(
                SET_MULTI_LEVEL_META_KEY,
                [None for _ in range(batch_size)],
            ),
            "open_relex": batch.get("open_relex", [[] for _ in range(batch_size)]),
            "set_open_relex": set_open_relex,
            "image_classification": batch.get("image_classification", [None for _ in range(batch_size)]),
            "object_detection": batch.get("object_detection", [None for _ in range(batch_size)]),
            "segmentation": batch.get("segmentation", [None for _ in range(batch_size)]),
            "audio_classification": batch.get("audio_classification", [None for _ in range(batch_size)]),
            "audio_segmentation": batch.get("audio_segmentation", [None for _ in range(batch_size)]),
            "objects": batch.get("objects", [[] for _ in range(batch_size)]),
            "image": batch.get("image", [None for _ in range(batch_size)]),
            "word_bboxes": batch.get("word_bboxes"),
            "word_page_ids": batch.get("word_page_ids"),
            "layout_input_mask": batch.get("layout_input_mask"),
            "page_input_mask": batch.get("page_input_mask"),
            "pixel_values": batch.get("pixel_values"),
            "vision_attention_mask": batch.get("vision_attention_mask"),
            "vision_input_mask": batch.get("vision_input_mask"),
            "image_sizes": batch.get("image_sizes"),
            "image_batch_idx": batch.get("image_batch_idx"),
            "image_page_ids": batch.get("image_page_ids"),
            "audio_values": batch.get("audio_values"),
            "audio_attention_mask": batch.get("audio_attention_mask"),
            "audio_input_mask": batch.get("audio_input_mask"),
            "bbox": batch.get("bbox"),
        }

        extraction_processor = self._extraction_task_processor()
        if extraction_processor is not None:
            batch_dict["span_idx"] = batch.get("span_idx")
            batch_dict["span_label"] = batch.get("span_label")
            batch_dict = extraction_processor.add_span_batch_fields(
                batch_dict, classes_mapping,
            )

        return batch_dict

    def collate_fn(self, batch_list, prepare_labels=True, *args, **kwargs):
        batch = self.collate_raw_batch(batch_list, **kwargs)
        return self.tokenize_and_prepare_labels(batch, prepare_labels)


class GLiNextTextProcessor(BaseGLiNextProcessor):
    """Processor for text models."""

    processor_task_names = _TEXT_TASKS

    def _canonicalize_structuring_tokens(self, item):
        """Use the configured inference splitter for raw structuring rows.

        A ``tokenized_text`` field is often emitted by dataset generators as
        a convenience.  Treating it as authoritative during training can
        silently create a different word sequence from inference.  Raw
        structuring values are character/text grounded, so retokenize them
        with the configured splitter before resolving spans.  Already
        resolved annotations and layout/PDF inputs keep their supplied word
        grid because their integer spans or boxes depend on it.
        """
        if not (
            item.get("structuring") or item.get("set_structuring")
        ) or not item.get("text"):
            return
        active_processors = [
            processor
            for task_name, processor in self.task_processors.items()
            if task_name in {"structuring", "set_structuring"}
        ]
        if active_processors and all(
            item.get(processor.resolved_flag)
            for processor in active_processors
        ):
            return
        if item.get("_glinext_tokens_are_authoritative"):
            return
        if any(
            item.get(key) is not None
            for key in ("word_bboxes", "bboxes", "bbox", "layout")
        ):
            return

        tokens = [
            token
            for token, _, _ in self.words_splitter(item["text"])
        ]
        if tokens:
            item["tokenized_text"] = tokens

    def _augment_label_item(self, item, batch, batch_idx):
        """Hook for modality processors to add per-item label context."""
        return None

    def _build_label_batch_list(self, batch):
        batch_list = []
        for i in range(len(batch["tokens"])):
            item = {
                "text": batch["text"][i],
                # Extraction spans were resolved before this raw batch was
                # assembled. Preserve that coordinate system when rebuilding
                # label items instead of interpreting token indices as character
                # offsets a second time.
                "tokenized_text": batch["tokens"][i],
                "_glinext_extraction_spans_resolved": True,
                "classification": batch["classification"][i],
                "extraction": batch["extraction"][i],
                "embedding": batch["embedding"][i],
                "objects": batch.get("objects", [[]])[i],
            }
            if "structuring" in batch and i < len(batch["structuring"]):
                item["structuring"] = batch["structuring"][i]
            if (
                "structuring_schema" in batch
                and i < len(batch["structuring_schema"])
            ):
                item["structuring_schema"] = batch["structuring_schema"][i]
            multi_level_meta = batch.get(MULTI_LEVEL_META_KEY)
            if (
                multi_level_meta is not None
                and i < len(multi_level_meta)
                and multi_level_meta[i] is not None
            ):
                item[MULTI_LEVEL_META_KEY] = multi_level_meta[i]
            if (
                "set_structuring" in batch
                and i < len(batch["set_structuring"])
            ):
                item["set_structuring"] = batch["set_structuring"][i]
            if (
                "set_structuring_schema" in batch
                and i < len(batch["set_structuring_schema"])
            ):
                item["set_structuring_schema"] = (
                    batch["set_structuring_schema"][i]
                )
            set_multi_level_meta = batch.get(SET_MULTI_LEVEL_META_KEY)
            if (
                set_multi_level_meta is not None
                and i < len(set_multi_level_meta)
                and set_multi_level_meta[i] is not None
            ):
                item[SET_MULTI_LEVEL_META_KEY] = set_multi_level_meta[i]
            if "open_relex" in batch and i < len(batch["open_relex"]):
                item["open_relex"] = batch["open_relex"][i]
            if (
                "set_open_relex" in batch
                and i < len(batch["set_open_relex"])
            ):
                item["set_open_relex"] = batch["set_open_relex"][i]
            for field in (
                "image_classification", "object_detection", "segmentation",
                "audio_classification", "audio_segmentation",
            ):
                values = batch.get(field)
                if values is not None and i < len(values) and values[i] is not None:
                    item[field] = values[i]
            self._augment_label_item(item, batch, i)
            batch_list.append(item)
        return batch_list

    def collate_raw_batch(self, batch_list, **kwargs):
        for item in batch_list:
            self._canonicalize_structuring_tokens(item)
            for proc in self.task_processors.values():
                proc.resolve_spans(item)

        classes_mapping = self.batch_generate_class_mappings(batch_list, **kwargs)
        preprocessed = self._preprocess_batch_text(batch_list, classes_mapping)

        texts = [item["tokens"] for item in preprocessed]
        seq_lengths = [len(t) for t in texts]
        set_processor = self.task_processors.get("set_open_relex")

        batch_dict = {
            "text": [item.get("text", "") for item in batch_list],
            "tokens": texts,
            "seq_length": torch.LongTensor(seq_lengths).unsqueeze(-1),
            "classes_mapping": classes_mapping,
            "classification": [item.get("classification", []) for item in batch_list],
            "extraction": [item.get("extraction", []) for item in batch_list],
            "embedding": [item.get("embedding", []) for item in batch_list],
            "structuring": [item.get("structuring", {}) for item in batch_list],
            "structuring_schema": [
                item.get("structuring_schema", {}) for item in batch_list
            ],
            MULTI_LEVEL_META_KEY: [
                item.get(MULTI_LEVEL_META_KEY) for item in batch_list
            ],
            "set_structuring": [
                item.get("set_structuring", {}) for item in batch_list
            ],
            "set_structuring_schema": [
                item.get("set_structuring_schema", {})
                for item in batch_list
            ],
            SET_MULTI_LEVEL_META_KEY: [
                item.get(SET_MULTI_LEVEL_META_KEY) for item in batch_list
            ],
            "open_relex": [item.get("open_relex", []) for item in batch_list],
            "set_open_relex": [
                (
                    set_processor._groups(item)
                    if set_processor is not None
                    else item.get("set_open_relex", [])
                )
                for item in batch_list
            ],
            "image_classification": [item.get("image_classification") for item in batch_list],
            "object_detection": [item.get("object_detection") for item in batch_list],
            "segmentation": [item.get("segmentation") for item in batch_list],
            "audio_classification": [item.get("audio_classification") for item in batch_list],
            "audio_segmentation": [item.get("audio_segmentation") for item in batch_list],
            "objects": [item.get("objects", []) for item in batch_list],
            "image": [item.get("image") for item in batch_list],
            "span_idx": [item.get("span_idx") for item in preprocessed],
            "span_label": [item.get("span_label") for item in preprocessed],
        }
        return self.create_batch_dict(batch_dict, classes_mapping)

    def tokenize_and_prepare_labels(self, batch, prepare_labels=True, *args, **kwargs):
        classes_mapping = batch["classes_mapping"]
        tokenize_kwargs = {}
        if batch.get("word_bboxes") is not None:
            tokenize_kwargs["word_bboxes"] = batch["word_bboxes"]
        if batch.get("word_page_ids") is not None:
            tokenize_kwargs["word_page_ids"] = batch["word_page_ids"]
        if batch.get("layout_input_mask") is not None:
            tokenize_kwargs["layout_input_mask"] = batch["layout_input_mask"]
        if batch.get("page_input_mask") is not None:
            tokenize_kwargs["page_input_mask"] = batch["page_input_mask"]
        tokenized_input = self.tokenize_inputs(batch["tokens"], classes_mapping, **tokenize_kwargs)
        tokenized_input["classes_mapping"] = classes_mapping

        if prepare_labels:
            max_seq_len = batch["seq_length"].max().item()
            sequence_lengths = tokenized_input.get("text_lengths")
            if isinstance(sequence_lengths, torch.Tensor):
                structuring_max_seq_len = int(sequence_lengths.max().item())
            else:
                structuring_max_seq_len = max_seq_len
            batch_list = self._build_label_batch_list(batch)

            cat_result = self.create_cat_labels(batch_list, classes_mapping)
            if cat_result is not None:
                tokenized_input["cat_labels"], tokenized_input["cat_batch_idx"] = cat_result

            ner_result = self.create_ner_labels(batch_list, classes_mapping, max_seq_len)
            if ner_result is not None:
                tokenized_input["ner_labels"], tokenized_input["ner_batch_idx"] = ner_result

            rel_result = self.create_joint_rel_labels(
                batch_list,
                classes_mapping,
                max_seq_len=max_seq_len,
                sequence_lengths=sequence_lengths,
            )
            if rel_result is not None:
                tokenized_input["rel_labels"] = rel_result["rel_labels"]
                tokenized_input["rel_pair_mask"] = rel_result.get("rel_pair_mask")
                tokenized_input["rel_mask"] = rel_result["rel_mask"]
                tokenized_input["rel_batch_idx"] = rel_result["rel_batch_idx"]
                tokenized_input["rel_span_idx"] = rel_result["rel_span_idx"]
                tokenized_input["rel_span_mask"] = rel_result["rel_span_mask"]
                tokenized_input["rel_span_class_idx"] = rel_result[
                    "rel_span_class_idx"
                ]

            open_rel_result = self.create_open_rel_labels(batch_list, classes_mapping, max_seq_len)
            if open_rel_result is not None:
                tokenized_input["open_rel_labels"] = open_rel_result["open_rel_labels"]
                tokenized_input["open_rel_mask"] = open_rel_result["open_rel_mask"]
                tokenized_input["open_rel_batch_idx"] = open_rel_result["open_rel_batch_idx"]
                tokenized_input["open_rel_count"] = open_rel_result["open_rel_count"]

            set_open_rel_result = self.create_set_open_rel_labels(
                batch_list,
                classes_mapping,
                max_seq_len,
            )
            if set_open_rel_result is not None:
                tokenized_input.update(set_open_rel_result)

            count_result = self.create_count_labels(batch_list, classes_mapping)
            if count_result is not None:
                tokenized_input["count_targets"] = count_result[0]
                tokenized_input["count_val"] = count_result[1]

            embedding_result = self.create_embedding_labels(batch_list)
            if embedding_result is not None:
                emb_texts = embedding_result.pop("embedding_texts")
                emb_tokenized = self.transformer_tokenizer(
                    emb_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding="longest",
                )
                tokenized_input["embedding_input_ids"] = emb_tokenized["input_ids"]
                tokenized_input["embedding_attention_mask"] = emb_tokenized["attention_mask"]
                tokenized_input["embedding_labels"] = embedding_result["embedding_labels"]
                tokenized_input["embedding_pair_idx"] = embedding_result["embedding_pair_idx"]

            extraction_processor = self._extraction_task_processor()
            if (
                getattr(self.config, "represent_spans", False)
                and extraction_processor is not None
            ):
                ner_span_result = extraction_processor.create_span_labels(
                    batch, classes_mapping,
                )
                if ner_span_result is not None:
                    tokenized_input.update(ner_span_result)

            if "structuring" in self.task_processors:
                struct_span_result = self.task_processors["structuring"].create_span_labels(
                    batch_list,
                    classes_mapping,
                    max_seq_len=structuring_max_seq_len,
                    sequence_lengths=sequence_lengths,
                    source_sequence_lengths=batch["seq_length"],
                )
                if struct_span_result is not None:
                    tokenized_input.update(struct_span_result)

            if "set_structuring" in self.task_processors:
                set_struct_span_result = self.task_processors[
                    "set_structuring"
                ].create_span_labels(
                    batch_list,
                    classes_mapping,
                    max_seq_len=structuring_max_seq_len,
                    sequence_lengths=sequence_lengths,
                    source_sequence_lengths=batch["seq_length"],
                )
                if set_struct_span_result is not None:
                    tokenized_input.update(set_struct_span_result)

            if "open_relex" in self.task_processors:
                open_rel_span_result = self.task_processors["open_relex"].create_span_labels(
                    batch_list, classes_mapping, max_seq_len=max_seq_len,
                )
                if open_rel_span_result is not None:
                    tokenized_input.update(open_rel_span_result)

            structuring_result = self.create_structuring_labels(
                batch_list,
                classes_mapping,
                structuring_max_seq_len,
                return_dict=True,
                sequence_lengths=sequence_lengths,
                source_sequence_lengths=batch["seq_length"],
            )
            if structuring_result is not None:
                tokenized_input.update(structuring_result)

            set_structuring_result = self.create_set_structuring_labels(
                batch_list,
                classes_mapping,
                structuring_max_seq_len,
                sequence_lengths=sequence_lengths,
                source_sequence_lengths=batch["seq_length"],
            )
            if set_structuring_result is not None:
                tokenized_input.update(set_structuring_result)

        if batch.get("bbox") is not None and "bbox" not in tokenized_input:
            tokenized_input["bbox"] = batch["bbox"]

        return tokenized_input


class GLiNextVisionProcessor(VisionProcessingMixin, BaseGLiNextProcessor):
    """Processor for vision models."""

    processor_task_names = _VISION_TASKS

    def collate_raw_batch(self, batch_list, **kwargs):
        classes_mapping = self.batch_generate_class_mappings(batch_list, **kwargs)
        batch = {
            "classes_mapping": classes_mapping,
            "objects": [item.get("objects", []) for item in batch_list],
            "image": [item.get("image") for item in batch_list],
            "labels": [item.get("labels") for item in batch_list],
            "classes": [item.get("classes") for item in batch_list],
            "all_labels": [item.get("all_labels") for item in batch_list],
            "true_labels": [item.get("true_labels") for item in batch_list],
            "name": [item.get("name") for item in batch_list],
            "image_classification": [item.get("image_classification") for item in batch_list],
            "object_detection": [item.get("object_detection") for item in batch_list],
            "segmentation": [item.get("segmentation") for item in batch_list],
        }
        batch.update(self._prepare_vision_batch(batch_list))
        return batch

    def _build_vision_label_batch_list(self, batch):
        batch_size = len(batch.get("objects", []))
        batch_list = []
        for i in range(batch_size):
            item = {
                "objects": batch["objects"][i],
            }
            for field in ("labels", "classes", "all_labels", "true_labels", "name", "image"):
                values = batch.get(field)
                if values is not None and i < len(values) and values[i] is not None:
                    item[field] = values[i]
            for field in ("image_classification", "object_detection", "segmentation"):
                values = batch.get(field)
                if values is not None and i < len(values) and values[i] is not None:
                    item[field] = values[i]
            if batch.get("image_sizes") is not None:
                item["_image_size"] = tuple(int(v) for v in batch["image_sizes"][i].tolist())
            batch_list.append(item)
        return batch_list

    def tokenize_and_prepare_labels(self, batch, prepare_labels=True, *args, **kwargs):
        classes_mapping = batch["classes_mapping"]
        tokenized_input = {"classes_mapping": classes_mapping}
        tokenized_input.update(self.prepare_task_label_encoder_inputs(classes_mapping, _VISION_TASKS))

        if prepare_labels:
            batch_list = self._build_vision_label_batch_list(batch)
            tokenized_input.update(
                self.create_task_labels(batch_list, classes_mapping, _VISION_TASKS)
            )

        if batch.get("pixel_values") is not None:
            tokenized_input["pixel_values"] = batch["pixel_values"]
        if batch.get("vision_attention_mask") is not None:
            tokenized_input["vision_attention_mask"] = batch["vision_attention_mask"]
        if batch.get("image_sizes") is not None:
            tokenized_input["image_sizes"] = batch["image_sizes"]
        return tokenized_input


class GLiNextAudioProcessor(AudioProcessingMixin, BaseGLiNextProcessor):
    """Processor for audio models."""

    processor_task_names = _AUDIO_TASKS

    def collate_raw_batch(self, batch_list, **kwargs):
        classes_mapping = self.batch_generate_class_mappings(batch_list, **kwargs)
        audio_fields = self._prepare_audio_batch(batch_list)
        batch = {
            "classes_mapping": classes_mapping,
            "segments": [item.get("segments") for item in batch_list],
            "audio_segments": [item.get("audio_segments") for item in batch_list],
            "labels": [item.get("labels") for item in batch_list],
            "classes": [item.get("classes") for item in batch_list],
            "all_labels": [item.get("all_labels") for item in batch_list],
            "true_labels": [item.get("true_labels") for item in batch_list],
            "name": [item.get("name") for item in batch_list],
            "duration": [item.get("duration") for item in batch_list],
            "audio_duration": [item.get("audio_duration") for item in batch_list],
            "sample_rate": [item.get("sample_rate") for item in batch_list],
            "audio_classification": [item.get("audio_classification") for item in batch_list],
            "audio_segmentation": [item.get("audio_segmentation") for item in batch_list],
        }
        batch.update(audio_fields)
        return batch

    def _build_audio_label_batch_list(self, batch):
        batch_size = batch["audio_values"].shape[0] if batch.get("audio_values") is not None else len(batch["labels"])
        batch_list = []
        for i in range(batch_size):
            item = {}
            for field in (
                "segments", "audio_segments", "labels", "classes", "all_labels",
                "true_labels", "name", "duration", "audio_duration", "sample_rate",
                self.duration_metadata_key, "audio_classification", "audio_segmentation",
            ):
                values = batch.get(field)
                if values is not None and i < len(values) and values[i] is not None:
                    item[field] = values[i]
            if batch.get("audio_attention_mask") is not None:
                item["_audio_num_samples"] = int(batch["audio_attention_mask"][i].sum().item())
            batch_list.append(item)
        return batch_list

    def tokenize_and_prepare_labels(self, batch, prepare_labels=True, *args, **kwargs):
        classes_mapping = batch["classes_mapping"]
        tokenized_input = {"classes_mapping": classes_mapping}
        tokenized_input.update(self.prepare_task_label_encoder_inputs(classes_mapping, _AUDIO_TASKS))

        if prepare_labels:
            batch_list = self._build_audio_label_batch_list(batch)
            tokenized_input.update(
                self.create_task_labels(batch_list, classes_mapping, _AUDIO_TASKS)
            )

        if batch.get("audio_values") is not None:
            tokenized_input["audio_values"] = batch["audio_values"]
        if batch.get("audio_attention_mask") is not None:
            tokenized_input["audio_attention_mask"] = batch["audio_attention_mask"]
        return tokenized_input


class GLiNextLayoutProcessor(LayoutProcessingMixin, VisionProcessingMixin, GLiNextTextProcessor):
    """Processor for text plus document-layout models."""

    processor_task_names = _TEXT_TASKS

    @staticmethod
    def _has_layout_image_payload(item: Dict[str, Any]) -> bool:
        return (
            item.get("image") is not None
            or item.get("images") is not None
            or item.get("pixel_values") is not None
        )

    @staticmethod
    def _as_list_payload(value):
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            if value.dim() == 4:
                return [value[i] for i in range(value.shape[0])]
            return [value]
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _layout_image_payloads(self, item: Dict[str, Any]):
        pixels = self._as_list_payload(item.get("pixel_values"))
        images = self._as_list_payload(item.get("images"))
        if not images:
            images = self._as_list_payload(item.get("image"))
        if pixels:
            payloads = [{"pixel_values": value} for value in pixels]
        else:
            payloads = [{"image": value} for value in images]

        raw_page_ids = item.get("image_page_ids")
        if raw_page_ids is None:
            raw_page_ids = item.get("pages")
        if raw_page_ids is None and len(payloads) == 1:
            raw_page_ids = [item.get("page", 0)]
        elif raw_page_ids is None:
            raw_page_ids = list(range(len(payloads)))
        page_ids = [self._normalize_page_id_value(value) for value in self._as_list_payload(raw_page_ids)]
        if len(page_ids) == 1 and len(payloads) > 1:
            page_ids = page_ids * len(payloads)
        if len(page_ids) != len(payloads):
            raise ValueError(
                f"image_page_ids/pages must contain one entry per image, got {len(page_ids)} and {len(payloads)}."
            )
        return list(zip(payloads, page_ids))

    def _prepare_layout_vision_batch(self, batch_list):
        image_tensors = []
        image_sizes = []
        image_batch_idx = []
        image_page_ids = []
        max_shape = None
        for batch_idx, item in enumerate(batch_list):
            for payload, page_id in self._layout_image_payloads(item):
                tensor, image_size = self.load_image(payload, getattr(self.config, "image_size", None))
                item["_image_size"] = image_size
                image_tensors.append(tensor)
                image_sizes.append(image_size)
                image_batch_idx.append(batch_idx)
                image_page_ids.append(page_id)
                shape = tuple(tensor.shape)
                if max_shape is None:
                    max_shape = list(shape)
                elif len(shape) != len(max_shape):
                    raise ValueError("All layout image tensors in a batch must have the same rank.")
                else:
                    max_shape = [max(current, int(dim)) for current, dim in zip(max_shape, shape)]
        if not image_tensors:
            return {}

        pixel_values = torch.zeros((len(image_tensors), *max_shape), dtype=torch.float)
        for idx, tensor in enumerate(image_tensors):
            slices = (idx, *[slice(0, int(dim)) for dim in tensor.shape])
            pixel_values[slices] = tensor

        return {
            "pixel_values": pixel_values,
            "image_sizes": torch.tensor(image_sizes, dtype=torch.long),
            "image_batch_idx": torch.tensor(image_batch_idx, dtype=torch.long),
            "image_page_ids": torch.tensor(image_page_ids, dtype=torch.long),
        }

    def collate_raw_batch(self, batch_list, **kwargs):
        layout_fields = self._prepare_layout_batch(batch_list)
        batch = GLiNextTextProcessor.collate_raw_batch(self, batch_list, **kwargs)
        batch.update(self._prepare_layout_vision_batch(batch_list))
        batch.update(layout_fields)
        return batch

    def tokenize_and_prepare_labels(self, batch, prepare_labels=True, *args, **kwargs):
        tokenized_input = GLiNextTextProcessor.tokenize_and_prepare_labels(
            self, batch, prepare_labels, *args, **kwargs,
        )
        if prepare_labels:
            classes_mapping = batch["classes_mapping"]
            max_seq_len = batch["seq_length"].max().item()
            batch_list = self._build_label_batch_list(batch)
            tokenized_input.update(
                self.create_task_labels(
                    batch_list,
                    classes_mapping,
                    (*_VISION_TASKS, *_AUDIO_TASKS),
                    max_seq_len=max_seq_len,
                )
            )
        if batch.get("pixel_values") is not None:
            tokenized_input["pixel_values"] = batch["pixel_values"]
        if batch.get("vision_attention_mask") is not None:
            tokenized_input["vision_attention_mask"] = batch["vision_attention_mask"]
        if batch.get("image_sizes") is not None:
            tokenized_input["image_sizes"] = batch["image_sizes"]
        if batch.get("image_batch_idx") is not None:
            tokenized_input["image_batch_idx"] = batch["image_batch_idx"]
        if batch.get("image_page_ids") is not None:
            tokenized_input["image_page_ids"] = batch["image_page_ids"]
        return tokenized_input


class GLiNextOmniProcessor(
    LayoutProcessingMixin,
    VisionProcessingMixin,
    AudioProcessingMixin,
    GLiNextTextProcessor,
):
    """Processor for models that combine text, vision, audio, and optional layout."""

    processor_task_names = _ALL_TASKS

    @staticmethod
    def _has_vision_payload(item: Dict[str, Any]) -> bool:
        return item.get("image") is not None or item.get("pixel_values") is not None

    @staticmethod
    def _has_audio_payload(item: Dict[str, Any]) -> bool:
        return any(
            item.get(key) is not None
            for key in ("audio", "audio_values", "mel_values", "mel_spectrogram", "audio_features")
        )

    @staticmethod
    def _requires_vision_payload(item: Dict[str, Any]) -> bool:
        return any(item.get(name) is not None for name in _VISION_TASKS) or bool(item.get("objects"))

    @staticmethod
    def _requires_audio_payload(item: Dict[str, Any]) -> bool:
        return any(item.get(name) is not None for name in _AUDIO_TASKS) or bool(
            item.get("segments") or item.get("audio_segments")
        )

    def _mark_optional_media_items(self, batch_list):
        for item in batch_list:
            has_vision = self._has_vision_payload(item)
            has_audio = self._has_audio_payload(item)

            if not has_vision:
                if self._requires_vision_payload(item):
                    raise ValueError(
                        "Omni vision task rows require 'image' or 'pixel_values'."
                    )
                item["_skip_vision_tasks"] = True
            else:
                item.pop("_skip_vision_tasks", None)

            if not has_audio:
                if self._requires_audio_payload(item):
                    raise ValueError(
                        "Omni audio task rows require 'audio', 'audio_values', or audio feature tensors."
                    )
                item["_skip_audio_tasks"] = True
            else:
                item.pop("_skip_audio_tasks", None)

    def _prepare_optional_vision_batch(self, batch_list):
        if not self._has_vision_inputs():
            return {}

        image_tensors = [None for _ in batch_list]
        image_sizes = [(0, 0) for _ in batch_list]
        tensor_shape = None
        any_payload = False

        for idx, item in enumerate(batch_list):
            if item.get("_skip_vision_tasks"):
                continue
            tensor, image_size = self.load_image(item, getattr(self.config, "image_size", None))
            item["_image_size"] = image_size
            image_tensors[idx] = tensor
            image_sizes[idx] = image_size
            any_payload = True

            shape = tuple(tensor.shape)
            if tensor_shape is None:
                tensor_shape = shape
            elif shape != tensor_shape:
                raise ValueError(
                    "Omni vision batches require one processed tensor shape. "
                    "Configure vision resizing instead of spatial padding, which "
                    "would misalign normalized boxes and dense patch coordinates."
                )

        if not any_payload:
            return {}

        pixel_values = torch.zeros((len(batch_list), *tensor_shape), dtype=torch.float)
        vision_input_mask = torch.zeros(len(batch_list), dtype=torch.long)
        for idx, tensor in enumerate(image_tensors):
            if tensor is None:
                continue
            slices = (idx, *[slice(0, int(dim)) for dim in tensor.shape])
            pixel_values[slices] = tensor
            vision_input_mask[idx] = 1

        return {
            "pixel_values": pixel_values,
            "image_sizes": torch.tensor(image_sizes, dtype=torch.long),
            "vision_input_mask": vision_input_mask,
        }

    def _prepare_optional_audio_batch(self, batch_list):
        if not self._has_audio_inputs():
            return {}

        audio_tensors = [None for _ in batch_list]
        audio_durations = [None for _ in batch_list]
        max_audio_shape = None
        max_audio_len = 0
        any_payload = False

        for idx, item in enumerate(batch_list):
            if item.get("_skip_audio_tasks"):
                continue
            tensor, time_length = self.load_audio(item)
            item["_audio_num_samples"] = time_length
            audio_tensors[idx] = tensor
            audio_durations[idx] = item.get(self.duration_metadata_key)
            any_payload = True

            shape = tuple(tensor.shape)
            if max_audio_shape is None:
                max_audio_shape = list(shape)
            elif len(shape) != len(max_audio_shape):
                raise ValueError("All audio tensors in an omni batch must have the same rank.")
            else:
                max_audio_shape = [max(current, int(dim)) for current, dim in zip(max_audio_shape, shape)]
            max_audio_len = max(max_audio_len, time_length)

        if not any_payload:
            return {}

        audio_values = torch.zeros((len(batch_list), *max_audio_shape), dtype=torch.float)
        audio_attention_mask = torch.zeros(len(batch_list), max_audio_len, dtype=torch.long)
        audio_input_mask = torch.zeros(len(batch_list), dtype=torch.long)
        for idx, tensor in enumerate(audio_tensors):
            if tensor is None:
                continue
            slices = (idx, *[slice(0, int(dim)) for dim in tensor.shape])
            audio_values[slices] = tensor
            audio_attention_mask[idx, :self._time_length(tensor)] = 1
            audio_input_mask[idx] = 1

        return {
            "audio_values": audio_values,
            "audio_attention_mask": audio_attention_mask,
            "audio_input_mask": audio_input_mask,
            self.duration_metadata_key: audio_durations,
        }

    def collate_raw_batch(self, batch_list, **kwargs):
        self._mark_optional_media_items(batch_list)
        layout_fields = self._prepare_layout_batch(batch_list)
        batch = GLiNextTextProcessor.collate_raw_batch(self, batch_list, **kwargs)
        batch.update(self._prepare_optional_vision_batch(batch_list))
        batch.update(self._prepare_optional_audio_batch(batch_list))
        batch.update(layout_fields)
        batch.update({
            "labels": [item.get("labels") for item in batch_list],
            "classes": [item.get("classes") for item in batch_list],
            "all_labels": [item.get("all_labels") for item in batch_list],
            "true_labels": [item.get("true_labels") for item in batch_list],
            "name": [item.get("name") for item in batch_list],
            "segments": [item.get("segments") for item in batch_list],
            "audio_segments": [item.get("audio_segments") for item in batch_list],
            "duration": [item.get("duration") for item in batch_list],
            "audio_duration": [item.get("audio_duration") for item in batch_list],
            "sample_rate": [item.get("sample_rate") for item in batch_list],
            "image_classification": [item.get("image_classification") for item in batch_list],
            "object_detection": [item.get("object_detection") for item in batch_list],
            "segmentation": [item.get("segmentation") for item in batch_list],
            "audio_classification": [item.get("audio_classification") for item in batch_list],
            "audio_segmentation": [item.get("audio_segmentation") for item in batch_list],
        })
        return batch

    def _build_label_batch_list(self, batch):
        batch_list = GLiNextTextProcessor._build_label_batch_list(self, batch)
        for i, item in enumerate(batch_list):
            for field in (
                "labels", "classes", "all_labels", "true_labels", "name",
                "segments", "audio_segments", "duration", "audio_duration",
                "sample_rate",
                "image_classification", "object_detection", "segmentation",
                "audio_classification", "audio_segmentation",
            ):
                values = batch.get(field)
                if values is not None and i < len(values) and values[i] is not None:
                    item[field] = values[i]
        return batch_list

    def tokenize_and_prepare_labels(self, batch, prepare_labels=True, *args, **kwargs):
        tokenized_input = GLiNextTextProcessor.tokenize_and_prepare_labels(
            self, batch, prepare_labels, *args, **kwargs,
        )
        if prepare_labels:
            classes_mapping = batch["classes_mapping"]
            max_seq_len = batch["seq_length"].max().item()
            batch_list = self._build_label_batch_list(batch)
            tokenized_input.update(
                self.create_task_labels(
                    batch_list,
                    classes_mapping,
                    (*_VISION_TASKS, *_AUDIO_TASKS),
                    max_seq_len=max_seq_len,
                )
            )
        if batch.get("pixel_values") is not None:
            tokenized_input["pixel_values"] = batch["pixel_values"]
        if batch.get("vision_attention_mask") is not None:
            tokenized_input["vision_attention_mask"] = batch["vision_attention_mask"]
        if batch.get("vision_input_mask") is not None:
            tokenized_input["vision_input_mask"] = batch["vision_input_mask"]
        if batch.get("image_sizes") is not None:
            tokenized_input["image_sizes"] = batch["image_sizes"]
        if batch.get("audio_values") is not None:
            tokenized_input["audio_values"] = batch["audio_values"]
        if batch.get("audio_attention_mask") is not None:
            tokenized_input["audio_attention_mask"] = batch["audio_attention_mask"]
        if batch.get("audio_input_mask") is not None:
            tokenized_input["audio_input_mask"] = batch["audio_input_mask"]
        return tokenized_input


def resolve_glinext_processor_class(config):
    """Resolve the concrete processor class from ``config.model_variant``."""
    variant = _normalize_variant(getattr(config, "model_variant", None))
    if variant == "layout":
        return GLiNextLayoutProcessor
    if variant == "vision":
        return GLiNextVisionProcessor
    if variant == "audio":
        return GLiNextAudioProcessor
    if variant == "omni":
        return GLiNextOmniProcessor
    if variant == "text":
        return GLiNextTextProcessor
    raise ValueError(f"Unknown GLiNExT model_variant: {getattr(config, 'model_variant', None)!r}")


class GLiNextProcessor(GLiNextOmniProcessor):
    """Backward-compatible processor class.

    Existing imports keep working and retain all modality capabilities. New
    construction paths should use :func:`resolve_glinext_processor_class`.
    """
    pass
