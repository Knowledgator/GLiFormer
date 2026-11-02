"""Images through a backbone's own vision tower.

Multimodal checkpoints such as Qwen3.5 ship a vision tower, a projector and the
placeholder tokens their language model was trained to read images from. This
module lays a batch out the way those backbones expect -- one placeholder block
per image, in front of the text -- so the stock multimodal forward encodes the
image, scatters its features over the placeholders and builds the 2D rotary
positions itself.

Rows keep one image each: ``pixel_values`` is ``(batch, patches, patch_dim)``
padded over patches and ``image_grid_thw`` is ``(batch, 3)``, all zeros for a
row without an image.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn


@dataclass(frozen=True)
class NativeVisionSpec:
    """Placeholder tokens and patch geometry of a backbone's vision tower."""

    image_token_id: int
    start_token_id: Optional[int]
    end_token_id: Optional[int]
    merge_size: int
    pad_token_id: int

    @property
    def marker_tokens(self) -> int:
        return int(self.start_token_id is not None) + int(self.end_token_id is not None)


@dataclass
class NativeVisionBatch:
    """A ``[image block | text]`` batch plus where its text and vision tokens ended up."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    mm_token_type_ids: torch.Tensor
    pixel_values: torch.Tensor  # (patches over all images, patch_dim)
    image_grid_thw: torch.Tensor  # (images, 3)
    text_index: torch.Tensor  # (batch, text length), positions in the sequence
    vision_index: torch.Tensor  # (batch, max image tokens), positions in the sequence
    vision_mask: torch.Tensor  # (batch, max image tokens)
    spatial_shape: torch.Tensor  # (batch, 2), merged grid height/width
    # The same layout on the CPU (pixel_values left empty), when it was built
    # from NativeVisionHost copies; the backbone reads its sizes from it
    # without waiting for the GPU.
    host: Optional["NativeVisionBatch"] = None


class NativeVisionHost:
    """CPU copies of the inputs that decide a native vision batch's layout.

    The data collator attaches one to each image batch. It is deliberately not
    a tensor container (no ``to``/``pin_memory``), so the trainer's move to the
    GPU leaves it on the CPU, and the layout sizes can be read without a
    device sync.
    """

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        image_grid_thw: torch.Tensor,
        vision_input_mask: Optional[torch.Tensor] = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.image_grid_thw = image_grid_thw
        self.vision_input_mask = vision_input_mask

    def matches(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], pixel_values: torch.Tensor) -> bool:
        """True when these copies describe the given (device) batch."""
        if tuple(self.input_ids.shape) != tuple(input_ids.shape):
            return False
        if self.image_grid_thw.shape != (input_ids.shape[0], 3) or pixel_values.dim() != 3:
            return False
        if (attention_mask is None) != (self.attention_mask is None):
            return False
        if attention_mask is not None and tuple(self.attention_mask.shape) != tuple(attention_mask.shape):
            return False
        return self.vision_input_mask is None or self.vision_input_mask.shape[0] == input_ids.shape[0]


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor.to(device)


def native_vision_tower(model: nn.Module) -> Optional[nn.Module]:
    """The vision tower a multimodal backbone carries, if it has one."""
    for name in ("visual", "vision_tower"):
        tower = getattr(model, name, None)
        if isinstance(tower, nn.Module):
            return tower
    return None


def native_vision_spec(model: nn.Module) -> Optional[NativeVisionSpec]:
    """Read the placeholder layout from a backbone config, or None if it has no vision tower."""
    config = getattr(model, "config", None)
    image_token_id = getattr(config, "image_token_id", None)
    vision_config = getattr(config, "vision_config", None)
    if native_vision_tower(model) is None or image_token_id is None or vision_config is None:
        return None
    text_config = getattr(config, "text_config", config)
    pad_token_id = getattr(text_config, "pad_token_id", None)
    if pad_token_id is None or pad_token_id == image_token_id:
        # Padding is masked out; it only must not read as an image placeholder.
        pad_token_id = 0 if image_token_id != 0 else 1
    return NativeVisionSpec(
        image_token_id=int(image_token_id),
        start_token_id=getattr(config, "vision_start_token_id", None),
        end_token_id=getattr(config, "vision_end_token_id", None),
        merge_size=int(getattr(vision_config, "spatial_merge_size", 1)),
        pad_token_id=int(pad_token_id),
    )


def num_image_tokens(image_grid_thw: torch.Tensor, merge_size: int) -> torch.Tensor:
    """Placeholder tokens per image: the patch grid after spatial merging."""
    return image_grid_thw.prod(-1) // (merge_size**2)


def build_native_vision_batch(
    spec: NativeVisionSpec,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    vision_input_mask: Optional[torch.Tensor] = None,
    host: Optional[NativeVisionHost] = None,
) -> Optional[NativeVisionBatch]:
    """Put each row's image block in front of its text.

    Every row reads ``[start, image tokens, end | text ... pad]`` with all padding
    on the right: the recurrent layers of hybrid backbones are left-to-right, and
    padding ahead of the content would run through their state. The text therefore
    starts at a per-row offset, kept in ``text_index``. Returns None when no row
    carries an image.

    With ``host`` copies of the inputs, the layout is built on the CPU and
    copied over asynchronously; otherwise its sizes are read from the device
    tensors, which waits for the GPU.
    """
    if host is not None and input_ids.device.type != "cpu" and host.matches(input_ids, attention_mask, pixel_values):
        return _build_from_host(spec, host, input_ids.device, pixel_values)
    batch_size, text_length = input_ids.shape
    device = input_ids.device
    if pixel_values.dim() != 3 or pixel_values.shape[0] != batch_size:
        raise ValueError(
            "Native vision expects pixel_values of shape (batch, patches, patch_dim), "
            f"got {tuple(pixel_values.shape)} for a batch of {batch_size}."
        )
    image_grid_thw = image_grid_thw.to(device=device, dtype=torch.long)
    if image_grid_thw.shape != (batch_size, 3):
        raise ValueError(
            f"image_grid_thw must have shape (batch, 3), got {tuple(image_grid_thw.shape)}."
        )

    num_patches = image_grid_thw.prod(-1)
    has_image = num_patches > 0
    if vision_input_mask is not None:
        has_image = has_image & vision_input_mask.to(device=device).bool()
    if not bool(has_image.any()):
        return None
    if int(num_patches.max()) > pixel_values.shape[1]:
        raise ValueError(
            f"image_grid_thw describes up to {int(num_patches.max())} patches but pixel_values "
            f"holds {pixel_values.shape[1]} per row."
        )

    image_tokens = torch.where(has_image, num_image_tokens(image_grid_thw, spec.merge_size), 0)
    block_lengths = torch.where(has_image, image_tokens + spec.marker_tokens, 0)
    block_width = int(block_lengths.max())
    max_tokens = int(image_tokens.max())
    has_start = int(spec.start_token_id is not None)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(device=device)

    positions = torch.arange(block_width + text_length, device=device).unsqueeze(0)
    block_end = block_lengths.unsqueeze(1)
    in_block = positions < block_end
    is_image_token = (positions >= has_start) & (positions < (has_start + image_tokens).unsqueeze(1)) & in_block
    text_index = block_end + torch.arange(text_length, device=device).unsqueeze(0)

    combined_ids = torch.full_like(positions.expand(batch_size, -1), spec.pad_token_id).to(input_ids.dtype)
    combined_ids.scatter_(1, text_index, input_ids)
    combined_ids[is_image_token] = spec.image_token_id
    if spec.start_token_id is not None:
        combined_ids[in_block & (positions == 0)] = spec.start_token_id
    if spec.end_token_id is not None:
        combined_ids[in_block & (positions == block_end - 1)] = spec.end_token_id
    combined_mask = in_block.to(attention_mask.dtype).scatter(1, text_index, attention_mask)

    token_slots = torch.arange(max_tokens, device=device).unsqueeze(0)
    vision_mask = token_slots < image_tokens.unsqueeze(1)
    vision_index = torch.where(vision_mask, has_start + token_slots, torch.zeros_like(token_slots))

    patch_mask = torch.arange(pixel_values.shape[1], device=pixel_values.device).unsqueeze(0) < torch.where(
        has_image, num_patches, 0
    ).to(pixel_values.device).unsqueeze(1)
    merged_grid = image_grid_thw[:, 1:] // spec.merge_size

    return NativeVisionBatch(
        input_ids=combined_ids,
        attention_mask=combined_mask,
        mm_token_type_ids=is_image_token.long(),
        pixel_values=pixel_values[patch_mask],
        image_grid_thw=image_grid_thw[has_image],
        text_index=text_index,
        vision_index=vision_index,
        vision_mask=vision_mask.to(attention_mask.dtype),
        spatial_shape=torch.where(has_image.unsqueeze(1), merged_grid, torch.zeros_like(merged_grid)),
    )


def _build_from_host(
    spec: NativeVisionSpec,
    host: NativeVisionHost,
    device: torch.device,
    pixel_values: torch.Tensor,
) -> Optional[NativeVisionBatch]:
    """The device-tensor layout of ``build_native_vision_batch``, computed on the CPU."""
    batch_size, max_patches = pixel_values.shape[:2]
    # Same code on CPU tensors; the patches only matter for their count here.
    cpu = build_native_vision_batch(
        spec,
        host.input_ids,
        host.attention_mask,
        torch.empty((batch_size, max_patches, 0)),
        host.image_grid_thw,
        host.vision_input_mask,
    )
    if cpu is None:
        return None
    grid = host.image_grid_thw.to(torch.long)
    num_patches = grid.prod(-1)
    has_image = num_patches > 0
    if host.vision_input_mask is not None:
        has_image = has_image & host.vision_input_mask.bool()
    patch_mask = torch.arange(max_patches).unsqueeze(0) < torch.where(has_image, num_patches, 0).unsqueeze(1)
    # pixel_values[patch_mask], as a gather of known size instead of a boolean index.
    patch_index = _to_device(patch_mask.flatten().nonzero().squeeze(1), device)
    pixels = pixel_values.flatten(0, 1).index_select(0, patch_index)
    return NativeVisionBatch(
        input_ids=_to_device(cpu.input_ids, device),
        attention_mask=_to_device(cpu.attention_mask, device),
        mm_token_type_ids=_to_device(cpu.mm_token_type_ids, device),
        pixel_values=pixels,
        image_grid_thw=_to_device(cpu.image_grid_thw, device),
        text_index=_to_device(cpu.text_index, device),
        vision_index=_to_device(cpu.vision_index, device),
        vision_mask=_to_device(cpu.vision_mask, device),
        spatial_shape=_to_device(cpu.spatial_shape, device),
        host=cpu,
    )


def gather_text_tokens(encoded: torch.Tensor, batch: NativeVisionBatch) -> torch.Tensor:
    """Each row's text states, back at the positions of the original ``input_ids``."""
    index = batch.text_index.unsqueeze(-1).expand(-1, -1, encoded.shape[-1])
    return encoded.gather(1, index)


def gather_vision_tokens(encoded: torch.Tensor, batch: NativeVisionBatch) -> torch.Tensor:
    """Pull each row's contextualised image tokens out of the encoded sequence."""
    index = batch.vision_index.unsqueeze(-1).expand(-1, -1, encoded.shape[-1])
    tokens = encoded.gather(1, index)
    return tokens * batch.vision_mask.unsqueeze(-1).to(tokens.dtype)


def uses_native_vision(config: Any) -> bool:
    return str(getattr(config, "vision_encoder_type", None) or "").lower() == "native"
