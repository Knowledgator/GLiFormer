"""Triton flash-attention kernels for the DeBERTa backbones.

Only the ``ops`` modules of ``flashdeberta`` are imported here: they are plain
Triton kernels, so this adapter does not inherit that package's coupling to the
``transformers`` DeBERTa classes.

Two kernels are wrapped, because DeBERTa attention scores are a sum of a
content term and several bias terms:

``disentangled``
    Rebuilds the c2p/p2c terms inside the kernel from the small
    ``(B, H, L, 2 * position_buckets)`` position projections, so no ``L x L``
    tensor is ever materialized. It masks with one key length per example, so
    it only stands in for plain right padding, and it has no place to put an
    extra additive bias.

``bias``
    Takes a full ``(B, H, L, L)`` additive bias, so it accepts anything the
    eager path can express -- the 2D layout bias, packed block masks, custom
    relative positions -- while still avoiding the softmax and probability
    tensors of the eager implementation.
"""

import functools
import warnings
from dataclasses import dataclass

import torch

#: Head dimensions the Triton kernels accept.
SUPPORTED_HEAD_DIMS = frozenset({16, 32, 64, 128})

#: Dtypes the Triton kernels accept.
SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

EAGER = "eager"
FLASH_AUTO = "flash"
FLASH_BIAS = "flash_bias"
FLASH_DISENTANGLED = "flash_disentangled"

#: Values accepted by ``LayoutDebertaConfig.attn_kernel``.
ATTN_KERNELS = (EAGER, FLASH_AUTO, FLASH_BIAS, FLASH_DISENTANGLED)

_KERNEL_ALIASES = {
    "flash": FLASH_AUTO,
    "flash_attention": FLASH_AUTO,
    "flash_attention_2": FLASH_AUTO,
    "flashdeberta": FLASH_AUTO,
    "flash_deberta": FLASH_AUTO,
    "bias": FLASH_BIAS,
    "flash_bias": FLASH_BIAS,
    "disentangled": FLASH_DISENTANGLED,
    "flash_disentangled": FLASH_DISENTANGLED,
}

_WARNED: set = set()


def warn_once(message: str) -> None:
    """Emit ``message`` the first time it is raised in this process."""
    if message in _WARNED:
        return
    _WARNED.add(message)
    warnings.warn(message, stacklevel=3)


def is_flash_kernel(value) -> str | None:
    """Return the kernel a user-facing attention setting names, else ``None``.

    Lenient by design: it is fed values like ``_attn_implementation`` that carry
    plenty of names this module knows nothing about ("sdpa", hub kernels), and
    those simply mean "not a flashdeberta kernel".
    """
    if value is True:
        return FLASH_AUTO
    if not isinstance(value, str):
        return None
    return _KERNEL_ALIASES.get(value.replace("-", "_").lower())


def normalize_attn_kernel(value) -> str:
    """Coerce a configured attention setting into one of :data:`ATTN_KERNELS`."""
    if value is None or value is False:
        return EAGER
    kernel = is_flash_kernel(value)
    if kernel is not None:
        return kernel
    if isinstance(value, str) and value.replace("-", "_").lower() in {EAGER, "sdpa", ""}:
        return EAGER
    raise ValueError(f"Unknown attn_kernel {value!r}. Expected one of {ATTN_KERNELS}.")


@functools.lru_cache(maxsize=1)
def _kernels():
    try:
        from flashdeberta.ops.flash_attention import flash_attention_with_disentangled
        from flashdeberta.ops.flash_attention_bias import flash_attention_with_bias
    except Exception:  # pragma: no cover - depends on the environment
        return None
    return flash_attention_with_disentangled, flash_attention_with_bias


def flash_kernels_available() -> bool:
    """Whether the ``flashdeberta`` Triton kernels can be imported."""
    return _kernels() is not None


def flash_runnable(query: torch.Tensor) -> bool:
    """Whether the kernels can run on this tensor (device, dtype, head dim)."""
    return (
        query.is_cuda
        and query.dtype in SUPPORTED_DTYPES
        and query.size(-1) in SUPPORTED_HEAD_DIMS
        and flash_kernels_available()
    )


def padding_lengths(attention_mask: torch.Tensor) -> torch.Tensor | None:
    """Per-example key lengths, when ``attention_mask`` is plain right padding.

    The disentangled kernel masks with a single length per example, so it can
    only replace masks of the form ``valid x valid`` where ``valid`` marks a
    prefix of the sequence. Left padding and packed block masks return ``None``
    and have to go through the bias kernel instead.
    """
    mask = attention_mask
    if mask.dim() == 4:
        if mask.size(1) != 1:
            return None
        mask = mask.squeeze(1)
    if mask.dim() not in {2, 3}:
        return None
    mask = mask.bool()
    if mask.dim() == 3:
        if mask.size(-1) != mask.size(-2):
            return None
        valid = mask.any(dim=-1)
    else:
        valid = mask

    lengths = valid.sum(dim=-1)
    positions = torch.arange(valid.size(-1), device=valid.device)
    prefix = positions.unsqueeze(0) < lengths.unsqueeze(-1)
    if not torch.equal(valid, prefix):
        return None
    if mask.dim() == 3 and not torch.equal(mask, prefix.unsqueeze(-1) & prefix.unsqueeze(-2)):
        return None
    return lengths.to(dtype=torch.int32)


def additive_mask_bias(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Additive form of a boolean attention mask, shaped ``(B, 1, Q, K)``.

    Masked cells get a large *finite* penalty rather than ``-inf``: a row that
    is masked out entirely then falls back to a uniform distribution, the way
    ``masked_fill`` + ``softmax`` does in the eager path, instead of producing
    NaNs that would spread across every token in the next layer. Halving
    ``finfo.min`` keeps the sum with the other bias terms finite in fp16.
    """
    mask = attention_mask.bool()
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() != 4:
        raise ValueError(f"Expected a 3D or 4D attention mask, got {tuple(attention_mask.shape)}")
    penalty = torch.finfo(dtype).min / 2
    bias = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    return bias.masked_fill_(~mask, penalty)


@dataclass
class FlashAttentionContext:
    """Flash-attention state shared by every layer of one encoder pass.

    The compute dtype is only known inside the attention module (autocast casts
    the projections, not the hidden states), so the additive mask bias is built
    on first use and then reused by the remaining layers.
    """

    kernel: str
    attention_mask: torch.Tensor
    seq_lengths: torch.Tensor | None = None
    _mask_bias: torch.Tensor | None = None

    def mask_bias(self, dtype: torch.dtype) -> torch.Tensor:
        if self._mask_bias is None or self._mask_bias.dtype != dtype:
            self._mask_bias = additive_mask_bias(self.attention_mask, dtype)
        return self._mask_bias


def flash_attention_bias(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Flash attention over ``softmax(q @ k.T * sm_scale + bias) @ v``.

    ``bias`` must be materialized at ``(B, H, Q, K)``: the kernel only
    broadcasts a size-1 *batch* dimension, and its backward pass accumulates
    the bias gradient without synchronizing across the head grid.
    """
    _, flash_with_bias = _kernels()
    return flash_with_bias(query, key, value, bias, False, sm_scale)


def flash_attention_disentangled(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lengths: torch.Tensor | None,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    sm_scale: float,
    position_buckets: int,
    max_relative_positions: int,
) -> torch.Tensor:
    """Flash attention that rebuilds the c2p/p2c bias inside the kernel."""
    flash_disentangled, _ = _kernels()
    return flash_disentangled(
        query,
        key,
        value,
        seq_lengths,
        pos_key,
        pos_query,
        False,
        sm_scale,
        position_buckets,
        max_relative_positions,
    )
