"""Rotary position embeddings (RoPE)."""

from typing import Optional, Tuple

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """
    Standard RoPE (rotary) embedding that returns (cos, sin) tensors for a batch of position_ids.
    """
    inv_freq: torch.Tensor

    def __init__(
        self,
        dim: int,
        base: float = 10_000.0,
        attention_scaling: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RotaryEmbedding requires even head_dim; got {dim}")

        self.dim = dim
        self.base = float(base)
        self.attention_scaling = float(attention_scaling)

        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x_like: torch.Tensor, position_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos = position_ids[:, None, :].float()

        device_type = x_like.device.type if (isinstance(x_like.device.type, str) and x_like.device.type != "mps") else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv.float() @ pos.float()).transpose(1, 2)
            emb = torch.cat([freqs, freqs], dim=-1)

            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x_like.dtype, device=x_like.device), sin.to(dtype=x_like.dtype, device=x_like.device)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate last-dim halves: (x1, x2) -> (-x2, x1)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to q. Shapes: q: [bs, seq, head_dim], cos/sin: [bs, seq, head_dim]."""
    return (q * cos) + (rotate_half(q) * sin)
