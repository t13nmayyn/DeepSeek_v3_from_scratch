import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class RotaryPositionalEmbedding1(nn.Module):
    """Applies Rotary Positional Embeddings (RoPE) to enhance positional awareness."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Precompute inverse frequencies for efficiency
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = x.shape
        positions = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        sinusoid = positions[:, None] * self.inv_freq[None, :]  # [seq_len, dim/2]
        sin, cos = torch.sin(sinusoid), torch.cos(sinusoid)
        sin = sin[None, None, :, :].expand(batch_size, num_heads, -1, -1)
        cos = cos[None, None, :, :].expand(batch_size, num_heads, -1, -1)

        # Rotate pairs of dimensions
        x_rot = x.view(batch_size, num_heads, seq_len, head_dim // 2, 2)
        x1, x2 = x_rot.unbind(dim=-1)
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.view(batch_size, num_heads, seq_len, head_dim)



# rope=RotaryPositionalEmbedding(8)
# print(rope(torch.randn(1,2,4,8)))



import torch
import torch.nn as nn
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int):
        """
        Rotary Embedding class that returns sin and cos frequencies for RoPE.
        """
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)  # [dim/2]

    def forward(self, seq_len: int, device: torch.device):
        """
        Returns sinusoidal values (sin and cos) used in RoPE.
        Output shape: [1, 1, seq_len, dim/2]
        """
        t = torch.arange(seq_len, device=device).float()  # [seq_len]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [seq_len, dim/2]
        sin = freqs.sin()[None, None, :, :]  # [1, 1, seq_len, dim/2]
        cos = freqs.cos()[None, None, :, :]  # [1, 1, seq_len, dim/2]
        return sin, cos


def apply_rotary(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embedding to tensor.
    x: [batch, heads, seq_len, head_dim]
    sin, cos: [1, 1, seq_len, head_dim/2]
    """
    b, h, t, d = x.shape
    x = x.view(b, h, t, d // 2, 2)
    x1, x2 = x.unbind(-1)
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)  # [batch, heads, seq_len, head_dim]
