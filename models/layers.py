"""Common neural layers used by Prompt-STDiff."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding module."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode timesteps.

        Args:
            t: Diffusion step [B]

        Returns:
            Time embedding [B, embed_dim]
        """
        half = self.embed_dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=device).float() / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.embed_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class MLP(nn.Module):
    """Two-layer MLP with SiLU activation and dropout."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualMLPBlock(nn.Module):
    """Residual MLP block for feature refinement."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = MLP(dim, hidden_dim, dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class SimpleGraphConv(nn.Module):
    """Simple graph convolution: A @ X followed by linear projection.

    Input x shape: [B, H, N, C]
    Adjacency shape: [N, N] or [B, N, N]
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if adj.dim() == 2:
            msg = torch.einsum("ij,bhjd->bhid", adj, x)
        elif adj.dim() == 3:
            msg = torch.einsum("bij,bhjd->bhid", adj, x)
        else:
            raise ValueError(f"Expected adj [N,N] or [B,N,N], got {adj.shape}")
        return self.linear(msg)
