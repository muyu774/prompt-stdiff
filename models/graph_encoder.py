"""Traffic branch graph encoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.layers import ResidualMLPBlock, SimpleGraphConv


class TrafficGraphEncoder(nn.Module):
    """Build h_traffic from x_t + history context + physical/semantic graphs.

    Inputs:
        x_t: [B, H, N, F]
        h_time: [B, H, N, C]
        a_phy: [N, N] or [B, N, N]
        a_sem: [N, N] or [B, N, N]
    Output:
        h_traffic: [B, H, N, C]
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim + hidden_dim, hidden_dim)
        self.gconv_phy = SimpleGraphConv(hidden_dim, hidden_dim)
        self.gconv_sem = SimpleGraphConv(hidden_dim, hidden_dim)
        self.fuse = nn.Linear(hidden_dim * 2, hidden_dim)
        self.refine = ResidualMLPBlock(hidden_dim, hidden_dim * 2, dropout=dropout)

    def forward(
        self,
        x_t: torch.Tensor,
        h_time: torch.Tensor,
        a_phy: torch.Tensor,
        a_sem: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([x_t, h_time], dim=-1)
        h = self.in_proj(x)

        h_phy = self.gconv_phy(h, a_phy)
        h_sem = self.gconv_sem(h, a_sem)

        h = self.fuse(torch.cat([h_phy, h_sem], dim=-1))
        h = self.refine(h)
        return h
