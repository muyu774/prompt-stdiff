"""Temporal encoder for history traffic sequences."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """Encode historical traffic sequence X_his into horizon-aligned context.

    Input:
        x_his: [B, T, N, F]
    Output:
        h_time: [B, H, N, C]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizon_steps: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.horizon_steps = horizon_steps
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_his: torch.Tensor) -> torch.Tensor:
        b, t, n, f = x_his.shape
        x = x_his.permute(0, 2, 1, 3).contiguous().view(b * n, t, f)  # [B*N, T, F]
        _, h_last = self.gru(x)
        h = h_last[-1].view(b, n, -1)  # [B, N, C]
        h = self.proj(h)
        h = h.unsqueeze(1).expand(-1, self.horizon_steps, -1, -1).contiguous()
        return h
