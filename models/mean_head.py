"""Deterministic traffic mean head for hybrid probabilistic forecasting."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.layers import MLP, SimpleGraphConv
from models.time_encoder import TemporalEncoder


class TrafficMeanHead(nn.Module):
    """Predict a deterministic future mean from history, graphs, and semantics.

    The diffusion branch can then model residual uncertainty around this mean. This
    gives MAE/RMSE a strong deterministic anchor while preserving probabilistic samples.
    """

    def __init__(
        self,
        input_dim: int,
        sem_dim: int,
        hidden_dim: int,
        horizon_steps: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizon_steps = int(horizon_steps)
        self.temporal_encoder = TemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon_steps=horizon_steps,
            num_layers=1,
            dropout=dropout,
        )
        self.horizon_emb = nn.Embedding(horizon_steps, hidden_dim)
        self.gconv_phy = SimpleGraphConv(hidden_dim, hidden_dim)
        self.gconv_sem = SimpleGraphConv(hidden_dim, hidden_dim)
        self.sem_proj = nn.Sequential(
            nn.Linear(sem_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fuse = MLP(
            in_dim=hidden_dim * 4,
            hidden_dim=hidden_dim * 2,
            out_dim=hidden_dim,
            dropout=dropout,
        )
        self.out = nn.Linear(hidden_dim, input_dim)

    def _expand_z_sem(self, z_sem: torch.Tensor, batch_size: int) -> torch.Tensor:
        if z_sem.dim() == 2:
            return z_sem.unsqueeze(0).expand(batch_size, -1, -1)
        if z_sem.dim() == 3 and z_sem.shape[0] == batch_size:
            return z_sem
        raise ValueError(f"Expected z_sem [N,D] or [B,N,D], got {tuple(z_sem.shape)}")

    def forward(
        self,
        x_his: torch.Tensor,
        a_phy: torch.Tensor,
        a_sem: torch.Tensor,
        z_sem: torch.Tensor,
    ) -> torch.Tensor:
        b, _, n, _ = x_his.shape

        h_time = self.temporal_encoder(x_his=x_his)  # [B,H,N,C]
        h_ids = torch.arange(self.horizon_steps, device=x_his.device)
        h_pos = self.horizon_emb(h_ids).view(1, self.horizon_steps, 1, -1)
        h_base = h_time + h_pos

        h_phy = self.gconv_phy(h_base, a_phy)
        h_sem_graph = self.gconv_sem(h_base, a_sem)

        z = self._expand_z_sem(z_sem, batch_size=b)
        z_feat = self.sem_proj(z).unsqueeze(1).expand(-1, self.horizon_steps, -1, -1)
        if z_feat.shape[2] != n:
            raise ValueError(f"Semantic node count mismatch: expected {n}, got {z_feat.shape[2]}")

        fused = self.fuse(torch.cat([h_base, h_phy, h_sem_graph, z_feat], dim=-1))
        return self.out(fused)
