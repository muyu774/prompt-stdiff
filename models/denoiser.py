"""Epsilon prediction network epsilon_theta."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.graph_encoder import TrafficGraphEncoder
from models.layers import MLP, ResidualMLPBlock, SinusoidalTimeEmbedding
from models.router import StepAwareCrossModalRouter
from models.time_encoder import TemporalEncoder


class EpsilonTheta(nn.Module):
    """Prompt-STDiff denoiser with two-branch cross-modal routing.

    Branches:
    - h_sem: semantic branch from Z_sem + timestep embedding.
    - h_traffic: numerical branch from x_t + X_his + A_phy + A_sem.

    Input:
        x_t: [B, H, N, F]
        t: [B]
        x_his: [B, T, N, F]
        a_phy: [N, N] or [B, N, N]
        a_sem: [N, N] or [B, N, N]
        z_sem: [N, D_sem] or [B, N, D_sem]
    Output:
        eps_hat: [B, H, N, F]
    """

    def __init__(
        self,
        input_dim: int,
        sem_dim: int,
        hidden_dim: int,
        horizon_steps: int,
        time_embed_dim: int,
        router_hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        semantic_dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizon_steps = horizon_steps
        self.hidden_dim = hidden_dim
        self.semantic_dropout_p = float(semantic_dropout_p)

        self.temporal_encoder = TemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon_steps=horizon_steps,
            num_layers=1,
            dropout=dropout,
        )
        self.traffic_encoder = TrafficGraphEncoder(
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.time_emb = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_to_hidden = nn.Linear(time_embed_dim, hidden_dim)

        self.sem_proj = nn.Sequential(
            nn.Linear(sem_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.router = StepAwareCrossModalRouter(
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim,
            router_hidden_dim=router_hidden_dim,
            dropout=dropout,
        )

        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim + input_dim, hidden_dim * 2, dropout=dropout)]
            + [ResidualMLPBlock(hidden_dim + input_dim, hidden_dim * 2, dropout=dropout) for _ in range(num_layers - 1)]
        )
        self.head = MLP(
            in_dim=hidden_dim + input_dim,
            hidden_dim=hidden_dim * 2,
            out_dim=input_dim,
            dropout=dropout,
        )

    def _expand_z_sem(self, z_sem: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Broadcast z_sem to [B, N, D_sem]."""
        if z_sem.dim() == 2:
            z_sem = z_sem.unsqueeze(0).expand(batch_size, -1, -1)
        elif z_sem.dim() == 3:
            if z_sem.shape[0] != batch_size:
                raise ValueError(
                    f"z_sem batch mismatch: expected {batch_size}, got {z_sem.shape[0]}"
                )
        else:
            raise ValueError(f"Expected z_sem [N,D] or [B,N,D], got {z_sem.shape}")
        return z_sem

    def _build_sem_branch(self, z_sem: torch.Tensor, t: torch.Tensor, horizon: int) -> torch.Tensor:
        """Build semantic branch h_sem.

        Returns:
            h_sem [B, H, N, C]
        """
        b, n, _ = z_sem.shape
        z_feat = self.sem_proj(z_sem)  # [B, N, C]
        t_feat = self.time_to_hidden(self.time_emb(t)).view(b, 1, 1, -1)  # [B,1,1,C]

        h_sem = z_feat.unsqueeze(1).expand(-1, horizon, -1, -1)
        h_sem = h_sem + t_feat
        return h_sem

    def _semantic_dropout(self, z_sem: torch.Tensor) -> torch.Tensor:
        """Apply semantic condition dropout on whole-sample basis.

        Eq.(18)-style behavior: with probability p_drop, semantic condition is set to zero.
        """
        if (not self.training) or self.semantic_dropout_p <= 0:
            return z_sem
        if self.semantic_dropout_p >= 1.0:
            return torch.zeros_like(z_sem)

        bsz = z_sem.shape[0]
        keep = torch.bernoulli(
            torch.full((bsz, 1, 1), 1.0 - self.semantic_dropout_p, device=z_sem.device, dtype=z_sem.dtype)
        )
        return z_sem * keep

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_his: torch.Tensor,
        a_phy: torch.Tensor,
        a_sem: torch.Tensor,
        z_sem: torch.Tensor,
    ) -> torch.Tensor:
        b, h, n, f = x_t.shape

        z = self._expand_z_sem(z_sem, batch_size=b)
        z = self._semantic_dropout(z)

        h_time = self.temporal_encoder(x_his=x_his)  # [B, H, N, C]
        h_traffic = self.traffic_encoder(
            x_t=x_t,
            h_time=h_time,
            a_phy=a_phy,
            a_sem=a_sem,
        )

        h_sem = self._build_sem_branch(z_sem=z, t=t, horizon=h)
        h_fused, _ = self.router(h_sem=h_sem, h_traffic=h_traffic, t=t)

        feat = torch.cat([x_t, h_fused], dim=-1)  # [B, H, N, F+C]
        for blk in self.blocks:
            feat = blk(feat)

        eps_hat = self.head(feat)  # [B, H, N, F]
        return eps_hat
