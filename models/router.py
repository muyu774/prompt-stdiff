"""Step-aware cross-modal router."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from models.layers import MLP, SinusoidalTimeEmbedding


class StepAwareCrossModalRouter(nn.Module):
    """Fuse semantic and traffic branches using diffusion-step awareness.

    Inputs:
        h_sem: [B, H, N, C]
        h_traffic: [B, H, N, C]
        t: [B]
    Outputs:
        h_fused: [B, H, N, C]
        gate: [B, H, N, C]
    """

    def __init__(
        self,
        hidden_dim: int,
        time_embed_dim: int,
        router_hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_proj = nn.Linear(time_embed_dim, hidden_dim)
        self.sem_proj = nn.Linear(hidden_dim, hidden_dim)
        self.trf_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate_mlp = MLP(
            in_dim=hidden_dim * 3,
            hidden_dim=router_hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        h_sem: torch.Tensor,
        h_traffic: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, _, c = h_sem.shape

        h_sem_aligned = self.sem_proj(h_sem)
        h_trf_aligned = self.trf_proj(h_traffic)
        t_vec = self.time_proj(self.time_emb(t)).view(b, 1, 1, c)
        t_feat = t_vec.expand_as(h_sem_aligned)

        gate_logits = self.gate_mlp(torch.cat([h_sem_aligned, h_trf_aligned, t_feat], dim=-1))
        gate = torch.sigmoid(gate_logits)

        h_fused = gate * h_sem_aligned + (1.0 - gate) * h_trf_aligned
        return h_fused, gate
