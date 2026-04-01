"""Prompt-STDiff model wrapper."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from models.denoiser import EpsilonTheta


class PromptSTDiff(nn.Module):
    """Main Prompt-STDiff model.

    This wrapper keeps denoiser interface stable for trainer/sampler.
    """

    def __init__(
        self,
        input_dim: int,
        sem_dim: int,
        hidden_dim: int,
        horizon_steps: int,
        time_embed_dim: int,
        router_hidden_dim: int,
        num_layers: int,
        dropout: float,
        semantic_dropout_p: float,
    ) -> None:
        super().__init__()
        self.epsilon_theta = EpsilonTheta(
            input_dim=input_dim,
            sem_dim=sem_dim,
            hidden_dim=hidden_dim,
            horizon_steps=horizon_steps,
            time_embed_dim=time_embed_dim,
            router_hidden_dim=router_hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            semantic_dropout_p=semantic_dropout_p,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_his: torch.Tensor,
        a_phy: torch.Tensor,
        a_sem: torch.Tensor,
        z_sem: torch.Tensor,
    ) -> torch.Tensor:
        """Predict epsilon noise tensor."""
        return self.epsilon_theta(
            x_t=x_t,
            t=t,
            x_his=x_his,
            a_phy=a_phy,
            a_sem=a_sem,
            z_sem=z_sem,
        )

    def model_fn(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Sampler-compatible model function."""
        return self.forward(
            x_t=x_t,
            t=t,
            x_his=cond["x_his"],
            a_phy=cond["a_phy"],
            a_sem=cond["a_sem"],
            z_sem=cond["z_sem"],
        )
