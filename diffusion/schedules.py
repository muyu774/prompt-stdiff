"""Diffusion beta schedule builders."""

from __future__ import annotations

import math

import torch


def linear_beta_schedule(
    timesteps: int,
    beta_start: float,
    beta_end: float,
) -> torch.Tensor:
    """Build linear beta schedule."""
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Build cosine beta schedule."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-6, 0.999)


def build_beta_schedule(config: dict) -> torch.Tensor:
    """Build beta schedule from config."""
    name = str(config["beta_schedule"]).lower()
    steps = int(config["num_steps"])

    if name == "linear":
        return linear_beta_schedule(
            timesteps=steps,
            beta_start=float(config["beta_start"]),
            beta_end=float(config["beta_end"]),
        )
    if name == "cosine":
        return cosine_beta_schedule(timesteps=steps)
    raise ValueError(f"Unsupported beta schedule: {name}")
