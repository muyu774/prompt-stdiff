"""Reverse diffusion sampler."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch

from diffusion.process import DiffusionProcess
from diffusion.noise_prior import SemanticGuidedDynamicNoisePrior


class DiffusionSampler:
    """DDPM sampler for traffic forecasting."""

    def __init__(
        self,
        process: DiffusionProcess,
        noise_prior: SemanticGuidedDynamicNoisePrior,
        sampling_steps: int | None = None,
    ) -> None:
        self.process = process
        self.noise_prior = noise_prior
        self.sampling_steps = int(sampling_steps) if sampling_steps is not None else int(process.num_steps)
        if self.sampling_steps != int(self.process.num_steps):
            raise ValueError(
                "Current sampler supports full-step ancestral sampling only. "
                f"Got sampling_steps={self.sampling_steps}, num_steps={self.process.num_steps}."
            )

    @torch.no_grad()
    def sample(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor],
        shape: Tuple[int, int, int, int],
        cond: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Sample predicted future traffic.

        Args:
            model_fn: Function that predicts epsilon.
            shape: Target shape [B, H, N, F].
            cond: Condition dict containing x_his, A_phy, A_sem, Z_sem.
            device: Torch device.
        """
        z_sem = cond["z_sem"]
        x_t = self.noise_prior.sample_initial(z_sem=z_sem, target_shape=shape).to(device)

        for step in reversed(range(self.process.num_steps)):
            t = torch.full((shape[0],), step, dtype=torch.long, device=device)
            eps_pred = model_fn(x_t, t, cond)
            x_t = self.process.p_sample(x_t=x_t, t=t, eps_pred=eps_pred)
        return x_t
