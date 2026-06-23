"""Reverse diffusion samplers."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import torch

from diffusion.process import DiffusionProcess
from diffusion.noise_prior import SemanticGuidedDynamicNoisePrior


class DiffusionSampler:
    """DDPM/DDIM sampler for traffic forecasting."""

    def __init__(
        self,
        process: DiffusionProcess,
        noise_prior: SemanticGuidedDynamicNoisePrior,
        sampling_steps: int | None = None,
        sampler_type: str = "ddpm",
    ) -> None:
        self.process = process
        self.noise_prior = noise_prior
        self.sampling_steps = int(sampling_steps) if sampling_steps is not None else int(process.num_steps)
        self.sampler_type = str(sampler_type).lower()
        if self.sampler_type not in {"ddpm", "ddim"}:
            raise ValueError(f"Unsupported sampler_type={sampler_type}. Use ddpm or ddim.")
        if self.sampler_type == "ddpm" and self.sampling_steps != int(self.process.num_steps):
            raise ValueError(
                "DDPM ancestral sampling supports full-step sampling only. "
                f"Got sampling_steps={self.sampling_steps}, num_steps={self.process.num_steps}."
            )
        self.timesteps = self._build_timesteps()

    def _build_timesteps(self) -> List[int]:
        """Build descending inference timesteps."""
        k = int(self.process.num_steps)
        s = max(1, min(int(self.sampling_steps), k))
        if self.sampler_type == "ddpm":
            return list(reversed(range(k)))
        # Deterministic DDIM: evenly spaced original DDPM steps, descending.
        steps = torch.linspace(0, k - 1, steps=s).round().long().unique(sorted=True)
        if int(steps[-1].item()) != k - 1:
            steps = torch.cat([steps, torch.tensor([k - 1], dtype=torch.long)])
            steps = steps.unique(sorted=True)
        return [int(x) for x in reversed(steps.tolist())]

    def _ddim_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        prev_step: int,
        eps_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Run one deterministic DDIM step with eta=0.

        Args:
            x_t: Current noisy tensor [B, H, N, F].
            t: Current timestep [B].
            prev_step: Next lower timestep, or -1 for x_0.
            eps_pred: Predicted epsilon [B, H, N, F].
        """
        x0_pred = self.process.predict_x0_from_eps(x_t=x_t, t=t, eps=eps_pred)
        if prev_step < 0:
            return x0_pred
        alpha_prev = self.process.alphas_cumprod[prev_step].to(device=x_t.device, dtype=x_t.dtype)
        sqrt_alpha_prev = torch.sqrt(alpha_prev).view(1, 1, 1, 1)
        sqrt_one_minus_alpha_prev = torch.sqrt(torch.clamp(1.0 - alpha_prev, min=0.0)).view(1, 1, 1, 1)
        return sqrt_alpha_prev * x0_pred + sqrt_one_minus_alpha_prev * eps_pred

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

        for idx, step in enumerate(self.timesteps):
            t = torch.full((shape[0],), step, dtype=torch.long, device=device)
            eps_pred = model_fn(x_t, t, cond)
            if self.sampler_type == "ddpm":
                x_t = self.process.p_sample(x_t=x_t, t=t, eps_pred=eps_pred)
            else:
                prev_step = self.timesteps[idx + 1] if idx + 1 < len(self.timesteps) else -1
                x_t = self._ddim_step(x_t=x_t, t=t, prev_step=prev_step, eps_pred=eps_pred)
        return x_t

    @torch.no_grad()
    def sample_ensemble(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor],
        shape: Tuple[int, int, int, int],
        cond: Dict[str, torch.Tensor],
        device: torch.device,
        num_samples: int,
    ) -> torch.Tensor:
        """Sample an ensemble with shape [S, B, H, N, F]."""
        samples = [
            self.sample(model_fn=model_fn, shape=shape, cond=cond, device=device)
            for _ in range(max(1, int(num_samples)))
        ]
        return torch.stack(samples, dim=0)
