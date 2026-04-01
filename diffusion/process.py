"""Core DDPM forward and reverse process."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Extract per-batch coefficients.

    Args:
        a: Lookup tensor [K].
        t: Timestep tensor [B].
        x_shape: Target shape for broadcasting.

    Returns:
        Coefficients shaped to [B, 1, 1, 1] for traffic tensors [B, H, N, F].
    """
    out = a.gather(0, t)
    view_shape = (t.shape[0],) + (1,) * (len(x_shape) - 1)
    return out.view(view_shape)


@dataclass
class PosteriorParams:
    """Posterior parameters for one reverse step."""

    mean: torch.Tensor
    variance: torch.Tensor
    x0_pred: torch.Tensor


class DiffusionProcess:
    """DDPM diffusion process for traffic tensor [B, H, N, F]."""

    def __init__(self, betas: torch.Tensor, clip_x0: bool = True) -> None:
        self.betas = betas
        self.num_steps = betas.shape[0]
        self.clip_x0 = clip_x0

        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=betas.dtype), self.alphas_cumprod[:-1]], dim=0
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def to(self, device: torch.device) -> "DiffusionProcess":
        """Move schedule tensors to target device."""
        for name in (
            "betas",
            "alphas",
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "sqrt_recip_alphas",
            "posterior_variance",
            "posterior_mean_coef1",
            "posterior_mean_coef2",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion sample q(x_t | x_0)."""
        sqrt_ab = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_1mab = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_ab * x_start + sqrt_1mab * noise

    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct x_0 from x_t and predicted epsilon."""
        sqrt_ab = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_1mab = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        x0 = (x_t - sqrt_1mab * eps) / torch.clamp(sqrt_ab, min=1e-8)
        if self.clip_x0:
            x0 = torch.clamp(x0, -3.0, 3.0)
        return x0

    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps_pred: torch.Tensor,
    ) -> PosteriorParams:
        """Compute posterior mean and variance p(x_{t-1}|x_t)."""
        x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred)
        coef1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = extract(self.posterior_mean_coef2, t, x_t.shape)
        mean = coef1 * x0_pred + coef2 * x_t
        var = extract(self.posterior_variance, t, x_t.shape)
        return PosteriorParams(mean=mean, variance=var, x0_pred=x0_pred)

    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
        """Sample one reverse step."""
        params = self.p_mean_variance(x_t=x_t, t=t, eps_pred=eps_pred)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).float().view(t.shape[0], 1, 1, 1)
        return params.mean + nonzero_mask * torch.sqrt(torch.clamp(params.variance, 1e-12)) * noise
