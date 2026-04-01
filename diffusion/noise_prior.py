"""Semantic-guided dynamic noise prior."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticGuidedDynamicNoisePrior(nn.Module):
    """Semantic-guided prior for diffusion noise/initialization.

    Default mode (`learn_sigma_prior=False`) follows the calibrated paper-style path:
    1) Z_sem -> mu_sem
    2) x_K = gamma * mu_sem + sqrt(1 - gamma^2) * eps

    Optional extension mode (`learn_sigma_prior=True`) predicts sigma as well.
    Optional extension mode (`learn_mu_prior=True`) learns a parametric projection for mu.
    """

    def __init__(
        self,
        sem_dim: int,
        horizon: int,
        out_dim: int,
        num_diffusion_steps: int,
        gamma: float = 0.35,
        gamma_schedule: str = "constant",
        learn_mu_prior: bool = False,
        learn_sigma_prior: bool = False,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.out_dim = out_dim
        self.gamma = gamma
        self.gamma_schedule = gamma_schedule
        self.num_diffusion_steps = num_diffusion_steps
        self.learn_mu_prior = learn_mu_prior
        self.learn_sigma_prior = learn_sigma_prior

        if learn_mu_prior:
            self.mu_proj = nn.Sequential(
                nn.Linear(sem_dim, sem_dim),
                nn.SiLU(),
                nn.Linear(sem_dim, horizon * out_dim),
            )
        if learn_sigma_prior:
            self.sigma_proj = nn.Sequential(
                nn.Linear(sem_dim, sem_dim),
                nn.SiLU(),
                nn.Linear(sem_dim, horizon * out_dim),
            )

    def _expand_z(self, z_sem: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Broadcast semantic embeddings to [B, N, D_sem]."""
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

    def mu_sem(self, z_sem: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Map semantic embeddings to mu_sem with shape [B, H, N, F]."""
        z = self._expand_z(z_sem, batch_size=batch_size)
        b, n, _ = z.shape
        if self.learn_mu_prior:
            mu = self.mu_proj(z).view(b, n, self.horizon, self.out_dim)
        else:
            # ASSUMPTION: deterministic projection by adaptive pooling is a stable default when mu prior is not learned.
            target_dim = self.horizon * self.out_dim
            pooled = F.adaptive_avg_pool1d(z.reshape(b * n, 1, z.shape[-1]), output_size=target_dim).squeeze(1)
            mu = pooled.view(b, n, self.horizon, self.out_dim)
        mu = mu.permute(0, 2, 1, 3).contiguous()  # [B, H, N, F]
        return mu

    def _gamma_t(self, t: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        """Get per-batch gamma values with shape [B,1,1,1]."""
        if self.gamma_schedule == "constant" or t is None:
            g = torch.full((batch_size,), self.gamma, device=device)
            return g.view(batch_size, 1, 1, 1)

        if self.gamma_schedule == "linear":
            # ASSUMPTION: gamma decays linearly with timestep index.
            tn = t.float() / max(self.num_diffusion_steps - 1, 1)
            g = self.gamma * (1.0 - tn)
            return g.view(batch_size, 1, 1, 1)

        raise ValueError(f"Unsupported gamma_schedule: {self.gamma_schedule}")

    def sample_noise(
        self,
        z_sem: torch.Tensor,
        target_shape: Tuple[int, int, int, int],
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample semantic-guided noise used in q_sample.

        Args:
            z_sem: [N, D_sem] or [B, N, D_sem]
            target_shape: (B, H, N, F)
            t: Diffusion step [B]
        """
        b, h, n, f = target_shape
        mu = self.mu_sem(z_sem=z_sem, batch_size=b)
        if mu.shape != target_shape:
            raise ValueError(f"mu_sem shape mismatch: expected {target_shape}, got {mu.shape}")

        eps = torch.randn(target_shape, device=mu.device)

        if self.learn_sigma_prior:
            z = self._expand_z(z_sem, batch_size=b)
            sigma = self.sigma_proj(z).view(b, n, h, f).permute(0, 2, 1, 3).contiguous()
            sigma = F.softplus(sigma) + 1e-4
            return mu + sigma * eps

        g = self._gamma_t(t=t, batch_size=b, device=mu.device)
        g = torch.clamp(g, 0.0, 0.999)
        return g * mu + torch.sqrt(1.0 - g * g) * eps

    def sample_initial(
        self,
        z_sem: torch.Tensor,
        target_shape: Tuple[int, int, int, int],
    ) -> torch.Tensor:
        """Sample x_K initialization for reverse process."""
        b = target_shape[0]
        if z_sem.dim() == 2 or z_sem.dim() == 3:
            device = z_sem.device
        else:
            raise ValueError(f"Expected z_sem [N,D] or [B,N,D], got {z_sem.shape}")
        t = torch.full((b,), self.num_diffusion_steps - 1, device=device, dtype=torch.long)
        return self.sample_noise(z_sem=z_sem, target_shape=target_shape, t=t)
