"""Loss functions for diffusion training."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def diffusion_loss(eps_pred: torch.Tensor, eps_true: torch.Tensor) -> torch.Tensor:
    """Backward-compatible wrapper for standard epsilon MSE loss."""
    return F.mse_loss(eps_pred, eps_true)


def diffusion_loss_with_x0(
    eps_pred: torch.Tensor,
    eps_true: torch.Tensor,
    x0_pred: torch.Tensor | None = None,
    x0_true: torch.Tensor | None = None,
    eps_weight: float = 1.0,
    x0_weight: float = 0.0,
    x0_loss_type: str = "l1",
) -> torch.Tensor:
    """Diffusion training loss with optional x0 reconstruction regularization.

    Args:
        eps_pred: Predicted epsilon.
        eps_true: Ground-truth epsilon noise.
        x0_pred: Reconstructed x0 from eps_pred and x_t.
        x0_true: Ground-truth clean target x0.
        eps_weight: Weight of epsilon MSE term.
        x0_weight: Weight of x0 auxiliary term.
        x0_loss_type: One of {"l1", "mse", "huber"}.
    """
    loss = float(eps_weight) * F.mse_loss(eps_pred, eps_true)

    if float(x0_weight) <= 0.0:
        return loss
    if x0_pred is None or x0_true is None:
        return loss

    typ = str(x0_loss_type).lower()
    if typ == "l1":
        x0_loss = F.l1_loss(x0_pred, x0_true)
    elif typ == "mse":
        x0_loss = F.mse_loss(x0_pred, x0_true)
    elif typ == "huber":
        x0_loss = F.smooth_l1_loss(x0_pred, x0_true)
    else:
        raise ValueError(f"Unsupported x0_loss_type: {x0_loss_type}")

    return loss + float(x0_weight) * x0_loss


def build_loss_dict(loss: torch.Tensor) -> Dict[str, float]:
    """Convert loss tensor to logging dict."""
    return {"loss": float(loss.detach().item())}
