"""Evaluation metrics for traffic forecasting."""

from __future__ import annotations

from typing import Dict

import torch


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error."""
    return torch.mean(torch.abs(pred - target))


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root mean squared error."""
    return torch.sqrt(torch.mean((pred - target) ** 2))


def mape(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-5,
    mask_threshold: float = 1.0,
) -> torch.Tensor:
    """Mean absolute percentage error with robust masking.

    Args:
        pred: Prediction tensor.
        target: Ground truth tensor.
        eps: Small value for denominator stability.
        mask_threshold: Only positions with |target| > mask_threshold are used.
    """
    denom = torch.clamp(torch.abs(target), min=eps)
    ratio = torch.abs((pred - target) / denom)

    if mask_threshold > 0:
        mask = torch.abs(target) > mask_threshold
        valid = mask.sum()
        if int(valid.item()) > 0:
            return ratio[mask].mean()
    return ratio.mean()


def crps_ensemble(pred_samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CRPS for ensemble samples.

    Args:
        pred_samples: [S, B, H, N, F]
        target: [B, H, N, F]

    Returns:
        Scalar CRPS.
    """
    if pred_samples.dim() != 5:
        raise ValueError(f"Expected pred_samples [S,B,H,N,F], got {pred_samples.shape}")

    term1 = torch.mean(torch.abs(pred_samples - target.unsqueeze(0)))

    s = pred_samples.shape[0]
    pairwise = torch.abs(pred_samples.unsqueeze(0) - pred_samples.unsqueeze(1))
    term2 = 0.5 * torch.mean(pairwise)

    if s <= 1:
        return term1
    return term1 - term2


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mape_eps: float = 1e-5,
    mape_mask_threshold: float = 1.0,
) -> Dict[str, float]:
    """Compute MAE/RMSE/MAPE for one prediction tensor.

    Args:
        pred: [B, H, N, F]
        target: [B, H, N, F]
    """
    out = {
        "mae": float(mae(pred, target).item()),
        "rmse": float(rmse(pred, target).item()),
        "mape": float(mape(pred, target, eps=mape_eps, mask_threshold=mape_mask_threshold).item()),
    }
    return out
