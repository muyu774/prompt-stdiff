"""Evaluation metrics for traffic forecasting."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

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


def nll(
    pred_samples: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Gaussian-fit negative log likelihood from ensemble samples.

    Args:
        pred_samples: Ensemble samples [S, B, H, N, F].
        target: Ground truth [B, H, N, F].
        eps: Minimum variance for numerical stability.

    Returns:
        Scalar Gaussian NLL averaged over all positions.
    """
    if pred_samples.dim() != 5:
        raise ValueError(f"Expected pred_samples [S,B,H,N,F], got {pred_samples.shape}")
    mean = pred_samples.mean(dim=0)
    var = torch.clamp(pred_samples.var(dim=0, unbiased=False), min=eps)
    return 0.5 * torch.mean(torch.log(2.0 * torch.pi * var) + ((target - mean) ** 2) / var)


def _central_interval(
    pred_samples: torch.Tensor,
    coverage: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return central prediction interval bounds over ensemble dim."""
    if pred_samples.dim() != 5:
        raise ValueError(f"Expected pred_samples [S,B,H,N,F], got {pred_samples.shape}")
    cov = float(coverage)
    if not (0.0 < cov < 1.0):
        raise ValueError(f"coverage must be in (0,1), got {coverage}")
    alpha = 1.0 - cov
    lower = torch.quantile(pred_samples, q=alpha / 2.0, dim=0)
    upper = torch.quantile(pred_samples, q=1.0 - alpha / 2.0, dim=0)
    return lower, upper


def winkler_score(
    pred_samples: torch.Tensor,
    target: torch.Tensor,
    coverage: float = 0.9,
) -> torch.Tensor:
    """Winkler interval score for a central prediction interval.

    Lower is better. The score rewards narrow intervals and penalizes misses.
    """
    lower, upper = _central_interval(pred_samples, coverage=coverage)
    alpha = 1.0 - float(coverage)
    width = upper - lower
    below = target < lower
    above = target > upper
    penalty_low = (2.0 / alpha) * (lower - target) * below.float()
    penalty_high = (2.0 / alpha) * (target - upper) * above.float()
    return torch.mean(width + penalty_low + penalty_high)


def picp(
    pred_samples: torch.Tensor,
    target: torch.Tensor,
    coverage: float = 0.9,
) -> torch.Tensor:
    """Prediction interval coverage probability for a central interval."""
    lower, upper = _central_interval(pred_samples, coverage=coverage)
    covered = (target >= lower) & (target <= upper)
    return covered.float().mean()


def mpiw(
    pred_samples: torch.Tensor,
    coverage: float = 0.9,
) -> torch.Tensor:
    """Mean prediction interval width for a central interval."""
    lower, upper = _central_interval(pred_samples, coverage=coverage)
    return torch.mean(upper - lower)


def sharpness(pred_samples: torch.Tensor) -> torch.Tensor:
    """Distribution sharpness as mean ensemble standard deviation."""
    if pred_samples.dim() != 5:
        raise ValueError(f"Expected pred_samples [S,B,H,N,F], got {pred_samples.shape}")
    return pred_samples.std(dim=0, unbiased=False).mean()


def reliability_curve(
    pred_samples: torch.Tensor,
    target: torch.Tensor,
    levels: Iterable[float] = tuple(i / 10.0 for i in range(1, 10)),
) -> Dict[str, float]:
    """Empirical coverage curve for nominal central intervals.

    Returns keys like ``reliability@10`` ... ``reliability@90`` where values are
    empirical coverage probabilities. A calibrated model should have
    ``reliability@90`` close to 0.90.
    """
    out: Dict[str, float] = {}
    for level in levels:
        pct = int(round(float(level) * 100))
        out[f"reliability@{pct}"] = float(picp(pred_samples, target, coverage=float(level)).item())
    return out


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mape_eps: float = 1e-5,
    mape_mask_threshold: float = 1.0,
) -> Dict[str, float]:
    """Compute point metrics and, when possible, probabilistic metrics.

    Args:
        pred: Point prediction [B, H, N, F] or ensemble [S, B, H, N, F].
        target: [B, H, N, F]
    """
    is_ensemble = pred.dim() == 5
    point_pred = pred.mean(dim=0) if is_ensemble else pred
    if point_pred.dim() != 4:
        raise ValueError(f"Expected pred [B,H,N,F] or [S,B,H,N,F], got {pred.shape}")

    out = {
        "mae": float(mae(point_pred, target).item()),
        "rmse": float(rmse(point_pred, target).item()),
        "mape": float(mape(point_pred, target, eps=mape_eps, mask_threshold=mape_mask_threshold).item()),
    }
    if is_ensemble:
        out.update(
            {
                "nll": float(nll(pred, target).item()),
                "winkler@90": float(winkler_score(pred, target, coverage=0.9).item()),
                "picp@90": float(picp(pred, target, coverage=0.9).item()),
                "mpiw@90": float(mpiw(pred, coverage=0.9).item()),
                "sharpness": float(sharpness(pred).item()),
            }
        )
        out.update(reliability_curve(pred, target))
    return out
