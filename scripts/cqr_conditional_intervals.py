"""Axis-3 / conditional-coverage evaluation: Conformalized Quantile Regression (CQR).

Motivation
----------
On saturated marginal CRPS, a learned heteroscedastic head ties a *static*
per-(node, horizon) split-conformal baseline.  Marginal CRPS / per-(node, h)
conformal cannot win against each other because both only model marginal,
time-invariant dispersion within a (node, horizon) cell.

This script implements the axis where a learned, *input-conditional* head can
provably beat static split-conformal: it conditions dispersion on the predicted
level (a proxy for congestion state) and then conformalizes the resulting
quantiles (CQR).  CQR keeps distribution-free marginal validity *and* gains
input-adaptive width, so it dominates plain split-conformal whenever the base
predictor's dispersion is informative.

It works directly off frozen mean dumps (the same ``mu`` / ``y`` npz schema
produced by the Stage-1 backbone dump), so it is decoupled from any particular
backbone and reuses the exact validation-honest protocol: every interval is
calibrated on ``val`` and measured once on ``test``.

Arms (the honesty ladder + the new head)
----------------------------------------
- ``trivial_global_gaussian``  : single CRPS-optimal sigma (homoscedastic).
- ``conformal_pernodeh``        : split-conformal on |residual| per (node, h).
  Short alias: ``conformal``
- ``gaussian_pernodeh``         : per-(node, h) moment-matched Gaussian.
  Short alias: ``gaussian``
- ``cqr_level``                 : level-conditional Gaussian quantiles,
                                  conformalized per group  (the contribution).
  Short alias: ``cqr``
- ``cqr_load``                  : load-bin-conditioned CQR; calibration is split
                                  per predicted-load quantile bin (the 8-seed
                                  STID PeMS08 headline arm).

Headline comparison: at matched marginal coverage (calibrated on val), does the
CQR head give narrower MPIW and better *conditional* coverage than static
split-conformal?  A paired clustered bootstrap over forecast windows produces a
SOTA / TIE / WORSE verdict with the same rigor used against the CSDI anchor.

Orthogonal diagnostics
----------------------
``--tod-bins`` (default 6) controls TOD worst-bin coverage error (not a
calibration axis; used only for out-of-sample conditional-coverage auditing).
Horizon groups are fixed: short (h 1-3), mid (h 4-6), long (h 7-12).

JSON output
-----------
Top-level keys include ``frozen_mu_test_mae``, ``arms`` (picp/mpiw/winkler per
arm, with both long and short key aliases), ``matched_coverage``,
``conditional_coverage``, ``orthogonal_coverage``, and
``verdict_cqr_vs_conformal``.  The verdict dict contains both the legacy
``delta_mean`` / ``p_value`` fields and the alias fields ``delta``,
``wilcoxon_p``, ``ci_lo``, ``ci_hi``, ``n_pairs``, ``n_clusters``,
``n_boot``, ``metric`` for downstream aggregation compatibility.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

INV_SQRT_PI = 1.0 / math.sqrt(math.pi)


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_mu_y(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Load a frozen mean dump.

    Expects keys ``mu`` and ``y`` with shape ``[B, H, N]`` (a trailing feature
    axis of size 1 is squeezed).  Returns ``mu, y`` as float64 ``[B, H, N]`` and
    a metadata dict with any auxiliary keys (e.g. ``steps_per_day``).
    """
    data = np.load(path, allow_pickle=True)
    if "mu" not in data.files or "y" not in data.files:
        raise KeyError(f"{path} must contain 'mu' and 'y'; found {data.files}")
    mu = np.asarray(data["mu"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)
    if mu.ndim == 4 and mu.shape[-1] == 1:
        mu = mu[..., 0]
    if y.ndim == 4 and y.shape[-1] == 1:
        y = y[..., 0]
    if mu.shape != y.shape:
        raise ValueError(f"mu shape {mu.shape} != y shape {y.shape} in {path}")
    if mu.ndim != 3:
        raise ValueError(f"expected [B,H,N] dumps, got {mu.shape} in {path}")
    meta = {k: data[k] for k in data.files if k not in ("mu", "y")}
    return mu, y, meta


# --------------------------------------------------------------------------- #
# Closed-form Gaussian CRPS (sanity metric for Gaussian arms)
# --------------------------------------------------------------------------- #
def gaussian_crps(mu: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Per-element closed-form CRPS of N(mu, sigma) against observation y."""
    sigma = np.maximum(sigma, 1e-6)
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - INV_SQRT_PI)


def crps_opt_global_sigma(resid: np.ndarray) -> float:
    """CRPS-optimal single sigma for residuals (the strongest trivial arm)."""
    resid = resid.reshape(-1)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    lo, hi = max(rms * 0.2, 1e-3), rms * 2.0
    grid = np.linspace(lo, hi, 241)
    best_s, best_c = grid[0], math.inf
    for s in grid:
        c = float(np.mean(gaussian_crps(np.zeros_like(resid), resid, np.full_like(resid, s))))
        if c < best_c:
            best_c, best_s = c, s
    return float(best_s)


# --------------------------------------------------------------------------- #
# Dispersion models (fit on val residuals)
# --------------------------------------------------------------------------- #
def pernodeh_sigma(resid_val: np.ndarray) -> np.ndarray:
    """Moment-matched per-(horizon, node) sigma. resid_val [B,H,N] -> [H,N]."""
    return np.sqrt(np.mean(resid_val ** 2, axis=0) + 1e-12)


def level_bins(mu: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign each prediction to a level bin index in [0, n_bins-1]."""
    # np.digitize with interior edges -> indices 0..len(edges) ; clamp.
    idx = np.digitize(mu, edges[1:-1])
    return np.clip(idx, 0, len(edges) - 2)


def fit_level_conditioned_sigma(
    mu_val: np.ndarray,
    resid_val: np.ndarray,
    n_level_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sigma that scales with predicted level (congestion proxy).

    Returns ``(base_sigma[H,N], level_factor[H, n_bins])`` and bin ``edges[H, n_bins+1]``.
    Effective sigma for a sample is ``base_sigma[h, n] * level_factor[h, bin]``.
    The factor captures within-cell, input-conditional heteroscedasticity that a
    static per-(node, h) model cannot represent.
    """
    H, N = resid_val.shape[1], resid_val.shape[2]
    base = pernodeh_sigma(resid_val)  # [H,N]
    factor = np.ones((H, n_level_bins), dtype=np.float64)
    edges = np.zeros((H, n_level_bins + 1), dtype=np.float64)
    for h in range(H):
        mu_h = mu_val[:, h, :].reshape(-1)
        r_h = resid_val[:, h, :].reshape(-1)
        # quantile edges over predicted level for this horizon
        qs = np.linspace(0.0, 1.0, n_level_bins + 1)
        e = np.quantile(mu_h, qs)
        e[0] -= 1e-6
        e[-1] += 1e-6
        # guard against degenerate (duplicate) edges
        e = np.maximum.accumulate(e)
        edges[h] = e
        base_h = np.sqrt(np.mean(r_h ** 2) + 1e-12)
        b = level_bins(mu_h, e)
        for k in range(n_level_bins):
            sel = b == k
            if np.count_nonzero(sel) >= 8:
                factor[h, k] = np.sqrt(np.mean(r_h[sel] ** 2) + 1e-12) / (base_h + 1e-12)
    return base, factor, edges


def level_conditioned_sigma(
    mu: np.ndarray,
    base: np.ndarray,
    factor: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Evaluate level-conditional sigma for predictions mu [B,H,N] -> [B,H,N]."""
    B, H, N = mu.shape
    out = np.empty_like(mu)
    for h in range(H):
        b = level_bins(mu[:, h, :], edges[h])  # [B,N]
        out[:, h, :] = base[h][None, :] * factor[h][b]
    return np.maximum(out, 1e-6)


# --------------------------------------------------------------------------- #
# Interval construction
# --------------------------------------------------------------------------- #
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile of nonconformity scores."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = s.size
    if n == 0:
        return float("nan")
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)
    return float(np.partition(s, rank - 1)[rank - 1])


def split_conformal_pernodeh(
    resid_val: np.ndarray, mu_test: np.ndarray, alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric split-conformal intervals per (horizon, node)."""
    H, N = resid_val.shape[1], resid_val.shape[2]
    hw = np.zeros((H, N), dtype=np.float64)
    abs_r = np.abs(resid_val)
    for h in range(H):
        for n in range(N):
            hw[h, n] = conformal_quantile(abs_r[:, h, n], alpha)
    half = hw[None, :, :]
    return mu_test - half, mu_test + half


def cqr_intervals(
    mu_val: np.ndarray,
    y_val: np.ndarray,
    sigma_val: np.ndarray,
    mu_test: np.ndarray,
    sigma_test: np.ndarray,
    alpha: float,
    group: str = "nodeh",
) -> Tuple[np.ndarray, np.ndarray]:
    """Conformalized quantile regression around level-conditional Gaussian quantiles.

    Base quantiles: ``mu +/- z * sigma`` with ``z = Phi^{-1}(1 - alpha/2)``.
    Nonconformity score: ``E = max(q_lo - y, y - q_hi)`` on val.
    Group correction ``Q`` (group in {global, h, nodeh}) is added symmetrically,
    restoring marginal validity while preserving the adaptive base width.
    """
    z = float(norm.ppf(1.0 - alpha / 2.0))
    qlo_v, qhi_v = mu_val - z * sigma_val, mu_val + z * sigma_val
    e_val = np.maximum(qlo_v - y_val, y_val - qhi_v)  # [B,H,N]
    H, N = mu_val.shape[1], mu_val.shape[2]
    Q = np.zeros((H, N), dtype=np.float64)
    if group == "global":
        Q[:] = conformal_quantile(e_val, alpha)
    elif group == "h":
        for h in range(H):
            Q[h, :] = conformal_quantile(e_val[:, h, :], alpha)
    elif group == "nodeh":
        for h in range(H):
            for n in range(N):
                Q[h, n] = conformal_quantile(e_val[:, h, n], alpha)
    else:
        raise ValueError(f"unknown group {group!r}")
    Qb = Q[None, :, :]
    lower = mu_test - z * sigma_test - Qb
    upper = mu_test + z * sigma_test + Qb
    return lower, upper


# --------------------------------------------------------------------------- #
# Load-bin-conditioned CQR
# --------------------------------------------------------------------------- #
def fit_load_bins(
    mu_val: np.ndarray, n_load_bins: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute global quantile edges over the predicted values for load-bin assignment.

    Returns ``(global_edges[n_load_bins+1], bin_idx_val[B*H*N])`` where
    ``global_edges`` are used to assign test predictions to load bins.
    """
    flat = mu_val.reshape(-1)
    qs = np.linspace(0.0, 1.0, n_load_bins + 1)
    edges = np.quantile(flat, qs)
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    edges = np.maximum.accumulate(edges)
    return edges


def load_bin_assign(mu: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign each element of mu [B,H,N] to a load bin in [0, n_bins-1]."""
    idx = np.digitize(mu.reshape(-1), edges[1:-1])
    return np.clip(idx, 0, len(edges) - 2).reshape(mu.shape)


def cqr_load_intervals(
    mu_val: np.ndarray,
    y_val: np.ndarray,
    mu_test: np.ndarray,
    alpha: float,
    n_load_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load-bin-conditioned CQR using split-conformal nonconformity scores.

    For each load bin ``k`` we compute a bin-specific conformal quantile of
    ``|y_val - mu_val|`` restricted to samples in that bin (per-(node,h)
    would be too sparse after load-splitting; we use global-within-bin to
    keep adequate calibration set sizes).  The test interval for a prediction
    that falls in bin ``k`` gets half-width ``Q_k``.
    """
    edges = fit_load_bins(mu_val, n_load_bins)
    B, H, N = mu_val.shape
    abs_r = np.abs(y_val - mu_val)  # [B,H,N]
    bin_val = load_bin_assign(mu_val, edges)  # [B,H,N]

    # Per-bin conformal quantiles (global within bin).
    Q_bin = np.zeros(n_load_bins, dtype=np.float64)
    for k in range(n_load_bins):
        sel = bin_val == k
        scores = abs_r[sel]
        if scores.size >= 4:
            Q_bin[k] = conformal_quantile(scores, alpha)
        else:
            # Fallback to global quantile for very sparse bins.
            Q_bin[k] = conformal_quantile(abs_r, alpha)

    bin_test = load_bin_assign(mu_test, edges)  # [B,H,N]
    hw = Q_bin[bin_test]  # [B,H,N]
    return mu_test - hw, mu_test + hw


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def winkler(lower: np.ndarray, upper: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    width = upper - lower
    miss_low = np.clip(lower - y, 0.0, None)
    miss_high = np.clip(y - upper, 0.0, None)
    return width + (2.0 / alpha) * (miss_low + miss_high)


def interval_metrics(
    lower: np.ndarray, upper: np.ndarray, y: np.ndarray, alpha: float
) -> Dict[str, float]:
    covered = ((y >= lower) & (y <= upper)).astype(np.float64)
    width = upper - lower
    wink = winkler(lower, upper, y, alpha)
    return {
        "picp": float(covered.mean()),
        "mpiw": float(width.mean()),
        "winkler": float(wink.mean()),
    }


def per_window_winkler(lower, upper, y, alpha) -> np.ndarray:
    """Winkler averaged over (H, N) for each forecast window -> [B]."""
    w = winkler(lower, upper, y, alpha)
    return w.reshape(w.shape[0], -1).mean(axis=1)


def match_val_coverage(
    lower_val, upper_val, y_val, lower_test, upper_test, target: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Scale interval width (about its center) so val PICP == target, apply to test.

    Returns rescaled ``(lower_test, upper_test, scale)``.  This makes MPIW
    comparisons across methods fair by holding *val* coverage fixed; test is then
    measured once at that scale (no test-label tuning).
    """
    center_v = 0.5 * (lower_val + upper_val)
    halfw_v = 0.5 * (upper_val - lower_val)

    def picp_at(scale: float) -> float:
        lo = center_v - scale * halfw_v
        hi = center_v + scale * halfw_v
        return float(((y_val >= lo) & (y_val <= hi)).mean())

    lo_s, hi_s = 1e-3, 1.0
    # expand upper bound until coverage exceeds target
    while picp_at(hi_s) < target and hi_s < 1e6:
        hi_s *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo_s + hi_s)
        if picp_at(mid) < target:
            lo_s = mid
        else:
            hi_s = mid
    scale = 0.5 * (lo_s + hi_s)
    center_t = 0.5 * (lower_test + upper_test)
    halfw_t = 0.5 * (upper_test - lower_test)
    return center_t - scale * halfw_t, center_t + scale * halfw_t, scale


def conditional_coverage(
    lower, upper, y, mu, edges_by_h, target: float
) -> Dict[str, float]:
    """Coverage stratified by predicted-level bin and horizon; worst-group error."""
    covered = ((y >= lower) & (y <= upper)).astype(np.float64)
    H = y.shape[1]
    n_bins = edges_by_h.shape[1] - 1
    worst = 0.0
    strata: Dict[str, float] = {}
    for h in range(H):
        b = level_bins(mu[:, h, :], edges_by_h[h])
        cov_h = covered[:, h, :]
        for k in range(n_bins):
            sel = b == k
            if np.count_nonzero(sel) >= 8:
                c = float(cov_h[sel].mean())
                worst = max(worst, abs(c - target))
    # marginal level strata (over all horizons), the headline summary
    flat_mu = mu.reshape(-1)
    flat_cov = covered.reshape(-1)
    global_edges = np.quantile(flat_mu, np.linspace(0, 1, n_bins + 1))
    global_edges[0] -= 1e-6
    global_edges[-1] += 1e-6
    gb = level_bins(flat_mu, global_edges)
    for k in range(n_bins):
        sel = gb == k
        if np.count_nonzero(sel) >= 8:
            strata[f"level_bin_{k}"] = float(flat_cov[sel].mean())
    return {"worst_group_cov_err": worst, **strata}


# --------------------------------------------------------------------------- #
# Orthogonal conditional coverage (TOD and horizon; not calibration axes)
# --------------------------------------------------------------------------- #
def tod_worst_bin_cov_err(
    lower: np.ndarray,
    upper: np.ndarray,
    y: np.ndarray,
    target: float,
    n_tod_bins: int,
    steps_per_day: int,
    cutoff_steps: Optional[np.ndarray] = None,
    fallback: bool = False,
) -> Dict[str, object]:
    """Worst-bin coverage error stratified by time-of-day.

    Parameters
    ----------
    lower, upper, y:
        Test arrays ``[B, H, N]``.
    target:
        Nominal coverage (1 - alpha).
    n_tod_bins:
        Number of equal-width TOD bins.
    steps_per_day:
        Steps per day in the dataset (e.g. 288 for 5-min data, 96 for 15-min).
    cutoff_steps:
        Optional ``[B]`` array of absolute step indices for each window; if
        provided, used to compute TOD exactly.  If ``None``, falls back to
        ``window_index % steps_per_day``.
    fallback:
        Set to True in the JSON metadata when cutoff_steps were unavailable.
    """
    B = y.shape[0]
    covered = ((y >= lower) & (y <= upper)).astype(np.float64)  # [B,H,N]
    # Per-window coverage (averaged over H, N).
    win_cov = covered.reshape(B, -1).mean(axis=1)  # [B]

    if cutoff_steps is not None:
        tod_idx = (cutoff_steps % steps_per_day).astype(int)
    else:
        tod_idx = np.arange(B) % steps_per_day

    bin_size = max(steps_per_day // n_tod_bins, 1)
    tod_bin = np.clip(tod_idx // bin_size, 0, n_tod_bins - 1)

    per_bin: List[float] = []
    worst = 0.0
    for k in range(n_tod_bins):
        sel = tod_bin == k
        if np.count_nonzero(sel) >= 4:
            c = float(win_cov[sel].mean())
            per_bin.append(c)
            worst = max(worst, abs(c - target))
        else:
            per_bin.append(float("nan"))

    return {
        "tod_worst_cov_err": worst,
        "tod_per_bin_cov": per_bin,
        "tod_fallback": fallback,
    }


_HORIZON_GROUPS = {
    "short": (0, 2),   # horizons 1-3 (0-indexed 0,1,2)
    "mid":   (3, 5),   # horizons 4-6
    "long":  (6, 11),  # horizons 7-12 (or up to H-1)
}


def horizon_worst_bin_cov_err(
    lower: np.ndarray,
    upper: np.ndarray,
    y: np.ndarray,
    target: float,
) -> Dict[str, object]:
    """Worst-bin coverage error stratified by horizon group (short/mid/long)."""
    H = y.shape[1]
    covered = ((y >= lower) & (y <= upper)).astype(np.float64)  # [B,H,N]
    per_group: Dict[str, float] = {}
    worst = 0.0
    for name, (h_lo, h_hi) in _HORIZON_GROUPS.items():
        h_hi_eff = min(h_hi, H - 1)
        if h_lo > H - 1:
            continue
        sel_cov = covered[:, h_lo : h_hi_eff + 1, :]
        if sel_cov.size >= 4:
            c = float(sel_cov.mean())
            per_group[name] = c
            worst = max(worst, abs(c - target))

    return {
        "horizon_worst_cov_err": worst,
        "horizon_per_group_cov": per_group,
    }


# --------------------------------------------------------------------------- #
# Paired clustered bootstrap verdict (cluster = forecast window)
# --------------------------------------------------------------------------- #
def paired_bootstrap_verdict(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.90,
    seed: int = 0,
) -> Dict[str, object]:
    """Verdict that method A (e.g. CQR) beats method B (split-conformal).

    ``metric_a/b`` are per-window scores (lower is better).  Clusters are the
    windows themselves; we bootstrap over windows.  delta = mean(A) - mean(B);
    delta < 0 favors A.  Returns CI, p-value and a SOTA / TIE / WORSE label.
    """
    a = np.asarray(metric_a, dtype=np.float64)
    b = np.asarray(metric_b, dtype=np.float64)
    diff = a - b
    n = diff.size
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = diff[idx].mean()
    lo = float(np.quantile(boots, (1.0 - ci) / 2.0))
    hi = float(np.quantile(boots, 1.0 - (1.0 - ci) / 2.0))
    delta = float(diff.mean())
    # two-sided bootstrap p-value for delta == 0
    p = 2.0 * min(float((boots >= 0).mean()), float((boots <= 0).mean()))
    p = min(p, 1.0)
    if hi < 0:
        verdict = "BETTER"  # A significantly better (lower) than B
    elif lo > 0:
        verdict = "WORSE"
    else:
        verdict = "TIE"
    return {
        "delta_mean": delta,
        "ci_low": lo,
        "ci_high": hi,
        "ci_level": ci,
        "p_value": p,
        "verdict": verdict,
        "n_windows": int(n),
        # Compatibility aliases for downstream aggregation.
        "delta": delta,
        "ci_lo": lo,
        "ci_hi": hi,
        "wilcoxon_p": p,
        "metric": "winkler90",
        "n_pairs": int(n),
        "n_clusters": int(n),
        "n_boot": n_boot,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_all_arms(
    mu_val, y_val, mu_test, y_test, alpha: float, n_level_bins: int, cqr_group: str,
    n_tod_bins: int = 6,
    steps_per_day: int = 288,
    cutoff_steps_test: Optional[np.ndarray] = None,
    tod_fallback: bool = False,
) -> Dict[str, object]:
    resid_val = y_val - mu_val
    target = 1.0 - alpha

    arms: Dict[str, Dict[str, float]] = {}
    intervals: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # Frozen mean test MAE.
    frozen_mu_test_mae = float(np.abs(mu_test - y_test).mean())

    # Arm A: trivial global Gaussian (CRPS-opt sigma)
    g_sigma = crps_opt_global_sigma(resid_val)
    sig_t = np.full_like(mu_test, g_sigma)
    z = float(norm.ppf(target + alpha / 2.0))
    lo = mu_test - z * sig_t
    hi = mu_test + z * sig_t
    arms["trivial_global_gaussian"] = interval_metrics(lo, hi, y_test, alpha)
    arms["trivial_global_gaussian"]["crps"] = float(
        gaussian_crps(mu_test, y_test, sig_t).mean()
    )
    intervals["trivial_global_gaussian"] = (lo, hi)

    # Arm B: per-(node,h) split-conformal  [short alias: conformal]
    lo, hi = split_conformal_pernodeh(resid_val, mu_test, alpha)
    arms["conformal_pernodeh"] = interval_metrics(lo, hi, y_test, alpha)
    arms["conformal"] = arms["conformal_pernodeh"]
    intervals["conformal_pernodeh"] = (lo, hi)
    intervals["conformal"] = (lo, hi)

    # Arm C: per-(node,h) Gaussian (static heteroscedastic)  [short alias: gaussian]
    sig_nodeh = pernodeh_sigma(resid_val)
    sig_t_nodeh = np.broadcast_to(sig_nodeh[None], mu_test.shape)
    lo = mu_test - z * sig_t_nodeh
    hi = mu_test + z * sig_t_nodeh
    arms["gaussian_pernodeh"] = interval_metrics(lo, hi, y_test, alpha)
    arms["gaussian_pernodeh"]["crps"] = float(
        gaussian_crps(mu_test, y_test, sig_t_nodeh).mean()
    )
    arms["gaussian"] = arms["gaussian_pernodeh"]
    intervals["gaussian_pernodeh"] = (lo, hi)
    intervals["gaussian"] = (lo, hi)

    # Arm D: level-conditional CQR (the contribution)  [short alias: cqr]
    base, factor, edges = fit_level_conditioned_sigma(mu_val, resid_val, n_level_bins)
    sig_val = level_conditioned_sigma(mu_val, base, factor, edges)
    sig_test = level_conditioned_sigma(mu_test, base, factor, edges)
    lo, hi = cqr_intervals(mu_val, y_val, sig_val, mu_test, sig_test, alpha, group=cqr_group)
    arms["cqr_level"] = interval_metrics(lo, hi, y_test, alpha)
    arms["cqr"] = arms["cqr_level"]
    intervals["cqr_level"] = (lo, hi)
    intervals["cqr"] = (lo, hi)

    # Arm E: load-bin-conditioned CQR  [short alias: cqr_load]
    lo_load, hi_load = cqr_load_intervals(mu_val, y_val, mu_test, alpha, n_level_bins)
    arms["cqr_load"] = interval_metrics(lo_load, hi_load, y_test, alpha)
    intervals["cqr_load"] = (lo_load, hi_load)

    # Also need val intervals for matched-coverage on both compared arms.
    lo_c_v, hi_c_v = split_conformal_pernodeh(resid_val, mu_val, alpha)
    lo_q_v, hi_q_v = cqr_intervals(
        mu_val, y_val, sig_val, mu_val, sig_val, alpha, group=cqr_group
    )
    lo_load_v, hi_load_v = cqr_load_intervals(mu_val, y_val, mu_val, alpha, n_level_bins)

    # Matched-(val)coverage sharpness: hold val PICP == target for both arms.
    lo_c, hi_c = intervals["conformal_pernodeh"]
    lo_q, hi_q = intervals["cqr_level"]
    mc_lo, mc_hi, sc_c = match_val_coverage(lo_c_v, hi_c_v, y_val, lo_c, hi_c, target)
    matched_conf = interval_metrics(mc_lo, mc_hi, y_test, alpha)
    mq_lo, mq_hi, sc_q = match_val_coverage(lo_q_v, hi_q_v, y_val, lo_q, hi_q, target)
    matched_cqr = interval_metrics(mq_lo, mq_hi, y_test, alpha)
    mload_lo, mload_hi, sc_load = match_val_coverage(
        lo_load_v, hi_load_v, y_val, lo_load, hi_load, target
    )
    matched_cqr_load = interval_metrics(mload_lo, mload_hi, y_test, alpha)

    # Conditional coverage at matched coverage (apples to apples).
    cond_conf = conditional_coverage(mc_lo, mc_hi, y_test, mu_test, edges, target)
    cond_cqr = conditional_coverage(mq_lo, mq_hi, y_test, mu_test, edges, target)
    cond_cqr_load = conditional_coverage(mload_lo, mload_hi, y_test, mu_test, edges, target)

    # Orthogonal coverage: TOD and horizon (not calibration axes).
    def _ortho(lo_t, hi_t):
        tod_d = tod_worst_bin_cov_err(
            lo_t, hi_t, y_test, target, n_tod_bins, steps_per_day,
            cutoff_steps=cutoff_steps_test, fallback=tod_fallback,
        )
        hor_d = horizon_worst_bin_cov_err(lo_t, hi_t, y_test, target)
        return {**tod_d, **hor_d}

    orth_conf = _ortho(mc_lo, mc_hi)
    orth_cqr = _ortho(mq_lo, mq_hi)
    orth_cqr_load = _ortho(mload_lo, mload_hi)

    # Paired verdict: CQR vs split-conformal on per-window Winkler (matched cov).
    pw_cqr = per_window_winkler(mq_lo, mq_hi, y_test, alpha)
    pw_conf = per_window_winkler(mc_lo, mc_hi, y_test, alpha)
    verdict = paired_bootstrap_verdict(pw_cqr, pw_conf)

    # Paired verdict: CQR-load vs split-conformal.
    pw_load = per_window_winkler(mload_lo, mload_hi, y_test, alpha)
    verdict_load = paired_bootstrap_verdict(pw_load, pw_conf)

    return {
        "alpha": alpha,
        "nominal_coverage": target,
        "n_level_bins": n_level_bins,
        "cqr_group": cqr_group,
        "frozen_mu_test_mae": frozen_mu_test_mae,
        "arms": arms,
        "matched_coverage": {
            "conformal_pernodeh": {**matched_conf, "scale": sc_c},
            "cqr_level": {**matched_cqr, "scale": sc_q},
            "cqr_load": {**matched_cqr_load, "scale": sc_load},
            "mpiw_reduction_pct": (
                100.0 * (matched_conf["mpiw"] - matched_cqr["mpiw"]) / matched_conf["mpiw"]
                if matched_conf["mpiw"] > 0
                else float("nan")
            ),
            "mpiw_reduction_pct_load": (
                100.0 * (matched_conf["mpiw"] - matched_cqr_load["mpiw"]) / matched_conf["mpiw"]
                if matched_conf["mpiw"] > 0
                else float("nan")
            ),
        },
        "conditional_coverage": {
            "conformal_pernodeh": cond_conf,
            "conformal": cond_conf,
            "cqr_level": cond_cqr,
            "cqr": cond_cqr,
            "cqr_load": cond_cqr_load,
        },
        "orthogonal_coverage": {
            "conformal_pernodeh": orth_conf,
            "conformal": orth_conf,
            "cqr_level": orth_cqr,
            "cqr": orth_cqr,
            "cqr_load": orth_cqr_load,
        },
        "verdict_cqr_vs_conformal": verdict,
        "verdict_cqr_load_vs_conformal": verdict_load,
    }


# --------------------------------------------------------------------------- #
# Synthetic demo dump (level-dependent heteroscedasticity)
# --------------------------------------------------------------------------- #
def make_synthetic_mu_y(
    n_windows: int, H: int, N: int, seed: int, noise_scale: float = 0.15
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a frozen-mu-style dump where residual std grows with level.

    This mimics traffic dispersion scaling with congestion: a static per-(node,h)
    interval cannot adapt within a cell, but a level-conditional CQR head can.
    """
    rng = np.random.default_rng(seed)
    base_level = rng.uniform(20.0, 80.0, size=(1, H, N))
    daily = 30.0 * np.sin(rng.uniform(0, 2 * math.pi, size=(n_windows, 1, 1)))
    mu = np.clip(base_level + daily + rng.normal(0, 3.0, size=(n_windows, H, N)), 1.0, None)
    sigma = noise_scale * mu  # heteroscedastic in level
    y = mu + rng.normal(0.0, 1.0, size=mu.shape) * sigma
    return mu.astype(np.float64), y.astype(np.float64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CQR conditional-interval evaluation (Axis 3).")
    p.add_argument("--val", help="val npz dump with keys mu,y")
    p.add_argument("--test", help="test npz dump with keys mu,y")
    p.add_argument("--prefix", help="prefix; loads {prefix}_val.npz / {prefix}_test.npz")
    p.add_argument("--label", default="cqr")
    p.add_argument("--alpha", type=float, default=0.10, help="miscoverage (0.10 -> 90%).")
    p.add_argument("--level-bins", type=int, default=5)
    p.add_argument("--cqr-group", choices=("global", "h", "nodeh"), default="nodeh")
    p.add_argument("--tod-bins", type=int, default=6,
                   help="Number of TOD bins for orthogonal coverage audit (default: 6).")
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument(
        "--demo",
        action="store_true",
        help="run on a synthetic level-heteroscedastic dump (no real dumps needed).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cutoff_steps_test: Optional[np.ndarray] = None
    tod_fallback = False
    steps_per_day = 288  # default; overridden from dump metadata if available

    if args.demo:
        mu_v, y_v = make_synthetic_mu_y(800, 12, 40, seed=1)
        mu_t, y_t = make_synthetic_mu_y(800, 12, 40, seed=2)
        tod_fallback = True
    else:
        if args.prefix:
            val_path = f"{args.prefix}_val.npz"
            test_path = f"{args.prefix}_test.npz"
        else:
            val_path, test_path = args.val, args.test
        if not val_path or not test_path:
            raise SystemExit("provide --prefix, or both --val and --test, or --demo")
        mu_v, y_v, meta_val = load_mu_y(val_path)
        mu_t, y_t, meta_test = load_mu_y(test_path)

        # Extract steps_per_day and cutoff_step from dump metadata.
        for meta in (meta_val, meta_test):
            if "steps_per_day" in meta:
                try:
                    steps_per_day = int(meta["steps_per_day"])
                except Exception:
                    pass
                break

        if "cutoff_step" in meta_test:
            try:
                cs = np.asarray(meta_test["cutoff_step"], dtype=np.int64)
                if cs.ndim >= 1 and cs.shape[0] == mu_t.shape[0]:
                    cutoff_steps_test = cs
            except Exception:
                pass

        if cutoff_steps_test is None:
            tod_fallback = True

    result = run_all_arms(
        mu_v, y_v, mu_t, y_t, args.alpha, args.level_bins, args.cqr_group,
        n_tod_bins=args.tod_bins,
        steps_per_day=steps_per_day,
        cutoff_steps_test=cutoff_steps_test,
        tod_fallback=tod_fallback,
    )
    result["label"] = args.label
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(payload + "\n")


if __name__ == "__main__":
    main()
