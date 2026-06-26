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
- ``gaussian_pernodeh``         : per-(node, h) moment-matched Gaussian.
- ``cqr_level``                 : level-conditional Gaussian quantiles,
                                  conformalized per group  (the contribution).

Headline comparison: at matched marginal coverage (calibrated on val), does the
CQR head give narrower MPIW and better *conditional* coverage than static
split-conformal?  A paired clustered bootstrap over forecast windows produces a
SOTA / TIE / WORSE verdict with the same rigor used against the CSDI anchor.
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
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_all_arms(
    mu_val, y_val, mu_test, y_test, alpha: float, n_level_bins: int, cqr_group: str
) -> Dict[str, object]:
    resid_val = y_val - mu_val
    target = 1.0 - alpha

    arms: Dict[str, Dict[str, float]] = {}
    intervals: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

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

    # Arm B: per-(node,h) split-conformal
    lo, hi = split_conformal_pernodeh(resid_val, mu_test, alpha)
    arms["conformal_pernodeh"] = interval_metrics(lo, hi, y_test, alpha)
    intervals["conformal_pernodeh"] = (lo, hi)

    # Arm C: per-(node,h) Gaussian (static heteroscedastic)
    sig_nodeh = pernodeh_sigma(resid_val)
    sig_t = np.broadcast_to(sig_nodeh[None], mu_test.shape)
    lo = mu_test - z * sig_t
    hi = mu_test + z * sig_t
    arms["gaussian_pernodeh"] = interval_metrics(lo, hi, y_test, alpha)
    arms["gaussian_pernodeh"]["crps"] = float(gaussian_crps(mu_test, y_test, sig_t).mean())
    intervals["gaussian_pernodeh"] = (lo, hi)

    # Arm D: level-conditional CQR (the contribution)
    base, factor, edges = fit_level_conditioned_sigma(mu_val, resid_val, n_level_bins)
    sig_val = level_conditioned_sigma(mu_val, base, factor, edges)
    sig_test = level_conditioned_sigma(mu_test, base, factor, edges)
    lo, hi = cqr_intervals(mu_val, y_val, sig_val, mu_test, sig_test, alpha, group=cqr_group)
    arms["cqr_level"] = interval_metrics(lo, hi, y_test, alpha)
    intervals["cqr_level"] = (lo, hi)

    # Also need val intervals for matched-coverage on both compared arms.
    lo_c_v, hi_c_v = split_conformal_pernodeh(resid_val, mu_val, alpha)
    lo_q_v, hi_q_v = cqr_intervals(
        mu_val, y_val, sig_val, mu_val, sig_val, alpha, group=cqr_group
    )

    # Matched-(val)coverage sharpness: hold val PICP == target for both arms.
    lo_c, hi_c = intervals["conformal_pernodeh"]
    lo_q, hi_q = intervals["cqr_level"]
    mc_lo, mc_hi, sc_c = match_val_coverage(lo_c_v, hi_c_v, y_val, lo_c, hi_c, target)
    matched_conf = interval_metrics(mc_lo, mc_hi, y_test, alpha)
    mq_lo, mq_hi, sc_q = match_val_coverage(lo_q_v, hi_q_v, y_val, lo_q, hi_q, target)
    matched_cqr = interval_metrics(mq_lo, mq_hi, y_test, alpha)

    # Conditional coverage at matched coverage (apples to apples).
    cond_conf = conditional_coverage(mc_lo, mc_hi, y_test, mu_test, edges, target)
    cond_cqr = conditional_coverage(mq_lo, mq_hi, y_test, mu_test, edges, target)

    # Paired verdict: CQR vs split-conformal on per-window Winkler (matched cov).
    pw_cqr = per_window_winkler(mq_lo, mq_hi, y_test, alpha)
    pw_conf = per_window_winkler(mc_lo, mc_hi, y_test, alpha)
    verdict = paired_bootstrap_verdict(pw_cqr, pw_conf)

    return {
        "alpha": alpha,
        "nominal_coverage": target,
        "n_level_bins": n_level_bins,
        "cqr_group": cqr_group,
        "arms": arms,
        "matched_coverage": {
            "conformal_pernodeh": {**matched_conf, "scale": sc_c},
            "cqr_level": {**matched_cqr, "scale": sc_q},
            "mpiw_reduction_pct": (
                100.0 * (matched_conf["mpiw"] - matched_cqr["mpiw"]) / matched_conf["mpiw"]
                if matched_conf["mpiw"] > 0
                else float("nan")
            ),
        },
        "conditional_coverage": {
            "conformal_pernodeh": cond_conf,
            "cqr_level": cond_cqr,
        },
        "verdict_cqr_vs_conformal": verdict,
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
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument(
        "--demo",
        action="store_true",
        help="run on a synthetic level-heteroscedastic dump (no real dumps needed).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo:
        mu_v, y_v = make_synthetic_mu_y(800, 12, 40, seed=1)
        mu_t, y_t = make_synthetic_mu_y(800, 12, 40, seed=2)
    else:
        if args.prefix:
            val_path = f"{args.prefix}_val.npz"
            test_path = f"{args.prefix}_test.npz"
        else:
            val_path, test_path = args.val, args.test
        if not val_path or not test_path:
            raise SystemExit("provide --prefix, or both --val and --test, or --demo")
        mu_v, y_v, _ = load_mu_y(val_path)
        mu_t, y_t, _ = load_mu_y(test_path)

    result = run_all_arms(
        mu_v, y_v, mu_t, y_t, args.alpha, args.level_bins, args.cqr_group
    )
    result["label"] = args.label
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(payload + "\n")


if __name__ == "__main__":
    main()
