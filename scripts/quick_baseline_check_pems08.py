"""Quick baseline diagnostics for PeMS08 (and compatible datasets).

This script is used to quickly verify whether the target task setting is reasonable
before spending GPU time on complex diffusion models.

It reports MAE/RMSE at horizons 3/6/12 for:
- persistence (last value)
- moving average
- seasonal naive
- ridge linear autoregression
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataio.split import split_time_series


def _load_array(data_path: Path) -> np.ndarray:
    """Load traffic array from npz/npy as [T, N, F]."""
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    if data_path.suffix == ".npz":
        bundle = np.load(data_path)
        if "data" in bundle:
            arr = bundle["data"]
        elif "x" in bundle:
            arr = bundle["x"]
        elif "arr_0" in bundle:
            arr = bundle["arr_0"]
        else:
            raise KeyError(f"No supported key in {data_path}, keys={bundle.files}")
    elif data_path.suffix == ".npy":
        arr = np.load(data_path)
    else:
        raise ValueError(f"Unsupported data file: {data_path}")

    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Expected [T,N,F], got {arr.shape}")
    return arr.astype(np.float32)


def _build_windows_2d(
    seq_2d: np.ndarray,
    history_steps: int,
    max_horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build windows on one split.

    Args:
        seq_2d: [T, N] for one split only.
        history_steps: history length.
        max_horizon: max target horizon.

    Returns:
        x_hist: [M, N, T_his]
        y_fut: [M, N, H_max]
    """
    t_total, n_nodes = seq_2d.shape
    max_start = t_total - history_steps - max_horizon
    if max_start < 0:
        raise ValueError(
            f"Split length is too short: T={t_total}, history={history_steps}, max_h={max_horizon}"
        )

    m = max_start + 1
    x_hist = np.zeros((m, n_nodes, history_steps), dtype=np.float32)
    y_fut = np.zeros((m, n_nodes, max_horizon), dtype=np.float32)
    for i, start in enumerate(range(m)):
        his_end = start + history_steps
        fut_end = his_end + max_horizon
        x_hist[i] = seq_2d[start:his_end].T
        y_fut[i] = seq_2d[his_end:fut_end].T
    return x_hist, y_fut


def _mae_rmse(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    err = pred - target
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return {"mae": mae, "rmse": rmse}


def _fit_ridge(
    x_hist: np.ndarray,
    y_h: np.ndarray,
    ridge_lambda: float,
    max_train_points: int,
    seed: int,
) -> np.ndarray:
    """Fit ridge linear model y = Xw + b.

    Args:
        x_hist: [M, N, T_his]
        y_h: [M, N]
    Returns:
        weight vector [T_his + 1], with last dim as bias.
    """
    m, n, t_his = x_hist.shape
    x = x_hist.reshape(m * n, t_his)
    y = y_h.reshape(m * n)

    if max_train_points > 0 and x.shape[0] > max_train_points:
        rng = np.random.default_rng(seed=seed)
        idx = rng.choice(x.shape[0], size=max_train_points, replace=False)
        x = x[idx]
        y = y[idx]

    ones = np.ones((x.shape[0], 1), dtype=np.float32)
    x_aug = np.concatenate([x, ones], axis=1)  # [P, T_his+1]

    xtx = x_aug.T @ x_aug
    reg = np.eye(xtx.shape[0], dtype=np.float32) * float(ridge_lambda)
    reg[-1, -1] = 0.0  # no bias regularization
    xty = x_aug.T @ y
    w = np.linalg.solve(xtx + reg, xty).astype(np.float32)
    return w


def _predict_ridge(x_hist: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Predict with fitted ridge model."""
    m, n, t_his = x_hist.shape
    x = x_hist.reshape(m * n, t_his)
    ones = np.ones((x.shape[0], 1), dtype=np.float32)
    x_aug = np.concatenate([x, ones], axis=1)
    y = x_aug @ w
    return y.reshape(m, n).astype(np.float32)


def _print_block(title: str, rows: List[Tuple[str, Dict[int, Dict[str, float]]]], horizons: List[int]) -> None:
    print(f"\n=== {title} ===")
    hdr = ["method"] + [f"h{h}_mae" for h in horizons] + [f"h{h}_rmse" for h in horizons]
    print(",".join(hdr))
    for name, metrics in rows:
        vals: List[str] = [name]
        for h in horizons:
            vals.append(f"{metrics[h]['mae']:.6f}")
        for h in horizons:
            vals.append(f"{metrics[h]['rmse']:.6f}")
        print(",".join(vals))


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick baseline diagnostics for PeMS08")
    parser.add_argument("--data_file", type=Path, default=Path("data/pems08/data.npz"))
    parser.add_argument("--feature_index", type=int, default=0)
    parser.add_argument("--history_steps", type=int, default=12)
    parser.add_argument("--horizons", type=str, default="3,6,12")
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--ma_window", type=int, default=3)
    parser.add_argument("--seasonal_lag", type=int, default=12)
    parser.add_argument("--ridge_lambda", type=float, default=1e-3)
    parser.add_argument("--max_train_points", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    arr = _load_array(args.data_file)
    fidx = int(args.feature_index)
    if fidx < 0 or fidx >= arr.shape[-1]:
        raise ValueError(f"feature_index out of range: {fidx}, F={arr.shape[-1]}")

    seq = arr[..., fidx]  # [T, N]
    horizons = [int(x) for x in str(args.horizons).split(",") if str(x).strip()]
    max_h = max(horizons)

    (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = split_time_series(
        total_steps=seq.shape[0],
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
    )
    tr_seq = seq[tr_s:tr_e]
    va_seq = seq[va_s:va_e]
    te_seq = seq[te_s:te_e]

    tr_x, tr_y = _build_windows_2d(tr_seq, history_steps=int(args.history_steps), max_horizon=max_h)
    va_x, va_y = _build_windows_2d(va_seq, history_steps=int(args.history_steps), max_horizon=max_h)
    te_x, te_y = _build_windows_2d(te_seq, history_steps=int(args.history_steps), max_horizon=max_h)

    print("data:", args.data_file)
    print(f"shape={arr.shape}, feature_index={fidx}")
    print(
        "split windows:",
        f"train={tr_x.shape[0]}",
        f"val={va_x.shape[0]}",
        f"test={te_x.shape[0]}",
        f"history={args.history_steps}",
        f"max_h={max_h}",
    )

    rows_val: List[Tuple[str, Dict[int, Dict[str, float]]]] = []
    rows_test: List[Tuple[str, Dict[int, Dict[str, float]]]] = []

    # Persistence
    m_val: Dict[int, Dict[str, float]] = {}
    m_test: Dict[int, Dict[str, float]] = {}
    for h in horizons:
        target_idx = h - 1
        p_val = va_x[:, :, -1]
        p_test = te_x[:, :, -1]
        m_val[h] = _mae_rmse(p_val, va_y[:, :, target_idx])
        m_test[h] = _mae_rmse(p_test, te_y[:, :, target_idx])
    rows_val.append(("persistence", m_val))
    rows_test.append(("persistence", m_test))

    # Moving average
    k = max(1, min(int(args.ma_window), int(args.history_steps)))
    m_val = {}
    m_test = {}
    for h in horizons:
        target_idx = h - 1
        p_val = va_x[:, :, -k:].mean(axis=2)
        p_test = te_x[:, :, -k:].mean(axis=2)
        m_val[h] = _mae_rmse(p_val, va_y[:, :, target_idx])
        m_test[h] = _mae_rmse(p_test, te_y[:, :, target_idx])
    rows_val.append((f"moving_avg{k}", m_val))
    rows_test.append((f"moving_avg{k}", m_test))

    # Seasonal naive
    lag = int(args.seasonal_lag)
    lag = max(1, min(lag, int(args.history_steps)))
    m_val = {}
    m_test = {}
    for h in horizons:
        target_idx = h - 1
        p_val = va_x[:, :, -lag]
        p_test = te_x[:, :, -lag]
        m_val[h] = _mae_rmse(p_val, va_y[:, :, target_idx])
        m_test[h] = _mae_rmse(p_test, te_y[:, :, target_idx])
    rows_val.append((f"seasonal_lag{lag}", m_val))
    rows_test.append((f"seasonal_lag{lag}", m_test))

    # Ridge linear AR (fit per horizon)
    m_val = {}
    m_test = {}
    for h in horizons:
        target_idx = h - 1
        w = _fit_ridge(
            x_hist=tr_x,
            y_h=tr_y[:, :, target_idx],
            ridge_lambda=float(args.ridge_lambda),
            max_train_points=int(args.max_train_points),
            seed=int(args.seed) + int(h),
        )
        p_val = _predict_ridge(va_x, w=w)
        p_test = _predict_ridge(te_x, w=w)
        m_val[h] = _mae_rmse(p_val, va_y[:, :, target_idx])
        m_test[h] = _mae_rmse(p_test, te_y[:, :, target_idx])
    rows_val.append(("ridge_ar", m_val))
    rows_test.append(("ridge_ar", m_test))

    _print_block("Validation", rows_val, horizons=horizons)
    _print_block("Test", rows_test, horizons=horizons)


if __name__ == "__main__":
    main()
