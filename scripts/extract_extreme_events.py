#!/usr/bin/env python3
"""Extract extreme traffic events from PeMS tensors.

The PeMS tensor usually contains numerical traffic observations but not textual event labels.
This script mines anomaly candidates (drop/spike) at node level and exports a table:
- dataset
- node_index
- event_id
- start_time / end_time
- anomaly stats

Output CSV is compatible with scripts/fetch_social_context.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class EventCandidate:
    """One extracted event candidate."""

    dataset: str
    node_index: int
    event_id: str
    start_time: str
    end_time: str
    location: str
    keywords: str
    t_idx: int
    value: float
    baseline: float
    zscore: float
    delta: float
    kind: str


def _load_tensor(data_file: Path) -> np.ndarray:
    """Load traffic tensor [T, N, F] from npz/npy."""
    if not data_file.exists():
        raise FileNotFoundError(f"data file not found: {data_file}")
    if data_file.suffix == ".npz":
        bundle = np.load(data_file)
        for k in ("data", "x", "arr_0"):
            if k in bundle:
                x = bundle[k]
                break
        else:
            raise KeyError(f"No supported key in {data_file}. keys={bundle.files}")
    elif data_file.suffix == ".npy":
        x = np.load(data_file)
    else:
        raise ValueError(f"Unsupported data file suffix: {data_file.suffix}")
    if x.ndim == 2:
        x = x[..., None]
    if x.ndim != 3:
        raise ValueError(f"Expected [T,N,F], got {x.shape}")
    return x.astype(np.float32)


def _rolling_stats(x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rolling mean/std for [T] series with causal window."""
    s = pd.Series(x)
    mu = s.rolling(window=window, min_periods=max(5, window // 3)).mean().to_numpy()
    sd = s.rolling(window=window, min_periods=max(5, window // 3)).std().to_numpy()
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, np.nan)
    return mu, sd


def extract_candidates(
    x: np.ndarray,
    dataset: str,
    start_time: pd.Timestamp,
    freq_minutes: int,
    feature_index: int,
    window: int,
    z_drop: float,
    z_spike: float,
    top_k: int,
    kind_mode: str,
) -> List[EventCandidate]:
    """Extract top-K anomaly candidates from tensor."""
    t_len, n, f = x.shape
    if feature_index < 0 or feature_index >= f:
        raise ValueError(f"feature_index out of range: {feature_index}, f={f}")
    arr = x[:, :, feature_index]  # [T, N]
    event_rows: List[EventCandidate] = []

    # Gather all scores first, then keep global top_k.
    scored: List[Tuple[float, int, int, float, float, float, float, str]] = []
    # (abs_z, t_idx, node, value, baseline, z, delta, kind)
    for node in range(n):
        s = arr[:, node]
        mu, sd = _rolling_stats(s, window=window)

        delta = np.full_like(s, np.nan)
        delta[1:] = s[1:] - s[:-1]

        z = (s - mu) / sd
        valid = np.isfinite(z) & np.isfinite(delta)
        if not np.any(valid):
            continue

        idx = np.where(valid)[0]
        for t in idx:
            zt = float(z[t])
            dt = float(delta[t])
            kind = ""
            # Drop event: unusually low value + sharp negative change.
            if zt <= -abs(z_drop) and dt < 0:
                kind = "drop"
            # Spike event: unusually high value + positive change.
            elif zt >= abs(z_spike) and dt > 0:
                kind = "spike"
            if not kind:
                continue
            if kind_mode in ("drop", "spike") and kind != kind_mode:
                continue
            scored.append(
                (
                    abs(zt),
                    int(t),
                    int(node),
                    float(s[t]),
                    float(mu[t]) if np.isfinite(mu[t]) else float("nan"),
                    zt,
                    dt,
                    kind,
                )
            )

    scored.sort(key=lambda x: x[0], reverse=True)
    scored = scored[: max(1, top_k)]

    for rank, (_, t_idx, node, value, base, zt, dt, kind) in enumerate(scored, start=1):
        ts = start_time + pd.Timedelta(minutes=freq_minutes * t_idx)
        # ASSUMPTION: event window is centered around anomaly step with +/- 30 minutes.
        st = ts - pd.Timedelta(minutes=30)
        ed = ts + pd.Timedelta(minutes=30)
        eid = f"{dataset}_{kind}_{rank:03d}_n{node}_t{t_idx}"
        kw = "traffic accident" if kind == "drop" else "traffic surge"
        event_rows.append(
            EventCandidate(
                dataset=dataset,
                node_index=node,
                event_id=eid,
                start_time=st.tz_convert("UTC").isoformat() if st.tzinfo else st.tz_localize("UTC").isoformat(),
                end_time=ed.tz_convert("UTC").isoformat() if ed.tzinfo else ed.tz_localize("UTC").isoformat(),
                location="",  # Fill later if geo metadata is available.
                keywords=kw,
                t_idx=t_idx,
                value=value,
                baseline=base,
                zscore=zt,
                delta=dt,
                kind=kind,
            )
        )
    return event_rows


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(description="Extract extreme event candidates from PeMS data tensor.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g., pems04 / pems08")
    parser.add_argument("--data_file", type=Path, required=True, help="Path to data.npz or data.npy")
    parser.add_argument("--out_csv", type=Path, required=True, help="Output CSV for extreme event candidates")
    parser.add_argument("--start_time", type=str, required=True, help="Global start time, e.g. 2018-01-01 00:00:00+00:00")
    parser.add_argument("--freq_minutes", type=int, default=5)
    parser.add_argument("--feature_index", type=int, default=0)
    parser.add_argument("--window", type=int, default=36, help="Rolling window in steps (36=3 hours for 5-min data)")
    parser.add_argument("--z_drop", type=float, default=3.0)
    parser.add_argument("--z_spike", type=float, default=3.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--kind", type=str, default="both", choices=["both", "drop", "spike"])
    args = parser.parse_args()

    x = _load_tensor(args.data_file)
    start_time = pd.Timestamp(args.start_time)
    if start_time.tzinfo is None:
        start_time = start_time.tz_localize("UTC")
    else:
        start_time = start_time.tz_convert("UTC")

    rows = extract_candidates(
        x=x,
        dataset=args.dataset,
        start_time=start_time,
        freq_minutes=int(args.freq_minutes),
        feature_index=int(args.feature_index),
        window=int(args.window),
        z_drop=float(args.z_drop),
        z_spike=float(args.z_spike),
        top_k=int(args.top_k),
        kind_mode=str(args.kind),
    )
    out_df = pd.DataFrame([r.__dict__ for r in rows])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Saved extreme events: {args.out_csv} rows={len(out_df)}")
    if len(out_df) > 0:
        print("kind counts:")
        print(out_df["kind"].value_counts().to_string())


if __name__ == "__main__":
    main()
