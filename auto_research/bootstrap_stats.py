#!/usr/bin/env python3
"""Multi-seed paired bootstrap CI over results.csv.

For each (dataset, horizon, setting) the metric is aggregated across seeds.
Paired bootstrap compares a target method vs reference methods and reports
mean difference with a 95% CI, to judge whether 0.01-0.03 CRPS gaps are real.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "auto_research"


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for c in ("mae", "rmse", "crps", "horizon", "seed"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int):
    """Return mean diff (a-b) and 95% CI via paired bootstrap over seeds."""
    rng = np.random.default_rng(seed)
    m = min(len(a), len(b))
    a, b = a[:m], b[:m]
    if m == 0:
        return float("nan"), (float("nan"), float("nan"))
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, m, size=m)
        diffs.append(float(np.mean(a[idx] - b[idx])))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(a - b)), (float(lo), float(hi))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=ROOT / "outputs" / "results.csv")
    p.add_argument("--metric", default="crps")
    p.add_argument("--target", default="Prompt-STDiff")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.csv.exists():
        print(f"[WARN] {args.csv} not found."); return
    df = load(args.csv)
    if args.metric not in df.columns:
        print(f"[WARN] metric {args.metric} not in columns."); return

    rows = []
    for (ds, h), grp in df.groupby(["dataset", "horizon"]):
        tgt = grp[grp["method"].str.contains(args.target, case=False, na=False)]
        if tgt.empty:
            continue
        tgt_vals = tgt.groupby("seed")[args.metric].mean().to_numpy()
        for ref_method, ref_grp in grp.groupby("method"):
            if str(ref_method).lower() == args.target.lower():
                continue
            ref_vals = ref_grp.groupby("seed")[args.metric].mean().to_numpy()
            mean_diff, (lo, hi) = paired_bootstrap(tgt_vals, ref_vals, args.n_boot, args.seed)
            sig = (lo > 0) or (hi < 0)
            rows.append({
                "dataset": ds, "horizon": int(h), "metric": args.metric,
                "target": args.target, "reference": ref_method,
                "target_mean": float(np.mean(tgt_vals)) if tgt_vals.size else float("nan"),
                "ref_mean": float(np.mean(ref_vals)) if ref_vals.size else float("nan"),
                "mean_diff(target-ref)": mean_diff, "ci95_low": lo, "ci95_high": hi,
                "significant": sig, "n_seeds_target": int(tgt_vals.size),
                "n_seeds_ref": int(ref_vals.size),
            })
    out = pd.DataFrame(rows).sort_values(["dataset", "horizon", "reference"])
    out.to_csv(OUT / "bootstrap_stats.csv", index=False)
    (OUT / "bootstrap_stats.md").write_text(
        "# Paired Bootstrap (multi-seed)\n\n"
        f"metric=`{args.metric}` target=`{args.target}` n_boot={args.n_boot}\n\n"
        + (out.to_markdown(index=False) if not out.empty else "_No comparable rows._"))
    print(f"[OK] {OUT/'bootstrap_stats.md'}")
    if not out.empty:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
