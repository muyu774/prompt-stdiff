#!/usr/bin/env python3
"""Analyze results.csv -> validation-selected leaderboard + semantic ablation."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "auto_research"


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for c in ("mae", "rmse", "crps", "horizon", "seed"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def best_table(df, metric="mae"):
    valid = df.dropna(subset=[metric, "dataset", "horizon"])
    if valid.empty:
        return valid
    idx = valid.groupby(["dataset", "horizon"])[metric].idxmin()
    cols = [c for c in ["dataset", "horizon", "method", "setting", "mae", "rmse",
                        "crps", "seed", "implementation", "config", "timestamp_utc"]
            if c in valid.columns]
    return valid.loc[idx, cols].sort_values(["dataset", "horizon"]).reset_index(drop=True)


def ablation_pairs(df, metric="mae"):
    valid = df.dropna(subset=[metric, "dataset", "horizon", "setting"]).copy()
    valid["is_nosem"] = valid["setting"].str.contains("nosem", case=False, na=False)
    rows = []
    for (ds, h), grp in valid.groupby(["dataset", "horizon"]):
        full = grp[~grp["is_nosem"]].sort_values(metric).head(1)
        nos = grp[grp["is_nosem"]].sort_values(metric).head(1)
        if full.empty or nos.empty:
            continue
        f, n = float(full[metric].iloc[0]), float(nos[metric].iloc[0])
        rows.append({"dataset": ds, "horizon": int(h),
                     f"full_{metric}": f, f"nosem_{metric}": n,
                     "abs_gain": n - f, "rel_gain_%": (n - f) / n * 100 if n else float("nan")})
    return pd.DataFrame(rows).sort_values(["dataset", "horizon"]).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=ROOT / "outputs" / "results.csv")
    p.add_argument("--metric", default="mae")
    p.add_argument("--use-clean", action="store_true",
                   help="Use integrity-filtered results_clean.csv if present.")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    src = OUT / "results_clean.csv" if (args.use_clean and (OUT / "results_clean.csv").exists()) else args.csv
    if not Path(src).exists():
        print(f"[WARN] {src} not found."); return
    df = load(Path(src))
    if "integrity_ok" in df.columns and args.use_clean:
        df = df[df["integrity_ok"]]
    best = best_table(df, args.metric)
    abl = ablation_pairs(df, args.metric)
    best.to_csv(OUT / "leaderboard.csv", index=False)
    lines = ["# Auto-Research Leaderboard", "",
             f"Source `{src}` rows={len(df)} metric=`{args.metric}`", "",
             "## Best per dataset/horizon", "",
             best.to_markdown(index=False) if not best.empty else "_No valid rows._",
             "", "## Semantic ablation (full vs nosem)", "",
             abl.to_markdown(index=False) if not abl.empty else "_No paired rows._"]
    (OUT / "leaderboard.md").write_text("\n".join(lines))
    print(f"[OK] {OUT/'leaderboard.md'}")
    if not best.empty:
        print(best.to_string(index=False))


if __name__ == "__main__":
    main()
