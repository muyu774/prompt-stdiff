#!/usr/bin/env python3
"""P0 integrity gate for results.csv (aligned to T-ITS checklist).

Checks:
1. CRPS unit sanity: flag rows whose CRPS is on a different scale.
2. Mean-preserving consistency: 'Ours' MAE close to frozen-mean MAE.
3. Test-leakage hint: flag sweep/scale settings in main-table-eligible rows.
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


def check_crps_units(df: pd.DataFrame) -> pd.Series:
    ok = pd.Series(True, index=df.index)
    if "crps" not in df.columns:
        return ok
    for ds, grp in df.groupby("dataset"):
        med = grp["crps"].dropna().median()
        if not np.isfinite(med) or med == 0:
            continue
        ratio = grp["crps"] / med
        bad = (ratio < 0.1) | (ratio > 10.0)
        ok.loc[grp.index[bad.fillna(False)]] = False
    return ok


def check_mean_preserving(df: pd.DataFrame, tol_rel: float = 0.02) -> pd.Series:
    ok = pd.Series(True, index=df.index)
    if not {"method", "mae", "dataset", "horizon"}.issubset(df.columns):
        return ok
    frozen = df[df["method"].str.contains("frozen", case=False, na=False)]
    ref = frozen.groupby(["dataset", "horizon"])["mae"].min()
    for idx, row in df.iterrows():
        if not str(row.get("method", "")).lower().startswith(("prompt", "ours", "residual")):
            continue
        key = (row["dataset"], row["horizon"])
        if key in ref.index and np.isfinite(ref[key]) and ref[key] > 0:
            if abs(row["mae"] - ref[key]) / ref[key] > tol_rel:
                ok.loc[idx] = False
    return ok


def check_leakage(df: pd.DataFrame) -> pd.Series:
    ok = pd.Series(True, index=df.index)
    if "setting" not in df.columns:
        return ok
    bad = df["setting"].astype(str).str.contains("sweep|scale|test_sweep", case=False, na=False)
    ok.loc[bad] = False
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=ROOT / "outputs" / "results.csv")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.csv.exists():
        print(f"[WARN] {args.csv} not found."); return
    df = load(args.csv)
    c1 = check_crps_units(df)
    c2 = check_mean_preserving(df)
    c3 = check_leakage(df)
    df["crps_unit_ok"] = c1
    df["mean_preserving_ok"] = c2
    df["no_leakage_ok"] = c3
    df["integrity_ok"] = c1 & c2 & c3
    df.to_csv(OUT / "results_clean.csv", index=False)

    n = len(df)
    lines = ["# Integrity Report", "",
             f"Source `{args.csv}` rows={n}", "",
             f"- CRPS unit OK: {int(c1.sum())}/{n}",
             f"- Mean-preserving OK: {int(c2.sum())}/{n}",
             f"- No-leakage OK: {int(c3.sum())}/{n}",
             f"- **Eligible for main table: {int(df['integrity_ok'].sum())}/{n}**", ""]
    bad = df[~df["integrity_ok"]]
    if not bad.empty:
        cols = [c for c in ["dataset", "method", "setting", "horizon", "mae", "crps",
                            "crps_unit_ok", "mean_preserving_ok", "no_leakage_ok"] if c in bad.columns]
        lines += ["## Flagged rows (excluded from main claims)", "", bad[cols].to_markdown(index=False)]
    (OUT / "integrity_report.md").write_text("\n".join(lines))
    print(f"[OK] {OUT/'integrity_report.md'}")
    print(f"[OK] {OUT/'results_clean.csv'}")
    print("\n".join(lines[:8]))


if __name__ == "__main__":
    main()
