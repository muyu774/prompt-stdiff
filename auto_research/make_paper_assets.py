#!/usr/bin/env python3
"""Generate paper assets: IEEEtran LaTeX main table + incident summary + plots."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "outputs" / "paper_assets"
INCIDENT = ROOT / "outputs" / "revision_8gpu_logs" / "incident"

PLOT_ALLOWLIST = [
    "scripts/plot_reliability.py",
    "scripts/plot_ddim_quality_speed.py",
    "scripts/plot_hyperparam_sensitivity.py",
    "scripts/plot_fig3_router_weights.py",
]


def _num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main_table_latex(csv_path: Path):
    if not csv_path.exists():
        print(f"[WARN] {csv_path} missing"); return
    df = _num(pd.read_csv(csv_path), ["mae", "rmse", "crps", "horizon"])
    df = df.dropna(subset=["dataset", "horizon", "method"])
    keep = df.sort_values("mae").drop_duplicates(
        subset=["dataset", "method", "setting", "horizon"], keep="first")
    ASSET.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Validation-selected results (physical units). "
             r"Lower MAE/RMSE/CRPS is better.}",
             r"\label{tab:auto_main}",
             r"\begin{tabular}{llccc}", r"\toprule",
             r"Dataset & Method & H & MAE & CRPS \\", r"\midrule"]
    for _, r in keep.sort_values(["dataset", "method", "horizon"]).iterrows():
        crps = "--" if pd.isna(r.get("crps")) else f"{r['crps']:.3f}"
        lines.append(f"{r['dataset']} & {r['method']} & {int(r['horizon'])} & "
                     f"{r['mae']:.3f} & {crps} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (ASSET / "main_table.tex").write_text("\n".join(lines))
    keep.to_csv(ASSET / "main_results_table.csv", index=False)
    print(f"[OK] {ASSET/'main_table.tex'}")


def incident_summary():
    if not INCIDENT.exists():
        print(f"[skip] {INCIDENT} missing"); return
    rows = []
    for jf in sorted(INCIDENT.glob("*.json")):
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue
        row = {"file": jf.name, "setting": d.get("setting"),
               "event_points": d.get("event_points"), "coverage": d.get("coverage")}
        rc = d.get("rho_mean_abs_error_over_half_width", {})
        if isinstance(rc, dict):
            row["rho_median"] = rc.get("q50")
            row["rho_q90"] = rc.get("q90")
        m = d.get("metrics", {})
        if isinstance(m, dict):
            for k in ("crps", "picp@90", "mpiw@90"):
                if k in m:
                    row[k] = m[k]
        rows.append(row)
    if rows:
        out = pd.DataFrame(rows)
        ASSET.mkdir(parents=True, exist_ok=True)
        out.to_csv(ASSET / "incident_summary.csv", index=False)
        (ASSET / "incident_summary.md").write_text(
            "# Incident/Drop Summary\n\n" + out.to_markdown(index=False))
        print(f"[OK] {ASSET/'incident_summary.md'}")


def run_plots():
    ASSET.mkdir(parents=True, exist_ok=True)
    for rel in PLOT_ALLOWLIST:
        s = ROOT / rel
        if not s.exists():
            print(f"[skip] {rel}"); continue
        log = ASSET / (Path(rel).stem + ".log")
        with log.open("w", encoding="utf-8") as f:
            rc = subprocess.run([sys.executable, str(s)], stdout=f,
                                stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
        print(f"[plot] {rel} rc={rc}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=ROOT / "outputs" / "results.csv")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    main_table_latex(args.csv)
    incident_summary()
    if not args.no_plots:
        run_plots()
    print(f"\nAssets in {ASSET}")


if __name__ == "__main__":
    main()
