"""Plot Prompt-STDiff hyperparameter sensitivity curves."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Dict, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/prompt_stdiff_mpl")

import matplotlib.pyplot as plt
import pandas as pd


FACTORS = ["gamma", "topk", "pdrop", "diffusion_steps"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot hyperparameter sensitivity")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/sensitivity/pems08_h12"))
    parser.add_argument("--horizon", type=int, default=12)
    return parser.parse_args()


def _parse_factor_value(setting: str) -> Tuple[str, float]:
    """Parse setting strings like sensitivity_gamma_0.3."""
    prefix = "sensitivity_"
    if not str(setting).startswith(prefix):
        return "", float("nan")
    rest = str(setting)[len(prefix) :]
    for factor in FACTORS:
        marker = factor + "_"
        if rest.startswith(marker):
            value_raw = rest[len(marker) :]
            return factor, float(value_raw)
    return "", float("nan")


def _latency_from_settings(settings_json: str) -> float:
    """Extract latency from settings_json."""
    try:
        settings = json.loads(settings_json)
        return float(settings.get("latency_ms_per_sample", float("nan")))
    except Exception:
        return float("nan")


def _plot_factor(df: pd.DataFrame, factor: str, out_dir: Path) -> None:
    """Plot MAE and CRPS for one factor."""
    sub = df[df["factor"] == factor].sort_values("value")
    if sub.empty:
        return
    fig, ax1 = plt.subplots(figsize=(5.6, 3.4), dpi=180)
    ax2 = ax1.twinx()
    ax1.plot(sub["value"], sub["mae"], marker="o", color="#b23a48", label="MAE")
    ax2.plot(sub["value"], sub["crps"], marker="s", linestyle="--", color="#245c7a", label="CRPS")
    ax1.set_xlabel(factor)
    ax1.set_ylabel("MAE", color="#b23a48")
    ax2.set_ylabel("CRPS", color="#245c7a")
    ax1.grid(axis="y", alpha=0.25)
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"sensitivity_{factor}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"sensitivity_{factor}.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_summary(df: pd.DataFrame, out_dir: Path) -> None:
    """Write markdown/CSV summary tables."""
    cols = ["factor", "value", "mae", "rmse", "crps", "latency_ms_per_sample", "setting", "config"]
    summary = df[cols].sort_values(["factor", "value"])
    summary.to_csv(out_dir / "sensitivity_summary.csv", index=False)
    lines = [
        "| Factor | Value | MAE | RMSE | CRPS | Latency ms/sample | Setting |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['factor']} | {row['value']:g} | {row['mae']:.6f} | "
            f"{row['rmse']:.6f} | {row['crps']:.6f} | {row['latency_ms_per_sample']:.6f} | "
            f"{row['setting']} |"
        )

    robust_lines = [
        "",
        "### Notes",
        "",
        "- Robust ranges should be selected where both MAE and CRPS remain close to the best value while latency is acceptable.",
        "- `gamma=0` is the no semantic dynamic prior setting and should match the `w/o Dynamic Prior` ablation when evaluated with the same checkpoint/protocol.",
        "- `p_drop` is a training-time semantic dropout hyperparameter; pure checkpoint evaluation with `model.eval()` will not activate dropout, so a full training sweep is required for final p_drop sensitivity.",
        "- Top-K sensitivity uses distinct semantic graph cache files per K to prevent silent cache reuse.",
    ]
    (out_dir / "sensitivity_summary.md").write_text("\n".join(lines + robust_lines) + "\n", encoding="utf-8")


def main() -> None:
    """Render all sensitivity plots."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    if "horizon" in df.columns:
        df = df[df["horizon"].astype(int) == int(args.horizon)].copy()
    parsed = df["setting"].apply(_parse_factor_value)
    df["factor"] = parsed.apply(lambda x: x[0])
    df["value"] = parsed.apply(lambda x: x[1])
    df["latency_ms_per_sample"] = df["settings_json"].apply(_latency_from_settings)
    df = df[df["factor"].isin(FACTORS)].copy()
    if df.empty:
        raise RuntimeError(f"No sensitivity rows found in {args.csv}")

    plt.rcParams["font.family"] = "Times New Roman"
    for factor in FACTORS:
        _plot_factor(df, factor=factor, out_dir=args.out_dir)
    _write_summary(df, out_dir=args.out_dir)
    print(f"Saved sensitivity plots and summary to {args.out_dir}")


if __name__ == "__main__":
    main()
