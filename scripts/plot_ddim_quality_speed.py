"""Plot DDIM quality/latency curves from result_writer CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Plot DDIM quality-speed sweep")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out_png", type=Path, default=Path("outputs/ddim_sweep/ddim_quality_speed.png"))
    parser.add_argument("--out_pdf", type=Path, default=Path("outputs/ddim_sweep/ddim_quality_speed.pdf"))
    parser.add_argument("--horizon", type=int, default=3)
    return parser.parse_args()


def _sampling_steps(row) -> int:
    """Parse sampling steps from settings_json or setting."""
    try:
        settings = json.loads(row.get("settings_json", "{}"))
        if "sampling_steps" in settings:
            return int(settings["sampling_steps"])
    except Exception:
        pass
    m = re.search(r"s(\d+)", str(row.get("setting", "")))
    if not m:
        return -1
    return int(m.group(1))


def _latency(row) -> float:
    """Parse latency from settings_json or notes."""
    try:
        settings = json.loads(row.get("settings_json", "{}"))
        if "latency_ms_per_sample" in settings:
            return float(settings["latency_ms_per_sample"])
    except Exception:
        pass
    m = re.search(r"latency_ms_per_sample=([0-9.]+)", str(row.get("notes", "")))
    if not m:
        return float("nan")
    return float(m.group(1))


def main() -> None:
    """Render plot."""
    args = parse_args()
    df = pd.read_csv(args.csv)
    df = df[df["horizon"].astype(int) == int(args.horizon)].copy()
    df["sampling_steps"] = df.apply(_sampling_steps, axis=1)
    df["latency_ms_per_sample"] = df.apply(_latency, axis=1)
    df = df.sort_values("sampling_steps", ascending=False)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax1 = plt.subplots(figsize=(6.2, 3.6), dpi=180)
    ax2 = ax1.twinx()

    ax1.plot(df["sampling_steps"], df["crps"].astype(float), marker="o", color="#b23a48", label="CRPS")
    ax2.plot(
        df["sampling_steps"],
        df["latency_ms_per_sample"].astype(float),
        marker="s",
        color="#245c7a",
        linestyle="--",
        label="Latency",
    )
    ax1.set_xlabel("DDIM sampling steps")
    ax1.set_ylabel("CRPS", color="#b23a48")
    ax2.set_ylabel("Latency (ms/sample)", color="#245c7a")
    ax1.grid(axis="y", alpha=0.25)
    ax1.invert_xaxis()
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(args.out_png, bbox_inches="tight")
    fig.savefig(args.out_pdf, bbox_inches="tight")
    print(f"Saved {args.out_png} and {args.out_pdf}")


if __name__ == "__main__":
    main()
