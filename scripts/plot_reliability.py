"""Plot probabilistic reliability diagrams from result_writer CSV files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/prompt_stdiff_mpl")

import matplotlib.pyplot as plt
import pandas as pd


LEVELS = [i / 10.0 for i in range(1, 10)]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot reliability diagram")
    parser.add_argument("--csv", type=Path, required=True, help="CSV produced by utils.result_writer")
    parser.add_argument("--out", type=Path, default=Path("outputs/reliability/reliability.png"))
    parser.add_argument("--out_pdf", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--methods", type=str, default="", help="Optional comma-separated method filter")
    parser.add_argument("--title", type=str, default="Reliability Diagram")
    return parser.parse_args()


def _extract_curve(settings_json: str) -> List[float]:
    """Extract reliability@10..90 values from one settings JSON string."""
    settings = json.loads(settings_json) if isinstance(settings_json, str) and settings_json else {}
    curve = []
    for pct in range(10, 100, 10):
        key = f"reliability@{pct}"
        if key not in settings:
            raise KeyError(f"Missing {key} in settings_json")
        curve.append(float(settings[key]))
    return curve


def _collect_curves(df: pd.DataFrame) -> Dict[Tuple[str, str], List[float]]:
    """Collect mean reliability curve by (method, setting)."""
    grouped: Dict[Tuple[str, str], List[List[float]]] = {}
    for _, row in df.iterrows():
        try:
            curve = _extract_curve(str(row.get("settings_json", "")))
        except Exception:
            continue
        key = (str(row.get("method", "")), str(row.get("setting", "")))
        grouped.setdefault(key, []).append(curve)

    out: Dict[Tuple[str, str], List[float]] = {}
    for key, curves in grouped.items():
        if not curves:
            continue
        mat = pd.DataFrame(curves).astype(float)
        out[key] = [float(x) for x in mat.mean(axis=0).tolist()]
    return out


def main() -> None:
    """Render reliability diagram."""
    args = parse_args()
    df = pd.read_csv(args.csv)
    if args.dataset is not None:
        df = df[df["dataset"].astype(str) == str(args.dataset)]
    if "horizon" in df.columns:
        df = df[df["horizon"].astype(int) == int(args.horizon)]
    if args.methods:
        keep = {x.strip() for x in args.methods.split(",") if x.strip()}
        df = df[df["method"].astype(str).isin(keep)]

    curves = _collect_curves(df)
    if not curves:
        raise RuntimeError(
            "No reliability curves found. Ensure the CSV rows contain settings_json "
            "with reliability@10..reliability@90 keys."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out_pdf is None:
        args.out_pdf = args.out.with_suffix(".pdf")
    else:
        args.out_pdf.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=180)
    ax.plot(LEVELS, LEVELS, color="black", linestyle=":", linewidth=1.6, label="Ideal")
    for (method, setting), empirical in curves.items():
        label = method if not setting else f"{method} ({setting})"
        ax.plot(LEVELS, empirical, marker="o", linewidth=1.8, label=label)

    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.set_title(args.title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out_pdf, bbox_inches="tight")
    print(f"Saved reliability diagram to {args.out} and {args.out_pdf}")


if __name__ == "__main__":
    main()
