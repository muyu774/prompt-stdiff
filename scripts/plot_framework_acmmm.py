#!/usr/bin/env python3
"""Plot ACM MM style framework diagram for Prompt-LLM / Prompt-STDiff."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _set_style() -> None:
    """Global figure style (ACM-like clean look)."""
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    mpl.rcParams["mathtext.fontset"] = "dejavuserif"
    mpl.rcParams["axes.linewidth"] = 0.8
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["savefig.pad_inches"] = 0.05


def _rounded(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    fc: str,
    ec: str = "#4A4A4A",
    lw: float = 1.0,
    r: float = 0.02,
    z: int = 1,
) -> FancyBboxPatch:
    """Draw rounded rectangle."""
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={r}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=z,
    )
    ax.add_patch(box)
    return box


def _arrow(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str = "#505050",
    lw: float = 1.4,
    ms: int = 10,
    ls: str = "-",
    z: int = 3,
) -> None:
    """Draw directed arrow."""
    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        linestyle=ls,
        color=color,
        zorder=z,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arr)


def plot_framework(out_pdf: Path, out_png: Path) -> None:
    """Render framework figure."""
    _set_style()

    # ACM-style restrained palette
    C_BG = "#FBFBFD"
    C_TITLE = "#1F2D3D"
    C_TEXT = "#1E1E1E"
    C_BORDER = "#5A6472"
    C_LEFT = "#EAF1FB"
    C_MID = "#EAF7EF"
    C_GRAPH = "#EEE8F8"
    C_PROMPT = "#FFF2E3"
    C_ACCENT_B = "#2E5B9A"
    C_ACCENT_G = "#2F855A"
    C_ACCENT_O = "#C97A1A"
    C_ACCENT_P = "#6B46C1"
    C_OUT = "#E7F9EE"

    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Top workflow track
    ax.text(0.15, 0.94, "Multi-modal Input & Encoding", ha="center", va="center", fontsize=16, color=C_TITLE, fontweight="bold")
    ax.text(0.60, 0.94, "Conditional Denoising Diffusion", ha="center", va="center", fontsize=16, color=C_TITLE, fontweight="bold")
    ax.text(0.92, 0.94, "Output", ha="center", va="center", fontsize=16, color=C_TITLE, fontweight="bold")
    _arrow(ax, 0.27, 0.94, 0.46, 0.94, color=C_BORDER, lw=1.0, ms=9)
    _arrow(ax, 0.74, 0.94, 0.88, 0.94, color=C_BORDER, lw=1.0, ms=9)

    # Left-top: temporal encoding
    _rounded(ax, 0.04, 0.66, 0.38, 0.24, fc=C_LEFT, ec=C_BORDER, lw=1.2, r=0.03)
    ax.text(0.23, 0.87, "Temporal Traffic Encoding", ha="center", va="center", fontsize=13, color=C_TEXT, fontweight="bold")

    _rounded(ax, 0.07, 0.72, 0.10, 0.10, fc="#DCEAFB", ec=C_BORDER, lw=0.9)
    ax.text(0.12, 0.77, "Historical\nSequence\n$X_{his}$", ha="center", va="center", fontsize=11, color=C_TEXT)

    _rounded(ax, 0.20, 0.72, 0.09, 0.10, fc="#DCEAFB", ec=C_BORDER, lw=0.9)
    ax.text(0.245, 0.77, "Temporal\nEncoder", ha="center", va="center", fontsize=11, color=C_TEXT)

    _rounded(ax, 0.315, 0.72, 0.08, 0.10, fc="#DCEAFB", ec=C_BORDER, lw=0.9)
    ax.text(0.355, 0.77, "Transformer\n+ 1D Conv", ha="center", va="center", fontsize=10.5, color=C_TEXT)

    _arrow(ax, 0.17, 0.77, 0.20, 0.77, color=C_ACCENT_B, lw=1.4)
    _arrow(ax, 0.29, 0.77, 0.315, 0.77, color=C_ACCENT_B, lw=1.4)
    _arrow(ax, 0.395, 0.77, 0.46, 0.77, color=C_ACCENT_B, lw=1.6)
    ax.text(0.425, 0.792, "$h_{time}$", fontsize=10.5, color=C_ACCENT_B)

    # Left-middle: graph construction
    _rounded(ax, 0.04, 0.42, 0.38, 0.20, fc=C_GRAPH, ec=C_BORDER, lw=1.2, r=0.03)
    ax.text(0.23, 0.59, "Dual Graph Construction", ha="center", va="center", fontsize=13, color=C_TEXT, fontweight="bold")

    _rounded(ax, 0.07, 0.47, 0.09, 0.08, fc="#E4D8FA", ec=C_BORDER, lw=0.9)
    ax.text(0.115, 0.51, "Distance\nMatrix", ha="center", va="center", fontsize=10.5, color=C_TEXT)
    _rounded(ax, 0.19, 0.47, 0.08, 0.08, fc="#E4D8FA", ec=C_BORDER, lw=0.9)
    ax.text(0.23, 0.51, "$A_{phy}$", ha="center", va="center", fontsize=13, color=C_TEXT)

    _rounded(ax, 0.07, 0.435, 0.09, 0.08, fc="#F9E9D5", ec=C_BORDER, lw=0.9)
    ax.text(0.115, 0.475, "Cosine\nSimilarity", ha="center", va="center", fontsize=10.5, color=C_TEXT)
    _rounded(ax, 0.19, 0.435, 0.08, 0.08, fc="#F9E9D5", ec=C_BORDER, lw=0.9)
    ax.text(0.23, 0.475, "$A_{sem}$", ha="center", va="center", fontsize=13, color=C_TEXT)

    _rounded(ax, 0.30, 0.44, 0.10, 0.10, fc="#EFEAFE", ec=C_BORDER, lw=0.9)
    ax.text(0.35, 0.49, "$G_{dual}$\n$(A_{phy},A_{sem})$", ha="center", va="center", fontsize=10.5, color=C_TEXT)
    _arrow(ax, 0.16, 0.51, 0.19, 0.51, color=C_ACCENT_P, lw=1.2)
    _arrow(ax, 0.16, 0.475, 0.19, 0.475, color=C_ACCENT_P, lw=1.2)
    _arrow(ax, 0.27, 0.51, 0.30, 0.51, color=C_ACCENT_P, lw=1.2)
    _arrow(ax, 0.27, 0.475, 0.30, 0.475, color=C_ACCENT_P, lw=1.2)
    _arrow(ax, 0.40, 0.49, 0.46, 0.49, color=C_ACCENT_P, lw=1.6)
    ax.text(0.42, 0.51, "$G_{dual}$", fontsize=10.5, color=C_ACCENT_P)

    # Left-bottom: prompt + frozen LLM
    _rounded(ax, 0.04, 0.12, 0.38, 0.24, fc=C_PROMPT, ec=C_BORDER, lw=1.2, r=0.03)
    ax.text(0.23, 0.33, "Prompt-based Semantic Extraction", ha="center", va="center", fontsize=13, color=C_TEXT, fontweight="bold")
    _rounded(ax, 0.07, 0.18, 0.12, 0.12, fc="#FCE7CE", ec=C_BORDER, lw=0.9)
    ax.text(0.13, 0.24, "POI / Weather /\nEvent Context", ha="center", va="center", fontsize=10.5, color=C_TEXT)
    _rounded(ax, 0.22, 0.18, 0.16, 0.12, fc="#FCE7CE", ec=C_BORDER, lw=0.9)
    ax.text(0.30, 0.24, "Frozen LLM\n(RoBERTa / Llama)", ha="center", va="center", fontsize=10.5, color=C_TEXT)
    _arrow(ax, 0.19, 0.24, 0.22, 0.24, color=C_ACCENT_O, lw=1.4)
    _arrow(ax, 0.38, 0.24, 0.46, 0.24, color=C_ACCENT_O, lw=1.6)
    ax.text(0.405, 0.262, "$Z_{sem}$", fontsize=10.5, color=C_ACCENT_O)

    # Middle main diffusion module
    _rounded(ax, 0.47, 0.10, 0.40, 0.80, fc=C_MID, ec=C_BORDER, lw=1.4, r=0.03)

    # Forward process strip
    _rounded(ax, 0.50, 0.79, 0.34, 0.08, fc="#DBEBDD", ec="#90A693", lw=0.8, r=0.015)
    ax.text(0.67, 0.85, "Forward process (training only)", ha="center", va="center", fontsize=11.5, color=C_TEXT, fontweight="bold")
    ax.text(0.53, 0.82, "$x_0$", fontsize=12, color=C_TEXT)
    ax.text(0.60, 0.82, "$x_1$", fontsize=12, color=C_TEXT)
    ax.text(0.67, 0.82, "$x_2$", fontsize=12, color=C_TEXT)
    ax.text(0.76, 0.82, "$\\cdots$", fontsize=12, color=C_TEXT)
    ax.text(0.82, 0.82, "$x_K$", fontsize=12, color=C_TEXT)
    _arrow(ax, 0.55, 0.82, 0.59, 0.82, color="#7A807C", lw=1.0, ms=8)
    _arrow(ax, 0.62, 0.82, 0.66, 0.82, color="#7A807C", lw=1.0, ms=8)
    _arrow(ax, 0.70, 0.82, 0.81, 0.82, color="#7A807C", lw=1.0, ms=8)

    # Reverse denoising core
    _rounded(ax, 0.50, 0.18, 0.34, 0.57, fc="#DDEDE0", ec="#90A693", lw=0.9, r=0.02)
    ax.text(0.67, 0.73, "Reverse Denoising Core", ha="center", va="center", fontsize=12.5, color=C_TEXT, fontweight="bold")

    # Step-aware router
    _rounded(ax, 0.52, 0.40, 0.12, 0.29, fc="#EEF4FF", ec="#8EA3C2", lw=0.9, r=0.015)
    ax.text(0.58, 0.67, "Step-aware\nRouter", ha="center", va="center", fontsize=11, color=C_TEXT, fontweight="bold")
    ax.text(0.535, 0.61, "1) $h_{time}$", fontsize=10.5, color=C_TEXT)
    ax.text(0.535, 0.56, "2) $Z_{sem}$", fontsize=10.5, color=C_TEXT)
    ax.text(0.535, 0.51, "3) $G_{dual}$", fontsize=10.5, color=C_TEXT)
    ax.text(0.535, 0.46, "4) step $k$", fontsize=10.5, color=C_TEXT)
    ax.text(0.58, 0.42, "fusion gate $\\alpha_k$", ha="center", va="center", fontsize=10, color=C_ACCENT_B)

    # Epsilon network
    _rounded(ax, 0.66, 0.34, 0.13, 0.35, fc="#EFFAF2", ec="#86A98E", lw=0.9, r=0.015)
    ax.text(0.725, 0.66, "Denoise Net\n$\\epsilon_\\theta$", ha="center", va="center", fontsize=11, color=C_TEXT, fontweight="bold")
    ax.text(0.725, 0.59, "Graph Attention", ha="center", va="center", fontsize=9.8, color=C_TEXT)
    ax.text(0.725, 0.54, "Temporal Self-Attn", ha="center", va="center", fontsize=9.8, color=C_TEXT)
    ax.text(0.725, 0.49, "Cross-modal Attn", ha="center", va="center", fontsize=9.8, color=C_TEXT)
    ax.text(0.725, 0.44, "Feed-forward", ha="center", va="center", fontsize=9.8, color=C_TEXT)
    _arrow(ax, 0.64, 0.54, 0.66, 0.54, color=C_ACCENT_G, lw=1.4)
    _arrow(ax, 0.79, 0.515, 0.83, 0.515, color=C_ACCENT_G, lw=1.4)
    ax.text(0.805, 0.535, "$\\hat{\\epsilon}$", fontsize=10, color=C_ACCENT_G)

    # Prior at bottom
    _rounded(ax, 0.56, 0.22, 0.07, 0.04, fc="#FBE9D6", ec="#BC8A4A", lw=0.8, r=0.01)
    _rounded(ax, 0.66, 0.22, 0.07, 0.04, fc="#FBE9D6", ec="#BC8A4A", lw=0.8, r=0.01)
    ax.text(0.595, 0.24, "$\\mu$-Net", ha="center", va="center", fontsize=10, color=C_TEXT)
    ax.text(0.695, 0.24, "$\\sigma$-Net", ha="center", va="center", fontsize=10, color=C_TEXT)
    ax.text(0.67, 0.20, "Semantic-guided prior: $x_K \\sim \\mathcal{N}(\\mu_\\theta(Z_{sem}),\\sigma_\\theta^2(Z_{sem}),\\mathcal{I})$",
            ha="center", va="center", fontsize=9.6, color=C_TEXT)

    # Flow from left modules into middle
    _arrow(ax, 0.42, 0.77, 0.50, 0.62, color=C_ACCENT_B, lw=1.6)
    _arrow(ax, 0.42, 0.49, 0.50, 0.53, color=C_ACCENT_P, lw=1.6)
    _arrow(ax, 0.42, 0.24, 0.56, 0.24, color=C_ACCENT_O, lw=1.6)

    # Right output panel
    _rounded(ax, 0.89, 0.50, 0.09, 0.11, fc="#F3F5F8", ec=C_BORDER, lw=0.9, r=0.015)
    ax.text(0.935, 0.565, "Training\n$\\mathcal{L}_{diff}=\\|\\epsilon-\\hat{\\epsilon}\\|_2^2$", ha="center", va="center", fontsize=10.5, color=C_TEXT)

    _rounded(ax, 0.89, 0.24, 0.09, 0.18, fc=C_OUT, ec=C_ACCENT_G, lw=1.0, r=0.018)
    ax.text(0.935, 0.37, "Inference", ha="center", va="center", fontsize=11.5, color=C_TEXT, fontweight="bold")
    ax.text(0.935, 0.32, "Predicted Flow\n$\\hat{x}_0$", ha="center", va="center", fontsize=10.8, color=C_TEXT)
    ax.plot([0.90, 0.97], [0.26, 0.26], color="#7EA985", lw=1.0)
    ax.vlines([0.905, 0.918, 0.931, 0.944, 0.957], 0.26, [0.30, 0.28, 0.33, 0.31, 0.35], color="#7EA985", lw=2.0)

    _arrow(ax, 0.87, 0.54, 0.89, 0.54, color="#7D7D7D", lw=1.5, ls="--")
    _arrow(ax, 0.87, 0.33, 0.89, 0.33, color=C_ACCENT_G, lw=1.7)

    # Save
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf")
    fig.savefig(out_png, format="png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """CLI args."""
    parser = argparse.ArgumentParser(description="Plot ACM MM-style framework figure.")
    parser.add_argument(
        "--out_pdf",
        type=str,
        default="outputs/figures/framework_acmmm.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--out_png",
        type=str,
        default="outputs/figures/framework_acmmm.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    out_pdf = Path(args.out_pdf)
    out_png = Path(args.out_png)
    plot_framework(out_pdf=out_pdf, out_png=out_png)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()

