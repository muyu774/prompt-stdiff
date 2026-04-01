"""Plot ACM-style Figure 3 (step-aware routing weights) from a trained checkpoint.

This script computes routing gates alpha_k from the real model on a chosen split,
then exports a publication-ready figure with Times New Roman font.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataio.traffic_dataset import build_dataloaders
from diffusion.process import DiffusionProcess
from diffusion.schedules import build_beta_schedule
from graph.graph_utils import to_torch
from graph.physical_graph import load_or_build_physical_graph
from graph.semantic_graph import load_or_build_semantic_graph
from models.prompt_stdiff import PromptSTDiff
from semantic.dynamic_context import maybe_load_dynamic_semantic_bank
from semantic.semantic_cache import load_semantic_embeddings
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.device import get_device
from utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot Figure 3 routing weights")
    parser.add_argument("--config", type=str, default="configs/pems03.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--gpu_id", type=int, default=None, choices=list(range(10)))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_batches", type=int, default=20, help="Use first N batches for statistics")
    parser.add_argument(
        "--noise_samples_per_step",
        type=int,
        default=1,
        help="Number of x_t noise draws per step per batch",
    )
    parser.add_argument("--out_prefix", type=str, default="outputs/fig3_router_weights")
    return parser.parse_args()


def _load_model_and_data(config: Dict, device: torch.device):
    """Build model/diffusion/data artifacts used for gate statistics."""
    dcfg = config["dataset"]
    mcfg = config["model"]
    diff_cfg = config["diffusion"]

    artifacts = build_dataloaders(config)
    split_loader = {
        "train": artifacts.train_loader,
        "val": artifacts.val_loader,
        "test": artifacts.test_loader,
    }

    root = Path(dcfg["data_root"]) / dcfg["name"]
    num_nodes = int(dcfg["num_nodes"])

    a_phy_np = load_or_build_physical_graph(
        file_path=root / dcfg["adjacency_file"],
        num_nodes=num_nodes,
        sigma=dcfg.get("physical_sigma", "auto"),
        normalize_mode=str(dcfg.get("physical_norm_mode", "sym")),
    )
    z_sem_np = load_semantic_embeddings(root / dcfg["semantic_embedding_file"])
    a_sem_np = load_or_build_semantic_graph(
        graph_path=root / dcfg["semantic_graph_file"],
        z_sem=z_sem_np,
        top_k=int(dcfg["semantic_top_k"]),
        rebuild=False,
        normalize_mode=str(dcfg.get("semantic_graph_norm_mode", "sym")),
        raw_graph_path=(
            root / dcfg["semantic_graph_raw_file"]
            if dcfg.get("semantic_graph_raw_file")
            else None
        ),
    )

    a_phy = to_torch(a_phy_np, device=device)
    a_sem = to_torch(a_sem_np, device=device)
    z_sem = torch.tensor(z_sem_np, dtype=torch.float32, device=device)

    model = PromptSTDiff(
        input_dim=int(dcfg["input_dim"]),
        sem_dim=int(z_sem_np.shape[1]),
        hidden_dim=int(mcfg["hidden_dim"]),
        horizon_steps=int(dcfg["horizon_steps"]),
        time_embed_dim=int(mcfg["time_embed_dim"]),
        router_hidden_dim=int(mcfg["router_hidden_dim"]),
        num_layers=int(mcfg["num_layers"]),
        dropout=float(mcfg["dropout"]),
        semantic_dropout_p=float(mcfg.get("semantic_dropout_p", 0.1)),
    ).to(device)

    process = DiffusionProcess(
        betas=build_beta_schedule(diff_cfg),
        clip_x0=bool(diff_cfg.get("clip_x0", True)),
    ).to(device)

    dynamic_bank = maybe_load_dynamic_semantic_bank(config, data_root=root, logger=get_logger())
    if dynamic_bank is not None and dynamic_bank.sem_dim != int(z_sem_np.shape[1]):
        get_logger().warning(
            "dynamic semantic dim mismatch: bank=%d, static=%d. Disable dynamic semantic for plotting.",
            dynamic_bank.sem_dim,
            int(z_sem_np.shape[1]),
        )
        dynamic_bank = None

    return model, process, split_loader, a_phy, a_sem, z_sem, dynamic_bank


@torch.no_grad()
def collect_alpha_stats(
    model: PromptSTDiff,
    process: DiffusionProcess,
    data_loader,
    a_phy: torch.Tensor,
    a_sem: torch.Tensor,
    z_sem: torch.Tensor,
    dynamic_bank,
    device: torch.device,
    max_batches: int,
    noise_samples_per_step: int,
) -> Dict[str, np.ndarray]:
    """Collect mean/std of alpha_k over batches and channels."""
    den = model.epsilon_theta
    model.eval()
    k_steps = int(process.num_steps)
    per_step_values: List[List[float]] = [[] for _ in range(k_steps)]

    for bi, batch in enumerate(data_loader, start=1):
        if bi > max_batches:
            break
        x_his = batch["x_his"].to(device=device, dtype=torch.float32)
        x_fut = batch["x_fut"].to(device=device, dtype=torch.float32)
        cutoff_step = batch["cutoff_step"].to(device=device, dtype=torch.long)
        bsz, horizon, num_nodes, _ = x_fut.shape

        if dynamic_bank is not None:
            z_sem_batch = dynamic_bank.compose(
                static_z_sem=z_sem,
                cutoff_steps=cutoff_step,
                num_nodes=num_nodes,
                device=device,
            )
        else:
            z_sem_batch = z_sem

        z = den._expand_z_sem(z_sem_batch, batch_size=bsz)
        h_time = den.temporal_encoder(x_his=x_his)

        for step in range(k_steps):
            t = torch.full((bsz,), step, dtype=torch.long, device=device)
            for _ in range(max(1, noise_samples_per_step)):
                noise = torch.randn_like(x_fut)
                x_t = process.q_sample(x_start=x_fut, t=t, noise=noise)
                h_traffic = den.traffic_encoder(
                    x_t=x_t,
                    h_time=h_time,
                    a_phy=a_phy,
                    a_sem=a_sem,
                )
                h_sem = den._build_sem_branch(z_sem=z, t=t, horizon=horizon)
                _, gate = den.router(h_sem=h_sem, h_traffic=h_traffic, t=t)  # [B,H,N,C]
                per_step_values[step].append(float(gate.mean().item()))

    means = np.array(
        [float(np.mean(v)) if len(v) > 0 else np.nan for v in per_step_values],
        dtype=np.float32,
    )
    stds = np.array(
        [float(np.std(v)) if len(v) > 0 else np.nan for v in per_step_values],
        dtype=np.float32,
    )
    ks = np.arange(1, k_steps + 1, dtype=np.int32)  # k=1..K
    return {"k": ks, "alpha_mean": means, "alpha_std": stds}


def plot_figure(stats: Dict[str, np.ndarray], out_prefix: Path) -> None:
    """Create publication-ready figure in PDF and PNG."""
    k = stats["k"]
    alpha_mean = np.clip(stats["alpha_mean"], 0.0, 1.0)
    alpha_std = np.nan_to_num(stats["alpha_std"], nan=0.0)
    alpha_low = np.clip(alpha_mean - alpha_std, 0.0, 1.0)
    alpha_high = np.clip(alpha_mean + alpha_std, 0.0, 1.0)

    num_mean = 1.0 - alpha_mean
    num_low = 1.0 - alpha_high
    num_high = 1.0 - alpha_low

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )

    fig, ax = plt.subplots(figsize=(3.3, 2.2), dpi=300)
    k_max = int(k.max())

    # Early vs late denoising background
    early_start = int(round(k_max * 0.6))
    late_end = int(round(k_max * 0.4))
    ax.axvspan(early_start, k_max, color="#d62728", alpha=0.06, lw=0)
    ax.axvspan(1, late_end, color="#1f77b4", alpha=0.06, lw=0)

    ax.plot(k, alpha_mean, color="#d62728", lw=1.8, label=r"Semantic guidance ($\alpha_k$)")
    ax.fill_between(k, alpha_low, alpha_high, color="#d62728", alpha=0.15, lw=0)

    ax.plot(
        k,
        num_mean,
        color="#1f77b4",
        lw=1.8,
        ls="--",
        label=r"Numerical refinement ($1-\alpha_k$)",
    )
    ax.fill_between(k, num_low, num_high, color="#1f77b4", alpha=0.12, lw=0)

    ax.set_xlim(k_max, 1)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Reverse diffusion step ($k$)")
    ax.set_ylabel("Routing weight")

    ticks = [k_max, int(round(k_max * 0.8)), int(round(k_max * 0.6)), int(round(k_max * 0.4)), int(round(k_max * 0.2)), 1]
    ticks = sorted(set([t for t in ticks if 1 <= t <= k_max]), reverse=True)
    ax.set_xticks(ticks)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))

    ax.text(int(round(k_max * 0.78)), 0.88, "Early denoising", color="#d62728", ha="center")
    ax.text(int(round(k_max * 0.22)), 0.12, "Late denoising", color="#1f77b4", ha="center")

    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="center right")

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.3)
    fig.savefig(str(out_prefix) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_prefix) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Main entry."""
    args = parse_args()
    logger = get_logger()

    config = load_config(args.config)
    device_arg = args.device
    if args.gpu_id is not None:
        device_arg = f"cuda:{int(args.gpu_id)}"
    device = get_device(device_arg)
    logger.info("Using device: %s", device)

    model, process, split_loader_map, a_phy, a_sem, z_sem, dynamic_bank = _load_model_and_data(
        config=config,
        device=device,
    )
    data_loader = split_loader_map[args.split]

    load_checkpoint(Path(args.ckpt), model=model, optimizer=None, map_location=str(device))
    logger.info("Loaded checkpoint: %s", args.ckpt)

    stats = collect_alpha_stats(
        model=model,
        process=process,
        data_loader=data_loader,
        a_phy=a_phy,
        a_sem=a_sem,
        z_sem=z_sem,
        dynamic_bank=dynamic_bank,
        device=device,
        max_batches=int(args.max_batches),
        noise_samples_per_step=int(args.noise_samples_per_step),
    )

    out_prefix = Path(args.out_prefix)
    np.savez(
        str(out_prefix) + "_stats.npz",
        k=stats["k"],
        alpha_mean=stats["alpha_mean"],
        alpha_std=stats["alpha_std"],
    )
    plot_figure(stats=stats, out_prefix=out_prefix)

    logger.info("Saved: %s.pdf", out_prefix)
    logger.info("Saved: %s.png", out_prefix)
    logger.info("Saved: %s_stats.npz", out_prefix)


if __name__ == "__main__":
    main()
