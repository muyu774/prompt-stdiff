"""Build node-group residual scale configs from frozen-mean validation residuals.

The generated configs keep the strong deterministic mean and learned
heteroscedastic head unchanged, then add a non-trainable node-group temperature
layer. Nodes are grouped by frozen-mean residual difficulty on a calibration
split, so the easy group can be sharpened while difficult nodes get wider
residual ensembles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataio.traffic_dataset import build_dataloaders
from models.mean_predictor import MeanPredictor
from utils.config import load_config
from utils.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build node-group residual scale configs.")
    parser.add_argument("--base_config", required=True, help="Base Prompt-STDiff config.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Directory for generated configs.")
    parser.add_argument("--split", choices=("train", "val"), default="val", help="Calibration split.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--num_groups", type=int, default=3)
    parser.add_argument("--hscale_start", type=float, default=1.06)
    parser.add_argument("--hscale_end", type=float, default=1.12)
    parser.add_argument("--tag_prefix", default="ng")
    return parser.parse_args()


def _loader(config: dict, split: str):
    artifacts = build_dataloaders(config)
    return artifacts.train_loader if split == "train" else artifacts.val_loader


@torch.no_grad()
def frozen_mean_node_mae(config: dict, split: str, device: torch.device) -> np.ndarray:
    """Return per-node frozen-mean MAE in normalized space."""
    loader = _loader(config, split)
    predictor = MeanPredictor(config=config, device=device).to(device)
    predictor.eval()

    num_nodes = int(config["dataset"]["num_nodes"])
    err_sum = torch.zeros(num_nodes, dtype=torch.float64, device=device)
    count = 0
    for batch in loader:
        x_his = batch["x_his"].to(device=device, dtype=torch.float32)
        x_fut = batch["x_fut"].to(device=device, dtype=torch.float32)
        cutoff_step = batch["cutoff_step"].to(device=device, dtype=torch.long)
        mu = predictor(batch={"x_his": x_his, "x_fut": x_fut, "cutoff_step": cutoff_step})
        err = torch.abs(x_fut - mu)  # [B,H,N,F]
        err_sum += err.double().sum(dim=(0, 1, 3))
        count += int(err.shape[0] * err.shape[1] * err.shape[3])
    return (err_sum / max(count, 1)).detach().cpu().numpy().astype(np.float32)


def equal_count_groups(scores: np.ndarray, num_groups: int) -> np.ndarray:
    """Assign group ids by score rank: 0=easiest, G-1=hardest."""
    if num_groups < 2:
        raise ValueError("num_groups must be >= 2")
    order = np.argsort(scores, kind="stable")
    groups = np.empty_like(order)
    splits = np.array_split(order, num_groups)
    for gid, idx in enumerate(splits):
        groups[idx] = gid
    return groups.astype(np.int64)


def _hscale(horizon: int, start: float, end: float) -> list[float]:
    return np.linspace(float(start), float(end), int(horizon)).round(4).astype(float).tolist()


def _scale_table(horizon: int, multipliers: Sequence[float]) -> list[list[float]]:
    return [[float(x) for x in multipliers] for _ in range(int(horizon))]


def main() -> None:
    args = parse_args()
    config = load_config(args.base_config)
    if args.gpu_id is not None:
        args.device = f"cuda:{int(args.gpu_id)}"
    device = get_device(args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scores = frozen_mean_node_mae(config, split=args.split, device=device)
    groups = equal_count_groups(scores, num_groups=int(args.num_groups))
    horizon = int(config["dataset"]["horizon_steps"])
    hscale = _hscale(horizon, args.hscale_start, args.hscale_end)

    # The variants are deliberately centered near 1.0 so the layer reallocates
    # interval width across node difficulty groups instead of merely inflating all nodes.
    variants = {
        "mild": [0.96, 1.00, 1.04],
        "medium": [0.92, 1.00, 1.08],
        "wide": [0.88, 1.00, 1.12],
        "asym": [0.94, 0.99, 1.10],
    }
    if int(args.num_groups) != 3:
        lo = np.linspace(0.94, 1.0, int(args.num_groups), endpoint=False)
        hi = np.array([1.08], dtype=float)
        variants = {"rank_linear": np.concatenate([lo, hi]).round(4).astype(float).tolist()}

    meta = {
        "base_config": str(args.base_config),
        "split": str(args.split),
        "num_groups": int(args.num_groups),
        "hscale": hscale,
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "group_counts": [int((groups == gid).sum()) for gid in range(int(args.num_groups))],
    }
    np.savez(args.out_dir / f"{args.tag_prefix}_node_groups.npz", scores=scores, groups=groups)
    with open(args.out_dir / f"{args.tag_prefix}_node_groups.json", "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    for name, multipliers in variants.items():
        if len(multipliers) != int(args.num_groups):
            raise ValueError(f"Variant {name} has {len(multipliers)} multipliers, expected {args.num_groups}")
        cfg = dict(config)
        cfg["model"] = dict(config["model"])
        cfg["train"] = dict(config["train"])
        cfg["model"]["residual_sample_scale"] = 1.0
        cfg["model"]["residual_horizon_scale"] = hscale
        cfg["model"]["residual_node_group_ids"] = [int(x) for x in groups.tolist()]
        cfg["model"]["residual_node_group_scale"] = _scale_table(horizon, multipliers)
        out = args.out_dir / f"{args.tag_prefix}_{name}.yaml"
        with open(out, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"{name}: multipliers={multipliers} config={out}")
    print(f"meta={args.out_dir / f'{args.tag_prefix}_node_groups.json'}")


if __name__ == "__main__":
    main()
