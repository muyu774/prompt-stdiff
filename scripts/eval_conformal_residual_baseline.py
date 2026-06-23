"""Evaluate split-conformal residual intervals around a frozen mean predictor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_experiment_and_record import _build_runtime
from trainers.evaluator import _inverse_with_scaler
from utils.config import load_config
from utils.device import get_device
from utils.metrics import mae as point_mae, rmse as point_rmse, mape as point_mape


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split-conformal residual interval baseline.")
    p.add_argument("--config", required=True)
    p.add_argument("--alpha", type=float, default=0.10, help="Miscoverage level, e.g. 0.10 for 90% interval.")
    p.add_argument("--split", choices=("test",), default="test")
    p.add_argument("--gpu_id", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--out_json", type=Path, required=True)
    return p.parse_args()


def _metric_feature(x: torch.Tensor, idx: Optional[int]) -> torch.Tensor:
    if idx is None:
        return x
    return x[..., int(idx):int(idx)+1]


def _quantile_conformal(values: np.ndarray, alpha: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    n = values.size
    rank = int(np.ceil((n + 1) * (1.0 - float(alpha))))
    rank = min(max(rank, 1), n)
    return float(np.partition(values, rank - 1)[rank - 1])


@torch.no_grad()
def collect_mean_and_target(config: Dict, split: str, device: torch.device):
    artifacts, model, _sampler, a_phy, a_sem, z_sem, dynamic_bank = _build_runtime(config, device)
    model.eval()
    loader = artifacts.val_loader if split == "val" else artifacts.test_loader
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    metric_idx = config.get("train", {}).get("metric_feature_index", None)
    for batch in loader:
        x_his = batch["x_his"].to(device=device, dtype=torch.float32)
        x_fut = batch["x_fut"].to(device=device, dtype=torch.float32)
        cutoff_step = batch["cutoff_step"].to(device=device, dtype=torch.long)
        b, h, n, f = x_fut.shape
        if dynamic_bank is not None:
            z_sem_batch = dynamic_bank.compose(static_z_sem=z_sem, cutoff_steps=cutoff_step, num_nodes=n, device=device)
        else:
            z_sem_batch = z_sem
        pred = model.predict_mean(
            x_his=x_his,
            a_phy=a_phy,
            a_sem=a_sem,
            z_sem=z_sem_batch,
            batch={"x_his": x_his, "x_fut": x_fut, "cutoff_step": cutoff_step},
        )
        pred_inv = _metric_feature(_inverse_with_scaler(pred, artifacts.scaler), metric_idx)
        target_inv = _metric_feature(_inverse_with_scaler(x_fut, artifacts.scaler), metric_idx)
        preds.append(pred_inv.cpu())
        targets.append(target_inv.cpu())
    return torch.cat(preds, dim=0), torch.cat(targets, dim=0)


def interval_metrics(pred: torch.Tensor, target: torch.Tensor, half_width: torch.Tensor, alpha: float, eps: float, mask_threshold: float):
    # pred/target [B,H,N,1], half_width [H]
    hw = half_width.view(1, -1, 1, 1).to(dtype=pred.dtype)
    lower = pred - hw
    upper = pred + hw
    covered = ((target >= lower) & (target <= upper)).float()
    width = (upper - lower)
    miss_low = (lower - target).clamp_min(0.0)
    miss_high = (target - upper).clamp_min(0.0)
    winkler = width + (2.0 / float(alpha)) * (miss_low + miss_high)
    return {
        "mae": float(point_mae(pred, target).item()),
        "rmse": float(point_rmse(pred, target).item()),
        "mape": float(point_mape(pred, target, eps=eps, mask_threshold=mask_threshold).item()),
        "picp@90": float(covered.mean().item()),
        "mpiw@90": float(width.mean().item()),
        "winkler@90": float(winkler.mean().item()),
        "sharpness": float(width.std().item()),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device_arg = args.device if args.gpu_id is None else f"cuda:{int(args.gpu_id)}"
    device = get_device(device_arg)
    alpha = float(args.alpha)
    eps = float(config.get("train", {}).get("mape_eps", 1e-5))
    mask_threshold = float(config.get("train", {}).get("mape_mask_threshold", 1.0))
    eval_horizons = [int(x) for x in config.get("train", {}).get("eval_horizons", [3, 6, 12])]

    val_pred, val_target = collect_mean_and_target(config, "val", device)
    test_pred, test_target = collect_mean_and_target(config, "test", device)
    abs_resid = (val_target - val_pred).abs().numpy()  # [B,H,N,1]
    h = abs_resid.shape[1]
    widths = np.asarray([_quantile_conformal(abs_resid[:, i, :, :].reshape(-1), alpha) for i in range(h)], dtype=np.float32)
    half_width = torch.tensor(widths, dtype=torch.float32)

    metrics = interval_metrics(test_pred, test_target, half_width, alpha, eps, mask_threshold)
    for hh in eval_horizons:
        idx = hh - 1
        if idx < 0 or idx >= h:
            continue
        sub = interval_metrics(test_pred[:, idx:idx+1], test_target[:, idx:idx+1], half_width[idx:idx+1], alpha, eps, mask_threshold)
        for k, v in sub.items():
            metrics[f"{k}@{hh}"] = v
    payload = {
        "method": "split_conformal_residual",
        "config": args.config,
        "alpha": alpha,
        "nominal_coverage": 1.0 - alpha,
        "half_width_by_horizon": widths.tolist(),
        "metrics": metrics,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
