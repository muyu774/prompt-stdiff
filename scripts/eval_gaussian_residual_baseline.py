"""Evaluate a heteroscedastic Gaussian residual baseline.

This script answers a narrow ablation question: if we keep the same frozen
mean predictor and the same learned residual scale head, do we still need
residual diffusion?  It replaces diffusion samples with i.i.d. Gaussian
standardized residuals, then runs the same de-standardization, centering, and
calibration path used by Prompt-STDiff.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Mapping, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_experiment_and_record import _build_runtime
from trainers.evaluator import _inverse_with_scaler
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.device import get_device
from utils.metrics import compute_all_metrics, crps_ensemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate heteroscedastic Gaussian residual baseline.")
    parser.add_argument("--config", required=True, help="Model config.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint containing the residual scale head.")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu_id", type=int, default=None, choices=list(range(10)))
    parser.add_argument("--num_eval_samples", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed for Gaussian residuals.")
    parser.add_argument("--out_json", type=Path, default=None)
    return parser.parse_args()


def _select_metric_feature(x: torch.Tensor, metric_feature_index: Optional[int]) -> torch.Tensor:
    if metric_feature_index is None:
        return x
    idx = int(metric_feature_index)
    if idx < 0 or idx >= int(x.shape[-1]):
        raise ValueError(f"metric_feature_index={idx} is out of range for feature dim={x.shape[-1]}")
    return x[..., idx : idx + 1]


def _append_prob_metrics(bucket: Dict[str, List[float]], values: Mapping[str, float]) -> None:
    for key, value in values.items():
        if key in {"mae", "rmse", "mape"}:
            continue
        bucket.setdefault(key, []).append(float(value))


@torch.no_grad()
def evaluate_gaussian_residual(
    *,
    model,
    data_loader,
    a_phy: torch.Tensor,
    a_sem: torch.Tensor,
    z_sem: torch.Tensor,
    device: torch.device,
    scaler,
    num_samples: int,
    dynamic_bank=None,
    eval_horizons: Optional[List[int]] = None,
    max_batches: Optional[int] = None,
    metric_feature_index: Optional[int] = None,
    mape_eps: float = 1e-5,
    mape_mask_threshold: float = 1.0,
    seed: int = 0,
) -> Dict[str, float]:
    if not bool(getattr(model, "use_mean_head", False)):
        raise ValueError("Gaussian residual baseline requires use_mean_head=true with a frozen mean predictor.")

    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    pred_list: List[torch.Tensor] = []
    target_list: List[torch.Tensor] = []
    crps_list: List[float] = []
    crps_by_h: Dict[int, List[float]] = {}
    prob_metric_lists: Dict[str, List[float]] = {}
    prob_metric_by_h: Dict[int, Dict[str, List[float]]] = {}

    seen = 0
    for batch in data_loader:
        x_his = batch["x_his"].to(device=device, dtype=torch.float32)
        x_fut = batch["x_fut"].to(device=device, dtype=torch.float32)
        cutoff_step = batch["cutoff_step"].to(device=device, dtype=torch.long)
        b, h, n, f = x_fut.shape

        if dynamic_bank is not None:
            z_sem_batch = dynamic_bank.compose(
                static_z_sem=z_sem,
                cutoff_steps=cutoff_step,
                num_nodes=n,
                device=device,
            )
        else:
            z_sem_batch = z_sem

        mean_pred = model.predict_mean(
            x_his=x_his,
            a_phy=a_phy,
            a_sem=a_sem,
            z_sem=z_sem_batch,
            batch={"x_his": x_his, "x_fut": x_fut, "cutoff_step": cutoff_step},
        )

        # Same reconstruction path as diffusion evaluation, except residual_std is Gaussian.
        residual_std = torch.randn(
            (int(num_samples), b, h, n, f),
            dtype=x_fut.dtype,
            device=device,
            generator=gen,
        )
        residual = model.unstandardize_residual(residual_std)
        residual = model.calibrate_residual_samples(residual, x_his=x_his, z_sem=z_sem_batch)
        ensemble = residual + mean_pred.unsqueeze(0)

        pred_mean = ensemble.mean(dim=0)
        pred_inv = _inverse_with_scaler(pred_mean, scaler)
        target_inv = _inverse_with_scaler(x_fut, scaler)
        ensemble_inv = _inverse_with_scaler(ensemble, scaler)

        pred_eval = _select_metric_feature(pred_inv, metric_feature_index)
        target_eval = _select_metric_feature(target_inv, metric_feature_index)
        ensemble_eval = _select_metric_feature(ensemble_inv, metric_feature_index)

        pred_list.append(pred_eval)
        target_list.append(target_eval)

        crps = crps_ensemble(ensemble_eval, target_eval)
        crps_list.append(float(crps.item()))
        prob_metrics = compute_all_metrics(
            ensemble_eval,
            target_eval,
            mape_eps=mape_eps,
            mape_mask_threshold=mape_mask_threshold,
        )
        _append_prob_metrics(prob_metric_lists, prob_metrics)

        if eval_horizons:
            for hh in eval_horizons:
                h_idx = int(hh) - 1
                if h_idx < 0 or h_idx >= int(target_eval.shape[1]):
                    continue
                ens_h = ensemble_eval[:, :, h_idx : h_idx + 1]
                tgt_h = target_eval[:, h_idx : h_idx + 1]
                crps_h = crps_ensemble(ens_h, tgt_h)
                crps_by_h.setdefault(int(hh), []).append(float(crps_h.item()))
                prob_h = compute_all_metrics(
                    ens_h,
                    tgt_h,
                    mape_eps=mape_eps,
                    mape_mask_threshold=mape_mask_threshold,
                )
                bucket = prob_metric_by_h.setdefault(int(hh), {})
                _append_prob_metrics(bucket, prob_h)

        seen += 1
        if max_batches is not None and int(max_batches) > 0 and seen >= int(max_batches):
            break

    if not pred_list:
        raise RuntimeError("No batches were evaluated.")

    pred_all = torch.cat(pred_list, dim=0)
    target_all = torch.cat(target_list, dim=0)
    metrics = compute_all_metrics(
        pred_all,
        target_all,
        mape_eps=mape_eps,
        mape_mask_threshold=mape_mask_threshold,
    )
    metrics["crps"] = float(np.mean(crps_list))
    for key, vals in prob_metric_lists.items():
        metrics[key] = float(np.mean(vals)) if vals else float("nan")

    if eval_horizons:
        for hh in eval_horizons:
            h_idx = int(hh) - 1
            if h_idx < 0 or h_idx >= int(target_all.shape[1]):
                continue
            point_h = compute_all_metrics(
                pred_all[:, h_idx : h_idx + 1],
                target_all[:, h_idx : h_idx + 1],
                mape_eps=mape_eps,
                mape_mask_threshold=mape_mask_threshold,
            )
            metrics[f"mae@{hh}"] = point_h["mae"]
            metrics[f"rmse@{hh}"] = point_h["rmse"]
            metrics[f"mape@{hh}"] = point_h["mape"]
            metrics[f"crps@{hh}"] = float(np.mean(crps_by_h.get(int(hh), [float("nan")])) )
            for key, vals in prob_metric_by_h.get(int(hh), {}).items():
                metrics[f"{key}@{hh}"] = float(np.mean(vals)) if vals else float("nan")
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.num_eval_samples is not None:
        config.setdefault("train", {})["num_eval_samples"] = int(args.num_eval_samples)
    device_arg = args.device if args.gpu_id is None else f"cuda:{int(args.gpu_id)}"
    device = get_device(device_arg)

    artifacts, model, _sampler, a_phy, a_sem, z_sem, dynamic_bank = _build_runtime(config, device)
    load_checkpoint(Path(args.ckpt), model=model, optimizer=None, map_location=str(device), strict=False)
    loader = artifacts.val_loader if args.split == "val" else artifacts.test_loader
    metrics = evaluate_gaussian_residual(
        model=model,
        data_loader=loader,
        a_phy=a_phy,
        a_sem=a_sem,
        z_sem=z_sem,
        device=device,
        scaler=artifacts.scaler,
        num_samples=int(config.get("train", {}).get("num_eval_samples", 20)),
        dynamic_bank=dynamic_bank,
        eval_horizons=[int(x) for x in config.get("train", {}).get("eval_horizons", [3, 6, 12])],
        max_batches=args.max_batches,
        metric_feature_index=config.get("train", {}).get("metric_feature_index", None),
        mape_eps=float(config.get("train", {}).get("mape_eps", 1e-5)),
        mape_mask_threshold=float(config.get("train", {}).get("mape_mask_threshold", 1.0)),
        seed=int(args.seed),
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": "heteroscedastic_gaussian_residual",
            "config": args.config,
            "ckpt": args.ckpt,
            "split": args.split,
            "seed": int(args.seed),
            "metrics": metrics,
        }
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
