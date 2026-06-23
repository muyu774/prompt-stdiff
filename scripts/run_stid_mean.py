#!/usr/bin/env python
"""Train/evaluate STID-style deterministic mean predictors on canonical splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataio.traffic_dataset import build_dataloaders
from models.stid_mean import STIDMeanConfig, STIDMeanModel
from utils.config import deep_merge, load_config
from utils.device import get_device
from utils.metrics import compute_all_metrics
from utils.result_writer import ExperimentResult, write_experiment_results
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train STID deterministic frozen mean candidate.")
    p.add_argument("--config", required=True)
    p.add_argument("--gpu_id", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--tag", required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--node_emb_dim", type=int, default=64)
    p.add_argument("--horizon_emb_dim", type=int, default=16)
    p.add_argument("--time_emb_dim", type=int, default=16)
    p.add_argument("--day_emb_dim", type=int, default=16)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no_time_embeddings", action="store_true")
    p.add_argument("--input_feature_index", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--max_train_batches", type=int, default=None)
    p.add_argument("--max_eval_batches", type=int, default=None)
    p.add_argument("--output_csv", default="outputs/results.csv")
    p.add_argument("--results_md", default="RESULTS.md")
    return p.parse_args()


def select_feature(x: torch.Tensor, idx: int) -> torch.Tensor:
    return x[..., int(idx) : int(idx) + 1]


def inverse(x: torch.Tensor, scaler: object | None) -> torch.Tensor:
    if scaler is None:
        return x
    arr = x.detach().cpu().numpy()
    inv = scaler.inverse_transform(arr)
    return torch.from_numpy(inv).to(x.device)


def evaluate_point(model: STIDMeanModel, loader, device, scaler, metric_feature_index: int, horizons, max_batches=None) -> dict:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            x_his = select_feature(batch["x_his"].to(device=device, dtype=torch.float32), metric_feature_index)
            x_fut = select_feature(batch["x_fut"].to(device=device, dtype=torch.float32), metric_feature_index)
            cutoff_step = batch["cutoff_step"].to(device=device)
            pred = model(x_his, cutoff_step=cutoff_step)
            preds.append(inverse(pred, scaler))
            targets.append(inverse(x_fut, scaler))
            if max_batches is not None and batch_idx >= int(max_batches):
                break
    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    out = compute_all_metrics(pred_all, target_all)
    for hh in horizons:
        idx = int(hh) - 1
        m = compute_all_metrics(pred_all[:, idx : idx + 1], target_all[:, idx : idx + 1])
        out[f"mae@{hh}"] = m["mae"]
        out[f"rmse@{hh}"] = m["rmse"]
        out[f"mape@{hh}"] = m["mape"]
    return out


def result_rows(metrics: Mapping[str, float], cfg: Mapping, args: argparse.Namespace, ckpt: Path) -> list[ExperimentResult]:
    horizons = [int(x) for x in cfg["train"].get("eval_horizons", [3, 6, 12])]
    settings = {
        "hidden_dim": args.hidden_dim,
        "node_emb_dim": args.node_emb_dim,
        "horizon_emb_dim": args.horizon_emb_dim,
        "time_emb_dim": args.time_emb_dim,
        "day_emb_dim": args.day_emb_dim,
        "use_time_embeddings": not bool(args.no_time_embeddings),
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "input_feature_index": args.input_feature_index,
    }
    rows = []
    for hh in horizons:
        rows.append(
            ExperimentResult(
                dataset=str(cfg["dataset"]["name"]),
                method="STID-Frozen-Mean-Candidate",
                setting=args.tag,
                horizon=hh,
                mae=float(metrics.get(f"mae@{hh}", metrics["mae"])),
                rmse=float(metrics.get(f"rmse@{hh}", metrics["rmse"])),
                crps=None,
                seed=int(args.seed),
                config=args.config,
                implementation="ours-stid-mean",
                checkpoint=str(ckpt),
                settings_json=json.dumps(settings, sort_keys=True),
                notes="deterministic stronger frozen mean candidate",
            )
        )
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if int(args.input_feature_index) != 0:
        raise ValueError("run_stid_mean currently expects input_feature_index=0 so scaler and output stay single-channel.")
    if args.batch_size is not None:
        cfg = deep_merge(cfg, {"dataset": {"batch_size": int(args.batch_size)}})
    # Keep the scaler single-channel. Otherwise PeMS04's 3-feature scaler would
    # broadcast [B,H,N,1] predictions to [B,H,N,3] during inverse_transform.
    cfg = deep_merge(cfg, {"dataset": {"input_dim": 1}})
    set_seed(int(args.seed))
    device = get_device(f"cuda:{args.gpu_id}" if args.gpu_id is not None else args.device)
    artifacts = build_dataloaders(cfg)
    dcfg = cfg["dataset"]
    horizons = [int(x) for x in cfg["train"].get("eval_horizons", [3, 6, 12])]

    model_cfg = STIDMeanConfig(
        num_nodes=int(dcfg["num_nodes"]),
        history_steps=int(dcfg["history_steps"]),
        horizon_steps=int(dcfg["horizon_steps"]),
        input_dim=1,
        output_dim=1,
        hidden_dim=int(args.hidden_dim),
        node_emb_dim=int(args.node_emb_dim),
        horizon_emb_dim=int(args.horizon_emb_dim),
        time_emb_dim=int(args.time_emb_dim),
        day_emb_dim=int(args.day_emb_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        use_time_embeddings=not bool(args.no_time_embeddings),
        steps_per_day=int(dcfg.get("steps_per_day", 288)),
        days_per_week=7,
    )
    model = STIDMeanModel(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(args.epochs), 1))

    save_dir = Path("outputs/checkpoints/stid_mean") / str(dcfg["name"]) / args.tag
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best.pt"
    last_path = save_dir / "last.pt"
    best_val = float("inf")
    bad = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0.0
        count = 0
        for batch_idx, batch in enumerate(artifacts.train_loader, start=1):
            x_his = select_feature(batch["x_his"].to(device=device, dtype=torch.float32), int(args.input_feature_index))
            x_fut = select_feature(batch["x_fut"].to(device=device, dtype=torch.float32), int(args.input_feature_index))
            cutoff_step = batch["cutoff_step"].to(device=device)
            pred = model(x_his, cutoff_step=cutoff_step)
            loss = F.l1_loss(pred, x_fut)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item())
            count += 1
            if args.max_train_batches is not None and batch_idx >= int(args.max_train_batches):
                break
        sched.step()
        train_loss = total / max(count, 1)
        val_metrics = evaluate_point(
            model,
            artifacts.val_loader,
            device=device,
            scaler=artifacts.scaler,
            metric_feature_index=int(args.input_feature_index),
            horizons=horizons,
            max_batches=args.max_eval_batches,
        )
        val_mae = float(val_metrics["mae"])
        print(f"[epoch {epoch:03d}] train_l1_norm={train_loss:.6f} val_mae={val_mae:.6f} val_rmse={val_metrics['rmse']:.6f}", flush=True)
        bundle = {
            "model": model.state_dict(),
            "model_config": model_cfg.to_dict(),
            "dataset": dict(dcfg),
            "input_feature_index": int(args.input_feature_index),
            "tag": args.tag,
            "epoch": epoch,
            "val_metrics": val_metrics,
        }
        torch.save(bundle, last_path)
        if val_mae < best_val:
            best_val = val_mae
            bad = 0
            torch.save(bundle, best_path)
            print(f"[epoch {epoch:03d}] saved best -> {best_path}", flush=True)
        else:
            bad += 1
            if bad >= int(args.patience):
                print(f"[early_stop] epoch={epoch} best_val_mae={best_val:.6f}", flush=True)
                break

    ckpt = torch.load(best_path if best_path.exists() else last_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate_point(
        model,
        artifacts.test_loader,
        device=device,
        scaler=artifacts.scaler,
        metric_feature_index=int(args.input_feature_index),
        horizons=horizons,
        max_batches=None,
    )
    print(f"[test] MAE={test_metrics['mae']:.6f} RMSE={test_metrics['rmse']:.6f}", flush=True)
    for hh in horizons:
        print(f"[test] H{hh} MAE={test_metrics[f'mae@{hh}']:.6f} RMSE={test_metrics[f'rmse@{hh}']:.6f}", flush=True)
    write_experiment_results(
        result_rows(test_metrics, cfg=cfg, args=args, ckpt=best_path if best_path.exists() else last_path),
        csv_path=Path(args.output_csv),
        md_path=Path(args.results_md),
        title="STID Frozen Mean Candidate Results",
    )


if __name__ == "__main__":
    main()
