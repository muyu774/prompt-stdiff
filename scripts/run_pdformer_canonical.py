#!/usr/bin/env python
"""Train/evaluate official PDFormer on this repo's canonical PeMS splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataio.traffic_dataset import build_dataloaders
from graph.physical_graph import load_or_build_physical_graph
from utils.config import deep_merge, load_config
from utils.device import get_device
from utils.metrics import compute_all_metrics
from utils.result_writer import ExperimentResult, write_experiment_results
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Canonical PDFormer deterministic mean candidate.")
    p.add_argument("--config", required=True)
    p.add_argument("--pdformer_repo", default="baselines/external_repos/PDFormer")
    p.add_argument("--gpu_id", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--tag", required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-2)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--skip_dim", type=int, default=256)
    p.add_argument("--enc_depth", type=int, default=4)
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--lape_dim", type=int, default=8)
    p.add_argument("--n_cluster", type=int, default=16)
    p.add_argument("--s_attn_size", type=int, default=3)
    p.add_argument("--far_mask_delta", type=float, default=7.0)
    p.add_argument("--dtw_delta", type=int, default=5)
    p.add_argument("--input_feature_index", type=int, default=0)
    p.add_argument("--no_time_features", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--max_train_batches", type=int, default=None)
    p.add_argument("--max_eval_batches", type=int, default=None)
    p.add_argument("--output_csv", default="outputs/results.csv")
    p.add_argument("--results_md", default="RESULTS.md")
    return p.parse_args()


def import_pdformer(repo: str):
    repo_path = str(Path(repo).resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    from libcity.model.traffic_flow_prediction.PDFormer import PDFormer  # type: ignore

    return PDFormer


def select_feature(x: torch.Tensor, idx: int) -> torch.Tensor:
    return x[..., int(idx) : int(idx) + 1]


def append_pdformer_time_features(
    x_flow: torch.Tensor,
    cutoff_step: torch.Tensor,
    steps_per_day: int,
    days_per_week: int = 7,
) -> torch.Tensor:
    """Append PDFormer official temporal channels: flow, tod scalar, dow one-hot."""
    if x_flow.dim() != 4 or x_flow.shape[-1] != 1:
        raise ValueError(f"Expected x_flow [B,T,N,1], got {tuple(x_flow.shape)}")
    b, t, n, _ = x_flow.shape
    device = x_flow.device
    start = cutoff_step.to(device=device, dtype=torch.long).view(b, 1) - int(t)
    offsets = torch.arange(t, device=device, dtype=torch.long).view(1, t)
    steps = start + offsets
    tod = torch.remainder(steps, int(steps_per_day)).to(dtype=x_flow.dtype) / float(steps_per_day)
    dow_idx = torch.remainder(torch.div(steps, int(steps_per_day), rounding_mode="floor"), int(days_per_week))
    dow = torch.nn.functional.one_hot(dow_idx, num_classes=int(days_per_week)).to(dtype=x_flow.dtype)
    tod = tod[:, :, None, None].expand(b, t, n, 1)
    dow = dow[:, :, None, :].expand(b, t, n, int(days_per_week))
    return torch.cat([x_flow, tod, dow], dim=-1)


def inverse(x: torch.Tensor, scaler: object | None) -> torch.Tensor:
    if scaler is None:
        return x
    inv = scaler.inverse_transform(x.detach().cpu().numpy())
    return torch.from_numpy(inv).to(x.device)


def normalized_laplacian_pe(adj: np.ndarray, lape_dim: int) -> np.ndarray:
    a = np.asarray(adj, dtype=np.float64)
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 1.0)
    deg = np.maximum(a.sum(axis=1), 1e-8)
    inv_sqrt = np.power(deg, -0.5)
    lap = np.eye(a.shape[0]) - (inv_sqrt[:, None] * a) * inv_sqrt[None, :]
    vals, vecs = np.linalg.eigh(lap)
    order = np.argsort(vals)
    vecs = vecs[:, order]
    start = 1 if vecs.shape[1] > 1 else 0
    pe = vecs[:, start : start + int(lape_dim)]
    if pe.shape[1] < int(lape_dim):
        pe = np.pad(pe, ((0, 0), (0, int(lape_dim) - pe.shape[1])), mode="constant")
    return pe.astype(np.float32)


def hop_and_dtw_masks(adj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(adj, dtype=np.float32)
    n = a.shape[0]
    connected = (a > 0).astype(np.float32)
    np.fill_diagonal(connected, 1.0)
    # Lightweight hop proxy: direct edges are 1, non-edges are far. This is
    # enough for PDFormer's mask semantics without an expensive all-pairs pass.
    sh = np.where(connected > 0, 1.0, 511.0).astype(np.float32)
    np.fill_diagonal(sh, 0.0)
    # Use graph proximity as a deterministic DTW proxy for semantic attention.
    dtw = np.where(connected > 0, 1.0, 2.0).astype(np.float32)
    np.fill_diagonal(dtw, 0.0)
    return sh, dtw


def collect_pattern_keys(train_loader, s_attn_size: int, n_cluster: int, feature_idx: int, seed: int) -> np.ndarray:
    chunks = []
    for batch in train_loader:
        x = select_feature(batch["x_his"], feature_idx)[:, :s_attn_size]  # [B,S,N,1]
        x = x.permute(0, 2, 1, 3).reshape(-1, s_attn_size, 1).numpy()
        chunks.append(x)
        if sum(c.shape[0] for c in chunks) >= max(n_cluster * 128, 4096):
            break
    arr = np.concatenate(chunks, axis=0)
    rng = np.random.default_rng(seed)
    if arr.shape[0] >= n_cluster:
        idx = rng.choice(arr.shape[0], size=n_cluster, replace=False)
        keys = arr[idx]
    else:
        idx = rng.choice(arr.shape[0], size=n_cluster, replace=True)
        keys = arr[idx]
    return keys.astype(np.float32)


def make_pdformer_config(cfg: Mapping[str, Any], args: argparse.Namespace, device: torch.device) -> dict:
    d = cfg["dataset"]
    return {
        "dataset": str(d["name"]),
        "device": device,
        "world_size": 1,
        "embed_dim": int(args.embed_dim),
        "skip_dim": int(args.skip_dim),
        "lape_dim": int(args.lape_dim),
        "geo_num_heads": 4,
        "sem_num_heads": 2,
        "t_num_heads": 2,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop": 0.0,
        "attn_drop": 0.0,
        "drop_path": float(args.drop_path),
        "s_attn_size": int(args.s_attn_size),
        "t_attn_size": 1,
        "enc_depth": int(args.enc_depth),
        "type_ln": "pre",
        "type_short_path": "hop",
        "output_dim": 1,
        "input_window": int(d["history_steps"]),
        "output_window": int(d["horizon_steps"]),
        "add_time_in_day": False,
        "add_day_in_week": False,
        "huber_delta": 1,
        "far_mask_delta": float(args.far_mask_delta),
        "dtw_delta": int(args.dtw_delta),
        "use_curriculum_learning": False,
        "step_size": 10**9,
        "max_epoch": int(args.epochs),
        "task_level": int(d["horizon_steps"]),
    }


def make_data_feature(cfg: Mapping[str, Any], args: argparse.Namespace, artifacts, device: torch.device) -> tuple[dict, torch.Tensor]:
    d = cfg["dataset"]
    root = Path(d["data_root"]) / str(d["name"])
    adj = load_or_build_physical_graph(
        root / d["adjacency_file"],
        num_nodes=int(d["num_nodes"]),
        sigma=d.get("physical_sigma", "auto"),
        add_loop=True,
        normalize=False,
    ).astype(np.float32)
    sh, dtw = hop_and_dtw_masks(adj)
    lap = normalized_laplacian_pe(adj, int(args.lape_dim))
    pattern_keys = collect_pattern_keys(
        artifacts.train_loader,
        s_attn_size=int(args.s_attn_size),
        n_cluster=int(args.n_cluster),
        feature_idx=int(args.input_feature_index),
        seed=int(args.seed),
    )
    use_time = not bool(args.no_time_features)
    feature = {
        "scaler": None,
        "adj_mx": adj,
        "sd_mx": adj,
        "sh_mx": sh,
        "ext_dim": 8 if use_time else 0,
        "num_nodes": int(d["num_nodes"]),
        "feature_dim": 9 if use_time else 1,
        "output_dim": 1,
        "num_batches": len(artifacts.train_loader),
        "dtw_matrix": dtw,
        "pattern_keys": pattern_keys,
    }
    return feature, torch.tensor(lap, dtype=torch.float32, device=device)


def run_eval(model, loader, lap_mx, device, scaler, feature_idx: int, horizons, max_batches=None) -> dict:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            x_his = select_feature(batch["x_his"].to(device=device, dtype=torch.float32), feature_idx)
            x_fut = select_feature(batch["x_fut"].to(device=device, dtype=torch.float32), feature_idx)
            if bool(getattr(model, "_prompt_stdiff_use_time_features", False)):
                x_his = append_pdformer_time_features(
                    x_his,
                    cutoff_step=batch["cutoff_step"].to(device=device),
                    steps_per_day=int(getattr(model, "_prompt_stdiff_steps_per_day", 288)),
                )
            pred = model.predict({"X": x_his, "y": x_fut}, lap_mx=lap_mx)
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


def rows_from_metrics(metrics: Mapping[str, float], cfg: Mapping, args: argparse.Namespace, ckpt: Path) -> list[ExperimentResult]:
    horizons = [int(x) for x in cfg["train"].get("eval_horizons", [3, 6, 12])]
    settings = {
        "embed_dim": args.embed_dim,
        "skip_dim": args.skip_dim,
        "enc_depth": args.enc_depth,
        "drop_path": args.drop_path,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "canonical_no_external": True,
        "use_generated_time_features": not bool(args.no_time_features),
    }
    return [
        ExperimentResult(
            dataset=str(cfg["dataset"]["name"]),
            method="PDFormer-Frozen-Mean-Candidate",
            setting=args.tag,
            horizon=int(hh),
            mae=float(metrics.get(f"mae@{hh}", metrics["mae"])),
            rmse=float(metrics.get(f"rmse@{hh}", metrics["rmse"])),
            crps=None,
            seed=int(args.seed),
            config=args.config,
            implementation="official-model-canonical-data",
            checkpoint=str(ckpt),
            settings_json=json.dumps(settings, sort_keys=True),
            notes="PDFormer official model on repo canonical split; no external time features",
        )
        for hh in horizons
    ]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if int(args.input_feature_index) != 0:
        raise ValueError("PDFormer canonical runner currently expects input_feature_index=0.")
    cfg = deep_merge(cfg, {"dataset": {"input_dim": 1, "batch_size": int(args.batch_size)}})
    set_seed(int(args.seed))
    device = get_device(f"cuda:{args.gpu_id}" if args.gpu_id is not None else args.device)
    artifacts = build_dataloaders(cfg)
    PDFormer = import_pdformer(args.pdformer_repo)
    data_feature, lap_mx = make_data_feature(cfg, args, artifacts, device=device)
    model_config = make_pdformer_config(cfg, args, device=device)
    use_time = not bool(args.no_time_features)
    model_config["add_time_in_day"] = use_time
    model_config["add_day_in_week"] = use_time
    model = PDFormer(model_config, data_feature).to(device)
    model._prompt_stdiff_use_time_features = use_time
    model._prompt_stdiff_steps_per_day = int(cfg["dataset"].get("steps_per_day", 288))
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(args.epochs), 1), eta_min=1e-4)
    horizons = [int(x) for x in cfg["train"].get("eval_horizons", [3, 6, 12])]
    save_dir = Path("outputs/checkpoints/pdformer_mean") / str(cfg["dataset"]["name"]) / args.tag
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
            if use_time:
                x_his = append_pdformer_time_features(
                    x_his,
                    cutoff_step=batch["cutoff_step"].to(device=device),
                    steps_per_day=int(cfg["dataset"].get("steps_per_day", 288)),
                )
            pred = model({"X": x_his, "y": x_fut}, lap_mx=lap_mx)
            loss = F.smooth_l1_loss(pred, x_fut, beta=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item())
            count += 1
            if args.max_train_batches is not None and batch_idx >= int(args.max_train_batches):
                break
        sched.step()
        val_metrics = run_eval(
            model,
            artifacts.val_loader,
            lap_mx=lap_mx,
            device=device,
            scaler=artifacts.scaler,
            feature_idx=int(args.input_feature_index),
            horizons=horizons,
            max_batches=args.max_eval_batches,
        )
        val_mae = float(val_metrics["mae"])
        print(f"[epoch {epoch:03d}] train_huber_norm={total / max(count, 1):.6f} val_mae={val_mae:.6f} val_rmse={val_metrics['rmse']:.6f}", flush=True)
        serial_feature = {k: v for k, v in data_feature.items() if k != "scaler"}
        bundle = {
            "model": model.state_dict(),
            "model_config": {k: v for k, v in model_config.items() if k != "device"},
            "data_feature": serial_feature,
            "lap_mx": lap_mx.detach().cpu().numpy().astype(np.float32),
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

    ckpt_path = best_path if best_path.exists() else last_path
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_metrics = run_eval(
        model,
        artifacts.test_loader,
        lap_mx=lap_mx,
        device=device,
        scaler=artifacts.scaler,
        feature_idx=int(args.input_feature_index),
        horizons=horizons,
        max_batches=None,
    )
    print(f"[test] MAE={test_metrics['mae']:.6f} RMSE={test_metrics['rmse']:.6f}", flush=True)
    for hh in horizons:
        print(f"[test] H{hh} MAE={test_metrics[f'mae@{hh}']:.6f} RMSE={test_metrics[f'rmse@{hh}']:.6f}", flush=True)
    write_experiment_results(
        rows_from_metrics(test_metrics, cfg=cfg, args=args, ckpt=ckpt_path),
        csv_path=Path(args.output_csv),
        md_path=Path(args.results_md),
        title="PDFormer Frozen Mean Candidate Results",
    )


if __name__ == "__main__":
    main()
