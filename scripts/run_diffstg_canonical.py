"""Run official DiffSTG on this repo's canonical PeMS forecasting windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
import types
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataio.traffic_dataset import load_traffic_array
from utils.config import load_config
from utils.device import get_device


class EasyDict(dict):
    """Small fallback for easydict.EasyDict used by DiffSTG configs."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class CanonicalDiffSTGDataset(Dataset):
    """Canonical PeMS windows as (future, history, pos_w, pos_d)."""

    def __init__(
        self,
        raw_metric: np.ndarray,
        windows: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        history_steps: int,
        points_per_day: int,
    ) -> None:
        self.raw_metric = raw_metric.astype(np.float32)
        self.windows = windows.astype(np.int64)
        self.mean = mean.astype(np.float32).reshape(1, -1)
        self.std = np.where(std.astype(np.float32).reshape(1, -1) < 1e-6, 1.0, std.reshape(1, -1))
        self.history_steps = int(history_steps)
        self.points_per_day = int(points_per_day)

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, index: int):
        his_start, his_end, fut_end = [int(x) for x in self.windows[index]]
        history = self.raw_metric[his_start:his_end]
        future = self.raw_metric[his_end:fut_end]
        history = ((history - self.mean) / self.std).astype(np.float32)[..., None]
        future = ((future - self.mean) / self.std).astype(np.float32)[..., None]

        idx = np.arange(his_start, his_end, dtype=np.int64)
        pos_w = ((idx // self.points_per_day) % 7).astype(np.int64)
        pos_d = (idx % self.points_per_day).astype(np.int64)[:, None]
        return (
            torch.from_numpy(future),
            torch.from_numpy(history),
            torch.from_numpy(pos_w),
            torch.from_numpy(pos_d),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate DiffSTG on canonical PeMS windows.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--canonical_npz", type=str, required=True)
    parser.add_argument("--diffstg_repo", type=str, default="baselines/external_repos/DiffSTG")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gpu_id", type=int, default=None, choices=list(range(16)))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--valid_interval", type=int, default=5)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=200)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--sample_steps", type=int, default=40)
    parser.add_argument("--sample_strategy", type=str, default="ddim_multi", choices=("ddpm", "ddim_multi", "ddim_one"))
    parser.add_argument("--nsample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--metric_feature_index", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--out_npz", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--identity_adj", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metric_scaler(canonical: Mapping[str, np.ndarray], metric_feature_index: int) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(canonical["scaler_mean"], dtype=np.float32)
    std = np.asarray(canonical["scaler_std"], dtype=np.float32)
    if mean.ndim == 3:
        mean = mean[..., metric_feature_index]
    if std.ndim == 3:
        std = std[..., metric_feature_index]
    return mean.reshape(-1), std.reshape(-1)


def _load_adjacency(config: Mapping, num_nodes: int, identity_adj: bool) -> np.ndarray:
    if identity_adj:
        return np.eye(num_nodes, dtype=np.float32)
    dcfg = config["dataset"]
    adj_path = Path(dcfg["data_root"]) / str(dcfg["name"]) / str(dcfg.get("adjacency_file", "adjacency.csv"))
    adj = np.eye(num_nodes, dtype=np.float32)
    if not adj_path.exists():
        return adj
    import csv

    with adj_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        src_key = "src" if "src" in cols else "from"
        dst_key = "dst" if "dst" in cols else "to"
        for row in reader:
            try:
                i = int(float(row[src_key]))
                j = int(float(row[dst_key]))
            except Exception:
                continue
            if 0 <= i < num_nodes and 0 <= j < num_nodes:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
    return adj.astype(np.float32)


def build_datasets(args: argparse.Namespace, config: Mapping) -> Tuple[Dataset, Dataset, Dataset, int, int, int, np.ndarray]:
    canonical = np.load(args.canonical_npz)
    dcfg = config["dataset"]
    raw_path = Path(dcfg["data_root"]) / str(dcfg["name"]) / str(dcfg["data_file"])
    raw = load_traffic_array(raw_path)
    metric_feature_index = args.metric_feature_index
    if metric_feature_index is None:
        metric_feature_index = int(config.get("train", {}).get("metric_feature_index", 0))
    raw_metric = raw[..., int(metric_feature_index)]
    mean, std = _metric_scaler(canonical, metric_feature_index=int(metric_feature_index))
    history_steps = int(dcfg["history_steps"])
    horizon_steps = int(dcfg["horizon_steps"])
    points_per_day = int(24 * 60 / int(dcfg.get("time_freq_minutes", 5)))
    adj = _load_adjacency(config, num_nodes=int(raw_metric.shape[1]), identity_adj=bool(args.identity_adj))

    kwargs = {
        "raw_metric": raw_metric,
        "mean": mean,
        "std": std,
        "history_steps": history_steps,
        "points_per_day": points_per_day,
    }
    return (
        CanonicalDiffSTGDataset(windows=canonical["train_windows"], **kwargs),
        CanonicalDiffSTGDataset(windows=canonical["val_windows"], **kwargs),
        CanonicalDiffSTGDataset(windows=canonical["test_windows"], **kwargs),
        int(raw_metric.shape[1]),
        history_steps,
        horizon_steps,
        adj,
    )


def _ensure_easydict_module() -> None:
    try:
        import easydict  # noqa: F401
        return
    except ModuleNotFoundError:
        module = types.ModuleType("easydict")
        module.EasyDict = EasyDict
        sys.modules["easydict"] = module


def import_diffstg(diffstg_repo: str):
    repo = Path(diffstg_repo).resolve()
    if not repo.exists():
        raise FileNotFoundError(f"DiffSTG repo not found: {repo}")
    _ensure_easydict_module()
    # DiffSTG has a top-level package named `utils`, which conflicts with this repo.
    for key in list(sys.modules):
        if key == "utils" or key.startswith("utils."):
            del sys.modules[key]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from algorithm.diffstg.model import DiffSTG  # type: ignore

    return DiffSTG


def make_model_config(
    args: argparse.Namespace,
    device: torch.device,
    num_nodes: int,
    history_steps: int,
    horizon_steps: int,
    adj: np.ndarray,
) -> EasyDict:
    cfg = EasyDict()
    cfg.N = int(args.diffusion_steps)
    cfg.sample_steps = int(args.sample_steps)
    cfg.sample_strategy = str(args.sample_strategy)
    cfg.device = device
    cfg.beta_end = float(args.beta_end)
    cfg.beta_schedule = "quad"
    cfg.epsilon_theta = "UGnet"
    cfg.T_h = int(history_steps)
    cfg.T_p = int(horizon_steps)
    cfg.V = int(num_nodes)
    cfg.F = 1
    cfg.d_h = int(args.hidden_size)
    cfg.C = int(args.hidden_size)
    cfg.n_channels = int(args.hidden_size)
    cfg.week_len = 7
    cfg.day_len = 288
    cfg.channel_multipliers = [1, 2]
    cfg.supports_len = 2
    cfg.A = adj.astype(np.float32)
    return cfg


def run_validation(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for future, history, pos_w, pos_d in loader:
            future = future.to(device)
            history = history.to(device)
            pos_w = pos_w.to(device)
            pos_d = pos_d.to(device)
            x = torch.cat((history, future), dim=1).transpose(1, 3)
            x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).transpose(1, 3)
            loss = model.loss(x, (x_masked, pos_w, pos_d))
            total += float(loss.item())
            count += 1
    return total / max(count, 1)


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    save_dir: Path,
) -> Path:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    best_path = save_dir / "best.pt"
    last_path = save_dir / "last.pt"
    best_loss = float("inf")

    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        count = 0
        for batch_idx, (future, history, pos_w, pos_d) in enumerate(train_loader, start=1):
            future = future.to(device)
            history = history.to(device)
            pos_w = pos_w.to(device)
            pos_d = pos_d.to(device)
            x = torch.cat((history, future), dim=1).transpose(1, 3)
            x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).transpose(1, 3)
            loss = 10.0 * model.loss(x, (x_masked, pos_w, pos_d))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            count += 1
            if args.max_train_batches is not None and batch_idx >= int(args.max_train_batches):
                break
        avg = total / max(count, 1)
        print(f"[epoch {epoch:03d}] train_loss={avg:.6f} time={time.time() - t0:.2f}s", flush=True)
        if epoch % int(args.valid_interval) == 0 or epoch == int(args.epochs):
            val_loss = run_validation(model, val_loader, device=device)
            scheduler.step(val_loss)
            print(f"[epoch {epoch:03d}] val_loss={val_loss:.6f}", flush=True)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), best_path)
                print(f"[epoch {epoch:03d}] saved best -> {best_path}", flush=True)
        torch.save(model.state_dict(), last_path)
    return best_path if best_path.exists() else last_path


def export_samples(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    horizon_steps: int,
    nsample: int,
    sample_strategy: str,
    sample_steps: int,
    out_npz: Path,
    max_eval_batches: Optional[int],
) -> None:
    model.eval()
    model.set_sample_strategy(sample_strategy)
    model.set_ddim_sample_steps(int(sample_steps))
    sample_chunks: List[np.ndarray] = []
    target_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for batch_idx, (future, history, pos_w, pos_d) in enumerate(test_loader, start=1):
            future = future.to(device)
            history = history.to(device)
            pos_w = pos_w.to(device)
            pos_d = pos_d.to(device)
            x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).transpose(1, 3)
            pred = model((x_masked, pos_w, pos_d), int(nsample))
            if pred.device != torch.device("cpu"):
                pred = pred.detach().cpu()
            # pred: [B,S,F,V,T], future: [B,H,V,F]
            pred_fut = pred[:, :, :, :, -horizon_steps:]
            samples_np = pred_fut.permute(1, 0, 4, 3, 2).numpy().astype(np.float32)
            target_np = future.detach().cpu().numpy().astype(np.float32)
            sample_chunks.append(samples_np)
            target_chunks.append(target_np)
            print(f"[export] batch={batch_idx} samples={samples_np.shape}", flush=True)
            if max_eval_batches is not None and batch_idx >= int(max_eval_batches):
                break
    samples_all = np.concatenate(sample_chunks, axis=1)
    target_all = np.concatenate(target_chunks, axis=0)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, samples=samples_all, target=target_all)
    print(f"[done] wrote {out_npz} samples={samples_all.shape} target={target_all.shape}", flush=True)


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    config = load_config(args.config)
    dataset_name = str(config["dataset"]["name"])
    device_arg = args.device if args.gpu_id is None else f"cuda:{int(args.gpu_id)}"
    device = get_device(device_arg)

    save_dir = Path(args.save_dir or f"outputs/prob_baselines/DiffSTG/{dataset_name}_run")
    out_npz = Path(args.out_npz or f"outputs/prob_baselines/DiffSTG/{dataset_name}_samples.npz")
    save_dir.mkdir(parents=True, exist_ok=True)

    train_set, val_set, test_set, num_nodes, history_steps, horizon_steps, adj = build_datasets(args, config)
    DiffSTG = import_diffstg(args.diffstg_repo)
    model_cfg = make_model_config(
        args=args,
        device=device,
        num_nodes=num_nodes,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        adj=adj,
    )
    with (save_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "model_config": dict(model_cfg), "num_nodes": num_nodes}, f, indent=2, default=str)

    train_loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=int(args.batch_size), shuffle=False, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=int(args.batch_size), shuffle=False, num_workers=0, drop_last=False)

    model = DiffSTG(model_cfg).to(device)
    if args.resume:
        ckpt = Path(args.resume)
        print(f"[load] {ckpt}", flush=True)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        best_path = ckpt
    else:
        best_path = train_model(model, train_loader, val_loader, device=device, args=args, save_dir=save_dir)
        print(f"[load best] {best_path}", flush=True)
        model.load_state_dict(torch.load(best_path, map_location=device))

    export_samples(
        model=model,
        test_loader=test_loader,
        device=device,
        horizon_steps=horizon_steps,
        nsample=int(args.nsample),
        sample_strategy=str(args.sample_strategy),
        sample_steps=int(args.sample_steps),
        out_npz=out_npz,
        max_eval_batches=args.max_eval_batches,
    )


if __name__ == "__main__":
    main()
