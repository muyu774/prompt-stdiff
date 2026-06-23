"""Run official CSDI forecasting on this repo's canonical PeMS windows.

The script keeps the official CSDI model code untouched. It adapts PeMS04/08
windows and scalers exported by ``scripts/export_canonical_setup.py`` into the
batch format expected by CSDI_Forecasting, then exports samples as:

    samples: [S, B, H, N, 1]
    target:  [B, H, N, 1]

Use ``scripts/eval_probabilistic_npz.py --space normalized`` to evaluate the
exported NPZ under the shared metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataio.traffic_dataset import load_traffic_array
from utils.config import load_config
from utils.device import get_device


class CanonicalPeMSForecastingDataset(Dataset):
    """Canonical PeMS windows in official CSDI forecasting format."""

    def __init__(
        self,
        raw_metric: np.ndarray,
        windows: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        history_steps: int,
    ) -> None:
        if raw_metric.ndim != 2:
            raise ValueError(f"Expected raw metric [T,N], got {raw_metric.shape}")
        if windows.ndim != 2 or windows.shape[1] != 3:
            raise ValueError(f"Expected windows [B,3], got {windows.shape}")
        self.raw_metric = raw_metric.astype(np.float32)
        self.windows = windows.astype(np.int64)
        self.mean = mean.astype(np.float32).reshape(1, -1)
        self.std = np.where(std.astype(np.float32).reshape(1, -1) < 1e-6, 1.0, std.reshape(1, -1))
        self.history_steps = int(history_steps)
        self.target_dim = int(raw_metric.shape[1])

        if self.mean.shape[1] not in (1, self.target_dim):
            raise ValueError(f"Scaler mean shape {self.mean.shape} is incompatible with N={self.target_dim}")
        if self.std.shape[1] not in (1, self.target_dim):
            raise ValueError(f"Scaler std shape {self.std.shape} is incompatible with N={self.target_dim}")

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        his_start, his_end, fut_end = [int(x) for x in self.windows[index]]
        seq = self.raw_metric[his_start:fut_end]
        seq = (seq - self.mean) / self.std
        observed_mask = np.isfinite(seq).astype(np.float32)
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        gt_mask = observed_mask.copy()
        gt_mask[self.history_steps :] = 0.0
        timepoints = np.arange(seq.shape[0], dtype=np.float32)

        return {
            "observed_data": torch.from_numpy(seq),
            "observed_mask": torch.from_numpy(observed_mask),
            "gt_mask": torch.from_numpy(gt_mask),
            "timepoints": torch.from_numpy(timepoints),
            "feature_id": torch.arange(self.target_dim, dtype=torch.float32),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate official CSDI on canonical PeMS windows.")
    parser.add_argument("--config", type=str, required=True, help="This repo dataset config, e.g. configs/pems04.yaml.")
    parser.add_argument("--canonical_npz", type=str, required=True)
    parser.add_argument("--csdi_repo", type=str, default="baselines/external_repos/CSDI")
    parser.add_argument("--csdi_config", type=str, default="config/base_forecasting.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gpu_id", type=int, default=None, choices=list(range(16)))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--valid_interval", type=int, default=5)
    parser.add_argument("--itr_per_epoch", type=float, default=1e8)
    parser.add_argument("--num_sample_features", type=int, default=64)
    parser.add_argument("--diffusion_steps", type=int, default=50)
    parser.add_argument("--nsample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric_feature_index", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--out_npz", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
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


def build_datasets(args: argparse.Namespace, config: Mapping) -> Tuple[Dataset, Dataset, Dataset, int, int]:
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

    train_set = CanonicalPeMSForecastingDataset(
        raw_metric=raw_metric,
        windows=canonical["train_windows"],
        mean=mean,
        std=std,
        history_steps=history_steps,
    )
    val_set = CanonicalPeMSForecastingDataset(
        raw_metric=raw_metric,
        windows=canonical["val_windows"],
        mean=mean,
        std=std,
        history_steps=history_steps,
    )
    test_set = CanonicalPeMSForecastingDataset(
        raw_metric=raw_metric,
        windows=canonical["test_windows"],
        mean=mean,
        std=std,
        history_steps=history_steps,
    )
    return train_set, val_set, test_set, int(raw_metric.shape[1]), horizon_steps


def load_csdi_config(args: argparse.Namespace, target_dim: int) -> Dict:
    csdi_repo = Path(args.csdi_repo).resolve()
    cfg_path = csdi_repo / args.csdi_config
    if not cfg_path.exists():
        raise FileNotFoundError(f"CSDI config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["train"]["epochs"] = int(args.epochs)
    cfg["train"]["batch_size"] = int(args.batch_size)
    cfg["train"]["lr"] = float(args.lr)
    cfg["train"]["itr_per_epoch"] = float(args.itr_per_epoch)
    cfg["diffusion"]["num_steps"] = int(args.diffusion_steps)
    cfg["model"]["num_sample_features"] = min(int(args.num_sample_features), int(target_dim))
    cfg["model"]["is_unconditional"] = int(cfg["model"].get("is_unconditional", 0))
    return cfg


def import_csdi(csdi_repo: str):
    repo = Path(csdi_repo).resolve()
    if not repo.exists():
        raise FileNotFoundError(f"CSDI repo not found: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from main_model import CSDI_Forecasting  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name == "linear_attention_transformer":
            raise ModuleNotFoundError(
                "Official CSDI requires `linear_attention_transformer`. Install it in the current "
                "environment with: `python -m pip install linear-attention-transformer einops` "
                "or use your preferred PyPI mirror."
            ) from exc
        raise

    return CSDI_Forecasting


def run_validation(model: torch.nn.Module, loader: DataLoader, max_batches: Optional[int]) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            loss = model(batch, is_train=0)
            total += float(loss.item())
            count += 1
            if max_batches is not None and batch_idx >= int(max_batches):
                break
    return total / max(count, 1)


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cfg: Mapping,
    save_dir: Path,
    valid_interval: int,
    max_train_batches: Optional[int],
) -> Path:
    optimizer = Adam(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=1e-6)
    epochs = int(train_cfg["epochs"])
    p1 = int(0.75 * epochs)
    p2 = int(0.9 * epochs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1)
    best_loss = float("inf")
    best_path = save_dir / "best.pth"
    last_path = save_dir / "last.pth"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        batches = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch, is_train=1)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            batches += 1
            if batch_idx >= float(train_cfg.get("itr_per_epoch", 1e8)):
                break
            if max_train_batches is not None and batch_idx >= int(max_train_batches):
                break
        scheduler.step()
        avg_train = train_loss / max(batches, 1)
        print(f"[epoch {epoch:03d}] train_loss={avg_train:.6f} time={time.time() - t0:.2f}s", flush=True)

        if epoch % int(valid_interval) == 0 or epoch == epochs:
            val_loss = run_validation(model, val_loader, max_batches=None)
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
    horizon_steps: int,
    nsample: int,
    out_npz: Path,
    max_eval_batches: Optional[int],
) -> None:
    model.eval()
    sample_chunks: List[np.ndarray] = []
    target_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader, start=1):
            samples, observed_data, target_mask, _, _ = model.evaluate(batch, int(nsample))
            # CSDI returns samples [B,S,K,L] and observed_data [B,K,L].
            samples_fut = samples[:, :, :, -horizon_steps:]
            target_fut = observed_data[:, :, -horizon_steps:]
            eval_mask = target_mask[:, :, -horizon_steps:]
            if float(eval_mask.min().item()) < 0.5:
                raise RuntimeError("Unexpected CSDI eval mask; future target positions are not fully selected.")
            samples_np = samples_fut.permute(1, 0, 3, 2).unsqueeze(-1).detach().cpu().numpy().astype(np.float32)
            target_np = target_fut.permute(0, 2, 1).unsqueeze(-1).detach().cpu().numpy().astype(np.float32)
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

    save_dir = Path(args.save_dir or f"outputs/prob_baselines/CSDI/{dataset_name}_run")
    out_npz = Path(args.out_npz or f"outputs/prob_baselines/CSDI/{dataset_name}_samples.npz")
    save_dir.mkdir(parents=True, exist_ok=True)

    train_set, val_set, test_set, target_dim, horizon_steps = build_datasets(args, config)
    train_loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=int(args.batch_size), shuffle=False, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=int(args.batch_size), shuffle=False, num_workers=0, drop_last=False)

    csdi_cfg = load_csdi_config(args, target_dim=target_dim)
    with (save_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "csdi_config": csdi_cfg, "target_dim": target_dim}, f, indent=2)

    CSDI_Forecasting = import_csdi(args.csdi_repo)
    model = CSDI_Forecasting(csdi_cfg, str(device), target_dim).to(device)

    if args.resume:
        ckpt_path = Path(args.resume)
        print(f"[load] {ckpt_path}", flush=True)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        best_path = ckpt_path
    else:
        best_path = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_cfg=csdi_cfg["train"],
            save_dir=save_dir,
            valid_interval=int(args.valid_interval),
            max_train_batches=args.max_train_batches,
        )
        print(f"[load best] {best_path}", flush=True)
        model.load_state_dict(torch.load(best_path, map_location=device))

    # Evaluation must use the full feature set, even if training sampled features.
    model.target_dim = target_dim
    export_samples(
        model=model,
        test_loader=test_loader,
        horizon_steps=int(horizon_steps),
        nsample=int(args.nsample),
        out_npz=out_npz,
        max_eval_batches=args.max_eval_batches,
    )


if __name__ == "__main__":
    main()
