"""Run official PriSTI on this repo's canonical PeMS forecasting windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Mapping, Optional, Tuple

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


class CanonicalPeMSPriSTIDataset(Dataset):
    """Canonical PeMS windows in PriSTI traffic batch format."""

    def __init__(
        self,
        raw_metric: np.ndarray,
        windows: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        history_steps: int,
        use_guide: bool,
    ) -> None:
        self.raw_metric = raw_metric.astype(np.float32)
        self.windows = windows.astype(np.int64)
        self.mean = mean.astype(np.float32).reshape(1, -1)
        self.std = np.where(std.astype(np.float32).reshape(1, -1) < 1e-6, 1.0, std.reshape(1, -1))
        self.history_steps = int(history_steps)
        self.use_guide = bool(use_guide)
        self.target_dim = int(raw_metric.shape[1])
        self.eval_length = int(self.windows[0, 2] - self.windows[0, 0])

        if self.mean.shape[1] not in (1, self.target_dim):
            raise ValueError(f"Scaler mean shape {self.mean.shape} incompatible with N={self.target_dim}")
        if self.std.shape[1] not in (1, self.target_dim):
            raise ValueError(f"Scaler std shape {self.std.shape} incompatible with N={self.target_dim}")

        self._torchcde = None
        if self.use_guide:
            try:
                import torchcde  # type: ignore

                self._torchcde = torchcde
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "PriSTI use_guide=true requires `torchcde`. Install with "
                    "`python -m pip install torchcde` or pass --no_use_guide."
                ) from exc

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        his_start, his_end, fut_end = [int(x) for x in self.windows[index]]
        seq = self.raw_metric[his_start:fut_end]
        seq = ((seq - self.mean) / self.std).astype(np.float32)
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

        observed_mask = np.ones_like(seq, dtype=np.float32)
        gt_mask = observed_mask.copy()
        gt_mask[self.history_steps :] = 0.0
        cond_mask = gt_mask.copy()

        out: Dict[str, torch.Tensor] = {
            "observed_data": torch.from_numpy(seq),
            "observed_mask": torch.from_numpy(observed_mask),
            "gt_mask": torch.from_numpy(gt_mask),
            "cond_mask": torch.from_numpy(cond_mask),
            "timepoints": torch.arange(seq.shape[0], dtype=torch.float32),
            "cut_length": torch.tensor(0, dtype=torch.long),
        }
        if self.use_guide:
            tmp_data = torch.from_numpy(seq).to(torch.float64)
            cond = torch.from_numpy(cond_mask).bool()
            itp_data = torch.where(cond, tmp_data, torch.tensor(float("nan"), dtype=torch.float64))
            coeffs = self._torchcde.linear_interpolation_coeffs(
                itp_data.permute(1, 0).unsqueeze(-1).to(torch.float32)
            ).squeeze(-1).permute(1, 0)
            coeffs = torch.nan_to_num(coeffs, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)
            out["coeffs"] = coeffs
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate PriSTI on canonical PeMS windows.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--canonical_npz", type=str, required=True)
    parser.add_argument("--pristi_repo", type=str, default="baselines/external_repos/PriSTI")
    parser.add_argument("--pristi_config", type=str, default="config/traffic.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gpu_id", type=int, default=None, choices=list(range(16)))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--valid_interval", type=int, default=10)
    parser.add_argument("--diffusion_steps", type=int, default=50)
    parser.add_argument("--nsample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric_feature_index", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--out_npz", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--no_use_guide", action="store_true")
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
    return adj


def build_datasets(args: argparse.Namespace, config: Mapping, use_guide: bool) -> Tuple[Dataset, Dataset, Dataset, int, int, int]:
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

    kwargs = {
        "raw_metric": raw_metric,
        "mean": mean,
        "std": std,
        "history_steps": history_steps,
        "use_guide": use_guide,
    }
    return (
        CanonicalPeMSPriSTIDataset(windows=canonical["train_windows"], **kwargs),
        CanonicalPeMSPriSTIDataset(windows=canonical["val_windows"], **kwargs),
        CanonicalPeMSPriSTIDataset(windows=canonical["test_windows"], **kwargs),
        int(raw_metric.shape[1]),
        history_steps,
        horizon_steps,
    )


def load_pristi_config(args: argparse.Namespace, use_guide: bool) -> Dict:
    cfg_path = Path(args.pristi_repo).resolve() / args.pristi_config
    if not cfg_path.exists():
        raise FileNotFoundError(f"PriSTI config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["train"]["epochs"] = int(args.epochs)
    cfg["train"]["batch_size"] = int(args.batch_size)
    cfg["train"]["lr"] = float(args.lr)
    cfg["train"]["valid_epoch_interval"] = int(args.valid_interval)
    cfg["diffusion"]["num_steps"] = int(args.diffusion_steps)
    cfg["diffusion"]["adj_file"] = "pems-bay"
    cfg["model"]["use_guide"] = bool(use_guide)
    cfg["model"]["is_unconditional"] = 0
    cfg["seed"] = int(args.seed)
    return cfg


def import_pristi(pristi_repo: str, adjacency: np.ndarray):
    repo = Path(pristi_repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import layers  # type: ignore
        import diff_models  # type: ignore
        from main_model import PriSTI_PemsBAY  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name == "torchcde":
            raise ModuleNotFoundError("PriSTI requires `torchcde`; install with `python -m pip install torchcde`.") from exc
        raise

    adj = adjacency.astype(np.float32)

    def _dataset_adj(thr: float = 0.1) -> np.ndarray:
        return adj

    layers.get_similarity_pemsbay = _dataset_adj
    diff_models.get_similarity_pemsbay = _dataset_adj
    return PriSTI_PemsBAY


def run_validation(model: torch.nn.Module, loader: DataLoader) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            loss = model(batch, is_train=0)
            total += float(loss.item())
            count += 1
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
    scheduler = None
    if bool(train_cfg.get("is_lr_decay", True)):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(0.75 * epochs), int(0.9 * epochs)],
            gamma=0.1,
        )
    best_path = save_dir / "best.pth"
    last_path = save_dir / "last.pth"
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        count = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch, is_train=1)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            count += 1
            if max_train_batches is not None and batch_idx >= int(max_train_batches):
                break
        if scheduler is not None:
            scheduler.step()
        avg = total / max(count, 1)
        print(f"[epoch {epoch:03d}] train_loss={avg:.6f} time={time.time() - t0:.2f}s", flush=True)
        if epoch % int(valid_interval) == 0 and epoch > epochs * 0.5:
            val_loss = run_validation(model, val_loader)
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
            samples_fut = samples[:, :, :, -horizon_steps:]
            target_fut = observed_data[:, :, -horizon_steps:]
            eval_mask = target_mask[:, :, -horizon_steps:]
            if float(eval_mask.min().item()) < 0.5:
                raise RuntimeError("Unexpected PriSTI eval mask; future positions are not fully selected.")
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
    use_guide = not bool(args.no_use_guide)

    save_dir = Path(args.save_dir or f"outputs/prob_baselines/PriSTI/{dataset_name}_run")
    out_npz = Path(args.out_npz or f"outputs/prob_baselines/PriSTI/{dataset_name}_samples.npz")
    save_dir.mkdir(parents=True, exist_ok=True)

    train_set, val_set, test_set, target_dim, history_steps, horizon_steps = build_datasets(args, config, use_guide=use_guide)
    adjacency = _load_adjacency(config, num_nodes=target_dim, identity_adj=bool(args.identity_adj))
    pristi_cfg = load_pristi_config(args, use_guide=use_guide)
    with (save_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "pristi_config": pristi_cfg,
                "target_dim": target_dim,
                "history_steps": history_steps,
                "horizon_steps": horizon_steps,
                "adjacency": "identity" if args.identity_adj else "config_or_identity",
            },
            f,
            indent=2,
        )

    train_loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=int(args.batch_size), shuffle=False, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=int(args.batch_size), shuffle=False, num_workers=0, drop_last=False)

    PriSTI_PemsBAY = import_pristi(args.pristi_repo, adjacency=adjacency)
    model = PriSTI_PemsBAY(pristi_cfg, str(device), target_dim=target_dim, seq_len=history_steps + horizon_steps).to(device)

    if args.resume:
        ckpt = Path(args.resume)
        print(f"[load] {ckpt}", flush=True)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        best_path = ckpt
    else:
        best_path = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_cfg=pristi_cfg["train"],
            save_dir=save_dir,
            valid_interval=int(args.valid_interval),
            max_train_batches=args.max_train_batches,
        )
        print(f"[load best] {best_path}", flush=True)
        model.load_state_dict(torch.load(best_path, map_location=device))

    export_samples(
        model=model,
        test_loader=test_loader,
        horizon_steps=horizon_steps,
        nsample=int(args.nsample),
        out_npz=out_npz,
        max_eval_batches=args.max_eval_batches,
    )


if __name__ == "__main__":
    main()
