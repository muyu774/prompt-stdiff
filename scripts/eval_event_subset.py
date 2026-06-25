"""Evaluate probabilistic forecasts on event-local traffic points.

The event CSV is expected to contain at least ``t_idx`` and ``node_index``.
For each event point, this script finds test forecast windows whose target
horizon contains that timestamp, then evaluates only the matching
``(window, horizon, node)`` positions. This keeps event performance from being
diluted by normal nodes in the same window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataio.scalers import StandardScaler
from dataio.traffic_dataset import build_dataloaders
from scripts.eval_probabilistic_npz import (
    SAMPLE_KEYS,
    TARGET_KEYS,
    _crps_ensemble_fast,
    _prepare_arrays,
    _select_key,
    _to_sample_first,
)
from scripts.run_experiment_and_record import _build_runtime
from trainers.evaluator import _inverse_with_scaler
from utils.checkpoint import load_checkpoint
from utils.config import deep_merge, load_config
from utils.device import get_device
from utils.metrics import compute_all_metrics


PROB_KEYS: Sequence[str] = (
    "crps",
    "nll",
    "winkler@90",
    "picp@90",
    "mpiw@90",
    "sharpness",
    "reliability@90",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate event-local probabilistic metrics.")
    parser.add_argument("--config", required=True, help="Dataset/model config.")
    parser.add_argument("--events_csv", required=True, type=Path, help="CSV with t_idx and node_index columns.")
    parser.add_argument("--method", default="model", help="Method label for JSON output.")
    parser.add_argument("--setting", default="event_subset", help="Setting label.")
    parser.add_argument("--out_json", type=Path, default=None, help="Optional JSON output path.")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pred_npz", type=Path, help="NPZ forecasts [S,B,H,N,F] or [B,S,H,N,F].")
    source.add_argument("--ckpt", type=Path, help="Prompt-STDiff checkpoint to sample.")

    parser.add_argument("--samples_key", default="auto")
    parser.add_argument("--target_key", default="auto")
    parser.add_argument("--sample_axis", default="auto", choices=("auto", "0", "1"))
    parser.add_argument("--space", default="normalized", choices=("normalized", "original"))
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--allow_truncate", action="store_true")

    parser.add_argument("--kind", default="all", help="Filter CSV kind column, e.g. drop/spike/all.")
    parser.add_argument("--keywords_contains", default="", help="Case-insensitive substring filter on keywords.")
    parser.add_argument("--max_events", type=int, default=None, help="Limit number of event rows after filtering.")
    parser.add_argument("--dedupe_positions", action="store_true", help="Deduplicate window/horizon/node positions.")

    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=4096, help="Event-point batch size for metric chunks.")
    parser.add_argument("--num_eval_samples", type=int, default=None, help="Samples for checkpoint evaluation.")
    parser.add_argument("--sampler", choices=("ddpm", "ddim"), default=None)
    parser.add_argument("--sampling_steps", type=int, default=None)
    parser.add_argument("--metric_feature_index", type=int, default=None)
    parser.add_argument("--eval_horizons", type=int, nargs="*", default=None)
    return parser.parse_args()


def _event_rows(events_csv: Path, kind: str, keywords_contains: str, max_events: Optional[int]) -> pd.DataFrame:
    if not events_csv.exists():
        raise FileNotFoundError(f"events_csv not found: {events_csv}")
    df = pd.read_csv(events_csv)
    missing = [c for c in ("t_idx", "node_index") if c not in df.columns]
    if missing:
        raise ValueError(f"events_csv missing required columns: {missing}; columns={list(df.columns)}")
    out = df.copy()
    if kind and kind.lower() != "all":
        if "kind" not in out.columns:
            raise ValueError("--kind was provided but CSV has no kind column.")
        out = out[out["kind"].astype(str).str.lower() == kind.lower()]
    if keywords_contains:
        if "keywords" not in out.columns:
            raise ValueError("--keywords_contains was provided but CSV has no keywords column.")
        needle = keywords_contains.lower()
        out = out[out["keywords"].fillna("").astype(str).str.lower().str.contains(needle, regex=False)]
    out = out.sort_values(["t_idx", "node_index"], kind="stable")
    if max_events is not None:
        out = out.head(int(max_events))
    return out.reset_index(drop=True)


def _loader_for_split(config: Mapping, split: str):
    artifacts = build_dataloaders(dict(config))
    loader = {
        "train": artifacts.train_loader,
        "val": artifacts.val_loader,
        "test": artifacts.test_loader,
    }[split]
    return artifacts, loader


def _event_positions(dataset, event_df: pd.DataFrame, n_windows: int, dedupe: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    if int(n_windows) > len(dataset):
        raise ValueError(f"n_windows={n_windows} exceeds dataset length={len(dataset)}")
    event_records = event_df[["t_idx", "node_index"]].astype({"t_idx": int, "node_index": int}).to_records(index=False)
    positions: List[Tuple[int, int, int, int, int]] = []
    for event_row, (t_idx, node_idx) in enumerate(event_records):
        for win_idx in range(int(n_windows)):
            _start, his_end, fut_end = dataset.indices[win_idx]
            cutoff = int(dataset.base_step) + int(his_end)
            fut_global = int(dataset.base_step) + int(fut_end)
            if cutoff <= int(t_idx) < fut_global:
                h_idx = int(t_idx) - cutoff
                positions.append((win_idx, h_idx, int(node_idx), int(event_row), int(t_idx)))
    if not positions:
        empty = np.asarray([], dtype=np.int64)
        return empty, empty, empty, event_df.iloc[[]].copy()

    arr = np.asarray(positions, dtype=np.int64)
    if dedupe:
        _, keep = np.unique(arr[:, :3], axis=0, return_index=True)
        keep = np.sort(keep)
        arr = arr[keep]
    used_events = event_df.iloc[np.unique(arr[:, 3])].copy()
    return arr[:, 0], arr[:, 1], arr[:, 2], used_events


def _feature_params(scaler: StandardScaler, feature_dim: int, metric_feature_index: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    mean = scaler.mean
    std = scaler.std
    if feature_dim == 1 and mean.shape[-1] > 1:
        idx = 0 if metric_feature_index is None else int(metric_feature_index)
        mean = mean[..., idx : idx + 1]
        std = std[..., idx : idx + 1]
    elif feature_dim != mean.shape[-1]:
        raise ValueError(f"feature_dim={feature_dim} incompatible with scaler feature_dim={mean.shape[-1]}")
    return mean.astype(np.float32), std.astype(np.float32)


def _inverse_points(
    arr: np.ndarray,
    scaler: Optional[StandardScaler],
    node_idx: np.ndarray,
    metric_feature_index: Optional[int],
) -> np.ndarray:
    if scaler is None:
        return arr.astype(np.float32, copy=False)
    mean, std = _feature_params(scaler, feature_dim=int(arr.shape[-1]), metric_feature_index=metric_feature_index)
    point_mean = mean[:, node_idx, :]
    point_std = std[:, node_idx, :]
    if arr.ndim == 3:
        return (arr * point_std).astype(np.float32) + point_mean.astype(np.float32)
    return (arr * point_std[0]).astype(np.float32) + point_mean[0].astype(np.float32)


def _reshape_event_arrays(samples: np.ndarray, target: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    # samples [S,M,F], target [M,F] -> [S,M,1,1,F], [M,1,1,F]
    sample_t = torch.as_tensor(samples[:, :, None, None, :], dtype=torch.float32)
    target_t = torch.as_tensor(target[:, None, None, :], dtype=torch.float32)
    return sample_t, target_t


def _metric_chunk(samples: torch.Tensor, target: torch.Tensor, device: torch.device) -> Dict[str, float]:
    samples = samples.to(device)
    target = target.to(device)
    out = compute_all_metrics(samples, target)
    out["crps"] = float(_crps_ensemble_fast(samples, target).item())
    return out


def _weighted_update(acc: Dict[str, float], metrics: Mapping[str, float], weight: int) -> None:
    acc["weight"] = float(acc.get("weight", 0.0) + int(weight))
    for k, v in metrics.items():
        acc[k] = float(acc.get(k, 0.0) + float(v) * int(weight))


def _weighted_finalize(acc: Mapping[str, float]) -> Dict[str, float]:
    w = float(acc.get("weight", 0.0))
    return {k: float(v) / w for k, v in acc.items() if k != "weight"} if w > 0 else {}


def _evaluate_event_arrays(
    samples: np.ndarray,
    target: np.ndarray,
    horizons: Iterable[int],
    device: torch.device,
    eval_batch_size: int,
    h_idx: np.ndarray,
) -> Dict[str, float]:
    metrics_acc: Dict[str, float] = {}
    horizon_acc: Dict[int, Dict[str, float]] = {int(h): {} for h in horizons}
    n = int(target.shape[0])
    for start in range(0, n, max(1, int(eval_batch_size))):
        end = min(start + int(eval_batch_size), n)
        sample_t, target_t = _reshape_event_arrays(samples[:, start:end], target[start:end])
        chunk_metrics = _metric_chunk(sample_t, target_t, device=device)
        _weighted_update(metrics_acc, chunk_metrics, weight=end - start)

        h_local = h_idx[start:end] + 1
        for h in horizon_acc:
            keep = np.flatnonzero(h_local == int(h))
            if keep.size == 0:
                continue
            h_sample_t, h_target_t = _reshape_event_arrays(
                samples[:, start:end][:, keep],
                target[start:end][keep],
            )
            _weighted_update(horizon_acc[h], _metric_chunk(h_sample_t, h_target_t, device=device), weight=int(keep.size))
    out = _weighted_finalize(metrics_acc)
    for h, acc in horizon_acc.items():
        for k, v in _weighted_finalize(acc).items():
            out[f"{k}@{h}"] = v
    return out


def _load_npz_event_arrays(args: argparse.Namespace, config: Mapping, device: torch.device):
    bundle = np.load(args.pred_npz, mmap_mode="r")
    sample_key = _select_key(bundle, args.samples_key, SAMPLE_KEYS, label="samples")
    if sample_key is None:
        raise KeyError(f"No sample key found in {args.pred_npz}; available={bundle.files}")
    target_key = _select_key(bundle, args.target_key, TARGET_KEYS, label="target")
    artifacts, loader = _loader_for_split(config, args.split)
    dataset = loader.dataset

    if target_key is not None:
        target = bundle[target_key].astype(np.float32, copy=False)
    else:
        target = np.concatenate([b["x_fut"].numpy().astype(np.float32) for b in loader], axis=0)
    samples = _to_sample_first(bundle[sample_key], sample_axis=str(args.sample_axis), target_windows=int(target.shape[0]))
    if samples.shape[1] != target.shape[0]:
        if not args.allow_truncate:
            raise ValueError(
                f"Forecast windows ({samples.shape[1]}) != target windows ({target.shape[0]}). "
                "Use --allow_truncate to compare the shared prefix."
            )
        n = min(int(samples.shape[1]), int(target.shape[0]))
        samples = samples[:, :n]
        target = target[:n]

    event_df = _event_rows(args.events_csv, kind=args.kind, keywords_contains=args.keywords_contains, max_events=args.max_events)
    win_idx, h_idx, node_idx, used_events = _event_positions(dataset, event_df, n_windows=int(target.shape[0]), dedupe=bool(args.dedupe_positions))
    if win_idx.size == 0:
        raise RuntimeError("No event points fall inside the selected split/prediction windows.")

    point_samples = samples[:, win_idx, h_idx, node_idx, :]
    point_target = target[win_idx, h_idx, node_idx, :]
    if args.space == "normalized":
        point_samples = _inverse_points(point_samples, artifacts.scaler, node_idx=node_idx, metric_feature_index=args.metric_feature_index)
        point_target = _inverse_points(point_target, artifacts.scaler, node_idx=node_idx, metric_feature_index=args.metric_feature_index)

    return point_samples, point_target, h_idx, event_df, used_events, int(target.shape[0])


@torch.no_grad()
def _load_model_event_arrays(args: argparse.Namespace, config: Mapping, device: torch.device):
    overrides: Dict = {"diffusion": {}}
    if args.sampler is not None:
        overrides["diffusion"]["sampler"] = str(args.sampler)
    if args.sampling_steps is not None:
        overrides["diffusion"]["sampling_steps"] = int(args.sampling_steps)
    if args.num_eval_samples is not None:
        overrides.setdefault("train", {})["num_eval_samples"] = int(args.num_eval_samples)
    config = deep_merge(dict(config), overrides)

    artifacts, model, sampler, a_phy, a_sem, z_sem, dynamic_bank = _build_runtime(config, device)
    from models.mean_predictor import get_mean_predictor_config

    load_checkpoint(
        args.ckpt,
        model=model,
        optimizer=None,
        map_location=str(device),
        strict=not bool(dict(get_mean_predictor_config(config)).get("type")),
    )
    model.eval()
    loader = {
        "train": artifacts.train_loader,
        "val": artifacts.val_loader,
        "test": artifacts.test_loader,
    }[args.split]
    event_df = _event_rows(args.events_csv, kind=args.kind, keywords_contains=args.keywords_contains, max_events=args.max_events)
    win_idx, h_idx, node_idx, used_events = _event_positions(loader.dataset, event_df, n_windows=len(loader.dataset), dedupe=bool(args.dedupe_positions))
    if win_idx.size == 0:
        raise RuntimeError("No event points fall inside the selected split windows.")

    by_window: Dict[int, List[int]] = {}
    for pos_i, w in enumerate(win_idx.tolist()):
        by_window.setdefault(int(w), []).append(pos_i)

    num_samples = int(args.num_eval_samples or config["train"].get("num_eval_samples", 20))
    out_samples: Optional[np.ndarray] = None
    out_target: Optional[np.ndarray] = None
    cursor = 0
    for batch in loader:
        b = int(batch["x_fut"].shape[0])
        batch_window_ids = range(cursor, cursor + b)
        needed = [(local_i, by_window[global_i]) for local_i, global_i in enumerate(batch_window_ids) if global_i in by_window]
        cursor += b
        if not needed:
            continue

        x_his = batch["x_his"].to(device=device, dtype=torch.float32)
        x_fut = batch["x_fut"].to(device=device, dtype=torch.float32)
        cutoff_step = batch["cutoff_step"].to(device=device, dtype=torch.long)
        bb, hh, nn, ff = x_fut.shape
        if dynamic_bank is not None:
            z_sem_batch = dynamic_bank.compose(static_z_sem=z_sem, cutoff_steps=cutoff_step, num_nodes=nn, device=device)
        else:
            z_sem_batch = z_sem
        cond = {"x_his": x_his, "a_phy": a_phy, "a_sem": a_sem, "z_sem": z_sem_batch}
        ensemble = sampler.sample_ensemble(
            model_fn=model.model_fn,
            shape=(bb, hh, nn, ff),
            cond=cond,
            device=device,
            num_samples=max(num_samples, 1),
        )
        if bool(getattr(model, "use_mean_head", False)):
            mean_pred = model.predict_mean(
                x_his=x_his,
                a_phy=a_phy,
                a_sem=a_sem,
                z_sem=z_sem_batch,
                batch={"x_his": x_his, "x_fut": x_fut, "cutoff_step": cutoff_step},
            )
            residual = model.unstandardize_residual(ensemble)
            residual = model.calibrate_residual_samples(residual, x_his=x_his, z_sem=z_sem_batch)
            ensemble = residual + mean_pred.unsqueeze(0)
            ensemble = model.apply_mean_correction(
                ensemble,
                x_his=x_his,
                z_sem=z_sem_batch,
                a_phy=a_phy,
            )
        if bool(config.get("model", {}).get("predict_residual", False)) and not bool(getattr(model, "uses_absolute_mean_predictor", False)):
            baseline = x_his[:, -1:, :, :].expand(-1, hh, -1, -1)
            ensemble = ensemble + baseline.unsqueeze(0)

        ensemble_inv = _inverse_with_scaler(ensemble, artifacts.scaler).detach().cpu().numpy().astype(np.float32)
        target_inv = _inverse_with_scaler(x_fut, artifacts.scaler).detach().cpu().numpy().astype(np.float32)
        if out_samples is None:
            out_samples = np.empty((num_samples, len(win_idx), ff), dtype=np.float32)
            out_target = np.empty((len(win_idx), ff), dtype=np.float32)
        assert out_target is not None
        for local_i, pos_ids in needed:
            for pos_i in pos_ids:
                out_samples[:, pos_i, :] = ensemble_inv[:, local_i, h_idx[pos_i], node_idx[pos_i], :]
                out_target[pos_i, :] = target_inv[local_i, h_idx[pos_i], node_idx[pos_i], :]

    if out_samples is None or out_target is None:
        raise RuntimeError("No event batches were sampled.")
    return out_samples, out_target, h_idx, event_df, used_events, len(loader.dataset)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.metric_feature_index is None:
        idx = config.get("train", {}).get("metric_feature_index", None)
        args.metric_feature_index = None if idx is None else int(idx)

    device_arg = args.device if args.gpu_id is None else f"cuda:{int(args.gpu_id)}"
    device = get_device(device_arg)
    if args.pred_npz is not None:
        samples, target, h_idx, event_df, used_events, n_windows = _load_npz_event_arrays(args, config=config, device=device)
        source = str(args.pred_npz)
    else:
        samples, target, h_idx, event_df, used_events, n_windows = _load_model_event_arrays(args, config=config, device=device)
        source = str(args.ckpt)

    eval_horizons = args.eval_horizons or [int(x) for x in config["train"].get("eval_horizons", [3, 6, 12])]
    metrics = _evaluate_event_arrays(
        samples=samples,
        target=target,
        horizons=eval_horizons,
        device=device,
        eval_batch_size=int(args.eval_batch_size),
        h_idx=h_idx,
    )
    result = {
        "dataset": str(config["dataset"]["name"]),
        "method": str(args.method),
        "setting": str(args.setting),
        "source": source,
        "events_csv": str(args.events_csv),
        "kind": str(args.kind),
        "keywords_contains": str(args.keywords_contains),
        "candidate_events": int(len(event_df)),
        "used_events": int(len(used_events)),
        "event_points": int(target.shape[0]),
        "split_windows": int(n_windows),
        "samples_shape": [int(x) for x in samples.shape],
        "target_shape": [int(x) for x in target.shape],
        "metrics": metrics,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n")


if __name__ == "__main__":
    main()
