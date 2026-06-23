"""Evaluate probabilistic baseline forecasts stored in a unified NPZ file.

Expected forecast shape is [S, B, H, N, F] by default, where S is the
ensemble/sample dimension. The script can also read [B, S, H, N, F] via
``--sample_axis 1`` or auto-detect common layouts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataio.scalers import StandardScaler
from dataio.traffic_dataset import build_dataloaders
from utils.config import load_config
from utils.device import get_device
from utils.metrics import compute_all_metrics
from utils.result_writer import ExperimentResult, write_experiment_results


SAMPLE_KEYS: Sequence[str] = (
    "samples",
    "pred_samples",
    "preds",
    "forecasts",
    "prediction",
)
TARGET_KEYS: Sequence[str] = (
    "target",
    "targets",
    "y",
    "x_fut",
)
PROB_KEYS: Sequence[str] = (
    "crps",
    "nll",
    "winkler@90",
    "picp@90",
    "mpiw@90",
    "sharpness",
    "reliability@10",
    "reliability@20",
    "reliability@30",
    "reliability@40",
    "reliability@50",
    "reliability@60",
    "reliability@70",
    "reliability@80",
    "reliability@90",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate external probabilistic traffic baselines from NPZ forecasts "
            "with the same split/scaler/metrics as Prompt-STDiff."
        )
    )
    parser.add_argument("--pred_npz", type=str, required=True, help="NPZ containing ensemble forecasts.")
    parser.add_argument("--config", type=str, required=True, help="Dataset config used for target/scaler.")
    parser.add_argument("--method", type=str, required=True, help="Baseline method name, e.g. CSDI.")
    parser.add_argument("--setting", type=str, default="probabilistic", help="Result setting label.")
    parser.add_argument("--implementation", type=str, default="external", help="Implementation label.")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional external checkpoint path.")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--space",
        type=str,
        default="normalized",
        choices=("normalized", "original"),
        help="Whether predictions in NPZ are in normalized or original traffic-value space.",
    )
    parser.add_argument("--samples_key", type=str, default="auto", help="NPZ key for samples.")
    parser.add_argument("--target_key", type=str, default="auto", help="Optional NPZ key for target.")
    parser.add_argument(
        "--sample_axis",
        type=str,
        default="auto",
        choices=("auto", "0", "1"),
        help="Sample axis: 0 for [S,B,H,N,F], 1 for [B,S,H,N,F].",
    )
    parser.add_argument("--metric_feature_index", type=int, default=None)
    parser.add_argument("--eval_horizons", type=int, nargs="*", default=None)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--allow_truncate", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gpu_id", type=int, default=None, choices=list(range(16)))
    parser.add_argument("--csv", type=Path, default=Path("outputs/results.csv"))
    parser.add_argument("--md", type=Path, default=Path("RESULTS.md"))
    parser.add_argument("--title", type=str, default="Probabilistic Baseline Results")
    parser.add_argument(
        "--crps_mae_warn_ratio",
        type=float,
        default=1.25,
        help=(
            "Warn when CRPS is much larger than point MAE. For sane ensemble "
            "forecasts these are usually on the same scale; a large gap often "
            "means normalized/original units are mixed."
        ),
    )
    parser.add_argument(
        "--interval_mae_warn_ratio",
        type=float,
        default=8.0,
        help="Warn when MPIW@90 is implausibly wide relative to MAE.",
    )
    parser.add_argument(
        "--fail_on_sanity_warning",
        action="store_true",
        help="Turn probabilistic metric sanity warnings into a non-zero exit.",
    )
    return parser.parse_args()


def _select_key(bundle: np.lib.npyio.NpzFile, requested: str, candidates: Sequence[str], label: str) -> Optional[str]:
    if requested != "auto":
        if requested not in bundle:
            raise KeyError(f"{label} key '{requested}' not found in {bundle.files}")
        return requested
    for key in candidates:
        if key in bundle:
            return key
    return None


def _to_sample_first(samples: np.ndarray, sample_axis: str, target_windows: Optional[int] = None) -> np.ndarray:
    if samples.ndim != 5:
        raise ValueError(f"Expected samples with shape [S,B,H,N,F] or [B,S,H,N,F], got {samples.shape}")
    if sample_axis == "0":
        return samples
    if sample_axis == "1":
        return np.moveaxis(samples, 1, 0)

    if target_windows is not None:
        if samples.shape[1] == int(target_windows):
            return samples
        if samples.shape[0] == int(target_windows):
            return np.moveaxis(samples, 1, 0)

    # Heuristic: the sample dimension is usually much smaller than number of windows.
    if samples.shape[0] <= samples.shape[1]:
        return samples
    if samples.shape[1] < samples.shape[0]:
        return np.moveaxis(samples, 1, 0)
    return samples


def _collect_targets(config: Mapping, split: str, max_windows: Optional[int]) -> Tuple[np.ndarray, Optional[StandardScaler]]:
    artifacts = build_dataloaders(dict(config))
    loader = {
        "train": artifacts.train_loader,
        "val": artifacts.val_loader,
        "test": artifacts.test_loader,
    }[split]
    chunks: List[np.ndarray] = []
    seen = 0
    for batch in loader:
        y = batch["x_fut"].detach().cpu().numpy().astype(np.float32)
        if max_windows is not None:
            remain = int(max_windows) - seen
            if remain <= 0:
                break
            y = y[:remain]
        chunks.append(y)
        seen += int(y.shape[0])
        if max_windows is not None and seen >= int(max_windows):
            break
    if not chunks:
        raise RuntimeError(f"No target windows collected for split={split}")
    return np.concatenate(chunks, axis=0), artifacts.scaler


def _feature_params(
    scaler: StandardScaler,
    feature_dim: int,
    metric_feature_index: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    mean = scaler.mean
    std = scaler.std
    if feature_dim == 1 and mean.shape[-1] > 1:
        idx = 0 if metric_feature_index is None else int(metric_feature_index)
        mean = mean[..., idx : idx + 1]
        std = std[..., idx : idx + 1]
    elif feature_dim != mean.shape[-1]:
        raise ValueError(
            f"Cannot inverse transform feature_dim={feature_dim} with scaler feature_dim={mean.shape[-1]}. "
            "Pass --metric_feature_index when evaluating a single metric channel."
        )
    return mean.astype(np.float32), std.astype(np.float32)


def _inverse_if_needed(
    arr: np.ndarray,
    scaler: Optional[StandardScaler],
    metric_feature_index: Optional[int],
) -> np.ndarray:
    if scaler is None:
        return arr.astype(np.float32, copy=False)
    mean, std = _feature_params(scaler, feature_dim=int(arr.shape[-1]), metric_feature_index=metric_feature_index)
    return (arr * std + mean).astype(np.float32, copy=False)


def _select_metric_feature(arr: np.ndarray, metric_feature_index: Optional[int]) -> np.ndarray:
    if metric_feature_index is None or arr.shape[-1] == 1:
        return arr
    idx = int(metric_feature_index)
    return arr[..., idx : idx + 1]


def _prepare_arrays(
    samples: np.ndarray,
    target: np.ndarray,
    scaler: Optional[StandardScaler],
    space: str,
    metric_feature_index: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    if space == "normalized":
        samples = _inverse_if_needed(samples, scaler=scaler, metric_feature_index=metric_feature_index)
        target = _inverse_if_needed(target, scaler=scaler, metric_feature_index=metric_feature_index)
    else:
        target = target.astype(np.float32, copy=False)
        samples = samples.astype(np.float32, copy=False)

    samples = _select_metric_feature(samples, metric_feature_index=metric_feature_index)
    target = _select_metric_feature(target, metric_feature_index=metric_feature_index)
    return samples, target


def _crps_ensemble_fast(pred_samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Memory-safe CRPS using sorted ensemble samples instead of SxS pairwise diffs."""
    if pred_samples.dim() != 5:
        raise ValueError(f"Expected pred_samples [S,B,H,N,F], got {pred_samples.shape}")
    s = int(pred_samples.shape[0])
    term1 = torch.mean(torch.abs(pred_samples - target.unsqueeze(0)))
    if s <= 1:
        return term1
    sorted_samples, _ = torch.sort(pred_samples, dim=0)
    weights = (2 * torch.arange(1, s + 1, device=pred_samples.device, dtype=pred_samples.dtype) - s - 1)
    view_shape = (s,) + (1,) * (sorted_samples.dim() - 1)
    # This equals 0.5 * E|X - X'| for the empirical ensemble.
    term2 = torch.mean(torch.sum(sorted_samples * weights.view(view_shape), dim=0) / float(s * s))
    return term1 - term2


def _new_bucket() -> MutableMapping[str, float]:
    return {"weight": 0.0}


def _accumulate(bucket: MutableMapping[str, float], metrics: Mapping[str, float], weight: int) -> None:
    bucket["weight"] = float(bucket.get("weight", 0.0) + weight)
    for key, value in metrics.items():
        bucket[key] = float(bucket.get(key, 0.0) + float(value) * weight)


def _finalize(bucket: Mapping[str, float]) -> Dict[str, float]:
    weight = float(bucket.get("weight", 0.0))
    if weight <= 0:
        return {}
    return {key: float(value) / weight for key, value in bucket.items() if key != "weight"}


def _prob_metrics_chunk(samples: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    metrics = compute_all_metrics(samples, target)
    metrics["crps"] = float(_crps_ensemble_fast(samples, target).item())
    return {key: metrics[key] for key in PROB_KEYS if key in metrics}


def evaluate_npz(
    samples_np: np.ndarray,
    target_np: np.ndarray,
    device: torch.device,
    eval_batch_size: int,
    eval_horizons: Iterable[int],
) -> Dict[str, float]:
    if samples_np.ndim != 5 or target_np.ndim != 4:
        raise ValueError(f"Expected samples [S,B,H,N,F] and target [B,H,N,F], got {samples_np.shape}, {target_np.shape}")
    if samples_np.shape[1:] != target_np.shape:
        raise ValueError(f"Samples/target shape mismatch: {samples_np.shape[1:]} vs {target_np.shape}")

    n_windows = int(target_np.shape[0])
    eval_batch_size = max(1, int(eval_batch_size))
    overall_bucket = _new_bucket()
    horizon_buckets = {int(h): _new_bucket() for h in eval_horizons}
    point_chunks: List[torch.Tensor] = []
    target_chunks: List[torch.Tensor] = []

    for start in range(0, n_windows, eval_batch_size):
        end = min(start + eval_batch_size, n_windows)
        samples = torch.as_tensor(samples_np[:, start:end], dtype=torch.float32, device=device)
        target = torch.as_tensor(target_np[start:end], dtype=torch.float32, device=device)
        weight = int(target.numel())

        _accumulate(overall_bucket, _prob_metrics_chunk(samples, target), weight=weight)
        point_chunks.append(samples.mean(dim=0).detach().cpu())
        target_chunks.append(target.detach().cpu())

        for h in horizon_buckets:
            if h < 1 or h > target.shape[1]:
                continue
            h_samples = samples[:, :, h - 1 : h]
            h_target = target[:, h - 1 : h]
            _accumulate(horizon_buckets[h], _prob_metrics_chunk(h_samples, h_target), weight=int(h_target.numel()))

    point_pred = torch.cat(point_chunks, dim=0)
    target_all = torch.cat(target_chunks, dim=0)
    metrics = compute_all_metrics(point_pred, target_all)
    metrics.update(_finalize(overall_bucket))

    for h, bucket in horizon_buckets.items():
        if h < 1 or h > target_all.shape[1]:
            continue
        point_h = compute_all_metrics(point_pred[:, h - 1 : h], target_all[:, h - 1 : h])
        prob_h = _finalize(bucket)
        for key, value in point_h.items():
            metrics[f"{key}@{h}"] = float(value)
        for key, value in prob_h.items():
            metrics[f"{key}@{h}"] = float(value)
    return metrics


def _rows_from_metrics(
    metrics: Mapping[str, float],
    config: Mapping,
    args: argparse.Namespace,
    samples_shape: Tuple[int, ...],
    target_shape: Tuple[int, ...],
    sample_key: str,
    target_key: Optional[str],
) -> List[ExperimentResult]:
    eval_horizons = args.eval_horizons or [int(x) for x in config["train"].get("eval_horizons", [3, 6, 12])]
    rows: List[ExperimentResult] = []
    base_settings = {
        "pred_npz": str(args.pred_npz),
        "sample_key": sample_key,
        "target_key": target_key or "config_dataloader",
        "samples_shape": list(samples_shape),
        "target_shape": list(target_shape),
        "space": args.space,
        "split": args.split,
        "eval_batch_size": int(args.eval_batch_size),
        "max_windows": args.max_windows,
        "sample_axis": args.sample_axis,
        "metric_feature_index": args.metric_feature_index,
    }
    for h in eval_horizons:
        if f"mae@{h}" not in metrics:
            continue
        row_settings = dict(base_settings)
        for key in PROB_KEYS:
            metric_key = f"{key}@{h}"
            if metric_key in metrics:
                row_settings[key] = float(metrics[metric_key])
        rows.append(
            ExperimentResult(
                dataset=str(config["dataset"]["name"]),
                method=args.method,
                setting=args.setting,
                horizon=int(h),
                mae=float(metrics[f"mae@{h}"]),
                rmse=float(metrics[f"rmse@{h}"]),
                crps=float(metrics[f"crps@{h}"]) if f"crps@{h}" in metrics else None,
                seed=int(config["train"].get("seed", 42)),
                config=args.config,
                implementation=args.implementation,
                checkpoint=str(args.checkpoint),
                settings_json=json.dumps(row_settings, sort_keys=True),
                notes=(
                    f"nll={row_settings.get('nll', float('nan'))}; "
                    f"picp@90={row_settings.get('picp@90', float('nan'))}; "
                    f"mpiw@90={row_settings.get('mpiw@90', float('nan'))}"
                ),
            )
        )
    return rows


def _probability_sanity_warnings(metrics: Mapping[str, float], args: argparse.Namespace) -> List[str]:
    warnings: List[str] = []
    mae = float(metrics.get("mae", float("nan")))
    crps = float(metrics.get("crps", float("nan")))
    mpiw90 = float(metrics.get("mpiw@90", float("nan")))
    picp90 = float(metrics.get("picp@90", float("nan")))

    if np.isfinite(mae) and mae > 0 and np.isfinite(crps):
        ratio = crps / mae
        if ratio > float(args.crps_mae_warn_ratio):
            warnings.append(
                "CRPS/MAE sanity warning: "
                f"crps={crps:.6g}, mae={mae:.6g}, ratio={ratio:.3f} "
                f"> {float(args.crps_mae_warn_ratio):.3f}. "
                "This often indicates a unit mismatch, wrong inverse_transform, "
                "or horizon/sample reduction bug."
            )
    if np.isfinite(mae) and mae > 0 and np.isfinite(mpiw90):
        ratio = mpiw90 / mae
        if ratio > float(args.interval_mae_warn_ratio):
            warnings.append(
                "MPIW/MAE sanity warning: "
                f"mpiw@90={mpiw90:.6g}, mae={mae:.6g}, ratio={ratio:.3f} "
                f"> {float(args.interval_mae_warn_ratio):.3f}. "
                "Intervals may be on the wrong scale or over-dispersed."
            )
    if np.isfinite(picp90) and (picp90 < 0.0 or picp90 > 1.0):
        warnings.append(f"PICP@90 sanity warning: expected [0,1], got {picp90:.6g}.")

    for key, value in metrics.items():
        if not np.isfinite(float(value)):
            warnings.append(f"Non-finite metric warning: {key}={value}.")
    return warnings


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.metric_feature_index is None:
        metric_feature_index = config.get("train", {}).get("metric_feature_index", None)
        args.metric_feature_index = None if metric_feature_index is None else int(metric_feature_index)

    device_arg = args.device if args.gpu_id is None else f"cuda:{int(args.gpu_id)}"
    device = get_device(device_arg)

    pred_path = Path(args.pred_npz)
    bundle = np.load(pred_path, mmap_mode="r")
    sample_key = _select_key(bundle, args.samples_key, SAMPLE_KEYS, label="samples")
    if sample_key is None:
        raise KeyError(f"No sample key found in {pred_path}. Tried {SAMPLE_KEYS}; available={bundle.files}")
    target_key = _select_key(bundle, args.target_key, TARGET_KEYS, label="target")

    target: np.ndarray
    scaler: Optional[StandardScaler]
    if target_key is None:
        target, scaler = _collect_targets(config, split=args.split, max_windows=args.max_windows)
    else:
        target = bundle[target_key].astype(np.float32, copy=False)
        if args.space == "normalized":
            _, scaler = _collect_targets(config, split=args.split, max_windows=1)
        else:
            scaler = None

    samples = _to_sample_first(
        bundle[sample_key],
        sample_axis=str(args.sample_axis),
        target_windows=int(target.shape[0]),
    )

    if args.max_windows is not None:
        samples = samples[:, : int(args.max_windows)]
        target = target[: int(args.max_windows)]

    if samples.shape[1] != target.shape[0]:
        if not args.allow_truncate:
            raise ValueError(
                f"Forecast windows ({samples.shape[1]}) != target windows ({target.shape[0]}). "
                "Use --allow_truncate to compare the shared prefix."
            )
        n = min(int(samples.shape[1]), int(target.shape[0]))
        samples = samples[:, :n]
        target = target[:n]

    samples, target = _prepare_arrays(
        samples=samples,
        target=target,
        scaler=scaler,
        space=args.space,
        metric_feature_index=args.metric_feature_index,
    )

    eval_horizons = args.eval_horizons or [int(x) for x in config["train"].get("eval_horizons", [3, 6, 12])]
    metrics = evaluate_npz(
        samples_np=samples,
        target_np=target,
        device=device,
        eval_batch_size=int(args.eval_batch_size),
        eval_horizons=eval_horizons,
    )
    sanity_warnings = _probability_sanity_warnings(metrics, args)
    for warning in sanity_warnings:
        print(f"[sanity-warning] {warning}", file=sys.stderr, flush=True)
    if sanity_warnings and bool(args.fail_on_sanity_warning):
        raise RuntimeError("Probability metric sanity checks failed. See [sanity-warning] lines above.")
    rows = _rows_from_metrics(
        metrics=metrics,
        config=config,
        args=args,
        samples_shape=tuple(int(x) for x in samples.shape),
        target_shape=tuple(int(x) for x in target.shape),
        sample_key=sample_key,
        target_key=target_key,
    )
    write_experiment_results(rows, csv_path=args.csv, md_path=args.md, title=args.title)
    print(json.dumps({"metrics": metrics, "rows": len(rows), "sanity_warnings": sanity_warnings}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
