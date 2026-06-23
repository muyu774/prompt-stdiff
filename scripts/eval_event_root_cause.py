"""Event-local mean-vs-dispersion root-cause diagnostics.

This script reuses event-local sample extraction and reports whether failures
are dominated by mean-level surprise or insufficient predictive width.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_event_subset import _load_model_event_arrays, _load_npz_event_arrays
from utils.config import load_config
from utils.device import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Event root-cause decomposition")
    p.add_argument("--config", required=True)
    p.add_argument("--events_csv", required=True, type=Path)
    p.add_argument("--method", default="model")
    p.add_argument("--setting", default="event_root_cause")
    p.add_argument("--out_json", required=True, type=Path)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pred_npz", type=Path)
    src.add_argument("--ckpt", type=Path)
    p.add_argument("--samples_key", default="auto")
    p.add_argument("--target_key", default="auto")
    p.add_argument("--sample_axis", default="auto", choices=("auto", "0", "1"))
    p.add_argument("--space", default="normalized", choices=("normalized", "original"))
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--allow_truncate", action="store_true")
    p.add_argument("--kind", default="all")
    p.add_argument("--keywords_contains", default="")
    p.add_argument("--max_events", type=int, default=None)
    p.add_argument("--dedupe_positions", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--gpu_id", type=int, default=None)
    p.add_argument("--num_eval_samples", type=int, default=None)
    p.add_argument("--sampler", choices=("ddpm", "ddim"), default=None)
    p.add_argument("--sampling_steps", type=int, default=None)
    p.add_argument("--metric_feature_index", type=int, default=None)
    p.add_argument("--lower_q", type=float, default=0.05)
    p.add_argument("--upper_q", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-6)
    return p.parse_args()


def summarize(values: np.ndarray) -> Mapping[str, float]:
    if values.size == 0:
        return {}
    qs = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
    out = {f"q{int(q*100):02d}": float(np.quantile(values, q)) for q in qs}
    out["mean"] = float(np.mean(values))
    out["std"] = float(np.std(values))
    return out


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

    # samples [S,M,F], target [M,F], already in original units from helper.
    y = target[..., 0].astype(np.float64)
    ens = samples[..., 0].astype(np.float64)
    qlo = np.quantile(ens, float(args.lower_q), axis=0)
    qhi = np.quantile(ens, float(args.upper_q), axis=0)
    center_mean = ens.mean(axis=0)
    center_mid = 0.5 * (qlo + qhi)
    half_width = np.maximum(0.5 * (qhi - qlo), float(args.eps))
    abs_mean_err = np.abs(y - center_mean)
    abs_mid_err = np.abs(y - center_mid)
    rho_mean = abs_mean_err / half_width
    rho_mid = abs_mid_err / half_width
    covered = (y >= qlo) & (y <= qhi)
    under = y < qlo
    over = y > qhi
    signed_mean_err = y - center_mean

    payload = {
        "dataset": str(config["dataset"]["name"]),
        "method": str(args.method),
        "setting": str(args.setting),
        "source": source,
        "events_csv": str(args.events_csv),
        "kind": str(args.kind),
        "candidate_events": int(len(event_df)),
        "used_events": int(len(used_events)),
        "event_points": int(y.shape[0]),
        "split_windows": int(n_windows),
        "interval": [float(args.lower_q), float(args.upper_q)],
        "coverage": float(np.mean(covered)),
        "under_rate": float(np.mean(under)),
        "over_rate": float(np.mean(over)),
        "mean_error": summarize(abs_mean_err),
        "signed_mean_error": summarize(signed_mean_err),
        "half_width": summarize(half_width),
        "rho_mean_abs_error_over_half_width": summarize(rho_mean),
        "rho_mid_abs_error_over_half_width": summarize(rho_mid),
        "horizon": {},
    }
    for h in sorted(set((h_idx + 1).tolist())):
        mask = (h_idx + 1) == h
        payload["horizon"][str(int(h))] = {
            "points": int(mask.sum()),
            "coverage": float(np.mean(covered[mask])),
            "mean_error_median": float(np.median(abs_mean_err[mask])),
            "half_width_median": float(np.median(half_width[mask])),
            "rho_median": float(np.median(rho_mean[mask])),
            "rho_q90": float(np.quantile(rho_mean[mask], 0.9)),
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
