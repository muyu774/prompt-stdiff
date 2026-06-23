"""Evaluate heteroscedastic Gaussian residual samples on event-local positions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_event_subset import (
    _event_positions,
    _event_rows,
    _evaluate_event_arrays,
    _inverse_points,
    _loader_for_split,
)
from scripts.run_experiment_and_record import _build_runtime
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gaussian residual event-subset evaluation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--events_csv", required=True, type=Path)
    parser.add_argument("--out_json", required=True, type=Path)
    parser.add_argument("--method", default="GaussianResidual")
    parser.add_argument("--setting", default="event_subset")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--kind", default="all")
    parser.add_argument("--keywords_contains", default="")
    parser.add_argument("--max_events", type=int, default=None)
    parser.add_argument("--dedupe_positions", action="store_true")
    parser.add_argument("--num_eval_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=4096)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.num_eval_samples is not None:
        config.setdefault("train", {})["num_eval_samples"] = int(args.num_eval_samples)
    device = get_device(args.device if args.gpu_id is None else f"cuda:{int(args.gpu_id)}")

    artifacts, model, _sampler, a_phy, a_sem, z_sem, dynamic_bank = _build_runtime(config, device)
    load_checkpoint(Path(args.ckpt), model=model, optimizer=None, map_location=str(device), strict=False)
    model.eval()

    # Build event positions against the canonical dataset for this split.
    artifacts2, loader = _loader_for_split(config, args.split)
    dataset = loader.dataset
    event_df = _event_rows(args.events_csv, args.kind, args.keywords_contains, args.max_events)

    # Generate Gaussian-residual samples window-by-window, then select event points.
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed))
    num_samples = int(config.get("train", {}).get("num_eval_samples", 20))
    metric_feature_index = config.get("train", {}).get("metric_feature_index", None)

    sample_chunks = []
    target_chunks = []
    processed = 0
    for batch in loader:
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
        residual_std = torch.randn((num_samples, b, h, n, f), dtype=x_fut.dtype, device=device, generator=gen)
        residual = model.unstandardize_residual(residual_std)
        residual = model.calibrate_residual_samples(residual, x_his=x_his, z_sem=z_sem_batch)
        ensemble = residual + mean_pred.unsqueeze(0)
        # Keep normalized arrays; event helper inverse-transforms selected points.
        sample_chunks.append(ensemble.detach().cpu().numpy().astype(np.float32))
        target_chunks.append(x_fut.detach().cpu().numpy().astype(np.float32))
        processed += b

    samples = np.concatenate(sample_chunks, axis=1)  # [S,B,H,N,F]
    target = np.concatenate(target_chunks, axis=0)   # [B,H,N,F]
    win_idx, h_idx, node_idx, used_events = _event_positions(dataset, event_df, int(target.shape[0]), args.dedupe_positions)
    if win_idx.size == 0:
        raise RuntimeError("No event-local forecast positions found for the requested subset.")

    event_samples = samples[:, win_idx, h_idx, node_idx, :]
    event_target = target[win_idx, h_idx, node_idx, :]
    # Inverse-transform selected event points to original traffic units.
    event_samples = _inverse_points(event_samples, artifacts.scaler, node_idx, metric_feature_index)
    event_target = _inverse_points(event_target, artifacts.scaler, node_idx, metric_feature_index)

    metrics = _evaluate_event_arrays(
        event_samples,
        event_target,
        horizons=[int(x) for x in config.get("train", {}).get("eval_horizons", [3, 6, 12])],
        device=device,
        eval_batch_size=int(args.eval_batch_size),
        h_idx=h_idx,
    )
    payload = {
        "method": args.method,
        "setting": args.setting,
        "config": args.config,
        "ckpt": args.ckpt,
        "events_csv": str(args.events_csv),
        "kind": args.kind,
        "split": args.split,
        "seed": int(args.seed),
        "num_samples": int(num_samples),
        "num_positions": int(win_idx.size),
        "used_events": int(len(used_events)),
        "metrics": metrics,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
