"""Evaluate a frozen mean predictor as the bare deterministic row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Mapping, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataio.traffic_dataset import build_dataloaders
from baselines.runners.run_agcrn import evaluate_agcrn
from models.mean_predictor import MeanPredictor
from utils.config import load_config
from utils.device import get_device
from utils.logger import get_logger
from utils.result_writer import ExperimentResult, write_experiment_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen AGCRN mean predictor.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gpu_id", type=int, default=None, choices=list(range(10)))
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--output_csv", type=str, default="outputs/results.csv")
    parser.add_argument("--results_md", type=str, default="RESULTS.md")
    return parser.parse_args()


def rows_from_metrics(metrics: Mapping[str, float], config: Mapping, config_path: str) -> List[ExperimentResult]:
    dcfg = config["dataset"]
    tcfg = config["train"]
    mean_cfg = config.get("mean_predictor", config.get("model", {}).get("mean_predictor", {}))
    rows: List[ExperimentResult] = []
    for hh in [int(x) for x in tcfg.get("eval_horizons", [3, 6, 12])]:
        rows.append(
            ExperimentResult(
                dataset=str(dcfg["name"]),
                method="Frozen-AGCRN-Mean",
                setting="bare-agcrn",
                horizon=hh,
                mae=float(metrics.get(f"mae@{hh}", metrics["mae"])),
                rmse=float(metrics.get(f"rmse@{hh}", metrics["rmse"])),
                crps=None,
                seed=int(tcfg.get("seed", 42)),
                config=config_path,
                implementation="official-frozen",
                checkpoint=str(dict(mean_cfg).get("pretrained_ckpt", "")),
                settings_json=json.dumps(dict(mean_cfg), sort_keys=True),
                notes="residual=0 bare deterministic mean row using shared split/scaler",
            )
        )
    return rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = get_logger()
    device_arg = args.device if args.gpu_id is None else f"cuda:{int(args.gpu_id)}"
    device = get_device(device_arg)

    artifacts = build_dataloaders(config)
    predictor = MeanPredictor(config=config, device=device).to(device)
    metric_feature_index = config["train"].get("metric_feature_index", None)
    metric_feature_index = None if metric_feature_index is None else int(metric_feature_index)
    eval_horizons = [int(x) for x in config["train"].get("eval_horizons", [3, 6, 12])]
    # Reuse the AGCRN runner evaluator so the bare row is bit-for-bit aligned
    # with the validated deterministic baseline path.
    metrics = evaluate_agcrn(
        model=predictor.model,
        loader=artifacts.test_loader,
        device=device,
        scaler=artifacts.scaler,
        composer=None,
        eval_horizons=eval_horizons,
        max_batches=args.max_batches,
        metric_feature_index=metric_feature_index,
        mape_eps=float(config["train"].get("mape_eps", 1e-5)),
        mape_mask_threshold=float(config["train"].get("mape_mask_threshold", 1.0)),
    )
    logger.info("Bare frozen AGCRN | MAE=%.6f RMSE=%.6f", metrics["mae"], metrics["rmse"])
    for hh in eval_horizons:
        logger.info(
            "Horizon %d | MAE=%.6f RMSE=%.6f",
            hh,
            metrics.get(f"mae@{hh}", metrics["mae"]),
            metrics.get(f"rmse@{hh}", metrics["rmse"]),
        )
    write_experiment_results(
        rows_from_metrics(metrics, config=config, config_path=args.config),
        csv_path=Path(args.output_csv),
        md_path=Path(args.results_md),
        title="Frozen AGCRN Mean Results",
    )


if __name__ == "__main__":
    main()
