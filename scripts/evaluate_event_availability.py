"""Evaluate event-subset performance under semantic availability regimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_experiment_and_record import _build_runtime
from semantic.availability import _event_kind, event_type_names
from trainers.evaluator import evaluate
from utils.checkpoint import load_checkpoint
from utils.config import deep_merge, load_config
from utils.device import get_device


SUBSET_TO_KIND = {"rain": "weather", "accident": "incident", "holiday": "calendar"}


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Evaluate FULL vs deploy-realistic event semantics")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0, choices=list(range(10)))
    parser.add_argument("--subsets", type=str, default="rain,accident,holiday")
    parser.add_argument("--deltas", type=str, default="0,5,15,30", help="Incident reporting lag minutes")
    parser.add_argument("--num_eval_samples", type=int, default=20)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/availability"))
    return parser.parse_args()


def _event_steps_for_subset(config: Dict, subset: str) -> np.ndarray:
    """Get event step indices for one subset kind."""
    dcfg = config["dataset"]
    data_root = Path(dcfg["data_root"]) / dcfg["name"]
    bank_file = (dcfg.get("dynamic_semantic", {}) or {}).get("bank_file", "dynamic_semantic_bank.npz")
    bank_path = data_root / str(bank_file)
    if not bank_path.exists():
        raise FileNotFoundError(f"Dynamic semantic bank is required for event subset eval: {bank_path}")
    bundle = np.load(bank_path, allow_pickle=True)
    steps = bundle["step_idx"].astype(np.int64)
    names = event_type_names(bank_path)
    if "event_type_id" in bundle:
        type_id = bundle["event_type_id"].astype(np.int64)
    else:
        events_csv = bank_path.parent / "dynamic_events.csv"
        if events_csv.exists():
            df = pd.read_csv(events_csv)
            df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
            if "event_type" in df.columns and len(df) == len(steps):
                lookup = {v: i for i, v in enumerate(names)}
                type_id = np.asarray([lookup.get(str(v), 0) for v in df["event_type"].fillna("unknown").astype(str)], dtype=np.int64)
            else:
                type_id = np.zeros_like(steps)
        else:
            type_id = np.zeros_like(steps)
    target_kind = SUBSET_TO_KIND[subset]
    keep_ids = [i for i, name in enumerate(names) if _event_kind(name) == target_kind]
    if not keep_ids:
        return np.asarray([], dtype=np.int64)
    mask = np.isin(type_id, np.asarray(keep_ids, dtype=np.int64))
    return steps[mask]


def _subset_indices(dataset, event_steps: np.ndarray) -> List[int]:
    """Return dataset indices whose forecast horizon contains at least one event step."""
    indices = []
    if event_steps.size == 0:
        return indices
    for idx, (_, his_end, fut_end) in enumerate(dataset.indices):
        global_his_end = int(dataset.base_step + his_end)
        global_fut_end = int(dataset.base_step + fut_end)
        if bool(((event_steps >= global_his_end) & (event_steps < global_fut_end)).any()):
            indices.append(idx)
    return indices


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render dataframe as markdown without tabulate."""
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _make_config(base: Dict, regime: str, delta_minutes: int, gamma_override: float | None = None) -> Dict:
    """Create availability config override."""
    override = {
        "dataset": {
            "dynamic_semantic": {
                "availability": {
                    "regime": regime,
                    "incident_lag_minutes": int(delta_minutes),
                }
            }
        },
        "train": {"eval_horizons": [12]},
    }
    if gamma_override is not None:
        override["model"] = {"gamma": float(gamma_override)}
    return deep_merge(base, override)


def _evaluate_one(config: Dict, ckpt: str, device: torch.device, subset_indices: List[int], args: argparse.Namespace) -> Dict[str, float]:
    """Evaluate one config on a subset of test windows."""
    artifacts, model, sampler, a_phy, a_sem, z_sem, dynamic_bank = _build_runtime(config, device)
    load_checkpoint(Path(ckpt), model=model, optimizer=None, map_location=str(device))
    model.eval()
    subset = Subset(artifacts.test_loader.dataset, subset_indices)
    loader = DataLoader(
        subset,
        batch_size=int(config["dataset"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return evaluate(
        model=model,
        sampler=sampler,
        data_loader=loader,
        a_phy=a_phy,
        a_sem=a_sem,
        z_sem=z_sem,
        device=device,
        scaler=artifacts.scaler,
        num_crps_samples=int(args.num_eval_samples),
        dynamic_bank=dynamic_bank,
        eval_horizons=[int(args.horizon)],
        max_batches=args.max_eval_batches,
        metric_feature_index=config["train"].get("metric_feature_index", None),
        mape_eps=float(config["train"].get("mape_eps", 1e-5)),
        mape_mask_threshold=float(config["train"].get("mape_mask_threshold", 1.0)),
    )


def main() -> None:
    """Run event subset availability analysis."""
    args = parse_args()
    base = load_config(args.config)
    subsets = [x.strip() for x in args.subsets.split(",") if x.strip()]
    deltas = [int(x) for x in args.deltas.split(",") if x.strip()]
    device = get_device(f"cuda:{int(args.gpu_id)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for subset in subsets:
        if subset not in SUBSET_TO_KIND:
            raise ValueError(f"Unsupported subset {subset}. Choose from {sorted(SUBSET_TO_KIND)}")
        event_steps = _event_steps_for_subset(base, subset=subset)
        # Build once only to access canonical test dataset indices.
        runtime_cfg = _make_config(base, regime="full", delta_minutes=0)
        artifacts, _, _, _, _, _, _ = _build_runtime(runtime_cfg, device)
        idxs = _subset_indices(artifacts.test_loader.dataset, event_steps=event_steps)
        if not idxs:
            rows.append({"subset": subset, "regime": "missing_or_empty", "delta_min": "", "windows": 0})
            continue

        no_prior_cfg = _make_config(base, regime="full", delta_minutes=0, gamma_override=0.0)
        no_prior = _evaluate_one(no_prior_cfg, args.ckpt, device, idxs, args)
        no_prior_mae = float(no_prior[f"mae@{args.horizon}"])
        no_prior_crps = float(no_prior[f"crps@{args.horizon}"])

        full_cfg = _make_config(base, regime="full", delta_minutes=0)
        full = _evaluate_one(full_cfg, args.ckpt, device, idxs, args)
        full_mae = float(full[f"mae@{args.horizon}"])
        full_crps = float(full[f"crps@{args.horizon}"])
        full_gain_mae = no_prior_mae - full_mae
        full_gain_crps = no_prior_crps - full_crps
        rows.append(
            {
                "subset": subset,
                "regime": "full",
                "delta_min": 0,
                "windows": len(idxs),
                "mae": full_mae,
                "crps": full_crps,
                "gain_mae_vs_gamma0": full_gain_mae,
                "gain_crps_vs_gamma0": full_gain_crps,
                "survival_mae_ratio": 1.0,
                "survival_crps_ratio": 1.0,
            }
        )

        for delta in deltas:
            deploy_cfg = _make_config(base, regime="deploy_realistic", delta_minutes=delta)
            deploy = _evaluate_one(deploy_cfg, args.ckpt, device, idxs, args)
            d_mae = float(deploy[f"mae@{args.horizon}"])
            d_crps = float(deploy[f"crps@{args.horizon}"])
            d_gain_mae = no_prior_mae - d_mae
            d_gain_crps = no_prior_crps - d_crps
            rows.append(
                {
                    "subset": subset,
                    "regime": "deploy_realistic",
                    "delta_min": delta,
                    "windows": len(idxs),
                    "mae": d_mae,
                    "crps": d_crps,
                    "gain_mae_vs_gamma0": d_gain_mae,
                    "gain_crps_vs_gamma0": d_gain_crps,
                    "survival_mae_ratio": d_gain_mae / full_gain_mae if abs(full_gain_mae) > 1e-8 else float("nan"),
                    "survival_crps_ratio": d_gain_crps / full_gain_crps if abs(full_gain_crps) > 1e-8 else float("nan"),
                }
            )

    df = pd.DataFrame(rows)
    csv_path = args.out_dir / "availability_event_subset_results.csv"
    md_path = args.out_dir / "availability_event_subset_results.md"
    df.to_csv(csv_path, index=False)
    note = (
        "FULL uses all event fields available up to cutoff_step. DEPLOY-REALISTIC keeps calendar/holiday/POI static, "
        "uses forecast-scaled weather semantics, and masks incident semantics until onset plus reporting lag Delta."
    )
    md_path.write_text("# Event Availability Analysis\n\n" + note + "\n\n" + _df_to_markdown(df) + "\n", encoding="utf-8")
    print(df.to_string(index=False))
    print(f"Saved {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
