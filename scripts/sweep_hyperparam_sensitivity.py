"""Hyperparameter sensitivity sweeps for Prompt-STDiff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, Iterable, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import deep_merge, load_config


SWEEPS = {
    "gamma": [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
    "topk": [3, 5, 10, 20],
    "pdrop": [0.0, 0.1, 0.2, 0.3],
    "diffusion_steps": [10, 25, 50, 100],
}


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Run hyperparameter sensitivity sweeps")
    parser.add_argument("--config", type=str, default="configs/pems08.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0, choices=list(range(10)))
    parser.add_argument("--factors", type=str, default="gamma,topk,pdrop,diffusion_steps")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--num_eval_samples", type=int, default=20)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--latency_batch_size", type=int, default=1)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/sensitivity/pems08_h12"))
    parser.add_argument("--append", action="store_true", help="Append to existing result files")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _factor_overrides(base_cfg: Dict[str, Any], factor: str, value: Any) -> Dict[str, Any]:
    """Build config overrides for one sweep point."""
    if factor == "gamma":
        return {"model": {"gamma": float(value)}}
    if factor == "topk":
        k = int(value)
        # Use distinct files so cached semantic graphs from other K values are never reused.
        return {
            "dataset": {
                "semantic_top_k": k,
                "semantic_graph_file": f"A_sem_topk{k}_norm.npy",
                "semantic_graph_raw_file": f"A_sem_topk{k}.npy",
                "semantic_graph_rebuild": True,
            }
        }
    if factor == "pdrop":
        return {"model": {"semantic_dropout_p": float(value)}}
    if factor == "diffusion_steps":
        steps = int(value)
        return {"diffusion": {"num_steps": steps, "sampling_steps": steps}}
    raise ValueError(f"Unsupported factor: {factor}")


def _write_temp_config(base_cfg: Dict[str, Any], factor: str, value: Any, out_dir: Path) -> Path:
    """Write one temporary config file."""
    cfg = deep_merge(base_cfg, _factor_overrides(base_cfg, factor=factor, value=value))
    cfg.setdefault("train", {})
    cfg["train"]["eval_horizons"] = [12]
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_value = str(value).replace(".", "p")
    path = out_dir / f"tmp_{factor}_{safe_value}.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
    return path


def _run_one(
    config_path: Path,
    factor: str,
    value: Any,
    args: argparse.Namespace,
    csv_path: Path,
    md_path: Path,
) -> None:
    """Run one eval-and-record command."""
    setting = f"sensitivity_{factor}_{value}"
    cmd: List[str] = [
        sys.executable,
        "scripts/run_experiment_and_record.py",
        "--config",
        str(config_path),
        "--ckpt",
        str(args.ckpt),
        "--gpu_id",
        str(args.gpu_id),
        "--num_eval_samples",
        str(args.num_eval_samples),
        "--latency_batch_size",
        str(args.latency_batch_size),
        "--method",
        "Prompt-STDiff",
        "--setting",
        setting,
        "--implementation",
        "ours",
        "--csv",
        str(csv_path),
        "--md",
        str(md_path),
        "--title",
        "PeMS08 H12 Hyperparameter Sensitivity",
    ]
    if args.max_eval_batches is not None:
        cmd.extend(["--max_eval_batches", str(args.max_eval_batches)])
    print("[run]", " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    """Run requested sweeps."""
    args = parse_args()
    factors = [x.strip() for x in args.factors.split(",") if x.strip()]
    base_cfg = load_config(args.config)
    base_cfg.setdefault("train", {})
    base_cfg["train"]["eval_horizons"] = [int(args.horizon)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "sensitivity_results.csv"
    md_path = args.out_dir / "SENSITIVITY_RESULTS.md"
    tmp_dir = args.out_dir / "tmp_configs"
    if not args.append:
        for p in (csv_path, md_path):
            if p.exists():
                p.unlink()
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    manifest = {
        "base_config": args.config,
        "ckpt": str(args.ckpt),
        "horizon": int(args.horizon),
        "factors": factors,
        "sweeps": {factor: SWEEPS[factor] for factor in factors},
        "consistency_check": "gamma=0 corresponds to w/o Dynamic Prior when all other settings match the ablation run.",
        "topk_cache_policy": "Each Top-K uses distinct A_sem_topk{K}.npy/A_sem_topk{K}_norm.npy with semantic_graph_rebuild=true.",
        "pdrop_note": "semantic_dropout_p affects training; in pure checkpoint evaluation model.eval() disables semantic dropout.",
    }
    (args.out_dir / "sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for factor in factors:
        if factor not in SWEEPS:
            raise ValueError(f"Unsupported factor={factor}. Choose from {sorted(SWEEPS)}")
        for value in SWEEPS[factor]:
            cfg_path = _write_temp_config(base_cfg, factor=factor, value=value, out_dir=tmp_dir)
            _run_one(cfg_path, factor=factor, value=value, args=args, csv_path=csv_path, md_path=md_path)

    if not args.dry_run:
        plot_cmd = [
            sys.executable,
            "scripts/plot_hyperparam_sensitivity.py",
            "--csv",
            str(csv_path),
            "--out_dir",
            str(args.out_dir),
            "--horizon",
            str(args.horizon),
        ]
        print("[plot]", " ".join(plot_cmd))
        subprocess.run(plot_cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
