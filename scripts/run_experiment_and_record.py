"""Evaluate a checkpoint, measure latency, and record results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataio.traffic_dataset import build_dataloaders
from diffusion.noise_prior import SemanticGuidedDynamicNoisePrior
from diffusion.process import DiffusionProcess
from diffusion.sampler import DiffusionSampler
from diffusion.schedules import build_beta_schedule
from graph.graph_utils import to_torch
from graph.physical_graph import load_or_build_physical_graph
from graph.semantic_graph import load_or_build_semantic_graph
from models.prompt_stdiff import PromptSTDiff
from models.mean_predictor import (
    MeanPredictor,
    compute_or_load_residual_standardizer,
    get_mean_predictor_config,
    residual_stats_path_from_config,
)
from semantic.availability import wrap_dynamic_bank_from_config
from semantic.dynamic_context import maybe_load_dynamic_semantic_bank
from semantic.semantic_cache import load_semantic_embeddings
from trainers.evaluator import evaluate
from utils.checkpoint import load_checkpoint
from utils.config import deep_merge, load_config
from utils.device import get_device
from utils.result_writer import ExperimentResult, write_experiment_results


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Evaluate and record Prompt-STDiff results")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gpu_id", type=int, default=None, choices=list(range(10)))
    parser.add_argument("--sampler", type=str, default=None, choices=["ddpm", "ddim"])
    parser.add_argument("--sampling_steps", type=int, default=None)
    parser.add_argument("--num_eval_samples", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--latency_batch_size", type=int, default=1)
    parser.add_argument("--latency_warmup", type=int, default=2)
    parser.add_argument("--latency_runs", type=int, default=3)
    parser.add_argument("--method", type=str, default="Prompt-STDiff")
    parser.add_argument("--setting", type=str, default="eval")
    parser.add_argument("--implementation", type=str, default="ours")
    parser.add_argument("--csv", type=Path, default=Path("outputs/results.csv"))
    parser.add_argument("--md", type=Path, default=Path("RESULTS.md"))
    parser.add_argument("--title", type=str, default="Experiment Results")
    return parser.parse_args()


def _load_semantic(config: Dict) -> np.ndarray:
    dcfg = config["dataset"]
    root = Path(dcfg["data_root"]) / dcfg["name"]
    return load_semantic_embeddings(root / dcfg["semantic_embedding_file"])


def _build_runtime(config: Dict, device: torch.device):
    """Build model, sampler, tensors, dataloaders."""
    dcfg = config["dataset"]
    mcfg = config["model"]
    diff_cfg = config["diffusion"]
    artifacts = build_dataloaders(config)
    data_root = Path(dcfg["data_root"]) / dcfg["name"]
    num_nodes = int(dcfg["num_nodes"])

    dynamic_bank = maybe_load_dynamic_semantic_bank(config, data_root=data_root, logger=_NullLogger())
    dynamic_bank = wrap_dynamic_bank_from_config(dynamic_bank, config=config, data_root=data_root)
    a_phy_np = load_or_build_physical_graph(
        file_path=data_root / dcfg["adjacency_file"],
        num_nodes=num_nodes,
        sigma=dcfg.get("physical_sigma", "auto"),
        normalize_mode=str(dcfg.get("physical_norm_mode", "sym")),
    )
    z_sem_np = _load_semantic(config)
    a_sem_np = load_or_build_semantic_graph(
        graph_path=data_root / dcfg["semantic_graph_file"],
        z_sem=z_sem_np,
        top_k=int(dcfg["semantic_top_k"]),
        rebuild=bool(dcfg.get("semantic_graph_rebuild", False)),
        normalize_mode=str(dcfg.get("semantic_graph_norm_mode", "sym")),
        raw_graph_path=(
            data_root / dcfg["semantic_graph_raw_file"]
            if dcfg.get("semantic_graph_raw_file")
            else None
        ),
    )
    sem_dim = int(z_sem_np.shape[1])
    mean_predictor = None
    mean_predictor_cfg = dict(get_mean_predictor_config(config))
    if mean_predictor_cfg.get("type"):
        mean_predictor = MeanPredictor(config=config, device=device).to(device)

    model = PromptSTDiff(
        input_dim=int(dcfg["input_dim"]),
        sem_dim=sem_dim,
        hidden_dim=int(mcfg["hidden_dim"]),
        horizon_steps=int(dcfg["horizon_steps"]),
        time_embed_dim=int(mcfg["time_embed_dim"]),
        router_hidden_dim=int(mcfg["router_hidden_dim"]),
        num_layers=int(mcfg["num_layers"]),
        dropout=float(mcfg["dropout"]),
        semantic_dropout_p=float(mcfg.get("semantic_dropout_p", 0.1)),
        use_semantic=bool(mcfg.get("use_semantic", True)),
        use_mean_head=bool(mcfg.get("use_mean_head", False)),
        mean_head_hidden_dim=(
            int(mcfg["mean_head_hidden_dim"])
            if mcfg.get("mean_head_hidden_dim") is not None
            else None
        ),
        mean_predictor=mean_predictor,
        center_residual_samples=bool(mcfg.get("center_residual_samples", False)),
        residual_sample_scale=float(mcfg.get("residual_sample_scale", 1.0)),
        residual_horizon_scale=mcfg.get("residual_horizon_scale", None),
        residual_node_group_ids=mcfg.get("residual_node_group_ids", None),
        residual_node_group_scale=mcfg.get("residual_node_group_scale", None),
        use_hetero_residual_scale=bool(mcfg.get("use_hetero_residual_scale", False)),
        hetero_scale_hidden_dim=(
            int(mcfg["hetero_scale_hidden_dim"])
            if mcfg.get("hetero_scale_hidden_dim") is not None
            else None
        ),
        hetero_scale_min=float(mcfg.get("hetero_scale_min", 0.2)),
        hetero_scale_max=float(mcfg.get("hetero_scale_max", 6.0)),
        hetero_scale_use_semantic=(
            bool(mcfg["hetero_scale_use_semantic"])
            if mcfg.get("hetero_scale_use_semantic") is not None
            else None
        ),
        use_incident_tail_scale=bool(mcfg.get("use_incident_tail_scale", False)),
        incident_tail_hidden_dim=(
            int(mcfg["incident_tail_hidden_dim"])
            if mcfg.get("incident_tail_hidden_dim") is not None
            else None
        ),
        incident_tail_min_scale=float(mcfg.get("incident_tail_min_scale", 0.85)),
        incident_tail_max_scale=float(mcfg.get("incident_tail_max_scale", 4.0)),
        incident_tail_use_semantic=bool(mcfg.get("incident_tail_use_semantic", True)),
        incident_tail_df=float(mcfg.get("incident_tail_df", 3.0)),
        use_incident_mean_correction=bool(mcfg.get("use_incident_mean_correction", False)),
        incident_correction_hidden_dim=(
            int(mcfg["incident_correction_hidden_dim"])
            if mcfg.get("incident_correction_hidden_dim") is not None
            else None
        ),
        incident_correction_use_semantic=bool(mcfg.get("incident_correction_use_semantic", True)),
        incident_correction_max_shift=float(mcfg.get("incident_correction_max_shift", 4.0)),
        incident_correction_graph_hops=int(mcfg.get("incident_correction_graph_hops", 2)),
        incident_correction_gate_bias=float(mcfg.get("incident_correction_gate_bias", -4.0)),
    ).to(device)
    if mean_predictor is not None and bool(mean_predictor_cfg.get("residual_standardize", True)):
        model.set_residual_standardizer(
            compute_or_load_residual_standardizer(
                path=residual_stats_path_from_config(config),
                mean_predictor=mean_predictor,
                train_loader=artifacts.train_loader,
                device=device,
                force_recompute=False,
                logger=_NullLogger(),
            )
        )
    process = DiffusionProcess(
        betas=build_beta_schedule(diff_cfg),
        clip_x0=bool(diff_cfg.get("clip_x0", True)),
    ).to(device)
    noise_prior = SemanticGuidedDynamicNoisePrior(
        sem_dim=sem_dim,
        horizon=int(dcfg["horizon_steps"]),
        out_dim=int(dcfg["input_dim"]),
        num_diffusion_steps=int(diff_cfg["num_steps"]),
        gamma=float(mcfg["gamma"]),
        gamma_schedule=str(mcfg.get("gamma_schedule", "constant")),
        learn_mu_prior=bool(mcfg.get("learn_mu_prior", False)),
        learn_sigma_prior=bool(mcfg.get("learn_sigma_prior", False)),
    ).to(device)
    sampler = DiffusionSampler(
        process=process,
        noise_prior=noise_prior,
        sampling_steps=int(diff_cfg.get("sampling_steps", diff_cfg["num_steps"])),
        sampler_type=str(diff_cfg.get("sampler", "ddpm")),
    )
    return (
        artifacts,
        model,
        sampler,
        to_torch(a_phy_np, device=device),
        to_torch(a_sem_np, device=device),
        torch.tensor(z_sem_np, dtype=torch.float32, device=device),
        dynamic_bank,
    )


class _NullLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


@torch.no_grad()
def _measure_latency_ms(
    model: PromptSTDiff,
    sampler: DiffusionSampler,
    loader,
    a_phy: torch.Tensor,
    a_sem: torch.Tensor,
    z_sem: torch.Tensor,
    device: torch.device,
    batch_size: int,
    warmup: int,
    runs: int,
    dynamic_bank=None,
) -> float:
    """Measure inference latency in ms/sample for one sampler.sample call."""
    batch = next(iter(loader))
    x_his = batch["x_his"][:batch_size].to(device=device, dtype=torch.float32)
    x_fut = batch["x_fut"][:batch_size].to(device=device, dtype=torch.float32)
    cutoff = batch["cutoff_step"][:batch_size].to(device=device, dtype=torch.long)
    b, h, n, f = x_fut.shape
    if dynamic_bank is not None:
        z_batch = dynamic_bank.compose(
            static_z_sem=z_sem,
            cutoff_steps=cutoff,
            num_nodes=n,
            device=device,
        )
    else:
        z_batch = z_sem
    cond = {"x_his": x_his, "a_phy": a_phy, "a_sem": a_sem, "z_sem": z_batch}

    for _ in range(max(0, warmup)):
        _ = sampler.sample(model_fn=model.model_fn, shape=(b, h, n, f), cond=cond, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    times: List[float] = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        _ = sampler.sample(model_fn=model.model_fn, shape=(b, h, n, f), cond=cond, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
    return 1000.0 * float(np.mean(times)) / max(1, b)


def main() -> None:
    """Run evaluation, latency measurement, and result recording."""
    args = parse_args()
    config = load_config(args.config)
    overrides: Dict = {"diffusion": {}}
    if args.sampler is not None:
        overrides["diffusion"]["sampler"] = args.sampler
    if args.sampling_steps is not None:
        overrides["diffusion"]["sampling_steps"] = int(args.sampling_steps)
    if args.num_eval_samples is not None:
        overrides.setdefault("train", {})["num_eval_samples"] = int(args.num_eval_samples)
    config = deep_merge(config, overrides)

    device_arg = args.device
    if args.gpu_id is not None:
        device_arg = f"cuda:{int(args.gpu_id)}"
    device = get_device(device_arg)
    artifacts, model, sampler, a_phy, a_sem, z_sem, dynamic_bank = _build_runtime(config, device)
    load_checkpoint(
        Path(args.ckpt),
        model=model,
        optimizer=None,
        map_location=str(device),
        strict=not bool(dict(get_mean_predictor_config(config)).get("type")),
    )
    model.eval()

    metrics = evaluate(
        model=model,
        sampler=sampler,
        data_loader=artifacts.test_loader,
        a_phy=a_phy,
        a_sem=a_sem,
        z_sem=z_sem,
        device=device,
        scaler=artifacts.scaler,
        num_crps_samples=int(config["train"].get("num_eval_samples", 20)),
        dynamic_bank=dynamic_bank,
        eval_horizons=[int(x) for x in config["train"].get("eval_horizons", [3, 6, 12])],
        max_batches=args.max_eval_batches,
        metric_feature_index=config["train"].get("metric_feature_index", None),
        mape_eps=float(config["train"].get("mape_eps", 1e-5)),
        mape_mask_threshold=float(config["train"].get("mape_mask_threshold", 1.0)),
        predict_residual=bool(config.get("model", {}).get("predict_residual", False)),
    )
    latency_ms = _measure_latency_ms(
        model=model,
        sampler=sampler,
        loader=artifacts.test_loader,
        a_phy=a_phy,
        a_sem=a_sem,
        z_sem=z_sem,
        device=device,
        batch_size=int(args.latency_batch_size),
        warmup=int(args.latency_warmup),
        runs=int(args.latency_runs),
        dynamic_bank=dynamic_bank,
    )
    base_settings = {
        "sampler": config["diffusion"].get("sampler", "ddpm"),
        "sampling_steps": int(config["diffusion"].get("sampling_steps", config["diffusion"]["num_steps"])),
        "num_eval_samples": int(config["train"].get("num_eval_samples", 20)),
        "latency_ms_per_sample": latency_ms,
        "latency_batch_size": int(args.latency_batch_size),
        "max_eval_batches": args.max_eval_batches,
    }
    rows: List[ExperimentResult] = []
    for h in [int(x) for x in config["train"].get("eval_horizons", [3, 6, 12])]:
        if f"mae@{h}" not in metrics:
            continue
        row_settings = dict(base_settings)
        for key in (
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
        ):
            metric_key = f"{key}@{h}"
            if metric_key in metrics:
                row_settings[key] = float(metrics[metric_key])
        rows.append(
            ExperimentResult(
                dataset=str(config["dataset"]["name"]),
                method=args.method,
                setting=args.setting,
                horizon=h,
                mae=float(metrics[f"mae@{h}"]),
                rmse=float(metrics[f"rmse@{h}"]),
                crps=float(metrics[f"crps@{h}"]),
                seed=int(config["train"]["seed"]),
                config=args.config,
                implementation=args.implementation,
                checkpoint=args.ckpt,
                settings_json=json.dumps(row_settings, sort_keys=True),
                notes=(
                    f"latency_ms_per_sample={latency_ms:.6f}; "
                    f"nll={row_settings.get('nll', float('nan'))}; "
                    f"picp@90={row_settings.get('picp@90', float('nan'))}"
                ),
            )
        )
    write_experiment_results(rows, csv_path=args.csv, md_path=args.md, title=args.title)
    print(json.dumps({"metrics": metrics, "latency_ms_per_sample": latency_ms}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
