"""Evaluate Prompt-STDiff checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

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
from utils.config import load_config
from utils.device import get_device
from utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Prompt-STDiff")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", type=str, default="test", choices=("val", "test"), help="Evaluation split.")
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|cuda:0...")
    parser.add_argument("--gpu_id", type=int, default=None, choices=list(range(10)), help="CUDA GPU id (0-9).")
    return parser.parse_args()


def load_semantic(config: dict) -> np.ndarray:
    """Load semantic embedding cache with strict/fallback policy."""
    dcfg = config["dataset"]
    mcfg = config["model"]
    data_root = Path(dcfg["data_root"]) / dcfg["name"]
    sem_path = data_root / dcfg["semantic_embedding_file"]
    semantic_required = bool(dcfg.get("semantic_required", True))
    allow_fallback = bool(dcfg.get("allow_random_semantic_fallback", False))

    if sem_path.exists():
        return load_semantic_embeddings(sem_path)

    if semantic_required and (not allow_fallback):
        raise FileNotFoundError(
            f"Semantic embeddings are required but missing: {sem_path}. "
            "Please generate semantic cache before evaluation."
        )

    # ASSUMPTION: evaluation fallback is only for debug/dry-run.
    rng = np.random.default_rng(seed=0)
    return rng.standard_normal((int(dcfg["num_nodes"]), int(mcfg["sem_dim"]))).astype(np.float32)


def main() -> None:
    """Main evaluation entry."""
    args = parse_args()
    config = load_config(args.config)
    logger = get_logger()

    dcfg = config["dataset"]
    mcfg = config["model"]
    diff_cfg = config["diffusion"]

    device_arg = args.device
    if args.gpu_id is not None:
        device_arg = f"cuda:{int(args.gpu_id)}"
    device = get_device(device_arg)

    artifacts = build_dataloaders(config)

    data_root = Path(dcfg["data_root"]) / dcfg["name"]
    num_nodes = int(dcfg["num_nodes"])
    dynamic_bank = maybe_load_dynamic_semantic_bank(config, data_root=data_root, logger=logger)
    dynamic_bank = wrap_dynamic_bank_from_config(dynamic_bank, config=config, data_root=data_root)

    a_phy_np = load_or_build_physical_graph(
        file_path=data_root / dcfg["adjacency_file"],
        num_nodes=num_nodes,
        sigma=dcfg.get("physical_sigma", "auto"),
        normalize_mode=str(dcfg.get("physical_norm_mode", "sym")),
    )
    z_sem_np = load_semantic(config)
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

    a_phy = to_torch(a_phy_np, device=device)
    a_sem = to_torch(a_sem_np, device=device)
    z_sem = torch.tensor(z_sem_np, dtype=torch.float32, device=device)
    sem_dim_data = int(z_sem_np.shape[1])
    sem_dim_cfg = int(mcfg["sem_dim"])
    if sem_dim_cfg != sem_dim_data:
        # ASSUMPTION: auto-align semantic dimension to loaded embedding dimension.
        logger.warning(
            "sem_dim mismatch: config=%d, embedding=%d. Using embedding dimension.",
            sem_dim_cfg,
            sem_dim_data,
        )
    if dynamic_bank is not None and dynamic_bank.sem_dim != sem_dim_data:
        logger.warning(
            "dynamic semantic dim mismatch: bank=%d, static=%d. Disabling dynamic semantic.",
            dynamic_bank.sem_dim,
            sem_dim_data,
        )
        dynamic_bank = None

    mean_predictor = None
    mean_predictor_cfg = dict(get_mean_predictor_config(config))
    if mean_predictor_cfg.get("type"):
        mean_predictor = MeanPredictor(config=config, device=device).to(device)

    model = PromptSTDiff(
        input_dim=int(dcfg["input_dim"]),
        sem_dim=sem_dim_data,
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
                logger=logger,
            )
        )

    betas = build_beta_schedule(diff_cfg)
    process = DiffusionProcess(
        betas=betas,
        clip_x0=bool(diff_cfg.get("clip_x0", True)),
    ).to(device)

    noise_prior = SemanticGuidedDynamicNoisePrior(
        sem_dim=sem_dim_data,
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

    _, _, _ = load_checkpoint(
        Path(args.ckpt),
        model=model,
        optimizer=None,
        map_location=str(device),
        strict=mean_predictor is None,
    )
    logger.info("Loaded checkpoint: %s", args.ckpt)

    eval_loader = artifacts.val_loader if args.split == "val" else artifacts.test_loader
    metrics = evaluate(
        model=model,
        sampler=sampler,
        data_loader=eval_loader,
        a_phy=a_phy,
        a_sem=a_sem,
        z_sem=z_sem,
        device=device,
        scaler=artifacts.scaler,
        num_crps_samples=int(config["train"].get("num_eval_samples", 20)),
        dynamic_bank=dynamic_bank,
        eval_horizons=[int(x) for x in config["train"].get("eval_horizons", [3, 6, 12])],
        metric_feature_index=config["train"].get("metric_feature_index", None),
        mape_eps=float(config["train"].get("mape_eps", 1e-5)),
        mape_mask_threshold=float(config["train"].get("mape_mask_threshold", 1.0)),
        predict_residual=bool(config.get("model", {}).get("predict_residual", False)),
    )

    logger.info(
        "%s metrics | MAE=%.6f RMSE=%.6f MAPE=%.6f CRPS=%.6f NLL=%.6f PICP90=%.6f MPIW90=%.6f WINKLER90=%.6f SHARPNESS=%.6f",
        args.split.capitalize(),
        metrics["mae"],
        metrics["rmse"],
        metrics["mape"],
        metrics["crps"],
        metrics.get("nll", float("nan")),
        metrics.get("picp@90", float("nan")),
        metrics.get("mpiw@90", float("nan")),
        metrics.get("winkler@90", float("nan")),
        metrics.get("sharpness", float("nan")),
    )
    for h in [int(x) for x in config["train"].get("eval_horizons", [3, 6, 12])]:
        k_mae = f"mae@{h}"
        k_rmse = f"rmse@{h}"
        k_mape = f"mape@{h}"
        k_crps = f"crps@{h}"
        if all(k in metrics for k in (k_mae, k_rmse, k_mape, k_crps)):
            logger.info(
                "Horizon %d | MAE=%.6f RMSE=%.6f MAPE=%.6f CRPS=%.6f NLL=%.6f PICP90=%.6f MPIW90=%.6f WINKLER90=%.6f REL90=%.6f",
                h,
                metrics[k_mae],
                metrics[k_rmse],
                metrics[k_mape],
                metrics[k_crps],
                metrics.get(f"nll@{h}", float("nan")),
                metrics.get(f"picp@90@{h}", float("nan")),
                metrics.get(f"mpiw@90@{h}", float("nan")),
                metrics.get(f"winkler@90@{h}", float("nan")),
                metrics.get(f"reliability@90@{h}", float("nan")),
            )


if __name__ == "__main__":
    main()
