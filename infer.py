"""Inference script for Prompt-STDiff."""

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
from semantic.dynamic_context import maybe_load_dynamic_semantic_bank
from semantic.semantic_cache import load_semantic_embeddings
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.device import get_device
from utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Inference with Prompt-STDiff")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", type=str, default="outputs/predictions.npy")
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|cuda:0...")
    parser.add_argument("--gpu_id", type=int, default=None, choices=list(range(10)), help="CUDA GPU id (0-9).")
    return parser.parse_args()


def _load_semantic(config: dict) -> np.ndarray:
    dcfg = config["dataset"]
    mcfg = config["model"]
    root = Path(dcfg["data_root"]) / dcfg["name"]
    sem_file = root / dcfg["semantic_embedding_file"]
    semantic_required = bool(dcfg.get("semantic_required", True))
    allow_fallback = bool(dcfg.get("allow_random_semantic_fallback", False))
    if sem_file.exists():
        return load_semantic_embeddings(sem_file)

    if semantic_required and (not allow_fallback):
        raise FileNotFoundError(
            f"Semantic embeddings are required but missing: {sem_file}. "
            "Please generate semantic cache before inference."
        )

    # ASSUMPTION: fallback random semantic embeddings are only for debug/dry run.
    return np.random.randn(int(dcfg["num_nodes"]), int(mcfg["sem_dim"])).astype(np.float32)


def main() -> None:
    """Run one full-split inference and save predictions."""
    args = parse_args()
    config = load_config(args.config)

    dcfg = config["dataset"]
    mcfg = config["model"]
    diff_cfg = config["diffusion"]
    logger = get_logger()

    artifacts = build_dataloaders(config)
    split_loader = {
        "train": artifacts.train_loader,
        "val": artifacts.val_loader,
        "test": artifacts.test_loader,
    }[args.split]

    device_arg = args.device
    if args.gpu_id is not None:
        device_arg = f"cuda:{int(args.gpu_id)}"
    device = get_device(device_arg)
    root = Path(dcfg["data_root"]) / dcfg["name"]
    dynamic_bank = maybe_load_dynamic_semantic_bank(config, data_root=root, logger=logger)

    a_phy_np = load_or_build_physical_graph(
        file_path=root / dcfg["adjacency_file"],
        num_nodes=int(dcfg["num_nodes"]),
        sigma=dcfg.get("physical_sigma", "auto"),
        normalize_mode=str(dcfg.get("physical_norm_mode", "sym")),
    )
    z_sem_np = _load_semantic(config)
    a_sem_np = load_or_build_semantic_graph(
        graph_path=root / dcfg["semantic_graph_file"],
        z_sem=z_sem_np,
        top_k=int(dcfg["semantic_top_k"]),
        rebuild=False,
        normalize_mode=str(dcfg.get("semantic_graph_norm_mode", "sym")),
        raw_graph_path=(
            root / dcfg["semantic_graph_raw_file"]
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
    ).to(device)

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
    )

    load_checkpoint(Path(args.ckpt), model=model, optimizer=None, map_location=str(device))
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in split_loader:
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
            cond = {
                "x_his": x_his,
                "a_phy": a_phy,
                "a_sem": a_sem,
                "z_sem": z_sem_batch,
            }
            pred = sampler.sample(
                model_fn=model.model_fn,
                shape=(b, h, n, f),
                cond=cond,
                device=device,
            )
            if artifacts.scaler is not None:
                pred_np = artifacts.scaler.inverse_transform(pred.detach().cpu().numpy())
            else:
                pred_np = pred.detach().cpu().numpy()
            preds.append(pred_np)

    pred_all = np.concatenate(preds, axis=0)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, pred_all)
    print(f"Saved predictions to {out_path} with shape {pred_all.shape}")


if __name__ == "__main__":
    main()
