"""Train Prompt-STDiff on PEMS datasets."""

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
from trainers.trainer import Trainer
from utils.config import load_config
from utils.device import get_device
from utils.logger import get_logger
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train Prompt-STDiff")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|cuda:0...")
    parser.add_argument("--gpu_id", type=int, default=None, choices=list(range(10)), help="CUDA GPU id (0-9).")
    return parser.parse_args()


def load_semantic_embeddings_or_fallback(config: dict, logger) -> np.ndarray:
    """Load cached semantic embeddings, fallback to random initialization."""
    dcfg = config["dataset"]
    mcfg = config["model"]
    root = Path(dcfg["data_root"]) / dcfg["name"]
    sem_path = root / dcfg["semantic_embedding_file"]

    if sem_path.exists():
        z_sem = load_semantic_embeddings(sem_path)
        return z_sem

    # ASSUMPTION: fallback random semantic embeddings if cache not found.
    logger.warning("semantic embeddings missing at %s, using random fallback", sem_path)
    z_sem = np.random.randn(int(dcfg["num_nodes"]), int(mcfg["sem_dim"])).astype(np.float32)
    sem_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(sem_path, z_sem)
    return z_sem


def main() -> None:
    """Main training entry."""
    args = parse_args()
    config = load_config(args.config)

    logger = get_logger()
    set_seed(int(config["train"]["seed"]))

    dcfg = config["dataset"]
    mcfg = config["model"]
    diff_cfg = config["diffusion"]

    artifacts = build_dataloaders(config)

    device_arg = args.device
    if args.gpu_id is not None:
        device_arg = f"cuda:{int(args.gpu_id)}"
    device = get_device(device_arg)
    logger.info("Using device: %s", device)

    data_root = Path(dcfg["data_root"]) / dcfg["name"]
    num_nodes = int(dcfg["num_nodes"])
    dynamic_bank = maybe_load_dynamic_semantic_bank(config, data_root=data_root, logger=logger)

    a_phy_np = load_or_build_physical_graph(
        file_path=data_root / dcfg["adjacency_file"],
        num_nodes=num_nodes,
        sigma=dcfg.get("physical_sigma", "auto"),
        normalize_mode=str(dcfg.get("physical_norm_mode", "sym")),
    )

    z_sem_np = load_semantic_embeddings_or_fallback(config, logger)
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    trainer = Trainer(
        model=model,
        process=process,
        sampler=sampler,
        noise_prior=noise_prior,
        optimizer=optimizer,
        train_loader=artifacts.train_loader,
        val_loader=artifacts.val_loader,
        a_phy=a_phy,
        a_sem=a_sem,
        z_sem=z_sem,
        device=device,
        config=config,
        scaler_obj=artifacts.scaler,
        dynamic_bank=dynamic_bank,
    )

    trainer.train()


if __name__ == "__main__":
    main()
