"""Efficiency benchmark for Prompt-STDiff and available baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

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
from semantic.dynamic_context import maybe_load_dynamic_semantic_bank
from semantic.semantic_cache import load_semantic_embeddings
from utils.config import load_config
from utils.device import get_device
from utils.result_writer import results_to_markdown


BASELINES = ("SpecSTG", "DiffSTG", "PDFormer")


class _NullLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Prompt-STDiff efficiency benchmark")
    parser.add_argument("--config", type=str, default="configs/pems08.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gpu_id", type=int, default=None, choices=list(range(10)))
    parser.add_argument("--max_train_batches", type=int, default=20)
    parser.add_argument("--latency_warmup", type=int, default=2)
    parser.add_argument("--latency_runs", type=int, default=3)
    parser.add_argument("--realistic_batch_size", type=int, default=None)
    parser.add_argument("--out_csv", type=Path, default=Path("outputs/efficiency/efficiency_table.csv"))
    parser.add_argument("--out_md", type=Path, default=Path("outputs/efficiency/efficiency_table.md"))
    return parser.parse_args()


def _load_runtime(config: Dict, device: torch.device):
    """Build Prompt-STDiff runtime objects."""
    dcfg = config["dataset"]
    mcfg = config["model"]
    diff_cfg = config["diffusion"]
    artifacts = build_dataloaders(config)
    data_root = Path(dcfg["data_root"]) / dcfg["name"]
    num_nodes = int(dcfg["num_nodes"])
    dynamic_bank = maybe_load_dynamic_semantic_bank(config, data_root=data_root, logger=_NullLogger())
    a_phy_np = load_or_build_physical_graph(
        file_path=data_root / dcfg["adjacency_file"],
        num_nodes=num_nodes,
        sigma=dcfg.get("physical_sigma", "auto"),
        normalize_mode=str(dcfg.get("physical_norm_mode", "sym")),
    )
    z_sem_np = load_semantic_embeddings(data_root / dcfg["semantic_embedding_file"])
    a_sem_np = load_or_build_semantic_graph(
        graph_path=data_root / dcfg["semantic_graph_file"],
        z_sem=z_sem_np,
        top_k=int(dcfg["semantic_top_k"]),
        rebuild=False,
        normalize_mode=str(dcfg.get("semantic_graph_norm_mode", "sym")),
        raw_graph_path=(
            data_root / dcfg["semantic_graph_raw_file"]
            if dcfg.get("semantic_graph_raw_file")
            else None
        ),
    )
    sem_dim = int(z_sem_np.shape[1])
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
        use_mean_head=bool(mcfg.get("use_mean_head", False)),
        mean_head_hidden_dim=(
            int(mcfg["mean_head_hidden_dim"])
            if mcfg.get("mean_head_hidden_dim") is not None
            else None
        ),
    ).to(device)
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
    tensors = {
        "a_phy": to_torch(a_phy_np, device=device),
        "a_sem": to_torch(a_sem_np, device=device),
        "z_sem": torch.tensor(z_sem_np, dtype=torch.float32, device=device),
    }
    return artifacts, model, process, sampler, tensors, dynamic_bank


def _make_cond(batch, tensors: Dict[str, torch.Tensor], device: torch.device, dynamic_bank=None):
    x_his = batch["x_his"].to(device=device, dtype=torch.float32)
    x_fut = batch["x_fut"].to(device=device, dtype=torch.float32)
    cutoff = batch["cutoff_step"].to(device=device, dtype=torch.long)
    _, _, n, _ = x_fut.shape
    if dynamic_bank is not None:
        z_sem = dynamic_bank.compose(
            static_z_sem=tensors["z_sem"],
            cutoff_steps=cutoff,
            num_nodes=n,
            device=device,
        )
    else:
        z_sem = tensors["z_sem"]
    cond = {"x_his": x_his, "a_phy": tensors["a_phy"], "a_sem": tensors["a_sem"], "z_sem": z_sem}
    return x_his, x_fut, cond


def _count_params(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _try_flops(model: torch.nn.Module, args: Tuple[torch.Tensor, ...]) -> Tuple[Optional[float], str]:
    """Try THOP/FVCore FLOPs for one forward call."""
    try:
        from thop import profile  # type: ignore

        flops, _ = profile(model, inputs=args, verbose=False)
        return float(flops), "thop"
    except Exception as thop_exc:
        try:
            from fvcore.nn import FlopCountAnalysis  # type: ignore

            flops = FlopCountAnalysis(model, args).total()
            return float(flops), "fvcore"
        except Exception as fv_exc:
            return None, f"unavailable: thop={type(thop_exc).__name__}, fvcore={type(fv_exc).__name__}"


def _measure_train_epoch(
    model: PromptSTDiff,
    process: DiffusionProcess,
    loader,
    tensors: Dict[str, torch.Tensor],
    device: torch.device,
    max_batches: int,
    dynamic_bank=None,
) -> Tuple[float, int, int]:
    """Measure partial epoch training time and peak memory."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    seen = 0
    for batch in loader:
        _, x_fut, cond = _make_cond(batch, tensors, device=device, dynamic_bank=dynamic_bank)
        b, h, n, f = x_fut.shape
        t = torch.randint(0, process.num_steps, (b,), device=device, dtype=torch.long)
        noise = torch.randn((b, h, n, f), device=device, dtype=x_fut.dtype)
        x_t = process.q_sample(x_start=x_fut, t=t, noise=noise)
        optimizer.zero_grad(set_to_none=True)
        eps = model(x_t=x_t, t=t, x_his=cond["x_his"], a_phy=cond["a_phy"], a_sem=cond["a_sem"], z_sem=cond["z_sem"])
        loss = torch.nn.functional.mse_loss(eps, noise)
        loss.backward()
        optimizer.step()
        seen += 1
        if seen >= max_batches:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        peak = 0
    elapsed = time.perf_counter() - t0
    total_batches = len(loader)
    epoch_est = elapsed * total_batches / max(1, seen)
    return epoch_est, total_batches, peak


@torch.no_grad()
def _latency_ms_per_sample(
    model: PromptSTDiff,
    sampler: DiffusionSampler,
    loader,
    tensors: Dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
    warmup: int,
    runs: int,
    dynamic_bank=None,
) -> float:
    """Measure inference latency for sampler.sample."""
    model.eval()
    batch = next(iter(loader))
    if batch["x_his"].shape[0] > batch_size:
        batch = {k: v[:batch_size] for k, v in batch.items()}
    _, x_fut, cond = _make_cond(batch, tensors, device=device, dynamic_bank=dynamic_bank)
    b, h, n, f = x_fut.shape
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


def _write_rows(rows: List[Dict[str, object]], csv_path: Path, md_path: Path) -> None:
    """Write benchmark rows as CSV and markdown."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(fieldnames) + " |", "|" + "|".join(["---"] * len(fieldnames)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[k]) for k in fieldnames) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run efficiency benchmark."""
    args = parse_args()
    config = load_config(args.config)
    device_arg = args.device
    if args.gpu_id is not None:
        device_arg = f"cuda:{int(args.gpu_id)}"
    device = get_device(device_arg)

    artifacts, model, process, sampler, tensors, dynamic_bank = _load_runtime(config, device)
    batch = next(iter(artifacts.train_loader))
    x_his, x_fut, cond = _make_cond(batch, tensors, device=device, dynamic_bank=dynamic_bank)
    b, h, n, f = x_fut.shape
    t = torch.zeros((b,), dtype=torch.long, device=device)
    x_t = torch.randn((b, h, n, f), device=device)
    flops, flops_backend = _try_flops(model, (x_t, t, x_his, cond["a_phy"], cond["a_sem"], cond["z_sem"]))
    epoch_time, total_batches, peak_memory = _measure_train_epoch(
        model=model,
        process=process,
        loader=artifacts.train_loader,
        tensors=tensors,
        device=device,
        max_batches=max(1, int(args.max_train_batches)),
        dynamic_bank=dynamic_bank,
    )
    realistic_bs = int(args.realistic_batch_size or config["dataset"]["batch_size"])
    latency_b1 = _latency_ms_per_sample(
        model=model,
        sampler=sampler,
        loader=artifacts.test_loader,
        tensors=tensors,
        device=device,
        batch_size=1,
        warmup=int(args.latency_warmup),
        runs=int(args.latency_runs),
        dynamic_bank=dynamic_bank,
    )
    latency_real = _latency_ms_per_sample(
        model=model,
        sampler=sampler,
        loader=artifacts.test_loader,
        tensors=tensors,
        device=device,
        batch_size=realistic_bs,
        warmup=int(args.latency_warmup),
        runs=int(args.latency_runs),
        dynamic_bank=dynamic_bank,
    )
    rows: List[Dict[str, object]] = [
        {
            "model": "Prompt-STDiff",
            "status": "available",
            "params": _count_params(model),
            "flops_forward": "" if flops is None else f"{flops:.0f}",
            "flops_backend": flops_backend,
            "train_time_epoch_sec_est": f"{epoch_time:.4f}",
            "train_time_total_sec_est": f"{epoch_time * int(config['train']['epochs']):.4f}",
            "train_batches": total_batches,
            "peak_gpu_mem_mb": f"{peak_memory / (1024 ** 2):.2f}" if peak_memory else "cpu_or_unavailable",
            "latency_b1_ms_per_sample": f"{latency_b1:.4f}",
            "latency_breal_ms_per_sample": f"{latency_real:.4f}",
            "sampler": config["diffusion"].get("sampler", "ddpm"),
            "sampling_steps": int(config["diffusion"].get("sampling_steps", config["diffusion"]["num_steps"])),
        }
    ]
    for name in BASELINES:
        rows.append(
            {
                "model": name,
                "status": "missing_in_repo",
                "params": "",
                "flops_forward": "",
                "flops_backend": "not measured",
                "train_time_epoch_sec_est": "",
                "train_time_total_sec_est": "",
                "train_batches": "",
                "peak_gpu_mem_mb": "",
                "latency_b1_ms_per_sample": "",
                "latency_breal_ms_per_sample": "",
                "sampler": "",
                "sampling_steps": "",
            }
        )
    _write_rows(rows, csv_path=args.out_csv, md_path=args.out_md)
    print(json.dumps({"rows": rows, "csv": str(args.out_csv), "md": str(args.out_md)}, indent=2))


if __name__ == "__main__":
    main()
