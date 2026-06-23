"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    best_metric: float,
    config: Dict[str, Any],
) -> None:
    """Save training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> Tuple[int, float, Dict[str, Any]]:
    """Load checkpoint into model/optimizer.

    Returns:
        (epoch, best_metric, config)
    """
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=strict)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    epoch = int(ckpt.get("epoch", 0))
    best_metric = float(ckpt.get("best_metric", float("inf")))
    config = dict(ckpt.get("config", {}))
    return epoch, best_metric, config
