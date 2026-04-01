"""Device utilities."""

from __future__ import annotations

import contextlib
from typing import ContextManager, Optional

import torch


def get_device(device: Optional[str] = None) -> torch.device:
    """Resolve execution device.

    Args:
        device: One of {"auto", "cpu", "cuda", "cuda:0", ...}. None equals "auto".

    Returns:
        torch.device
    """
    req = (device or "auto").strip().lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req == "cpu":
        return torch.device("cpu")
    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested device '{device}', but CUDA is not available.")
        return torch.device(device)
    raise ValueError(f"Unsupported device spec: {device}")


def autocast_context(enabled: bool) -> ContextManager:
    """Return autocast context manager wrapper."""
    if enabled and torch.cuda.is_available():
        return torch.cuda.amp.autocast()
    return contextlib.nullcontext()
