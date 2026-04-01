"""Graph utility functions for adjacency processing."""

from __future__ import annotations

import numpy as np
import torch


def add_self_loops(adj: np.ndarray, value: float = 1.0) -> np.ndarray:
    """Add self loops to adjacency matrix."""
    out = adj.copy()
    np.fill_diagonal(out, value)
    return out


def symmetrize(adj: np.ndarray, mode: str = "max") -> np.ndarray:
    """Symmetrize adjacency matrix.

    Args:
        adj: Input adjacency [N, N].
        mode: `max` or `avg`.
    """
    if mode == "max":
        return np.maximum(adj, adj.T)
    if mode == "avg":
        return 0.5 * (adj + adj.T)
    raise ValueError(f"Unsupported mode: {mode}")


def row_normalize(adj: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Apply row normalization to adjacency matrix."""
    deg = adj.sum(axis=1, keepdims=True)
    deg = np.where(deg < eps, 1.0, deg)
    return adj / deg


def to_torch(adj: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert numpy adjacency to float tensor on device."""
    return torch.tensor(adj, dtype=torch.float32, device=device)
