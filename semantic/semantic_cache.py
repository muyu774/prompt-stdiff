"""Semantic embedding cache IO utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_semantic_embeddings(path: Path, z_sem: np.ndarray) -> None:
    """Save semantic embeddings to `.npy` file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, z_sem.astype(np.float32))


def load_semantic_embeddings(path: Path) -> np.ndarray:
    """Load semantic embeddings from `.npy` file."""
    if not path.exists():
        raise FileNotFoundError(f"Semantic embeddings not found: {path}")
    z_sem = np.load(path)
    if z_sem.ndim != 2:
        raise ValueError(f"Expected z_sem [N, D], got {z_sem.shape}")
    return z_sem.astype(np.float32)
