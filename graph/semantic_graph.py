"""Semantic graph construction from node embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np

from graph.graph_utils import add_self_loops, row_normalize, symmetrize

NormMode = Literal["none", "row", "sym"]


def cosine_similarity_matrix(z_sem: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute cosine similarity matrix.

    Args:
        z_sem: Semantic embeddings [N, D].

    Returns:
        Similarity matrix [N, N].
    """
    if z_sem.ndim != 2:
        raise ValueError(f"Expected z_sem [N, D], got {z_sem.shape}")
    if not np.isfinite(z_sem).all():
        raise ValueError("z_sem contains NaN/Inf values, cannot build semantic graph.")

    norms = np.linalg.norm(z_sem, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    zn = z_sem / norms
    return zn @ zn.T


def symmetric_normalize(adj: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Apply symmetric normalization D^{-1/2} A D^{-1/2}."""
    deg = adj.sum(axis=1)
    inv_sqrt = np.power(np.maximum(deg, eps), -0.5)
    return (inv_sqrt[:, None] * adj) * inv_sqrt[None, :]


def normalize_semantic_graph(adj: np.ndarray, mode: NormMode = "sym") -> np.ndarray:
    """Normalize semantic adjacency with selected mode."""
    if mode == "none":
        return adj.astype(np.float32)
    if mode == "row":
        return row_normalize(adj).astype(np.float32)
    if mode == "sym":
        return symmetric_normalize(adj).astype(np.float32)
    raise ValueError(f"Unsupported normalize mode: {mode}")


def build_topk_semantic_graph(
    z_sem: np.ndarray,
    top_k: int,
    add_loop: bool = True,
    clamp_nonnegative: bool = True,
    sym_mode: str = "max",
) -> np.ndarray:
    """Build semantic graph with cosine similarity + Top-K sparsification.

    Args:
        z_sem: Semantic embeddings [N, D].
        top_k: Number of neighbors retained per node.
        add_loop: Whether to add self-loops.
        clamp_nonnegative: Clamp negative similarities to zero.
        sym_mode: Symmetrization mode in {"max", "avg"}.

    Returns:
        Raw semantic adjacency [N, N].
    """
    sim = cosine_similarity_matrix(z_sem)
    n = sim.shape[0]
    top_k = max(1, min(top_k, n - 1))

    adj = np.zeros_like(sim, dtype=np.float32)
    for i in range(n):
        row = sim[i].copy()
        row[i] = -np.inf
        idx = np.argpartition(row, -top_k)[-top_k:]
        adj[i, idx] = sim[i, idx]

    if clamp_nonnegative:
        adj = np.maximum(adj, 0.0)

    adj = symmetrize(adj, mode=sym_mode)
    if add_loop:
        adj = add_self_loops(adj, value=1.0)
    return adj.astype(np.float32)


def semantic_graph_stats(adj: np.ndarray) -> Dict[str, float]:
    """Compute simple graph statistics for sanity check."""
    nonzero = (adj > 0).astype(np.float32)
    degree = nonzero.sum(axis=1)
    return {
        "nonzero_ratio": float(nonzero.mean()),
        "avg_degree": float(degree.mean()),
        "min_degree": float(degree.min()),
        "max_degree": float(degree.max()),
    }


def _is_valid_cached_graph(cached: np.ndarray, expected_n: int) -> bool:
    """Validate cached graph shape/values."""
    return cached.shape == (expected_n, expected_n) and np.isfinite(cached).all()


def load_or_build_semantic_graph(
    graph_path: Path,
    z_sem: np.ndarray,
    top_k: int,
    rebuild: bool = False,
    normalize_mode: NormMode = "sym",
    raw_graph_path: Optional[Path] = None,
) -> np.ndarray:
    """Load semantic adjacency from cache or build from embeddings.

    Args:
        graph_path: Path of returned graph (usually normalized graph).
        z_sem: Semantic embedding [N, D].
        top_k: Top-K neighbors.
        rebuild: Force rebuild.
        normalize_mode: Normalization mode for returned graph.
        raw_graph_path: Optional path to save/load raw graph.

    Returns:
        Semantic adjacency [N, N].
    """
    expected_n = int(z_sem.shape[0])
    if z_sem.ndim != 2:
        raise ValueError(f"Expected z_sem [N, D], got {z_sem.shape}")
    if not np.isfinite(z_sem).all():
        raise ValueError("z_sem contains NaN/Inf values, cannot build semantic graph.")

    if graph_path.exists() and not rebuild:
        cached = np.load(graph_path).astype(np.float32)
        if _is_valid_cached_graph(cached, expected_n):
            return cached
        rebuild = True

    raw_adj: Optional[np.ndarray] = None
    if raw_graph_path is not None and raw_graph_path.exists() and not rebuild:
        raw_cached = np.load(raw_graph_path).astype(np.float32)
        if _is_valid_cached_graph(raw_cached, expected_n):
            raw_adj = raw_cached

    if raw_adj is None:
        raw_adj = build_topk_semantic_graph(z_sem=z_sem, top_k=top_k)

    graph = normalize_semantic_graph(raw_adj, mode=normalize_mode)

    graph_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(graph_path, graph)

    if raw_graph_path is not None:
        raw_graph_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(raw_graph_path, raw_adj.astype(np.float32))

    return graph.astype(np.float32)
