"""Physical graph construction and loading."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from graph.graph_utils import add_self_loops, row_normalize, symmetrize


def load_or_build_physical_graph(
    file_path: Path,
    num_nodes: int,
    sigma: Union[float, str] = "auto",
    add_loop: bool = True,
    normalize: bool = True,
    normalize_mode: str = "sym",
) -> np.ndarray:
    """Load physical adjacency from file.

    Supported file formats:
    - `.npy`: direct adjacency [N, N]
    - `.csv`: edge list with columns: src, dst, distance

    Notes:
    - `sigma="auto"` estimates Gaussian scale from positive distance statistics.
    - `normalize_mode="sym"` applies D^{-1/2} A D^{-1/2} (paper-aligned default).

    Returns:
        Physical adjacency matrix [N, N].
    """
    if not file_path.exists():
        # ASSUMPTION: if no physical graph is found, use identity adjacency.
        adj = np.eye(num_nodes, dtype=np.float32)
        return adj

    if file_path.suffix == ".npy":
        adj = np.load(file_path).astype(np.float32)
    elif file_path.suffix == ".csv":
        edges = pd.read_csv(file_path)
        if edges.shape[1] < 2:
            raise ValueError("CSV adjacency needs at least 2 columns: src,dst")
        src_col = edges.columns[0]
        dst_col = edges.columns[1]
        dist_col: Optional[str] = edges.columns[2] if edges.shape[1] >= 3 else None

        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        sigma_value: Optional[float] = None
        if dist_col is not None:
            valid_dist = pd.to_numeric(edges[dist_col], errors="coerce").to_numpy()
            valid_dist = valid_dist[np.isfinite(valid_dist)]
            valid_dist = valid_dist[valid_dist > 0]
            if isinstance(sigma, str) and sigma.lower() == "auto":
                # ASSUMPTION: use distance std as Gaussian scale for Eq.(4) when sigma is unspecified.
                sigma_value = float(np.std(valid_dist)) if valid_dist.size > 0 else 1.0
            else:
                sigma_value = max(float(sigma), 1e-6)

        for _, row in edges.iterrows():
            s = int(row[src_col])
            d = int(row[dst_col])
            if dist_col is None:
                w = 1.0
            else:
                dist = float(row[dist_col])
                # ASSUMPTION: Gaussian kernel over edge distance.
                scale = sigma_value if sigma_value is not None else 1.0
                w = np.exp(-((dist / max(scale, 1e-6)) ** 2))
            adj[s, d] = max(adj[s, d], w)
    else:
        raise ValueError(f"Unsupported adjacency file: {file_path}")

    if adj.shape != (num_nodes, num_nodes):
        raise ValueError(f"Expected adjacency [{num_nodes},{num_nodes}], got {adj.shape}")

    adj = symmetrize(adj, mode="max")
    if add_loop:
        adj = add_self_loops(adj, value=1.0)
    if normalize:
        if normalize_mode == "row":
            adj = row_normalize(adj)
        elif normalize_mode == "sym":
            deg = adj.sum(axis=1)
            inv_sqrt = np.power(np.maximum(deg, 1e-8), -0.5)
            adj = (inv_sqrt[:, None] * adj) * inv_sqrt[None, :]
        else:
            raise ValueError(f"Unsupported normalize_mode: {normalize_mode}")
    return adj.astype(np.float32)
