"""Build semantic graph files from semantic embeddings.

Outputs:
- raw adjacency (A_sem)
- normalized adjacency (A_sem_norm)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph.semantic_graph import (
    build_topk_semantic_graph,
    normalize_semantic_graph,
    semantic_graph_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build semantic graph from embeddings")
    parser.add_argument("--z_sem", type=Path, required=True, help="Path to semantic_embeddings.npy")
    parser.add_argument("--a_sem_out", type=Path, required=True, help="Path to save raw semantic adjacency")
    parser.add_argument(
        "--a_sem_norm_out",
        type=Path,
        required=True,
        help="Path to save normalized semantic adjacency",
    )
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument(
        "--norm_mode",
        type=str,
        default="sym",
        choices=["none", "row", "sym"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    z_sem = np.load(args.z_sem).astype(np.float32)
    a_sem = build_topk_semantic_graph(z_sem=z_sem, top_k=int(args.top_k))
    a_sem_norm = normalize_semantic_graph(a_sem, mode=args.norm_mode)

    args.a_sem_out.parent.mkdir(parents=True, exist_ok=True)
    args.a_sem_norm_out.parent.mkdir(parents=True, exist_ok=True)

    np.save(args.a_sem_out, a_sem)
    np.save(args.a_sem_norm_out, a_sem_norm)

    raw_stats = semantic_graph_stats(a_sem)
    norm_stats = semantic_graph_stats(a_sem_norm)

    print(f"Saved raw graph: {args.a_sem_out} shape={a_sem.shape}")
    print(f"Saved norm graph: {args.a_sem_norm_out} shape={a_sem_norm.shape}")
    print("Raw stats:", raw_stats)
    print("Norm stats:", norm_stats)


if __name__ == "__main__":
    main()
