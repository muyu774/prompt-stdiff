"""Build semantic metadata CSV from node mapping for prompt construction.

Use this script to create a richer metadata table consumed by semantic/offline_encoder.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DIRECTIONS = ["northbound", "southbound", "eastbound", "westbound"]


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _compute_degree(num_nodes: int, adjacency_csv: Path) -> np.ndarray:
    """Compute undirected degree from adjacency csv.

    ASSUMPTION: edges are treated as unweighted for weak semantic role labeling.
    """
    if not adjacency_csv.exists():
        return np.zeros((num_nodes,), dtype=np.float32)

    edges = pd.read_csv(adjacency_csv)
    if not set(["src", "dst"]).issubset(edges.columns):
        # try legacy columns
        edges.columns = [str(c).strip().lower() for c in edges.columns]
    if not set(["src", "dst"]).issubset(edges.columns):
        return np.zeros((num_nodes,), dtype=np.float32)

    deg = np.zeros((num_nodes,), dtype=np.float32)
    for _, row in edges.iterrows():
        try:
            s = int(row["src"])
            d = int(row["dst"])
        except Exception:
            continue
        if 0 <= s < num_nodes:
            deg[s] += 1
        if 0 <= d < num_nodes:
            deg[d] += 1
    return deg


def _label_by_degree(deg: np.ndarray) -> Dict[str, List[str]]:
    """Assign weak semantic labels from node degree.

    ASSUMPTION: high-degree sensors are likely corridor/backbone roads.
    """
    q1, q2, q3 = np.quantile(deg, [0.25, 0.5, 0.75])

    road_type: List[str] = []
    functional_region: List[str] = []
    poi_category: List[str] = []

    for d in deg:
        if d >= q3:
            road_type.append("highway")
            functional_region.append("commuter_corridor")
            poi_category.append("transport_hub")
        elif d >= q2:
            road_type.append("arterial")
            functional_region.append("mixed")
            poi_category.append("office")
        elif d >= q1:
            road_type.append("urban_road")
            functional_region.append("commercial")
            poi_category.append("shopping")
        else:
            road_type.append("suburban_road")
            functional_region.append("residential")
            poi_category.append("residential")

    return {
        "road_type": road_type,
        "functional_region": functional_region,
        "poi_category": poi_category,
    }


def build_metadata(
    node_mapping_csv: Path,
    out_csv: Path,
    adjacency_csv: Path | None,
    mode: str,
    district_bins: int,
) -> None:
    """Build output metadata table for semantic prompt generation."""
    df = _safe_read_csv(node_mapping_csv).copy()
    if "node_id" not in df.columns or "node_index" not in df.columns:
        raise ValueError("node_mapping.csv must contain columns: node_id,node_index")

    df = df.sort_values("node_index", kind="stable").reset_index(drop=True)
    n = len(df)

    df["road_name"] = ""
    df["direction"] = [DIRECTIONS[i % len(DIRECTIONS)] for i in range(n)]
    df["district"] = [f"zone_{i * district_bins // max(n, 1)}" for i in range(n)]
    df["traffic_pattern_hint"] = ""

    if mode == "weak_labels":
        if adjacency_csv is None:
            raise ValueError("--adjacency_csv is required when mode=weak_labels")
        deg = _compute_degree(num_nodes=n, adjacency_csv=adjacency_csv)
        labels = _label_by_degree(deg)
        for k, v in labels.items():
            df[k] = v
        df["traffic_pattern_hint"] = [
            "morning commuter traffic is common" if x == "commuter_corridor" else "traffic is relatively stable outside rush hour"
            for x in df["functional_region"].tolist()
        ]
    else:
        # template mode: leave semantic slots blank for manual editing.
        df["road_type"] = ""
        df["functional_region"] = ""
        df["poi_category"] = ""

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved semantic metadata template: {out_csv}")
    print(f"Rows: {len(df)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build semantic metadata CSV")
    parser.add_argument("--node_mapping_csv", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--adjacency_csv", type=Path, default=None)
    parser.add_argument("--mode", type=str, default="template", choices=["template", "weak_labels"])
    parser.add_argument("--district_bins", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_metadata(
        node_mapping_csv=args.node_mapping_csv,
        out_csv=args.out_csv,
        adjacency_csv=args.adjacency_csv,
        mode=args.mode,
        district_bins=int(args.district_bins),
    )


if __name__ == "__main__":
    main()
