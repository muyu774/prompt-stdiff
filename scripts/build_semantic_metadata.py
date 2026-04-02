"""Build semantic metadata CSV from node mapping for prompt construction.

Use this script to create a richer metadata table consumed by semantic/offline_encoder.py.
Modes:
- template: create empty editable fields.
- weak_labels: derive coarse labels from graph degree (ASSUMPTION).
- external_match: deterministic merge from external static metadata table.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DIRECTIONS = ["northbound", "southbound", "eastbound", "westbound"]
SEMANTIC_COLUMNS = [
    "road_name",
    "road_type",
    "direction",
    "district",
    "functional_region",
    "poi_category",
    "traffic_pattern_hint",
]


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


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


def _merge_external_metadata(base_df: pd.DataFrame, external_csv: Path) -> pd.DataFrame:
    """Merge external static semantic metadata by node key.

    ASSUMPTION: external metadata provides deterministic node-level attributes from
    public road-network/POI processing pipeline.
    """
    ext = _normalize_columns(_safe_read_csv(external_csv))
    if ("node_index" not in ext.columns) and ("node_id" not in ext.columns):
        raise ValueError(
            "external_csv must contain either node_index or node_id for deterministic matching."
        )

    available_sem_cols = [c for c in SEMANTIC_COLUMNS if c in ext.columns]
    if not available_sem_cols:
        raise ValueError(
            f"external_csv has no semantic columns. Expected at least one of {SEMANTIC_COLUMNS}."
        )

    base = base_df.copy()
    if "node_index" in ext.columns:
        ext2 = ext[["node_index"] + available_sem_cols].copy()
        ext2["node_index"] = pd.to_numeric(ext2["node_index"], errors="coerce")
        base["node_index"] = pd.to_numeric(base["node_index"], errors="coerce")
        merged = base.merge(ext2, on="node_index", how="left", suffixes=("", "_ext"))
    else:
        ext2 = ext[["node_id"] + available_sem_cols].copy()
        ext2["node_id"] = ext2["node_id"].astype(str)
        base["node_id"] = base["node_id"].astype(str)
        merged = base.merge(ext2, on="node_id", how="left", suffixes=("", "_ext"))

    for col in SEMANTIC_COLUMNS:
        ext_col = f"{col}_ext"
        if ext_col in merged.columns:
            merged[col] = merged[ext_col].where(
                merged[ext_col].notna() & (merged[ext_col].astype(str).str.strip() != ""),
                merged.get(col, ""),
            )
            merged = merged.drop(columns=[ext_col])
        elif col not in merged.columns:
            merged[col] = ""

    return merged


def build_metadata(
    node_mapping_csv: Path,
    out_csv: Path,
    adjacency_csv: Path | None,
    mode: str,
    district_bins: int,
    external_csv: Path | None,
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
    elif mode == "template":
        # template mode: leave semantic slots blank for manual editing.
        df["road_type"] = ""
        df["functional_region"] = ""
        df["poi_category"] = ""
    elif mode == "external_match":
        if external_csv is None:
            raise ValueError("--external_csv is required when mode=external_match")
        df = _merge_external_metadata(base_df=df, external_csv=external_csv)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved semantic metadata template: {out_csv}")
    print(f"Rows: {len(df)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build semantic metadata CSV")
    parser.add_argument("--node_mapping_csv", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--adjacency_csv", type=Path, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="template",
        choices=["template", "weak_labels", "external_match"],
    )
    parser.add_argument(
        "--external_csv",
        type=Path,
        default=None,
        help="External static metadata table used when mode=external_match.",
    )
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
        external_csv=args.external_csv,
    )


if __name__ == "__main__":
    main()
