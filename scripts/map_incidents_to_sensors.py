"""Map incident coordinates to sensor indices via geo radius and topology filtering.

Input incidents CSV requires:
- latitude
- longitude

Output keeps original columns and adds:
- node_index
- map_distance_m
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def _load_sensor_points(
    sensor_csv: Path,
    idx_col: str,
    lat_col: str,
    lon_col: str,
) -> pd.DataFrame:
    df = _normalize_columns(pd.read_csv(sensor_csv))
    i_col = idx_col.strip().lower()
    la_col = lat_col.strip().lower()
    lo_col = lon_col.strip().lower()
    if not {i_col, la_col, lo_col}.issubset(df.columns):
        raise ValueError(
            f"sensor_csv must include {i_col},{la_col},{lo_col}. available={list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "node_index": pd.to_numeric(df[i_col], errors="coerce"),
            "lat": pd.to_numeric(df[la_col], errors="coerce"),
            "lon": pd.to_numeric(df[lo_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["node_index", "lat", "lon"]).copy()
    out["node_index"] = out["node_index"].astype(int)
    out = out.drop_duplicates(subset=["node_index"], keep="first").reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid sensor rows found in sensor_csv.")
    return out


def _load_graph_neighbors(adjacency_csv: Optional[Path], num_nodes: int) -> Dict[int, List[int]]:
    if adjacency_csv is None or (not adjacency_csv.exists()):
        return {}
    edges = _normalize_columns(pd.read_csv(adjacency_csv))
    src_col = "src" if "src" in edges.columns else ("from" if "from" in edges.columns else None)
    dst_col = "dst" if "dst" in edges.columns else ("to" if "to" in edges.columns else None)
    if src_col is None or dst_col is None:
        return {}

    neighbors: Dict[int, List[int]] = {i: [] for i in range(int(num_nodes))}
    for _, row in edges.iterrows():
        s = pd.to_numeric(row[src_col], errors="coerce")
        d = pd.to_numeric(row[dst_col], errors="coerce")
        if pd.isna(s) or pd.isna(d):
            continue
        si = int(s)
        di = int(d)
        if 0 <= si < num_nodes and 0 <= di < num_nodes:
            neighbors[si].append(di)
            neighbors[di].append(si)
    return neighbors


def _bfs_within_hops(neighbors: Dict[int, List[int]], start: int, max_hops: int) -> set[int]:
    if max_hops < 0:
        return set(neighbors.keys())
    seen = {int(start)}
    frontier = {int(start)}
    for _ in range(int(max_hops)):
        nxt: set[int] = set()
        for u in frontier:
            for v in neighbors.get(u, []):
                if v not in seen:
                    seen.add(v)
                    nxt.add(v)
        if not nxt:
            break
        frontier = nxt
    return seen


def _haversine_m(lat: float, lon: float, lat_arr: np.ndarray, lon_arr: np.ndarray) -> np.ndarray:
    r = 6371000.0
    phi1 = np.deg2rad(lat)
    phi2 = np.deg2rad(lat_arr)
    dphi = np.deg2rad(lat_arr - lat)
    dlambda = np.deg2rad(lon_arr - lon)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 1e-12)))
    return r * c


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map incidents to sensors")
    parser.add_argument("--incidents_csv", type=Path, required=True)
    parser.add_argument("--sensor_csv", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--incident_lat_col", type=str, default="latitude")
    parser.add_argument("--incident_lon_col", type=str, default="longitude")
    parser.add_argument("--sensor_idx_col", type=str, default="node_index")
    parser.add_argument("--sensor_lat_col", type=str, default="latitude")
    parser.add_argument("--sensor_lon_col", type=str, default="longitude")
    parser.add_argument("--radius_m", type=float, default=2000.0)
    parser.add_argument("--adjacency_csv", type=Path, default=None)
    parser.add_argument("--topology_hops", type=int, default=3)
    parser.add_argument("--assign_global_if_unmapped", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    incidents = _normalize_columns(pd.read_csv(args.incidents_csv))
    sensors = _load_sensor_points(
        sensor_csv=args.sensor_csv,
        idx_col=args.sensor_idx_col,
        lat_col=args.sensor_lat_col,
        lon_col=args.sensor_lon_col,
    )

    ilat = args.incident_lat_col.strip().lower()
    ilon = args.incident_lon_col.strip().lower()
    if ilat not in incidents.columns or ilon not in incidents.columns:
        raise ValueError(
            f"incidents_csv must contain {ilat},{ilon}. available={list(incidents.columns)}"
        )

    node_arr = sensors["node_index"].to_numpy(dtype=np.int64)
    lat_arr = sensors["lat"].to_numpy(dtype=np.float64)
    lon_arr = sensors["lon"].to_numpy(dtype=np.float64)
    neighbors = _load_graph_neighbors(
        adjacency_csv=args.adjacency_csv,
        num_nodes=int(node_arr.max()) + 1,
    )

    out_rows: List[dict] = []
    for _, row in incidents.iterrows():
        lat = pd.to_numeric(row.get(ilat), errors="coerce")
        lon = pd.to_numeric(row.get(ilon), errors="coerce")
        if pd.isna(lat) or pd.isna(lon):
            continue

        dist = _haversine_m(float(lat), float(lon), lat_arr=lat_arr, lon_arr=lon_arr)
        in_radius = dist <= float(args.radius_m)
        mapped_nodes = node_arr[in_radius]

        if mapped_nodes.size > 0 and neighbors:
            anchor = int(node_arr[int(np.argmin(dist))])
            allowed = _bfs_within_hops(
                neighbors=neighbors,
                start=anchor,
                max_hops=int(args.topology_hops),
            )
            mapped_nodes = np.asarray([n for n in mapped_nodes.tolist() if int(n) in allowed], dtype=np.int64)

        if mapped_nodes.size == 0 and bool(args.assign_global_if_unmapped):
            r = row.to_dict()
            r["node_index"] = -1
            r["map_distance_m"] = np.nan
            out_rows.append(r)
            continue

        for node in mapped_nodes.tolist():
            node = int(node)
            node_dist = float(dist[np.where(node_arr == node)[0][0]])
            r = row.to_dict()
            r["node_index"] = node
            r["map_distance_m"] = node_dist
            out_rows.append(r)

    out_df = pd.DataFrame(out_rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Saved mapped incidents: {args.out_csv}")
    print(f"rows={len(out_df)}")


if __name__ == "__main__":
    main()
