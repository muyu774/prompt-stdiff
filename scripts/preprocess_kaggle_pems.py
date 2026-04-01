"""Preprocess Kaggle PEMS raw tables into Prompt-STDiff training format.

This script converts long-format traffic tables into dense tensor `data.npz` with shape [T, N, F].
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TIME_CANDIDATES = [
    "timestamp",
    "time",
    "datetime",
    "date_time",
    "date",
]
NODE_CANDIDATES = [
    "node_id",
    "sensor_id",
    "station_id",
    "detector_id",
    "id",
    "node",
    "sensor",
    "station",
]
FEATURE_CANDIDATES = [
    "flow",
    "traffic_flow",
    "speed",
    "occupancy",
    "volume",
]

EDGE_SRC_CANDIDATES = ["from", "src", "source"]
EDGE_DST_CANDIDATES = ["to", "dst", "target"]
EDGE_WEIGHT_CANDIDATES = ["distance", "cost", "weight"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase snake-style."""
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def pick_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """Pick first matched column from candidates."""
    col_set = set(columns)
    for c in candidates:
        if c in col_set:
            return c
    return None


def infer_feature_cols(df: pd.DataFrame, exclude: Sequence[str], max_features: int = 3) -> List[str]:
    """Infer numeric feature columns.

    ASSUMPTION: if canonical feature names are absent, use remaining numeric columns.
    """
    exclude_set = set(exclude)

    named = [c for c in FEATURE_CANDIDATES if c in df.columns and c not in exclude_set]
    if named:
        return named[:max_features]

    numeric_cols = [
        c
        for c in df.columns
        if c not in exclude_set and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        raise ValueError("No numeric feature columns found.")
    return numeric_cols[:max_features]


def load_long_table(path: Path) -> pd.DataFrame:
    """Load one CSV/Parquet table."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file: {path}")
    return normalize_columns(df)


def long_to_dense_tensor(
    df: pd.DataFrame,
    time_col: str,
    node_col: str,
    feature_cols: Sequence[str],
    freq: Optional[str] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Convert long table to dense tensor [T, N, F].

    Returns:
        - data: np.ndarray [T, N, F]
        - node_ids: sorted node IDs
        - features: feature column names
    """
    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col, node_col])

    node_ids = sorted(work[node_col].astype(str).unique().tolist())
    node_order = {nid: i for i, nid in enumerate(node_ids)}
    work["_node_idx"] = work[node_col].astype(str).map(node_order)

    if freq is not None:
        # ASSUMPTION: raw data can be aggregated to fixed frequency by mean.
        work = (
            work.set_index(time_col)
            .groupby("_node_idx")[list(feature_cols)]
            .resample(freq)
            .mean()
            .reset_index()
        )

    time_values = np.sort(work[time_col].dropna().unique())
    time_order = {t: i for i, t in enumerate(time_values)}
    work["_time_idx"] = work[time_col].map(time_order)

    t_len = len(time_values)
    n_len = len(node_ids)
    f_len = len(feature_cols)

    dense = np.full((t_len, n_len, f_len), np.nan, dtype=np.float32)
    for fi, feat in enumerate(feature_cols):
        sub = work[["_time_idx", "_node_idx", feat]].dropna()
        dense[sub["_time_idx"].to_numpy(dtype=np.int64), sub["_node_idx"].to_numpy(dtype=np.int64), fi] = sub[
            feat
        ].to_numpy(dtype=np.float32)

    # Fill missing values by temporal forward/backward fill per node-feature.
    tensor = pd.DataFrame(
        dense.reshape(t_len, n_len * f_len),
        index=pd.to_datetime(time_values),
    )
    tensor = tensor.ffill().bfill().fillna(0.0)
    dense = tensor.to_numpy(dtype=np.float32).reshape(t_len, n_len, f_len)
    return dense, node_ids, list(feature_cols)


def find_packaged_npz(raw_root: Path, split_name: str) -> Optional[Path]:
    """Find packaged npz file for one split (e.g., PEMS03.npz)."""
    key = split_name.lower()
    candidates = sorted(raw_root.rglob("*.npz"))
    matched = [p for p in candidates if key in p.stem.lower() or key in p.parent.name.lower()]
    return matched[0] if matched else None


def load_packaged_tensor(npz_path: Path, feature_indices: Optional[Sequence[int]] = None) -> np.ndarray:
    """Load packaged traffic tensor from npz and optionally select feature channels."""
    bundle = np.load(npz_path)
    if "data" in bundle:
        arr = bundle["data"]
    elif "x" in bundle:
        arr = bundle["x"]
    elif "arr_0" in bundle:
        arr = bundle["arr_0"]
    else:
        raise KeyError(f"No supported key in {npz_path}. keys={bundle.files}")

    if arr.ndim == 2:
        # ASSUMPTION: [T, N] is expanded to one feature channel.
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Expected packaged data [T,N,F], got {arr.shape} from {npz_path}")

    arr = arr.astype(np.float32)
    if feature_indices:
        arr = arr[:, :, list(feature_indices)]
    return arr


def load_node_ids_from_txt(txt_path: Path) -> Optional[List[str]]:
    """Load node IDs from txt file when provided by raw dataset."""
    if not txt_path.exists():
        return None
    lines = [x.strip() for x in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    node_ids = [x for x in lines if x]
    return node_ids if node_ids else None


def _build_id_to_idx(node_ids: Sequence[str]) -> Dict[str, int]:
    """Build robust string/int key mapping to contiguous node index."""
    mapping: Dict[str, int] = {}
    for idx, nid in enumerate(node_ids):
        nid_str = str(nid)
        mapping[nid_str] = idx
        try:
            mapping[str(int(float(nid_str)))] = idx
        except (TypeError, ValueError):
            pass
    return mapping


def _map_node_value(v: object, id_to_idx: Dict[str, int], num_nodes: int) -> Optional[int]:
    """Map node ID value to contiguous index."""
    key = str(v)
    if key in id_to_idx:
        return id_to_idx[key]

    try:
        iv = int(float(key))
    except (TypeError, ValueError):
        return None

    if str(iv) in id_to_idx:
        return id_to_idx[str(iv)]

    if 0 <= iv < num_nodes:
        return iv
    if 1 <= iv <= num_nodes:
        # ASSUMPTION: if IDs are 1-based contiguous indices, convert to 0-based.
        return iv - 1
    return None


def convert_edge_table_to_adjacency(
    edge_csv: Path,
    out_adj_csv: Path,
    node_ids: Optional[Sequence[str]],
    num_nodes: int,
) -> None:
    """Convert edge csv (from/to/distance|cost) to standardized adjacency.csv."""
    if not edge_csv.exists():
        pd.DataFrame(columns=["src", "dst", "distance"]).to_csv(out_adj_csv, index=False)
        return

    df = normalize_columns(pd.read_csv(edge_csv))
    src_col = pick_column(df.columns, EDGE_SRC_CANDIDATES)
    dst_col = pick_column(df.columns, EDGE_DST_CANDIDATES)
    if src_col is None or dst_col is None:
        pd.DataFrame(columns=["src", "dst", "distance"]).to_csv(out_adj_csv, index=False)
        return

    weight_col = pick_column(df.columns, EDGE_WEIGHT_CANDIDATES)
    id_to_idx = _build_id_to_idx(node_ids) if node_ids is not None else {}

    rows = []
    for _, row in df.iterrows():
        s = _map_node_value(row[src_col], id_to_idx=id_to_idx, num_nodes=num_nodes)
        d = _map_node_value(row[dst_col], id_to_idx=id_to_idx, num_nodes=num_nodes)
        if s is None or d is None:
            continue

        if weight_col is None:
            w = 1.0
        else:
            try:
                w = float(row[weight_col])
            except (TypeError, ValueError):
                w = 1.0
        rows.append((s, d, w))

    out = pd.DataFrame(rows, columns=["src", "dst", "distance"])
    out.to_csv(out_adj_csv, index=False)


def preprocess_packaged_split(
    npz_path: Path,
    out_dir: Path,
    feature_indices: Optional[Sequence[int]],
) -> None:
    """Preprocess packaged split where npz provides [T,N,F] and csv provides edges."""
    data = load_packaged_tensor(npz_path=npz_path, feature_indices=feature_indices)
    _, n_len, f_len = data.shape

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "data.npz", data=data)

    txt_path = npz_path.with_suffix(".txt")
    csv_path = npz_path.with_suffix(".csv")
    node_ids = load_node_ids_from_txt(txt_path)
    if node_ids is None or len(node_ids) != n_len:
        # ASSUMPTION: if txt IDs are missing/misaligned, use contiguous node indices.
        node_ids = [str(i) for i in range(n_len)]

    pd.DataFrame({"node_id": node_ids, "node_index": np.arange(n_len)}).to_csv(
        out_dir / "node_mapping.csv", index=False
    )
    pd.DataFrame({"feature": [f"feature_{i}" for i in range(f_len)]}).to_csv(
        out_dir / "features.csv", index=False
    )

    convert_edge_table_to_adjacency(
        edge_csv=csv_path,
        out_adj_csv=out_dir / "adjacency.csv",
        node_ids=node_ids,
        num_nodes=n_len,
    )
    print(f"[OK] {npz_path} -> {out_dir / 'data.npz'} | shape={data.shape}")


def find_split_file(raw_root: Path, split_name: str) -> Path:
    """Find one long-table file for a split name by keyword match."""
    candidates = sorted([p for p in raw_root.rglob("*") if p.suffix.lower() in {".csv", ".parquet", ".pq"}])
    key = split_name.lower()

    matched = [p for p in candidates if key in p.stem.lower() or key in p.parent.name.lower()]
    if matched:
        return matched[0]

    raise FileNotFoundError(
        f"No raw file found for split '{split_name}' under {raw_root}. "
        "Please check raw paths or rename files/directories with split keywords."
    )


def preprocess_one_split(
    raw_file: Path,
    out_dir: Path,
    time_col: Optional[str],
    node_col: Optional[str],
    feature_cols: Optional[Sequence[str]],
    freq: Optional[str],
    max_features: int,
) -> None:
    """Preprocess one split and write outputs."""
    df = load_long_table(raw_file)

    tcol = time_col or pick_column(df.columns, TIME_CANDIDATES)
    ncol = node_col or pick_column(df.columns, NODE_CANDIDATES)
    if tcol is None or ncol is None:
        raise ValueError(
            f"Cannot infer time/node columns for {raw_file}. "
            f"columns={list(df.columns)}"
        )

    feats = list(feature_cols) if feature_cols else infer_feature_cols(df, exclude=[tcol, ncol], max_features=max_features)

    data, node_ids, feat_names = long_to_dense_tensor(
        df=df,
        time_col=tcol,
        node_col=ncol,
        feature_cols=feats,
        freq=freq,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "data.npz", data=data)

    pd.DataFrame({"node_id": node_ids, "node_index": np.arange(len(node_ids))}).to_csv(
        out_dir / "node_mapping.csv", index=False
    )
    pd.DataFrame({"feature": feat_names}).to_csv(out_dir / "features.csv", index=False)

    # Create placeholder adjacency if absent.
    adj_path = out_dir / "adjacency.csv"
    if not adj_path.exists():
        # ASSUMPTION: if no physical edges are available in raw files, start from identity graph.
        pd.DataFrame(columns=["src", "dst", "distance"]).to_csv(adj_path, index=False)

    print(f"[OK] {raw_file} -> {out_dir / 'data.npz'} | shape={data.shape}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Preprocess Kaggle PEMS raw dataset")
    parser.add_argument("--raw_root", type=Path, required=True, help="Raw dataset root (unzipped kaggle folder)")
    parser.add_argument("--out_root", type=Path, default=Path("data"), help="Output root for pems03/pems04/pems08")
    parser.add_argument(
        "--splits",
        type=str,
        default="pems03,pems04,pems08",
        help="Comma separated split names to process",
    )
    parser.add_argument("--time_col", type=str, default=None)
    parser.add_argument("--node_col", type=str, default=None)
    parser.add_argument("--feature_cols", type=str, default=None, help="Comma separated feature columns")
    parser.add_argument(
        "--feature_indices",
        type=str,
        default=None,
        help="Optional feature channel indices for packaged npz, e.g. 0 or 0,1",
    )
    parser.add_argument("--freq", type=str, default=None, help="Optional resample frequency, e.g. 5min")
    parser.add_argument("--max_features", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """CLI entry."""
    args = parse_args()

    feature_cols = None
    if args.feature_cols:
        feature_cols = [x.strip().lower() for x in args.feature_cols.split(",") if x.strip()]
    feature_indices = None
    if args.feature_indices:
        feature_indices = [int(x.strip()) for x in args.feature_indices.split(",") if x.strip()]

    split_list = [x.strip().lower() for x in args.splits.split(",") if x.strip()]
    for split_name in split_list:
        out_dir = args.out_root / split_name
        packaged_npz = find_packaged_npz(args.raw_root, split_name)
        if packaged_npz is not None:
            preprocess_packaged_split(
                npz_path=packaged_npz,
                out_dir=out_dir,
                feature_indices=feature_indices,
            )
        else:
            raw_file = find_split_file(args.raw_root, split_name)
            preprocess_one_split(
                raw_file=raw_file,
                out_dir=out_dir,
                time_col=args.time_col.lower() if args.time_col else None,
                node_col=args.node_col.lower() if args.node_col else None,
                feature_cols=feature_cols,
                freq=args.freq,
                max_features=int(args.max_features),
            )


if __name__ == "__main__":
    main()
