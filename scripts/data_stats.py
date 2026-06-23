"""Data-transparency stats for dynamic event availability."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataio.canonical_setup import load_canonical_traffic_array
from dataio.split import split_time_series
from dataio.windowing import build_window_indices
from utils.config import load_config


def _event_kind(name: str) -> str:
    """Map raw event type to coarse subset kind without importing torch modules."""
    s = str(name).lower()
    if "weather" in s or "rain" in s or "storm" in s:
        return "weather"
    if "incident" in s or "accident" in s or "collision" in s or "crash" in s or "roadwork" in s or "hazard" in s:
        return "incident"
    if "holiday" in s or "calendar" in s:
        return "calendar"
    if "poi" in s:
        return "poi"
    return "other"


def _event_type_names(bank_path: Path) -> List[str]:
    """Return event type names from bank or dynamic_events.csv."""
    bundle = np.load(bank_path, allow_pickle=True)
    if "event_type_vocab" in bundle:
        return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in bundle["event_type_vocab"].tolist()]
    events_csv = bank_path.parent / "dynamic_events.csv"
    if events_csv.exists():
        df = pd.read_csv(events_csv)
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        if "event_type" in df.columns and len(df) == int(bundle["step_idx"].shape[0]):
            return sorted(set(df["event_type"].fillna("unknown").astype(str).tolist()))
    return ["unknown"]


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Print dynamic event window-count tables")
    parser.add_argument("--config", action="append", required=True, help="Dataset config; repeatable")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/data_stats"))
    return parser.parse_args()


def _global_windows(total_steps: int, history: int, horizon: int, ratios: Tuple[float, float, float]):
    """Build global window arrays by split."""
    ranges = split_time_series(total_steps, *ratios)
    out = {}
    for name, (s, e) in zip(("train", "val", "test"), ranges):
        local = build_window_indices(e - s, history, horizon)
        out[name] = np.asarray([(s + a, s + b, s + c) for a, b, c in local], dtype=np.int64)
    return out, dict(zip(("train", "val", "test"), ranges))


def _window_event_counts(windows: np.ndarray, event_steps: np.ndarray, event_type: np.ndarray, type_names: List[str]):
    """Count forecast windows whose target horizon contains at least one event type."""
    rows = []
    for tid, name in enumerate(type_names):
        steps = event_steps[event_type == tid]
        if steps.size == 0:
            count = 0
        else:
            contains = np.zeros((windows.shape[0],), dtype=bool)
            for i, (_, his_end, fut_end) in enumerate(windows):
                contains[i] = bool(((steps >= his_end) & (steps < fut_end)).any())
            count = int(contains.sum())
        rows.append({"event_type": name, "event_kind": _event_kind(name), "window_count": count})
    return rows


def _non_overlap_report(windows_by_split: Dict[str, np.ndarray]) -> Dict[str, bool]:
    """Verify exact forecast windows do not overlap across split datasets."""
    sets = {k: set(map(tuple, v.tolist())) for k, v in windows_by_split.items()}
    return {
        "train_val_disjoint": sets["train"].isdisjoint(sets["val"]),
        "train_test_disjoint": sets["train"].isdisjoint(sets["test"]),
        "val_test_disjoint": sets["val"].isdisjoint(sets["test"]),
    }


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a small dataframe as GitHub-flavored markdown without tabulate."""
    if df.empty:
        return "_No rows._"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def _event_type_ids(bundle, bank_path: Path, n_events: int, type_names: List[str]) -> np.ndarray:
    """Load event_type_id from bank, with dynamic_events.csv fallback."""
    if "event_type_id" in bundle:
        return bundle["event_type_id"].astype(np.int64)
    events_csv = bank_path.parent / "dynamic_events.csv"
    if events_csv.exists():
        df = pd.read_csv(events_csv)
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        if "event_type" in df.columns and len(df) == int(n_events):
            lookup = {v: i for i, v in enumerate(type_names)}
            return np.asarray([lookup.get(str(v), 0) for v in df["event_type"].fillna("unknown").astype(str)], dtype=np.int64)
    return np.zeros((n_events,), dtype=np.int64)


def main() -> None:
    """Run stats for one or more configs."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    balance_rows = []

    for cfg_path in args.config:
        cfg = load_config(cfg_path)
        dcfg = cfg["dataset"]
        dataset = str(dcfg["name"])
        data_root = Path(dcfg["data_root"]) / dataset
        data = load_canonical_traffic_array(data_root / dcfg["data_file"])
        if data.shape[-1] > int(dcfg["input_dim"]):
            data = data[..., : int(dcfg["input_dim"])]

        windows, ranges = _global_windows(
            total_steps=data.shape[0],
            history=int(dcfg["history_steps"]),
            horizon=int(dcfg["horizon_steps"]),
            ratios=(float(dcfg["train_ratio"]), float(dcfg["val_ratio"]), float(dcfg["test_ratio"])),
        )
        overlap = _non_overlap_report(windows)

        bank_file = ((dcfg.get("dynamic_semantic", {}) or {}).get("bank_file", "dynamic_semantic_bank.npz"))
        bank_path = data_root / str(bank_file)
        if not bank_path.exists():
            print(f"[warn] {dataset}: missing dynamic bank {bank_path}; writing zero event stats")
            type_names = ["missing_bank"]
            event_steps = np.asarray([], dtype=np.int64)
            event_type = np.asarray([], dtype=np.int64)
        else:
            bundle = np.load(bank_path, allow_pickle=True)
            event_steps = bundle["step_idx"].astype(np.int64)
            type_names = _event_type_names(bank_path)
            event_type = _event_type_ids(bundle, bank_path=bank_path, n_events=len(event_steps), type_names=type_names)

        for split, win in windows.items():
            rows = _window_event_counts(win, event_steps, event_type, type_names)
            for row in rows:
                row.update(
                    {
                        "dataset": dataset,
                        "split": split,
                        "total_windows": int(win.shape[0]),
                        "window_ratio": float(row["window_count"]) / max(1, int(win.shape[0])),
                    }
                )
                all_rows.append(row)
            balance_rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "range_start": int(ranges[split][0]),
                    "range_end": int(ranges[split][1]),
                    "total_windows": int(win.shape[0]),
                    **overlap,
                }
            )

    counts_df = pd.DataFrame(all_rows)
    balance_df = pd.DataFrame(balance_rows)
    counts_csv = args.out_dir / "event_window_counts.csv"
    balance_csv = args.out_dir / "split_balance.csv"
    counts_df.to_csv(counts_csv, index=False)
    balance_df.to_csv(balance_csv, index=False)

    md = [
        "# Data Transparency Stats",
        "",
        "## Event Window Counts",
        _df_to_markdown(counts_df),
        "",
        "## Split Balance and Non-overlap",
        _df_to_markdown(balance_df),
        "",
    ]
    (args.out_dir / "data_stats.md").write_text("\n".join(md), encoding="utf-8")
    print(counts_df.to_string(index=False))
    print(balance_df.to_string(index=False))
    print(f"Saved {counts_csv}, {balance_csv}, {args.out_dir / 'data_stats.md'}")


if __name__ == "__main__":
    main()
