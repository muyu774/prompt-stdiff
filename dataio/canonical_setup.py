"""Canonical split/scaler export for external baseline fairness.

External baselines must consume the exported windows and scaler statistics from
this module instead of rebuilding their own splits. This keeps Prompt-STDiff,
official baseline repos, and reimplemented controls on the same chronological
6:2:2 protocol and the same training-only scaler.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np

from dataio.scalers import StandardScaler
from dataio.split import split_time_series
from dataio.windowing import build_window_indices


SplitRange = Tuple[int, int]


@dataclass
class CanonicalSetupMetadata:
    """Metadata describing one canonical baseline export."""

    dataset: str
    data_file: str
    raw_shape: Tuple[int, int, int]
    used_shape: Tuple[int, int, int]
    input_dim: int
    history_steps: int
    horizon_steps: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    scaler: str
    scaler_mode: str
    train_range: SplitRange
    val_range: SplitRange
    test_range: SplitRange
    train_windows: int
    val_windows: int
    test_windows: int
    window_columns: Tuple[str, str, str]
    note: str


@dataclass
class CanonicalSetup:
    """In-memory canonical split/scaler artifacts."""

    metadata: CanonicalSetupMetadata
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    train_windows: np.ndarray
    val_windows: np.ndarray
    test_windows: np.ndarray


def load_canonical_traffic_array(data_path: Path) -> np.ndarray:
    """Load traffic array with the same file semantics as the main dataloader.

    This local NumPy-only loader avoids importing PyTorch when exporting
    canonical setup files for external repositories.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Traffic file not found: {data_path}")

    if data_path.suffix == ".npz":
        bundle = np.load(data_path)
        for key in ("data", "x", "arr_0"):
            if key in bundle:
                arr = bundle[key]
                break
        else:
            raise KeyError(f"No supported key found in {data_path}. keys={bundle.files}")
    elif data_path.suffix == ".npy":
        arr = np.load(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    if arr.ndim == 2:
        # Keep identical behavior to dataio.traffic_dataset.load_traffic_array.
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Expected [T, N, F], got {arr.shape}")
    return arr.astype(np.float32)


def _globalize_windows(local_windows: Iterable[Tuple[int, int, int]], base_step: int) -> np.ndarray:
    """Convert split-local windows to global [his_start, his_end, fut_end]."""
    rows = []
    for start, his_end, fut_end in local_windows:
        rows.append((base_step + start, base_step + his_end, base_step + fut_end))
    if not rows:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)


def build_canonical_setup(config: Mapping) -> CanonicalSetup:
    """Build canonical setup from the same config fields as build_dataloaders."""
    dcfg = config["dataset"]
    data_root = Path(str(dcfg["data_root"])) / str(dcfg["name"])
    data_path = data_root / str(dcfg["data_file"])

    raw_data = load_canonical_traffic_array(data_path)
    input_dim = int(dcfg["input_dim"])
    if raw_data.shape[-1] < input_dim:
        raise ValueError(
            f"input_dim={input_dim} but loaded feature dim is {raw_data.shape[-1]}."
        )
    # Keep this identical to dataio.traffic_dataset.build_dataloaders.
    data = raw_data[..., :input_dim] if raw_data.shape[-1] > input_dim else raw_data

    train_range, val_range, test_range = split_time_series(
        total_steps=data.shape[0],
        train_ratio=float(dcfg["train_ratio"]),
        val_ratio=float(dcfg["val_ratio"]),
        test_ratio=float(dcfg["test_ratio"]),
    )
    tr_s, tr_e = train_range
    va_s, va_e = val_range
    te_s, te_e = test_range

    scaler_name = str(dcfg.get("scaler", "standard"))
    scaler_mode = str(dcfg.get("scaler_mode", "per_node"))
    if scaler_name != "standard":
        raise ValueError(
            "Canonical external-baseline export currently expects dataset.scaler=standard "
            f"to match Prompt-STDiff; got {scaler_name}."
        )
    scaler = StandardScaler.fit(data[tr_s:tr_e], mode=scaler_mode)

    history_steps = int(dcfg["history_steps"])
    horizon_steps = int(dcfg["horizon_steps"])
    train_windows = _globalize_windows(
        build_window_indices(tr_e - tr_s, history_steps, horizon_steps),
        base_step=tr_s,
    )
    val_windows = _globalize_windows(
        build_window_indices(va_e - va_s, history_steps, horizon_steps),
        base_step=va_s,
    )
    test_windows = _globalize_windows(
        build_window_indices(te_e - te_s, history_steps, horizon_steps),
        base_step=te_s,
    )

    metadata = CanonicalSetupMetadata(
        dataset=str(dcfg["name"]),
        data_file=str(data_path),
        raw_shape=tuple(int(x) for x in raw_data.shape),
        used_shape=tuple(int(x) for x in data.shape),
        input_dim=input_dim,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        train_ratio=float(dcfg["train_ratio"]),
        val_ratio=float(dcfg["val_ratio"]),
        test_ratio=float(dcfg["test_ratio"]),
        scaler=scaler_name,
        scaler_mode=scaler_mode,
        train_range=(int(tr_s), int(tr_e)),
        val_range=(int(va_s), int(va_e)),
        test_range=(int(te_s), int(te_e)),
        train_windows=int(train_windows.shape[0]),
        val_windows=int(val_windows.shape[0]),
        test_windows=int(test_windows.shape[0]),
        window_columns=("his_start", "his_end", "fut_end"),
        note=(
            "Windows are global step indices using inclusive-exclusive slices: "
            "x_his=data[his_start:his_end], x_fut=data[his_end:fut_end]. "
            "Scaler statistics are fit on the unnormalized training range only."
        ),
    )

    return CanonicalSetup(
        metadata=metadata,
        scaler_mean=scaler.mean.astype(np.float32),
        scaler_std=scaler.std.astype(np.float32),
        train_windows=train_windows,
        val_windows=val_windows,
        test_windows=test_windows,
    )


def save_canonical_setup(setup: CanonicalSetup, out_prefix: Path) -> Dict[str, Path]:
    """Save canonical setup as NPZ plus JSON metadata.

    Args:
        setup: In-memory canonical setup.
        out_prefix: Output path without extension or with a desired stem.

    Returns:
        Mapping with ``npz`` and ``json`` paths.
    """
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = out_prefix.with_suffix(".npz")
    json_path = out_prefix.with_suffix(".json")

    np.savez(
        npz_path,
        scaler_mean=setup.scaler_mean,
        scaler_std=setup.scaler_std,
        train_windows=setup.train_windows,
        val_windows=setup.val_windows,
        test_windows=setup.test_windows,
        split_ranges=np.asarray(
            [
                setup.metadata.train_range,
                setup.metadata.val_range,
                setup.metadata.test_range,
            ],
            dtype=np.int64,
        ),
    )
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(setup.metadata), f, indent=2, sort_keys=True)
        f.write("\n")
    return {"npz": npz_path, "json": json_path}


def load_canonical_setup(npz_path: Path) -> CanonicalSetup:
    """Load a canonical setup exported by ``save_canonical_setup``."""
    json_path = npz_path.with_suffix(".json")
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not json_path.exists():
        raise FileNotFoundError(json_path)

    with json_path.open("r", encoding="utf-8") as f:
        meta_raw = json.load(f)
    bundle = np.load(npz_path)
    metadata = CanonicalSetupMetadata(
        dataset=str(meta_raw["dataset"]),
        data_file=str(meta_raw["data_file"]),
        raw_shape=tuple(int(x) for x in meta_raw["raw_shape"]),
        used_shape=tuple(int(x) for x in meta_raw["used_shape"]),
        input_dim=int(meta_raw["input_dim"]),
        history_steps=int(meta_raw["history_steps"]),
        horizon_steps=int(meta_raw["horizon_steps"]),
        train_ratio=float(meta_raw["train_ratio"]),
        val_ratio=float(meta_raw["val_ratio"]),
        test_ratio=float(meta_raw["test_ratio"]),
        scaler=str(meta_raw["scaler"]),
        scaler_mode=str(meta_raw["scaler_mode"]),
        train_range=tuple(int(x) for x in meta_raw["train_range"]),
        val_range=tuple(int(x) for x in meta_raw["val_range"]),
        test_range=tuple(int(x) for x in meta_raw["test_range"]),
        train_windows=int(meta_raw["train_windows"]),
        val_windows=int(meta_raw["val_windows"]),
        test_windows=int(meta_raw["test_windows"]),
        window_columns=tuple(str(x) for x in meta_raw["window_columns"]),
        note=str(meta_raw["note"]),
    )
    return CanonicalSetup(
        metadata=metadata,
        scaler_mean=bundle["scaler_mean"].astype(np.float32),
        scaler_std=bundle["scaler_std"].astype(np.float32),
        train_windows=bundle["train_windows"].astype(np.int64),
        val_windows=bundle["val_windows"].astype(np.int64),
        test_windows=bundle["test_windows"].astype(np.int64),
    )
