"""Traffic dataset and dataloader builders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataio.scalers import StandardScaler
from dataio.split import split_time_series
from dataio.windowing import build_window_indices


@dataclass
class DatasetArtifacts:
    """Artifacts returned by dataset preparation."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    scaler: Optional[StandardScaler]
    raw_data: np.ndarray


class TrafficWindowDataset(Dataset):
    """Windowed traffic dataset.

    Each item returns:
    - x_his: [T, N, F]
    - x_fut: [H, N, F]
    - cutoff_step: scalar, global forecast-start step index for strict temporal truncation.
    """

    def __init__(
        self,
        data: np.ndarray,
        history_steps: int,
        horizon_steps: int,
        base_step: int = 0,
    ) -> None:
        """Initialize the dataset.

        Args:
            data: Traffic data in shape [T_total, N, F].
            history_steps: History length T.
            horizon_steps: Forecast horizon H.
            base_step: Global start step offset for this split.
        """
        if data.ndim != 3:
            raise ValueError(f"Expected data shape [T, N, F], got {data.shape}.")

        self.data = data.astype(np.float32)
        self.history_steps = history_steps
        self.horizon_steps = horizon_steps
        self.base_step = int(base_step)
        self.indices = build_window_indices(
            total_steps=data.shape[0],
            history_steps=history_steps,
            horizon_steps=horizon_steps,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start, his_end, fut_end = self.indices[idx]
        x_his = self.data[start:his_end]  # [T, N, F]
        x_fut = self.data[his_end:fut_end]  # [H, N, F]
        cutoff_step = self.base_step + his_end

        return {
            "x_his": torch.from_numpy(x_his),
            "x_fut": torch.from_numpy(x_fut),
            "cutoff_step": torch.tensor(cutoff_step, dtype=torch.long),
        }


def load_traffic_array(data_path: Path) -> np.ndarray:
    """Load traffic array from npz/npy.

    Supported format:
    - npz with keys in priority: data, x, arr_0
    - npy as direct [T, N, F]
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
        # ASSUMPTION: if data is [T, N], expand to single feature channel.
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Expected [T, N, F], got {arr.shape}")
    return arr.astype(np.float32)


def build_dataloaders(config: Dict) -> DatasetArtifacts:
    """Build train/val/test dataloaders from config."""
    dcfg = config["dataset"]
    supported = dcfg.get("pems_supported", [])
    if supported and dcfg["name"] not in supported:
        raise ValueError(f"Unsupported dataset {dcfg['name']}. Supported: {supported}")
    data_root = Path(dcfg["data_root"]) / dcfg["name"]
    data_path = data_root / dcfg["data_file"]

    data = load_traffic_array(data_path)
    expected_input_dim = int(dcfg["input_dim"])
    if data.shape[-1] < expected_input_dim:
        raise ValueError(
            f"input_dim={expected_input_dim} but loaded data feature dim is {data.shape[-1]}."
        )
    if data.shape[-1] > expected_input_dim:
        # ASSUMPTION: use first `input_dim` channels when raw data has extra features.
        data = data[..., :expected_input_dim]

    (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = split_time_series(
        total_steps=data.shape[0],
        train_ratio=float(dcfg["train_ratio"]),
        val_ratio=float(dcfg["val_ratio"]),
        test_ratio=float(dcfg["test_ratio"]),
    )

    scaler: Optional[StandardScaler]
    if dcfg.get("scaler", "standard") == "standard":
        scaler_mode = str(dcfg.get("scaler_mode", "per_node"))
        scaler = StandardScaler.fit(data[tr_s:tr_e], mode=scaler_mode)
        data_scaled = scaler.transform(data)
    else:
        scaler = None
        data_scaled = data

    history_steps = int(dcfg["history_steps"])
    horizon_steps = int(dcfg["horizon_steps"])

    # Build split arrays first, then window each split to avoid split leakage.
    train_data = data_scaled[tr_s:tr_e]
    val_data = data_scaled[va_s:va_e]
    test_data = data_scaled[te_s:te_e]

    train_set = TrafficWindowDataset(train_data, history_steps, horizon_steps, base_step=tr_s)
    val_set = TrafficWindowDataset(val_data, history_steps, horizon_steps, base_step=va_s)
    test_set = TrafficWindowDataset(test_data, history_steps, horizon_steps, base_step=te_s)

    batch_size = int(dcfg["batch_size"])
    num_workers = int(dcfg.get("num_workers", 0))
    train_shuffle = bool(dcfg.get("train_shuffle", False))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return DatasetArtifacts(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler=scaler,
        raw_data=data,
    )
