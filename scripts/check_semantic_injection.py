"""Smoke-check semantic injection adapters for fair baseline experiments.

The script intentionally does not train a baseline. It verifies the reusable
adapter contract used by external GWNet/AGCRN/PDFormer/DiffSTG/PriSTI/SpecSTG
wrappers:

- ``use_semantic=False`` returns the original input/condition objects.
- ``use_semantic=True`` composes semantics via the same cached Z_sem and
  ``cutoff_step`` path used by Prompt-STDiff.
- Deterministic adapters append semantic channels over the history axis.
- Diffusion adapters append projected semantics to the condition mapping.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataio.traffic_dataset import load_traffic_array
from semantic_injection import (
    BatchSemanticComposer,
    DeterministicSemanticInputAdapter,
    DiffusionSemanticConditionAdapter,
)
from semantic.semantic_cache import load_semantic_embeddings
from utils.config import load_config
from utils.device import get_device
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Check semantic injection adapters")
    parser.add_argument("--config", type=str, default="configs/pems04.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--d_proj", type=int, default=16)
    parser.add_argument("--batch_index", type=int, default=0)
    return parser.parse_args()


def _semantic_path(config: dict[str, Any]) -> Path:
    """Resolve static semantic cache path from dataset config."""
    dcfg = config["dataset"]
    return Path(dcfg["data_root"]) / dcfg["name"] / dcfg["semantic_embedding_file"]


def main() -> None:
    """Run adapter contract checks and print a compact report."""
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["train"]["seed"]))
    # Keep the smoke test portable in restricted CI/sandbox environments.
    device = get_device(args.device)

    dcfg = config["dataset"]
    data_path = Path(dcfg["data_root"]) / dcfg["name"] / dcfg["data_file"]
    data = load_traffic_array(data_path)
    input_dim = int(dcfg["input_dim"])
    data = data[..., :input_dim]
    history_steps = int(dcfg["history_steps"])
    horizon_steps = int(dcfg["horizon_steps"])
    start = int(args.batch_index)
    his_end = start + history_steps
    fut_end = his_end + horizon_steps
    if fut_end > data.shape[0]:
        raise ValueError(f"Not enough data for one window: fut_end={fut_end}, total={data.shape[0]}")
    batch = {
        "x_his": torch.from_numpy(data[start:his_end]).unsqueeze(0),
        "x_fut": torch.from_numpy(data[his_end:fut_end]).unsqueeze(0),
        "cutoff_step": torch.tensor([his_end], dtype=torch.long),
    }

    x_his = batch["x_his"].to(device=device, dtype=torch.float32)
    x_fut = batch["x_fut"].to(device=device, dtype=torch.float32)

    z_np = load_semantic_embeddings(_semantic_path(config))
    z_static = torch.tensor(z_np, dtype=torch.float32, device=device)
    composer = BatchSemanticComposer(static_z_sem=z_static)
    z_batch = composer.compose(batch=batch, device=device, num_nodes=x_his.shape[2])

    # Disabled deterministic adapter must be exact no-op.
    det_off = DeterministicSemanticInputAdapter(
        sem_dim=z_batch.shape[-1],
        d_proj=int(args.d_proj),
        use_semantic=False,
    ).to(device)
    x_off = det_off(x_his)
    assert x_off is x_his, "use_semantic=False must return the original x_his object."

    det_on = DeterministicSemanticInputAdapter(
        sem_dim=z_batch.shape[-1],
        d_proj=int(args.d_proj),
        use_semantic=True,
    ).to(device)
    x_on = det_on(x_his, z_batch=z_batch)
    expected_dim = int(x_his.shape[-1]) + int(args.d_proj)
    assert tuple(x_on.shape[:-1]) == tuple(x_his.shape[:-1])
    assert int(x_on.shape[-1]) == expected_dim

    cond = {"x_his": x_his, "x_fut": x_fut}
    diff_off = DiffusionSemanticConditionAdapter(
        sem_dim=z_batch.shape[-1],
        d_proj=int(args.d_proj),
        use_semantic=False,
    ).to(device)
    cond_off = diff_off(cond)
    assert cond_off is cond, "use_semantic=False must return the original condition mapping."

    diff_on = DiffusionSemanticConditionAdapter(
        sem_dim=z_batch.shape[-1],
        d_proj=int(args.d_proj),
        use_semantic=True,
        out_key="z_sem_proj",
    ).to(device)
    cond_on = diff_on(cond, z_batch=z_batch)
    assert "z_sem_proj" in cond_on
    assert tuple(cond_on["z_sem_proj"].shape) == (
        int(x_his.shape[0]),
        int(x_his.shape[2]),
        int(args.d_proj),
    )

    print("Semantic injection adapter check passed.")
    print(f"config={args.config}")
    print(f"x_his={tuple(x_his.shape)}")
    print(f"z_batch={tuple(z_batch.shape)}")
    print(f"deterministic_augmented={tuple(x_on.shape)}")
    print(f"diffusion_condition[z_sem_proj]={tuple(cond_on['z_sem_proj'].shape)}")


if __name__ == "__main__":
    main()
