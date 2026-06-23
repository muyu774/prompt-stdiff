"""Smoke-check GWNet/AGCRN/PDFormer semantic wrappers.

This script uses tiny toy baseline modules to verify that the deterministic
semantic wrappers preserve expected input layouts and append semantic channels
only when enabled. It is intentionally independent of official baseline repos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.deterministic_semantic import (  # noqa: E402
    AGCRNSemanticWrapper,
    GWNetSemanticWrapper,
    PDFormerSemanticWrapper,
    adjusted_input_dim,
)
from semantic_injection import BatchSemanticComposer  # noqa: E402
from semantic.semantic_cache import load_semantic_embeddings  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.device import get_device  # noqa: E402


class ShapeRecorder(nn.Module):
    """Toy baseline that records the input shape it receives."""

    def __init__(self) -> None:
        super().__init__()
        self.last_shape: tuple[int, ...] | None = None

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Record shape and return a harmless tensor."""
        self.last_shape = tuple(x.shape)
        return x.mean()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Check deterministic semantic wrappers")
    parser.add_argument("--config", type=str, default="configs/pems03.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--d_proj", type=int, default=16)
    return parser.parse_args()


def _load_one_window(config: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one dummy canonical [B,T,N,F] window and real semantic batch."""
    dcfg = config["dataset"]
    t = int(dcfg["history_steps"])
    n = int(dcfg["num_nodes"])
    f = int(dcfg["input_dim"])
    x_his = torch.zeros((1, t, n, f), device=device, dtype=torch.float32)
    cutoff = torch.tensor([t], dtype=torch.long)
    sem_path = Path(dcfg["data_root"]) / dcfg["name"] / dcfg["semantic_embedding_file"]
    z_static = torch.tensor(load_semantic_embeddings(sem_path), dtype=torch.float32, device=device)
    composer = BatchSemanticComposer(static_z_sem=z_static)
    z_batch = composer.compose(
        batch={"cutoff_step": cutoff},
        device=device,
        num_nodes=n,
    )
    return x_his, z_batch


def _assert_no_projection_params(module: nn.Module) -> None:
    """Assert disabled wrapper does not add trainable semantic projection params."""
    semantic_param_count = sum(
        p.numel()
        for name, p in module.named_parameters()
        if "semantic_adapter" in name
    )
    if semantic_param_count != 0:
        raise AssertionError(f"Disabled wrapper added semantic params: {semantic_param_count}")


def main() -> None:
    """Run wrapper checks."""
    args = parse_args()
    config = load_config(args.config)
    device = get_device(args.device)
    x_his, z_batch = _load_one_window(config, device=device)
    b, t, n, f = x_his.shape

    off_cfg = dict(config)
    off_cfg["baseline"] = {"use_semantic": False, "semantic_proj_dim": int(args.d_proj)}
    on_cfg = dict(config)
    on_cfg["baseline"] = {"use_semantic": True, "semantic_proj_dim": int(args.d_proj)}

    assert adjusted_input_dim(off_cfg) == f
    assert adjusted_input_dim(on_cfg) == f + int(args.d_proj)

    checks = [
        ("GWNet", GWNetSemanticWrapper, (b, f + int(args.d_proj), n, t)),
        ("AGCRN", AGCRNSemanticWrapper, (b, t, n, f + int(args.d_proj))),
        ("PDFormer", PDFormerSemanticWrapper, (b, t, n, f + int(args.d_proj))),
    ]

    for name, wrapper_cls, expected_on_shape in checks:
        base_off = ShapeRecorder()
        wrapped_off = wrapper_cls(
            base_model=base_off,
            sem_dim=z_batch.shape[-1],
            d_proj=int(args.d_proj),
            use_semantic=False,
        ).to(device)
        _assert_no_projection_params(wrapped_off)
        _ = wrapped_off(x_his)
        if base_off.last_shape is None:
            raise AssertionError(f"{name} disabled wrapper did not call baseline.")

        base_on = ShapeRecorder()
        wrapped_on = wrapper_cls(
            base_model=base_on,
            sem_dim=z_batch.shape[-1],
            d_proj=int(args.d_proj),
            use_semantic=True,
        ).to(device)
        _ = wrapped_on(x_his, z_batch=z_batch)
        if base_on.last_shape != expected_on_shape:
            raise AssertionError(
                f"{name} semantic shape mismatch: got {base_on.last_shape}, expected {expected_on_shape}"
            )
        print(f"{name}: off_shape={base_off.last_shape} on_shape={base_on.last_shape}")

    print("Deterministic semantic wrapper check passed.")


if __name__ == "__main__":
    main()
