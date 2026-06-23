"""Semantic wrappers for deterministic traffic forecasting baselines.

These wrappers are designed for official or third-party implementations of
GWNet, AGCRN, and PDFormer. They do not reimplement those models. Instead, they
provide a thin, auditable semantic-conditioning layer:

1. Compose ``z_batch`` from the same cached Prompt-STDiff semantic sources.
2. Project and broadcast it over the history axis.
3. Concatenate it to ``x_his`` before calling the original baseline.

When semantic conditioning is disabled, the adapter has no projection
parameters and returns the original canonical ``x_his`` tensor before any
optional layout conversion.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn

from models.semantic_adapter import (
    DeterministicSemanticInputAdapter,
    build_semantic_adapter_config,
)


SUPPORTED_LAYOUTS = {"BTNF", "BFNT", "BFTN", "BNTF"}


def permute_layout(x: torch.Tensor, src: str, dst: str) -> torch.Tensor:
    """Permute a 4D traffic tensor between named layouts.

    Layout symbols:
        B: batch, T: history/time, N: nodes, F: features/channels.

    Args:
        x: 4D tensor.
        src: Source layout, e.g. ``BTNF``.
        dst: Destination layout, e.g. ``BFNT`` for common GWNet code.

    Returns:
        Tensor with axes permuted to ``dst``.
    """
    src = src.upper()
    dst = dst.upper()
    if src == dst:
        return x
    if src not in SUPPORTED_LAYOUTS or dst not in SUPPORTED_LAYOUTS:
        raise ValueError(f"Unsupported layout conversion: {src} -> {dst}")
    if x.dim() != 4:
        raise ValueError(f"Expected a 4D tensor for layout {src}, got {tuple(x.shape)}")
    order = [src.index(axis) for axis in dst]
    return x.permute(*order).contiguous()


def deterministic_semantic_extra_dim(config: Mapping[str, Any]) -> int:
    """Return extra input channels added by semantic conditioning."""
    adapter_cfg = build_semantic_adapter_config(config)
    return int(adapter_cfg.d_proj) if adapter_cfg.use_semantic else 0


def adjusted_input_dim(config: Mapping[str, Any], base_input_dim: Optional[int] = None) -> int:
    """Return baseline input dimension after optional semantic augmentation.

    Args:
        config: Shared Prompt-STDiff config containing ``dataset`` and optional
            ``baseline`` blocks.
        base_input_dim: Optional explicit original feature dimension. When not
            provided, ``config["dataset"]["input_dim"]`` is used.
    """
    if base_input_dim is None:
        base_input_dim = int(config["dataset"]["input_dim"])
    return int(base_input_dim) + deterministic_semantic_extra_dim(config)


class DeterministicSemanticBaselineWrapper(nn.Module):
    """Wrap an existing deterministic baseline with semantic input injection.

    The wrapper expects canonical dataloader tensors ``x_his`` in ``[B,T,N,F]``.
    It calls the original baseline with the requested ``baseline_layout``.

    Args:
        base_model: Existing baseline model instance.
        sem_dim: Dimension of cached semantic embeddings.
        d_proj: Projection dimension appended to traffic features.
        use_semantic: Whether to enable semantic augmentation.
        baseline_layout: Layout expected by ``base_model.forward``.
        dropout: Dropout on projected semantic features.
    """

    def __init__(
        self,
        base_model: nn.Module,
        sem_dim: int,
        d_proj: int,
        use_semantic: bool = False,
        baseline_layout: str = "BTNF",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.baseline_layout = baseline_layout.upper()
        if self.baseline_layout not in SUPPORTED_LAYOUTS:
            raise ValueError(f"Unsupported baseline_layout={baseline_layout}")
        self.semantic_adapter = DeterministicSemanticInputAdapter(
            sem_dim=int(sem_dim),
            d_proj=int(d_proj),
            use_semantic=bool(use_semantic),
            dropout=float(dropout),
        )

    @property
    def use_semantic(self) -> bool:
        """Whether semantic conditioning is enabled."""
        return bool(self.semantic_adapter.use_semantic)

    @property
    def extra_input_dim(self) -> int:
        """Number of extra semantic channels appended to traffic input."""
        return int(self.semantic_adapter.extra_input_dim)

    def forward(
        self,
        x_his: torch.Tensor,
        z_batch: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Forward through semantic adapter and the original baseline.

        Args:
            x_his: Canonical historical tensor [B, T, N, F].
            z_batch: Optional semantic tensor [B, N, D_sem], required when
                ``use_semantic=True``.
            *args: Extra positional arguments passed to the baseline.
            **kwargs: Extra keyword arguments passed to the baseline.
        """
        x_aug = self.semantic_adapter(x_his, z_batch=z_batch)
        x_base = permute_layout(x_aug, src="BTNF", dst=self.baseline_layout)
        return self.base_model(x_base, *args, **kwargs)


class GWNetSemanticWrapper(DeterministicSemanticBaselineWrapper):
    """Semantic wrapper for Graph WaveNet-style implementations.

    ASSUMPTION: most Graph WaveNet repositories expect input layout [B, F, N, T].
    Override ``baseline_layout`` if your implementation differs.
    """

    def __init__(
        self,
        base_model: nn.Module,
        sem_dim: int,
        d_proj: int,
        use_semantic: bool = False,
        baseline_layout: str = "BFNT",
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=base_model,
            sem_dim=sem_dim,
            d_proj=d_proj,
            use_semantic=use_semantic,
            baseline_layout=baseline_layout,
            dropout=dropout,
        )


class AGCRNSemanticWrapper(DeterministicSemanticBaselineWrapper):
    """Semantic wrapper for AGCRN-style implementations.

    ASSUMPTION: common AGCRN code consumes [B, T, N, F].
    """

    def __init__(
        self,
        base_model: nn.Module,
        sem_dim: int,
        d_proj: int,
        use_semantic: bool = False,
        baseline_layout: str = "BTNF",
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=base_model,
            sem_dim=sem_dim,
            d_proj=d_proj,
            use_semantic=use_semantic,
            baseline_layout=baseline_layout,
            dropout=dropout,
        )


class PDFormerSemanticWrapper(DeterministicSemanticBaselineWrapper):
    """Semantic wrapper for PDFormer-style implementations.

    ASSUMPTION: common PDFormer dataloaders provide [B, T, N, F].
    """

    def __init__(
        self,
        base_model: nn.Module,
        sem_dim: int,
        d_proj: int,
        use_semantic: bool = False,
        baseline_layout: str = "BTNF",
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=base_model,
            sem_dim=sem_dim,
            d_proj=d_proj,
            use_semantic=use_semantic,
            baseline_layout=baseline_layout,
            dropout=dropout,
        )
