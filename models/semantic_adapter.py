"""Semantic adapters for fair baseline conditioning.

This module intentionally does not modify any baseline interface. It provides
small reusable adapters that baseline entry points can opt into via
``baseline.use_semantic`` while keeping ``use_semantic=False`` as a no-op path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, cast

import torch
import torch.nn as nn

from semantic.dynamic_context import DynamicSemanticBank, maybe_load_dynamic_semantic_bank
from semantic.semantic_cache import load_semantic_embeddings


@dataclass
class SemanticAdapterConfig:
    """Configuration for loading and composing semantic tensors.

    Attributes:
        use_semantic: Whether a baseline consumes semantic conditioning.
        static_path: Path to cached static node semantic embeddings, shape [N, D].
        d_proj: Projection dimension used before feeding semantics into a baseline.
        dynamic_bank: Optional dynamic semantic bank using the same cutoff protocol
            as Prompt-STDiff.
    """

    use_semantic: bool
    static_path: Path
    d_proj: int
    dynamic_bank: Optional[DynamicSemanticBank] = None


def _resolve_semantic_path(config: Mapping[str, Any]) -> Path:
    """Resolve the static semantic embedding path from the existing dataset config."""
    dcfg = config["dataset"]
    data_root = Path(str(dcfg["data_root"])) / str(dcfg["name"])
    return data_root / str(dcfg["semantic_embedding_file"])


def build_semantic_adapter_config(
    config: Mapping[str, Any],
    logger: Optional[Any] = None,
) -> SemanticAdapterConfig:
    """Build adapter config from the shared project config.

    The expected opt-in block is:

    ```yaml
    baseline:
      use_semantic: true
      semantic_proj_dim: 16
    ```

    If the block is absent, ``use_semantic`` defaults to ``False`` so existing
    baseline runs remain unchanged.
    """
    bcfg = dict(config.get("baseline", {}) or {})
    use_semantic = bool(bcfg.get("use_semantic", False))
    # ASSUMPTION: by default, use the same hidden dimension as Prompt-STDiff's
    # semantic branch so diffusion baselines receive a comparable condition size.
    default_proj_dim = int(dict(config.get("model", {}) or {}).get("hidden_dim", 128))
    d_proj = int(bcfg.get("semantic_proj_dim", bcfg.get("d_proj", default_proj_dim)))

    dcfg = config["dataset"]
    data_root = Path(str(dcfg["data_root"])) / str(dcfg["name"])
    dynamic_bank = None
    if use_semantic:
        dynamic_bank = maybe_load_dynamic_semantic_bank(
            dict(config),
            data_root=data_root,
            logger=logger,
        )

    return SemanticAdapterConfig(
        use_semantic=use_semantic,
        static_path=_resolve_semantic_path(config),
        d_proj=d_proj,
        dynamic_bank=dynamic_bank,
    )


def maybe_build_batch_semantic_composer(
    config: Mapping[str, Any],
    device: torch.device,
    logger: Optional[Any] = None,
) -> Optional["BatchSemanticComposer"]:
    """Build a composer only when ``baseline.use_semantic`` is enabled.

    This helper is the safest baseline integration point: the disabled path does
    not read semantic files, load dynamic banks, allocate tensors, or consume RNG.
    """
    adapter_cfg = build_semantic_adapter_config(config, logger=logger)
    if not adapter_cfg.use_semantic:
        return None
    z_np = load_semantic_embeddings(adapter_cfg.static_path)
    z = torch.tensor(z_np, dtype=torch.float32, device=device)
    return BatchSemanticComposer(static_z_sem=z, dynamic_bank=adapter_cfg.dynamic_bank)


class BatchSemanticComposer:
    """Compose per-batch semantic tensors from Prompt-STDiff's exact sources.

    Static semantics come from ``semantic_embeddings.npy``. If a dynamic semantic
    bank is enabled in the config, the same ``DynamicSemanticBank.compose`` path
    used by Prompt-STDiff is used with ``batch["cutoff_step"]``. This guarantees
    identical temporal truncation for baseline conditioning.
    """

    def __init__(
        self,
        static_z_sem: torch.Tensor,
        dynamic_bank: Optional[DynamicSemanticBank] = None,
    ) -> None:
        """Initialize the composer.

        Args:
            static_z_sem: Static node semantics [N, D].
            dynamic_bank: Optional dynamic event bank.
        """
        if static_z_sem.dim() != 2:
            raise ValueError(f"Expected static_z_sem [N,D], got {tuple(static_z_sem.shape)}")
        self.static_z_sem = static_z_sem
        self.dynamic_bank = dynamic_bank

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        device: torch.device,
        logger: Optional[Any] = None,
    ) -> "BatchSemanticComposer":
        """Load semantic cache and dynamic bank from the shared config."""
        adapter_cfg = build_semantic_adapter_config(config, logger=logger)
        z_np = load_semantic_embeddings(adapter_cfg.static_path)
        z = torch.tensor(z_np, dtype=torch.float32, device=device)
        return cls(static_z_sem=z, dynamic_bank=adapter_cfg.dynamic_bank)

    @property
    def sem_dim(self) -> int:
        """Semantic embedding dimension."""
        return int(self.static_z_sem.shape[-1])

    def compose(
        self,
        batch: Mapping[str, torch.Tensor],
        device: torch.device,
        num_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        """Compose semantic tensor for one mini-batch.

        Args:
            batch: Batch from ``build_dataloaders`` containing ``cutoff_step``.
            device: Output device.
            num_nodes: Optional node count sanity check.

        Returns:
            Semantic tensor [B, N, D].
        """
        if "cutoff_step" not in batch:
            raise KeyError("BatchSemanticComposer requires batch['cutoff_step'] for strict truncation.")

        cutoff_step = batch["cutoff_step"].to(device=device, dtype=torch.long)
        if cutoff_step.dim() != 1:
            cutoff_step = cutoff_step.view(-1)

        n = int(num_nodes) if num_nodes is not None else int(self.static_z_sem.shape[0])
        if self.dynamic_bank is not None:
            return self.dynamic_bank.compose(
                static_z_sem=self.static_z_sem,
                cutoff_steps=cutoff_step,
                num_nodes=n,
                device=device,
            )

        bsz = int(cutoff_step.shape[0])
        return self.static_z_sem.to(device=device).unsqueeze(0).expand(bsz, -1, -1)


class DeterministicSemanticInputAdapter(nn.Module):
    """Append semantic features to deterministic baseline inputs.

    When enabled, ``z_batch`` [B, N, D] is projected to [B, N, d_proj], broadcast
    over the history axis, and concatenated with ``x_his``:
    [B, T, N, F] -> [B, T, N, F + d_proj].

    When disabled, the exact original ``x_his`` tensor object is returned.
    """

    def __init__(
        self,
        sem_dim: int,
        d_proj: int,
        use_semantic: bool = False,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the deterministic semantic input adapter."""
        super().__init__()
        self.use_semantic = bool(use_semantic)
        self.d_proj = int(d_proj)
        self.proj: Optional[nn.Linear]
        if self.use_semantic:
            self.proj = nn.Linear(int(sem_dim), self.d_proj)
        else:
            # Keep the disabled path parameter-free and RNG-free so baseline
            # construction remains byte-identical when use_semantic=false.
            self.proj = None
        self.dropout = nn.Dropout(float(dropout))

    @property
    def extra_input_dim(self) -> int:
        """Additional feature channels appended to x_his."""
        return self.d_proj if self.use_semantic else 0

    def forward(self, x_his: torch.Tensor, z_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return original or semantic-augmented history tensor.

        Args:
            x_his: Historical traffic tensor [B, T, N, F].
            z_batch: Batch semantic tensor [B, N, D].
        """
        if not self.use_semantic:
            return x_his
        if z_batch is None:
            raise ValueError("z_batch is required when use_semantic=True.")
        if x_his.dim() != 4:
            raise ValueError(f"Expected x_his [B,T,N,F], got {tuple(x_his.shape)}")
        if z_batch.dim() != 3:
            raise ValueError(f"Expected z_batch [B,N,D], got {tuple(z_batch.shape)}")
        if int(z_batch.shape[0]) != int(x_his.shape[0]) or int(z_batch.shape[1]) != int(x_his.shape[2]):
            raise ValueError(
                "Semantic batch shape mismatch: "
                f"x_his={tuple(x_his.shape)}, z_batch={tuple(z_batch.shape)}"
            )

        bsz, steps, _, _ = x_his.shape
        proj = cast(nn.Linear, self.proj)
        z_proj = self.dropout(proj(z_batch))  # [B, N, d_proj]
        z_time = z_proj.unsqueeze(1).expand(bsz, steps, -1, -1)
        return torch.cat([x_his, z_time], dim=-1)


class DiffusionSemanticConditionAdapter(nn.Module):
    """Append semantic projections to diffusion-baseline conditioning dicts.

    This adapter is deliberately generic because DiffSTG, PriSTI, and SpecSTG
    implementations often name their conditioning tensors differently. The
    default behavior adds ``cond[out_key] = projected_z`` while preserving every
    existing key.

    When disabled, the original condition mapping is returned unchanged.
    """

    def __init__(
        self,
        sem_dim: int,
        d_proj: int,
        use_semantic: bool = False,
        out_key: str = "z_sem_proj",
        dropout: float = 0.0,
    ) -> None:
        """Initialize the diffusion semantic condition adapter."""
        super().__init__()
        self.use_semantic = bool(use_semantic)
        self.out_key = str(out_key)
        self.d_proj = int(d_proj)
        self.proj: Optional[nn.Linear]
        if self.use_semantic:
            self.proj = nn.Linear(int(sem_dim), self.d_proj)
        else:
            # Keep the disabled path parameter-free and RNG-free.
            self.proj = None
        self.dropout = nn.Dropout(float(dropout))

    def project(self, z_batch: torch.Tensor) -> torch.Tensor:
        """Project semantic tensor [B, N, D] to [B, N, d_proj]."""
        if z_batch.dim() != 3:
            raise ValueError(f"Expected z_batch [B,N,D], got {tuple(z_batch.shape)}")
        if self.proj is None:
            raise RuntimeError("Semantic projection is unavailable when use_semantic=False.")
        return self.dropout(self.proj(z_batch))

    def forward(
        self,
        cond: Mapping[str, torch.Tensor],
        z_batch: Optional[torch.Tensor] = None,
    ) -> Mapping[str, torch.Tensor]:
        """Return original or semantic-augmented condition mapping."""
        if not self.use_semantic:
            return cond
        if z_batch is None:
            raise ValueError("z_batch is required when use_semantic=True.")
        out: Dict[str, torch.Tensor] = dict(cond)
        out[self.out_key] = self.project(z_batch)
        return out
