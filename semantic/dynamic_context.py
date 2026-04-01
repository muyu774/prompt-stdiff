"""Dynamic semantic context bank with strict temporal truncation support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


@dataclass
class DynamicSemanticBank:
    """Dynamic semantic event bank.

    Stored arrays:
    - step_idx: [E] integer step index on the same timeline as traffic data.
    - node_index: [E] node id in [0, N-1], or -1 for global events.
    - embedding: [E, D] event semantic embeddings.
    """

    step_idx: torch.Tensor
    node_index: torch.Tensor
    embedding: torch.Tensor
    fusion_alpha: float
    recency_tau_steps: Optional[float]
    strict_truncation: bool

    @classmethod
    def from_npz(
        cls,
        npz_path: Path,
        fusion_alpha: float,
        recency_tau_steps: Optional[float],
        strict_truncation: bool,
    ) -> "DynamicSemanticBank":
        """Load dynamic semantic bank from npz file."""
        if not npz_path.exists():
            raise FileNotFoundError(f"Dynamic semantic bank not found: {npz_path}")

        bundle = np.load(npz_path)
        if not all(k in bundle for k in ("step_idx", "embedding")):
            raise KeyError(
                f"{npz_path} must contain keys: step_idx, embedding, optional node_index. "
                f"Found keys={bundle.files}"
            )

        step_idx = torch.tensor(bundle["step_idx"], dtype=torch.long)
        embedding = torch.tensor(bundle["embedding"], dtype=torch.float32)

        if "node_index" in bundle:
            node_index = torch.tensor(bundle["node_index"], dtype=torch.long)
        else:
            # ASSUMPTION: missing node_index means all events are global events.
            node_index = torch.full((step_idx.shape[0],), -1, dtype=torch.long)

        if step_idx.ndim != 1:
            raise ValueError(f"step_idx must be [E], got {tuple(step_idx.shape)}")
        if embedding.ndim != 2:
            raise ValueError(f"embedding must be [E,D], got {tuple(embedding.shape)}")
        if node_index.ndim != 1:
            raise ValueError(f"node_index must be [E], got {tuple(node_index.shape)}")
        if not (step_idx.shape[0] == embedding.shape[0] == node_index.shape[0]):
            raise ValueError("step_idx/node_index/embedding must have same first dimension E.")

        return cls(
            step_idx=step_idx,
            node_index=node_index,
            embedding=embedding,
            fusion_alpha=float(fusion_alpha),
            recency_tau_steps=float(recency_tau_steps) if recency_tau_steps is not None else None,
            strict_truncation=bool(strict_truncation),
        )

    @property
    def sem_dim(self) -> int:
        """Semantic embedding dimension D."""
        return int(self.embedding.shape[1])

    def _time_mask(self, cutoff_step: int) -> torch.Tensor:
        """Build temporal mask according to strict truncation setting."""
        if self.strict_truncation:
            # Paper-style strict truncation: keep events observed up to forecast start.
            return self.step_idx <= int(cutoff_step)
        # ASSUMPTION: non-strict mode allows a one-step look-ahead context for ablation/debug.
        return self.step_idx <= int(cutoff_step) + 1

    def _aggregate_one(
        self,
        cutoff_step: int,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Aggregate dynamic semantic delta for one sample.

        Returns:
            delta_z: [N, D]
        """
        mask = self._time_mask(cutoff_step=cutoff_step)
        if int(mask.sum().item()) == 0:
            return torch.zeros((num_nodes, self.sem_dim), dtype=torch.float32, device=device)

        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        emb = self.embedding[idx].to(device=device)
        nodes = self.node_index[idx].to(device=device)
        steps = self.step_idx[idx].to(device=device)

        if self.recency_tau_steps is not None and self.recency_tau_steps > 0:
            age = torch.clamp(float(cutoff_step) - steps.float(), min=0.0)
            weights = torch.exp(-age / float(self.recency_tau_steps))
        else:
            weights = torch.ones((emb.shape[0],), dtype=emb.dtype, device=device)

        delta = torch.zeros((num_nodes, self.sem_dim), dtype=torch.float32, device=device)

        # Global events: node_index == -1
        global_mask = nodes < 0
        if bool(global_mask.any()):
            g_emb = emb[global_mask]
            g_w = weights[global_mask].unsqueeze(-1)
            g_mean = (g_emb * g_w).sum(dim=0, keepdim=True) / torch.clamp(g_w.sum(dim=0, keepdim=True), min=1e-8)
            delta = delta + g_mean

        # Node-specific events: node_index in [0, N-1]
        node_mask = (nodes >= 0) & (nodes < num_nodes)
        if bool(node_mask.any()):
            n_idx = nodes[node_mask].long()
            n_emb = emb[node_mask]
            n_w = weights[node_mask].unsqueeze(-1)

            sum_emb = torch.zeros((num_nodes, self.sem_dim), dtype=torch.float32, device=device)
            sum_w = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)

            sum_emb.index_add_(0, n_idx, n_emb * n_w)
            sum_w.index_add_(0, n_idx, n_w)

            valid = sum_w.squeeze(-1) > 0
            if bool(valid.any()):
                sum_emb[valid] = sum_emb[valid] / torch.clamp(sum_w[valid], min=1e-8)
                delta[valid] = delta[valid] + sum_emb[valid]

        return delta

    def compose(
        self,
        static_z_sem: torch.Tensor,
        cutoff_steps: torch.Tensor,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compose batch semantic embeddings with dynamic context.

        Args:
            static_z_sem: [N, D] or [B, N, D]
            cutoff_steps: [B] forecast-start step indices.
            num_nodes: Number of nodes N.
            device: Target device.

        Returns:
            z_sem_batch: [B, N, D]
        """
        if cutoff_steps.ndim != 1:
            raise ValueError(f"cutoff_steps must be [B], got {tuple(cutoff_steps.shape)}")

        bsz = int(cutoff_steps.shape[0])

        if static_z_sem.dim() == 2:
            base = static_z_sem.unsqueeze(0).expand(bsz, -1, -1).to(device=device)
        elif static_z_sem.dim() == 3:
            if int(static_z_sem.shape[0]) != bsz:
                raise ValueError(
                    f"static_z_sem batch mismatch: expected {bsz}, got {static_z_sem.shape[0]}"
                )
            base = static_z_sem.to(device=device)
        else:
            raise ValueError(f"static_z_sem must be [N,D] or [B,N,D], got {tuple(static_z_sem.shape)}")

        if int(base.shape[1]) != int(num_nodes):
            raise ValueError(f"num_nodes mismatch: expected {num_nodes}, got {base.shape[1]}")
        if int(base.shape[2]) != int(self.sem_dim):
            raise ValueError(
                f"semantic dim mismatch: static {base.shape[2]} vs dynamic bank {self.sem_dim}"
            )

        deltas = []
        cutoff_cpu = cutoff_steps.detach().cpu().long()
        for i in range(bsz):
            delta = self._aggregate_one(
                cutoff_step=int(cutoff_cpu[i].item()),
                num_nodes=num_nodes,
                device=device,
            )
            deltas.append(delta)

        delta_batch = torch.stack(deltas, dim=0)  # [B, N, D]
        return base + float(self.fusion_alpha) * delta_batch


def maybe_load_dynamic_semantic_bank(config: dict, data_root: Path, logger) -> Optional[DynamicSemanticBank]:
    """Conditionally load dynamic semantic bank from config."""
    dcfg = config.get("dataset", {})
    dynamic_cfg = dcfg.get("dynamic_semantic", {}) or {}
    if not bool(dynamic_cfg.get("enabled", False)):
        return None

    bank_file = dynamic_cfg.get("bank_file")
    if not bank_file:
        logger.warning("dynamic_semantic.enabled=true but bank_file is missing. Skipping dynamic semantic.")
        return None

    bank_path = data_root / str(bank_file)
    if not bank_path.exists():
        logger.warning("dynamic semantic bank not found: %s. Skipping dynamic semantic.", bank_path)
        return None

    bank = DynamicSemanticBank.from_npz(
        npz_path=bank_path,
        fusion_alpha=float(dynamic_cfg.get("fusion_alpha", 0.35)),
        recency_tau_steps=dynamic_cfg.get("recency_tau_steps", 288),
        strict_truncation=bool(dynamic_cfg.get("strict_truncation", True)),
    )
    logger.info(
        "Loaded dynamic semantic bank: %s | events=%d dim=%d alpha=%.3f strict=%s",
        bank_path,
        int(bank.step_idx.shape[0]),
        bank.sem_dim,
        float(bank.fusion_alpha),
        str(bank.strict_truncation),
    )
    return bank
