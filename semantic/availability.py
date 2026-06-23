"""Dynamic semantic availability regimes.

This module wraps ``DynamicSemanticBank`` without changing its public
``compose(...)`` signature. The wrapper filters event components according to
deployment availability and then delegates aggregation to the existing bank
logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from semantic.dynamic_context import DynamicSemanticBank


FULL = "full"
DEPLOY_REALISTIC = "deploy_realistic"


def _decode_vocab(vocab: np.ndarray) -> List[str]:
    """Decode string vocab arrays from npz."""
    out: List[str] = []
    for item in vocab.tolist():
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def _event_kind(name: str) -> str:
    """Map raw event type names to coarse availability kinds."""
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


@dataclass
class DynamicSemanticMetadata:
    """Event metadata associated with a dynamic semantic bank."""

    event_type_id: torch.Tensor
    event_type_vocab: List[str]
    event_kind: List[str]

    @classmethod
    def from_npz(cls, npz_path: Path, expected_events: int) -> "DynamicSemanticMetadata":
        """Load event type metadata from a dynamic semantic bank npz."""
        bundle = np.load(npz_path, allow_pickle=True)
        if "event_type_id" in bundle and "event_type_vocab" in bundle:
            type_id = torch.tensor(bundle["event_type_id"], dtype=torch.long)
            vocab = _decode_vocab(bundle["event_type_vocab"])
        elif (npz_path.parent / "dynamic_events.csv").exists():
            type_id, vocab = _load_event_type_from_csv(npz_path.parent / "dynamic_events.csv", expected_events)
        else:
            type_id = torch.zeros((expected_events,), dtype=torch.long)
            vocab = ["unknown"]
        if int(type_id.shape[0]) != int(expected_events):
            raise ValueError(
                f"event_type_id length mismatch: expected {expected_events}, got {type_id.shape[0]}"
            )
        kinds = [_event_kind(x) for x in vocab]
        return cls(event_type_id=type_id, event_type_vocab=vocab, event_kind=kinds)

    def kind_mask(self, kinds: Sequence[str]) -> torch.Tensor:
        """Return event-level mask for coarse event kinds."""
        keep_ids = [i for i, k in enumerate(self.event_kind) if k in set(kinds)]
        if not keep_ids:
            return torch.zeros_like(self.event_type_id, dtype=torch.bool)
        ids = torch.tensor(keep_ids, dtype=torch.long, device=self.event_type_id.device)
        return (self.event_type_id.unsqueeze(-1) == ids.unsqueeze(0)).any(dim=-1)


class AvailabilityAwareDynamicSemanticBank:
    """Wrapper around DynamicSemanticBank with deploy-time availability regimes."""

    def __init__(
        self,
        bank: DynamicSemanticBank,
        metadata: DynamicSemanticMetadata,
        regime: str = FULL,
        incident_lag_steps: int = 0,
        weather_forecast_scale: float = 0.75,
    ) -> None:
        """Initialize wrapper.

        Args:
            bank: Existing DynamicSemanticBank.
            metadata: Event metadata loaded from the bank NPZ.
            regime: ``full`` or ``deploy_realistic``.
            incident_lag_steps: Incident reporting lag in 5-min steps.
            weather_forecast_scale: ASSUMPTION: weather forecast semantics are
                available at t0 but less certain than realized weather, so their
                dynamic delta is scaled.
        """
        self.bank = bank
        self.metadata = metadata
        self.regime = str(regime).lower()
        self.incident_lag_steps = int(max(0, incident_lag_steps))
        self.weather_forecast_scale = float(weather_forecast_scale)
        if self.regime not in {FULL, DEPLOY_REALISTIC}:
            raise ValueError(f"Unsupported availability regime: {regime}")

    @property
    def sem_dim(self) -> int:
        """Semantic embedding dimension."""
        return self.bank.sem_dim

    def _base_for_static(self, static_z_sem: torch.Tensor, bsz: int, device: torch.device) -> torch.Tensor:
        """Broadcast static semantic tensor."""
        if static_z_sem.dim() == 2:
            return static_z_sem.unsqueeze(0).expand(bsz, -1, -1).to(device=device)
        if static_z_sem.dim() == 3:
            if int(static_z_sem.shape[0]) != int(bsz):
                raise ValueError(f"static_z_sem batch mismatch: expected {bsz}, got {static_z_sem.shape[0]}")
            return static_z_sem.to(device=device)
        raise ValueError(f"static_z_sem must be [N,D] or [B,N,D], got {tuple(static_z_sem.shape)}")

    def _visible_mask(self, cutoff_step: int) -> torch.Tensor:
        """Build event visibility mask under the selected regime."""
        if self.regime == FULL:
            return self.bank._time_mask(cutoff_step=cutoff_step)

        base_mask = self.bank._time_mask(cutoff_step=cutoff_step)
        incident_mask = self.metadata.kind_mask(["incident"])
        if bool(incident_mask.any()):
            # Incident is unavailable until onset + reporting lag <= cutoff_step.
            visible_incident = self.bank.step_idx + int(self.incident_lag_steps) <= int(cutoff_step)
            base_mask = base_mask & ((~incident_mask) | visible_incident)
        return base_mask

    def _aggregate_one_with_mask(
        self,
        cutoff_step: int,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Aggregate dynamic delta using the existing bank fields plus visibility mask."""
        mask = self._visible_mask(cutoff_step=cutoff_step)
        if int(mask.sum().item()) == 0:
            return torch.zeros((num_nodes, self.sem_dim), dtype=torch.float32, device=device)

        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        emb = self.bank.embedding[idx].to(device=device)
        nodes = self.bank.node_index[idx].to(device=device)
        steps = self.bank.step_idx[idx].to(device=device)

        if self.regime == DEPLOY_REALISTIC:
            weather_event = self.metadata.kind_mask(["weather"])[idx].to(device=device)
            if bool(weather_event.any()):
                emb = emb.clone()
                emb[weather_event] = emb[weather_event] * self.weather_forecast_scale

        if self.bank.recency_tau_steps is not None and self.bank.recency_tau_steps > 0:
            age = torch.clamp(float(cutoff_step) - steps.float(), min=0.0)
            weights = torch.exp(-age / float(self.bank.recency_tau_steps))
        else:
            weights = torch.ones((emb.shape[0],), dtype=emb.dtype, device=device)

        delta = torch.zeros((num_nodes, self.sem_dim), dtype=torch.float32, device=device)
        global_mask = nodes < 0
        if bool(global_mask.any()):
            g_emb = emb[global_mask]
            g_w = weights[global_mask].unsqueeze(-1)
            g_mean = (g_emb * g_w).sum(dim=0, keepdim=True) / torch.clamp(g_w.sum(dim=0, keepdim=True), min=1e-8)
            delta = delta + g_mean

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
        """Compose z_sem under the selected availability regime.

        Signature mirrors DynamicSemanticBank.compose without modifying it.
        """
        if cutoff_steps.ndim != 1:
            raise ValueError(f"cutoff_steps must be [B], got {tuple(cutoff_steps.shape)}")
        bsz = int(cutoff_steps.shape[0])
        base = self._base_for_static(static_z_sem, bsz=bsz, device=device)
        deltas = []
        cutoff_cpu = cutoff_steps.detach().cpu().long()
        for i in range(bsz):
            deltas.append(
                self._aggregate_one_with_mask(
                    cutoff_step=int(cutoff_cpu[i].item()),
                    num_nodes=num_nodes,
                    device=device,
                )
            )
        delta_batch = torch.stack(deltas, dim=0)
        return base + float(self.bank.fusion_alpha) * delta_batch


def lag_minutes_to_steps(delta_minutes: int, freq_minutes: int = 5) -> int:
    """Convert reporting lag minutes to timeline steps."""
    return int(np.ceil(max(0, int(delta_minutes)) / float(freq_minutes)))


def wrap_dynamic_bank_for_availability(
    bank: Optional[DynamicSemanticBank],
    npz_path: Path,
    regime: str = FULL,
    incident_lag_minutes: int = 0,
    freq_minutes: int = 5,
    weather_forecast_scale: float = 0.75,
) -> Optional[AvailabilityAwareDynamicSemanticBank | DynamicSemanticBank]:
    """Return a bank or wrapper according to availability regime."""
    if bank is None:
        return None
    regime_l = str(regime).lower()
    if regime_l == FULL:
        return bank
    metadata = DynamicSemanticMetadata.from_npz(npz_path=npz_path, expected_events=int(bank.step_idx.shape[0]))
    return AvailabilityAwareDynamicSemanticBank(
        bank=bank,
        metadata=metadata,
        regime=regime_l,
        incident_lag_steps=lag_minutes_to_steps(incident_lag_minutes, freq_minutes=freq_minutes),
        weather_forecast_scale=weather_forecast_scale,
    )


def _load_event_type_from_csv(events_csv: Path, expected_events: int) -> tuple[torch.Tensor, List[str]]:
    """Fallback event type metadata from dynamic_events.csv.

    ASSUMPTION: when old banks do not store event_type_id, dynamic_events.csv rows
    are in the same filtered order used to build the bank. If row counts differ,
    callers fall back to unknown metadata.
    """
    df = pd.read_csv(events_csv)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "event_type" not in df.columns or len(df) != int(expected_events):
        return torch.zeros((expected_events,), dtype=torch.long), ["unknown"]
    vals = df["event_type"].fillna("unknown").astype(str).tolist()
    vocab = sorted(set(vals))
    lookup = {v: i for i, v in enumerate(vocab)}
    ids = torch.tensor([lookup[v] for v in vals], dtype=torch.long)
    return ids, vocab


def wrap_dynamic_bank_from_config(
    bank: Optional[DynamicSemanticBank],
    config: dict,
    data_root: Path,
) -> Optional[AvailabilityAwareDynamicSemanticBank | DynamicSemanticBank]:
    """Apply configured availability regime to an already-loaded dynamic bank."""
    dynamic_cfg = (config.get("dataset", {}) or {}).get("dynamic_semantic", {}) or {}
    availability_cfg = dynamic_cfg.get("availability", {}) or {}
    regime = str(availability_cfg.get("regime", FULL)).lower()
    if bank is None or regime == FULL:
        return bank
    bank_file = dynamic_cfg.get("bank_file", "dynamic_semantic_bank.npz")
    return wrap_dynamic_bank_for_availability(
        bank=bank,
        npz_path=data_root / str(bank_file),
        regime=regime,
        incident_lag_minutes=int(availability_cfg.get("incident_lag_minutes", 0)),
        freq_minutes=int((config.get("dataset", {}) or {}).get("time_freq_minutes", 5)),
        weather_forecast_scale=float(availability_cfg.get("weather_forecast_scale", 0.75)),
    )


def event_type_names(npz_path: Path) -> List[str]:
    """Return dynamic bank event type vocabulary."""
    bundle = np.load(npz_path, allow_pickle=True)
    if "event_type_vocab" not in bundle:
        events_csv = npz_path.parent / "dynamic_events.csv"
        if events_csv.exists():
            _, vocab = _load_event_type_from_csv(events_csv, expected_events=int(bundle["step_idx"].shape[0]))
            return vocab
        return ["unknown"]
    return _decode_vocab(bundle["event_type_vocab"])
