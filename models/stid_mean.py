"""STID-style deterministic mean predictor for traffic forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class STIDMeanConfig:
    num_nodes: int
    history_steps: int
    horizon_steps: int
    input_dim: int = 1
    output_dim: int = 1
    hidden_dim: int = 128
    node_emb_dim: int = 64
    horizon_emb_dim: int = 16
    time_emb_dim: int = 16
    day_emb_dim: int = 16
    num_layers: int = 3
    dropout: float = 0.1
    use_time_embeddings: bool = True
    steps_per_day: int = 288
    days_per_week: int = 7

    @classmethod
    def from_mapping(cls, cfg: Mapping) -> "STIDMeanConfig":
        return cls(
            num_nodes=int(cfg["num_nodes"]),
            history_steps=int(cfg["history_steps"]),
            horizon_steps=int(cfg["horizon_steps"]),
            input_dim=int(cfg.get("input_dim", 1)),
            output_dim=int(cfg.get("output_dim", 1)),
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            node_emb_dim=int(cfg.get("node_emb_dim", 64)),
            horizon_emb_dim=int(cfg.get("horizon_emb_dim", 16)),
            time_emb_dim=int(cfg.get("time_emb_dim", 16)),
            day_emb_dim=int(cfg.get("day_emb_dim", 16)),
            num_layers=int(cfg.get("num_layers", 3)),
            dropout=float(cfg.get("dropout", 0.1)),
            use_time_embeddings=bool(cfg.get("use_time_embeddings", True)),
            steps_per_day=int(cfg.get("steps_per_day", 288)),
            days_per_week=int(cfg.get("days_per_week", 7)),
        )

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class STIDMeanModel(nn.Module):
    """Simple STID-family point forecaster.

    The model predicts each node independently from its history, while learned
    node and horizon identities let the shared MLP specialize spatially and by
    lead time. It is intentionally deterministic so it can be frozen as a mean
    predictor for residual diffusion.
    """

    def __init__(self, cfg: STIDMeanConfig | Mapping) -> None:
        super().__init__()
        self.cfg = cfg if isinstance(cfg, STIDMeanConfig) else STIDMeanConfig.from_mapping(cfg)
        c = self.cfg
        self.node_emb = nn.Parameter(torch.empty(c.num_nodes, c.node_emb_dim))
        self.horizon_emb = nn.Parameter(torch.empty(c.horizon_steps, c.horizon_emb_dim))
        if c.use_time_embeddings:
            self.time_in_day_emb = nn.Embedding(c.steps_per_day, c.time_emb_dim)
            self.day_in_week_emb = nn.Embedding(c.days_per_week, c.day_emb_dim)
        else:
            self.time_in_day_emb = None
            self.day_in_week_emb = None

        in_dim = c.history_steps * c.input_dim + c.node_emb_dim + c.horizon_emb_dim
        if c.use_time_embeddings:
            in_dim += c.time_emb_dim + c.day_emb_dim
        layers = []
        dim = in_dim
        for _ in range(max(int(c.num_layers), 1) - 1):
            layers.extend([nn.Linear(dim, c.hidden_dim), nn.ReLU(), nn.Dropout(c.dropout)])
            dim = c.hidden_dim
        layers.append(nn.Linear(dim, c.output_dim))
        self.mlp = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.node_emb)
        nn.init.xavier_uniform_(self.horizon_emb)
        if self.time_in_day_emb is not None:
            nn.init.xavier_uniform_(self.time_in_day_emb.weight)
        if self.day_in_week_emb is not None:
            nn.init.xavier_uniform_(self.day_in_week_emb.weight)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x_his: torch.Tensor, cutoff_step: torch.Tensor | None = None, *_: object, **__: object) -> torch.Tensor:
        """Predict future mean.

        Args:
            x_his: normalized history [B,T,N,F].

        Returns:
            normalized forecast [B,H,N,F_out].
        """
        if x_his.dim() != 4:
            raise ValueError(f"Expected x_his [B,T,N,F], got {tuple(x_his.shape)}")
        b, t, n, f = x_his.shape
        c = self.cfg
        if t != c.history_steps or n != c.num_nodes or f != c.input_dim:
            raise ValueError(
                "STIDMeanModel input shape mismatch: "
                f"got [B,{t},{n},{f}], expected [B,{c.history_steps},{c.num_nodes},{c.input_dim}]"
            )
        hist = x_his.permute(0, 2, 1, 3).reshape(b, n, t * f)
        hist = hist[:, None, :, :].expand(-1, c.horizon_steps, -1, -1)
        node = self.node_emb[None, None, :, :].expand(b, c.horizon_steps, -1, -1)
        horizon = self.horizon_emb[None, :, None, :].expand(b, -1, n, -1)
        parts = [hist, node, horizon]
        if c.use_time_embeddings:
            if cutoff_step is None:
                raise ValueError("cutoff_step is required when STID time embeddings are enabled.")
            steps = cutoff_step.to(device=x_his.device, dtype=torch.long).view(b, 1)
            leads = torch.arange(c.horizon_steps, device=x_his.device, dtype=torch.long).view(1, c.horizon_steps)
            future_steps = steps + leads
            tod_idx = torch.remainder(future_steps, c.steps_per_day)
            dow_idx = torch.remainder(torch.div(future_steps, c.steps_per_day, rounding_mode="floor"), c.days_per_week)
            tod = self.time_in_day_emb(tod_idx)[:, :, None, :].expand(-1, -1, n, -1)
            dow = self.day_in_week_emb(dow_idx)[:, :, None, :].expand(-1, -1, n, -1)
            parts.extend([tod, dow])
        feat = torch.cat(parts, dim=-1)
        out = self.mlp(feat.reshape(b * c.horizon_steps * n, -1))
        return out.view(b, c.horizon_steps, n, c.output_dim)


class FrozenSTIDMean(nn.Module):
    """Checkpoint wrapper exposing the same forward signature as MeanPredictor."""

    def __init__(self, ckpt_path: str, device: Optional[torch.device] = None) -> None:
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location=device or "cpu")
        cfg = STIDMeanConfig.from_mapping(ckpt["model_config"])
        self.model = STIDMeanModel(cfg)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.checkpoint_config = ckpt

    @torch.no_grad()
    def forward(self, batch: Mapping[str, torch.Tensor], z_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        del z_batch
        return self.model(batch["x_his"], cutoff_step=batch.get("cutoff_step"))
