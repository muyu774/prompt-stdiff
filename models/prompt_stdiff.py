"""Prompt-STDiff model wrapper."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from models.denoiser import EpsilonTheta
from models.mean_head import TrafficMeanHead
from models.mean_predictor import MeanPredictor, ResidualStandardizer


class ResidualScaleHead(nn.Module):
    """Predict input-dependent residual sample scale.

    The scale is dimensionless and multiplies residual samples after residual
    un-standardization. Training targets are standardized residual magnitudes,
    so the head learns where the diffusion ensemble should be locally wider or
    narrower without moving the ensemble mean.
    """

    def __init__(
        self,
        input_dim: int,
        sem_dim: int,
        hidden_dim: int,
        horizon_steps: int,
        min_scale: float = 0.2,
        max_scale: float = 6.0,
        use_semantic: bool = True,
    ) -> None:
        super().__init__()
        if min_scale <= 0 or max_scale <= 0 or min_scale >= max_scale:
            raise ValueError(f"Invalid residual scale bounds: min={min_scale}, max={max_scale}")
        self.input_dim = int(input_dim)
        self.sem_dim = int(sem_dim)
        self.horizon_steps = int(horizon_steps)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.use_semantic = bool(use_semantic)

        hist_dim = int(input_dim) * 3
        in_dim = hist_dim + int(sem_dim)
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.horizon_emb = nn.Embedding(horizon_steps, hidden_dim)
        self.scale_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def _expand_z_sem(self, z_sem: torch.Tensor, batch_size: int, num_nodes: int) -> torch.Tensor:
        if z_sem.dim() == 2:
            z = z_sem.unsqueeze(0).expand(batch_size, -1, -1)
        elif z_sem.dim() == 3:
            z = z_sem
        else:
            raise ValueError(f"Expected z_sem [N,D] or [B,N,D], got {z_sem.shape}")
        if z.shape[0] != batch_size or z.shape[1] != num_nodes:
            raise ValueError(f"z_sem shape mismatch: expected [B={batch_size},N={num_nodes},D], got {z.shape}")
        if not self.use_semantic:
            z = torch.zeros_like(z)
        return z

    def forward(self, x_his: torch.Tensor, z_sem: torch.Tensor) -> torch.Tensor:
        """Return positive scale tensor [B,H,N,F]."""
        b, _t, n, _f = x_his.shape
        hist_last = x_his[:, -1]
        hist_mean = x_his.mean(dim=1)
        hist_std = x_his.std(dim=1, unbiased=False)
        z = self._expand_z_sem(z_sem, batch_size=b, num_nodes=n)
        node_in = torch.cat([hist_last, hist_mean, hist_std, z], dim=-1)
        node_feat = self.node_encoder(node_in)  # [B,N,C]

        h_ids = torch.arange(self.horizon_steps, device=x_his.device)
        h_feat = self.horizon_emb(h_ids).view(1, self.horizon_steps, 1, -1)
        feat = node_feat.unsqueeze(1) + h_feat
        raw = self.scale_head(feat)
        lo = torch.log(torch.tensor(self.min_scale, dtype=raw.dtype, device=raw.device))
        hi = torch.log(torch.tensor(self.max_scale, dtype=raw.dtype, device=raw.device))
        log_scale = lo + (hi - lo) * torch.sigmoid(raw)
        return torch.exp(log_scale)


class IncidentTailScaleHead(nn.Module):
    """Predict an incident-conditioned heavy-tail residual multiplier.

    This head is intentionally separate from the base heteroscedastic scale:
    the base scale handles routine node/horizon uncertainty, while this
    multiplier gives semantic/event states a focused path to widen residual
    ensembles under incident-like regimes.
    """

    def __init__(
        self,
        input_dim: int,
        sem_dim: int,
        hidden_dim: int,
        horizon_steps: int,
        min_scale: float = 0.85,
        max_scale: float = 4.0,
        use_semantic: bool = True,
    ) -> None:
        super().__init__()
        if min_scale <= 0 or max_scale <= 0 or min_scale >= max_scale:
            raise ValueError(f"Invalid incident tail scale bounds: min={min_scale}, max={max_scale}")
        self.input_dim = int(input_dim)
        self.sem_dim = int(sem_dim)
        self.horizon_steps = int(horizon_steps)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.use_semantic = bool(use_semantic)

        hist_dim = int(input_dim) * 4
        in_dim = hist_dim + int(sem_dim)
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.horizon_emb = nn.Embedding(horizon_steps, hidden_dim)
        self.tail_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def _expand_z_sem(self, z_sem: torch.Tensor, batch_size: int, num_nodes: int) -> torch.Tensor:
        if z_sem.dim() == 2:
            z = z_sem.unsqueeze(0).expand(batch_size, -1, -1)
        elif z_sem.dim() == 3:
            z = z_sem
        else:
            raise ValueError(f"Expected z_sem [N,D] or [B,N,D], got {z_sem.shape}")
        if z.shape[0] != batch_size or z.shape[1] != num_nodes:
            raise ValueError(f"z_sem shape mismatch: expected [B={batch_size},N={num_nodes},D], got {z.shape}")
        if not self.use_semantic:
            z = torch.zeros_like(z)
        return z

    def forward(self, x_his: torch.Tensor, z_sem: torch.Tensor) -> torch.Tensor:
        """Return positive incident tail multiplier [B,H,N,F]."""
        b, _t, n, _f = x_his.shape
        hist_last = x_his[:, -1]
        hist_mean = x_his.mean(dim=1)
        hist_std = x_his.std(dim=1, unbiased=False)
        # Recent drop/surge magnitude is a lightweight event-state proxy that
        # complements external semantic embeddings without leaking future data.
        hist_delta = x_his[:, -1] - x_his[:, 0]
        z = self._expand_z_sem(z_sem, batch_size=b, num_nodes=n)
        node_in = torch.cat([hist_last, hist_mean, hist_std, hist_delta, z], dim=-1)
        node_feat = self.node_encoder(node_in)

        h_ids = torch.arange(self.horizon_steps, device=x_his.device)
        h_feat = self.horizon_emb(h_ids).view(1, self.horizon_steps, 1, -1)
        raw = self.tail_head(node_feat.unsqueeze(1) + h_feat)
        lo = torch.log(torch.tensor(self.min_scale, dtype=raw.dtype, device=raw.device))
        hi = torch.log(torch.tensor(self.max_scale, dtype=raw.dtype, device=raw.device))
        log_scale = lo + (hi - lo) * torch.sigmoid(raw)
        return torch.exp(log_scale)


class PromptSTDiff(nn.Module):
    """Main Prompt-STDiff model.

    This wrapper keeps denoiser interface stable for trainer/sampler.
    """

    def __init__(
        self,
        input_dim: int,
        sem_dim: int,
        hidden_dim: int,
        horizon_steps: int,
        time_embed_dim: int,
        router_hidden_dim: int,
        num_layers: int,
        dropout: float,
        semantic_dropout_p: float,
        use_semantic: bool = True,
        use_mean_head: bool = False,
        mean_head_hidden_dim: int | None = None,
        mean_predictor: MeanPredictor | None = None,
        center_residual_samples: bool = False,
        residual_sample_scale: float = 1.0,
        residual_horizon_scale: list[float] | tuple[float, ...] | None = None,
        residual_node_group_ids: list[int] | tuple[int, ...] | None = None,
        residual_node_group_scale: list[list[float]] | tuple[tuple[float, ...], ...] | None = None,
        use_hetero_residual_scale: bool = False,
        hetero_scale_hidden_dim: int | None = None,
        hetero_scale_min: float = 0.2,
        hetero_scale_max: float = 6.0,
        hetero_scale_use_semantic: bool | None = None,
        use_incident_tail_scale: bool = False,
        incident_tail_hidden_dim: int | None = None,
        incident_tail_min_scale: float = 0.85,
        incident_tail_max_scale: float = 4.0,
        incident_tail_use_semantic: bool = True,
        incident_tail_df: float = 3.0,
    ) -> None:
        super().__init__()
        self.mean_predictor = mean_predictor
        self.use_external_mean_predictor = mean_predictor is not None
        self.residual_standardizer: ResidualStandardizer | None = None
        self.use_mean_head = bool(use_mean_head) or self.use_external_mean_predictor
        self.center_residual_samples = bool(center_residual_samples)
        self.residual_sample_scale = float(residual_sample_scale)
        self.residual_horizon_scale = (
            [float(x) for x in residual_horizon_scale]
            if residual_horizon_scale is not None
            else None
        )
        if self.residual_horizon_scale is not None and len(self.residual_horizon_scale) != int(horizon_steps):
            raise ValueError(
                f"residual_horizon_scale length must equal horizon_steps={horizon_steps}, "
                f"got {len(self.residual_horizon_scale)}"
            )
        self.residual_node_group_ids = (
            [int(x) for x in residual_node_group_ids]
            if residual_node_group_ids is not None
            else None
        )
        self.residual_node_group_scale = (
            [[float(v) for v in row] for row in residual_node_group_scale]
            if residual_node_group_scale is not None
            else None
        )
        if (self.residual_node_group_ids is None) != (self.residual_node_group_scale is None):
            raise ValueError("residual_node_group_ids and residual_node_group_scale must be provided together")
        if self.residual_node_group_scale is not None:
            if len(self.residual_node_group_scale) != int(horizon_steps):
                raise ValueError(
                    "residual_node_group_scale must have one row per horizon step: "
                    f"expected {horizon_steps}, got {len(self.residual_node_group_scale)}"
                )
            width = len(self.residual_node_group_scale[0])
            if width <= 0:
                raise ValueError("residual_node_group_scale must contain at least one group")
            if any(len(row) != width for row in self.residual_node_group_scale):
                raise ValueError("residual_node_group_scale rows must have equal length")
            if self.residual_node_group_ids and max(self.residual_node_group_ids) >= width:
                raise ValueError(
                    "residual_node_group_ids reference a group outside residual_node_group_scale width"
                )
        self.use_hetero_residual_scale = bool(use_hetero_residual_scale)
        self.use_incident_tail_scale = bool(use_incident_tail_scale)
        self.incident_tail_df = float(incident_tail_df)
        if self.incident_tail_df <= 0:
            raise ValueError(f"incident_tail_df must be positive, got {incident_tail_df}")
        self.epsilon_theta = EpsilonTheta(
            input_dim=input_dim,
            sem_dim=sem_dim,
            hidden_dim=hidden_dim,
            horizon_steps=horizon_steps,
            time_embed_dim=time_embed_dim,
            router_hidden_dim=router_hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            semantic_dropout_p=semantic_dropout_p,
            use_semantic=use_semantic,
        )
        self.mean_head = (
            TrafficMeanHead(
                input_dim=input_dim,
                sem_dim=sem_dim,
                hidden_dim=int(mean_head_hidden_dim or hidden_dim),
                horizon_steps=horizon_steps,
                dropout=dropout,
            )
            if self.use_mean_head
            and not self.use_external_mean_predictor
            else None
        )
        self.residual_scale_head = (
            ResidualScaleHead(
                input_dim=input_dim,
                sem_dim=sem_dim,
                hidden_dim=int(hetero_scale_hidden_dim or hidden_dim),
                horizon_steps=horizon_steps,
                min_scale=float(hetero_scale_min),
                max_scale=float(hetero_scale_max),
                use_semantic=(
                    bool(use_semantic)
                    if hetero_scale_use_semantic is None
                    else bool(hetero_scale_use_semantic)
                ),
            )
            if self.use_hetero_residual_scale
            else None
        )
        self.incident_tail_scale_head = (
            IncidentTailScaleHead(
                input_dim=input_dim,
                sem_dim=sem_dim,
                hidden_dim=int(incident_tail_hidden_dim or hetero_scale_hidden_dim or hidden_dim),
                horizon_steps=horizon_steps,
                min_scale=float(incident_tail_min_scale),
                max_scale=float(incident_tail_max_scale),
                use_semantic=bool(incident_tail_use_semantic),
            )
            if self.use_incident_tail_scale
            else None
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_his: torch.Tensor,
        a_phy: torch.Tensor,
        a_sem: torch.Tensor,
        z_sem: torch.Tensor,
    ) -> torch.Tensor:
        """Predict epsilon noise tensor."""
        return self.epsilon_theta(
            x_t=x_t,
            t=t,
            x_his=x_his,
            a_phy=a_phy,
            a_sem=a_sem,
            z_sem=z_sem,
        )

    def model_fn(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Sampler-compatible model function."""
        return self.forward(
            x_t=x_t,
            t=t,
            x_his=cond["x_his"],
            a_phy=cond["a_phy"],
            a_sem=cond["a_sem"],
            z_sem=cond["z_sem"],
        )

    def predict_mean(
        self,
        x_his: torch.Tensor,
        a_phy: torch.Tensor,
        a_sem: torch.Tensor,
        z_sem: torch.Tensor,
        batch: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Predict deterministic future mean, or zeros when disabled."""
        if self.mean_predictor is not None:
            if batch is None:
                batch = {"x_his": x_his}
            return self.mean_predictor(batch=batch, z_batch=z_sem)
        if self.mean_head is None:
            b = x_his.shape[0]
            h = self.epsilon_theta.horizon_steps
            n = x_his.shape[2]
            f = x_his.shape[3]
            return torch.zeros((b, h, n, f), dtype=x_his.dtype, device=x_his.device)
        return self.mean_head(x_his=x_his, a_phy=a_phy, a_sem=a_sem, z_sem=z_sem)

    @property
    def uses_absolute_mean_predictor(self) -> bool:
        """Whether the mean predictor outputs absolute future targets."""
        return self.mean_predictor is not None

    def set_residual_standardizer(self, standardizer: ResidualStandardizer | None) -> None:
        self.residual_standardizer = standardizer

    def standardize_residual(self, residual: torch.Tensor) -> torch.Tensor:
        if self.residual_standardizer is None:
            return residual
        return self.residual_standardizer.standardize(residual)

    def unstandardize_residual(self, residual_std: torch.Tensor) -> torch.Tensor:
        if self.residual_standardizer is None:
            return residual_std
        return self.residual_standardizer.unstandardize(residual_std)

    def predict_residual_scale(self, x_his: torch.Tensor, z_sem: torch.Tensor) -> torch.Tensor:
        """Predict local residual scale [B,H,N,F], or ones when disabled."""
        if self.residual_scale_head is None:
            b = x_his.shape[0]
            h = self.epsilon_theta.horizon_steps
            n = x_his.shape[2]
            f = x_his.shape[3]
            return torch.ones((b, h, n, f), dtype=x_his.dtype, device=x_his.device)
        return self.residual_scale_head(x_his=x_his, z_sem=z_sem)

    def predict_incident_tail_scale(self, x_his: torch.Tensor, z_sem: torch.Tensor) -> torch.Tensor:
        """Predict incident-conditioned residual multiplier [B,H,N,F]."""
        if self.incident_tail_scale_head is None:
            b = x_his.shape[0]
            h = self.epsilon_theta.horizon_steps
            n = x_his.shape[2]
            f = x_his.shape[3]
            return torch.ones((b, h, n, f), dtype=x_his.dtype, device=x_his.device)
        return self.incident_tail_scale_head(x_his=x_his, z_sem=z_sem)

    def predict_total_residual_scale(self, x_his: torch.Tensor, z_sem: torch.Tensor) -> torch.Tensor:
        """Return the train-time local residual scale before post-hoc calibration."""
        return self.predict_residual_scale(x_his=x_his, z_sem=z_sem) * self.predict_incident_tail_scale(
            x_his=x_his,
            z_sem=z_sem,
        )

    def residual_scale_nll_loss(
        self,
        residual_std_target: torch.Tensor,
        x_his: torch.Tensor,
        z_sem: torch.Tensor,
        loss_type: str = "gaussian",
    ) -> torch.Tensor:
        """NLL auxiliary loss for residual scale heads.

        ``student_t`` is useful for incident regimes where residuals have
        heavy tails; constants independent of scale are omitted because the
        term is used as an auxiliary training objective.
        """
        scale = self.predict_total_residual_scale(x_his=x_his, z_sem=z_sem).clamp_min(1e-6)
        target = residual_std_target.detach()
        normalized = target / scale
        if str(loss_type).lower() in {"student_t", "student-t", "t"}:
            nu = torch.tensor(self.incident_tail_df, dtype=target.dtype, device=target.device)
            return torch.mean(torch.log(scale) + 0.5 * (nu + 1.0) * torch.log1p((normalized**2) / nu))
        if str(loss_type).lower() in {"gaussian", "normal"}:
            return torch.mean(torch.log(scale) + 0.5 * normalized**2)
        raise ValueError(f"Unsupported residual scale loss_type: {loss_type}")

    def calibrate_residual_samples(
        self,
        residual: torch.Tensor,
        x_his: torch.Tensor | None = None,
        z_sem: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calibrate sampled residuals before adding the frozen mean predictor.

        For a strong deterministic mean, residual diffusion is primarily used
        for uncertainty. Centering preserves the AGCRN point forecast while the
        residual sample spread still contributes probabilistic information.
        """
        out = residual
        if self.center_residual_samples and out.dim() >= 5:
            out = out - out.mean(dim=0, keepdim=True)
        if self.residual_scale_head is not None:
            if x_his is None or z_sem is None:
                raise ValueError("x_his and z_sem are required when use_hetero_residual_scale=true")
            scale = self.predict_residual_scale(x_his=x_his, z_sem=z_sem)
            if out.dim() == 5:
                scale = scale.unsqueeze(0)
            out = out * scale
        if self.incident_tail_scale_head is not None:
            if x_his is None or z_sem is None:
                raise ValueError("x_his and z_sem are required when use_incident_tail_scale=true")
            tail_scale = self.predict_incident_tail_scale(x_his=x_his, z_sem=z_sem)
            if out.dim() == 5:
                tail_scale = tail_scale.unsqueeze(0)
            out = out * tail_scale
        if self.residual_horizon_scale is not None:
            h_scale = torch.tensor(
                self.residual_horizon_scale,
                dtype=out.dtype,
                device=out.device,
            ).view(1, -1, 1, 1)
            if out.dim() == 5:
                h_scale = h_scale.unsqueeze(0)
            out = out * h_scale
        if self.residual_node_group_scale is not None and self.residual_node_group_ids is not None:
            n = int(out.shape[-2])
            if len(self.residual_node_group_ids) != n:
                raise ValueError(
                    "residual_node_group_ids length must match node dimension: "
                    f"expected {n}, got {len(self.residual_node_group_ids)}"
                )
            group_ids = torch.tensor(self.residual_node_group_ids, dtype=torch.long, device=out.device)
            scale_table = torch.tensor(
                self.residual_node_group_scale,
                dtype=out.dtype,
                device=out.device,
            )
            node_scale = scale_table[:, group_ids].view(1, -1, n, 1)
            if out.dim() == 5:
                node_scale = node_scale.unsqueeze(0)
            out = out * node_scale
        if self.residual_sample_scale != 1.0:
            out = out * self.residual_sample_scale
        return out
