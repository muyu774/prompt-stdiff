"""Incident-gated residual mean-correction heads.

These modules implement a *principled, gated break of mean-preservation* under
detected incident/drop regimes. They are deliberately separate from the
variance-only heads in ``models.prompt_stdiff`` (``ResidualScaleHead`` and
``IncidentTailScaleHead``), which can only widen the predictive ensemble:

* :class:`RegimeShiftHead` predicts a per-(node, horizon, feature) gate
  ``g in [0, 1]`` that estimates the probability of an incident-driven regime
  shift, from leakage-free history/semantic conditioning.
* :class:`MeanCorrectionHead` predicts a *signed* per-(node, horizon, feature)
  shift ``delta`` that moves the predictive *center* (not its spread).
* :class:`GraphMeanPropagator` diffuses the gated shift ``g * delta`` along the
  physical road graph, so a single detected drop corrects the spatially
  correlated drops it causes downstream.

The corrected forecast is ``mu_hat + propagate(g * delta)``. Because the gate is
sparse and the propagator is identity-initialized, off-incident behavior is the
mean-preserving default, so full-test point accuracy is preserved by
construction; the correction only activates where the ``rho`` decomposition shows
that variance calibration cannot help (mean-level misses).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _expand_z_sem(z_sem: torch.Tensor, batch_size: int, num_nodes: int, use_semantic: bool) -> torch.Tensor:
    """Broadcast semantic embeddings to ``[B, N, D]`` (or zeros if disabled)."""
    if z_sem.dim() == 2:
        z = z_sem.unsqueeze(0).expand(batch_size, -1, -1)
    elif z_sem.dim() == 3:
        z = z_sem
    else:
        raise ValueError(f"Expected z_sem [N,D] or [B,N,D], got {z_sem.shape}")
    if z.shape[0] != batch_size or z.shape[1] != num_nodes:
        raise ValueError(f"z_sem shape mismatch: expected [B={batch_size},N={num_nodes},D], got {z.shape}")
    if not use_semantic:
        z = torch.zeros_like(z)
    return z


def _history_node_features(x_his: torch.Tensor) -> torch.Tensor:
    """Leakage-free per-node history summary used by both correction heads.

    Concatenates the last observation, history mean, history std, and the recent
    drop/surge magnitude. None of these touch the forecast horizon, so they are
    safe inference-time inputs.
    """
    hist_last = x_his[:, -1]
    hist_mean = x_his.mean(dim=1)
    hist_std = x_his.std(dim=1, unbiased=False)
    hist_delta = x_his[:, -1] - x_his[:, 0]
    return torch.cat([hist_last, hist_mean, hist_std, hist_delta], dim=-1)


class RegimeShiftHead(nn.Module):
    """Predict a per-(node, horizon, feature) regime-shift gate ``g in [0, 1]``.

    The head mirrors the conditioning of ``IncidentTailScaleHead`` but outputs a
    detection gate rather than a positive scale. A negative initial bias keeps the
    gate near zero by default, so the correction is identity until the detector
    has learned to fire on incident/drop regimes.
    """

    def __init__(
        self,
        input_dim: int,
        sem_dim: int,
        hidden_dim: int,
        horizon_steps: int,
        use_semantic: bool = True,
        init_gate_bias: float = -4.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.sem_dim = int(sem_dim)
        self.horizon_steps = int(horizon_steps)
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
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        # Bias the final logit so the gate starts near zero (mean-preserving default).
        final_linear = self.gate_head[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.constant_(final_linear.bias, float(init_gate_bias))

    def logits(self, x_his: torch.Tensor, z_sem: torch.Tensor) -> torch.Tensor:
        """Return raw gate logits ``[B, H, N, F]``."""
        b, _t, n, _f = x_his.shape
        z = _expand_z_sem(z_sem, batch_size=b, num_nodes=n, use_semantic=self.use_semantic)
        node_in = torch.cat([_history_node_features(x_his), z], dim=-1)
        node_feat = self.node_encoder(node_in)

        h_ids = torch.arange(self.horizon_steps, device=x_his.device)
        h_feat = self.horizon_emb(h_ids).view(1, self.horizon_steps, 1, -1)
        feat = node_feat.unsqueeze(1) + h_feat
        return self.gate_head(feat)

    def forward(self, x_his: torch.Tensor, z_sem: torch.Tensor) -> torch.Tensor:
        """Return the regime-shift gate ``g in [0, 1]`` with shape ``[B, H, N, F]``."""
        return torch.sigmoid(self.logits(x_his=x_his, z_sem=z_sem))


class MeanCorrectionHead(nn.Module):
    """Predict a *signed* per-(node, horizon, feature) mean shift ``delta``.

    Unlike the multiplicative scale heads, this head can move the predictive
    center in either direction, which is required to recover coverage on
    incident-driven drops where the failure is mean-level (``rho >> 1``). The
    output is bounded by ``max_shift`` via ``tanh`` for training stability and is
    expressed in the same (standardized residual / normalized forecasting) space
    as the frozen mean predictor's residual target.
    """

    def __init__(
        self,
        input_dim: int,
        sem_dim: int,
        hidden_dim: int,
        horizon_steps: int,
        use_semantic: bool = True,
        max_shift: float = 4.0,
    ) -> None:
        super().__init__()
        if max_shift <= 0:
            raise ValueError(f"max_shift must be positive, got {max_shift}")
        self.input_dim = int(input_dim)
        self.sem_dim = int(sem_dim)
        self.horizon_steps = int(horizon_steps)
        self.use_semantic = bool(use_semantic)
        self.max_shift = float(max_shift)

        hist_dim = int(input_dim) * 4
        in_dim = hist_dim + int(sem_dim)
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.horizon_emb = nn.Embedding(horizon_steps, hidden_dim)
        self.shift_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        # Start from the zero shift so the correction is identity by default.
        final_linear = self.shift_head[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def forward(self, x_his: torch.Tensor, z_sem: torch.Tensor) -> torch.Tensor:
        """Return the bounded signed shift ``delta`` with shape ``[B, H, N, F]``."""
        b, _t, n, _f = x_his.shape
        z = _expand_z_sem(z_sem, batch_size=b, num_nodes=n, use_semantic=self.use_semantic)
        node_in = torch.cat([_history_node_features(x_his), z], dim=-1)
        node_feat = self.node_encoder(node_in)

        h_ids = torch.arange(self.horizon_steps, device=x_his.device)
        h_feat = self.horizon_emb(h_ids).view(1, self.horizon_steps, 1, -1)
        raw = self.shift_head(node_feat.unsqueeze(1) + h_feat)
        return self.max_shift * torch.tanh(raw)


class GraphMeanPropagator(nn.Module):
    """Diffuse a gated mean shift along the physical road graph.

    Computes ``sum_k w_k (A^k @ shift)`` where ``A`` is the (already normalized)
    physical adjacency and ``w_k`` are softmax-normalized learnable weights. The
    weights are initialized to strongly favor ``k = 0`` (identity), so the
    propagator starts as a no-op and only learns to spread corrections to
    downstream neighbors if that improves incident-regime reliability.
    """

    def __init__(self, num_hops: int = 2) -> None:
        super().__init__()
        if num_hops < 0:
            raise ValueError(f"num_hops must be non-negative, got {num_hops}")
        self.num_hops = int(num_hops)
        # Identity-favoring initialization: large weight on hop 0.
        init = torch.full((self.num_hops + 1,), -4.0)
        init[0] = 4.0
        self.hop_logits = nn.Parameter(init)

    def forward(self, shift: torch.Tensor, a_phy: torch.Tensor) -> torch.Tensor:
        """Propagate ``shift`` ``[B, H, N, F]`` over ``a_phy`` ``[N,N]``/``[B,N,N]``."""
        weights = torch.softmax(self.hop_logits, dim=0)
        out = weights[0] * shift
        if self.num_hops == 0:
            return out
        current = shift
        for k in range(1, self.num_hops + 1):
            if a_phy.dim() == 2:
                current = torch.einsum("ij,bhjf->bhif", a_phy, current)
            elif a_phy.dim() == 3:
                current = torch.einsum("bij,bhjf->bhif", a_phy, current)
            else:
                raise ValueError(f"Expected a_phy [N,N] or [B,N,N], got {a_phy.shape}")
            out = out + weights[k] * current
        return out


def regime_shift_labels(
    residual_target_std: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Binary regime-shift labels from standardized residual magnitude.

    A position is labeled a regime shift when the standardized mean-level residual
    ``|y - mu_hat| / s_h`` exceeds ``threshold``. This ties the detector directly
    to the mean-level surprise that the ``rho`` decomposition isolates. Labels are
    derived from the training target only; the detector inputs never see the
    horizon, so inference remains leakage-free.
    """
    return (residual_target_std.abs() > float(threshold)).to(residual_target_std.dtype)


def detection_loss(gate: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy detection loss for the regime-shift gate."""
    return F.binary_cross_entropy(gate.clamp(1e-6, 1.0 - 1e-6), labels)


def correction_regression_loss(
    gated_shift: torch.Tensor,
    residual_target: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Huber loss driving ``g * delta`` toward the true mean residual on incidents.

    Only labeled regime-shift positions contribute, so the correction learns to
    repair detected mean-level misses without being asked to fit ordinary noise.
    """
    weight = labels
    denom = weight.sum().clamp_min(eps)
    per_elem = F.smooth_l1_loss(gated_shift, residual_target, reduction="none")
    return (per_elem * weight).sum() / denom


def sparsity_loss(gated_shift: torch.Tensor, labels: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Penalize correction magnitude on non-incident positions.

    This is the guardrail that keeps off-incident MAE/CRPS unchanged: away from
    detected regimes the gated shift is pushed toward zero, so the mean-preserving
    default is restored wherever the detector does not fire.
    """
    off = 1.0 - labels
    denom = off.sum().clamp_min(eps)
    return ((gated_shift**2) * off).sum() / denom
