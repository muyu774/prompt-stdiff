"""Frozen deterministic mean predictors for residual diffusion."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn

from baselines.runners.run_agcrn import (
    build_model as build_agcrn_model,
    load_checkpoint as load_agcrn_checkpoint,
    normalize_output,
    select_input_feature,
)
from models.stid_mean import STIDMeanConfig, STIDMeanModel
from scripts.run_pdformer_canonical import append_pdformer_time_features, import_pdformer
from utils.config import deep_merge, load_config


def get_mean_predictor_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return mean-predictor config from top-level or model-nested blocks."""
    top = dict(config.get("mean_predictor", {}) or {})
    nested = dict(dict(config.get("model", {}) or {}).get("mean_predictor", {}) or {})
    return deep_merge(top, nested)


def residual_stats_path_from_config(config: Mapping[str, Any]) -> Path:
    """Resolve residual stats cache path, with a deterministic default."""
    mean_cfg = dict(get_mean_predictor_config(config))
    if mean_cfg.get("residual_stats_file"):
        return Path(str(mean_cfg["residual_stats_file"]))
    dcfg = dict(config.get("dataset", {}) or {})
    name = str(dcfg.get("name", "dataset"))
    horizon = int(dcfg.get("horizon_steps", 12))
    return Path("outputs/residual_stats") / f"residual_stats_{name}_h{horizon}.npz"


class MeanPredictor(nn.Module):
    """Frozen deterministic mean predictor used before residual diffusion.

    Currently supports ``type=agcrn``, ``type=stid``, and ``type=pdformer``. It returns
    ``mu_hat`` in the same normalized space as ``batch["x_fut"]`` with shape [B,H,N,F].
    """

    def __init__(self, config: Mapping[str, Any], device: torch.device) -> None:
        super().__init__()
        self.config = dict(config)
        self.mean_cfg = dict(get_mean_predictor_config(self.config))
        self.type = str(self.mean_cfg.get("type", "none")).lower()
        self.freeze = bool(self.mean_cfg.get("freeze", True))
        self.device = device

        if self.type not in {"agcrn", "stid", "pdformer"}:
            raise ValueError(
                f"Unsupported mean_predictor.type={self.type!r}. "
                "Expected 'agcrn', 'stid', or 'pdformer'."
            )

        ckpt = self.mean_cfg.get("pretrained_ckpt")
        if not ckpt:
            raise ValueError(f"mean_predictor.pretrained_ckpt is required for frozen {self.type}.")
        self.pretrained_ckpt = Path(str(ckpt))
        if not self.pretrained_ckpt.exists():
            raise FileNotFoundError(f"Frozen {self.type} checkpoint not found: {self.pretrained_ckpt}")

        if self.type == "stid":
            bundle = torch.load(self.pretrained_ckpt, map_location=device)
            stid_cfg = STIDMeanConfig.from_mapping(bundle["model_config"])
            model = STIDMeanModel(stid_cfg).to(device)
            model.load_state_dict(bundle["model"])
            model.eval()
            if self.freeze:
                for param in model.parameters():
                    param.requires_grad_(False)
            self.model = model
            self.implementation = "ours-stid-mean"
            self.notes = []
            self.agcrn_config = {}
            self.stid_config = stid_cfg.to_dict()
            self.input_feature_index = int(bundle.get("input_feature_index", self.mean_cfg.get("input_feature_index", 0)))
            return

        if self.type == "pdformer":
            bundle = torch.load(self.pretrained_ckpt, map_location=device, weights_only=False)
            pdformer_repo = str(self.mean_cfg.get("pdformer_repo", "baselines/external_repos/PDFormer"))
            PDFormer = import_pdformer(pdformer_repo)
            model_config = dict(bundle["model_config"])
            model_config["device"] = device
            data_feature = dict(bundle["data_feature"])
            model = PDFormer(model_config, data_feature).to(device)
            model.load_state_dict(bundle["model"])
            model.eval()
            if self.freeze:
                for param in model.parameters():
                    param.requires_grad_(False)
            self.model = model
            self.implementation = "official-pdformer-frozen-mean"
            self.notes = []
            self.agcrn_config = {}
            self.pdformer_config = model_config
            self.pdformer_data_feature = data_feature
            self.pdformer_lap_mx = torch.tensor(
                bundle["lap_mx"],
                device=device,
                dtype=torch.float32,
            )
            self.input_feature_index = int(bundle.get("input_feature_index", self.mean_cfg.get("input_feature_index", 0)))
            self.use_time_features = bool(
                self.mean_cfg.get(
                    "use_time_features",
                    bool(model_config.get("add_time_in_day", False))
                    or bool(model_config.get("add_day_in_week", False))
                    or int(data_feature.get("ext_dim", 0)) > 0,
                )
            )
            source_config = self.mean_cfg.get("source_config", None)
            base_config = load_config(str(source_config)) if source_config else self.config
            self.steps_per_day = int(
                self.mean_cfg.get(
                    "steps_per_day",
                    dict(base_config.get("dataset", {}) or {}).get("steps_per_day", 288),
                )
            )
            self.horizon_steps = int(model_config.get("output_window", dict(base_config.get("dataset", {}) or {}).get("horizon_steps", 12)))
            return

        agcrn_config = self._build_agcrn_config()
        agcrn_args = SimpleNamespace(
            model_file=str(self.mean_cfg.get("model_file", "")),
            agcrn_repo=str(self.mean_cfg.get("agcrn_repo", "baselines/external_repos/AGCRN")),
            fallback=bool(self.mean_cfg.get("fallback", False)),
            skip_agcrn_init=bool(self.mean_cfg.get("skip_agcrn_init", True)),
        )
        model, implementation, notes = build_agcrn_model(agcrn_config, args=agcrn_args, device=device)
        setattr(model, "_run_config", agcrn_config)
        load_agcrn_checkpoint(self.pretrained_ckpt, model=model, device=device)
        model.eval()
        if self.freeze:
            for param in model.parameters():
                param.requires_grad_(False)
        self.model = model
        self.implementation = implementation
        self.notes = notes
        self.agcrn_config = agcrn_config

    def _build_agcrn_config(self) -> Mapping[str, Any]:
        source_config = self.mean_cfg.get("source_config", None)
        base_config = load_config(str(source_config)) if source_config else self.config
        baseline_cfg = dict(self.mean_cfg.get("baseline", {}) or {})
        if "input_feature_index" not in baseline_cfg:
            metric_idx = dict(base_config.get("train", {}) or {}).get("metric_feature_index", None)
            if metric_idx is not None:
                baseline_cfg["input_feature_index"] = int(metric_idx)
        if "use_semantic" not in baseline_cfg:
            baseline_cfg["use_semantic"] = bool(self.mean_cfg.get("use_semantic", False))
        if "semantic_proj_dim" not in baseline_cfg and self.mean_cfg.get("semantic_proj_dim") is not None:
            baseline_cfg["semantic_proj_dim"] = int(self.mean_cfg["semantic_proj_dim"])
        return deep_merge(base_config, {"baseline": baseline_cfg})

    @torch.no_grad()
    def forward(self, batch: Mapping[str, torch.Tensor], z_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return frozen mean prediction ``mu_hat`` in normalized data space."""
        x_his = batch["x_his"].to(device=self.device, dtype=torch.float32)
        x_fut = batch["x_fut"].to(device=self.device, dtype=torch.float32)
        if self.type == "stid":
            idx = int(getattr(self, "input_feature_index", 0))
            x_in = x_his[..., idx : idx + 1]
            pred = self.model(x_in)
            target_shape = x_fut[..., idx : idx + 1].shape
            return normalize_output(pred, target_shape)
        if self.type == "pdformer":
            idx = int(getattr(self, "input_feature_index", 0))
            x_in = x_his[..., idx : idx + 1]
            if bool(getattr(self, "use_time_features", False)):
                if "cutoff_step" not in batch:
                    raise KeyError("PDFormer mean predictor requires batch['cutoff_step'] for generated time features.")
                x_in = append_pdformer_time_features(
                    x_in,
                    cutoff_step=batch["cutoff_step"].to(device=self.device),
                    steps_per_day=int(getattr(self, "steps_per_day", 288)),
                )
            if "x_fut" in batch:
                y = x_fut[..., idx : idx + 1]
            else:
                b, _, n, _ = x_his.shape
                y = torch.zeros(
                    (b, int(getattr(self, "horizon_steps", 12)), n, 1),
                    device=self.device,
                    dtype=x_his.dtype,
                )
            if hasattr(self.model, "predict"):
                pred = self.model.predict({"X": x_in, "y": y}, lap_mx=self.pdformer_lap_mx)
            else:
                pred = self.model({"X": x_in, "y": y}, lap_mx=self.pdformer_lap_mx)
            target_shape = y.shape
            return normalize_output(pred, target_shape)
        x_in = select_input_feature(x_his, config=self.agcrn_config)
        target_shape = x_fut[..., :1].shape if int(x_fut.shape[-1]) != int(x_in.shape[-1]) else x_fut.shape
        pred = self.model(x_in, z_batch=z_batch)
        return normalize_output(pred, target_shape)


class ResidualStandardizer(nn.Module):
    """Standardize residual targets with cached train-set statistics."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        if mean.shape != std.shape:
            raise ValueError(f"Residual stats shape mismatch: mean={mean.shape}, std={std.shape}")
        self.register_buffer("mean", mean.view(1, *mean.shape))
        self.register_buffer("std", torch.clamp(std.view(1, *std.shape), min=1e-6))

    @classmethod
    def from_npz(cls, path: Path, device: torch.device) -> "ResidualStandardizer":
        data = torch.load(path, map_location=device) if path.suffix == ".pt" else None
        if data is not None:
            mean = data["mean"].to(device=device, dtype=torch.float32)
            std = data["std"].to(device=device, dtype=torch.float32)
            return cls(mean=mean, std=std)

        import numpy as np

        bundle = np.load(path)
        mean = torch.tensor(bundle["mean"], device=device, dtype=torch.float32)
        std = torch.tensor(bundle["std"], device=device, dtype=torch.float32)
        return cls(mean=mean, std=std)

    def standardize(self, residual: torch.Tensor) -> torch.Tensor:
        return (residual - self.mean) / self.std

    def unstandardize(self, residual_std: torch.Tensor) -> torch.Tensor:
        return residual_std * self.std.unsqueeze(0) + self.mean.unsqueeze(0)


@torch.no_grad()
def compute_or_load_residual_standardizer(
    *,
    path: Path,
    mean_predictor: MeanPredictor,
    train_loader: Any,
    device: torch.device,
    force_recompute: bool = False,
    logger: Optional[Any] = None,
) -> ResidualStandardizer:
    """Compute residual stats over the train split or load the cached file.

    Stats are per horizon and feature, shape [H,1,F] so they broadcast over
    [B,H,N,F] while sharing statistics across nodes.
    """
    if path.exists() and not force_recompute:
        if logger is not None:
            logger.info("Loading residual stats: %s", path)
        return ResidualStandardizer.from_npz(path, device=device)

    if logger is not None:
        logger.info("Computing residual stats over train loader: %s", path)

    count: Optional[torch.Tensor] = None
    sum_r: Optional[torch.Tensor] = None
    sumsq_r: Optional[torch.Tensor] = None
    for batch in train_loader:
        x_fut = batch["x_fut"].to(device=device, dtype=torch.float32)
        mu = mean_predictor(batch)
        if mu.shape != x_fut.shape:
            x_fut = x_fut[..., : mu.shape[-1]]
        residual = x_fut - mu
        reduce_dims = (0, 2)
        batch_count = torch.tensor(
            residual.shape[0] * residual.shape[2],
            device=device,
            dtype=torch.float32,
        )
        batch_sum = residual.sum(dim=reduce_dims, keepdim=False).unsqueeze(1)
        batch_sumsq = (residual * residual).sum(dim=reduce_dims, keepdim=False).unsqueeze(1)
        if count is None:
            count = batch_count
            sum_r = batch_sum
            sumsq_r = batch_sumsq
        else:
            count = count + batch_count
            sum_r = sum_r + batch_sum
            sumsq_r = sumsq_r + batch_sumsq

    if count is None or sum_r is None or sumsq_r is None:
        raise RuntimeError("Cannot compute residual stats: empty train loader.")
    mean = sum_r / count
    var = torch.clamp(sumsq_r / count - mean * mean, min=1e-6)
    std = torch.sqrt(var)

    path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.savez(
        path,
        mean=mean.detach().cpu().numpy().astype(np.float32),
        std=std.detach().cpu().numpy().astype(np.float32),
    )
    if logger is not None:
        logger.info(
            "Saved residual stats: %s mean_abs=%.6f std_min=%.6f std_max=%.6f",
            path,
            float(mean.abs().mean().item()),
            float(std.min().item()),
            float(std.max().item()),
        )
    return ResidualStandardizer(mean=mean, std=std)
