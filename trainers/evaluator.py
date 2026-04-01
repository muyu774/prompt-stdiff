"""Evaluation utilities for Prompt-STDiff."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion.sampler import DiffusionSampler
from models.prompt_stdiff import PromptSTDiff
from semantic.dynamic_context import DynamicSemanticBank
from utils.metrics import compute_all_metrics, crps_ensemble


def _inverse_with_scaler(x: torch.Tensor, scaler: Optional[object]) -> torch.Tensor:
    """Inverse transform normalized tensor with numpy scaler."""
    if scaler is None:
        return x
    x_np = x.detach().cpu().numpy()
    inv = scaler.inverse_transform(x_np)
    return torch.from_numpy(inv).to(x.device)


@torch.no_grad()
def evaluate(
    model: PromptSTDiff,
    sampler: DiffusionSampler,
    data_loader: DataLoader,
    a_phy: torch.Tensor,
    a_sem: torch.Tensor,
    z_sem: torch.Tensor,
    device: torch.device,
    scaler: Optional[object] = None,
    num_crps_samples: int = 20,
    dynamic_bank: Optional[DynamicSemanticBank] = None,
    eval_horizons: Optional[List[int]] = None,
    logger: Optional[object] = None,
    log_interval: int = 10,
    max_batches: Optional[int] = None,
    metric_feature_index: Optional[int] = None,
    mape_eps: float = 1e-5,
    mape_mask_threshold: float = 1.0,
) -> Dict[str, float]:
    """Evaluate model with MAE/RMSE/MAPE and CRPS."""
    model.eval()

    pred_list = []
    target_list = []
    crps_list = []
    crps_by_h: Dict[int, List[float]] = {}

    def _select_metric_feature(x: torch.Tensor) -> torch.Tensor:
        if metric_feature_index is None:
            return x
        f_total = int(x.shape[-1])
        idx = int(metric_feature_index)
        if idx < 0 or idx >= f_total:
            raise ValueError(
                f"metric_feature_index out of range: {idx}, but feature dim is {f_total}."
            )
        return x[..., idx : idx + 1]

    seen_batches = 0
    for batch_idx, batch in enumerate(data_loader, start=1):
        x_his = batch["x_his"].to(device=device, dtype=torch.float32)
        x_fut = batch["x_fut"].to(device=device, dtype=torch.float32)
        cutoff_step = batch["cutoff_step"].to(device=device, dtype=torch.long)

        b, h, n, f = x_fut.shape
        if dynamic_bank is not None:
            z_sem_batch = dynamic_bank.compose(
                static_z_sem=z_sem,
                cutoff_steps=cutoff_step,
                num_nodes=n,
                device=device,
            )
        else:
            z_sem_batch = z_sem

        cond = {
            "x_his": x_his,
            "a_phy": a_phy,
            "a_sem": a_sem,
            "z_sem": z_sem_batch,
        }

        # CRPS and point metrics are both computed from the same Monte-Carlo ensemble.
        sample_stack = []
        for _ in range(max(num_crps_samples, 1)):
            s = sampler.sample(
                model_fn=model.model_fn,
                shape=(b, h, n, f),
                cond=cond,
                device=device,
            )
            sample_stack.append(s)
        ensemble = torch.stack(sample_stack, dim=0)  # [S, B, H, N, F]

        pred_mean = ensemble.mean(dim=0)
        pred_inv = _inverse_with_scaler(pred_mean, scaler)
        target_inv = _inverse_with_scaler(x_fut, scaler)
        pred_list.append(_select_metric_feature(pred_inv))
        target_list.append(_select_metric_feature(target_inv))

        ensemble_inv = _inverse_with_scaler(ensemble, scaler)
        ensemble_eval = _select_metric_feature(ensemble_inv)
        target_eval = _select_metric_feature(target_inv)
        crps = crps_ensemble(ensemble_eval, target_eval)
        crps_list.append(float(crps.item()))

        if eval_horizons:
            h_total = int(target_inv.shape[1])
            for hh in eval_horizons:
                h_idx = int(hh) - 1
                if h_idx < 0 or h_idx >= h_total:
                    continue
                crps_h = crps_ensemble(
                    ensemble_eval[:, :, h_idx : h_idx + 1, ...],
                    target_eval[:, h_idx : h_idx + 1, ...],
                )
                crps_by_h.setdefault(int(hh), []).append(float(crps_h.item()))

        if logger is not None and log_interval > 0 and (batch_idx % log_interval == 0):
            logger.info(
                "Validation progress | batch %d/%d",
                batch_idx,
                len(data_loader),
            )

        seen_batches += 1
        if max_batches is not None and max_batches > 0 and seen_batches >= max_batches:
            if logger is not None:
                logger.info(
                    "Validation early stop at max_batches=%d (processed %d/%d batches)",
                    int(max_batches),
                    seen_batches,
                    len(data_loader),
                )
            break

    if not pred_list or not target_list:
        raise RuntimeError("No validation batches were processed. Check val loader and max_batches.")

    pred_all = torch.cat(pred_list, dim=0)
    target_all = torch.cat(target_list, dim=0)

    metrics = compute_all_metrics(
        pred_all,
        target_all,
        mape_eps=mape_eps,
        mape_mask_threshold=mape_mask_threshold,
    )
    metrics["crps"] = float(np.mean(crps_list)) if crps_list else float("nan")

    if eval_horizons:
        h_total = int(target_all.shape[1])
        for hh in eval_horizons:
            h_idx = int(hh) - 1
            if h_idx < 0 or h_idx >= h_total:
                continue
            m_h = compute_all_metrics(
                pred_all[:, h_idx : h_idx + 1, ...],
                target_all[:, h_idx : h_idx + 1, ...],
                mape_eps=mape_eps,
                mape_mask_threshold=mape_mask_threshold,
            )
            metrics[f"mae@{hh}"] = m_h["mae"]
            metrics[f"rmse@{hh}"] = m_h["rmse"]
            metrics[f"mape@{hh}"] = m_h["mape"]
            vals = crps_by_h.get(int(hh), [])
            metrics[f"crps@{hh}"] = float(np.mean(vals)) if vals else float("nan")
    return metrics
