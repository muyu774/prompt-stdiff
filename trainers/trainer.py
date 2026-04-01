"""Training loop for Prompt-STDiff."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import time

import torch

from diffusion.noise_prior import SemanticGuidedDynamicNoisePrior
from diffusion.process import DiffusionProcess
from diffusion.sampler import DiffusionSampler
from models.prompt_stdiff import PromptSTDiff
from semantic.dynamic_context import DynamicSemanticBank
from trainers.evaluator import evaluate
from trainers.losses import build_loss_dict, diffusion_loss_with_x0
from utils.checkpoint import save_checkpoint
from utils.device import autocast_context
from utils.logger import get_logger


class Trainer:
    """Prompt-STDiff trainer."""

    def __init__(
        self,
        model: PromptSTDiff,
        process: DiffusionProcess,
        sampler: DiffusionSampler,
        noise_prior: SemanticGuidedDynamicNoisePrior,
        optimizer: torch.optim.Optimizer,
        train_loader,
        val_loader,
        a_phy: torch.Tensor,
        a_sem: torch.Tensor,
        z_sem: torch.Tensor,
        device: torch.device,
        config: Dict,
        scaler_obj: Optional[object] = None,
        dynamic_bank: Optional[DynamicSemanticBank] = None,
    ) -> None:
        self.model = model
        self.process = process
        self.sampler = sampler
        self.noise_prior = noise_prior
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.a_phy = a_phy
        self.a_sem = a_sem
        self.z_sem = z_sem
        self.device = device
        self.config = config
        self.scaler_obj = scaler_obj
        self.dynamic_bank = dynamic_bank

        self.logger = get_logger()
        self.best_mae = float("inf")

        tcfg = config["train"]
        self.epochs = int(tcfg["epochs"])
        self.grad_clip = float(tcfg.get("grad_clip", 0.0))
        self.log_interval = int(tcfg.get("log_interval", 50))
        self.eval_interval = int(tcfg.get("eval_interval", 1))
        self.num_eval_samples = int(tcfg.get("num_eval_samples", 20))
        # ASSUMPTION: use fewer MC samples during in-training validation for speed;
        # keep `num_eval_samples` for final standalone evaluation.
        self.train_num_eval_samples = int(tcfg.get("train_num_eval_samples", self.num_eval_samples))
        max_eval_batches_cfg = tcfg.get("max_eval_batches", None)
        self.max_eval_batches = (
            int(max_eval_batches_cfg)
            if max_eval_batches_cfg is not None
            else None
        )
        # ASSUMPTION: update last checkpoint every epoch to avoid losing progress on interruption.
        self.save_last_every_epoch = bool(tcfg.get("save_last_every_epoch", True))
        self.eval_horizons = [int(x) for x in tcfg.get("eval_horizons", [3, 6, 12])]
        self.metric_feature_index = tcfg.get("metric_feature_index", None)
        self.mape_eps = float(tcfg.get("mape_eps", 1e-5))
        self.mape_mask_threshold = float(tcfg.get("mape_mask_threshold", 1.0))
        self.loss_eps_weight = float(tcfg.get("loss_eps_weight", 1.0))
        self.loss_x0_weight = float(tcfg.get("loss_x0_weight", 0.0))
        self.loss_x0_type = str(tcfg.get("loss_x0_type", "l1"))

        self.save_dir = Path(tcfg["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.use_amp = bool(tcfg.get("mixed_precision", False))
        amp_enabled = self.use_amp and torch.cuda.is_available()
        try:
            self.amp_scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)
        except TypeError:
            # ASSUMPTION: fallback for older torch versions.
            self.amp_scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    def train(self) -> None:
        """Run full training loop."""
        for epoch in range(1, self.epochs + 1):
            train_stats = self.train_one_epoch(epoch)
            self.logger.info(
                "Epoch %d | train_loss=%.6f",
                epoch,
                train_stats["loss"],
            )

            if epoch % self.eval_interval == 0:
                self.logger.info(
                    "Epoch %d | start validation (val_batches=%d, nsample=%d, diffusion_steps=%d, max_batches=%s)",
                    epoch,
                    len(self.val_loader),
                    self.train_num_eval_samples,
                    int(self.process.num_steps),
                    str(self.max_eval_batches) if self.max_eval_batches is not None else "all",
                )
                t0 = time.time()
                val_metrics = evaluate(
                    model=self.model,
                    sampler=self.sampler,
                    data_loader=self.val_loader,
                    a_phy=self.a_phy,
                    a_sem=self.a_sem,
                    z_sem=self.z_sem,
                    device=self.device,
                    scaler=self.scaler_obj,
                    num_crps_samples=self.train_num_eval_samples,
                    dynamic_bank=self.dynamic_bank,
                    eval_horizons=self.eval_horizons,
                    logger=self.logger,
                    log_interval=max(1, len(self.val_loader) // 10),
                    max_batches=self.max_eval_batches,
                    metric_feature_index=self.metric_feature_index,
                    mape_eps=self.mape_eps,
                    mape_mask_threshold=self.mape_mask_threshold,
                )
                self.logger.info("Epoch %d | validation finished in %.2fs", epoch, time.time() - t0)
                self.logger.info(
                    "Epoch %d | val_mae=%.6f val_rmse=%.6f val_mape=%.6f val_crps=%.6f",
                    epoch,
                    val_metrics["mae"],
                    val_metrics["rmse"],
                    val_metrics["mape"],
                    val_metrics["crps"],
                )
                for h in self.eval_horizons:
                    k_mae = f"mae@{h}"
                    k_rmse = f"rmse@{h}"
                    k_crps = f"crps@{h}"
                    if k_mae in val_metrics and k_rmse in val_metrics and k_crps in val_metrics:
                        self.logger.info(
                            "Epoch %d | horizon=%d | MAE=%.6f RMSE=%.6f CRPS=%.6f",
                            epoch,
                            h,
                            val_metrics[k_mae],
                            val_metrics[k_rmse],
                            val_metrics[k_crps],
                        )

                if val_metrics["mae"] < self.best_mae:
                    self.best_mae = val_metrics["mae"]
                    save_checkpoint(
                        path=self.save_dir / "best.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        best_metric=self.best_mae,
                        config=self.config,
                    )

            if self.save_last_every_epoch:
                save_checkpoint(
                    path=self.save_dir / "last.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    best_metric=self.best_mae,
                    config=self.config,
                )

        save_checkpoint(
            path=self.save_dir / "last.pt",
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.epochs,
            best_metric=self.best_mae,
            config=self.config,
        )

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one epoch of diffusion training."""
        self.model.train()

        running_loss = 0.0
        steps = 0

        for step, batch in enumerate(self.train_loader, start=1):
            x_his = batch["x_his"].to(device=self.device, dtype=torch.float32)
            x_fut = batch["x_fut"].to(device=self.device, dtype=torch.float32)
            cutoff_step = batch["cutoff_step"].to(device=self.device, dtype=torch.long)

            b, h, n, f = x_fut.shape
            if self.dynamic_bank is not None:
                z_sem_batch = self.dynamic_bank.compose(
                    static_z_sem=self.z_sem,
                    cutoff_steps=cutoff_step,
                    num_nodes=n,
                    device=self.device,
                )
            else:
                z_sem_batch = self.z_sem

            t = torch.randint(
                low=0,
                high=self.process.num_steps,
                size=(b,),
                device=self.device,
                dtype=torch.long,
            )

            noise = torch.randn((b, h, n, f), device=self.device, dtype=x_fut.dtype)
            x_t = self.process.q_sample(x_start=x_fut, t=t, noise=noise)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast_context(self.use_amp):
                eps_pred = self.model(
                    x_t=x_t,
                    t=t,
                    x_his=x_his,
                    a_phy=self.a_phy,
                    a_sem=self.a_sem,
                    z_sem=z_sem_batch,
                )
                x0_pred = self.process.predict_x0_from_eps(x_t=x_t, t=t, eps=eps_pred)
                loss = diffusion_loss_with_x0(
                    eps_pred=eps_pred,
                    eps_true=noise,
                    x0_pred=x0_pred,
                    x0_true=x_fut,
                    eps_weight=self.loss_eps_weight,
                    x0_weight=self.loss_x0_weight,
                    x0_loss_type=self.loss_x0_type,
                )

            if self.amp_scaler.is_enabled():
                self.amp_scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.amp_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.amp_scaler.step(self.optimizer)
                self.amp_scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            running_loss += float(loss.detach().item())
            steps += 1

            if step % self.log_interval == 0:
                self.logger.info(
                    "Epoch %d Step %d | loss=%.6f",
                    epoch,
                    step,
                    float(loss.detach().item()),
                )

        mean_loss = running_loss / max(steps, 1)
        return build_loss_dict(torch.tensor(mean_loss))
