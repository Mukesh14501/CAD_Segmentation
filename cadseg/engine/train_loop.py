# cadseg/engine/train_loop.py
from __future__ import annotations
from typing import Callable, Optional, Dict, Any, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from cadseg.config import TrainConfig
from cadseg.engine.optimizer import build_optimizer, build_scheduler
from cadseg.utils.ckpt import CheckpointManager
from cadseg.utils.logging import log_kv
from cadseg.models.builder import freeze_encoder_stages, unfreeze_all


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Trainer:
    model: torch.nn.Module
    loss_fn: torch.nn.Module
    cfg: TrainConfig
    save_dir: str
    device: Optional[torch.device] = None

    def __post_init__(self):
        self.device = self.device or _pick_device()
        self.model.to(self.device)
        self.loss_fn.to(self.device)

        # Optimizer (built now; scheduler waits for loader length)
        self.optimizer = build_optimizer(self.model, self.cfg)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.cfg.amp and self.device.type == "cuda"))

        self.ckpt = CheckpointManager(save_dir=torch.path.Path(self.save_dir) if hasattr(torch, "path") else __import__("pathlib").Path(self.save_dir),
                                      monitor=self.cfg.save_metric, mode="max")

        # Encoder warmup freeze
        if self.cfg.freeze_encoder_stages:
            freeze_encoder_stages(self.model, self.cfg.freeze_encoder_stages)

    def _build_scheduler(self, steps_per_epoch: int):
        self.scheduler, self.sched_mode = build_scheduler(self.optimizer, self.cfg, steps_per_epoch)

    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> Tuple[float, float]:
        self.model.train(True)
        running_loss = 0.0
        num = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", ncols=100, leave=False)
        for step, batch in enumerate(pbar, start=1):
            imgs = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True) if "mask" in batch else None

            self.optimizer.zero_grad(set_to_none=True)

            if self.cfg.amp and self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(imgs)
                    loss = self.loss_fn(logits, masks)
                self.scaler.scale(loss).backward()
                if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # CPU or MPS or AMP disabled
                logits = self.model(imgs)
                loss = self.loss_fn(logits, masks)
                loss.backward()
                if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                    clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            num += imgs.size(0)

            # LR schedule per batch?
            if self.scheduler is not None and self.sched_mode == "batch":
                self.scheduler.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"})

        avg_loss = running_loss / max(1, num)
        # LR schedule per epoch?
        if self.scheduler is not None and self.sched_mode == "epoch":
            self.scheduler.step()

        return avg_loss, num

    def fit(
        self,
        train_loader: DataLoader,
        *,
        validate_fn: Optional[Callable[[int], Dict[str, Any]]] = None,
        start_epoch: int = 0,
    ) -> Dict[str, Any]:
        """
        Train for cfg.epochs. If validate_fn is provided, it will be called at the end of each epoch:
            metrics = validate_fn(epoch)
        Trainer expects `metrics` to include the monitored key cfg.save_metric (e.g., "macro_mIoU").
        If no validate_fn, we'll monitor negative training loss instead (so lower loss → "higher" metric).
        """
        # Build scheduler now that we know steps_per_epoch
        self._build_scheduler(steps_per_epoch=max(1, len(train_loader)))

        best_metric = None
        epochs_no_improve = 0

        for epoch in range(start_epoch, self.cfg.epochs):
            # Unfreeze encoder at the configured epoch
            if epoch == self.cfg.unfreeze_at_epoch:
                unfreeze_all(self.model)

            train_loss, _ = self._train_one_epoch(train_loader, epoch)

            # Validation (optional, Step 8 will provide validate_fn)
            if validate_fn is not None:
                metrics = validate_fn(epoch) or {}
                monitor_value = float(metrics.get(self.cfg.save_metric, float("nan")))
            else:
                metrics = {"train_loss": train_loss}
                # monitor negative loss to "maximize" (so lower loss appears better)
                monitor_value = -train_loss

            # Plateau LR requires metric
            if self.scheduler is not None and self.sched_mode == "plateau":
                self.scheduler.step(monitor_value)

            # Save checkpoints
            self.ckpt.save(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                monitor_value=monitor_value,
                extra={"metrics": metrics},
            )

            # Logging summary
            log_kv(
                f"[epoch {epoch}]",
                {"train_loss": f"{train_loss:.4f}", self.cfg.save_metric: f"{monitor_value:.4f}"}
            )

            # Early stopping
            if best_metric is None or monitor_value > best_metric:
                best_metric = monitor_value
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.cfg.early_stop_patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {self.cfg.early_stop_patience} epochs).")
                    break

        return {"best": best_metric}
