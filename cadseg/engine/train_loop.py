# cadseg/engine/train_loop.py
from __future__ import annotations
from typing import Callable, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

import math
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


class _NullCheckpointManager:
    """No-op stand-in when we don't want local checkpoint files."""
    def save(
        self,
        *,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        scaler: Optional[torch.cuda.amp.GradScaler],
        monitor_value: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        return  # intentionally do nothing


@dataclass
class Trainer:
    model: torch.nn.Module
    loss_fn: torch.nn.Module
    cfg: TrainConfig
    save_dir: Optional[str] = None
    device: Optional[torch.device] = None

    def __post_init__(self):
        self.device = self.device or _pick_device()
        self.model.to(self.device)
        self.loss_fn.to(self.device)

        # Optimizer/scaler
        self.optimizer = build_optimizer(self.model, self.cfg)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.cfg.amp and self.device.type == "cuda"))

        # Checkpoint manager: real or no-op
        if self.save_dir:
            self.ckpt = CheckpointManager(
                save_dir=Path(self.save_dir),
                monitor=self.cfg.save_metric,
                mode="max",
            )
        else:
            self.ckpt = _NullCheckpointManager()

        # Optional encoder freeze
        if self.cfg.freeze_encoder_stages:
            freeze_encoder_stages(self.model, self.cfg.freeze_encoder_stages)

        self.scheduler = None
        self.sched_mode = None

    def _build_scheduler(self, steps_per_epoch: int):
        self.scheduler, self.sched_mode = build_scheduler(self.optimizer, self.cfg, steps_per_epoch)

    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> Tuple[float, int]:
        self.model.train(True)
        running_loss = 0.0
        seen = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", ncols=100, leave=False)
        for step, batch in enumerate(pbar, start=1):
            imgs = batch["image"].to(self.device, non_blocking=True)
            masks = batch.get("mask")
            if masks is not None:
                masks = masks.to(self.device, non_blocking=True)

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
                logits = self.model(imgs)
                loss = self.loss_fn(logits, masks)
                loss.backward()
                if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                    clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            seen += imgs.size(0)

            # Per-batch schedule
            if self.scheduler is not None and self.sched_mode == "batch":
                self.scheduler.step()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

        avg_loss = running_loss / max(1, seen)

        # Per-epoch schedule
        if self.scheduler is not None and self.sched_mode == "epoch":
            self.scheduler.step()

        return avg_loss, seen

    def fit(
        self,
        train_loader: DataLoader,
        *,
        validate_fn: Optional[Callable[[int], Dict[str, Any]]] = None,
        start_epoch: int = 0,
    ) -> Dict[str, Any]:
        def _is_better(curr: float, best: Optional[float], min_delta: float = 0.0) -> bool:
            if best is None:
                return True
            return (curr - best) > min_delta  # maximize with tolerance

        # Build scheduler when we know steps/epoch
        self._build_scheduler(steps_per_epoch=max(1, len(train_loader)))

        best_metric: Optional[float] = None
        epochs_no_improve = 0
        monitor_key = self.cfg.save_metric if validate_fn is not None else "neg_train_loss"
        min_delta = getattr(self.cfg, "early_stop_min_delta", 0.0)

        for epoch in range(start_epoch, self.cfg.epochs):
            # Timed unfreeze
            if epoch == getattr(self.cfg, "unfreeze_at_epoch", -1):
                unfreeze_all(self.model)

            train_loss, _ = self._train_one_epoch(train_loader, epoch)

            if validate_fn is not None:
                metrics = validate_fn(epoch) or {}
                raw_val = metrics.get(self.cfg.save_metric, None)
                monitor_value = float(raw_val) if (raw_val is not None) else float("nan")
            else:
                metrics = {"train_loss": train_loss}
                monitor_value = -float(train_loss)  # maximize negative loss

            invalid = (
                (monitor_value is None)
                or math.isnan(monitor_value)
                or math.isinf(monitor_value)
            )

            # Plateaulike schedulers
            if self.scheduler is not None and self.sched_mode == "plateau" and not invalid:
                self.scheduler.step(monitor_value)

            # Save (no-op if _NullCheckpointManager)
            self.ckpt.save(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                monitor_value=(monitor_value if not invalid else (best_metric if best_metric is not None else float("-inf"))),
                extra={"metrics": metrics},
            )

            # Log summary line
            log_kv(
                f"[epoch {epoch}]",
                {
                    "train_loss": f"{train_loss:.4f}",
                    monitor_key: f"{monitor_value:.4f}" if not invalid else "NaN",
                },
            )

            improved = (not invalid) and _is_better(monitor_value, best_metric, min_delta=min_delta)
            if improved:
                best_metric = monitor_value
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= getattr(self.cfg, "early_stop_patience", float("inf")):
                    print(f"Early stopping at epoch {epoch} (no improvement for {self.cfg.early_stop_patience} epochs).")
                    break

        return {"best": best_metric}
