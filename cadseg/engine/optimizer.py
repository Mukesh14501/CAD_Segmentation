# cadseg/engine/optimizer.py
from __future__ import annotations
from typing import Tuple, Literal, Optional
import math
import torch
from torch.optim import Optimizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler, OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau

from cadseg.config import TrainConfig
from cadseg.models.builder import split_encoder_decoder_params


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> Optimizer:
    """
    AdamW with optional discriminative LR:
      - encoder params @ cfg.lr * 0.5
      - decoder/head params @ cfg.lr
    """
    enc_params, dec_params = split_encoder_decoder_params(model)
    if enc_params and dec_params:
        param_groups = [
            {"params": enc_params, "lr": cfg.lr * 0.5, "weight_decay": cfg.weight_decay},
            {"params": dec_params, "lr": cfg.lr,       "weight_decay": cfg.weight_decay},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": cfg.lr, "weight_decay": cfg.weight_decay}]
    return AdamW(param_groups)


def build_scheduler(
    optimizer: Optimizer,
    cfg: TrainConfig,
    steps_per_epoch: int,
) -> Tuple[Optional[_LRScheduler], Literal["batch", "epoch", "plateau"]]:
    """
    Returns (scheduler, mode) where mode is:
      - "batch"   → call scheduler.step() every optimizer step
      - "epoch"   → call scheduler.step() every epoch
      - "plateau" → call scheduler.step(metric) every epoch
    """
    sched = cfg.sched.lower().strip()
    if sched == "onecycle":
        # OneCycleLR needs total steps (epochs * steps_per_epoch)
        total_steps = max(1, cfg.epochs * max(1, steps_per_epoch))
        sch = OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in optimizer.param_groups],
            total_steps=total_steps,
            pct_start=0.1,
            div_factor=10.0,
            final_div_factor=10.0,
            anneal_strategy="cos",
        )
        return sch, "batch"

    if sched == "cosine":
        # Cosine over epochs, step each epoch
        T_max = max(1, cfg.epochs)
        sch = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=cfg.lr * 0.01)
        return sch, "epoch"

    if sched == "reduce_on_plateau":
        sch = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=False)
        return sch, "plateau"

    # none/unknown
    return None, "epoch"
