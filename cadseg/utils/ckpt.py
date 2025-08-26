# cadseg/utils/ckpt.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import torch


@dataclass
class CheckpointManager:
    save_dir: Path
    monitor: str = "macro_mIoU"   # key in metrics dict to maximize
    mode: str = "max"             # "max" or "min"

    def __post_init__(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_value: Optional[float] = None
        self.best_path = self.save_dir / "best.pth"
        self.last_path = self.save_dir / "last.pth"

    def is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        return (value > self.best_value) if self.mode == "max" else (value < self.best_value)

    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Any,
        monitor_value: float,
        extra: Optional[Dict[str, Any]] = None,
    ):
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None and hasattr(scheduler, "state_dict") else None,
            "scaler": scaler.state_dict() if scaler is not None and hasattr(scaler, "state_dict") else None,
            "monitor": self.monitor,
            "monitor_value": monitor_value,
        }
        if extra:
            state["extra"] = extra

        # Save last every epoch
        torch.save(state, self.last_path)

        # Save best if improved
        if self.is_better(monitor_value):
            torch.save(state, self.best_path)
            self.best_value = monitor_value

    def load_best(self, model: torch.nn.Module) -> Optional[int]:
        if not self.best_path.exists():
            return None
        state = torch.load(self.best_path, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)
        return int(state.get("epoch", 0))

    def load_last(self, model: torch.nn.Module, optimizer=None, scheduler=None, scaler=None) -> Optional[int]:
        if not self.last_path.exists():
            return None
        state = torch.load(self.last_path, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)
        if optimizer is not None and state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler") is not None and hasattr(scaler, "load_state_dict"):
            scaler.load_state_dict(state["scaler"])
        return int(state.get("epoch", 0))
