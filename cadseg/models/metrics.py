# cadseg/models/metrics.py
from __future__ import annotations
from typing import Dict, Optional, List, Tuple
import math
import numpy as np
import torch


def _to_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


def _binarize(probs: torch.Tensor, thresholds: float | List[float] | torch.Tensor = 0.5) -> torch.Tensor:
    """
    probs: BxCxHxW
    thresholds: scalar, list length C, or tensor [C]
    return: binary mask BxCxHxW in {0,1}
    """
    if not torch.is_tensor(thresholds):
        if isinstance(thresholds, (list, tuple)):
            thresholds = torch.tensor(thresholds, dtype=probs.dtype, device=probs.device)
        else:
            thresholds = torch.tensor([float(thresholds)], dtype=probs.dtype, device=probs.device)
    if thresholds.numel() == 1:
        thr = thresholds.view(1, 1, 1, 1)
    else:
        thr = thresholds.view(1, -1, 1, 1)
    return (probs >= thr).to(probs.dtype)


def _safe_div(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return a / (b + eps)


@torch.no_grad()
def confusion_counts(
    preds_bin: torch.Tensor,  # BxCxHxW {0,1}
    targets: torch.Tensor,    # BxCxHxW {0,1}
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns per-class (C,) tensors: TP, FP, FN, TN summed over batch and pixels.
    """
    assert preds_bin.shape == targets.shape, "Shape mismatch for confusion_counts"
    B, C, H, W = preds_bin.shape
    p = preds_bin.view(B, C, -1)
    t = targets.view(B, C, -1)
    TP = (p * t).sum(dim=(0, 2))
    FP = (p * (1 - t)).sum(dim=(0, 2))
    FN = ((1 - p) * t).sum(dim=(0, 2))
    TN = ((1 - p) * (1 - t)).sum(dim=(0, 2))
    return TP, FP, FN, TN


@torch.no_grad()
def iou_dice_from_conf(TP: torch.Tensor, FP: torch.Tensor, FN: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    iou = _safe_div(TP, TP + FP + FN)
    dice = _safe_div(2 * TP, 2 * TP + FP + FN)
    return iou, dice


class SegmentationMetrics:
    """
    Streaming metrics accumulator for multi-label segmentation.
    Call update(logits, targets) per batch, then compute().
    """
    def __init__(self, num_classes: int, thresholds: float | List[float] = 0.5):
        self.C = int(num_classes)
        if isinstance(thresholds, (list, tuple)):
            assert len(thresholds) == self.C, "thresholds must have length C"
        self.thresholds = thresholds
        self.reset()

    def reset(self):
        device = torch.device("cpu")
        self.TP = torch.zeros(self.C, dtype=torch.float64, device=device)
        self.FP = torch.zeros(self.C, dtype=torch.float64, device=device)
        self.FN = torch.zeros(self.C, dtype=torch.float64, device=device)
        self.TN = torch.zeros(self.C, dtype=torch.float64, device=device)
        self.support = torch.zeros(self.C, dtype=torch.float64, device=device)  # GT positives

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits: BxCxHxW (raw)
        targets: BxCxHxW float in {0,1}
        """
        probs = _to_probs(logits)
        preds_bin = _binarize(probs, self.thresholds)
        TP, FP, FN, TN = confusion_counts(preds_bin, targets)
        self.TP += TP.double().cpu()
        self.FP += FP.double().cpu()
        self.FN += FN.double().cpu()
        self.TN += TN.double().cpu()
        self.support += (targets.view(targets.shape[0], targets.shape[1], -1).sum(dim=(0, 2))).double().cpu()

    @torch.no_grad()
    def compute(self) -> Dict:
        iou, dice = iou_dice_from_conf(self.TP, self.FP, self.FN)  # (C,)
        prec = _safe_div(self.TP, self.TP + self.FP)
        rec = _safe_div(self.TP, self.TP + self.FN)

        # Macro averages over classes that have GT support > 0
        present = self.support > 0
