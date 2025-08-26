# cadseg/models/losses.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Dice (multi-label aware)
# ----------------------------
class DiceLoss(nn.Module):
    """
    Multi-label Dice loss over channels (expects logits with shape BxCxHxW).
    Targets should be float in {0,1} with shape BxCxHxW.
    """
    def __init__(self, eps: float = 1e-6, class_weights: Optional[List[float]] = None):
        super().__init__()
        self.eps = eps
        self.class_weights = None
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        # per-class dice
        dims = (0, 2, 3)
        inter = (probs * targets).sum(dim=dims)
        den = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2 * inter + self.eps) / (den + self.eps)  # (C,)
        loss_c = 1.0 - dice
        if self.class_weights is not None:
            w = self.class_weights.to(logits.device, dtype=loss_c.dtype)
            loss = (loss_c * w).sum() / (w.sum() + self.eps)
        else:
            loss = loss_c.mean()
        return loss


# ----------------------------
# Tversky (multi-label)
# ----------------------------
class TverskyLoss(nn.Module):
    """
    Tversky loss (multi-label), good when FN must be penalized more than FP.
    alpha → weight on FP, beta → weight on FN  (common: alpha=0.7, beta=0.3)
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, eps: float = 1e-6):
        super().__init__()
        self.alpha, self.beta, self.eps = float(alpha), float(beta), float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        t = targets.float()
        dims = (0, 2, 3)
        TP = (p * t).sum(dims)
        FP = (p * (1 - t)).sum(dims)
        FN = ((1 - p) * t).sum(dims)
        tversky = (TP + self.eps) / (TP + self.alpha * FP + self.beta * FN + self.eps)
        return (1.0 - tversky).mean()


# ----------------------------
# Focal (multi-label, binary-per-channel)
# ----------------------------
class FocalLoss(nn.Module):
    """
    Focal loss for multi-label segmentation using BCE base.
    alpha can be:
      - None (no class balancing),
      - scalar in [0,1] applied to positives,
      - list/1D tensor of length C (per-class alpha for positives).
    gamma: focusing parameter (2.0 is common).
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[List[float] | float] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE with logits per element
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")  # BxCxHxW
        pt = torch.exp(-bce)  # = 1 - p_t
        # alpha weighting (positives only)
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                a = torch.tensor(self.alpha, dtype=bce.dtype, device=logits.device).view(1, -1, 1, 1)
            else:
                a = torch.tensor(float(self.alpha), dtype=bce.dtype, device=logits.device)
                a = a.view(1, 1, 1, 1)
            # targets=1 gets alpha, targets=0 gets (1-alpha)
            alpha_t = targets * a + (1 - targets) * (1 - a)
            bce = alpha_t * bce
        # focal modulator
        loss = ((1 - pt) ** self.gamma) * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ----------------------------
# Combo wrappers
# ----------------------------
@dataclass
class LossConfig:
    name: str                    # "dice_bce" | "dice_focal" | "tversky" | "dice_ce"
    focal_gamma: float = 2.0
    tversky_alpha: float = 0.7
    tversky_beta: float = 0.3
    # For BCEWithLogits, pos_weight boosts positives per class (list length C).
    # This is NOT the same as alpha in Focal (but similar effect).
    bce_pos_weight: Optional[List[float]] = None
    # Optional per-class weights for Dice
    dice_class_weights: Optional[List[float]] = None


class ComboLoss(nn.Module):
    """
    Flexible combination:
      - "dice_bce": 0.5*Dice + 0.5*BCEWithLogits(pos_weight)
      - "dice_focal": 0.5*Dice + 0.5*Focal(gamma[, alpha])
      - "tversky": Tversky(alpha,beta)
      - "dice_ce": 0.5*Dice + 0.5*BCEWithLogits (same as dice_bce; CE not used in multi-label)
    """
    def __init__(self, cfg: LossConfig, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.num_classes = int(num_classes)

        name = cfg.name.lower()
        self.dice = DiceLoss(class_weights=cfg.dice_class_weights)

        if name in {"dice_bce", "dice_ce"}:
            pos_w = None
            if cfg.bce_pos_weight is not None:
                assert len(cfg.bce_pos_weight) == self.num_classes, "bce_pos_weight must have length C"
                pos_w = torch.tensor(cfg.bce_pos_weight, dtype=torch.float32)
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        if name == "dice_focal":
            # You can reuse bce_pos_weight semantics as alpha if you like (optional)
            alpha = cfg.bce_pos_weight if cfg.bce_pos_weight is not None else None
            self.focal = FocalLoss(gamma=cfg.focal_gamma, alpha=alpha)

        if name == "tversky":
            self.tversky = TverskyLoss(alpha=cfg.tversky_alpha, beta=cfg.tversky_beta)

        self.name = name

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # move optional tensors
        if hasattr(self, "bce") and isinstance(self.bce, nn.BCEWithLogitsLoss):
            if self.bce.pos_weight is not None:
                self.bce.pos_weight = self.bce.pos_weight.to(*args, **kwargs)
        if hasattr(self, "dice") and self.dice.class_weights is not None:
            self.dice.class_weights = self.dice.class_weights.to(*args, **kwargs)
        if hasattr(self, "focal") and isinstance(self.focal.alpha, torch.Tensor):
            self.focal.alpha = self.focal.alpha.to(*args, **kwargs)
        return self

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 🔧 ensure float targets for all component losses
        t = targets.float()

        if self.name == "dice_bce" or self.name == "dice_ce":
            return 0.5 * self.dice(logits, t) + 0.5 * self.bce(logits, t)
        if self.name == "dice_focal":
            return 0.5 * self.dice(logits, t) + 0.5 * self.focal(logits, t)
        if self.name == "tversky":
            return self.tversky(logits, t)
        raise ValueError(f"Unknown loss name: {self.name}")


# ----------------------------
# Builders
# ----------------------------
def build_loss(
    name: str,
    num_classes: int,
    *,
    bce_pos_weight: Optional[List[float]] = None,
    dice_class_weights: Optional[List[float]] = None,
    focal_gamma: float = 2.0,
    tversky_alpha: float = 0.7,
    tversky_beta: float = 0.3,
) -> ComboLoss:
    cfg = LossConfig(
        name=name,
        focal_gamma=focal_gamma,
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta,
        bce_pos_weight=bce_pos_weight,
        dice_class_weights=dice_class_weights,
    )
    return ComboLoss(cfg, num_classes)
