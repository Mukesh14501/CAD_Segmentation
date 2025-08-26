# cadseg/dataio/transforms.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import math

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _coerce_aug_cfg(aug_cfg):
    """Accept either a plain dict or our AugConfig dataclass."""
    if isinstance(aug_cfg, dict):
        return {
            "train": aug_cfg.get("train", {}) or {},
            "valid": aug_cfg.get("valid", {}) or {},
            "infer": aug_cfg.get("infer", {}) or {},
        }
    # dataclass AugConfig with .train/.valid/.infer
    if hasattr(aug_cfg, "train") and hasattr(aug_cfg, "valid") and hasattr(aug_cfg, "infer"):
        return {
            "train": aug_cfg.train or {},
            "valid": aug_cfg.valid or {},
            "infer": aug_cfg.infer or {},
        }
    return {"train": {}, "valid": {}, "infer": {}}



def _maybe_add_normalize(ops: List[A.BasicTransform], normalize: str | None) -> None:
    """
    normalize:
      - "imagenet" -> A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
      - "none" or None -> no normalization (image stays in [0,1])
    """
    if normalize is None or normalize.lower() == "none":
        return
    if normalize.lower() == "imagenet":
        ops.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        return
    raise ValueError(f"Unknown normalize option: {normalize}")


def _maybe_add_blur(ops: List[A.BasicTransform], ksize: int) -> None:
    if ksize and ksize > 0:
        ops.append(A.GaussianBlur(blur_limit=(ksize, ksize), p=0.15))


def _build_train_ops(cfg: Dict[str, Any], ref_size: int) -> List[A.BasicTransform]:
    """
    CAD-safe train pipeline:
      - Horizontal/Vertical flips
      - 90-degree rotations
      - Mild scale jitter (±)
      - Tiny translation (≈ few pixels) — no elastic/perspective
      - Mild brightness/contrast, optional blur/noise
    """
    flip = bool(cfg.get("flip", True))
    rotate90 = bool(cfg.get("rotate90", True))
    scale_jitter = float(cfg.get("scale_jitter", 0.1))
    bc_strength = float(cfg.get("brightness_contrast", 0.1))
    noise_std = float(cfg.get("gaussian_noise_std", 0.01))
    blur_ksize = int(cfg.get("gaussian_blur_ksize", 0))
    tiny_affine_px = float(cfg.get("tiny_affine_px", 2))  # ~2 px default

    # convert pixel translation to percent using reference size (e.g., tile_size=1024)
    translate_percent = max(0.0, tiny_affine_px / max(1.0, float(ref_size)))
    scale_low = max(0.0, 1.0 - scale_jitter)
    scale_high = 1.0 + scale_jitter

    ops: List[A.BasicTransform] = []

    if flip:
        ops += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]
    if rotate90:
        ops.append(A.RandomRotate90(p=0.5))

    # small geometric changes without bending lines
    ops.append(
        A.Affine(
            scale=(scale_low, scale_high),
            translate_percent={"x": (-translate_percent, translate_percent),
                               "y": (-translate_percent, translate_percent)},
            rotate=0, shear=0,
            interpolation=cv2.INTER_LINEAR,
            cval=0, cval_mask=0, p=0.8
        )
    )

    # mild photometric changes (safe for CAD renders)
    if bc_strength and bc_strength > 0:
        ops.append(A.RandomBrightnessContrast(
            brightness_limit=bc_strength, contrast_limit=bc_strength, p=0.5
        ))
    if noise_std and noise_std > 0:
        ops.append(A.GaussNoise(var_limit=(1e-6, noise_std**2), p=0.15))
    _maybe_add_blur(ops, blur_ksize)

    return ops


def _build_valid_ops(cfg: Dict[str, Any], tile_size: int) -> List[A.BasicTransform]:
    """
    Validation pipeline: size-preserving except optional long-side fit and pad.
    """
    ops: List[A.BasicTransform] = []
    # Optional long-side resize (keep aspect ratio)
    # You can specify: valid: { fit_long_side: 2048 } in aug.yaml
    fit_long = None
    if isinstance(cfg, dict):
        fit_long = cfg.get("fit_long_side", None)
    if fit_long is not None:
        ops.append(A.LongestMaxSize(max_size=int(fit_long), interpolation=cv2.INTER_LINEAR))

    # Ensure at least tile_size (useful if very small)
    ops.append(A.PadIfNeeded(min_height=tile_size, min_width=tile_size,
                             border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0))
    return ops


class PadToMultiple(A.DualTransform):
    """
    Pad image & mask so H and W are multiples of `step`.
    Useful to align with a tiling stride (e.g., step = tile_size - overlap).
    """
    def __init__(self, step: int, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.step = int(step)

    def apply(self, img, **params):
        return self._pad_to_multiple(img, self.step, is_mask=False)

    def apply_to_mask(self, mask, **params):
        return self._pad_to_multiple(mask, self.step, is_mask=True)

    @staticmethod
    def _pad_to_multiple(arr, step: int, is_mask: bool):
        h, w = arr.shape[:2]
        new_h = int(math.ceil(h / step) * step) if step > 0 else h
        new_w = int(math.ceil(w / step) * step) if step > 0 else w
        if new_h == h and new_w == w:
            return arr
        border_mode = cv2.BORDER_CONSTANT
        value = 0
        return cv2.copyMakeBorder(arr, 0, new_h - h, 0, new_w - w, border_mode, value=value)


def _build_infer_ops(cfg: Dict[str, Any], tile_size: int, overlap: int) -> List[A.BasicTransform]:
    """
    Inference pipeline: optional pad-to-multiple so that (H,W) align with your tiling grid.
    """
    ops: List[A.BasicTransform] = []
    # `infer: { pad_to_tile: true }` pads to tile_size minimum
    if bool(cfg.get("pad_to_tile", True)):
        ops.append(A.PadIfNeeded(min_height=tile_size, min_width=tile_size,
                                 border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0))
    # Pad to a multiple of (tile_size - overlap) to simplify stitching windows
    step = max(1, int(tile_size - overlap))
    ops.append(PadToMultiple(step=step))
    return ops


def build_transforms(
    aug_cfg,
    stage: str,
    *,
    normalize: str = "imagenet",
    tile_size: int = 1024,
    tile_overlap: int = 256,
) -> A.Compose:
    aug_cfg = _coerce_aug_cfg(aug_cfg)  # if you added this earlier
    stage = stage.lower()
    if stage not in {"train", "valid", "infer"}:
        raise ValueError("stage must be one of: train | valid | infer")

    ops: List[A.BasicTransform] = []

    if stage == "train":
        ops += _build_train_ops(aug_cfg.get("train", {}), ref_size=tile_size)
        # 🔧 ensure all train tiles are exactly tile_size x tile_size
        ops.append(A.PadIfNeeded(
            min_height=tile_size, min_width=tile_size,
            border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0
        ))
    elif stage == "valid":
        ops += _build_valid_ops(aug_cfg.get("valid", {}), tile_size=tile_size)
        # (valid already pads to tile_size inside _build_valid_ops)
    else:
        ops += _build_infer_ops(aug_cfg.get("infer", {}), tile_size=tile_size, overlap=tile_overlap)

    _maybe_add_normalize(ops, normalize)
    ops.append(ToTensorV2(transpose_mask=True))
    return A.Compose(ops)