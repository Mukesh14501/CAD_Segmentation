# cadseg/engine/infer_loop.py
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path
import os
import json

import numpy as np
import torch
import cv2
from tqdm import tqdm

from cadseg.config import DatasetConfig, InferConfig
from cadseg.dataio.masks import list_images, image_id_from_path, load_image_rgb
from cadseg.engine.eval_loop import infer_full_image_logits, _pick_device
from cadseg.models.postprocess import postprocess_logits


# ----------------------------
# TTA helpers (flip-only)
# ----------------------------
def _tta_variants_list(tta_cfg: Optional[Sequence[str]]) -> List[str]:
    valid = {"hflip", "vflip"}
    tta = [t for t in (tta_cfg or []) if t in valid]
    # Always include "none" pass
    return ["none"] + [t for t in tta if t in valid]


def _apply_flip(img: np.ndarray, kind: str) -> np.ndarray:
    if kind == "hflip":
        return np.ascontiguousarray(img[:, ::-1, :])
    if kind == "vflip":
        return np.ascontiguousarray(img[::-1, :, :])
    return img


def _invert_flip(logits: np.ndarray, kind: str) -> np.ndarray:
    # logits shape (C,H,W)
    if kind == "hflip":
        return logits[:, :, ::-1]
    if kind == "vflip":
        return logits[:, ::-1, :]
    return logits


def _merge_logits(logits_list: List[np.ndarray], mode: str = "mean") -> np.ndarray:
    """
    logits_list: list of (C,H,W)
    mode: "mean" | "gmean" | "max"
    """
    if len(logits_list) == 1:
        return logits_list[0]
    mode = (mode or "mean").lower()
    if mode == "max":
        return np.maximum.reduce(logits_list)
    # For mean/gmean we better convert to probs to avoid extreme logits effect,
    # but we followed a logits fusion approach earlier; here we'll go via probs.
    probs = [1.0 / (1.0 + np.exp(-x)) for x in logits_list]
    if mode == "gmean":
        # Geometric mean of probs
        eps = 1e-6
        logsum = np.zeros_like(probs[0])
        for p in probs:
            logsum += np.log(np.clip(p, eps, 1.0))
        gmean = np.exp(logsum / len(probs))
        # back to logits
        gmean = np.clip(gmean, eps, 1.0 - eps)
        return np.log(gmean / (1.0 - gmean))
    # mean
    m = np.mean(probs, axis=0)
    eps = 1e-6
    m = np.clip(m, eps, 1.0 - eps)
    return np.log(m / (1.0 - m))


# ----------------------------
# Overlay utilities
# ----------------------------
def _make_palette(C: int, seed: int = 13) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # HSV evenly spaced → convert to RGB
    hues = np.linspace(0, 1, C, endpoint=False)
    cols = []
    for h in hues:
        # simple HSV→RGB
        i = int(h * 6.0) % 6
        f = h * 6.0 - i
        q = 1 - f
        if i == 0:
            r, g, b = 1, f, 0
        elif i == 1:
            r, g, b = q, 1, 0
        elif i == 2:
            r, g, b = 0, 1, f
        elif i == 3:
            r, g, b = 0, q, 1
        elif i == 4:
            r, g, b = f, 0, 1
        else:
            r, g, b = 1, 0, q
        cols.append([int(255*r), int(255*g), int(255*b)])
    return np.array(cols, dtype=np.uint8)  # RGB


def make_overlay(
    img_rgb: np.ndarray,                  # (H,W,3) uint8 RGB
    masks_bin: np.ndarray,                # (C,H,W) uint8 {0,1}
    class_names: List[str],
    alpha: float = 0.5,
) -> np.ndarray:
    """Return overlay image in RGB uint8."""
    H, W = img_rgb.shape[:2]
    C = masks_bin.shape[0]
    palette = _make_palette(C)  # RGB
    color_mask = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(C):
        m = masks_bin[c].astype(bool)
        if not m.any():
            continue
        color_mask[m] = palette[c][None, None, :].astype(np.float32)
    overlay = (1 - alpha) * img_rgb.astype(np.float32) + alpha * color_mask
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


# ----------------------------
# Batch inference
# ----------------------------
def infer_folder(
    model: torch.nn.Module,
    ds_cfg: DatasetConfig,
    class_names: List[str],
    infer_cfg: Optional[InferConfig] = None,
    *,
    thresholds: Union[float, Sequence[float]] = 0.5,
    out_dir: Union[str, Path] = "outputs/infer",
    save_binary: bool = True,
    save_overlay: bool = True,
    save_probs_npz: bool = False,
    tile_bs: int = 8,
    tta_list: Optional[Sequence[str]] = None,
    merge: str = "mean",
    device: Optional[torch.device] = None,
    limit: Optional[int] = None,
    normalize: str = "imagenet",
) -> Dict[str, str]:
    """
    Run inference on all images in ds_cfg.images_path and save results under out_dir/<image_id>/.
    Returns a dict with summary info.
    """
    device = device or _pick_device()
    model.eval().to(device)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    images = list_images(ds_cfg.images_path)
    if limit is not None:
        images = images[:int(limit)]

    tta_modes = _tta_variants_list(tta_list if tta_list is not None else getattr(infer_cfg, "tta", ["hflip", "vflip"]))
    merge_mode = merge or (infer_cfg.merge if infer_cfg is not None else "mean")
    min_area = getattr(infer_cfg, "min_component_area", None)

    for ip in tqdm(images, desc="Infer", ncols=100):
        iid = image_id_from_path(ip)
        img = load_image_rgb(ip)
        H, W = img.shape[:2]

        # TTA runs
        logits_list: List[np.ndarray] = []
        for mode in tta_modes:
            img_aug = _apply_flip(img, mode)
            logits = infer_full_image_logits(model, img_aug, ds_cfg, infer_cfg, tile_bs=tile_bs, device=device, normalize=normalize)  # (C,H,W)
            logits = _invert_flip(logits, mode)
            logits_list.append(logits)

        logits_merged = _merge_logits(logits_list, merge_mode)  # (C,H,W)

        # Post-process
        probs, masks_bin = postprocess_logits(
            logits_merged, thresholds=thresholds, min_component_area=min_area
        )  # (C,H,W) each

        # Save
        odir = out_dir / iid
        odir.mkdir(parents=True, exist_ok=True)

        if save_binary:
            for c, cname in enumerate(class_names):
                out_mask = (masks_bin[c] * 255).astype(np.uint8)
                cv2.imwrite(str(odir / f"{c:02d}_{cname}.png"), out_mask)

        if save_probs_npz:
            np.savez_compressed(odir / "probs.npz", probs=probs.astype(np.float16))

        if save_overlay:
            overlay_rgb = make_overlay(img, masks_bin, class_names, alpha=0.45)
            # cv2 wants BGR
            overlay_bgr = overlay_rgb[:, :, ::-1]
            cv2.imwrite(str(odir / "overlay.png"), overlay_bgr)

        # Also keep a simple JSON manifest per image
        meta = {
            "image_id": iid,
            "size_hw": [H, W],
            "classes": class_names,
            "thresholds": thresholds if isinstance(thresholds, (list, tuple)) else [float(thresholds)] * len(class_names),
            "tta": tta_modes,
            "merge": merge_mode,
            "min_component_area": min_area,
        }
        with open(odir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return {"out_dir": str(out_dir), "num_images": len(images)}
