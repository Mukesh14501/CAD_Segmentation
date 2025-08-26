# cadseg/engine/eval_loop.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from cadseg.config import DatasetConfig, InferConfig
from cadseg.dataio.masks import (
    list_images, image_id_from_path, load_image_rgb, load_binary_mask, per_class_mask_path
)
from cadseg.dataio.transforms import build_transforms
from cadseg.dataio.tiling import generate_grid, stitch_probs
from cadseg.models.metrics import SegmentationMetrics


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def infer_full_image_logits(
    model: torch.nn.Module,
    img: np.ndarray,                       # (H,W,3) RGB uint8
    ds_cfg: DatasetConfig,
    infer_cfg: Optional[InferConfig] = None,
    *,
    tile_bs: int = 8,
    device: Optional[torch.device] = None,
    normalize: str = "imagenet",
) -> np.ndarray:
    """
    Runs tiled inference on a single image and returns stitched LOGITS map (C,H,W) in float32.
    We stitch logits (not probs), then downstream can sigmoid + threshold globally.
    """
    device = device or _pick_device()
    model.eval()
    model.to(device)

    tile = ds_cfg.tile_size
    overlap = ds_cfg.tile_overlap
    H, W = img.shape[:2]

    tfm = build_transforms(
        aug_cfg={"infer": {"pad_to_tile": True}},
        stage="infer",
        normalize=normalize,
        tile_size=tile,
        tile_overlap=overlap,
    )

    windows = generate_grid(H, W, tile, overlap)
    tiles_tensor: List[torch.Tensor] = []
    tiles_sizes: List[Tuple[int, int]] = []  # (h_valid,w_valid) per window
    tiles_windows: List[Tuple[int, int, int, int]] = []

    logits_tiles: List[np.ndarray] = []

    def flush_batch():
        if not tiles_tensor:
            return
        batch = torch.stack(tiles_tensor, dim=0).to(device, non_blocking=True)  # (B,3,tile,tile)
        logits = model(batch)  # (B,C,tile,tile)
        logits = logits.detach().float().cpu().numpy()
        # Slice back to valid region per tile and collect
        for i, (h, w) in enumerate(tiles_sizes):
            lt = logits[i, :, :h, :w].copy()  # (C,h,w)
            logits_tiles.append(lt)
        tiles_tensor.clear()
        tiles_sizes.clear()

    for (y0, x0, y1, x1) in windows:
        patch = img[y0:y1, x0:x1, :]
        h, w = patch.shape[:2]
        out = tfm(image=patch, mask=None)
        t = out["image"]  # (3,tile,tile)
        tiles_tensor.append(t)
        tiles_sizes.append((h, w))
        tiles_windows.append((y0, x0, y1, x1))
        if len(tiles_tensor) >= tile_bs:
            flush_batch()
    flush_batch()

    # Stitch logits
    C = logits_tiles[0].shape[0] if logits_tiles else model(torch.zeros(1, 3, tile, tile, device=device)).shape[1]
    logits_full = stitch_probs(logits_tiles, tiles_windows, out_shape=(C, H, W))  # (C,H,W)
    return logits_full.astype(np.float32)


def _stack_multilabel_masks_for_id(
    ds_cfg: DatasetConfig, class_names: List[str], image_id: str, size_hw: Tuple[int, int]
) -> np.ndarray:
    """Load per-class binary masks for one image ID and stack => (C,H,W) float32 in {0.,1.}"""
    H, W = size_hw
    ms = []
    for cname in class_names:
        mp = per_class_mask_path(ds_cfg.masks_path, cname, image_id)
        if mp.exists():
            m = load_binary_mask(mp)
        else:
            m = np.zeros((H, W), dtype=np.uint8)
        if m.shape[:2] != (H, W):
            raise ValueError(f"Mask size mismatch for {image_id}:{cname}. Got {m.shape[:2]}, expected {(H,W)}")
        ms.append(m.astype(np.float32))
    M = np.stack(ms, axis=0)  # (C,H,W)
    return M


@torch.inference_mode()
def evaluate_dataset(
    model: torch.nn.Module,
    ds_cfg: DatasetConfig,
    class_names: List[str],
    infer_cfg: Optional[InferConfig] = None,
    *,
    thresholds: Optional[List[float] | float] = 0.5,
    tile_bs: int = 8,
    device: Optional[torch.device] = None,
    limit: Optional[int] = None,
    normalize: str = "imagenet",
    progress: bool = True,
) -> Dict:
    """
    Full-image evaluation over the dataset root (images + per-class masks).
    Returns aggregated metrics dict from SegmentationMetrics.compute().
    """
    device = device or _pick_device()
    model.eval()
    images = list_images(ds_cfg.images_path)
    if limit is not None:
        images = images[:int(limit)]

    metrics = SegmentationMetrics(num_classes=len(class_names), thresholds=thresholds)

    it = enumerate(images)
    if progress:
        it = tqdm(images, desc="Valid (tiled)", ncols=100)
        def _wrap(): 
            for p in images:
                yield p
        it = _wrap()

    for ip in it:
        if not isinstance(ip, Path):
            ip = Path(ip)
        iid = image_id_from_path(ip)
        img = load_image_rgb(ip)
        H, W = img.shape[:2]

        logits = infer_full_image_logits(
            model, img, ds_cfg, infer_cfg, tile_bs=tile_bs, device=device, normalize=normalize
        )  # (C,H,W)
        targets = _stack_multilabel_masks_for_id(ds_cfg, class_names, iid, (H, W))  # (C,H,W)

        # to tensors with batch dim
        lt = torch.from_numpy(logits).unsqueeze(0)  # (1,C,H,W)
        tt = torch.from_numpy(targets).unsqueeze(0)  # (1,C,H,W)
        metrics.update(lt, tt)

    return metrics.compute()
