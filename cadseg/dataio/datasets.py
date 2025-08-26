"""
Dataset stubs:
- CADSegDataset: full images and masks (multi-label).
- TiledDataset: tile view over large canvases (with overlap).
- InferenceDataset: images only, for inference-time tiling.
"""


# cadseg/dataio/datasets.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from cadseg.config import DatasetConfig
from cadseg.dataio.transforms import build_transforms
from cadseg.dataio.masks import (
    list_images,
    image_id_from_path,
    load_image_rgb,
    load_binary_mask,
    per_class_mask_path,
)

# ----------------------------
# Helpers
# ----------------------------
def _stack_multilabel_masks(
    masks_root: Path,
    class_names: List[str],
    image_id: str,
    size_hw: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Load per-class binary masks and stack into (H, W, C) with values {0,1}.
    If a mask file is missing for a class, use an all-zero mask.
    Optionally enforce (H, W) to `size_hw`.
    """
    masks: List[np.ndarray] = []
    H, W = None, None
    for cname in class_names:
        mp = per_class_mask_path(masks_root, cname, image_id)
        if mp.exists():
            m = load_binary_mask(mp)
        else:
            # fallback to zero mask (no positives for this class)
            if size_hw is None and (H is None or W is None):
                raise FileNotFoundError(
                    f"Mask missing for class='{cname}', image_id='{image_id}', and size cannot be inferred. "
                    f"Provide size_hw or ensure at least one class mask exists."
                )
            h_ref, w_ref = size_hw if size_hw is not None else (H, W)
            m = np.zeros((h_ref, w_ref), dtype=np.uint8)
        if H is None or W is None:
            H, W = m.shape[:2]
        if size_hw is not None and m.shape[:2] != size_hw:
            raise ValueError(
                f"Mask size mismatch for {image_id}:{cname}. Expected {size_hw}, got {m.shape[:2]}"
            )
        masks.append(m)
    M = np.stack(masks, axis=-1).astype(np.uint8)  # (H, W, C)
    return M


# ----------------------------
# Full-image dataset
# ----------------------------
class CADSegDataset(Dataset):
    """
    Returns full images and multi-label masks (for small images or debugging).
    __getitem__ returns:
      - image: float tensor (3,H,W)
      - mask : float tensor (C,H,W) in {0.,1.}
      - meta : dict(id, size_hw)
    """
    def __init__(
        self,
        ds_cfg: DatasetConfig,
        class_names: List[str],
        stage: str = "train",
        aug_cfg: Optional[Dict[str, Any]] = None,
        normalize: str = "imagenet",
    ):
        self.ds_cfg = ds_cfg
        self.class_names = class_names
        self.stage = stage
        self.C = len(class_names)
        self.images_dir = ds_cfg.images_path
        self.masks_dir = ds_cfg.masks_path

        self.image_paths: List[Path] = list_images(self.images_dir)
        self.ids: List[str] = [image_id_from_path(p) for p in self.image_paths]

        self.transform = build_transforms(
            aug_cfg or {},
            stage=self.stage,
            normalize=normalize,
            tile_size=ds_cfg.tile_size,
            tile_overlap=ds_cfg.tile_overlap,
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ip = self.image_paths[idx]
        iid = self.ids[idx]

        img = load_image_rgb(ip)  # (H,W,3), uint8
        H, W = img.shape[:2]
        mask = _stack_multilabel_masks(self.masks_dir, self.class_names, iid, size_hw=(H, W))

        # Albumentations expects dict; ToTensorV2 gives tensors
        out = self.transform(image=img, mask=mask)
        image_t = out["image"]  # (3,H,W) float32
        mask_t = out["mask"]    # (C,H,W) float32

        meta = {"id": iid, "size_hw": (H, W)}
        return {"image": image_t, "mask": mask_t, "meta": meta}


# ----------------------------
# Tiled training dataset
# ----------------------------
@dataclass
class TileInfo:
    image_index: int
    id: str
    y0: int
    x0: int
    y1: int
    x1: int
    has_pos_any: bool
    has_pos_per_class: List[bool]


def _generate_grid(H: int, W: int, tile: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    """
    Generate [y0,x0,y1,x1] windows covering the image with the given tile size and overlap.
    Ensures flush to image borders.
    """
    stride = max(1, tile - overlap)
    ys = list(range(0, max(1, H - tile + 1), stride))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if len(ys) == 0: ys = [0]
    if len(xs) == 0: xs = [0]
    # force last tile to touch the border
    if ys[-1] + tile < H:
        ys.append(H - tile)
    if xs[-1] + tile < W:
        xs.append(W - tile)

    windows = []
    for y0 in ys:
        for x0 in xs:
            y1 = min(H, y0 + tile)
            x1 = min(W, x0 + tile)
            windows.append((y0, x0, y1, x1))
    return windows


class TiledDataset(Dataset):
    """
    Generates overlapping tiles from large CAD canvases for training/validation.

    __getitem__ returns:
      - image: float tensor (3,tile,tile)
      - mask : float tensor (C,tile,tile)
      - meta : dict(id, y0, x0, y1, x1, has_pos_any, has_pos_per_class)
    """
    def __init__(
        self,
        ds_cfg: DatasetConfig,
        class_names: List[str],
        stage: str = "train",
        aug_cfg: Optional[Dict[str, Any]] = None,
        normalize: str = "imagenet",
        preload_index: bool = True,
        min_positive_area: Optional[int] = None,
    ):
        assert stage in {"train", "valid"}, "TiledDataset is intended for train/valid"
        self.ds_cfg = ds_cfg
        self.class_names = class_names
        self.C = len(class_names)
        self.stage = stage
        self.tile = ds_cfg.tile_size
        self.overlap = ds_cfg.tile_overlap
        self.images_dir = ds_cfg.images_path
        self.masks_dir = ds_cfg.masks_path
        self.min_positive_area = int(min_positive_area) if min_positive_area is not None else ds_cfg.min_positive_area

        self.image_paths: List[Path] = list_images(self.images_dir)
        self.ids: List[str] = [image_id_from_path(p) for p in self.image_paths]

        self.transform = build_transforms(
            aug_cfg or {},
            stage=self.stage,
            normalize=normalize,
            tile_size=self.tile,
            tile_overlap=self.overlap,
        )

        # Build tile index (compute which tiles contain positives per class)
        self.tiles: List[TileInfo] = []
        if preload_index:
            self._build_tile_index()

    def _build_tile_index(self) -> None:
        for i, ip in enumerate(self.image_paths):
            iid = self.ids[i]
            img = load_image_rgb(ip)
            H, W = img.shape[:2]
            # Load all masks stacked
            M = _stack_multilabel_masks(self.masks_dir, self.class_names, iid, size_hw=(H, W))  # (H,W,C)

            for (y0, x0, y1, x1) in _generate_grid(H, W, self.tile, self.overlap):
                m_tile = M[y0:y1, x0:x1, :]  # (h,w,C)
                # small tolerance if edge tile is smaller than tile size; pad will happen in transforms if needed
                pos_per_class = [(m_tile[..., c].sum() >= self.min_positive_area) for c in range(self.C)]
                has_any = any(pos_per_class)
                self.tiles.append(
                    TileInfo(
                        image_index=i,
                        id=iid,
                        y0=y0, x0=x0, y1=y1, x1=x1,
                        has_pos_any=has_any,
                        has_pos_per_class=pos_per_class,
                    )
                )

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        t = self.tiles[idx]
        ip = self.image_paths[t.image_index]
        iid = self.ids[t.image_index]

        img = load_image_rgb(ip)
        H, W = img.shape[:2]
        # Crop window (note: may be smaller than tile at borders; transform will pad)
        img_c = img[t.y0:t.y1, t.x0:t.x1, :]

        # Load masks and crop
        M = _stack_multilabel_masks(self.masks_dir, self.class_names, iid, size_hw=(H, W))  # (H,W,C)
        m_c = M[t.y0:t.y1, t.x0:t.x1, :]

        out = self.transform(image=img_c, mask=m_c)
        image_t = out["image"]  # (3,tile,tile)
        mask_t = out["mask"]    # (C,tile,tile)

        meta = {
            "id": t.id,
            "y0": t.y0, "x0": t.x0, "y1": t.y1, "x1": t.x1,
            "has_pos_any": t.has_pos_any,
            "has_pos_per_class": t.has_pos_per_class,
        }
        return {"image": image_t, "mask": mask_t, "meta": meta}


# ----------------------------
# Inference dataset (full-image, no masks)
# ----------------------------
class InferenceDataset(Dataset):
    """
    Returns full images prepared for inference (no masks).
    __getitem__ returns:
      - image: float tensor (3,H',W') after pad/normalize
      - meta : dict(id, size_hw_before, size_hw_after)
    """
    def __init__(
        self,
        ds_cfg: DatasetConfig,
        stage: str = "infer",
        aug_cfg: Optional[Dict[str, Any]] = None,
        normalize: str = "imagenet",
    ):
        assert stage == "infer", "InferenceDataset should be used with stage='infer'"
        self.ds_cfg = ds_cfg
        self.images_dir = ds_cfg.images_path
        self.image_paths: List[Path] = list_images(self.images_dir)
        self.ids: List[str] = [image_id_from_path(p) for p in self.image_paths]

        self.transform = build_transforms(
            aug_cfg or {},
            stage="infer",
            normalize=normalize,
            tile_size=ds_cfg.tile_size,
            tile_overlap=ds_cfg.tile_overlap,
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ip = self.image_paths[idx]
        iid = self.ids[idx]
        img = load_image_rgb(ip)
        H, W = img.shape[:2]
        out = self.transform(image=img, mask=None)
        image_t = out["image"]
        H2, W2 = image_t.shape[-2:]
        meta = {"id": iid, "size_hw_before": (H, W), "size_hw_after": (H2, W2), "image_path": str(ip)}
        return {"image": image_t, "meta": meta}
