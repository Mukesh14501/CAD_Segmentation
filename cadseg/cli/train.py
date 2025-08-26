# cadseg/cli/train.py
"""
CLI: train.py
End-to-end starter training script:
- Reads configs & (optional) train/valid split lists
- Builds tiled datasets and a class-balanced batch sampler
- Constructs UNet++(ResNet-34) model + Dice/BCE (or other) loss
- Trains with AMP, encoder warmup freeze/unfreeze, checkpointing, early stopping
- Runs stitched full-image validation each epoch (no tile leakage)

Usage (minimal):
  python -m cadseg.cli.train --configs configs --run_dir runs/exp1

Optional:
  # If you have splits in data/splits/train.txt and valid.txt (one image_id per line)
  python -m cadseg.cli.train --run_dir runs/exp1

  # Quick smoke run with fewer val images
  python -m cadseg.cli.train --limit_valid 10 --run_dir runs/exp_debug
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from cadseg.config import load_configs
from cadseg.dataio.datasets import TiledDataset
from cadseg.dataio.sampling import ClassBalancedBatchSampler
from cadseg.dataio.collate import default_collate
from cadseg.dataio.masks import list_images, image_id_from_path, load_image_rgb, per_class_mask_path, load_binary_mask
from cadseg.dataio.tiling import generate_grid
from cadseg.dataio.transforms import build_transforms
from cadseg.engine.eval_loop import infer_full_image_logits, _pick_device
from cadseg.models.builder import build_model
from cadseg.models.losses import build_loss
from cadseg.models.metrics import SegmentationMetrics
from cadseg.engine.train_loop import Trainer
import os

os.environ["http_proxy"] = "http://proxy50.adm.toyota.co.jp:15520"
os.environ["https_proxy"] = "http://proxy50.adm.toyota.co.jp:15520"

# ---------------------------
# Utility helpers
# ---------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def _read_id_list(path: Path) -> List[str]:
    if not path.exists():
        return []
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s)
    return ids

def _filter_dataset_to_ids(ds: TiledDataset, keep_ids: List[str]) -> None:
    """Mutate a TiledDataset to keep only images with IDs in keep_ids, then rebuild tile index."""
    if not keep_ids:
        return
    idset = set(keep_ids)
    new_paths = []
    new_ids = []
    for p, iid in zip(ds.image_paths, ds.ids):
        if iid in idset:
            new_paths.append(p)
            new_ids.append(iid)
    ds.image_paths = new_paths
    ds.ids = new_ids
    ds.tiles = []
    ds._build_tile_index()  # rebuild

def _auto_split_if_missing(images_dir: Path, seed: int, train_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
    """Create an in-memory random split by image_id if no split files present."""
    paths = list_images(images_dir)
    ids = [image_id_from_path(p) for p in paths]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_train = int(round(train_ratio * len(ids)))
    return ids[:n_train], ids[n_train:]

def _maybe_save_splits(root: Path, train_ids: List[str], valid_ids: List[str]) -> None:
    """Persist splits if not present so future runs are reproducible."""
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in [("train.txt", train_ids), ("valid.txt", valid_ids)]:
        p = splits_dir / name
        if not p.exists():
            with open(p, "w", encoding="utf-8") as f:
                f.write("\n".join(ids) + "\n")

# ---------------------------
# Validation (full-image, stitched)
# ---------------------------

@torch.inference_mode()
def _evaluate_on_ids(
    model: torch.nn.Module,
    class_names: List[str],
    ds_cfg,
    image_id_list: List[str],
    *,
    thresholds: float | List[float] = 0.5,
    tile_bs: int = 8,
    device: Optional[torch.device] = None,
    limit: Optional[int] = None,
) -> Dict:
    """
    Run stitched validation ONLY on the provided image IDs.
    Returns metrics dict with an extra top-level key 'macro_mIoU' for Trainer.
    """
    device = device or _pick_device()
    model.eval().to(device)

    # Map id -> path
    all_paths = list_images(ds_cfg.images_path)
    id2path = {image_id_from_path(p): p for p in all_paths}

    metrics = SegmentationMetrics(num_classes=len(class_names), thresholds=thresholds)

    count = 0
    # matched = 0
    for iid in image_id_list:
        # matched += 1
        if limit is not None and count >= limit:
            break
        if iid not in id2path:
            print(f"[warn] validation id '{iid}' not found under {ds_cfg.images_path}")
            continue

        ip = id2path[iid]
        img = load_image_rgb(ip)
        H, W = img.shape[:2]

        logits = infer_full_image_logits(model, img, ds_cfg, None, tile_bs=tile_bs, device=device)  # (C,H,W)

        # Build GT multi-label mask (C,H,W)
        ms = []
        for cname in class_names:
            mp = per_class_mask_path(ds_cfg.masks_path, cname, iid)
            if mp.exists():
                m = load_binary_mask(mp)
            else:
                m = np.zeros((H, W), dtype=np.uint8)
            ms.append(m.astype(np.float32))
        target = np.stack(ms, axis=0)

        # Update metrics
        lt = torch.from_numpy(logits).unsqueeze(0)
        tt = torch.from_numpy(target).unsqueeze(0)
        metrics.update(lt, tt)

        count += 1
    
    # print(f"[valid] evaluated images: {matched} / {len(image_id_list)}")

    try:
        out = metrics.compute()
    except Exception as e:
        print(f"[valid] metrics.compute() failed: {e}")
        out = None

    # If nothing was evaluated or something went wrong, return a safe default
    if out is None:
        out = {
            "per_class": {},
            "macro_present": {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "num_present_classes": 0},
            "macro_all":     {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0},
            "micro":         {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0},
        }

    out["macro_mIoU"] = out["macro_all"]["iou"]
    return out
    # out = metrics.compute()
    # # Provide a top-level macro metric key for Trainer monitoring
    # out["macro_mIoU"] = out["macro_all"]["iou"]
    # return out


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=str, default="configs", help="Path to configs directory.")
    ap.add_argument("--run_dir", type=str, default="", help="Directory to write checkpoints/metrics (auto if empty).")
    ap.add_argument("--limit_valid", type=int, default=0, help="Validate on only N images (speed/debug).")
    ap.add_argument("--tile_bs", type=int, default=8, help="Tile batch size during validation.")
    args = ap.parse_args()

    cfg = load_configs(args.configs)
    ds_cfg, md_cfg, tr_cfg, aug_cfg = cfg.dataset, cfg.model, cfg.train, cfg.aug

    set_seed(ds_cfg.seed)

    # Resolve / create run dir
    run_dir = Path(args.run_dir) if args.run_dir else Path(f"runs/exp_{_timestamp()}")
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ----- Splits -----
    splits_dir = ds_cfg.root_path / "splits"
    train_ids = _read_id_list(splits_dir / "train.txt")
    valid_ids = _read_id_list(splits_dir / "valid.txt")

    if not train_ids or not valid_ids:
        train_ids, valid_ids = _auto_split_if_missing(ds_cfg.images_path, seed=ds_cfg.seed, train_ratio=0.8)
        _maybe_save_splits(ds_cfg.root_path, train_ids, valid_ids)

    class_names = ds_cfg.classes
    C = len(class_names)

    # ----- Datasets & Loaders -----
    train_ds = TiledDataset(
        ds_cfg, class_names, stage="train", aug_cfg=aug_cfg, preload_index=False,
        min_positive_area=ds_cfg.min_positive_area,
    )
    _filter_dataset_to_ids(train_ds, train_ids)

    valid_ds = TiledDataset(
        ds_cfg, class_names, stage="valid", aug_cfg=aug_cfg, preload_index=False,
        min_positive_area=ds_cfg.min_positive_area,
    )
    _filter_dataset_to_ids(valid_ds, valid_ids)

    # Class-balanced sampler for training
    sampler = ClassBalancedBatchSampler(
        train_ds,
        batch_size=tr_cfg.batch_size,
        pos_fraction=0.75,
        batches_per_epoch=None,
        seed=ds_cfg.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=ds_cfg.num_workers,
        pin_memory=True,
        collate_fn=default_collate,
    )

    # ----- Model & Loss -----
    model = build_model(md_cfg)

    # Loss weights (optional): from train.yaml class_weights or leave None
    bce_pos_weight = tr_cfg.class_weights if tr_cfg.class_weights is not None else None
    dice_cls_w = tr_cfg.class_weights if tr_cfg.class_weights is not None else None

    loss_fn = build_loss(
        name=tr_cfg.loss,
        num_classes=C,
        bce_pos_weight=bce_pos_weight,
        dice_class_weights=dice_cls_w,
        focal_gamma=tr_cfg.focal_gamma,
        tversky_alpha=tr_cfg.tversky_alpha,
        tversky_beta=tr_cfg.tversky_beta,
    )

    # ----- Trainer -----
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        cfg=tr_cfg,
        save_dir=str(ckpt_dir),
    )

    device = _pick_device()

    # Validation callback using full-image stitched eval on the VALID split only
    def _validate(epoch: int) -> Dict:
        metrics = _evaluate_on_ids(
            model, class_names, ds_cfg, valid_ids,
            thresholds=0.5, tile_bs=args.tile_bs, device=device,
            limit=(args.limit_valid if args.limit_valid > 0 else None),
        )
        # Persist a small metrics snapshot per epoch
        (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        with open(run_dir / "metrics" / f"epoch_{epoch:03d}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[valid] epoch={epoch} macro_mIoU={metrics['macro_mIoU']:.4f}  "
              f"macroDice={metrics['macro_all']['dice']:.4f}")
        return metrics

    # ----- Fit -----
    hist = trainer.fit(train_loader, validate_fn=_validate, start_epoch=0)

    print("\nTraining complete.")
    print(f"Run directory: {run_dir.resolve()}")
    if (ckpt_dir / "best.pth").exists():
        print(f"Best checkpoint: {ckpt_dir / 'best.pth'}")

if __name__ == "__main__":
    main()
