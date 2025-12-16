# cadseg/cli/train.py
"""
CLI: train.py
Simple end-to-end trainer with:
- Random train/val split over the entire dataset
- Fresh timestamped run directory under runs/<YYYYmmdd-HHMMSS>
- Optional resume from the latest previous best checkpoint via --resume

Usage:
  python -m cadseg.cli.train            # train from scratch
  python -m cadseg.cli.train --resume   # load latest best.pth then train
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random
import time
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from cadseg.config import load_configs
from cadseg.dataio.datasets import TiledDataset
from cadseg.dataio.sampling import ClassBalancedBatchSampler
from cadseg.dataio.collate import default_collate
from cadseg.dataio.masks import (
    list_images, image_id_from_path, load_image_rgb,
    per_class_mask_path, load_binary_mask,
)
from cadseg.engine.eval_loop import infer_full_image_logits, _pick_device
from cadseg.models.builder import build_model
from cadseg.models.losses import build_loss
from cadseg.models.metrics import SegmentationMetrics
from cadseg.engine.train_loop import Trainer

import mlflow
import mlflow.pytorch

# (Optional) corporate proxy env (safe no-op if not used)
os.environ.setdefault("http_proxy", os.environ.get("http_proxy", ""))
os.environ.setdefault("https_proxy", os.environ.get("https_proxy", ""))

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("CAD_Segmentation")
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
    ids: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s)
    return ids

def _write_id_list(path: Path, ids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(ids) + "\n")

def _filter_dataset_to_ids(ds: TiledDataset, keep_ids: List[str]) -> None:
    """Mutate a TiledDataset to keep only images with IDs in keep_ids, then rebuild tile index."""
    if not keep_ids:
        return
    idset = set(keep_ids)
    new_paths, new_ids = [], []
    for p, iid in zip(ds.image_paths, ds.ids):
        if iid in idset:
            new_paths.append(p)
            new_ids.append(iid)
    ds.image_paths = new_paths
    ds.ids = new_ids
    ds.tiles = []
    ds._build_tile_index()  # rebuild

def _all_image_ids(images_dir: Path) -> List[str]:
    paths = list_images(images_dir)
    ids = []
    for p in paths:
        stem = image_id_from_path(p)
        if stem.isdigit():
            ids.append(stem)
    ids.sort(key=lambda s: int(s))
    return ids

def _discover_classes(masks_root: Path) -> List[str]:
    """Return sorted list of class names from subfolders under masks_root."""
    if not masks_root.exists():
        return []
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}
    classes = []
    for sub in sorted(p for p in masks_root.iterdir() if p.is_dir()):
        has_files = any(fp.suffix.lower() in exts for fp in sub.glob("*"))
        if has_files:
            classes.append(sub.name)
    return classes

def _meta_dir(run_dir: Path) -> Path:
    d = run_dir / "meta"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _train_manifest_path(run_dir: Path) -> Path:
    return _meta_dir(run_dir) / "train_ids.json"

def _classes_manifest_path(run_dir: Path) -> Path:
    return _meta_dir(run_dir) / "classes.json"

def _save_train_ids(run_dir: Path, ids: List[str]) -> None:
    _train_manifest_path(run_dir).write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")

def _save_classes(run_dir: Path, classes: List[str]) -> None:
    _classes_manifest_path(run_dir).write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")

def _maybe_save_splits(root: Path, train_ids: List[str], valid_ids: List[str]) -> None:
    """Persist splits (overwrite with current results)."""
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    _write_id_list(splits_dir / "train.txt", train_ids)
    _write_id_list(splits_dir / "valid.txt", valid_ids)

# ---------------------------
# New: simple random split over the entire dataset
# ---------------------------
def _random_split_ids(
    images_dir: Path,
    seed: int,
    valid_ratio: float = 0.2,
) -> Tuple[List[str], List[str], Dict[str, int]]:
    ids = _all_image_ids(images_dir)
    rng = random.Random(seed)
    rng.shuffle(ids)
    k_valid = int(round(valid_ratio * len(ids)))
    valid_ids = sorted(ids[:k_valid], key=lambda s: int(s))
    train_ids = sorted(ids[k_valid:], key=lambda s: int(s))
    counters = {
        "total": len(ids),
        "train": len(train_ids),
        "valid": len(valid_ids),
    }
    return train_ids, valid_ids, counters

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
    for iid in image_id_list:
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

        lt = torch.from_numpy(logits).unsqueeze(0).float()
        tt = torch.from_numpy(target).unsqueeze(0).float()
        metrics.update(lt, tt)

        count += 1

    out = metrics.compute()
    out["macro_mIoU"] = out["macro_all"]["iou"]
    return out

def _load_checkpoint_flexible(model, state):
    model_state = model.state_dict()
    loaded, skipped_from_ckpt, missing_in_model, unexpected_in_ckpt = 0, 0, 0, 0
    new_state = {}

    for k, v in state["model"].items():
        if k in model_state and model_state[k].shape == v.shape:
            new_state[k] = v
            loaded += 1
        else:
            skipped_from_ckpt += 1

    unexpected_in_ckpt = len(state["model"]) - (loaded + skipped_from_ckpt)
    missing_in_model = len(model_state) - len(new_state)

    model_state.update(new_state)
    model.load_state_dict(model_state, strict=False)
    return {
        "loaded": loaded,
        "skipped_from_ckpt": skipped_from_ckpt,
        "missing_in_model": missing_in_model,
        "unexpected_in_ckpt": unexpected_in_ckpt,
    }

def _find_latest_best_checkpoint(runs_dir: Path) -> Optional[Path]:
    """Scan runs/*/checkpoints/best.pth and return the newest by mtime, if any."""
    candidates = []
    if runs_dir.exists():
        for run in runs_dir.iterdir():
            if not run.is_dir():
                continue
            p = run / "checkpoints" / "best.pth"
            if p.exists():
                candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="If set, load latest previous best.pth before training.")
    args = ap.parse_args()

    # Load configs (path fixed to 'configs' as per your project structure)
    cfg = load_configs("configs")
    ds_cfg, md_cfg, tr_cfg, aug_cfg = cfg.dataset, cfg.model, cfg.train, cfg.aug

    # Reproducible randomness for the random split
    set_seed(ds_cfg.seed)

    # Fresh timestamped run directory
    run_dir = Path("runs") / _timestamp()
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] Run directory: {run_dir.resolve()}")

    # ----- Auto-discover classes from masks directory -----
    discovered = _discover_classes(ds_cfg.masks_path)
    if discovered:
        if set(discovered) != set(ds_cfg.classes):
            print(f"[classes] Overriding dataset classes from masks: {ds_cfg.classes} -> {discovered}")
        class_names = discovered
    else:
        class_names = list(ds_cfg.classes)  # fallback to config
    md_cfg.num_classes = len(class_names)
    C = len(class_names)

    # ----- Random split over the entire dataset -----
    valid_ratio = getattr(ds_cfg, "valid_ratio", 0.2)  # default 20% valid if not present
    train_ids, valid_ids, ctrs = _random_split_ids(ds_cfg.images_path, seed=ds_cfg.seed, valid_ratio=valid_ratio)
    _maybe_save_splits(ds_cfg.root_path, train_ids, valid_ids)
    print(f"[split] total={ctrs['total']}  train={ctrs['train']}  valid={ctrs['valid']}  (valid_ratio={valid_ratio})")

    # ----- Datasets & Loaders -----
    train_ds = TiledDataset(
        ds_cfg, class_names, stage="train", aug_cfg=aug_cfg, preload_index=False,
        min_positive_area=ds_cfg.min_positive_area,
    )
    valid_ds = TiledDataset(
        ds_cfg, class_names, stage="valid", aug_cfg=aug_cfg, preload_index=False,
        min_positive_area=ds_cfg.min_positive_area,
    )
    _filter_dataset_to_ids(train_ds, train_ids)
    _filter_dataset_to_ids(valid_ds, valid_ids)

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

    # ----- Model, optional resume, loss -----
    model = build_model(md_cfg)

    if args.resume:
        latest_ckpt = _find_latest_best_checkpoint(Path("runs"))
        if latest_ckpt is None:
            print("[resume] No previous best.pth found under runs/*; proceeding from scratch.")
        else:
            try:
                state = torch.load(latest_ckpt, map_location="cpu")
                stats = _load_checkpoint_flexible(model, state)
                print(f"[resume] Loaded '{latest_ckpt}' "
                      f"(loaded={stats['loaded']}, skipped={stats['skipped_from_ckpt']}, "
                      f"missing_in_model={stats['missing_in_model']}, unexpected_in_ckpt={stats['unexpected_in_ckpt']}).")
            except Exception as e:
                print(f"[resume] Failed to load '{latest_ckpt}': {e}")
                print("[resume] Proceeding from scratch.")

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

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        cfg=tr_cfg,
        save_dir=str(ckpt_dir),
    )

    device = _pick_device()

    def _next_epoch_offset(dirpath: Path) -> int:
        """Return max existing epoch index + 1, or 0 if none."""
        max_idx = -1
        for p in dirpath.glob("epoch_*.json"):
            stem = p.stem  # e.g., "epoch_012"
            try:
                idx = int(stem.split("_")[-1])
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                pass
        return max_idx + 1

    epoch_offset = _next_epoch_offset(metrics_dir)
    print(f"[metrics] starting global epoch index at = {epoch_offset}")

    # Validation callback using full-image stitched eval on the VALID split only
    def _validate(epoch: int) -> Dict:
        metrics = _evaluate_on_ids(
            model, class_names, ds_cfg, valid_ids,
            thresholds=0.5, tile_bs=getattr(tr_cfg, "tile_bs", 8), device=device,
            limit=None,
        )

        global_epoch = epoch_offset + epoch
        metrics.setdefault("meta", {})
        metrics["meta"].update({
            "epoch_local": int(epoch),
            "epoch_global": int(global_epoch),
            "classes": class_names,
        })

        with open(metrics_dir / f"epoch_{global_epoch:03d}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)  

        print(f"[valid] epoch_local={epoch} global={global_epoch}  "
              f"macro_mIoU={metrics['macro_mIoU']:.4f}  "
              f"macroDice={metrics['macro_all']['dice']:.4f}")
        
        return metrics

    # ----- Fit -----
    trainer.fit(train_loader, validate_fn=_validate, start_epoch=0)

    print("\nTraining complete.")
    print(f"Run directory: {run_dir.resolve()}")
    if (ckpt_dir / "best.pth").exists():
        print(f"Best checkpoint: {ckpt_dir / 'best.pth'}")

    # Save run-level manifests (optional)
    try:
        _save_train_ids(run_dir, train_ids)
        _save_classes(run_dir, class_names)
    except Exception as e:
        print(f"[meta] Failed to save run manifests: {e}")

if __name__ == "__main__":
    main()


