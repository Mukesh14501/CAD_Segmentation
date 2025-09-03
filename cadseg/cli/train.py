# cadseg/cli/train.py
"""
CLI: train.py
End-to-end trainer with automatic class discovery & safe fine-tune on class-set changes.

- Auto-discovers classes from data/<...>/masks/<class_name>/ subfolders
- Rebuilds model head to len(classes) dynamically
- Loads existing checkpoints safely (skips head if class count changed)
- Fine-tunes on full train set when classes changed (to avoid forgetting)
- Optionally fine-tunes only "new IDs" otherwise

Usage:
  python -m cadseg.cli.train --configs configs --run_dir runs/exp1
  # quick debug:
  python -m cadseg.cli.train --configs configs --run_dir runs/exp_debug --limit_valid 10
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

# (Optional) corporate proxy env (safe no-op if not used)
os.environ.setdefault("http_proxy", os.environ.get("http_proxy", ""))
os.environ.setdefault("https_proxy", os.environ.get("https_proxy", ""))


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
# Class discovery & metadata
# ---------------------------
def _discover_classes(masks_root: Path) -> List[str]:
    """Return sorted list of class names from subfolders under masks_root."""
    if not masks_root.exists():
        return []
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}
    classes = []
    for sub in sorted(p for p in masks_root.iterdir() if p.is_dir()):
        # keep folder only if it contains any mask file
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


def _load_prev_train_ids(run_dir: Path) -> List[str]:
    p = _train_manifest_path(run_dir)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _save_train_ids(run_dir: Path, ids: List[str]) -> None:
    _train_manifest_path(run_dir).write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_prev_classes(run_dir: Path) -> List[str]:
    p = _classes_manifest_path(run_dir)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _save_classes(run_dir: Path, classes: List[str]) -> None:
    _classes_manifest_path(run_dir).write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_checkpoint_flexible(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Load only params with matching names AND shapes (skip head safely)."""
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    msd = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in msd and msd[k].shape == v.shape}
    # use strict=False so missing keys (e.g., new head) are left initialized
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return {
        "loaded": len(filtered),
        "skipped_from_ckpt": len(state) - len(filtered),
        "missing_in_model": len(missing),
        "unexpected_in_ckpt": len(unexpected),
    }


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

        # Update metrics (logits + targets as float)
        lt = torch.from_numpy(logits).unsqueeze(0).float()
        tt = torch.from_numpy(target).unsqueeze(0).float()
        metrics.update(lt, tt)

        count += 1

    out = metrics.compute()
    out["macro_mIoU"] = out["macro_all"]["iou"]
    return out


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=str, default="configs", help="Path to configs directory.")
    ap.add_argument("--run_dir", type=str, default="", help="Directory to write checkpoints/metrics (auto if empty).")
    ap.add_argument("--limit_valid", type=int, default=0, help="Validate on only N images (speed/debug).")
    ap.add_argument("--tile_bs", type=int, default=8, help="Tile batch size during validation.")
    # Fine-tune oriented flags
    ap.add_argument("--finetune_epochs", type=int, default=5, help="Epochs for fine-tuning if checkpoint exists.")
    ap.add_argument("--finetune_lr", type=float, default=None, help="Optional LR override for fine-tuning.")
    ap.add_argument("--freeze_encoder_stages_ft", type=int, default=0,
                    help="Freeze first N encoder stages during fine-tuning.")
    ap.add_argument("--force_full_finetune", action="store_true",
                    help="Fine-tune on ALL train IDs (not only new).")
    
    ap.add_argument("--replay_old_frac", type=float, default=0.5,
                help="When classes changed, fraction of OLD train IDs to replay (0..1). Default 0.5")

    args = ap.parse_args()

    cfg = load_configs(args.configs)
    ds_cfg, md_cfg, tr_cfg, aug_cfg = cfg.dataset, cfg.model, cfg.train, cfg.aug

    set_seed(ds_cfg.seed)

    # Resolve / create run dir
    run_dir = Path(args.run_dir) if args.run_dir else Path(f"runs/exp_{_timestamp()}")
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ----- Auto-discover classes from masks directory -----
    discovered = _discover_classes(ds_cfg.masks_path)
    if discovered:
        if set(discovered) != set(ds_cfg.classes):
            print(f"[classes] Overriding dataset classes from masks: {ds_cfg.classes} -> {discovered}")
        class_names = discovered
    else:
        class_names = list(ds_cfg.classes)  # fallback to config

    # ensure model.num_classes matches discovered classes
    md_cfg.num_classes = len(class_names)
    C = len(class_names)

    # ----- Splits -----
    splits_dir = ds_cfg.root_path / "splits"
    train_ids = _read_id_list(splits_dir / "train.txt")
    valid_ids = _read_id_list(splits_dir / "valid.txt")

    if not train_ids or not valid_ids:
        train_ids, valid_ids = _auto_split_if_missing(ds_cfg.images_path, seed=ds_cfg.seed, train_ratio=0.8)
        _maybe_save_splits(ds_cfg.root_path, train_ids, valid_ids)

    # ----- Fine-tune mode detection & delta -----
    best_ckpt = (ckpt_dir / "best.pth")
    has_ckpt = best_ckpt.exists()

    prev_train_ids = _load_prev_train_ids(run_dir)
    prev_classes = _load_prev_classes(run_dir)
    classes_changed = bool(prev_classes) and (prev_classes != class_names)

    new_train_ids = [iid for iid in train_ids if iid not in set(prev_train_ids)]

    finetune_mode = has_ckpt  # choose FT if a best checkpoint already exists
    if finetune_mode:
        print(f"[info] Found checkpoint for fine-tuning: {best_ckpt}")
        if classes_changed:
            print(f"[info] Class set changed: {prev_classes} -> {class_names}. "
                  f"Will fine-tune on FULL train set and load checkpoint head safely.")
        elif not args.force_full_finetune and len(new_train_ids) > 0:
            print(f"[info] Fine-tuning on NEW images only: {len(new_train_ids)} / {len(train_ids)}")
        elif not args.force_full_finetune:
            print("[warn] No NEW train IDs detected since last run. Falling back to full train set for fine-tuning.")

    # ----- Datasets & Loaders -----
    train_ds = TiledDataset(
    ds_cfg, class_names, stage="train", aug_cfg=aug_cfg, preload_index=False,
    min_positive_area=ds_cfg.min_positive_area,
    )
    valid_ds = TiledDataset(
        ds_cfg, class_names, stage="valid", aug_cfg=aug_cfg, preload_index=False,
        min_positive_area=ds_cfg.min_positive_area,
    )

    # ----- Choose which train IDs to use -----
    rng = random.Random(ds_cfg.seed)

    # Base: full train split
    train_ids_effective = list(train_ids)

    if finetune_mode and classes_changed:
        # Split current TRAIN IDs into "old" and "new" relative to previous run
        old_in_train = [iid for iid in train_ids if iid in set(prev_train_ids)]
        new_in_train = [iid for iid in train_ids if iid not in set(prev_train_ids)]

        # Keep all NEW images that are in the train split
        keep_new = list(new_in_train)  # ≈80% of total NEW images

        # Randomly replay a fraction of OLD images from the train split
        frac = max(0.0, min(1.0, float(args.replay_old_frac)))
        k_old = int(round(frac * len(old_in_train)))
        keep_old = rng.sample(old_in_train, k_old) if 0 < k_old < len(old_in_train) else list(old_in_train)

        train_ids_effective = keep_new + keep_old
        rng.shuffle(train_ids_effective)

        print(f"[ft] Class change replay -> using NEW(train)={len(keep_new)}  "
            f"+ OLD(train)={len(keep_old)}/{len(old_in_train)} (replay_old_frac={frac}).")
    elif finetune_mode and (not classes_changed) and (not args.force_full_finetune) and len(new_train_ids) > 0:
        # Prior behavior: fine-tune only on NEW images since last run
        train_ids_effective = list(new_train_ids)
        print(f"[ft] Using NEW(train) only: {len(train_ids_effective)} images.")
    else:
        print(f"[ft] Using FULL train split: {len(train_ids_effective)} images.")

    # Apply the selection to datasets
    _filter_dataset_to_ids(train_ds, train_ids_effective)
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

    # ----- Load checkpoint if fine-tuning (safe when classes changed) -----
    if finetune_mode:
        try:
            state = torch.load(best_ckpt, map_location="cpu")
            stats = _load_checkpoint_flexible(model, state)
            print(f"[ft] Restored {stats['loaded']} tensors from checkpoint "
                  f"(skipped_from_ckpt={stats['skipped_from_ckpt']}, "
                  f"missing_in_model={stats['missing_in_model']}, "
                  f"unexpected_in_ckpt={stats['unexpected_in_ckpt']}).")
        except Exception as e:
            print(f"[ft] Failed to load checkpoint '{best_ckpt}': {e}")
            print("[ft] Proceeding with training from scratch...")
            finetune_mode = False  # fallback

        # Adjust training config for FT
        if finetune_mode:
            if args.finetune_lr is not None:
                tr_cfg.lr = args.finetune_lr
                print(f"[ft] Overriding LR for fine-tune: {tr_cfg.lr}")
            if args.finetune_epochs is not None and args.finetune_epochs > 0:
                tr_cfg.epochs = args.finetune_epochs
                print(f"[ft] Overriding epochs for fine-tune: {tr_cfg.epochs}")
            if args.freeze_encoder_stages_ft and args.freeze_encoder_stages_ft > 0:
                tr_cfg.freeze_encoder_stages = args.freeze_encoder_stages_ft
                print(f"[ft] Freezing first {tr_cfg.freeze_encoder_stages} encoder stages during FT")

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

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

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
                # ignore any unexpected files
                pass
        return max_idx + 1

    epoch_offset = _next_epoch_offset(metrics_dir)
    print(f"[metrics] continuing from global epoch index = {epoch_offset}")
    
    # Validation callback using full-image stitched eval on the VALID split only
    def _validate(epoch: int) -> Dict:
        metrics = _evaluate_on_ids(
            model, class_names, ds_cfg, valid_ids,
            thresholds=0.5, tile_bs=args.tile_bs, device=device,
            limit=(args.limit_valid if args.limit_valid > 0 else None),
        )

        # compute global epoch index (so fine-tuning appends instead of overwriting)
        global_epoch = epoch_offset + epoch

        # Optional: embed a tiny meta block (handy when skimming files later)
        metrics.setdefault("meta", {})
        metrics["meta"].update({
            "epoch_local": int(epoch),
            "epoch_global": int(global_epoch),
            "classes": class_names,
        })

        # Persist metrics snapshot per epoch (no overwrite)
        with open(metrics_dir / f"epoch_{global_epoch:03d}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"[valid] epoch_local={epoch} global={global_epoch}  "
            f"macro_mIoU={metrics['macro_mIoU']:.4f}  "
            f"macroDice={metrics['macro_all']['dice']:.4f}")

        return metrics


    # ----- Fit -----
    hist = trainer.fit(train_loader, validate_fn=_validate, start_epoch=0)

    print("\nFine-tuning complete." if finetune_mode else "\nTraining complete.")
    print(f"Run directory: {run_dir.resolve()}")
    if (ckpt_dir / "best.pth").exists():
        print(f"Best checkpoint: {ckpt_dir / 'best.pth'}")

    # Save manifests for next run comparison
    try:
        _save_train_ids(run_dir, train_ids)
        _save_classes(run_dir, class_names)
    except Exception as e:
        print(f"[meta] Failed to save manifests: {e}")


if __name__ == "__main__":
    main()
