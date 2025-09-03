# cadseg/cli/train.py
"""
CLI: train.py
End-to-end trainer with automatic class discovery, split policy, and safe fine-tune.

Split policy (owned by trainer; API only ingests):
- If NEW CLASSES are added:
    train = 80% of NEW data + 50% of OLD train
    val   = OLD val + 20% of NEW data
- If NO NEW CLASSES:
    train = OLD train + 80% of NEW data
    val   = OLD val   + 20% of NEW data

'NEW data' = images with numeric ID > last_trained_max_id in data/meta/state.json

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

# ---------------------------
# Data/meta state (dataset-level)
# ---------------------------
def _data_meta_dir(ds_root: Path) -> Path:
    d = ds_root / "meta"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _state_json_path(ds_root: Path) -> Path:
    return _data_meta_dir(ds_root) / "state.json"

def _read_state(ds_root: Path) -> Dict:
    p = _state_json_path(ds_root)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"last_upload_max_id": 0, "last_trained_max_id": 0, "prev_classes": []}

def _write_state(ds_root: Path, state: Dict) -> None:
    _state_json_path(ds_root).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

def _all_image_ids(images_dir: Path) -> List[str]:
    paths = list_images(images_dir)
    ids = []
    for p in paths:
        stem = image_id_from_path(p)
        if stem.isdigit():
            ids.append(stem)
    ids.sort(key=lambda s: int(s))
    return ids

# ---------------------------
# Class discovery & run-level manifests
# ---------------------------
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

def _maybe_save_splits(root: Path, train_ids: List[str], valid_ids: List[str]) -> None:
    """Persist splits (overwrite with current policy results)."""
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    _write_id_list(splits_dir / "train.txt", train_ids)
    _write_id_list(splits_dir / "valid.txt", valid_ids)

# ---------------------------
# Split policy builder (the heart of your requirement)
# ---------------------------
def _rebuild_splits_per_policy(
    *,
    ds_root: Path,
    images_dir: Path,
    discovered_classes: List[str],
    seed: int,
    train_ratio_new: float = 0.8,
    old_fraction_replay: float = 0.5,
) -> Tuple[List[str], List[str], Dict[str, int], bool]:
    """
    Build splits following the user's policy.

    Returns:
        train_ids, valid_ids, counters, new_classes_added
    """
    rng = random.Random(seed)
    splits_dir = ds_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    old_train = _read_id_list(splits_dir / "train.txt")
    old_valid = _read_id_list(splits_dir / "valid.txt")

    state = _read_state(ds_root)
    prev_classes = state.get("prev_classes", [])
    last_trained_max_id = int(state.get("last_trained_max_id", 0))

    # Detect class change
    new_classes_added = (len(prev_classes) > 0) and (set(discovered_classes) != set(prev_classes))
    if not prev_classes:
        # First run: treat as "no new classes" for policy; we'll still do 80/20 on the entire set as NEW.
        new_classes_added = False

    all_ids_sorted = _all_image_ids(images_dir)
    # NEW = strictly greater than last_trained_max_id
    new_ids = [iid for iid in all_ids_sorted if int(iid) > last_trained_max_id]

    # deterministically split NEW into 80/20 by order
    k_train_new = int(round(train_ratio_new * len(new_ids)))
    new_train = list(new_ids[:k_train_new])
    new_valid = list(new_ids[k_train_new:])

    def _uniq(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for s in seq:
            if s not in seen:
                out.append(s)
                seen.add(s)
        return out

    if new_classes_added:
        # A) NEW CLASSES ADDED
        # - train = 80% NEW + 50% OLD_TRAIN (sample)
        # - val   = OLD_VAL + 20% NEW
        k_replay = int(round(old_fraction_replay * len(old_train)))
        replay_old = rng.sample(old_train, k_replay) if k_replay > 0 else []
        train_ids = _uniq(replay_old + new_train)
        valid_ids = _uniq(old_valid + new_valid)
    else:
        # B) NO NEW CLASSES ADDED
        # - train = OLD_TRAIN + 80% NEW
        # - val   = OLD_VAL   + 20% NEW
        train_ids = _uniq(old_train + new_train)
        valid_ids = _uniq(old_valid + new_valid)

    _maybe_save_splits(ds_root, train_ids, valid_ids)

    counters = {
        "new_total": len(new_ids),
        "new_train": len(new_train),
        "new_valid": len(new_valid),
        "old_train_before": len(old_train),
        "old_valid_before": len(old_valid),
        "train_total_now": len(train_ids),
        "valid_total_now": len(valid_ids),
    }
    return train_ids, valid_ids, counters, new_classes_added

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
                    help="Ignore split deltas and just train on current FULL train split.")
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

    # ----- Build/extend splits PER POLICY (owned by trainer) -----
    train_ids, valid_ids, ctrs, new_classes_added = _rebuild_splits_per_policy(
        ds_root=ds_cfg.root_path,
        images_dir=ds_cfg.images_path,
        discovered_classes=class_names,
        seed=ds_cfg.seed,
        train_ratio_new=0.8,
        old_fraction_replay=args.replay_old_frac,
    )

    print(f"[policy] new_classes_added={new_classes_added}")
    print(f"[splits] Δnew={ctrs['new_total']}  new_train={ctrs['new_train']}  new_valid={ctrs['new_valid']}  "
          f"train_total_now={ctrs['train_total_now']}  valid_total_now={ctrs['valid_total_now']}")

    # ----- Datasets & Loaders -----
    train_ds = TiledDataset(
        ds_cfg, class_names, stage="train", aug_cfg=aug_cfg, preload_index=False,
        min_positive_area=ds_cfg.min_positive_area,
    )
    valid_ds = TiledDataset(
        ds_cfg, class_names, stage="valid", aug_cfg=aug_cfg, preload_index=False,
        min_positive_area=ds_cfg.min_positive_area,
    )

    # Apply our splits to datasets
    _filter_dataset_to_ids(train_ds, train_ids)
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

    # ----- (Optional) checkpoint restore for fine-tuning -----
    best_ckpt = (ckpt_dir / "best.pth")
    has_ckpt = best_ckpt.exists()

    if has_ckpt:
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

        # Adjust training config for FT (epochs/LR/freeze) if checkpoint existed
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
                pass
        return max_idx + 1

    epoch_offset = _next_epoch_offset(metrics_dir)
    print(f"[metrics] continuing from global epoch index = {epoch_offset}")

    # Validation callback using full-image stitched eval on the VALID split only
    def _validate(epoch: int) -> Dict:
        # IMPORTANT: validate on the *current* valid split we've built
        metrics = _evaluate_on_ids(
            model, class_names, ds_cfg, valid_ids,
            thresholds=0.5, tile_bs=args.tile_bs, device=device,
            limit=(args.limit_valid if args.limit_valid > 0 else None),
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

    # ---- Update DATA meta (dataset-level state) ----
    try:
        state = _read_state(ds_cfg.root_path)
        all_ids_sorted = _all_image_ids(ds_cfg.images_path)
        state["last_trained_max_id"] = int(all_ids_sorted[-1]) if all_ids_sorted else state.get("last_trained_max_id", 0)
        state["prev_classes"] = class_names
        _write_state(ds_cfg.root_path, state)
        print(f"[data-meta] last_trained_max_id={state['last_trained_max_id']}  prev_classes={state['prev_classes']}")
    except Exception as e:
        print(f"[data-meta] Failed to update dataset state.json: {e}")

if __name__ == "__main__":
    main()
