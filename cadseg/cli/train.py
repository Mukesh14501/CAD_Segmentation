# cadseg/cli/train.py
"""
CLI: train.py (MLflow-integrated)
- Random train/val split over the entire dataset
- Logs ALL params/metrics/artifacts to MLflow
- Keeps the best checkpoint in-memory only (no local files)
- Logs and registers the best model to MLflow Model Registry

Usage:
  python -m cadseg.cli.train
  python -m cadseg.cli.train --resume   # tries to load latest model from MLflow Registry
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import random
import time
import os
from copy import deepcopy

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
from mlflow.tracking import MlflowClient

# (Optional) corporate proxy env (safe no-op if not used)
os.environ.setdefault("http_proxy", os.environ.get("http_proxy", ""))
os.environ.setdefault("https_proxy", os.environ.get("https_proxy", ""))

# ---------- MLflow defaults (change if you use a server/S3/DB backend) ----------
mlflow.set_tracking_uri("file:./mlruns")
DEFAULT_EXPERIMENT_NAME = "CAD_Segmentation"
DEFAULT_REGISTERED_MODEL = "CAD_Segmentation"
mlflow.set_experiment(DEFAULT_EXPERIMENT_NAME)


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
    counters = {"total": len(ids), "train": len(train_ids), "valid": len(valid_ids)}
    return train_ids, valid_ids, counters

def _to_plain(obj: Any) -> Any:
    """Best-effort conversion of config objects to plain dicts for MLflow logging."""
    try:
        # dataclass?
        from dataclasses import asdict, is_dataclass
        if is_dataclass(obj):
            return asdict(obj)
    except Exception:
        pass
    # objects with __dict__
    if hasattr(obj, "__dict__"):
        out = {}
        for k, v in obj.__dict__.items():
            if not k.startswith("_"):
                out[k] = _to_plain(v)
        return out
    # lists/tuples
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    # dicts
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    # primitives
    return obj

def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dicts for MLflow params/metrics."""
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            # cast list to json string for params; numbers pass through
            if isinstance(v, (list, tuple, set)):
                items[new_key] = json.dumps(list(v), ensure_ascii=False)
            else:
                items[new_key] = v
    return items

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
) -> Dict[str, Any]:
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

    out: Dict[str, Any] = metrics.compute()
    out["macro_mIoU"] = float(out["macro_all"]["iou"])
    return out

def _load_checkpoint_flexible(model, state):
    model_state = model.state_dict()
    loaded, skipped_from_ckpt = 0, 0
    new_state = {}

    for k, v in state["model"].items():
        if k in model_state and model_state[k].shape == v.shape:
            new_state[k] = v
            loaded += 1
        else:
            skipped_from_ckpt += 1

    model_state.update(new_state)
    model.load_state_dict(model_state, strict=False)
    missing_in_model = len(model_state) - len(new_state)
    unexpected_in_ckpt = len(state["model"]) - (loaded + skipped_from_ckpt)
    return {
        "loaded": loaded,
        "skipped_from_ckpt": skipped_from_ckpt,
        "missing_in_model": missing_in_model,
        "unexpected_in_ckpt": unexpected_in_ckpt,
    }

def _try_resume_from_registry(model, registered_model_name: str) -> None:
    """
    If --resume is passed, try to load the latest version from MLflow Model Registry.
    Prefers 'Production', then 'Staging', else highest version number.
    """
    try:
        client = MlflowClient()
        mv = None
        # prefer stage
        for stage in ["Production", "Staging"]:
            vs = client.get_latest_versions(registered_model_name, stages=[stage])
            if vs:
                mv = vs[0]
                break
        if mv is None:
            # fallback to highest version
            all_versions = client.search_model_versions(f"name='{registered_model_name}'")
            if all_versions:
                mv = sorted(all_versions, key=lambda x: int(x.version))[-1]
        if mv is None:
            print(f"[resume] No versions found for registered model '{registered_model_name}'.")
            return
        uri = f"models:/{registered_model_name}/{mv.version}"
        print(f"[resume] Loading model from registry: {uri}")
        loaded = mlflow.pytorch.load_model(uri)
        # load weights into current architecture
        state = {"model": loaded.state_dict()}
        stats = _load_checkpoint_flexible(model, state)
        print(f"[resume] Loaded from registry (loaded={stats['loaded']}, skipped={stats['skipped_from_ckpt']}).")
    except Exception as e:
        print(f"[resume] Failed to resume from registry: {e}")


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="If set, try to load the latest registered model from MLflow before training.")
    ap.add_argument("--registered_model_name", type=str, default=DEFAULT_REGISTERED_MODEL,
                    help="MLflow Registered Model name to use.")
    args = ap.parse_args()

    # Load configs (path fixed to 'configs' as per your project structure)
    cfg = load_configs("configs")
    ds_cfg, md_cfg, tr_cfg, aug_cfg = cfg.dataset, cfg.model, cfg.train, cfg.aug

    # Reproducible randomness for the random split
    set_seed(ds_cfg.seed)

    # ----- Auto-discover classes from masks directory -----
    discovered = _discover_classes(ds_cfg.masks_path)
    if discovered and set(discovered) != set(ds_cfg.classes):
        print(f"[classes] Overriding dataset classes from masks: {ds_cfg.classes} -> {discovered}")
        class_names = discovered
    else:
        class_names = list(ds_cfg.classes)
    md_cfg.num_classes = len(class_names)
    C = len(class_names)

    # ----- Random split over the entire dataset -----
    valid_ratio = getattr(ds_cfg, "valid_ratio", 0.2)
    train_ids, valid_ids, ctrs = _random_split_ids(ds_cfg.images_path, seed=ds_cfg.seed, valid_ratio=valid_ratio)
    print(f"[split] total={ctrs['total']}  train={ctrs['train']}  valid={ctrs['valid']}  (valid_ratio={valid_ratio})")

    # ----- Datasets & Loader -----
    train_ds = TiledDataset(
        ds_cfg, class_names, stage="train", aug_cfg=aug_cfg, preload_index=False,
        min_positive_area=ds_cfg.min_positive_area,
    )
    valid_ds = TiledDataset(
        ds_cfg, class_names, stage="valid", aug_cfg=aug_cfg, preload_index=False,
        min_positive_area=ds_cfg.min_positive_area,
    )

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

    # ----- Model / Loss -----
    model = build_model(md_cfg)

    # Optional resume from MLflow Model Registry
    if args.resume:
        _try_resume_from_registry(model, args.registered_model_name)

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
        save_dir=None,  # <-- disable trainer file I/O; we’ll handle model saving via MLflow
    )

    device = _pick_device()

    # ---------- MLflow run ----------
    run_name = f"run-{_timestamp()}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        # ---- Log params (flattened) ----
        params = {
            "env.device": str(device),
            "env.cuda_available": torch.cuda.is_available(),
            "env.torch_version": torch.__version__,
            "env.seed": int(ds_cfg.seed),
            "data.num_classes": int(C),
            "data.class_names": json.dumps(class_names, ensure_ascii=False),
            "split.valid_ratio": valid_ratio,
            "split.counts": json.dumps(ctrs),
        }
        # dump configs
        params.update({f"cfg.dataset.{k}": v for k, v in _flatten_dict(_to_plain(ds_cfg)).items()})
        params.update({f"cfg.model.{k}": v for k, v in _flatten_dict(_to_plain(md_cfg)).items()})
        params.update({f"cfg.train.{k}": v for k, v in _flatten_dict(_to_plain(tr_cfg)).items()})
        params.update({f"cfg.aug.{k}": v for k, v in _flatten_dict(_to_plain(aug_cfg)).items()})
        # Convert all values to strings where needed
        params = {k: (str(v) if isinstance(v, (dict, list, tuple)) else v) for k, v in params.items()}
        mlflow.log_params(params)

        # Log split & classes as artifacts
        mlflow.log_text("\n".join(train_ids) + "\n", artifact_file="splits/train.txt")
        mlflow.log_text("\n".join(valid_ids) + "\n", artifact_file="splits/valid.txt")
        mlflow.log_text(json.dumps(class_names, ensure_ascii=False, indent=2), artifact_file="meta/classes.json")

        # ------ In-memory best checkpoint tracking ------
        best_metric = -1.0
        best_state_dict = None

        # Validation callback using stitched eval on VALID split
        def _validate(epoch: int) -> Dict[str, Any]:
            nonlocal best_metric, best_state_dict

            # ---- Eval ----
            metrics = _evaluate_on_ids(
                model, class_names, ds_cfg, valid_ids,
                thresholds=0.5,
                tile_bs=getattr(tr_cfg, "tile_bs", 8),
                device=device,
                limit=None,
            )

            # ---- Meta ----
            metrics.setdefault("meta", {}).update({
                "epoch_local": int(epoch),
                "classes": class_names,
            })

            # ---- Log numerics (flattened) ----
            flat = _flatten_dict(metrics)
            numeric = {k: float(v) for k, v in flat.items() if isinstance(v, (int, float, np.floating))}

            # Short aliases
            macro_miou = float(metrics.get("macro_mIoU", float("nan")))
            macro_dice = float(metrics.get("macro_all", {}).get("dice", float("nan")))
            numeric["macro_mIoU"] = macro_miou
            numeric["macroDice"] = macro_dice

            mlflow.log_metrics(numeric, step=epoch)

            # ---- Persist full structured metrics for this epoch ----
            mlflow.log_dict(metrics, f"metrics/epoch_{epoch:03d}.json")

            # ---- Per-class scalars (so they render in MLflow UI) ----
            pc = metrics.get("per_class") or {}
            for key in ("iou", "dice", "precision", "recall", "tp", "fp", "fn", "tn", "support_px"):
                vals = pc.get(key, [])
                for i, val in enumerate(vals if isinstance(vals, list) else []):
                    if isinstance(val, (int, float, np.floating)):
                        mlflow.log_metric(f"per_class.{key}[{i}]", float(val), step=epoch)

            # ---- Track best ----
            if macro_miou > best_metric:
                best_metric = macro_miou
                best_state_dict = deepcopy(model.state_dict())

            print(f"[valid] epoch={epoch}  macro_mIoU={macro_miou:.4f}  macroDice={macro_dice:.4f}")
            return metrics


        # ----- Fit -----
        trainer.fit(train_loader, validate_fn=_validate, start_epoch=0)

        print("\nTraining complete.")

        # ----- Log and Register the BEST model -----
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        else:
            print("[warn] best_state_dict was None; logging the final model weights.")

        # Log model artifact to this run
        artifact_path = "model"

        # Logs the model to the run *and* registers it under the given name.
        # (Requires MLflow >= 2.x where flavors support `registered_model_name`.)
        info = mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=args.registered_model_name,
        )

        print(f"[mlflow] Logged & registered model as '{args.registered_model_name}'.")
        print(f"[mlflow] Model URI: {info.model_uri}")


if __name__ == "__main__":
    main()
