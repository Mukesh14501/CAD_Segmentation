# cadseg/cli/train.py
"""
CLI: train.py (MLflow-integrated)
- Random train/val split over the entire dataset
- Logs params/metrics/artifacts to MLflow
- Tracks best checkpoint in-memory (no local ckpt files)
- Logs & registers the best model in the MLflow Model Registry

Usage:
  python -m cadseg.cli.train
  python -m cadseg.cli.train --resume
"""
from __future__ import annotations

import argparse
import json
import os
import time
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


# ---------------------------
# MLflow defaults (override in your env if needed)
# ---------------------------
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))

DEFAULT_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "CAD_Segmentation")
DEFAULT_REGISTERED_MODEL = os.getenv("MLFLOW_REGISTERED_MODEL", "CAD_Segmentation")
mlflow.set_experiment(DEFAULT_EXPERIMENT)

# Optional proxy passthrough (no-op if unset)
os.environ["http_proxy"] = "http://proxy50.adm.toyota.co.jp:15520"
os.environ["https_proxy"] = "http://proxy50.adm.toyota.co.jp:15520"

# ---------------------------
# Utilities
# ---------------------------


def _ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _all_image_ids(images_dir: Path) -> List[str]:
    ids: List[str] = []
    for p in list_images(images_dir):
        stem = image_id_from_path(p)
        if stem.isdigit():
            ids.append(stem)
    ids.sort(key=lambda s: int(s))
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

def _discover_classes(masks_root: Path) -> List[str]:
    if not masks_root.exists():
        return []
    img_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}
    classes = []
    for sub in sorted(p for p in masks_root.iterdir() if p.is_dir()):
        if any(fp.suffix.lower() in img_exts for fp in sub.glob("*")):
            classes.append(sub.name)
    return classes

def _random_split_ids(images_dir: Path, seed: int, valid_ratio: float) -> Tuple[List[str], List[str], Dict[str, int]]:
    ids = _all_image_ids(images_dir)
    rng = random.Random(seed)
    rng.shuffle(ids)
    k_valid = int(round(valid_ratio * len(ids)))
    valid_ids = sorted(ids[:k_valid], key=lambda s: int(s))
    train_ids = sorted(ids[k_valid:], key=lambda s: int(s))
    return train_ids, valid_ids, {"total": len(ids), "train": len(train_ids), "valid": len(valid_ids)}

def _to_plain(obj: Any) -> Any:
    try:
        from dataclasses import asdict, is_dataclass
        if is_dataclass(obj):
            return asdict(obj)
    except Exception:
        pass
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: _to_plain(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return obj

def _flatten(d: Dict[str, Any], prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        nk = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, nk, sep))
        else:
            out[nk] = json.dumps(list(v), ensure_ascii=False) if isinstance(v, (list, tuple, set)) else v
    return out

def _evaluate_on_ids(
    model: torch.nn.Module,
    class_names: List[str],
    ds_cfg,
    image_ids: List[str],
    *,
    thresholds: float | List[float] = 0.5,
    tile_bs: int = 8,
    device: Optional[torch.device] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Stitched validation over provided image IDs.
    Returns metrics dict; also populates top-level 'macro_mIoU' for convenience.
    """
    device = device or _pick_device()
    model.eval().to(device)

    id2path = {image_id_from_path(p): p for p in list_images(ds_cfg.images_path)}
    metrics = SegmentationMetrics(num_classes=len(class_names), thresholds=thresholds)

    processed = 0
    print(image_ids)
    for iid in image_ids:
        if limit is not None and processed >= limit:
            break
        if iid not in id2path:
            print(f"[warn] validation id '{iid}' not found in {ds_cfg.images_path}")
            continue

        img = load_image_rgb(id2path[iid])
        H, W = img.shape[:2]

        logits = infer_full_image_logits(model, img, ds_cfg, None, tile_bs=tile_bs, device=device)  # (C,H,W)

        # Build gt (C,H,W)
        gt = []
        for cname in class_names:
            mp = per_class_mask_path(ds_cfg.masks_path, cname, iid)
            m = load_binary_mask(mp) if mp.exists() else np.zeros((H, W), dtype=np.uint8)
            gt.append(m.astype(np.float32))
        target = np.stack(gt, axis=0)

        metrics.update(
            torch.from_numpy(logits).unsqueeze(0).float(),
            torch.from_numpy(target).unsqueeze(0).float(),
        )
        processed += 1

    out: Dict[str, Any] = metrics.compute()
    out["macro_mIoU"] = float(out["macro_all"]["iou"])
    return out

def _load_checkpoint_flexible(model: torch.nn.Module, state: Dict[str, Any]) -> Dict[str, int]:
    model_state = model.state_dict()
    loaded = 0
    skipped_from_ckpt = 0
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

def _try_resume_from_registry(model: torch.nn.Module, name: str) -> None:
    """
    Prefer 'Production', then 'Staging', else highest version.
    Loads registry weights into the current architecture (flexible match).
    """
    try:
        client = MlflowClient()
        mv = None
        for stage in ("Production", "Staging"):
            vs = client.get_latest_versions(name, stages=[stage])
            if vs:
                mv = vs[0]
                break
        if mv is None:
            versions = client.search_model_versions(f"name='{name}'")
            if versions:
                mv = sorted(versions, key=lambda x: int(x.version))[-1]
        if mv is None:
            print(f"[resume] No versions found for '{name}'.")
            return
        uri = f"models:/{name}/{mv.version}"
        print(f"[resume] Loading {uri}")
        loaded = mlflow.pytorch.load_model(uri)
        stats = _load_checkpoint_flexible(model, {"model": loaded.state_dict()})
        print(f"[resume] Loaded (loaded={stats['loaded']}, skipped={stats['skipped_from_ckpt']}).")
    except Exception as e:
        print(f"[resume] Failed to resume from registry: {e}")


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    best_epoch = -1
    best_metric = -1.0
    best_metrics_blob = {}
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Try loading latest model from MLflow Registry.")
    parser.add_argument("--registered_model_name", type=str, default=DEFAULT_REGISTERED_MODEL)
    args = parser.parse_args()

    # ---- Load configs ----
    cfg = load_configs("configs")
    ds_cfg, md_cfg, tr_cfg, aug_cfg = cfg.dataset, cfg.model, cfg.train, cfg.aug

    # ---- Seed & device ----
    set_seed(ds_cfg.seed)
    device = _pick_device()

    # ---- Classes ----
    discovered = _discover_classes(ds_cfg.masks_path)
    class_names = discovered if discovered and set(discovered) != set(ds_cfg.classes) else list(ds_cfg.classes)
    if discovered and set(discovered) != set(ds_cfg.classes):
        print(f"[classes] Overriding dataset classes: {ds_cfg.classes} -> {class_names}")
    md_cfg.num_classes = len(class_names)
    C = len(class_names)

    # ---- Split ----
    valid_ratio = float(getattr(ds_cfg, "valid_ratio", 0.2))
    train_ids, valid_ids, counts = _random_split_ids(ds_cfg.images_path, seed=ds_cfg.seed, valid_ratio=valid_ratio)
    print(f"[split] total={counts['total']} train={counts['train']} valid={counts['valid']} (valid_ratio={valid_ratio})")

    # ---- Datasets & Loader ----


    train_ds = TiledDataset(
        ds_cfg, class_names, stage="train", aug_cfg=aug_cfg, preload_index=False,
        min_positive_area=ds_cfg.min_positive_area
 
    )

    valid_ds = TiledDataset(
        ds_cfg, class_names, stage="valid", aug_cfg=aug_cfg, preload_index=False,
        min_positive_area=ds_cfg.min_positive_area
    )

    _filter_dataset_to_ids(train_ds, train_ids)
    _filter_dataset_to_ids(valid_ds, valid_ids)


    sampler = ClassBalancedBatchSampler(
            train_ds,
            batch_size=tr_cfg.batch_size,
            pos_fraction=0.5,        
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

    # ---- Model & Loss ----
    model = build_model(md_cfg)

    if args.resume:
        _try_resume_from_registry(model, args.registered_model_name)

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
        save_dir=None,  # disable on-disk checkpoints; MLflow will own persistence
    )

    # ----------------- MLflow run -----------------
    run_name = f"cadseg-{_ts()}"
    with mlflow.start_run(run_name=run_name) as run:
        # Tags help filtering in UI
        mlflow.set_tags({
            "project": "cadseg",
            "task": "segmentation",
            "device": str(device),
            "num_classes": str(C),
            "classes": json.dumps(class_names, ensure_ascii=False),
        })

        # Params
        param_blob = {
            "env": {
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "device": str(device),
                "seed": int(ds_cfg.seed),
            },
            "split": {"valid_ratio": valid_ratio, "counts": counts},
            "cfg": {
                "dataset": _to_plain(ds_cfg),
                "model": _to_plain(md_cfg),
                "train": _to_plain(tr_cfg),
                "aug": _to_plain(aug_cfg),
            },
        }
        flat_params = _flatten(param_blob)
        # Cast non-primitive values to str for MLflow params
        flat_params = {k: (str(v) if not isinstance(v, (str, int, float, bool)) else v) for k, v in flat_params.items()}
        mlflow.log_params(flat_params)

        # Artifacts (splits & classes)
        mlflow.log_text("\n".join(train_ids) + "\n", artifact_file="splits/train.txt")
        mlflow.log_text("\n".join(valid_ids) + "\n", artifact_file="splits/valid.txt")
        mlflow.log_text(json.dumps(class_names, ensure_ascii=False, indent=2), artifact_file="meta/classes.json")

        # Best-tracking (in-memory)
        best_metric = -1.0
        best_state: Optional[Dict[str, torch.Tensor]] = None

        def _validate(epoch: int) -> Dict[str, Any]:
            nonlocal best_metric, best_state, best_epoch, best_metrics_blob

            metrics = _evaluate_on_ids(
                model, class_names, ds_cfg, valid_ids,
                thresholds=0.5,
                tile_bs=getattr(tr_cfg, "tile_bs", 8),
                device=device,
                limit=None,
            )
            metrics.setdefault("meta", {}).update({"epoch_local": int(epoch), "classes": class_names})

            # Log numeric metrics
            flat = _flatten(metrics)
            numeric = {k: float(v) for k, v in flat.items() if isinstance(v, (int, float, np.floating))}
            macro_miou = float(metrics.get("macro_mIoU", float("nan")))
            macro_dice = float(metrics.get("macro_all", {}).get("dice", float("nan")))
            numeric["macro_mIoU"] = macro_miou
            numeric["macroDice"] = macro_dice
            mlflow.log_metrics(numeric, step=epoch)

            # Persist full JSON for the epoch
            mlflow.log_dict(metrics, f"metrics/epoch_{epoch:03d}.json")

            # Per-class scalar convenience (for MLflow charts)
            pc = metrics.get("per_class") or {}
            for key in ("iou", "dice", "precision", "recall", "tp", "fp", "fn", "tn", "support_px"):
                vals = pc.get(key, [])
                if not isinstance(vals, list):
                    continue
                for i, val in enumerate(vals):
                    if isinstance(val, (int, float, np.floating)):
                        # Use dots (or slashes) — both are allowed by MLflow
                        mlflow.log_metric(f"per_class.{key}.{i}", float(val), step=epoch)

            # Track best on macro_mIoU
            macro_miou = float(metrics.get("macro_mIoU", float("nan")))
            if macro_miou > best_metric:
                best_metric = macro_miou
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch
                best_metrics_blob = metrics

            print(f"[valid] epoch={epoch} macro_mIoU={macro_miou:.4f} macroDice={macro_dice:.4f}")
            return metrics

        # ---- Train ----
        trainer.fit(train_loader, validate_fn=_validate, start_epoch=0)
        print("\n[train] complete.")
        mlflow.log_param("best_epoch", best_epoch)
        mlflow.log_metric("best/macro_mIoU", best_metric)

        # 2) Also flatten and log best per-class metrics so they appear in the overview
        if best_metrics_blob:
            flat_best = {k: v for k, v in _flatten(best_metrics_blob).items()
                        if isinstance(v, (int, float))}
            # rename keys to "best/..."
            flat_best = {f"best/{k}": float(v) for k, v in flat_best.items()}
            mlflow.log_metrics(flat_best)
            mlflow.log_dict(best_metrics_blob, "metrics/best.json")

            
        # ---- Log & Register BEST model ----
        if best_state is not None:
            model.load_state_dict(best_state)
        else:
            print("[warn] best_state was None; logging final weights.")

        info = mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=args.registered_model_name,  # logs AND registers
        )
        print(f"[mlflow] Logged & registered as '{args.registered_model_name}'.")
        print(f"[mlflow] Model URI: {info.model_uri}")


if __name__ == "__main__":
    main()
