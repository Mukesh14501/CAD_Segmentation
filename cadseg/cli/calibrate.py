# cadseg/cli/calibrate.py
"""
CLI: calibrate.py
- Computes per-class probability thresholds that maximize F1 or IoU on a validation set.
- Saves to the path configured in infer.yaml (`thresholds_file`), unless overridden.
Note: This implementation accumulates per-pixel probabilities in memory; for extremely large canvases
consider using --limit during early experiments.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from cadseg.config import load_configs
from cadseg.models.builder import build_model
from cadseg.engine.eval_loop import infer_full_image_logits, _pick_device
from cadseg.dataio.masks import list_images, image_id_from_path, load_image_rgb, per_class_mask_path, load_binary_mask


def _load_ckpt(model: torch.nn.Module, ckpt_path: Path) -> None:
    state = torch.load(ckpt_path, map_location="cpu")
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)


def _gather_flat_probs_targets(
    model: torch.nn.Module,
    images: List[Path],
    classes: List[str],
    ds_cfg,
    *,
    tile_bs: int,
    device: torch.device,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Returns:
      probs_per_class: list of C arrays, each shape (N_pixels_total_for_class_eval,)
      targs_per_class: list of C arrays (0/1)
    """
    C = len(classes)
    probs_per_class = [ [] for _ in range(C) ]
    targs_per_class = [ [] for _ in range(C) ]

    for ip in tqdm(images, desc="Calibrate (collect)", ncols=100):
        iid = image_id_from_path(ip)
        img = load_image_rgb(ip)
        H, W = img.shape[:2]
        logits = infer_full_image_logits(model, img, ds_cfg, tile_bs=tile_bs, device=device)  # (C,H,W)
        probs = 1 / (1 + np.exp(-logits))  # sigmoid

        # Load GT masks per class
        for c, cname in enumerate(classes):
            mp = per_class_mask_path(ds_cfg.masks_path, cname, iid)
            if mp.exists():
                m = load_binary_mask(mp)
            else:
                m = np.zeros((H, W), dtype=np.uint8)
            # Flatten
            probs_per_class[c].append(probs[c].reshape(-1).astype(np.float32))
            targs_per_class[c].append(m.reshape(-1).astype(np.uint8))

    probs_per_class = [ np.concatenate(xs, axis=0) if xs else np.zeros((0,), np.float32) for xs in probs_per_class ]
    targs_per_class = [ np.concatenate(xs, axis=0) if xs else np.zeros((0,), np.uint8) for xs in targs_per_class ]
    return probs_per_class, targs_per_class


def _best_threshold_1d(probs: np.ndarray, targs: np.ndarray, metric: str = "f1") -> float:
    """
    Sweep thresholds in [0.05..0.95] and select the one that maximizes the chosen metric.
    """
    if probs.size == 0:
        return 0.5
    ts = np.linspace(0.05, 0.95, 19, dtype=np.float32)
    best_t, best_v = 0.5, -1.0
    for t in ts:
        pred = (probs >= t).astype(np.uint8)
        TP = np.sum((pred == 1) & (targs == 1))
        FP = np.sum((pred == 1) & (targs == 0))
        FN = np.sum((pred == 0) & (targs == 1))
        if metric == "iou":
            denom = TP + FP + FN
            v = (TP / denom) if denom > 0 else 0.0
        else:  # f1 default
            denom = 2 * TP + FP + FN
            v = (2 * TP / denom) if denom > 0 else 0.0
        if v > best_v:
            best_v, best_t = v, float(t)
    return best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=str, default="configs", help="Path to configs dir.")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pth).")
    ap.add_argument("--metric", type=str, default="f1", choices=["f1", "iou"], help="Metric to optimize.")
    ap.add_argument("--out", type=str, default="", help="Override output path (defaults to infer.yaml: thresholds_file).")
    ap.add_argument("--limit", type=int, default=0, help="Limit #images during calibration (for speed/memory).")
    ap.add_argument("--tile_bs", type=int, default=8, help="Tile batch size during inference.")
    args = ap.parse_args()

    cfg = load_configs(args.configs)
    ds, md = cfg.dataset, cfg.model
    class_names = ds.classes

    model = build_model(md)
    _load_ckpt(model, Path(args.ckpt))
    device = _pick_device()
    model.to(device).eval()

    images = list_images(ds.images_path)
    if args.limit and args.limit > 0:
        images = images[:args.limit]

    probs_per_class, targs_per_class = _gather_flat_probs_targets(
        model, images, class_names, ds, tile_bs=args.tile_bs, device=device
    )

    thresholds = {}
    for c, cname in enumerate(class_names):
        t = _best_threshold_1d(probs_per_class[c], targs_per_class[c], metric=args.metric)
        thresholds[cname] = round(float(t), 4)

    out_path = Path(args.out) if args.out else Path(cfg.infer.thresholds_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    print(f"Saved thresholds to: {out_path}")
    print(json.dumps(thresholds, indent=2))

if __name__ == "__main__":
    main()
