from __future__ import annotations
import io
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import cv2
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import StreamingResponse, JSONResponse

from cadseg.config import load_configs
from cadseg.models.builder import build_model
from cadseg.engine.eval_loop import infer_full_image_logits, _pick_device
from cadseg.models.postprocess import postprocess_logits
from cadseg.engine.infer_loop import make_overlay

# -----------------------------
# App & Globals
# -----------------------------
app = FastAPI(title="CADSeg Inference API", version="1.0")

CFG_DIR = os.getenv("CFG_DIR", "configs")
CKPT_PATH = os.getenv("MODEL_CKPT", "runs/debug_exp/checkpoints/best.pth")
TILE_BS = int(os.getenv("TILE_BS", "4"))  # smaller batch lowers peak RAM/VRAM

cfg = load_configs(CFG_DIR)
model = build_model(cfg.model)

# Load checkpoint (strict)
state = torch.load(CKPT_PATH, map_location="cpu")
if isinstance(state, dict) and "model" in state:
    state = state["model"]
model.load_state_dict(state, strict=True)

# Device
DEVICE = _pick_device()
model.to(DEVICE).eval()

# Thresholds per class
thr_path = Path(getattr(cfg.infer, "thresholds_file", "configs/thresholds.json"))
if thr_path.exists():
    import json
    th = json.loads(thr_path.read_text())
    if isinstance(th, dict):
        THRESHOLDS = [float(th.get(c, 0.5)) for c in cfg.dataset.classes]
    else:
        THRESHOLDS = [float(x) for x in th]
else:
    THRESHOLDS = [0.5 for _ in cfg.dataset.classes]

# Per-class min component area for small-blob removal (optional)
MIN_AREA = getattr(cfg.infer, "min_component_area", None)

# Default TTA list and merge mode
DEFAULT_TTA = getattr(cfg.infer, "tta", ["hflip", "vflip"]) or []
MERGE_MODE = getattr(cfg.infer, "merge", "mean")


# ---------------
# TTA helpers
# ---------------
def _apply_flip(img: np.ndarray, kind: str) -> np.ndarray:
    if kind == "hflip":
        return np.ascontiguousarray(img[:, ::-1, :])
    if kind == "vflip":
        return np.ascontiguousarray(img[::-1, :, :])
    return img


def _invert_flip(arr: np.ndarray, kind: str) -> np.ndarray:
    # arr: (C,H,W) logits
    if kind == "hflip":
        return arr[:, :, ::-1]
    if kind == "vflip":
        return arr[:, ::-1, :]
    return arr


def _merge_logits(logits_list: List[np.ndarray], mode: str = "mean") -> np.ndarray:
    if len(logits_list) == 1:
        return logits_list[0]
    mode = (mode or "mean").lower()
    if mode == "max":
        return np.maximum.reduce(logits_list)
    # probs fusion for mean/gmean → back to logits
    probs = [1.0 / (1.0 + np.exp(-x)) for x in logits_list]
    if mode == "gmean":
        eps = 1e-6
        logsum = np.zeros_like(probs[0])
        for p in probs:
            logsum += np.log(np.clip(p, eps, 1.0))
        gmean = np.exp(logsum / len(probs))
        gmean = np.clip(gmean, eps, 1.0 - eps)
        return np.log(gmean / (1.0 - gmean))
    m = np.mean(probs, axis=0)
    eps = 1e-6
    m = np.clip(m, eps, 1.0 - eps)
    return np.log(m / (1.0 - m))


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok", "device": str(DEVICE), "classes": cfg.dataset.classes}


@app.post("/segment", response_class=StreamingResponse)
async def segment(
    file: UploadFile = File(...),
    tta: Optional[str] = Query(None, description="comma-separated: hflip,vflip"),
    merge: Optional[str] = Query(None, description="mean|gmean|max"),
    tile_bs: Optional[int] = Query(None, description="tile batch size"),
    return_type: str = Query("overlay", description="overlay|mask|both (PNG or ZIP for both)"),
):
    # Read image bytes → RGB np.ndarray
    data = await file.read()
    buf = np.frombuffer(data, dtype=np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return JSONResponse(status_code=400, content={"error": "Cannot decode image"})
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # TTA list
    tta_list = [s.strip() for s in (tta.split(",") if tta else DEFAULT_TTA) if s.strip()]
    merge_mode = (merge or MERGE_MODE)
    bs = int(tile_bs or TILE_BS)

    # Tiled inference with optional TTA
    logits_list: List[np.ndarray] = []
    for mode in (["none"] + tta_list) if tta_list else ["none"]:
        img_aug = _apply_flip(img, mode)
        logits = infer_full_image_logits(
            model, img_aug, cfg.dataset, cfg.infer, tile_bs=bs, device=DEVICE, normalize="imagenet"
        )  # (C,H,W)
        logits = _invert_flip(logits, mode)
        logits_list.append(logits)
    logits_merged = _merge_logits(logits_list, merge_mode)

    # Post-process → probs & binary masks
    probs, masks_bin = postprocess_logits(
        logits_merged, thresholds=THRESHOLDS, min_component_area=MIN_AREA
    )  # (C,H,W)

    # Build overlay
    overlay_rgb = make_overlay(img, masks_bin, cfg.dataset.classes, alpha=0.45)
    overlay_bgr = overlay_rgb[:, :, ::-1]
    ok, png = cv2.imencode(".png", overlay_bgr)
    if not ok:
        return JSONResponse(status_code=500, content={"error": "PNG encoding failed"})

    return StreamingResponse(io.BytesIO(png.tobytes()), media_type="image/png")


# Optional: return JSON with simple stats (area per class)
@app.post("/segment_json")
async def segment_json(
    file: UploadFile = File(...),
    tta: Optional[str] = Query(None),
    merge: Optional[str] = Query(None),
    tile_bs: Optional[int] = Query(None),
):
    data = await file.read()
    buf = np.frombuffer(data, dtype=np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return JSONResponse(status_code=400, content={"error": "Cannot decode image"})
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    tta_list = [s.strip() for s in (tta.split(",") if tta else DEFAULT_TTA) if s.strip()]
    merge_mode = (merge or MERGE_MODE)
    bs = int(tile_bs or TILE_BS)

    logits_list: List[np.ndarray] = []
    for mode in (["none"] + tta_list) if tta_list else ["none"]:
        img_aug = _apply_flip(img, mode)
        logits = infer_full_image_logits(model, img_aug, cfg.dataset, cfg.infer, tile_bs=bs, device=DEVICE)
        logits = _invert_flip(logits, mode)
        logits_list.append(logits)
    logits_merged = _merge_logits(logits_list, merge_mode)

    probs, masks_bin = postprocess_logits(logits_merged, thresholds=THRESHOLDS, min_component_area=MIN_AREA)

    areas = masks_bin.reshape(masks_bin.shape[0], -1).sum(axis=1).tolist()  # pixels per class
    return {
        "classes": cfg.dataset.classes,
        "areas_px": areas,
        "thresholds": THRESHOLDS,
        "image_size": [int(img.shape[0]), int(img.shape[1])],
    }
