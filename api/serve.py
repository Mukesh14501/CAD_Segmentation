from __future__ import annotations
import io
import os
from pathlib import Path
from typing import List, Optional, Tuple

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
import os
os.environ["http_proxy"] = "http://proxy50.adm.toyota.co.jp:15520"
os.environ["https_proxy"] = "http://proxy50.adm.toyota.co.jp:15520"

app = FastAPI(title="CADSeg Inference API", version="1.0")

CFG_DIR = os.getenv("CFG_DIR", "configs")
TILE_BS = int(os.getenv("TILE_BS", "4"))  # smaller batch lowers peak RAM/VRAM

# --- Helpers ---------------------------------------------------
def _find_latest_best_checkpoint(runs_dir: Path) -> Optional[Path]:
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
    # choose newest by modification time
    return max(candidates, key=lambda p: p.stat().st_mtime)

def _find_latest_metrics_json(run_dir: Path) -> Optional[Path]:
    """
    Given a run directory like runs/20250909-142750,
    pick newest metrics JSON inside run_dir/metrics.
    """
    mdir = run_dir / "metrics"
    if not mdir.exists():
        return None
    jsons = [p for p in mdir.glob("*.json") if p.is_file()]
    if not jsons:
        return None
    return max(jsons, key=lambda p: p.stat().st_mtime)

def _load_meta_classes(metrics_path: Path) -> Optional[List[str]]:
    import json
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        meta = data.get("meta", {})
        classes = meta.get("classes")
        if isinstance(classes, list) and all(isinstance(c, str) for c in classes):
            return classes
    except Exception:
        pass
    return None

def _normalize_thresholds(thr: List[float], n_classes: int) -> List[float]:
    """
    Ensure thresholds length == n_classes by trimming or padding with 0.5.
    """
    if len(thr) == n_classes:
        return thr
    if len(thr) > n_classes:
        return thr[:n_classes]
    return thr + [0.5] * (n_classes - len(thr))

def _set_num_classes_on_cfg(cfg, n_classes: int):
    """
    Try to set the model's output channels on config before build_model().
    Supports common field names used in configs.
    """
    # Common names across different templates
    for attr in ("num_classes", "out_channels", "classes", "n_classes"):
        if hasattr(cfg.model, attr):
            setattr(cfg.model, attr, n_classes)

# --- Load configs first (we'll adjust based on metrics) -------
cfg = load_configs(CFG_DIR)

# --- Resolve checkpoint path (env or newest best.pth) ---------
_ckpt_env = os.getenv("MODEL_CKPT", "").strip()
if _ckpt_env:
    ckpt_path = Path(_ckpt_env)
else:
    ckpt_path = _find_latest_best_checkpoint(Path("runs"))  # auto-pick newest best.pth

if not ckpt_path or not ckpt_path.exists():
    raise RuntimeError(
        "No model checkpoint found. "
        "Either set MODEL_CKPT to an existing .pth or ensure runs/*/checkpoints/best.pth exists."
    )

# --- Find corresponding run dir and newest metrics ------------
# ckpt structure: runs/<run_id>/checkpoints/best.pth
run_dir = ckpt_path.parent.parent  # .../runs/<run_id>
metrics_path = _find_latest_metrics_json(run_dir)
metrics_classes: Optional[List[str]] = _load_meta_classes(metrics_path) if metrics_path else None

# --- Determine class list -------------------------------------
# Priority: metrics meta.classes -> cfg.dataset.classes
if metrics_classes and len(metrics_classes) > 0:
    CLASS_NAMES = metrics_classes
else:
    CLASS_NAMES = list(cfg.dataset.classes)  # fallback to config

N_CLASSES = len(CLASS_NAMES)

# --- Rebuild model with correct head size ---------------------
# 1) Update cfg pieces that depend on classes
cfg.dataset.classes = CLASS_NAMES
_set_num_classes_on_cfg(cfg, N_CLASSES)

# 2) Build the model with the correct output channels
model = build_model(cfg.model)

# 3) Load the checkpoint (strict). The checkpoint should match this head size.
state = torch.load(ckpt_path, map_location="cpu")
if isinstance(state, dict) and "model" in state:
    state = state["model"]
model.load_state_dict(state, strict=True)

# --- Device ----------------------------------------------------
DEVICE = _pick_device()
model.to(DEVICE).eval()

# --- Thresholds per class -------------------------------------
thr_path = Path(getattr(cfg.infer, "thresholds_file", "configs/thresholds.json"))
if thr_path.exists():
    import json
    try:
        th = json.loads(thr_path.read_text(encoding="utf-8"))
        if isinstance(th, dict):
            # map by class name; default 0.5 when missing
            THRESHOLDS = [float(th.get(c, 0.5)) for c in CLASS_NAMES]
        else:
            # assume list; normalize length
            raw = [float(x) for x in th]
            THRESHOLDS = _normalize_thresholds(raw, N_CLASSES)
    except Exception:
        THRESHOLDS = [0.5] * N_CLASSES
else:
    THRESHOLDS = [0.5] * N_CLASSES

# --- Per-class min component area (optional) ------------------
MIN_AREA = getattr(cfg.infer, "min_component_area", None)

# --- Default TTA list and merge mode --------------------------
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

def _normalize_min_area(min_area_cfg, class_names: List[str]) -> Optional[List[int]]:
    """
    Accepts:
      - None -> None
      - int/float -> broadcast to all classes
      - list/tuple of numbers:
          * if len==1 -> broadcast
          * if len==n_classes -> use as-is
      - dict {class_name: number} -> map in CLASS_NAMES order, default 0 for missing
    Returns a list[int] length == n_classes, or None.
    """
    if min_area_cfg is None:
        return None

    n = len(class_names)

    # scalar
    if isinstance(min_area_cfg, (int, float)):
        v = int(min_area_cfg)
        return [v] * n

    # list/tuple
    if isinstance(min_area_cfg, (list, tuple)):
        vals = [int(x) for x in min_area_cfg]
        if len(vals) == n:
            return vals
        if len(vals) == 1:
            return [vals[0]] * n
        # fallback: trim or pad with zeros
        return (vals + [0] * n)[:n]

    # dict keyed by class name
    if isinstance(min_area_cfg, dict):
        out = []
        for c in class_names:
            v = min_area_cfg.get(c, 0)
            out.append(int(v))
        return out

    # unknown type -> disable
    return None

_raw_min_area = getattr(cfg.infer, "min_component_area", None)
MIN_AREA = _normalize_min_area(_raw_min_area, CLASS_NAMES)


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "classes": CLASS_NAMES,
        "checkpoint": str(ckpt_path),
        "metrics_json": str(metrics_path) if metrics_path else None,
    }

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
)

    # Build overlay
    overlay_rgb = make_overlay(img, masks_bin, CLASS_NAMES, alpha=0.45)
    overlay_bgr = overlay_rgb[:, :, ::-1]
    ok, png = cv2.imencode(".png", overlay_bgr)
    if not ok:
        return JSONResponse(status_code=500, content={"error": "PNG encoding failed"})

    return StreamingResponse(io.BytesIO(png.tobytes()), media_type="image/png")


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
        logits = infer_full_image_logits(
            model, img_aug, cfg.dataset, cfg.infer, tile_bs=bs, device=DEVICE
        )
        logits = _invert_flip(logits, mode)
        logits_list.append(logits)
    logits_merged = _merge_logits(logits_list, merge_mode)

    probs, masks_bin = postprocess_logits(
        logits_merged, thresholds=THRESHOLDS, min_component_area=MIN_AREA
    )

    areas = masks_bin.reshape(masks_bin.shape[0], -1).sum(axis=1).tolist()  # pixels per class
    return {
        "classes": CLASS_NAMES,
        "areas_px": areas,
        "thresholds": THRESHOLDS,
        "image_size": [int(img.shape[0]), int(img.shape[1])],
        "checkpoint": str(ckpt_path),
        "metrics_json": str(metrics_path) if metrics_path else None,
    }
