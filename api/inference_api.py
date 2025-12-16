from __future__ import annotations
import io
import os
import base64
import json
import tempfile
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import torch
import cv2
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse

from cadseg.config import load_configs
from cadseg.engine.eval_loop import infer_full_image_logits, _pick_device
from cadseg.models.postprocess import postprocess_logits

import mlflow
import mlflow.pytorch
from mlflow import MlflowClient
from mlflow import artifacts as mlart

# =========================
# MLflow: load Production model
# =========================
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "CAD_Segmentation")
STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")

MODEL_URI = f"models:/{MODEL_NAME}/{STAGE}"
model: torch.nn.Module = mlflow.pytorch.load_model(MODEL_URI)  # registry load only
model.eval()

# Resolve concrete version + run_id for artifacts
_client = MlflowClient()
try:
    _vers = _client.get_latest_versions(MODEL_NAME, stages=[STAGE])
    _mv = _vers[0] if _vers else None
    MODEL_VERSION = _mv.version if _mv else None
    RUN_ID = _mv.run_id if _mv else None
except Exception:
    MODEL_VERSION, RUN_ID = None, None

def _load_classes_from_mlflow() -> Optional[List[str]]:
    """
    Try, in order:
      1) models:/<name>/<stage>/classes.json (inside the registered model directory)
      2) runs:/<run_id>/classes.json (root artifacts of the training run)
    """
    tmpdir = tempfile.mkdtemp(prefix="cadseg-classes-")

    # Attempt (1): within the registered model directory
    try:
        local = mlart.download_artifacts(artifact_uri=f"{MODEL_URI}/classes.json", dst_path=tmpdir)
        if local and Path(local).is_file():
            with open(local, "r", encoding="utf-8") as f:
                arr = json.load(f)
            if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
                return arr
    except Exception:
        pass

    # Attempt (2): from the training run's artifact root
    if RUN_ID:
        try:
            local = mlart.download_artifacts(artifact_uri=f"runs:/{RUN_ID}/classes.json", dst_path=tmpdir)
            if local and Path(local).is_file():
                with open(local, "r", encoding="utf-8") as f:
                    arr = json.load(f)
                if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
                    return arr
        except Exception:
            pass

    return None

# =========================
# App & Globals
# =========================
app = FastAPI(title="CADSeg Inference (JSON masks)", version="1.0")

CFG_DIR = os.getenv("CFG_DIR", "configs")
TILE_BS = int(os.getenv("TILE_BS", "4"))              # batch for tiled infer
DEFAULT_TTA = os.getenv("TTA", "hflip,vflip").strip() # comma list or ""
MERGE_MODE = os.getenv("MERGE_MODE", "mean")          # mean|gmean|max

# -------------------------
# Helpers
# -------------------------
def _normalize_thresholds(thr, n_classes: int) -> List[float]:
    if thr is None:
        return [0.5] * n_classes
    if isinstance(thr, (list, tuple)):
        vals = [float(x) for x in thr]
        if len(vals) == n_classes:
            return vals
        if len(vals) > n_classes:
            return vals[:n_classes]
        return vals + [0.5] * (n_classes - len(vals))
    if isinstance(thr, (int, float)):
        return [float(thr)] * n_classes
    return [0.5] * n_classes

def _apply_flip(img: np.ndarray, kind: str) -> np.ndarray:
    if kind == "hflip":
        return np.ascontiguousarray(img[:, ::-1, :])
    if kind == "vflip":
        return np.ascontiguousarray(img[::-1, :, :])
    return img

def _invert_flip(arr: np.ndarray, kind: str) -> np.ndarray:
    # arr shape: (C,H,W)
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

    probs = [1.0 / (1.0 + np.exp(-x)) for x in logits_list]
    eps = 1e-6
    if mode == "gmean":
        logsum = np.zeros_like(probs[0])
        for p in probs:
            logsum += np.log(np.clip(p, eps, 1.0))
        gmean = np.exp(logsum / len(probs))
        gmean = np.clip(gmean, eps, 1.0 - eps)
        return np.log(gmean / (1.0 - gmean))

    m = np.mean(probs, axis=0)
    m = np.clip(m, eps, 1.0 - eps)
    return np.log(m / (1.0 - m))

def _normalize_min_area(min_area_cfg, class_names: List[str]) -> Optional[List[int]]:
    if min_area_cfg is None:
        return None
    n = len(class_names)
    if isinstance(min_area_cfg, (int, float)):
        return [int(min_area_cfg)] * n
    if isinstance(min_area_cfg, (list, tuple)):
        vals = [int(x) for x in min_area_cfg]
        if len(vals) == n:
            return vals
        if len(vals) == 1:
            return [vals[0]] * n
        return (vals + [0] * n)[:n]
    if isinstance(min_area_cfg, dict):
        return [int(min_area_cfg.get(c, 0)) for c in class_names]
    return None

def _encode_mask_png_to_b64(mask_bool: np.ndarray) -> str:
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    ok, png = cv2.imencode(".png", mask_u8)
    if not ok:
        raise RuntimeError("PNG encoding failed for a mask")
    return base64.b64encode(png.tobytes()).decode("ascii")

# -------------------------
# Load config + classes from MLflow
# -------------------------
cfg = load_configs(CFG_DIR)

CLASSES_FROM_ARTIFACT = _load_classes_from_mlflow()
if CLASSES_FROM_ARTIFACT and len(CLASSES_FROM_ARTIFACT) > 0:
    CLASS_NAMES = list(CLASSES_FROM_ARTIFACT)
else:
    # Fallback to config if artifact missing
    CLASS_NAMES = list(cfg.dataset.classes)

N_CLASSES = len(CLASS_NAMES)
DEVICE = _pick_device()
model.to(DEVICE).eval()

# thresholds
thr_path = Path(getattr(cfg.infer, "thresholds_file", "configs/thresholds.json"))
if thr_path.exists():
    try:
        th = json.loads(thr_path.read_text(encoding="utf-8"))
        if isinstance(th, dict):
            THRESHOLDS = [float(th.get(c, 0.5)) for c in CLASS_NAMES]
        else:
            THRESHOLDS = _normalize_thresholds(th, N_CLASSES)
    except Exception:
        THRESHOLDS = [0.5] * N_CLASSES
else:
    THRESHOLDS = [0.5] * N_CLASSES

# min connected-component area
MIN_AREA = _normalize_min_area(getattr(cfg.infer, "min_component_area", None), CLASS_NAMES)

# =========================
# Single endpoint: /inference
# =========================
@app.post("/inference")
async def inference(
    file: UploadFile = File(...),
    tta: Optional[str] = Query(None, description="comma-separated: hflip,vflip"),
    merge: Optional[str] = Query(None, description="mean|gmean|max"),
    tile_bs: Optional[int] = Query(None, description="tile batch size"),
):
    """
    Returns JSON:
      {
        "classes": [...],
        "image_size": [H, W],
        "model_registry": {"name": "...", "stage": "...", "version": "...", "uri": "..."},
        "thresholds": [...],
        "masks": [
          {"rule": "<class_name>", "mask_b64": "<base64 PNG>"},
          ...
        ]
      }
    """
    data = await file.read()
    buf = np.frombuffer(data, dtype=np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return JSONResponse(status_code=400, content={"error": "Cannot decode image"})
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # HWC, RGB

    tta_cfg = tta if tta is not None else DEFAULT_TTA
    tta_list = [s.strip() for s in (tta_cfg.split(",") if tta_cfg else []) if s.strip()]
    merge_mode = (merge or MERGE_MODE)
    bs = int(tile_bs or TILE_BS)

    with torch.no_grad():
        logits_list: List[np.ndarray] = []
        modes = (["none"] + tta_list) if tta_list else ["none"]
        for m in modes:
            img_aug = _apply_flip(img, m)
            logits = infer_full_image_logits(
                model, img_aug, cfg.dataset, cfg.infer, tile_bs=bs, device=DEVICE, normalize="imagenet"
            )  # (C,H,W) logits, numpy
            logits = _invert_flip(logits, m)
            logits_list.append(logits)

        logits_merged = _merge_logits(logits_list, merge_mode)

        probs, masks_bin = postprocess_logits(
            logits_merged,
            thresholds=THRESHOLDS,
            min_component_area=MIN_AREA
        )  # probs: (C,H,W) in [0,1], masks_bin: (C,H,W) bool

    H, W = img.shape[:2]

    masks_out: List[Dict[str, str]] = []
    for ci, cname in enumerate(CLASS_NAMES):
        mask_b64 = _encode_mask_png_to_b64(masks_bin[ci])
        masks_out.append({"rule": cname, "mask_b64": mask_b64})
    
    x = {
        "classes": CLASS_NAMES,
        "image_size": [int(H), int(W)],
        "model_registry": {
            "name": MODEL_NAME,
            "stage": STAGE,
            "version": MODEL_VERSION,
            "uri": MODEL_URI,
            "run_id": RUN_ID,
        },
        "thresholds": THRESHOLDS,
        "masks": masks_out,
    }

    

    return x
