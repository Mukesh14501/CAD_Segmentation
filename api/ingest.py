from __future__ import annotations
import os
import json
import re
import zipfile
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

from cadseg.config import load_configs

app = FastAPI(title="CADSeg Inference API", version="1.0")

# -----------------------------
# Config (lazy load recommended)
# -----------------------------
from functools import lru_cache

@lru_cache(maxsize=1)
def get_cfg():
    cfg_dir = os.getenv("CFG_DIR", "configs")
    try:
        return load_configs(cfg_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to load configs from {cfg_dir}: {e}")

# -----------------------------
# FS helpers
# -----------------------------
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def _read_file_bytes(p: Path) -> bytes:
    with open(p, "rb") as f:
        return f.read()

def _walk_files(root: Path, exts=None):
    for p in root.rglob("*"):
        if p.is_file():
            if exts is None or p.suffix.lower() in exts:
                yield p

def _scan_existing_index(images_out: Path) -> Tuple[int, int]:
    """
    Scan existing images named like '^\d+\.(png|jpg|jpeg|tif|tiff|bmp)$'
    Return (max_index, existing_zero_pad_width). If none, (0,0).
    """
    pat = re.compile(r"^(\d+)\.(?:png|jpg|jpeg|tif|tiff|bmp)$", re.IGNORECASE)
    max_idx = 0
    zpad = 0
    if not images_out.exists():
        return 0, 0
    for p in images_out.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        idx_str = m.group(1)
        idx = int(idx_str)
        max_idx = max(max_idx, idx)
        zpad = max(zpad, len(idx_str))
    return max_idx, zpad

# -----------------------------
# Image / mask decoding
# -----------------------------
def _decode_to_rgb(data: bytes) -> Optional[np.ndarray]:
    buf = np.frombuffer(data, dtype=np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 2:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    if im.shape[2] == 4:
        b, g, r, a = cv2.split(im)
        rgb = cv2.merge([r, g, b]).astype(np.float32)
        alpha = (a.astype(np.float32) / 255.0)[..., None]
        bg = np.full_like(rgb, 255.0)
        out = rgb * alpha + bg * (1.0 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def _read_mask_to_binary(path: Path) -> np.ndarray:
    data = _read_file_bytes(path)
    buf = np.frombuffer(data, dtype=np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise ValueError(f"Cannot decode mask: {path}")
    if im.ndim == 3:
        if im.shape[2] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return (im > 0).astype(np.uint8)

# -----------------------------
# Dataset meta (data/meta/state.json)
# -----------------------------
def _meta_dir_data(root_path: Path) -> Path:
    d = root_path / "meta"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _state_json_path(root_path: Path) -> Path:
    return _meta_dir_data(root_path) / "state.json"

def _read_state(root_path: Path) -> Dict:
    p = _state_json_path(root_path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    # default shape
    return {
        "last_upload_max_id": 0,
        "last_trained_max_id": 0,
        "prev_classes": [],
    }

def _write_state(root_path: Path, state: Dict) -> None:
    _state_json_path(root_path).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

# -----------------------------
# API: Ingest only (no splits)
# -----------------------------
@app.post("/ingest_labelstudio")
async def ingest_labelstudio(
    images_zip: UploadFile = File(..., description="ZIP with raw CAD images"),
    masks_zip: UploadFile = File(..., description="ZIP with Label Studio exported masks"),
    overwrite: bool = Form(True),
):
    """
    Ingest two zips (images + masks) for CADSeg training.

    - Continues numbering from existing images (keeps/expands zero-padding).
    - Maps Label Studio task IDs to newly assigned image IDs:
        * If some images contain 'task-<id>' in their path/name, they get assigned first.
        * Remaining images are assigned by arrival order.
        * Every mask 'task-<id>' is mapped to the corresponding new image index.
    - Writes masks into masks/<class>/<ID>.png (OR-merge multiples).
    - Updates data/meta/state.json with 'last_upload_max_id'.  (Splits are owned by the trainer.)
    """
    cfg = get_cfg()
    images_out = Path(cfg.dataset.images_path)
    masks_out_root = Path(cfg.dataset.masks_path)
    root_path = Path(cfg.dataset.root_path)

    _ensure_dir(images_out)
    _ensure_dir(masks_out_root)

    # --- scan existing image index & padding ---
    current_max, existing_pad = _scan_existing_index(images_out)

    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        # save uploads
        images_zip_path = tmp / "images.zip"; images_zip_path.write_bytes(await images_zip.read())
        masks_zip_path  = tmp / "masks.zip";  masks_zip_path.write_bytes(await masks_zip.read())

        # extract
        images_dir = tmp / "images"; _ensure_dir(images_dir)
        masks_dir  = tmp / "masks";  _ensure_dir(masks_dir)
        with zipfile.ZipFile(images_zip_path, "r") as z:
            z.extractall(images_dir)
        with zipfile.ZipFile(masks_zip_path, "r") as z:
            z.extractall(masks_dir)

        # collect images
        img_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        image_files = sorted([p for p in _walk_files(images_dir, img_exts)], key=lambda p: _natural_key(p.name))
        if not image_files:
            raise HTTPException(status_code=400, detail="No images found in images ZIP")

        N = len(image_files)
        start_idx = current_max + 1
        end_idx   = start_idx + N - 1
        zpad = max(existing_pad, len(str(end_idx)))

        # Build possible mapping task-id -> image file (if images contain task-<id> in path/name)
        tid_pat = re.compile(r"task[-_](\d+)", re.IGNORECASE)
        img_task_map: Dict[int, Path] = {}
        imgs_no_tid: List[Path] = []
        for p in image_files:
            m = tid_pat.search(str(p).replace("\\", "/"))
            if m:
                img_task_map[int(m.group(1))] = p
            else:
                imgs_no_tid.append(p)

        # Parse masks and group parts per (task, class)
        mask_pat = re.compile(r"task-(\d+).*?-tag-([A-Za-z0-9_]+)-")
        mask_files = [p for p in _walk_files(masks_dir, None)]
        groups: Dict[Tuple[int, str], List[Path]] = {}
        task_ids = set()
        for p in mask_files:
            m = mask_pat.search(p.name) or mask_pat.search(str(p).replace("\\", "/"))
            if not m:
                continue
            task = int(m.group(1)); cls = m.group(2)
            task_ids.add(task)
            groups.setdefault((task, cls), []).append(p)

        if not task_ids:
            # It's valid to ingest images without masks, but warn
            class_names = []
            written_masks = 0
        else:
            # Determine new IDs and save images, building:
            # - task_to_new_idx (for masks)
            task_to_new_idx: Dict[int, int] = {}
            id_map: Dict[int, str] = {}  # new index -> filename

            # Assign indices incrementally
            assigned_indices: set[int] = set()
            next_free_idx = start_idx

            def _next_index() -> int:
                nonlocal next_free_idx
                idx = next_free_idx
                next_free_idx += 1
                return idx

            # Write function to actually save an image to the numbered target
            def _save_image_to_index(src_path: Path, idx: int) -> str:
                target_name = f"{idx:0{zpad}d}.png"
                target_path = images_out / target_name
                if target_path.exists() and not overwrite:
                    return target_name
                rgb = _decode_to_rgb(_read_file_bytes(src_path))
                if rgb is None:
                    raise HTTPException(status_code=400, detail=f"Cannot decode image: {src_path.name}")
                ok = cv2.imwrite(str(target_path), rgb[:, :, ::-1])
                if not ok:
                    raise HTTPException(status_code=500, detail=f"Failed to save image: {target_name}")
                return target_name

            # 1) images with known task ids
            for tid, src in img_task_map.items():
                idx = _next_index()
                fname = _save_image_to_index(src, idx)
                id_map[idx] = fname
                task_to_new_idx[tid] = idx
                assigned_indices.add(idx)

            # 2) remaining images (no task id on filename): assign in remaining order
            for src in imgs_no_tid:
                idx = _next_index()
                fname = _save_image_to_index(src, idx)
                id_map[idx] = fname

            # 3) ensure every task id gets an index (if task didn’t appear in image filenames)
            all_indices = list(range(start_idx, start_idx + N))
            free_indices = [i for i in all_indices if i not in assigned_indices]
            fi = 0
            for t in sorted(task_ids):
                if t not in task_to_new_idx:
                    if fi >= len(free_indices):
                        raise HTTPException(status_code=500, detail="Internal mapping error (free index exhausted).")
                    task_to_new_idx[t] = free_indices[fi]; fi += 1

            # ----- Write masks -----
            class_names = sorted({cls for (_, cls) in groups.keys()})
            for cls in class_names:
                _ensure_dir(masks_out_root / cls)

            written_masks = 0
            for (task, cls), parts in groups.items():
                idx = task_to_new_idx[task]
                # load & OR-union parts
                combined = None
                for part in parts:
                    m = _read_mask_to_binary(part)
                    if combined is None:
                        combined = m.astype(np.uint8)
                    else:
                        if combined.shape != m.shape:
                            m = cv2.resize(m, (combined.shape[1], combined.shape[0]), interpolation=cv2.INTER_NEAREST)
                        combined = np.maximum(combined, m)
                if combined is None:
                    continue
                target_name = f"{idx:0{zpad}d}.png"
                target_path = masks_out_root / cls / target_name
                if target_path.exists() and not overwrite:
                    continue
                if cv2.imwrite(str(target_path), (combined * 255).astype(np.uint8)):
                    written_masks += 1

        # Save any images that may not have been written (case: no masks at all)
        if not task_ids:
            # When no masks ZIP or no task-ids present, we still need to write all images sequentially.
            next_idx = start_idx
            for src in image_files:
                target_name = f"{next_idx:0{zpad}d}.png"
                target_path = images_out / target_name
                if (not target_path.exists()) or overwrite:
                    rgb = _decode_to_rgb(_read_file_bytes(src))
                    if rgb is None:
                        raise HTTPException(status_code=400, detail=f"Cannot decode image: {src.name}")
                    if not cv2.imwrite(str(target_path), rgb[:, :, ::-1]):
                        raise HTTPException(status_code=500, detail=f"Failed to save image: {target_name}")
                next_idx += 1

        # ---- Update dataset meta/state.json ----
        state = _read_state(root_path)
        prev_upload_max = int(state.get("last_upload_max_id", 0))
        state["last_upload_max_id"] = max(prev_upload_max, end_idx)
        # leave last_trained_max_id and prev_classes untouched; trainer will update
        _write_state(root_path, state)

        return {
            "existing_max_index": current_max,
            "start_index": start_idx,
            "end_index": end_idx,
            "zpad_used": zpad,
            "images_written_estimate": N,     # images saved equals N unless overwrite=False skipped existing
            "classes_detected": class_names if task_ids else [],
            "masks_written": written_masks if task_ids else 0,
            "meta_updated": {"last_upload_max_id": state["last_upload_max_id"]},
        }
