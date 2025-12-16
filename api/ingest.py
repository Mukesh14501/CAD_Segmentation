from __future__ import annotations
import os
import json
import re
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from fastapi import FastAPI, File, UploadFile, Form, HTTPException

from cadseg.config import load_configs

app = FastAPI(title="CADSeg Inference API", version="1.0")

os.environ.setdefault("http_proxy", os.environ.get("http_proxy", ""))
os.environ.setdefault("https_proxy", os.environ.get("https_proxy", ""))

# -----------------------------
# Config (lazy load)
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

def _walk_files(root: Path, exts=None):
    for p in root.rglob("*"):
        if p.is_file():
            if exts is None or p.suffix.lower() in exts:
                yield p

def _scan_existing_index(images_out: Path) -> int:
    """
    Scan existing images named like '^\d+\.(png|jpg|jpeg|tif|tiff|bmp)$'
    Return max_index. If none, 0.
    """
    pat = re.compile(r"^(\d+)\.(?:png|jpg|jpeg|tif|tiff|bmp)$", re.IGNORECASE)
    max_idx = 0
    if not images_out.exists():
        return 0
    for p in images_out.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        max_idx = max(max_idx, idx)
    return max_idx

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
    return {
        "last_upload_max_id": 0,
        "last_trained_max_id": 0,
        "prev_classes": [],
    }

def _write_state(root_path: Path, state: Dict) -> None:
    _state_json_path(root_path).write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

# -----------------------------
# API: Ingest (no processing)
# -----------------------------
@app.post("/ingest_labelstudio")
async def ingest_labelstudio(
    images_zip: UploadFile = File(..., description="ZIP with raw CAD images"),
    masks_zip: UploadFile = File(..., description="ZIP with Label Studio exported masks"),
    overwrite: bool = Form(True),
):
    """
    Ingest two zips (images + masks) for CADSeg training with minimal logic:
      - No decoding, no validation, no resizing, no alpha handling.
      - Just extract and copy files.
      - Continues numbering from existing images.
      - Final saved names have NO leading zeros (e.g., '06.png' -> '6.png').
      - Masks: for each (task, class), the FIRST mask found is copied to masks/<class>/<ID>.png
               (no merging of multiple parts).
      - Updates data/meta/state.json with 'last_upload_max_id'.
    """
    cfg = get_cfg()
    images_out = Path(cfg.dataset.images_path)
    masks_out_root = Path(cfg.dataset.masks_path)
    root_path = Path(cfg.dataset.root_path)

    _ensure_dir(images_out)
    _ensure_dir(masks_out_root)

    # current max index (existing images like '12.png')
    current_max = _scan_existing_index(images_out)

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

        # collect images (as-is)
        img_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        image_files = sorted([p for p in _walk_files(images_dir, img_exts)], key=lambda p: _natural_key(p.name))
        if not image_files:
            raise HTTPException(status_code=400, detail="No images found in images ZIP")

        N = len(image_files)
        start_idx = current_max + 1
        end_idx   = start_idx + N - 1

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

        # Parse masks and group per (task, class)
        # We will copy only the FIRST mask file per (task, class); no merging/processing.
        mask_pat = re.compile(r"task-(\d+).*?-tag-([A-Za-z0-9_]+)-", re.IGNORECASE)
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

        # Assign indices
        task_to_new_idx: Dict[int, int] = {}
        assigned_indices: set[int] = set()
        next_free_idx = start_idx

        def _next_index() -> int:
            nonlocal next_free_idx
            idx = next_free_idx
            next_free_idx += 1
            return idx

        # Save function: copy file bytes as-is; strip leading zeros in final name
        def _copy_to_images(src_path: Path, idx: int) -> str:
            # ensure plain number without leading zeros
            target_name = f"{idx}.png"
            target_path = images_out / target_name
            if target_path.exists() and not overwrite:
                return target_name
            # copy file bytes directly; if not PNG source, we still save as .png filename
            # (as requested: no validation/processing). This keeps things simple.
            with open(src_path, "rb") as rf, open(target_path, "wb") as wf:
                shutil.copyfileobj(rf, wf)
            return target_name

        # 1) images with known task ids
        id_map: Dict[int, str] = {}  # new index -> filename
        for tid, src in img_task_map.items():
            idx = _next_index()
            fname = _copy_to_images(src, idx)
            id_map[idx] = fname
            task_to_new_idx[tid] = idx
            assigned_indices.add(idx)

        # 2) remaining images (no task id)
        for src in imgs_no_tid:
            idx = _next_index()
            fname = _copy_to_images(src, idx)
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

        # ----- Write masks (FIRST file only per (task, class)) -----
        class_names = sorted({cls for (_, cls) in groups.keys()})
        for cls in class_names:
            _ensure_dir(masks_out_root / cls)

        written_masks = 0
        for (task, cls), parts in groups.items():
            idx = task_to_new_idx[task]
            target_name = f"{idx}.png"  # no leading zeros
            target_path = masks_out_root / cls / target_name
            if target_path.exists() and not overwrite:
                continue
            # copy the FIRST mask file as-is
            src = parts[0]
            with open(src, "rb") as rf, open(target_path, "wb") as wf:
                shutil.copyfileobj(rf, wf)
            written_masks += 1

        # Case: no masks; still need to ensure all images were copied.
        if not task_ids:
            # Copy all images sequentially if somehow missed above (normally already done).
            # This block is effectively a no-op in this simpler flow, but kept for clarity.
            pass

        # ---- Update dataset meta/state.json ----
        state = _read_state(root_path)
        prev_upload_max = int(state.get("last_upload_max_id", 0))
        state["last_upload_max_id"] = max(prev_upload_max, end_idx)
        _write_state(root_path, state)

        return {
            "existing_max_index": current_max,
            "start_index": start_idx,
            "end_index": end_idx,
            "zpad_used": 0,  # always 0 now; filenames have no leading zeros
            "images_written_estimate": N,
            "classes_detected": class_names,
            "masks_written": written_masks,
            "meta_updated": {"last_upload_max_id": state["last_upload_max_id"]},
            "notes": "No image/mask processing performed; files copied as-is. First mask per (task,class) only.",
        }
