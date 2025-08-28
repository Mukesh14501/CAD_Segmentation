# cadseg/dataio/ingest.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import shutil
import random
import re
import json

NUM_RE = re.compile(r"^(\d+)\.(png|jpg|jpeg|tif|tiff)$", re.IGNORECASE)

def _read_lines(p: Path) -> List[str]:
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

def _append_lines(p: Path, lines: List[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

def _scan_max_id(images_dir: Path) -> int:
    max_id = 0
    if not images_dir.exists():
        return 0
    for fn in images_dir.iterdir():
        m = NUM_RE.match(fn.name)
        if m:
            n = int(m.group(1))
            if n > max_id:
                max_id = n
    return max_id

def _collect_numeric_files(dirpath: Path) -> List[Path]:
    if not dirpath.exists():
        return []
    out = []
    for p in dirpath.iterdir():
        if p.is_file() and NUM_RE.match(p.name):
            out.append(p)
    return sorted(out, key=lambda p: int(NUM_RE.match(p.name).group(1)))

def _copy_with_new_id(src: Path, dst_dir: Path, new_id: int) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    ext = src.suffix.lower()
    dst = dst_dir / f"{new_id}{ext}"
    shutil.copy2(src, dst)
    return dst

def ingest_week(
    dataset_root: Path,
    weekly_drop_root: Path,
    *,
    seed: int = 42,
    train_ratio: float = 0.9,
) -> Dict:
    """
    Ingest a weekly drop into canonical data tree.

    Expected canonical tree:
    data/
      images/
      masks/
        rule_00/
        rule_01/
        ...

    Expected weekly drop shape (two common patterns supported):
      A) <weekly_drop_root>/
           images/1.png,2.png,...
           masks/rule_xx/1.png,2.png,...
      B) <weekly_drop_root>/
           1.png,2.png,...               (images in root)
           masks/rule_xx/1.png,...

    Returns summary dict including id map {old->new} and split assignment.
    """

    images_root = dataset_root / "images"
    masks_root  = dataset_root / "masks"
    splits_dir  = dataset_root / "splits"
    meta_dir    = dataset_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Detect source image dir
    src_images_dir = weekly_drop_root / "images"
    if not src_images_dir.exists():
        src_images_dir = weekly_drop_root  # flat layout

    # Collect new image files (numeric names only)
    new_imgs = _collect_numeric_files(src_images_dir)
    if not new_imgs:
        raise FileNotFoundError(
            f"No numeric images like '1.png' found under: {src_images_dir}"
        )

    # Determine class subdirs present in weekly masks
    src_masks_root = weekly_drop_root / "masks"
    class_dirs: List[Path] = []
    if src_masks_root.exists():
        for sub in sorted(src_masks_root.iterdir()):
            if sub.is_dir():
                class_dirs.append(sub)

    # Compute ID offset from current max id in canonical images/
    start_max = _scan_max_id(images_root)
    offset = start_max

    rng = random.Random(seed)

    id_map: Dict[str, str] = {}  # "old_id_str" -> "new_id_str"
    assigned_train: List[str] = []
    assigned_valid: List[str] = []

    # Copy images and same-index masks with offset
    for src_img in new_imgs:
        m = NUM_RE.match(src_img.name)
        assert m is not None
        old_id = int(m.group(1))
        new_id = offset + old_id  # 1->offset+1, 2->offset+2, ...

        # Copy image
        dst_img = _copy_with_new_id(src_img, images_root, new_id)

        # Copy masks in each class dir if present
        for cdir in class_dirs:
            src_mask = cdir / f"{old_id}{src_img.suffix}"
            if src_mask.exists():
                dst_cdir = masks_root / cdir.name
                _copy_with_new_id(src_mask, dst_cdir, new_id)
            else:
                # If mask missing, we leave it absent; your loader treats missing as all-zero
                pass

        id_map[str(old_id)] = str(new_id)

    # New global max (for info)
    new_global_max = _scan_max_id(images_root)
    (meta_dir / "latest_id.txt").write_text(str(new_global_max), encoding="utf-8")

    # 90/10 split: deterministic but shuffled IDs
    new_ids = sorted(id_map.values(), key=lambda s: int(s))
    rng.shuffle(new_ids)
    n_train = int(round(train_ratio * len(new_ids)))
    new_train_ids = new_ids[:n_train]
    new_valid_ids = new_ids[n_train:]

    # Append to splits
    train_txt = splits_dir / "train.txt"
    valid_txt = splits_dir / "valid.txt"
    _append_lines(train_txt, new_train_ids)
    _append_lines(valid_txt, new_valid_ids)

    # Write an ingest log (nice for audit)
    summary = {
        "weekly_drop_root": str(weekly_drop_root),
        "offset_applied": offset,
        "id_map": id_map,
        "new_train_ids": new_train_ids,
        "new_valid_ids": new_valid_ids,
        "new_global_max_id": new_global_max,
    }
    (meta_dir / "ingest_log.jsonl").open("a", encoding="utf-8").write(
        json.dumps(summary, ensure_ascii=False) + "\n"
    )

    return summary
