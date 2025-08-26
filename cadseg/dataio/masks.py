"""
Mask IO and morphology utilities:
- load/save binary masks
- multi-label stacking (C channels)
- small-component removal, dilation/erosion (very mild)
- optional RLE helpers
"""

# cadseg/dataio/masks.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

def list_images(images_dir: Path) -> List[Path]:
    files = [p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    files.sort()
    return files

def image_id_from_path(p: Path) -> str:
    return p.stem  # unique id derived from filename

def load_image_rgb(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

def load_binary_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    # enforce binary {0,1}
    m = (m > 0).astype(np.uint8)
    return m

def validate_mask_values(mask: np.ndarray) -> bool:
    uniq = np.unique(mask)
    return set(uniq.tolist()) <= {0, 1}

def check_alignment(img: np.ndarray, mask: np.ndarray, img_path: Path, mask_path: Path) -> None:
    if img.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Size mismatch: image {img_path.name} has {img.shape[:2]}, "
            f"mask {mask_path.name} has {mask.shape[:2]}"
        )

def per_class_mask_path(masks_root: Path, class_name: str, image_id: str) -> Path:
    # masks/<class_name>/<image_id>.png
    return masks_root / class_name / f"{image_id}.png"

def scan_dataset(
    images_dir: Path,
    masks_dir: Path,
    class_names: List[str],
    max_images: Optional[int] = None,
) -> Dict[str, Dict]:
    """Scan dataset and return a manifest with basic stats."""
    manifest: Dict[str, Dict] = {}
    images = list_images(images_dir)
    if max_images:
        images = images[:max_images]

    for ip in images:
        iid = image_id_from_path(ip)
        entry = {
            "image_path": str(ip),
            "masks": {},
            "present_classes": [],
            "size": None,
        }
        im = load_image_rgb(ip)
        H, W = im.shape[:2]
        entry["size"] = [int(H), int(W)]

        for cname in class_names:
            mp = per_class_mask_path(masks_dir, cname, iid)
            if mp.exists():
                m = load_binary_mask(mp)
                check_alignment(im, m, ip, mp)
                if not validate_mask_values(m):
                    raise ValueError(f"Non-binary values in {mp}")
                area = int(m.sum())
                entry["masks"][cname] = {
                    "path": str(mp),
                    "present": bool(area > 0),
                    "area_px": area,
                }
                if area > 0:
                    entry["present_classes"].append(cname)
            else:
                # Missing mask file = assumed all-zero mask (no positives for that class)
                entry["masks"][cname] = {
                    "path": None,
                    "present": False,
                    "area_px": 0,
                }

        manifest[iid] = entry
    return manifest

def summarize_manifest(manifest: Dict[str, Dict], class_names: List[str]) -> Dict:
    n_images = len(manifest)
    per_class = {c: {"images_with_positive": 0, "total_positive_px": 0} for c in class_names}
    missing_masks = {c: 0 for c in class_names}

    for _, rec in manifest.items():
        for c in class_names:
            m = rec["masks"][c]
            if m["path"] is None:
                missing_masks[c] += 1
            if m["area_px"] > 0:
                per_class[c]["images_with_positive"] += 1
                per_class[c]["total_positive_px"] += m["area_px"]

    summary = {
        "num_images": n_images,
        "per_class": per_class,
        "missing_masks": missing_masks,
    }
    return summary

def save_summary_json(summary: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

def sanity_check_dataset(
    images_dir: Path,
    masks_dir: Path,
    class_names: List[str],
    min_positive_area: int = 1,
    sample_limit: Optional[int] = None,
    summary_out: Optional[Path] = None,
) -> Dict:
    """
    Validates:
      - All images load.
      - Each present mask aligns with its image spatially.
      - Masks are binary {0,1}.
      - Reports per-class coverage and missing mask files.
    """
    manifest = scan_dataset(images_dir, masks_dir, class_names, max_images=sample_limit)
    summary = summarize_manifest(manifest, class_names)

    # Warn on zero coverage
    zero_cov = [c for c, s in summary["per_class"].items() if s["images_with_positive"] == 0]
    if zero_cov:
        print(f"[warn] No positive pixels found for classes: {zero_cov}")

    # Warn if all masks are missing for a class (files not present)
    fully_missing = [c for c, n in summary["missing_masks"].items() if n == len(manifest)]
    if fully_missing:
        print(f"[warn] Mask files entirely missing for classes: {fully_missing}")

    if summary_out is not None:
        save_summary_json(summary, summary_out)

    return {"manifest": manifest, "summary": summary}
