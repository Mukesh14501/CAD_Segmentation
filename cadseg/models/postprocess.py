# cadseg/models/postprocess.py
from __future__ import annotations
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np

try:
    from skimage.morphology import remove_small_objects
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _as_thresholds(thr: Union[float, Sequence[float]], C: int) -> np.ndarray:
    if isinstance(thr, (list, tuple, np.ndarray)):
        arr = np.asarray(thr, dtype=np.float32)
        if arr.size != C:
            raise ValueError(f"Threshold length {arr.size} != num classes {C}")
        return arr.astype(np.float32)
    return np.full((C,), float(thr), dtype=np.float32)


def apply_thresholds(
    probs: np.ndarray,                   # (C,H,W) float in [0,1]
    thresholds: Union[float, Sequence[float]] = 0.5,
) -> np.ndarray:
    """Return binary masks (C,H,W) in {0,1} using per-class thresholds."""
    C, H, W = probs.shape
    thr = _as_thresholds(thresholds, C).reshape(C, 1, 1)
    return (probs >= thr).astype(np.uint8)


def remove_small_components_per_class(
    masks_bin: np.ndarray,               # (C,H,W) uint8 {0,1}
    min_areas: Optional[Sequence[int]] = None,
    connectivity: int = 1,
) -> np.ndarray:
    """Remove connected components smaller than `min_areas[c]` for each class."""
    C, H, W = masks_bin.shape
    if min_areas is None:
        return masks_bin
    if len(min_areas) != C:
        raise ValueError(f"min_areas length {len(min_areas)} != num classes {C}")

    out = masks_bin.copy()
    for c in range(C):
        area = int(min_areas[c]) if min_areas[c] is not None else 0
        if area <= 1:
            continue
        if HAS_SKIMAGE:
            # skimage expects bool
            cleaned = remove_small_objects(out[c].astype(bool), min_size=area, connectivity=connectivity)
            out[c] = cleaned.astype(np.uint8)
        else:
            # Fallback: naive OpenCV connected components
            import cv2
            nb, labels = cv2.connectedComponents(out[c].astype(np.uint8), connectivity=4 if connectivity == 1 else 8)
            # Count sizes
            if nb > 1:
                sizes = np.bincount(labels.ravel())
                # 0 is background
                remove = np.where(sizes < area)[0]
                # Exclude background idx 0
                remove = remove[remove != 0]
                for rid in remove:
                    out[c][labels == rid] = 0
    return out


def postprocess_logits(
    logits: np.ndarray,                  # (C,H,W) float (logits)
    *,
    thresholds: Union[float, Sequence[float]] = 0.5,
    min_component_area: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert logits → probs → binary masks → (optional) small-blob removal.
    Returns (probs, masks_bin) with shapes (C,H,W).
    """
    probs = sigmoid_np(logits.astype(np.float32))
    masks_bin = apply_thresholds(probs, thresholds)
    masks_bin = remove_small_components_per_class(masks_bin, min_component_area)
    return probs, masks_bin
