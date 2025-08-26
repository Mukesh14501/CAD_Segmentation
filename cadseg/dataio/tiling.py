# cadseg/dataio/tiling.py
from __future__ import annotations
from typing import Tuple, List
import numpy as np

def generate_grid(H: int, W: int, tile: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    """
    Public version of grid generator (also used by inference loops).
    Returns list of windows (y0,x0,y1,x1) covering the image.
    """
    stride = max(1, tile - overlap)
    ys = list(range(0, max(1, H - tile + 1), stride))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if len(ys) == 0: ys = [0]
    if len(xs) == 0: xs = [0]
    if ys[-1] + tile < H:
        ys.append(H - tile)
    if xs[-1] + tile < W:
        xs.append(W - tile)

    windows = []
    for y0 in ys:
        for x0 in xs:
            y1 = min(H, y0 + tile)
            x1 = min(W, x0 + tile)
            windows.append((y0, x0, y1, x1))
    return windows


def hann_window_1d(n: int) -> np.ndarray:
    """Symmetric Hann window (1D) of length n in [0,1]."""
    if n <= 1:
        return np.ones((n,), dtype=np.float32)
    w = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / (n - 1)))
    return w.astype(np.float32)


def make_weight_window(h: int, w: int) -> np.ndarray:
    """
    2D separable Hann window -> emphasizes center, tapers edges.
    Returns (h,w) float32 in [0,1].
    """
    wy = hann_window_1d(h)
    wx = hann_window_1d(w)
    W2d = np.outer(wy, wx)
    # avoid zeros to prevent division instability; clamp minimum
    eps = 1e-6
    W2d = np.clip(W2d, eps, 1.0)
    return W2d.astype(np.float32)


def stitch_probs(
    tiles: List[np.ndarray],
    windows: List[Tuple[int, int, int, int]],
    out_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Blend tile probabilities into a full-canvas probability map.
    Args:
      tiles: list of (C,h,w) float arrays in [0,1] or logits (if you blend logits, do sigmoid after)
      windows: list of (y0,x0,y1,x1) matching tiles
      out_shape: (C,H,W) of final canvas
    Returns:
      probs: (C,H,W) float32
    """
    assert len(tiles) == len(windows), "tiles/windows length mismatch"
    C, H, W = out_shape
    acc = np.zeros((C, H, W), dtype=np.float32)
    denom = np.zeros((1, H, W), dtype=np.float32)  # broadcast per channel

    for t, (y0, x0, y1, x1) in zip(tiles, windows):
        # t shape (C,h,w)
        _, h, w = t.shape
        wgt = make_weight_window(h, w)  # (h,w)
        acc[:, y0:y1, x0:x1] += t * wgt[None, ...]
        denom[:, y0:y1, x0:x1] += wgt[None, ...]
    eps = 1e-8
    probs = acc / (denom + eps)
    return probs.astype(np.float32)
