# safer default_collate
from __future__ import annotations
from typing import List, Dict, Any
import torch

def default_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    # Filter out None or malformed samples
    batch = [b for b in batch if b is not None and "image" in b and isinstance(b["image"], torch.Tensor)]
    if not batch:
        raise RuntimeError(
            "Empty batch produced by DataLoader. Likely causes: "
            "(1) ClassBalancedBatchSampler picked only invalid/empty tiles, "
            "(2) Dataset __getitem__ returns None when min_positive_area is too strict, "
            "(3) Masks/classes mismatch leading to no positives."
        )

    images = torch.stack([b["image"] for b in batch], dim=0)

    out: Dict[str, Any] = {"image": images}
    if "mask" in batch[0] and isinstance(batch[0]["mask"], torch.Tensor):
        masks = [b["mask"] for b in batch if b.get("mask") is not None]
        if len(masks) == len(batch):
            out["mask"] = torch.stack(masks, dim=0)

    # Pass through any optional fields if all samples have them
    for k in ("meta", "id", "tile_idx"):
        if all(k in b for b in batch):
            out[k] = [b[k] for b in batch]
    return out
