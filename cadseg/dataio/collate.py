"""
Custom collate functions for tile batches, dtype control, channel stacking.
"""

# cadseg/dataio/collate.py
from __future__ import annotations
from typing import Dict, Any, List
import torch

def default_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate for tile/full-image samples with metadata.
    - Stacks 'image' and 'mask' if present.
    - 'meta' is returned as a list of dicts (one per sample).
    """
    images = [b["image"] for b in batch]
    images = torch.stack(images, dim=0)

    collated: Dict[str, Any] = {"image": images, "meta": [b.get("meta", {}) for b in batch]}

    if "mask" in batch[0] and batch[0]["mask"] is not None:
        masks = [b["mask"] for b in batch]
        collated["mask"] = torch.stack(masks, dim=0)

    return collated
