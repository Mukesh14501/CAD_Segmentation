# cadseg/cli/validate.py
"""
CLI: validate.py
- Loads configs and model (optionally a checkpoint)
- Runs tiled validation over full images (no leakage from tiling)
- Prints per-class + macro metrics
"""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import torch

from cadseg.config import load_configs
from cadseg.models.builder import build_model
from cadseg.engine.eval_loop import evaluate_dataset


def _load_ckpt(model: torch.nn.Module, ckpt_path: Path) -> None:
    state = torch.load(ckpt_path, map_location="cpu")
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=str, default="configs", help="Path to configs dir.")
    ap.add_argument("--ckpt", type=str, default="", help="Path to model checkpoint (.pth).")
    ap.add_argument("--thresholds", type=str, default="", help="Path to thresholds.json (optional).")
    ap.add_argument("--limit", type=int, default=0, help="Limit #images for quick check.")
    ap.add_argument("--tile_bs", type=int, default=8, help="Tile batch size during inference.")
    args = ap.parse_args()

    cfg = load_configs(args.configs)
    ds, md = cfg.dataset, cfg.model
    class_names = ds.classes

    model = build_model(md)

    if args.ckpt:
        _load_ckpt(model, Path(args.ckpt))

    thresholds = 0.5
    if args.thresholds:
        with open(args.thresholds, "r", encoding="utf-8") as f:
            th = json.load(f)
        # Accept dict {class_name: thr} or list
        if isinstance(th, dict):
            thresholds = [float(th.get(c, 0.5)) for c in class_names]
        else:
            thresholds = th

    metrics = evaluate_dataset(
        model, ds, class_names, cfg.infer,
        thresholds=thresholds,
        tile_bs=args.tile_bs,
        limit=(args.limit if args.limit > 0 else None),
    )

    print("\n=== Validation Metrics ===")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
