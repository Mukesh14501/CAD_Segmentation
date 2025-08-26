# cadseg/cli/infer.py
"""
CLI: infer.py
- Loads configs, model checkpoint, thresholds
- Runs tiled inference with optional TTA
- Writes per-class binary masks and a color overlay per image
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch

from cadseg.config import load_configs
from cadseg.models.builder import build_model
from cadseg.engine.infer_loop import infer_folder


def _load_ckpt(model: torch.nn.Module, ckpt_path: Path) -> None:
    state = torch.load(ckpt_path, map_location="cpu")
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)


def _load_thresholds(path: Path, class_names: list[str]) -> list[float]:
    if not path.exists():
        return [0.5 for _ in class_names]
    with open(path, "r", encoding="utf-8") as f:
        th = json.load(f)
    if isinstance(th, dict):
        return [float(th.get(c, 0.5)) for c in class_names]
    return [float(x) for x in th]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=str, default="configs", help="Path to configs dir.")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pth).")
    ap.add_argument("--out_dir", type=str, default="outputs/infer", help="Directory to save outputs.")
    ap.add_argument("--tile_bs", type=int, default=8, help="Tile batch size during inference.")
    ap.add_argument("--tta", type=str, default="", help="Comma-separated TTA: hflip,vflip (empty = use configs).")
    ap.add_argument("--merge", type=str, default="", help="Fusion: mean|gmean|max (empty = use configs).")
    ap.add_argument("--limit", type=int, default=0, help="Limit #images (for quick tests).")
    ap.add_argument("--save_probs", action="store_true", help="Save probs.npz per image.")
    ap.add_argument("--no_overlay", action="store_true", help="Do not save overlay.png.")
    ap.add_argument("--no_binary", action="store_true", help="Do not save per-class binary PNGs.")
    ap.add_argument("--thresholds", type=str, default="", help="Override thresholds file (JSON).")
    args = ap.parse_args()

    cfg = load_configs(args.configs)
    ds, md, inf = cfg.dataset, cfg.model, cfg.infer
    class_names = ds.classes

    model = build_model(md)
    _load_ckpt(model, Path(args.ckpt))

    # thresholds
    thr_path = Path(args.thresholds) if args.thresholds else Path(inf.thresholds_file)
    thresholds = _load_thresholds(thr_path, class_names)

    # TTA list
    tta_list = [t.strip() for t in args.tta.split(",") if t.strip()] if args.tta else None

    result = infer_folder(
        model=model,
        ds_cfg=ds,
        class_names=class_names,
        infer_cfg=inf,
        thresholds=thresholds,
        out_dir=args.out_dir,
        save_binary=not args.no_binary,
        save_overlay=not args.no_overlay,
        save_probs_npz=args.save_probs,
        tile_bs=args.tile_bs,
        tta_list=tta_list,
        merge=(args.merge if args.merge else (inf.merge if hasattr(inf, "merge") else "mean")),
        limit=(args.limit if args.limit > 0 else None),
    )
    print(f"\nSaved outputs to: {result['out_dir']}  (images processed: {result['num_images']})")

if __name__ == "__main__":
    main()
