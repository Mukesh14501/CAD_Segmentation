# cadseg/cli/export.py
"""
CLI: export.py
- Exports the trained model to TorchScript and/or ONNX.
- Optionally runs a quick ONNX Runtime sanity check vs PyTorch on a dummy input.

Examples:
  # Export both TorchScript and ONNX with dynamic axes, and verify ONNX numerics
  python -m cadseg.cli.export --configs configs \
      --ckpt runs/exp1/checkpoints/best.pth \
      --out_dir runs/exp1/export --jit trace --onnx --dynamic --check_onnx

  # Export TorchScript-only (script mode)
  python -m cadseg.cli.export --ckpt runs/exp1/checkpoints/best.pth --jit script

  # Export ONNX only with opset 17 and no checks
  python -m cadseg.cli.export --ckpt runs/exp1/checkpoints/best.pth --onnx --opset 17
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from cadseg.config import load_configs
from cadseg.models.builder import build_model, count_params


def _load_ckpt(model: torch.nn.Module, ckpt_path: Path) -> None:
    state = torch.load(ckpt_path, map_location="cpu")
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)


def _dummy_input(h: int, w: int, device: torch.device) -> torch.Tensor:
    # single RGB image, normalized-ish [0,1]
    x = torch.rand(1, 3, h, w, device=device, dtype=torch.float32)
    return x


def export_torchscript(model: torch.nn.Module, out_path: Path, mode: str = "trace", h: int = 1024, w: int = 1024):
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "script":
        ts = torch.jit.script(model)
    else:
        # trace with a dummy example
        example = _dummy_input(h, w, device=next(model.parameters()).device)
        ts = torch.jit.trace(model, example, strict=False)
    # Optimize for inference where available
    try:
        ts = torch.jit.optimize_for_inference(ts)
    except Exception:
        pass

    ts_path = out_path.with_suffix(".ts")
    ts.save(str(ts_path))
    return ts_path


def export_onnx(
    model: torch.nn.Module,
    out_path: Path,
    opset: int = 17,
    dynamic: bool = True,
    h: int = 1024,
    w: int = 1024,
):
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = _dummy_input(h, w, device=next(model.parameters()).device)
    input_names = ["input"]
    output_names = ["logits"]

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        }

    torch.onnx.export(
        model, x, str(out_path),
        export_params=True, opset_version=opset,
        do_constant_folding=True,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    return out_path


def _ort_check(onnx_path: Path, model: torch.nn.Module, h: int, w: int, atol: float = 1e-3, rtol: float = 1e-2) -> Dict[str, Any]:
    """
    Compare ORT vs PyTorch on a single random input.
    Returns error stats.
    """
    try:
        import onnxruntime as ort
    except Exception as e:
        return {"ok": False, "reason": f"onnxruntime not installed: {e}"}

    device = next(model.parameters()).device
    model.eval()
    x = _dummy_input(h, w, device=device)
    with torch.no_grad():
        y_pt = model(x).cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    y_ort = sess.run(None, {"input": x.cpu().numpy()})[0]

    # Align shapes and compute errors
    if y_ort.shape != y_pt.shape:
        return {"ok": False, "reason": f"shape mismatch ORT {y_ort.shape} vs PT {y_pt.shape}"}

    diff = np.abs(y_ort - y_pt)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    rel = diff / (np.abs(y_pt) + 1e-6)
    max_rel = float(rel.max())
    mean_rel = float(rel.mean())

    ok = (max_abs <= atol + rtol * np.abs(y_pt).max() + 1e-6)
    return {
        "ok": ok,
        "max_abs": round(max_abs, 6),
        "mean_abs": round(mean_abs, 6),
        "max_rel": round(max_rel, 6),
        "mean_rel": round(mean_rel, 6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=str, default="configs", help="Path to configs dir.")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pth).")
    ap.add_argument("--out_dir", type=str, default="runs/export", help="Output directory for exported artifacts.")
    # TorchScript
    ap.add_argument("--jit", type=str, default="", choices=["", "trace", "script"], help="Export TorchScript mode.")
    # ONNX
    ap.add_argument("--onnx", action="store_true", help="Export ONNX as well.")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    ap.add_argument("--dynamic", action="store_true", help="Use dynamic axes for ONNX (batch/height/width).")
    ap.add_argument("--check_onnx", action="store_true", help="Run ONNX Runtime numeric sanity check.")
    # Dummy input size for export/checks (defaults to dataset tile_size)
    ap.add_argument("--h", type=int, default=0, help="Dummy height for export/checks.")
    ap.add_argument("--w", type=int, default=0, help="Dummy width for export/checks.")
    args = ap.parse_args()

    cfg = load_configs(args.configs)
    ds, md = cfg.dataset, cfg.model

    # Build & load model
    model = build_model(md)
    _load_ckpt(model, Path(args.ckpt))
    model.eval()

    # Device: keep on CPU for portability (small memory footprint during export)
    device = torch.device("cpu")
    model.to(device)

    # Dummy example size
    H = int(args.h) if args.h > 0 else int(ds.tile_size)
    W = int(args.w) if args.w > 0 else int(ds.tile_size)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    meta = {
        "encoder": md.encoder,
        "arch": md.arch,
        "num_classes": md.num_classes,
        "in_channels": md.in_channels,
        "classes": ds.classes,
        "tile_size": ds.tile_size,
        "export_h": H, "export_w": W,
    }
    with open(out_dir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Exports
    ts_path = None
    onnx_path = None

    if args.jit:
        ts_path = export_torchscript(model, out_dir / "model", mode=args.jit, h=H, w=W)
        print(f"[export] TorchScript saved: {ts_path}")

    if args.onnx:
        onnx_path = export_onnx(model, out_dir / "model.onnx", opset=args.opset, dynamic=args.dynamic, h=H, w=W)
        print(f"[export] ONNX saved: {onnx_path}")

        if args.check_onnx:
            stats = _ort_check(onnx_path, model, H, W)
            with open(out_dir / "onnx_check.json", "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            print(f"[export] ONNX check: {stats}")

    # Param counts
    tot, trn = count_params(model)
    print(f"[export] Params: total={tot:,} trainable={trn:,}")
    print(f"[export] Done. Artifacts in: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
