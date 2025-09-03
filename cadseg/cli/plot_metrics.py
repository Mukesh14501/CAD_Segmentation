# cadseg/cli/plot_metrics.py
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

def _read_metrics_files(run_dir: Path) -> List[Dict]:
    metr_dir = run_dir / "metrics"
    if not metr_dir.exists():
        raise FileNotFoundError(f"Metrics folder not found: {metr_dir}")
    files = sorted(metr_dir.glob("epoch_*.json"),
                   key=lambda p: int(re.findall(r"epoch_(\d+)", p.stem)[0]))
    out = []
    for p in files:
        try:
            d = json.loads(p.read_text())
            d["_epoch"] = int(re.findall(r"epoch_(\d+)", p.stem)[0])
            out.append(d)
        except Exception as e:
            print(f"[warn] skipping {p.name}: {e}")
    if not out:
        raise RuntimeError(f"No readable metrics in {metr_dir}")
    return out

def _extract_series(history: List[Dict], key_path: List[str], default: float = np.nan) -> np.ndarray:
    vals = []
    for d in history:
        x = d
        try:
            for k in key_path:
                x = x[k]
            vals.append(float(x))
        except Exception:
            vals.append(default)
    return np.array(vals, dtype=float)

def _smooth(y: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    if np.isnan(y).all() or len(y) < 3:
        return y
    out = y.copy()
    m = np.isnan(out)
    if m.any():
        # simple fill for gaps
        out[m] = np.interp(np.flatnonzero(m), np.flatnonzero(~m), out[~m])
    for i in range(1, len(out)):
        out[i] = alpha * out[i] + (1 - alpha) * out[i-1]
    return out

def _save_csv(run_dir: Path, epochs: np.ndarray, series: Dict[str, np.ndarray]):
    import csv
    outp = run_dir / "plots" / "metrics_summary.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    fields = ["epoch"] + list(series.keys())
    with outp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for i in range(len(epochs)):
            row = [int(epochs[i])] + [float(series[k][i]) for k in series]
            w.writerow(row)
    print(f"[save] {outp}")

def _plot_lines(run_dir: Path, title: str, epochs: np.ndarray, lines: Dict[str, np.ndarray], fname: str):
    outdir = run_dir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8,4.5))
    for name, y in lines.items():
        plt.plot(epochs, y, label=name)
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    outp = outdir / fname
    plt.tight_layout()
    plt.savefig(outp, dpi=150)
    plt.close()
    print(f"[save] {outp}")

def main():
    ap = argparse.ArgumentParser(description="Plot epoch vs metrics from run_dir/metrics/epoch_*.json")
    ap.add_argument("--run_dir", required=True, help="e.g., runs/exp1")
    ap.add_argument("--smooth", type=float, default=0.0, help="EMA smoothing alpha (0=off, e.g., 0.25)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    hist = _read_metrics_files(run_dir)

    epochs = np.array([d["_epoch"] for d in hist], dtype=int)

    # Macro (recommended for stakeholders)
    macro_iou   = _extract_series(hist, ["macro_all","iou"])
    macro_dice  = _extract_series(hist, ["macro_all","dice"])
    macro_prec  = _extract_series(hist, ["macro_all","precision"])
    macro_rec   = _extract_series(hist, ["macro_all","recall"])

    # Micro / accuracy (optional)
    micro_iou   = _extract_series(hist, ["micro","iou"])
    micro_dice  = _extract_series(hist, ["micro","dice"])
    accuracy    = _extract_series(hist, ["micro","accuracy"])

    # Optional: train_loss if you added it to the saved metrics
    train_loss  = _extract_series(hist, ["train_loss"], default=np.nan)

    if args.smooth > 0:
        macro_iou  = _smooth(macro_iou, args.smooth)
        macro_dice = _smooth(macro_dice, args.smooth)
        macro_prec = _smooth(macro_prec, args.smooth)
        macro_rec  = _smooth(macro_rec, args.smooth)
        micro_iou  = _smooth(micro_iou, args.smooth)
        micro_dice = _smooth(micro_dice, args.smooth)
        accuracy   = _smooth(accuracy, args.smooth)
        train_loss = _smooth(train_loss, args.smooth)

    # Save CSV summary
    _save_csv(run_dir, epochs, {
        "macro_iou": macro_iou,
        "macro_dice": macro_dice,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "micro_iou": micro_iou,
        "micro_dice": micro_dice,
        "accuracy": accuracy,
        "train_loss": train_loss,
    })

    # Plots
    _plot_lines(run_dir, "Macro IoU / Dice", epochs,
                {"mIoU (macro)": macro_iou, "Dice (macro)": macro_dice},
                "epoch_vs_macro_iou_dice.png")

    _plot_lines(run_dir, "Precision / Recall (macro)", epochs,
                {"Precision": macro_prec, "Recall": macro_rec},
                "epoch_vs_precision_recall.png")

    if not np.isnan(train_loss).all():
        _plot_lines(run_dir, "Train Loss", epochs, {"Train Loss": train_loss},
                    "epoch_vs_train_loss.png")

    _plot_lines(run_dir, "Accuracy (micro)", epochs, {"Accuracy": accuracy},
                "epoch_vs_accuracy.png")

if __name__ == "__main__":
    main()
