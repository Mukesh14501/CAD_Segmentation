# cadseg/cli/ingest_week.py
from __future__ import annotations
import argparse
from pathlib import Path
import json
from cadseg.dataio.ingest import ingest_week

def main():
    ap = argparse.ArgumentParser(description="Ingest a weekly data drop (renumber + split 90/10).")
    ap.add_argument("--data_root", type=str, default="data", help="Canonical dataset root (contains images/, masks/, splits/).")
    ap.add_argument("--weekly_root", type=str, required=True, help="Path to weekly drop folder (or extracted ZIP).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.9, help="Train fraction for new IDs.")
    args = ap.parse_args()

    out = ingest_week(Path(args.data_root), Path(args.weekly_root), seed=args.seed, train_ratio=args.train_ratio)
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
