# cadseg/cli/prepare_dataset.py
"""
CLI: prepare_dataset.py
- Loads configs
- Validates data contract (images/masks alignment, binary masks)
- Emits dataset summary JSON to data/meta/dataset_summary.json
"""
from pathlib import Path
import argparse
from cadseg.config import load_configs
from cadseg.dataio.masks import sanity_check_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=str, default="configs", help="Path to configs directory.")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of images scanned (0 = all).")
    args = ap.parse_args()

    cfg = load_configs(args.configs)
    ds = cfg.dataset

    print("[prepare_dataset] Using dataset root:", ds.root_path)
    print("[prepare_dataset] Images:", ds.images_path)
    print("[prepare_dataset] Masks :", ds.masks_path)
    print("[prepare_dataset] Classes:", ds.classes, f"(C={ds.C})")

    limit = args.limit if args.limit and args.limit > 0 else None

    out_json = ds.root_path / "meta" / "dataset_summary.json"
    result = sanity_check_dataset(
        images_dir=ds.images_path,
        masks_dir=ds.masks_path,
        class_names=ds.classes,
        min_positive_area=ds.min_positive_area,
        sample_limit=limit,
        summary_out=out_json,
    )

    summary = result["summary"]
    print("\n=== Dataset Summary ===")
    print("Num images:", summary["num_images"])
    print("Missing masks per class:", summary["missing_masks"])
    print("Per-class positives (images_with_positive):")
    for c, s in summary["per_class"].items():
        print(f"  - {c}: {s['images_with_positive']} images, {s['total_positive_px']} px total")

    print("\nSaved summary to:", out_json)

if __name__ == "__main__":
    main()
