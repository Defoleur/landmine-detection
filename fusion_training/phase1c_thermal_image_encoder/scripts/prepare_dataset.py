#!/usr/bin/env python3
"""Create an image manifest from split_1 records and processed PNGs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase1c_thermal_image_encoder.data import prepare_records, write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/thermal_landmine_splits/split_1"))
    parser.add_argument("--processed-root", type=Path, default=Path("datasets/thermal_landmine_splits/split_1_processed"))
    parser.add_argument("--manifest", type=Path, default=Path("phase1c_thermal_image_encoder/artifacts/image_manifest.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("phase1c_thermal_image_encoder/artifacts/image_report.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("prim", "condition"), default="prim",
                        help="prim: keep groups together; condition: exclude incomplete conditions and balance season/daytime and mine type")
    parser.add_argument("--mine-type-field", default="exact_class")
    args = parser.parse_args()
    records, diagnostics = prepare_records(args.dataset_root, args.processed_root, args.seed, args.mode, args.mine_type_field)
    write_manifest(records, args.manifest)
    report = {"records": len(records), "splits": {split: sum(r["split"] == split for r in records)
                                                    for split in ("train", "val", "test")},
              "dataset_root": str(args.dataset_root.resolve()),
              "processed_root": str(args.processed_root.resolve()), "seed": args.seed,
              "mode": args.mode, **diagnostics}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
