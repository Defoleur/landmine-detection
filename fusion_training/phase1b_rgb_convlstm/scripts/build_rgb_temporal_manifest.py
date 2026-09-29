#!/usr/bin/env python3
"""Build one 12-condition sample per landmine point."""

import argparse
import json
from pathlib import Path

from phase1b_rgb_convlstm.data import build_point_manifest, write_temporal_manifest
from phase1b_rgb_convlstm.data.feature_cache import attach_cache_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("phase1b_rgb_convlstm/artifacts/temporal_manifest.json"))
    parser.add_argument("--report", type=Path, default=Path("phase1b_rgb_convlstm/artifacts/temporal_report.json"))
    parser.add_argument("--target-condition", default="autumn_day")
    parser.add_argument("--target-mask-dataset", type=Path,
                        help="Dataset containing one full-scene target mask per mine/point")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, help="Attach paths from an existing dataset-level feature cache")
    args = parser.parse_args()
    samples, report = build_point_manifest(args.dataset_root, args.target_condition, args.seed,
                                           target_mask_dataset=args.target_mask_dataset)
    if args.cache_dir:
        samples = attach_cache_paths(samples, args.cache_dir)
        report["feature_cache"] = str(args.cache_dir.resolve())
    write_temporal_manifest(samples, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ("source_frames", "candidate_samples", "accepted_samples", "rejected_samples", "splits")}, indent=2))


if __name__ == "__main__":
    main()
