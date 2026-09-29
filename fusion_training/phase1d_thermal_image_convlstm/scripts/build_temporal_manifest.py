#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from phase1d_thermal_image_convlstm.sequence import (
    build_sequence_manifest,
    write_manifest,
)


def main():
    p = argparse.ArgumentParser(
        description="Build fixed-condition sequences from processed thermal images."
    )
    p.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=Path("datasets/dataset_thermal_points_20260920_151536"),
    )
    p.add_argument(
        "--processed-root",
        type=Path,
        default=Path("datasets/dataset_thermal_points_20260920_151536_processed"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "phase1d_thermal_image_convlstm/artifacts/image_temporal_manifest.json"
        ),
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path(
            "phase1d_thermal_image_convlstm/artifacts/image_temporal_report.json"
        ),
    )
    p.add_argument(
        "--target-mask-dataset",
        type=Path,
        default=Path("datasets/dataset_thermal_points_masks_20260922_111237"),
        help="Dataset containing one target mask per mine and point",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--merge-mine-variants", action="store_true")
    p.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
    )
    a = p.parse_args()
    if a.split_ratios and not (
        all(0 < ratio < 1 for ratio in a.split_ratios) and sum(a.split_ratios) <= 1
    ):
        raise SystemExit(
            "--split-ratios must contain positive ratios totaling at most 1"
        )
    sequences, report = build_sequence_manifest(
        dataset_root=a.dataset_root,
        processed_root=a.processed_root,
        seed=a.seed,
        require_complete=not a.allow_incomplete,
        target_mask_dataset=a.target_mask_dataset,
        merge_mine_variants=a.merge_mine_variants,
        split_ratios=a.split_ratios,
    )
    write_manifest(sequences, a.output)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
