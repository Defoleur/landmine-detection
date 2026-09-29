#!/usr/bin/env python3
"""Build a fixed twelve-condition RGB/rendered-thermal manifest."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from phase1d_thermal_convlstm.sequence import (
    CONDITIONS,
    base_mine_id,
    discover_target_masks,
)

CONDITION_RE = re.compile(r"_(spring|summer|autumn|winter)_(morning|day|evening)$")
POINT_RE = re.compile(r"_(p\d{2})_obstacle_mine_(.+)$")


def discover(dataset_root, processed_root):
    root, processed = (
        Path(dataset_root).resolve(),
        Path(processed_root).resolve() / "images",
    )
    records, rejected = [], []
    for thermal_image in sorted(processed.glob("*.png")):
        condition_match = CONDITION_RE.search(thermal_image.stem)
        point_match = POINT_RE.search(
            thermal_image.stem[: condition_match.start()]
            if condition_match
            else thermal_image.stem
        )
        if not condition_match or not point_match:
            rejected.append(
                {"path": str(thermal_image), "reason": "unparseable_stem"}
            )
            continue
        stem = thermal_image.stem
        record = {
            "frame_id": stem,
            "mine_id": point_match.group(2),
            "point_number": int(point_match.group(1)[1:]),
            "condition": f"{condition_match.group(1)}_{condition_match.group(2)}",
            "rgb_path": str(root / "images" / f"{stem}.png"),
            "thermal_image_path": str(thermal_image),
            "metadata_path": str(root / "metadata" / f"{stem}.json"),
        }
        missing = [
            key
            for key in (
                "rgb_path",
                "thermal_image_path",
                "metadata_path",
            )
            if not Path(record[key]).exists()
        ]
        if missing:
            rejected.append(
                {"frame_id": stem, "reason": "missing_paths", "paths": missing}
            )
            continue
        record.update(json.loads(Path(record["metadata_path"]).read_text()))
        records.append(record)
    if not records:
        raise ValueError("No complete records discovered")
    return records, rejected


def build_sequences(records, seed=42, target_masks=None):
    groups = defaultdict(dict)
    for record in records:
        key = (record["mine_id"], record["point_number"])
        if record["condition"] in groups[key]:
            raise ValueError(f"Duplicate condition: {key} {record['condition']}")
        groups[key][record["condition"]] = record
    incomplete = [
        {"key": list(key), "missing": sorted(set(CONDITIONS) - set(values))}
        for key, values in groups.items()
        if set(values) != set(CONDITIONS)
    ]
    groups = {
        key: values for key, values in groups.items() if set(values) == set(CONDITIONS)
    }
    if target_masks is not None:
        source_keys = set(groups)
        target_keys = set(target_masks)
        missing = sorted(source_keys - target_keys)
        extra = sorted(target_keys - source_keys)
        if missing or extra:
            raise ValueError(
                "Target mask point identities do not match image sequences: "
                f"missing={missing[:3]} extra={extra[:3]}"
            )
    rng = random.Random(seed)
    by_base = defaultdict(list)
    for key in groups:
        by_base[base_mine_id(key[0])].append(key)
    split = {}
    for mine, keys in sorted(by_base.items()):
        keys = sorted(keys)
        rng.shuffle(keys)
        train_count, val_count = int(len(keys) * 0.8), int(len(keys) * 0.1)
        if train_count < 1 or val_count < 1 or train_count + val_count >= len(keys):
            raise ValueError(f"Insufficient points for split group {mine}: {len(keys)}")
        for index, key in enumerate(keys):
            split[key] = (
                "train"
                if index < train_count
                else "val" if index < train_count + val_count else "test"
            )
    sequences = []
    for (mine, point), values in sorted(groups.items()):
        frames = [values[condition] for condition in CONDITIONS]
        target_path = (
            target_masks[(mine, point)]["target_path"]
            if target_masks is not None
            else frames[0].get("target_path", "")
        )
        sequences.append(
            {
                "sequence_id": f"{mine}_p{point:02d}",
                "mine_id": mine,
                "point_number": point,
                "split": split[(mine, point)],
                "conditions": list(CONDITIONS),
                "frames": frames,
                "target_path": target_path,
            }
        )
    return sequences, incomplete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/dataset_fusion_20260921_201549"),
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("datasets/dataset_fusion_20260921_201549_processed"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "phase2_rgb_thermal_image_fusion/artifacts/image_fusion_manifest.json"
        ),
    )
    parser.add_argument(
        "--target-mask-dataset",
        type=Path,
        default=Path("datasets/dataset_fusion_empty_20260922_141207"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "phase2_rgb_thermal_image_fusion/artifacts/image_fusion_manifest_report.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records, rejected = discover(args.dataset_root, args.processed_root)
    target_masks = discover_target_masks(args.target_mask_dataset)
    sequences, incomplete = build_sequences(records, args.seed, target_masks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sequences, indent=2) + "\n")
    report = {
        "source_records": len(records),
        "sequences": len(sequences),
        "rejected": rejected + incomplete,
        "splits": {
            name: sum(item["split"] == name for item in sequences)
            for name in ("train", "val", "test")
        },
        "conditions": list(CONDITIONS),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "processed_root": str(Path(args.processed_root).resolve()),
        "target_mask_dataset": str(Path(args.target_mask_dataset).resolve()),
        "target_mask_points": len(target_masks),
        "seed": args.seed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
