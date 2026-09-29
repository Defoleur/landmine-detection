#!/usr/bin/env python3
"""Validate image temporal splits, conditions, and target masks."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from phase1d_thermal_image_convlstm.sequence import CONDITIONS, read_manifest


def validate(sequences):
    errors = []
    mine_points = defaultdict(list)
    sequence_ids = set()
    for item in sequences:
        sequence_id = item.get("sequence_id")
        if sequence_id in sequence_ids:
            errors.append(f"duplicate sequence: {sequence_id}")
        sequence_ids.add(sequence_id)
        mine_points[item["mine_id"]].append(item)
        split = item.get("split")
        if split not in {"train", "val", "test"}:
            errors.append(f"invalid split: {sequence_id}")
        if item.get("conditions") != list(CONDITIONS):
            errors.append(
                f"incomplete or incorrectly ordered conditions: {sequence_id}"
            )
        if len(item.get("conditions", [])) != len(set(item.get("conditions", []))):
            errors.append(f"duplicate condition: {sequence_id}")
        target_path = item.get("target_path", "")
        if not target_path or not Path(target_path).exists():
            errors.append(f"missing target mask: {sequence_id}")
        for frame in item.get("frames", []):
            if not Path(frame["image_path"]).exists():
                errors.append(f"missing image: {frame['frame_id']}")
    split_counts = {}
    for mine, mine_sequences in sorted(mine_points.items()):
        counts = {
            split: sum(item["split"] == split for item in mine_sequences)
            for split in ("train", "val", "test")
        }
        split_counts[mine] = counts
        if counts != {"train": 8, "val": 1, "test": 1}:
            errors.append(f"invalid point split for {mine}: {counts}")
    return {
        "valid": not errors,
        "errors": errors,
        "sequences": len(sequences),
        "mines": len(mine_points),
        "split_counts": split_counts,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--report", type=Path)
    a = p.parse_args()
    result = validate(read_manifest(a.manifest))
    print(json.dumps(result, indent=2))
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(result, indent=2) + "\n")
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
