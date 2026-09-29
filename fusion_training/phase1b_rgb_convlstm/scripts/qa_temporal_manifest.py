#!/usr/bin/env python3
"""Run split, completeness, duplicate, and per-landmine point checks."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from phase1b_rgb_convlstm.data import CONDITIONS, read_temporal_manifest


def validate(samples):
    errors = []
    mine_points = defaultdict(list)
    for sample in samples:
        mine_points[sample["mine_id"]].append(sample)
        if len(sample["conditions"]) != len(set(sample["conditions"])):
            errors.append(f"duplicate condition: {sample['sequence_id']}")
        if set(sample["conditions"]) != set(CONDITIONS):
            errors.append(f"incomplete conditions: {sample['sequence_id']}")
    for mine, mine_samples in mine_points.items():
        counts = {split: sum(sample["split"] == split for sample in mine_samples)
                  for split in ("train", "val", "test")}
        if counts != {"train": 8, "val": 1, "test": 1}:
            errors.append(f"invalid point split for {mine}: {counts}")
    return {"samples": len(samples), "mines": len(mine_points),
            "split_counts": {mine: {split: sum(sample["split"] == split for sample in mine_samples)
                                     for split in ("train", "val", "test")}
                              for mine, mine_samples in sorted(mine_points.items())},
            "errors": errors, "valid": not errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate(read_temporal_manifest(args.manifest))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
