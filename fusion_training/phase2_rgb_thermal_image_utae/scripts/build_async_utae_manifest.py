#!/usr/bin/env python3
"""Build an asynchronous U-TAE manifest from the existing fusion dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase2_rgb_thermal_image_fusion.scripts.build_image_fusion_manifest import build_sequences, discover
from phase2_rgb_thermal_image_utae.manifest import add_derived_observation_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path,
                        help="Convert an existing fusion manifest instead of discovering source files")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.input_manifest and (args.dataset_root is None or args.processed_root is None):
        parser.error("--dataset-root and --processed-root are required without --input-manifest")
    if args.input_manifest:
        sequences = json.loads(args.input_manifest.read_text())
        rejected, incomplete = [], []
    else:
        records, rejected = discover(args.dataset_root, args.processed_root)
        sequences, incomplete = build_sequences(records, seed=args.seed)
    sequences = add_derived_observation_metadata(sequences, args.year)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sequences, indent=2) + "\n")
    print(json.dumps({"sequences": len(sequences), "rejected": len(rejected), "incomplete": len(incomplete),
                      "timestamp_source": "derived_from_condition", "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
