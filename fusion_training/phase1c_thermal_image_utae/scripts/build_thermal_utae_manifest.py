#!/usr/bin/env python3
"""Convert a legacy thermal-image manifest to the timestamped U-TAE contract."""

import argparse
import json
from pathlib import Path

from phase2_rgb_thermal_image_utae.manifest import add_derived_observation_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--year", type=int, default=2026)
    args = parser.parse_args()
    items = json.loads(args.input.read_text())
    converted = add_derived_observation_metadata(items, args.year)
    problems = []
    for sequence in converted:
        sequence["observations"] = [item for item in sequence["observations"] if item["modality"] == "thermal"]
        frames = sequence.get("frames", [])
        if len(frames) != 12:
            problems.append(f"{sequence.get('sequence_id', '<unknown>')}: expected 12 frames, got {len(frames)}")
        if not sequence.get("target_path") or not Path(sequence["target_path"]).exists():
            problems.append(f"{sequence.get('sequence_id', '<unknown>')}: missing target path")
        for frame in frames:
            if not frame.get("image_path") or not Path(frame["image_path"]).exists():
                problems.append(f"{sequence.get('sequence_id', '<unknown>')}: missing frame image")
            if not frame.get("target_path") or not Path(frame["target_path"]).exists():
                problems.append(f"{sequence.get('sequence_id', '<unknown>')}: missing frame target")
    if problems:
        raise SystemExit("thermal U-TAE manifest validation failed:\n" + "\n".join(problems[:20]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(converted, indent=2) + "\n")
    report = {
        "sequences": len(converted),
        "splits": {split: sum(item.get("split") == split for item in converted)
                   for split in ("train", "val", "test")},
        "thermal_observations": sum(len(item.get("observations", [])) for item in converted),
        "year": args.year,
    }
    report_path = args.report or args.output.with_name(args.output.stem + "_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
