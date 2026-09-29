#!/usr/bin/env python3
"""Validate image fusion manifest paths, slots, duplicates, and image formats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from phase1d_thermal_convlstm.sequence import CONDITIONS


def validate(manifest):
    errors, ids, counts = [], set(), {name: 0 for name in ("train", "val", "test")}
    for sequence in manifest:
        sid = sequence.get("sequence_id")
        if sid in ids:
            errors.append(f"duplicate sequence_id {sid}")
        ids.add(sid)
        counts[sequence.get("split", "")] = counts.get(sequence.get("split", ""), 0) + 1
        frames = sequence.get("frames", [])
        if [frame.get("condition") for frame in frames] != list(CONDITIONS):
            errors.append(f"{sid}: conditions are not fixed twelve slots")
        seen = set()
        for frame in frames:
            if frame["condition"] in seen:
                errors.append(f"{sid}: duplicate condition {frame['condition']}")
            seen.add(frame["condition"])
            for key in (
                "rgb_path",
                "thermal_image_path",
                "metadata_path",
            ):
                if not Path(frame[key]).exists():
                    errors.append(f"{sid}: missing {key}")
            for key in ("rgb_path", "thermal_image_path"):
                try:
                    with Image.open(frame[key]) as image:
                        if (
                            image.width < 2
                            or image.height < 2
                            or image.mode not in ("RGB", "RGBA", "L")
                        ):
                            errors.append(f"{sid}: invalid {key} image")
                except (OSError, ValueError) as exc:
                    errors.append(f"{sid}: unreadable {key}: {exc}")
        if not Path(sequence["target_path"]).exists():
            errors.append(f"{sid}: missing target")
    return {
        "sequences": len(manifest),
        "splits": counts,
        "errors": errors,
        "valid": not errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate(json.loads(args.manifest.read_text()))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
