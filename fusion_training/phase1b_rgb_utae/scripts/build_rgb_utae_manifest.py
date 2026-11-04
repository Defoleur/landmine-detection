#!/usr/bin/env python3
"""Convert a legacy RGB temporal manifest to the timestamped U-TAE contract."""

import argparse
import json
from pathlib import Path

from phase2_rgb_thermal_image_utae.manifest import add_derived_observation_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2026)
    args = parser.parse_args()
    items = json.loads(args.input.read_text())
    converted = add_derived_observation_metadata(items, args.year)
    for sequence in converted:
        sequence["observations"] = [item for item in sequence["observations"] if item["modality"] == "rgb"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(converted, indent=2) + "\n")


if __name__ == "__main__":
    main()
