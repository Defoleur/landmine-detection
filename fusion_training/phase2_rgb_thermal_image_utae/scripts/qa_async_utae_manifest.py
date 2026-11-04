#!/usr/bin/env python3
"""Validate asynchronous U-TAE manifest timestamps and observation identities."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def validate(items):
    errors = []
    for sequence in items:
        ids = set()
        previous = -1
        for observation in sorted(sequence.get("observations", []), key=lambda item: item["arrival_index"]):
            if observation["observation_id"] in ids:
                errors.append(f"{sequence['sequence_id']}: duplicate observation ID")
            ids.add(observation["observation_id"])
            acquired = datetime.fromisoformat(observation["acquired_at_utc"].replace("Z", "+00:00"))
            available = datetime.fromisoformat(observation["available_at_utc"].replace("Z", "+00:00"))
            if available < acquired:
                errors.append(f"{sequence['sequence_id']}: availability precedes acquisition")
            if observation["arrival_index"] <= previous:
                errors.append(f"{sequence['sequence_id']}: arrival order is not strict")
            previous = observation["arrival_index"]
    return {"valid": not errors, "sequence_count": len(items), "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    result = validate(json.loads(args.manifest.read_text()))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
