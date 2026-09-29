#!/usr/bin/env python3
"""Create a positive-only manifest from a direct RGB mask dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from phase1a_rgb_encoder.data import dataset_report, read_manifest, write_manifest


def has_positive_mask(record: dict) -> bool:
    mask_path = record.get("mask_path", "")
    if not mask_path:
        raise ValueError(f"Record has no mask_path: {record.get('acquisition_id', '<unknown>')}")
    with Image.open(mask_path) as mask:
        return bool(np.asarray(mask).max() > 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Source manifest generated from dataset_rgb_main_mask")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output positive-only manifest")
    parser.add_argument("--report", type=Path, required=True,
                        help="Output JSON dataset report")
    args = parser.parse_args()

    records = read_manifest(args.manifest)
    positive_records = [record for record in records if has_positive_mask(record)]
    if not positive_records:
        raise ValueError("No positive records found in source manifest")

    write_manifest(positive_records, args.output)
    report = dataset_report(positive_records)
    report.update({
        "source_manifest": str(args.manifest.resolve()),
        "filter": "mask_path contains at least one nonzero pixel",
        "source_records": len(records),
        "excluded_empty_records": len(records) - len(positive_records),
    })
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
