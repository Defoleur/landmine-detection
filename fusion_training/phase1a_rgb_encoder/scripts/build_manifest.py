#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from phase1a_rgb_encoder.data import assign_spatial_splits, dataset_report, discover_records, write_manifest

parser = argparse.ArgumentParser()
parser.add_argument("dataset_root", type=Path)
parser.add_argument("--output", type=Path, default=Path("artifacts/dataset_manifest.csv"))
parser.add_argument("--report", type=Path, default=Path("artifacts/dataset_report.json"))
parser.add_argument("--block-size", type=float, default=10.0)
args = parser.parse_args()
records = assign_spatial_splits(discover_records(args.dataset_root, args.block_size))
write_manifest(records, args.output)
args.report.parent.mkdir(parents=True, exist_ok=True)
args.report.write_text(json.dumps(dataset_report(records), indent=2))
print(json.dumps(dataset_report(records), indent=2))
