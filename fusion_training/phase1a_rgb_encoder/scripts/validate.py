#!/usr/bin/env python3
"""Evaluate an RGB checkpoint on the validation or test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from phase1a_rgb_encoder.data import RGBDataset, read_manifest
from phase1a_rgb_encoder.losses import masked_bce_with_logits
from phase1a_rgb_encoder.metrics import SegmentationMetrics
from phase1a_rgb_encoder.models import RGBSegmentationModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("phase1a_rgb_encoder/artifacts/dataset_rgb_main_manifest.csv"),
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, help="Write the metrics JSON to this path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = payload.get("config", {})
    records = [record for record in read_manifest(args.manifest) if record["split"] == args.split]
    if not records:
        raise ValueError(f"No records found for split {args.split!r}")

    tile_size = int(config.get("data", {}).get("tile_size", 640))
    loader = DataLoader(
        RGBDataset(records, augment=False, tile_size=tile_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
    )
    model = RGBSegmentationModel(pretrained=False).to(args.device).eval()
    model.load_state_dict(payload.get("model_state_dict", payload))

    metrics = SegmentationMetrics()
    total_loss = 0.0
    total_samples = 0
    with torch.inference_mode():
        for batch in loader:
            rgb = batch["rgb"].to(args.device)
            valid = batch["valid"].to(args.device)
            target = batch["target"].to(args.device)
            target_valid = batch["target_valid"].to(args.device)
            logits, _ = model(rgb, valid, batch.get("timestamp"))
            loss = masked_bce_with_logits(logits, target, target_valid, valid)
            evaluation_mask = target_valid.bool() & valid.bool()
            metrics.update(logits, target, evaluation_mask, args.threshold)
            batch_size = rgb.shape[0]
            total_loss += float(loss) * batch_size
            total_samples += batch_size

    result = {
        "split": args.split,
        "samples": len(records),
        "tile_size": tile_size,
        "threshold": args.threshold,
        "loss": total_loss / max(total_samples, 1),
        **metrics.compute(),
    }
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
