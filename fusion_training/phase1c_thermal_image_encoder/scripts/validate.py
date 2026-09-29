#!/usr/bin/env python3
"""Evaluate an image checkpoint on the validation or test split."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from phase1a_rgb_encoder.metrics import SegmentationMetrics
from phase1c_thermal_image_encoder.data import ImageSegmentationDataset, read_manifest
from phase1c_thermal_image_encoder.models import ImageSegmentationModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("phase1c_thermal_image_encoder/artifacts/image_manifest.jsonl"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = payload.get("config", {})
    records = [r for r in read_manifest(args.manifest) if r["split"] == args.split]
    tile_size = int(config.get("data", {}).get("tile_size", 640))
    loader = DataLoader(ImageSegmentationDataset(records, tile_size), batch_size=args.batch_size, shuffle=False)
    model = ImageSegmentationModel(bool(config.get("model", {}).get("pretrained", False))).to(args.device).eval()
    model.load_state_dict(payload.get("model_state_dict", payload), strict=False)
    metrics = SegmentationMetrics()
    total = 0.0
    with torch.inference_mode():
        for batch in loader:
            logits, _ = model(batch["image"].to(args.device))
            target = batch["target"].to(args.device)
            total += float(torch.nn.functional.binary_cross_entropy_with_logits(logits, target))
            metrics.update(logits, target, batch["target_validity"].to(args.device).bool())
    print(json.dumps({"split": args.split, "samples": len(records), "loss": total / max(len(loader), 1),
                      **metrics.compute()}, indent=2))


if __name__ == "__main__":
    main()
