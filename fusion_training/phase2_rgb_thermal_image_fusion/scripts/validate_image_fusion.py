#!/usr/bin/env python3
"""Run finite-loss validation for an image fusion checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from phase1a_rgb_encoder.losses import masked_bce_with_logits
from phase1a_rgb_encoder.metrics import SegmentationMetrics
from phase2_rgb_thermal_image_fusion.data import (
    ImageFusionSequenceDataset,
    read_manifest,
)
from phase2_rgb_thermal_image_fusion.models import ImageFusionSegmenter


def load_compatible(model, payload):
    state = payload.get("model_state_dict", payload)
    expected = model.state_dict()
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(
        {
            key: value
            for key, value in state.items()
            if key in expected and expected[key].shape == value.shape
        },
        strict=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    samples = [
        item for item in read_manifest(args.manifest) if item["split"] == args.split
    ]
    loader = DataLoader(
        ImageFusionSequenceDataset(samples, args.tile_size),
        batch_size=1,
    )
    model = ImageFusionSegmenter().to(args.device)
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    load_compatible(model, payload)
    model.eval()
    total = 0.0
    metrics = SegmentationMetrics()
    with torch.inference_mode():
        for batch in loader:
            logits, _ = model(
                batch["rgb_sequence"].to(args.device),
                batch["thermal_image"].to(args.device),
                batch["rgb_available"].to(args.device),
                batch["thermal_available"].to(args.device),
                freeze_branches=True,
            )
            target = batch["target"].to(args.device)
            valid = batch["target_validity"].to(args.device)
            loss = masked_bce_with_logits(logits, target, valid, valid)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite validation loss")
            total += float(loss)
            metrics.update(logits, target, valid.bool())
    result = {
        "split": args.split,
        "count": len(samples),
        "loss": total / max(len(loader), 1),
        **metrics.compute(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
