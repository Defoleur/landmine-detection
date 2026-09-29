#!/usr/bin/env python3
"""Infer one manifest sequence and write probability and comparison PNGs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image

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
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    sequence = next(
        item
        for item in read_manifest(args.manifest)
        if item["sequence_id"] == args.sequence_id
    )
    dataset = ImageFusionSequenceDataset([sequence], args.tile_size)
    batch = {
        key: value.unsqueeze(0) if torch.is_tensor(value) else value
        for key, value in dataset[0].items()
    }
    model = ImageFusionSegmenter().to(args.device).eval()
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    load_compatible(model, payload)
    with torch.inference_mode():
        logits, _ = model(
            batch["rgb_sequence"].to(args.device),
            batch["thermal_image"].to(args.device),
            batch["rgb_available"].to(args.device),
            batch["thermal_available"].to(args.device),
            freeze_branches=True,
        )
    probability = torch.sigmoid(logits[0, 0]).cpu().numpy()
    prediction = (probability >= args.threshold).astype(np.uint8) * 255
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        (colormaps["inferno"](probability.clip(0, 1))[..., :3] * 255).astype(np.uint8)
    ).save(
        args.output_dir / "probability.png"
    )
    Image.fromarray(prediction).save(args.output_dir / "prediction.png")
    if Path(sequence["target_path"]).exists():
        with Image.open(sequence["target_path"]) as target:
            target.convert("L").resize(
                (args.tile_size, args.tile_size), Image.Resampling.NEAREST
            ).save(args.output_dir / "target.png")
    print(
        json.dumps(
            {
                "sequence_id": args.sequence_id,
                "threshold": args.threshold,
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
