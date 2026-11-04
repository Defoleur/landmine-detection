#!/usr/bin/env python3
"""Run one asynchronous U-TAE fusion prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from phase2_rgb_thermal_image_fusion.data import ImageFusionSequenceDataset, read_manifest
from phase2_rgb_thermal_image_utae.models import AsyncUTAESegmenter
from phase2_rgb_thermal_image_utae.manifest import id_time_features, legacy_modality_ages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    sample = next(item for item in read_manifest(args.manifest) if item["sequence_id"] == args.sequence_id)
    item = ImageFusionSequenceDataset([sample], args.tile_size)[0]
    model = AsyncUTAESegmenter().to(args.device).eval()
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload), strict=False)
    with torch.inference_mode():
        rgb_ages, thermal_ages = legacy_modality_ages(
            item["season_ids"].unsqueeze(0), item["daytime_ids"].unsqueeze(0),
            item["rgb_available"].unsqueeze(0), item["thermal_available"].unsqueeze(0),
        )
        logits = model(
            item["rgb_sequence"].unsqueeze(0).to(args.device),
            item["thermal_image"].unsqueeze(0).to(args.device),
            item["rgb_available"].unsqueeze(0).to(args.device),
            item["thermal_available"].unsqueeze(0).to(args.device),
            rgb_time_features=id_time_features(item["season_ids"].unsqueeze(0), item["daytime_ids"].unsqueeze(0)).to(args.device),
            thermal_time_features=id_time_features(item["season_ids"].unsqueeze(0), item["daytime_ids"].unsqueeze(0)).to(args.device),
            rgb_ages=rgb_ages.to(args.device), thermal_ages=thermal_ages.to(args.device),
        )
    prediction = (torch.sigmoid(logits[0, 0]) >= args.threshold).to(torch.uint8).cpu().numpy() * 255
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prediction).save(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
