#!/usr/bin/env python3
"""Run three-channel thermal-image U-TAE inference for one sequence."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from phase1b_rgb_utae.time_encoding import ids_to_time_features
from phase1d_thermal_image_convlstm.sequence import read_manifest
from phase1d_thermal_image_convlstm.temporal_data import ImageSequenceDataset
from phase1d_thermal_image_utae.temporal_models import ThermalImageUTAE


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
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    sample = next(item for item in read_manifest(args.manifest) if item["sequence_id"] == args.sequence_id)
    item = ImageSequenceDataset([sample], args.tile_size)[0]
    model = ThermalImageUTAE().to(args.device).eval()
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload), strict=False)
    with torch.inference_mode():
        logits = model(item["image"].unsqueeze(0).to(args.device),
                       item["condition_present"].unsqueeze(0).to(args.device),
                       ids_to_time_features(item["season_ids"].unsqueeze(0), item["daytime_ids"].unsqueeze(0)).to(args.device))
    mask = (torch.sigmoid(logits[0, 0]) >= args.threshold).to(torch.uint8).numpy() * 255
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(args.output)


if __name__ == "__main__":
    main()
