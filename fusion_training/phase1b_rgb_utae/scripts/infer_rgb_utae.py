#!/usr/bin/env python3
"""Run RGB U-TAE inference for one sequence."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from phase1a_rgb_encoder.models import RGBEncoder, RGBSegmentationDecoder
from phase1b_rgb_convlstm.data import PointConditionDataset, read_temporal_manifest
from phase1b_rgb_utae.models import RGBUTAE
from phase1b_rgb_utae.time_encoding import ids_to_time_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    sample = next(item for item in read_temporal_manifest(args.manifest) if item["sequence_id"] == args.sequence_id)
    item = PointConditionDataset([sample], args.tile_size)[0]
    model = RGBUTAE(RGBEncoder(), RGBSegmentationDecoder()).eval()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload), strict=False)
    with torch.inference_mode():
        logits = model(item["rgb"].unsqueeze(0), item["condition_present"].unsqueeze(0),
                       ids_to_time_features(item["season_ids"].unsqueeze(0), item["daytime_ids"].unsqueeze(0)))
    mask = (torch.sigmoid(logits[0, 0]) >= args.threshold).to(torch.uint8).numpy() * 255
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(args.output)


if __name__ == "__main__":
    main()
