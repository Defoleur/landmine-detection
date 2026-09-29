#!/usr/bin/env python3
"""Run endpoint segmentation on one 12-condition point sample."""

import argparse
from pathlib import Path

import torch
import yaml

from phase1b_rgb_convlstm.data import PointConditionDataset, read_temporal_manifest
from phase1b_rgb_convlstm.models import RGBTemporalSegmentationModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("phase1b_rgb_convlstm/config.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    sample = next(x for x in read_temporal_manifest(args.manifest) if x["sequence_id"] == args.sequence_id)
    dataset = PointConditionDataset([sample], config["data"].get("tile_size", 256))
    model = RGBTemporalSegmentationModel(use_condition_present=config["model"].get("use_condition_present", False)).eval()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload), strict=False)
    with torch.inference_mode():
        logits, _ = model(dataset[0]["rgb"].unsqueeze(0), condition_present=dataset[0]["condition_present"].unsqueeze(0))
    mask = (torch.sigmoid(logits[0, 0]) >= 0.5).to(torch.uint8).numpy() * 255
    from PIL import Image
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
