#!/usr/bin/env python3
"""Export the reusable RGBTemporalEncoder without optimizer state."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from phase1b_rgb_convlstm.models import RGBTemporalSegmentationModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("phase1b_rgb_convlstm/config.yaml"))
    parser.add_argument("--source-encoder-checkpoint", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    model = RGBTemporalSegmentationModel(use_condition_present=config["model"].get("use_condition_present", False)).eval()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload), strict=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.temporal.state_dict(), "config": config["model"],
                "source_encoder_checkpoint": str(args.source_encoder_checkpoint) if args.source_encoder_checkpoint else ""}, args.output)
    (args.output.with_suffix(".json")).write_text(json.dumps({"architecture": "RGBTemporalEncoder", "config": config["model"],
                                                               "source_encoder_checkpoint": str(args.source_encoder_checkpoint) if args.source_encoder_checkpoint else ""}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
