#!/usr/bin/env python3
"""Export the reusable image encoder without the decoder or optimizer."""

import argparse
import json
from pathlib import Path

import torch

from phase1c_thermal_image_encoder.models import ImageSegmentationModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    model = ImageSegmentationModel(bool(config.get("model", {}).get("pretrained", False))).eval()
    model.load_state_dict(payload.get("model_state_dict", payload), strict=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"architecture": "ImageEncoder", "input_channels": 3,
                "scales": ["1/2", "1/4", "1/8"], "channels": [32, 64, 128]}
    torch.save({"model_state_dict": model.encoder.state_dict(), "config": config,
                "feature_contract": metadata}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
