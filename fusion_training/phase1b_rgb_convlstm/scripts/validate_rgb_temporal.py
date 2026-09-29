#!/usr/bin/env python3
"""Evaluate a temporal checkpoint on the deterministic validation split."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from phase1b_rgb_convlstm.data import PointConditionDataset, read_temporal_manifest
from phase1b_rgb_convlstm.models import RGBTemporalSegmentationModel
from phase1b_rgb_convlstm.scripts.train_rgb_temporal import evaluate, load_phase1a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("phase1b_rgb_convlstm/config.yaml"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    samples = [x for x in read_temporal_manifest(args.manifest) if x["split"] == "val"]
    loader = DataLoader(PointConditionDataset(samples, config["data"].get("tile_size", 256)), batch_size=1)
    model = RGBTemporalSegmentationModel(use_condition_present=config["model"].get("use_condition_present", False)).to(args.device)
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload), strict=False)
    loss, metrics = evaluate(model, loader, args.device)
    print(json.dumps({"val_loss": loss, **metrics}, indent=2))


if __name__ == "__main__":
    main()
