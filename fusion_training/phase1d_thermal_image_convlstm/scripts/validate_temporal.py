#!/usr/bin/env python3
"""Evaluate an image temporal checkpoint on the validation split."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from phase1d_thermal_image_convlstm.sequence import read_manifest
from phase1d_thermal_image_convlstm.temporal_data import ImageSequenceDataset
from phase1d_thermal_image_convlstm.temporal_models import (
    ThermalImageTemporalSegmentationModel,
)
from phase1d_thermal_image_convlstm.scripts.train_temporal import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    seq = [x for x in read_manifest(a.manifest) if x["split"] == "val"]
    loader = DataLoader(
        ImageSequenceDataset(seq, a.tile_size),
        batch_size=1,
        shuffle=False,
        num_workers=a.workers,
        pin_memory=a.device.startswith("cuda"),
    )
    model = ThermalImageTemporalSegmentationModel(use_condition_present=True).to(
        a.device
    )
    payload = torch.load(a.checkpoint, map_location=a.device, weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload), strict=False)
    loss, metrics = evaluate(model, loader, a.device)
    print(json.dumps({"val_loss": loss, **metrics}, indent=2))


if __name__ == "__main__":
    main()
