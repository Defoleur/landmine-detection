#!/usr/bin/env python3
"""Validate one thermal-image U-TAE checkpoint."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from phase1a_rgb_encoder.losses import masked_bce_with_logits
from phase1a_rgb_encoder.metrics import SegmentationMetrics
from phase1b_rgb_utae.time_encoding import ids_to_time_features
from phase1d_thermal_image_convlstm.sequence import read_manifest
from phase1d_thermal_image_convlstm.temporal_data import (ImageSequenceDataset,
    make_virtual_sequences, variable_image_sequence_collate)
from phase1d_thermal_image_utae.temporal_models import ThermalImageUTAE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--virtual-validation-samples-per-sequence", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    samples = make_virtual_sequences(
        [item for item in read_manifest(args.manifest) if item["split"] == "val"],
        args.virtual_validation_samples_per_sequence,
        seed=10000,
    )
    loader = DataLoader(ImageSequenceDataset(samples, args.tile_size), batch_size=1, collate_fn=variable_image_sequence_collate)
    model = ThermalImageUTAE().to(args.device).eval()
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload), strict=False)
    total = 0.0
    metrics = SegmentationMetrics()
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["image"].to(args.device), batch["condition_present"].to(args.device),
                           ids_to_time_features(batch["season_ids"], batch["daytime_ids"]).to(args.device))
            target = batch["target"].to(args.device)
            valid = batch["target_validity"].to(args.device)
            total += float(masked_bce_with_logits(logits, target, valid, valid))
            metrics.update(logits, target, valid.bool(), threshold=args.threshold)
    values = {key: float(value) for key, value in metrics.compute().items()}
    print({"validation_loss": total / max(len(loader), 1), "samples": len(samples), **values})


if __name__ == "__main__":
    main()
