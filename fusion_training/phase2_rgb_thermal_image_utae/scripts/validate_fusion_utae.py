#!/usr/bin/env python3
"""Validate an asynchronous RGB/thermal U-TAE checkpoint."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from phase1a_rgb_encoder.losses import masked_bce_with_logits
from phase1a_rgb_encoder.metrics import SegmentationMetrics
from phase2_rgb_thermal_image_fusion.data import (ImageFusionSequenceDataset,
    make_virtual_sequences, read_manifest, variable_sequence_collate)
from phase2_rgb_thermal_image_utae.manifest import id_time_features, legacy_modality_ages
from phase2_rgb_thermal_image_utae.models import AsyncUTAESegmenter


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
    loader = DataLoader(ImageFusionSequenceDataset(samples, args.tile_size), batch_size=1,
                        collate_fn=variable_sequence_collate)
    model = AsyncUTAESegmenter().to(args.device).eval()
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload), strict=False)
    total = 0.0
    metrics = SegmentationMetrics()
    with torch.inference_mode():
        for batch in loader:
            time_features = id_time_features(batch["season_ids"], batch["daytime_ids"]).to(args.device)
            rgb_ages, thermal_ages = legacy_modality_ages(
                batch["season_ids"], batch["daytime_ids"], batch["rgb_available"], batch["thermal_available"]
            )
            logits = model(
                batch["rgb_sequence"].to(args.device), batch["thermal_image"].to(args.device),
                batch["rgb_available"].to(args.device), batch["thermal_available"].to(args.device),
                rgb_time_features=time_features, thermal_time_features=time_features,
                rgb_ages=rgb_ages.to(args.device), thermal_ages=thermal_ages.to(args.device),
            )
            target = batch["target"].to(args.device)
            valid = batch["target_validity"].to(args.device)
            total += float(masked_bce_with_logits(logits, target, valid, valid))
            metrics.update(logits, target, valid.bool(), threshold=args.threshold)
    values = {key: float(value) for key, value in metrics.compute().items()}
    print({"validation_loss": total / max(len(loader), 1), "samples": len(samples), **values})


if __name__ == "__main__":
    main()
