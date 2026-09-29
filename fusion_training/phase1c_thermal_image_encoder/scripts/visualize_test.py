#!/usr/bin/env python3
"""Save qualitative image predictions from the held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from phase1c_thermal_image_encoder.data import IMAGE_MEAN, IMAGE_STD, ImageSegmentationDataset, read_manifest
from phase1c_thermal_image_encoder.models import ImageSegmentationModel
from phase1w_valid.visualization_designs import save_designs


def feature_heatmap(feature):
    activation = feature[0].float().pow(2).mean(dim=0).sqrt().cpu().numpy()
    maximum = float(activation.max())
    return activation / maximum if maximum > 0 else activation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("phase1c_thermal_image_encoder/artifacts/image_manifest.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("phase1c_thermal_image_encoder/artifacts/test_visualizations"))
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--positive-only", action="store_true")
    parser.add_argument("--negative-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--design", choices=("all", "adaptive"), default="all")
    args = parser.parse_args()
    if args.positive_only and args.negative_only:
        raise SystemExit("Use either --positive-only or --negative-only")
    records = [r for r in read_manifest(args.manifest) if r["split"] == "test"]
    if args.positive_only or args.negative_only:
        records = [r for r in records if bool(r["positive"]) == args.positive_only]
    records = records[:args.num_samples]
    if not records:
        raise SystemExit("No test samples matched the requested selection")
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = payload.get("config", {})
    tile_size = int(config.get("data", {}).get("tile_size", 640))
    dataset = ImageSegmentationDataset(records, tile_size)
    model = ImageSegmentationModel(bool(config.get("model", {}).get("pretrained", False))).to(args.device).eval()
    model.load_state_dict(payload.get("model_state_dict", payload), strict=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mean = np.asarray(IMAGE_MEAN)[:, None, None]
    std = np.asarray(IMAGE_STD)[:, None, None]
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            logits, encoded = model(sample["image"].unsqueeze(0).to(args.device))
            probability = torch.sigmoid(logits[0, 0]).cpu().numpy()
            prediction = probability >= args.threshold
            image = np.clip(sample["image"].numpy() * std + mean, 0, 1).transpose(1, 2, 0)
            target = sample["target"][0].numpy() > 0.5
            error = np.zeros((*target.shape, 3), dtype=np.float32)
            error[target & prediction] = (0.1, 0.8, 0.1)
            error[target & ~prediction] = (1.0, 0.1, 0.1)
            error[~target & prediction] = (1.0, 0.7, 0.0)
            features = [(feature_heatmap(encoded.features[scale]), f"Encoder output {scale}")
                        for scale in ("1/2", "1/4", "1/8")]
            panels = [(image, "Thermal image"), (target, "Ground truth"),
                      (probability, "Prediction probability"), (prediction, f"Prediction >= {args.threshold:g}"),
                      (error, "TP green / FN red / FP orange"),
                      (target.astype(float) - prediction.astype(float), "Prediction error"),
                      *features]
            save_designs(
                [(title, panel) for panel, title in panels],
                args.output_dir,
                "1c_image",
                f"{index:04d}_{sample['metadata']['sample_id']}",
                args.design,
            )
    print(f"Saved {len(records)} image test visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
