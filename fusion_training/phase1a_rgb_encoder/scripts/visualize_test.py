#!/usr/bin/env python3
"""Save qualitative RGB landmine predictions from the held-out test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from phase1a_rgb_encoder.data import IMAGENET_MEAN, IMAGENET_STD, RGBDataset, read_manifest
from phase1a_rgb_encoder.models import RGBSegmentationModel
from phase1w_valid.visualization_designs import save_designs


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("phase1a_rgb_encoder/artifacts/dataset_rgb_main_manifest.csv"))
    parser.add_argument("--checkpoint", type=Path, required=True, help="Full RGB segmentation checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("phase1a_rgb_encoder/artifacts/test_visualizations"))
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--positive-only", action="store_true", help="Visualize test images containing labels only")
    parser.add_argument("--negative-only", action="store_true", help="Visualize test images with empty masks only")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--design", choices=("all", "adaptive"), default="all")
    return parser.parse_args()


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    return (tensor.cpu() * std + mean).permute(1, 2, 0).clamp(0, 1).numpy()


def overlay(image, mask, color):
    result = image.copy()
    mask = mask.astype(bool)
    result[mask] = 0.55 * result[mask] + 0.45 * np.asarray(color)
    return result


def feature_heatmap(feature):
    activation = feature[0].float().pow(2).mean(dim=0).sqrt().cpu().numpy()
    maximum = float(activation.max())
    return activation / maximum if maximum > 0 else activation


def main():
    args = arguments()
    records = [r for r in read_manifest(args.manifest) if r["split"] == "test"]
    dataset = RGBDataset(records, augment=False, tile_size=args.tile_size)
    if args.positive_only and args.negative_only:
        raise ValueError("Use either --positive-only or --negative-only, not both")
    if args.positive_only or args.negative_only:
        def has_landmine(record):
            if record.get("mask_path"):
                with Image.open(record["mask_path"]) as mask:
                    return bool(np.asarray(mask).max())
            return bool(Path(record["label_path"]).read_text().strip())
        indices = [i for i, record in enumerate(records)
                   if has_landmine(record) == args.positive_only]
    else:
        indices = list(range(len(dataset)))
    indices = indices[:args.num_samples]
    if not indices:
        raise ValueError("No test samples matched the requested selection")

    model = RGBSegmentationModel(pretrained=False).to(args.device)
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload))
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for output_index, dataset_index in enumerate(indices):
            sample = dataset[dataset_index]
            rgb = sample["rgb"].unsqueeze(0).to(args.device)
            valid = sample["valid"].unsqueeze(0).to(args.device)
            logits, encoded = model(rgb, valid, sample["timestamp"])
            probability = torch.sigmoid(logits[0, 0]).cpu().numpy()
            prediction = probability >= args.threshold
            target = sample["target"][0].numpy() > 0.5
            image = denormalize(sample["rgb"])
            target_overlay = overlay(image, target, (1, 0, 0))
            prediction_overlay = overlay(image, prediction, (0, 1, 0))
            error = np.zeros((*target.shape, 3), dtype=np.float32)
            error[target & prediction] = (0.1, 0.8, 0.1)
            error[target & ~prediction] = (1.0, 0.1, 0.1)
            error[~target & prediction] = (1.0, 0.7, 0.0)

            features = [(feature_heatmap(encoded.features[scale]), f"Encoder output {scale}")
                        for scale in ("1/2", "1/4", "1/8")]
            panels = [(image, "RGB"), (target_overlay, "Ground truth (red)"),
                      (probability, "Prediction probability"),
                      (prediction_overlay, f"Prediction >= {args.threshold:g} (green)"),
                      (error, "TP green / FN red / FP orange"),
                      (target.astype(float) - prediction.astype(float), "Prediction error"),
                      *features]
            save_designs(
                [(title, panel) for panel, title in panels],
                args.output_dir,
                "1a",
                f"{output_index:04d}_{sample['acquisition_id']}",
                args.design,
            )
    print(f"Saved {len(indices)} test visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
