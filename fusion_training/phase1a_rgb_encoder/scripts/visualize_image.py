#!/usr/bin/env python3
"""Visualize segmentation prediction for one RGB image."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from phase1a_rgb_encoder.data import IMAGENET_MEAN, IMAGENET_STD, polygons_to_mask
from phase1a_rgb_encoder.models import RGBSegmentationModel


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Full RGB segmentation checkpoint; or use --run-dir with --epoch")
    parser.add_argument("--run-dir", type=Path, help="Run directory containing epochs/epoch_NNNN/checkpoint.pt")
    parser.add_argument("--epoch", type=int, help="Epoch number to select when using --run-dir")
    parser.add_argument("--label", type=Path, help="Optional YOLO polygon label file")
    parser.add_argument("--mask", type=Path, help="Optional binary or palette ground-truth mask PNG")
    parser.add_argument("--output", type=Path, help="Output image path; defaults to a filename based on the input image")
    parser.add_argument("--postfix", default="_visualization", help="Suffix applied to the input stem for default output names")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--crop-square", action="store_true", help="Center-crop the image to a square before resizing")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_image(path, tile_size, crop_square=False):
    with Image.open(path) as source:
        original = source.convert("RGB")
        source_size = original.size
        if crop_square:
            width, height = original.size
            side = min(width, height)
            left = (width - side) // 2
            top = (height - side) // 2
            crop_box = (left, top, left + side, top + side)
            original = original.crop(crop_box)
        else:
            crop_box = (0, 0, original.width, original.height)
        resized = original.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    return array, ((tensor - mean) / std).unsqueeze(0), crop_box, source_size


def overlay(image, mask, color):
    result = image.copy()
    mask = mask.astype(bool)
    result[mask] = 0.55 * result[mask] + 0.45 * np.asarray(color)
    return result


def main():
    args = arguments()
    if not args.checkpoint:
        if not (args.run_dir and args.epoch is not None):
            raise SystemExit("--checkpoint is required unless --run-dir and --epoch are given")
        args.checkpoint = args.run_dir / f"epochs/epoch_{args.epoch:04d}" / "checkpoint.pt"
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    output = args.output
    if output is None:
        if args.run_dir:
            output = (args.run_dir / f"epochs/epoch_{args.epoch:04d}" / "artifacts"
                      / f"{args.image.stem}{args.postfix}.png")
        else:
            output = Path("phase1a_rgb_encoder/artifacts/image_visualizations") / f"{args.image.stem}{args.postfix}.png"

    image, tensor, crop_box, source_size = load_image(args.image, args.tile_size, args.crop_square)
    valid = torch.ones((1, 1, args.tile_size, args.tile_size), device=args.device)
    model = RGBSegmentationModel(pretrained=False).to(args.device).eval()
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload))
    with torch.inference_mode():
        logits, _ = model(tensor.to(args.device), valid)
    probability = torch.sigmoid(logits[0, 0]).cpu().numpy()
    prediction = probability >= args.threshold

    target = None
    if args.label and args.mask:
        raise ValueError("Use either --label or --mask, not both")
    if args.label or args.mask:
        if args.mask:
            with Image.open(args.mask) as mask_image:
                target = np.asarray(mask_image) > 0
            if target.shape != (source_size[1], source_size[0]):
                raise ValueError(f"Mask shape {target.shape} does not match image {(source_size[1], source_size[0])}")
            target = Image.fromarray(target.astype(np.uint8) * 255)
        else:
            target = Image.fromarray((polygons_to_mask(args.label, source_size) * 255).astype(np.uint8))
        target = target.crop(crop_box)
        target = np.asarray(target.resize(
            (args.tile_size, args.tile_size), Image.Resampling.NEAREST)) > 127

    if target is None:
        figure, axes = plt.subplots(1, 3, figsize=(15, 5))
        panels = [(image, "RGB input"), (probability, "Prediction probability"),
                  (overlay(image, prediction, (0, 1, 0)), f"Prediction >= {args.threshold:g} (green)")]
    else:
        error = np.zeros((*target.shape, 3), dtype=np.float32)
        error[target & prediction] = (0.1, 0.8, 0.1)
        error[target & ~prediction] = (1.0, 0.1, 0.1)
        error[~target & prediction] = (1.0, 0.7, 0.0)
        figure, axes = plt.subplots(2, 3, figsize=(15, 9))
        panels = [(image, "RGB input"), (overlay(image, target, (1, 0, 0)), "Ground truth (red)"),
                  (probability, "Prediction probability"),
                  (overlay(image, prediction, (0, 1, 0)), f"Prediction >= {args.threshold:g} (green)"),
                  (error, "TP green / FN red / FP orange"),
                  (target.astype(float) - prediction.astype(float), "Prediction error")]
    for axis, (panel, title) in zip(np.asarray(axes).flat, panels):
        axis.imshow(panel, cmap="inferno" if title == "Prediction probability" else None,
                    vmin=0 if title == "Prediction probability" else None,
                    vmax=1 if title == "Prediction probability" else None)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(args.image.name)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(f"Saved image visualization to {output}")


if __name__ == "__main__":
    main()
