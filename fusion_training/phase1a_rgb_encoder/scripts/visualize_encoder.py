#!/usr/bin/env python3
"""Visualize encoder activations for one RGB image without ground truth."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from phase1a_rgb_encoder.data import IMAGENET_MEAN, IMAGENET_STD
from phase1a_rgb_encoder.models import RGBEncoder


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to an RGB image")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Exported encoder_rgb.pt checkpoint")
    parser.add_argument("--output", type=Path, help="Output image path; defaults to a filename based on the input image")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--crop-square", action="store_true", help="Center-crop the image to a square before resizing")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_image(path: Path, tile_size: int, crop_square: bool = False):
    with Image.open(path) as source:
        image = source.convert("RGB")
        if crop_square:
            width, height = image.size
            side = min(width, height)
            left = (width - side) // 2
            top = (height - side) // 2
            image = image.crop((left, top, left + side, top + side))
        image = image.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    return array, ((tensor - mean) / std).unsqueeze(0)


def activation_map(feature):
    # Mean absolute activation shows where the feature bank responds spatially.
    return feature.detach().abs().mean(dim=1)[0].cpu().numpy()


def main():
    args = arguments()
    image, tensor = load_image(args.image, args.tile_size, args.crop_square)
    valid = torch.ones((1, 1, args.tile_size, args.tile_size), device=args.device)
    encoder = RGBEncoder.from_pretrained(args.checkpoint, map_location=args.device).to(args.device).eval()
    with torch.inference_mode():
        encoder_output = encoder(tensor.to(args.device), valid)

    output_path = args.output or Path("phase1a_rgb_encoder/artifacts/encoder_visualizations") / f"{args.image.stem}_encoder_visualization.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [(image, "RGB input", None)]
    for scale in RGBEncoder.scales:
        panels.append((activation_map(encoder_output.features[scale]), f"Mean |activation| ({scale})", "magma"))
    for scale in RGBEncoder.scales:
        panels.append((encoder_output.valid[scale][0, 0].cpu().numpy(), f"Validity ({scale})", "gray"))

    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, (panel, title, cmap) in zip(axes.flat, panels):
        axis.imshow(panel, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(f"Encoder activations: {args.image.name}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved encoder visualization to {output_path}")


if __name__ == "__main__":
    main()
