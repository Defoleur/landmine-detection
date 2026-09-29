#!/usr/bin/env python3
"""Run the image encoder on one rendered thermal PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from phase1c_thermal_image_encoder.data import IMAGE_MEAN, IMAGE_STD
from phase1c_thermal_image_encoder.models import ImageSegmentationModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with Image.open(args.image) as source:
        image = source.convert("RGB").resize((args.tile_size, args.tile_size), Image.Resampling.BILINEAR)
    shown = np.asarray(image, dtype=np.float32) / 255.0
    tensor = (shown.transpose(2, 0, 1) - np.asarray(IMAGE_MEAN)[:, None, None]) / np.asarray(IMAGE_STD)[:, None, None]
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = payload.get("config", {})
    model = ImageSegmentationModel(bool(config.get("model", {}).get("pretrained", False))).to(args.device).eval()
    model.load_state_dict(payload.get("model_state_dict", payload), strict=False)
    with torch.inference_mode():
        logits, _ = model(torch.from_numpy(tensor[None].astype(np.float32)).to(args.device))
    probability = torch.sigmoid(logits[0, 0]).cpu().numpy()
    prediction = probability >= args.threshold
    panels = [(shown, "Rendered thermal image"), (probability, "Prediction probability"),
              (prediction, f"Prediction >= {args.threshold:g}")]
    if args.mask:
        with Image.open(args.mask) as mask:
            target = np.asarray(mask.convert("L").resize((args.tile_size, args.tile_size), Image.Resampling.NEAREST)) > 0
        panels.append((target, "Ground truth"))
    figure, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    for axis, (panel, title) in zip(np.atleast_1d(axes), panels):
        axis.imshow(panel, cmap="magma" if panel.ndim == 2 else None, vmin=0 if panel.ndim == 2 else None,
                    vmax=1 if panel.ndim == 2 else None)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
