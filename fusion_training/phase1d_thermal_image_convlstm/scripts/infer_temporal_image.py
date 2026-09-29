#!/usr/bin/env python3
"""Run image ConvLSTM inference on one processed thermal image."""

import argparse
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw
from phase1c_thermal_image_encoder.data import IMAGE_MEAN, IMAGE_STD
from phase1d_thermal_image_convlstm.temporal_models import (
    ThermalImageTemporalSegmentationModel,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--probability-output", type=Path)
    p.add_argument("--repeat", type=int, choices=range(1, 13), default=1)
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if not 0 <= a.threshold <= 1:
        raise SystemExit("--threshold must be between 0 and 1")

    with Image.open(a.image) as source:
        image_array = (
            np.asarray(
                source.convert("RGB").resize(
                    (a.tile_size, a.tile_size), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            ).transpose(2, 0, 1)
            / 255.0
        )
    image = torch.from_numpy(image_array)
    image = (image - torch.tensor(IMAGE_MEAN)[:, None, None]) / torch.tensor(IMAGE_STD)[
        :, None, None
    ]
    sequence = image.unsqueeze(0).repeat(1, a.repeat, 1, 1, 1)
    condition_present = torch.ones((1, a.repeat), dtype=torch.bool)
    model = (
        ThermalImageTemporalSegmentationModel(use_condition_present=True)
        .to(a.device)
        .eval()
    )
    model.load_state_dict(
        torch.load(a.checkpoint, map_location=a.device, weights_only=False).get(
            "model_state_dict", {}
        ),
        strict=False,
    )
    with torch.inference_mode():
        logits, _ = model(
            sequence.to(a.device),
            condition_present.to(a.device),
            freeze_spatial=True,
        )
    probability = torch.sigmoid(logits[0, 0]).cpu().numpy().clip(0, 1)
    prediction = probability >= a.threshold
    probability_output = a.probability_output or a.output.with_name(
        f"{a.output.stem}_probability{a.output.suffix}"
    )
    visualization = a.output.with_name(f"{a.output.stem}_comparison{a.output.suffix}")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    probability_output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((prediction.astype(np.uint8) * 255)).save(a.output)
    probability_rgb = Image.fromarray(
        (colormaps["inferno"](probability)[..., :3] * 255).astype(np.uint8)
    )
    probability_rgb.save(probability_output)

    canvas = Image.new("RGB", (a.tile_size * 3, a.tile_size), "white")
    canvas.paste(
        Image.fromarray((image_array.transpose(1, 2, 0) * 255).astype(np.uint8)), (0, 0)
    )
    canvas.paste(
        probability_rgb.resize((a.tile_size, a.tile_size)),
        (a.tile_size, 0),
    )
    canvas.paste(
        Image.fromarray((prediction.astype(np.uint8) * 255)).convert("RGB"),
        (a.tile_size * 2, 0),
    )
    draw = ImageDraw.Draw(canvas)
    for index, label in enumerate(("input", "probability", "prediction")):
        draw.rectangle(
            (index * a.tile_size, 0, (index + 1) * a.tile_size - 1, 20),
            fill="black",
        )
        draw.text((index * a.tile_size + 3, 3), label, fill="white")
    canvas.save(visualization)
    print(f"wrote mask: {a.output}")
    print(f"wrote probability: {probability_output}")
    print(f"wrote visualization: {visualization}")


if __name__ == "__main__":
    main()
