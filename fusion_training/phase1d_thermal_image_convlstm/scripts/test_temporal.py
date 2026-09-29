#!/usr/bin/env python3
"""Evaluate and visualize an image temporal checkpoint on the test split."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from phase1a_rgb_encoder.losses import masked_bce_with_logits
from phase1a_rgb_encoder.metrics import SegmentationMetrics
from phase1c_thermal_image_encoder.data import IMAGE_MEAN, IMAGE_STD
from phase1d_thermal_image_convlstm.sequence import read_manifest
from phase1d_thermal_image_convlstm.temporal_data import ImageSequenceDataset
from phase1d_thermal_image_convlstm.temporal_models import (
    ThermalImageTemporalSegmentationModel,
)
from phase1w_valid.visualization_designs import save_designs


def denormalize_image(image):
    mean = np.asarray(IMAGE_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(IMAGE_STD, dtype=np.float32)[:, None, None]
    image = np.clip(image.numpy() * std + mean, 0, 1)
    return (image.transpose(1, 2, 0) * 255).astype(np.uint8)


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _image_panel(image, size):
    return Image.fromarray(denormalize_image(image)).resize(
        (size, size), Image.Resampling.BILINEAR
    )


def _mask_panel(mask, size, threshold=None):
    values = mask.detach().cpu().numpy()
    values = values >= threshold if threshold is not None else values > 0.5
    return (
        Image.fromarray((values.astype(np.uint8) * 255), mode="L")
        .resize((size, size), Image.Resampling.NEAREST)
        .convert("RGB")
    )


def _probability_panel(probability, size):
    values = np.clip(probability.detach().cpu().numpy(), 0, 1)
    return (
        Image.fromarray(
            (colormaps["inferno"](values)[..., :3] * 255).astype(np.uint8)
        ).resize((size, size), Image.Resampling.BILINEAR)
    )


def _feature_panel(feature, size):
    activation = feature.detach().float().cpu().pow(2).mean(dim=0).sqrt().numpy()
    maximum = float(activation.max())
    if maximum > 0:
        activation /= maximum
    return (
        Image.fromarray((activation * 255).astype(np.uint8), mode="L")
        .resize((size, size), Image.Resampling.BILINEAR)
        .convert("RGB")
    )


def _error_panel(target, prediction, size):
    target_array = target.detach().cpu().numpy() > 0.5
    predicted_array = prediction.detach().cpu().numpy() >= 0.5
    overlay = np.zeros((*target_array.shape, 3), dtype=np.uint8)
    overlay[target_array & predicted_array] = (0, 190, 0)
    overlay[~target_array & predicted_array] = (220, 0, 0)
    overlay[target_array & ~predicted_array] = (0, 80, 220)
    return Image.fromarray(overlay).resize((size, size), Image.Resampling.NEAREST)


def save_visualization(
    sample, images, probability, target, details, output_path, tile_size, threshold,
    design="all",
):
    thumb = max(int(tile_size * 1.5), 128)
    label_height = 20
    input_panels = [_image_panel(images[index], thumb) for index in range(images.shape[0])]
    input_labels = list(sample["conditions"])
    prediction = probability >= threshold
    output_panels = [
        _mask_panel(target, thumb),
        _probability_panel(probability, thumb),
        _mask_panel(probability, thumb, threshold),
        _error_panel(target, prediction, thumb),
    ]
    output_labels = [
        "ground truth",
        "probability",
        f"prediction >= {threshold:g}",
        "TP green FP red FN blue",
    ]
    for scale in ("1/2", "1/4", "1/8"):
        output_panels.append(_feature_panel(details["features"][scale][0], thumb))
        output_labels.append(f"LSTM {scale}")

    save_designs(
        list(zip(input_labels, input_panels)), output_path.parent / "designs", "1d_image_inputs",
        output_path.stem, design, columns=3,
    )
    save_designs(
        list(zip(output_labels, output_panels)), output_path.parent / "designs", "1d_image_outputs",
        output_path.stem, design, columns=4,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--visualization-dir", type=Path)
    p.add_argument("--metrics-output", type=Path)
    p.add_argument("--count", type=int, default=0)
    p.add_argument(
        "--mine-id", "--landmine", action="append", dest="mine_ids",
        help="Only evaluate these exact landmine IDs; repeat for multiple IDs.",
    )
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--design", choices=("all", "adaptive"), default="all")
    a = p.parse_args()
    if a.count < 0 or not 0 <= a.threshold <= 1:
        raise SystemExit(
            "--count must be non-negative and --threshold must be between 0 and 1"
        )
    seq = [x for x in read_manifest(a.manifest) if x["split"] == "test"]
    if a.mine_ids:
        requested_mines = {
            value.casefold().removeprefix("obstacle_mine_") for value in a.mine_ids
        }
        seq = [
            item
            for item in seq
            if str(item.get("mine_id", "")).casefold().removeprefix("obstacle_mine_")
            in requested_mines
        ]
    seq = seq[: a.count] if a.count else seq
    loader = DataLoader(ImageSequenceDataset(seq, a.tile_size), batch_size=1)
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
    metrics = SegmentationMetrics()
    loss = 0.0
    with torch.inference_mode():
        for i, b in enumerate(loader):
            logits, details = model(
                b["image"].to(a.device),
                b["condition_present"].to(a.device),
                freeze_spatial=True,
            )
            target = b["target"].to(a.device)
            valid = b["target_validity"].to(a.device)
            loss += float(masked_bce_with_logits(logits, target, valid, valid))
            metrics.update(logits, target, valid.bool(), threshold=a.threshold)
            if a.visualization_dir:
                details_cpu = {
                    "features": {
                        key: value.cpu() for key, value in details["features"].items()
                    }
                }
                save_visualization(
                    seq[i],
                    b["image"][0].cpu(),
                    torch.sigmoid(logits[0, 0]).cpu(),
                    target[0, 0].cpu(),
                    details_cpu,
                    a.visualization_dir / f"{i:04d}_{seq[i]['sequence_id']}.png",
                    a.tile_size,
                    a.threshold,
                    a.design,
                )
    result = {
        "sample_count": len(seq),
        "test_loss": loss / max(len(loader), 1),
        **metrics.compute(),
    }
    metrics_output = a.metrics_output
    if metrics_output is None and a.visualization_dir:
        metrics_output = a.visualization_dir / "test_metrics.json"
    if metrics_output:
        metrics_output.parent.mkdir(parents=True, exist_ok=True)
        metrics_output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
