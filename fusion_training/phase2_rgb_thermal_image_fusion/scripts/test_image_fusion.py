#!/usr/bin/env python3
"""Evaluate image fusion and save labeled RGB/thermal prediction panels."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from phase1a_rgb_encoder.losses import masked_bce_with_logits
from phase1a_rgb_encoder.metrics import SegmentationMetrics
from phase1d_thermal_convlstm.sequence import CONDITIONS
from phase2_rgb_thermal_image_fusion.data import (
    ImageFusionSequenceDataset,
    RGB_MEAN,
    RGB_STD,
    read_manifest,
)
from phase2_rgb_thermal_image_fusion.models import ImageFusionSegmenter
from phase1w_valid.visualization_designs import save_designs


def load_compatible(model, payload):
    state = payload.get("model_state_dict", payload)
    expected = model.state_dict()
    state = {key.removeprefix("module."): value for key, value in state.items()}
    state = {
        key: value
        for key, value in state.items()
        if key in expected and expected[key].shape == value.shape
    }
    model.load_state_dict(state, strict=False)


def _gray_panel(values, size, mask=False):
    image = Image.fromarray(
        (np.clip(np.asarray(values), 0, 1) * 255).astype(np.uint8)
    ).convert("RGB")
    return image.resize(
        (size, size), Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
    )


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _rgb_panel(values, size):
    values = np.asarray(values).transpose(1, 2, 0)
    values = np.clip(values * np.asarray(RGB_STD) + np.asarray(RGB_MEAN), 0, 1)
    return Image.fromarray((values * 255).astype(np.uint8)).resize(
        (size, size), Image.Resampling.BILINEAR
    )


def _thermal_panel(values, size):
    values = np.asarray(values).transpose(1, 2, 0)
    values = np.clip(values * np.asarray(RGB_STD) + np.asarray(RGB_MEAN), 0, 1)
    return Image.fromarray((values * 255).astype(np.uint8)).resize(
        (size, size), Image.Resampling.BILINEAR
    )


def _overlay_panel(image, probability, threshold, size, color):
    shown = np.asarray(image.resize((size, size)), dtype=np.float32)
    mask = probability >= threshold
    mask = (
        np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (size, size), Image.Resampling.NEAREST
            )
        )
        > 0
    )
    shown[mask] = shown[mask] * 0.55 + np.asarray(color, dtype=np.float32) * 0.45
    return Image.fromarray(np.clip(shown, 0, 255).astype(np.uint8))


def _probability_panel(probability, size):
    values = np.asarray(probability)
    resized = np.asarray(
        Image.fromarray((np.clip(values, 0, 1) * 255).astype(np.uint8)).resize(
            (size, size), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0
    return Image.fromarray((colormaps["inferno"](resized)[..., :3] * 255).astype(np.uint8))


def _mask_panel(mask, size, threshold=0.5):
    values = np.asarray(mask) >= threshold
    return (
        Image.fromarray((values.astype(np.uint8) * 255), mode="L")
        .resize((size, size), Image.Resampling.NEAREST)
        .convert("RGB")
    )


def _error_panel(target, probability, size, threshold):
    target_values = np.asarray(target) > 0.5
    predicted = np.asarray(probability) >= threshold
    error = np.zeros((*target_values.shape, 3), dtype=np.uint8)
    error[target_values & predicted] = (30, 205, 30)
    error[target_values & ~predicted] = (235, 35, 35)
    error[~target_values & predicted] = (245, 150, 20)
    return Image.fromarray(error).resize((size, size), Image.Resampling.NEAREST)


def _feature_panel(feature, size):
    activation = feature[0].detach().float().pow(2).mean(dim=0).sqrt()
    maximum = float(activation.max())
    if maximum > 0:
        activation /= maximum
    return _gray_panel(activation.cpu().numpy(), size)


def _availability_panel(availability, size):
    values = np.asarray(availability, dtype=np.uint8)
    panel = np.repeat(values[None, :], max(size // 8, 1), axis=0)
    return (
        Image.fromarray(panel * 255, mode="L")
        .resize((size, size), Image.Resampling.NEAREST)
        .convert("RGB")
    )


def save_visualization(
    batch,
    logits,
    details,
    output,
    threshold=0.5,
    size=160,
    design="all",
    overlay_condition="summer_day",
):
    rgb, thermal = batch["rgb_sequence"][0].cpu(), batch["thermal_image"][0].cpu()
    target, probability = (
        batch["target"][0, 0].cpu().numpy(),
        torch.sigmoid(logits[0, 0]).cpu().numpy(),
    )
    input_panels, input_labels = [], []
    for index in range(rgb.shape[0]):
        input_panels += [
            _rgb_panel(rgb[index], size),
            _thermal_panel(thermal[index], size),
        ]
        input_labels += [f"RGB {CONDITIONS[index]}", f"Thermal {CONDITIONS[index]}"]

    prediction = probability >= threshold
    output_panels = [
        _mask_panel(target, size),
        _probability_panel(probability, size),
        _mask_panel(probability, size, threshold),
        _error_panel(target, probability, size, threshold),
    ]
    output_labels = [
        "Ground truth",
        "Fusion probability",
        f"Prediction >= {threshold:g}",
        "TP green / FN red / FP orange",
    ]
    overlay_index = CONDITIONS.index(overlay_condition)
    rgb_overlay = _rgb_panel(rgb[overlay_index], size)
    thermal_overlay = _thermal_panel(thermal[overlay_index], size)
    output_panels += [
        _overlay_panel(rgb_overlay, probability, threshold, size, (40, 220, 60)),
        _overlay_panel(thermal_overlay, probability, threshold, size, (40, 220, 60)),
    ]
    output_labels += [
        "RGB із накладеним прогнозом",
        "Теплове зображення із накладеним прогнозом",
    ]
    for scale in ("1/2", "1/4", "1/8"):
        output_panels.append(_feature_panel(details["features"][scale], size))
        output_labels.append(f"Fused features {scale}")
    save_designs(
        list(zip(input_labels, input_panels)),
        output.parent / "designs",
        "2_inputs",
        output.stem,
        design,
        columns=6,
    )
    save_designs(
        list(zip(output_labels, output_panels)),
        output.parent / "designs",
        "2_outputs",
        output.stem,
        design,
        columns=3,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--visualization-dir", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument(
        "--mine-id",
        action="append",
        dest="mine_ids",
        default=[],
        help="Generate visualizations only for this mine ID; repeat for multiple mines.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Limit selected test sequences after applying --mine-id filters.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Choose test sequences randomly instead of taking them in manifest order.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used with --random (default: 42).",
    )
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--overlay-condition",
        choices=CONDITIONS,
        default="summer_day",
        help="Condition used for the RGB and thermal prediction overlays.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--design", choices=("all", "adaptive"), default="all")
    args = parser.parse_args()
    if args.count < 0 or args.tile_size <= 0 or not 0 <= args.threshold <= 1:
        raise SystemExit("invalid count, tile-size, or threshold")
    samples = [item for item in read_manifest(args.manifest) if item["split"] == "test"]
    if args.mine_ids:
        samples = [item for item in samples if item.get("mine_id") in args.mine_ids]
    if args.random:
        randomizer = random.Random(args.seed)
        if args.count:
            samples = randomizer.sample(samples, min(args.count, len(samples)))
        else:
            randomizer.shuffle(samples)
    elif args.count:
        samples = samples[: args.count]
    if not samples:
        raise SystemExit("No test sequences found")
    loader = DataLoader(
        ImageFusionSequenceDataset(samples, args.tile_size), batch_size=1
    )
    model = ImageFusionSegmenter().to(args.device).eval()
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    load_compatible(model, payload)
    metrics = SegmentationMetrics()
    total_loss = 0.0
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            logits, details = model(
                batch["rgb_sequence"].to(args.device),
                batch["thermal_image"].to(args.device),
                batch["rgb_available"].to(args.device),
                batch["thermal_available"].to(args.device),
                freeze_branches=True,
            )
            probability = torch.sigmoid(logits)
            target = batch["target"].to(args.device)
            valid = batch["target_validity"].to(args.device)
            total_loss += float(masked_bce_with_logits(logits, target, valid, valid))
            metrics.update(logits, target, valid.bool(), threshold=args.threshold)
            if args.visualization_dir is not None and (
                not args.mine_ids or samples[index].get("mine_id") in args.mine_ids
            ):
                save_visualization(
                    batch,
                    logits.cpu(),
                    details,
                    args.visualization_dir
                    / f"{index:04d}_{samples[index]['sequence_id']}.png",
                    args.threshold,
                    design=args.design,
                    overlay_condition=args.overlay_condition,
                )
    result = {
        "count": len(samples),
        "test_loss": total_loss / max(len(loader), 1),
        "threshold": args.threshold,
        **metrics.compute(),
    }
    output = args.metrics_output or (
        args.visualization_dir / "test_metrics.json"
        if args.visualization_dir is not None
        else Path("test_metrics.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
