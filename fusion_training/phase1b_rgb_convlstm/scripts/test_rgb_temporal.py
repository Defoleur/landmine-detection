#!/usr/bin/env python3
"""Evaluate and visualize a temporal checkpoint on the held-out test split."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from phase1b_rgb_convlstm.data import (CachedPointConditionDataset, PointConditionDataset,
                                       attach_target_mask_paths, read_temporal_manifest)
from phase1a_rgb_encoder.models import RGBSegmentationModel
from phase1a_rgb_encoder.scripts.visualize_image import load_image as load_phase1a_image
from phase1b_rgb_convlstm.models import CachedRGBTemporalSegmentationModel, RGBTemporalSegmentationModel
from phase1b_rgb_convlstm.scripts.train_rgb_temporal import evaluate
from phase1w_valid.visualization_designs import save_designs


def sample_mine_type(sample):
    """Read type from old manifests when mine_type is not embedded."""
    if sample.get("mine_type"):
        return str(sample["mine_type"])
    image_path = Path(sample["image_paths"][0])
    metadata_path = image_path.parent.parent / "metadata" / f"{image_path.stem}.json"
    if not metadata_path.exists():
        return ""
    metadata = json.loads(metadata_path.read_text())
    mine_id = sample.get("mine_id", metadata.get("mine", ""))
    for item in metadata.get("visible_mines") or metadata.get("mine_masks") or []:
        if item.get("sim_name") == mine_id and item.get("mine_type"):
            return str(item["mine_type"])
    name = str(mine_id).removeprefix("obstacle_mine_")
    return f"Mine_{name.rsplit('_', 1)[0].upper()}" if "_" in name else f"Mine_{name.upper()}"


def _image_for_sample(sample, tile_size):
    target_index = sample["conditions"].index(sample["target_condition"])
    with Image.open(sample["image_paths"][target_index]) as image:
        return image.convert("RGB").resize((tile_size, tile_size), Image.Resampling.BILINEAR)


def _mask_image(mask, tile_size, threshold=0.5):
    array = (mask.detach().cpu().squeeze().numpy() >= threshold).astype(np.uint8) * 255
    return Image.fromarray(array).resize((tile_size, tile_size), Image.Resampling.NEAREST).convert("L")


def _error_overlay(target, prediction):
    target_array = np.asarray(target) > 0
    predicted_array = np.asarray(prediction) > 0
    overlay = np.zeros((*target_array.shape, 3), dtype=np.uint8)
    overlay[target_array & predicted_array] = (0, 190, 0)
    overlay[~target_array & predicted_array] = (220, 0, 0)
    overlay[target_array & ~predicted_array] = (0, 80, 220)
    return Image.fromarray(overlay)


def _probability_heatmap(probability, size):
    resized = np.asarray(Image.fromarray((np.clip(probability, 0, 1) * 255).astype(np.uint8)).resize(
        size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    rgb = (colormaps["inferno"](resized)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def _feature_heatmap(feature, size):
    activation = feature.detach().float().cpu().pow(2).mean(dim=0).sqrt().numpy()
    maximum = float(activation.max())
    if maximum > 0:
        activation /= maximum
    return Image.fromarray((activation * 255).astype(np.uint8), mode="L").resize(
        size, Image.Resampling.BILINEAR
    ).convert("RGB")


def _prediction_overlay(image, probability, size, threshold=0.5, color=(0, 255, 0)):
    shown = np.asarray(image.convert("RGB").resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
    mask = np.asarray(Image.fromarray((np.clip(probability, 0, 1) * 255).astype(np.uint8)).resize(
        size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0 >= threshold
    shown[mask] = 0.55 * shown[mask] + 0.45 * np.array(color, dtype=np.float32)
    return Image.fromarray(shown.astype(np.uint8))


def save_lstm_encoder_visualization(sample, features, output_path, tile_size, scale=2.0, design="all"):
    """Visualize the temporal encoder feature magnitude before decoding."""
    thumb = max(int(round(tile_size * scale)), 64)
    font_size = max(12, thumb // 32)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    label_height = font_size + 6
    panels = []
    for index, scale_name in enumerate(("1/2", "1/4", "1/8")):
        panel = _feature_heatmap(features[scale_name][0], (thumb, thumb))
        panels.append((f"LSTM output {scale_name}", panel))
    target_image_path = sample.get("target_image_path")
    if target_image_path:
        with Image.open(target_image_path) as target_image:
            target_rgb = target_image.convert("RGB").resize((tile_size, tile_size), Image.Resampling.BILINEAR)
    else:
        target_rgb = _image_for_sample(sample, tile_size)
    target_rgb = target_rgb.resize((thumb, thumb), Image.Resampling.BILINEAR)
    panels.append(("target RGB", target_rgb))
    save_designs(panels, output_path.parent.parent / "designs", "1b", output_path.stem, design)


def save_visualization(sample, prediction, target, output_path, tile_size, scale=2.0,
                       phase1a_prediction=None, lstm_threshold=0.5, phase1a_threshold=0.5,
                       design="all", features=None):
    thumb = max(int(round(tile_size * scale)), 64)
    font_size = max(12, thumb // 32)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    label_height = font_size + 6
    comparison = phase1a_prediction is not None
    target_mask = _mask_image(target, tile_size).resize((thumb, thumb), Image.Resampling.NEAREST)
    predicted_mask = _mask_image(prediction, tile_size, lstm_threshold).resize((thumb, thumb), Image.Resampling.NEAREST)
    target_array = np.asarray(target_mask) > 0
    predicted_array = np.asarray(predicted_mask) > 0
    probability = prediction.detach().cpu().squeeze().numpy()
    panels = [_probability_heatmap(probability, (thumb, thumb)), target_mask.convert("RGB"), predicted_mask.convert("RGB"),
              _error_overlay(target_array, predicted_array)]
    labels = ["probability", "ground truth", f"prediction >= {lstm_threshold:g}", "TP green FP red FN blue"]
    if features is not None:
        for scale_name in ("1/2", "1/4", "1/8"):
            panels.append(_feature_heatmap(features[scale_name][0], (thumb, thumb)))
            labels.append(f"LSTM {scale_name}")
    if comparison:
        baseline_mask = _mask_image(phase1a_prediction, tile_size, phase1a_threshold).resize(
            (thumb, thumb), Image.Resampling.NEAREST)
        panels.extend([baseline_mask.convert("RGB"), _error_overlay(target_array, np.asarray(baseline_mask) > 0)])
        labels.extend(["baseline prediction", "baseline TP green FP red FN blue"])
    input_panels = []
    for index, (condition, image_path) in enumerate(zip(sample["conditions"], sample["image_paths"])):
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize((thumb, thumb), Image.Resampling.BILINEAR)
        input_panels.append((condition, image))
    save_designs(
        input_panels,
        output_path.parent.parent / "designs",
        "1b_inputs",
        output_path.stem,
        design,
        columns=3,
    )
    save_designs(
        list(zip(labels, panels)),
        output_path.parent.parent / "designs",
        "1b_outputs",
        output_path.stem,
        design,
        columns=4,
    )


def save_phase1a_visualization(sample, predictions, target, output_path, tile_size, scale=2.0,
                               phase1a_threshold=0.5, design="all"):
    """Save a separate single-image baseline mosaic for all 12 conditions."""
    thumb = max(int(round(tile_size * scale)), 64)
    font_size = max(12, thumb // 32)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    label_height = font_size + 6
    condition_panels = []
    for index, condition in enumerate(sample["conditions"]):
        with Image.open(sample["image_paths"][index]) as image:
            shown = _prediction_overlay(image, predictions[index], (thumb, thumb), threshold=phase1a_threshold)
        condition_panels.append((f"{condition} >= {phase1a_threshold:g} (green)", shown))

    target_image_path = sample.get("target_image_path")
    if target_image_path:
        with Image.open(target_image_path) as target_image:
            target_rgb = target_image.convert("RGB").resize((tile_size, tile_size), Image.Resampling.BILINEAR)
    else:
        target_rgb = _image_for_sample(sample, tile_size)
    target_rgb = target_rgb.resize((thumb, thumb), Image.Resampling.BILINEAR)
    target_mask = _mask_image(target, tile_size).resize((thumb, thumb), Image.Resampling.NEAREST)
    target_index = sample["conditions"].index(sample["target_condition"])
    target_prediction = _prediction_overlay(target_rgb, predictions[target_index], (thumb, thumb),
                                            threshold=phase1a_threshold)
    target_prediction_mask = np.asarray(Image.fromarray((predictions[target_index] * 255).astype(np.uint8)).resize(
        (thumb, thumb), Image.Resampling.BILINEAR)) >= int(phase1a_threshold * 255)
    panels = (target_rgb, _prediction_overlay(target_rgb, np.asarray(target_mask) / 255.0,
                                               (thumb, thumb), color=(255, 0, 0)), target_prediction,
              _error_overlay(np.asarray(target_mask) > 0, target_prediction_mask))
    labels = ("RGB", "ground truth (red)",
              f"prediction >= {phase1a_threshold:g} (green)",
              "TP green FP red FN blue")
    save_designs(
        condition_panels,
        output_path.parent.parent / "designs",
        "1a_inputs",
        output_path.stem,
        design,
        columns=3,
    )
    save_designs(
        list(zip(labels, panels)),
        output_path.parent.parent / "designs",
        "1a_outputs",
        output_path.stem,
        design,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase1a-checkpoint", type=Path,
                        help="Optional Phase 1A RGB encoder+decoder checkpoint for side-by-side comparison")
    parser.add_argument("--config", type=Path, default=Path("phase1b_rgb_convlstm/config.yaml"))
    parser.add_argument("--feature-source", choices=("online", "cache"), default="online")
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--target-mask-dataset", type=Path,
                        help="Override targets with full-scene masks matched by mine and point number")
    parser.add_argument("--count", type=int, default=0, help="Number of test samples to evaluate; 0 means all")
    parser.add_argument("--mine-type", nargs="+", help="Only evaluate these landmine types, e.g. Mine_PMN4 Mine_PMD6")
    parser.add_argument(
        "--mine-id", "--landmine", action="append", dest="mine_ids",
        help="Only evaluate these exact landmine IDs; repeat for multiple IDs.",
    )
    parser.add_argument("--visualization-dir", type=Path, help="Directory for test prediction montages")
    parser.add_argument("--visualization-scale", type=float, default=2.0,
                        help="Display scale relative to tile size; default: 2 (1024x1024 montage)")
    parser.add_argument("--lstm-threshold", type=float, default=0.5,
                        help="Threshold for Phase 1B masks and metrics")
    parser.add_argument("--phase1a-threshold", type=float, default=0.5,
                        help="Threshold for Phase 1A masks and metrics")
    parser.add_argument("--metrics-output", type=Path, help="Path for JSON metrics output")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--design", choices=("all", "adaptive"), default="all")
    args = parser.parse_args()
    if args.count < 0:
        parser.error("--count must be non-negative")
    if args.visualization_scale <= 0:
        parser.error("--visualization-scale must be positive")
    if not 0 <= args.lstm_threshold <= 1 or not 0 <= args.phase1a_threshold <= 1:
        parser.error("--lstm-threshold and --phase1a-threshold must be between 0 and 1")
    cached = args.feature_source == "cache"
    if cached and not args.cache_manifest:
        parser.error("--cache-manifest is required with --feature-source cache")
    config = yaml.safe_load(args.config.read_text())
    samples = [x for x in read_temporal_manifest(args.manifest) if x["split"] == "test"]
    if args.target_mask_dataset:
        samples = attach_target_mask_paths(samples, args.target_mask_dataset)
    requested_types = {value.casefold() for value in (args.mine_type or [])}
    if requested_types:
        samples = [sample for sample in samples if sample_mine_type(sample).casefold() in requested_types]
    if args.mine_ids:
        requested_mines = {
            value.casefold().removeprefix("obstacle_mine_") for value in args.mine_ids
        }
        samples = [
            sample
            for sample in samples
            if str(sample.get("mine_id", "")).casefold().removeprefix("obstacle_mine_")
            in requested_mines
        ]
    if args.count:
        samples = samples[:args.count]
    if not samples:
        requested = ", ".join(args.mine_type or args.mine_ids or []) or "all types"
        raise SystemExit(f"No test samples matched {requested}")
    tile_size = config["data"].get("tile_size", 256)
    dataset_type = CachedPointConditionDataset if cached else PointConditionDataset
    dataset = dataset_type(samples, tile_size)
    loader = DataLoader(dataset, batch_size=1)
    model_type = CachedRGBTemporalSegmentationModel if cached else RGBTemporalSegmentationModel
    model = model_type(use_condition_present=config["model"].get("use_condition_present", False),
                       temporal_mean=config["model"].get("temporal_mean", False)).to(args.device)
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload), strict=False)
    compare = args.phase1a_checkpoint is not None
    if compare:
        phase1a_model = RGBSegmentationModel(pretrained=False).to(args.device).eval()
        phase1a_payload = torch.load(args.phase1a_checkpoint, map_location=args.device, weights_only=False)
        phase1a_model.load_state_dict(phase1a_payload.get("model_state_dict", phase1a_payload), strict=False)
    metrics = None
    if args.visualization_dir or compare:
        from phase1a_rgb_encoder.metrics import SegmentationMetrics
        from phase1a_rgb_encoder.losses import masked_bce_with_logits
        metrics_accumulator = SegmentationMetrics()
        phase1a_metrics_accumulator = SegmentationMetrics() if compare else None
        total_loss = 0.0
        phase1a_total_loss = 0.0
        model.eval()
        phase1b_visualization_dir = args.visualization_dir / "phase1b" if args.visualization_dir else None
        phase1a_visualization_dir = args.visualization_dir / "phase1a" if args.visualization_dir and compare else None
        lstm_visualization_dir = args.visualization_dir / "lstm_encoder" if args.visualization_dir else None
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                if cached:
                    features = {scale: value.to(args.device) for scale, value in batch["features"].items()}
                    logits, temporal_details = model(features, condition_present=batch["condition_present"].to(args.device))
                else:
                    logits, temporal_details = model(
                        batch["rgb"].to(args.device),
                        condition_present=batch["condition_present"].to(args.device), freeze_spatial=True)
                target = batch["target"].to(args.device)
                valid = batch["target_validity"].to(args.device)
                total_loss += float(masked_bce_with_logits(logits, target, valid, valid))
                metrics_accumulator.update(logits, target, valid.bool(), threshold=args.lstm_threshold)
                phase1a_logits = None
                phase1a_predictions = []
                if compare:
                    target_index = samples[index]["conditions"].index(samples[index]["target_condition"])
                    for condition_index in range(len(samples[index]["conditions"])):
                        _, baseline_rgb, _, _ = load_phase1a_image(
                            Path(samples[index]["image_paths"][condition_index]), tile_size)
                        baseline_rgb = baseline_rgb.to(args.device)
                        baseline_valid = torch.ones(
                            (1, 1, tile_size, tile_size), device=args.device, dtype=baseline_rgb.dtype)
                        condition_logits, _ = phase1a_model(baseline_rgb, baseline_valid)
                        probability = torch.sigmoid(condition_logits[0, 0]).cpu().numpy()
                        phase1a_predictions.append(probability)
                        if condition_index == target_index:
                            phase1a_logits = condition_logits
                    phase1a_total_loss += float(masked_bce_with_logits(phase1a_logits, target, valid, valid))
                    phase1a_metrics_accumulator.update(
                        phase1a_logits, target, valid.bool(), threshold=args.phase1a_threshold)
                if args.visualization_dir:
                    save_lstm_encoder_visualization(
                        samples[index], temporal_details["features"],
                        lstm_visualization_dir / f"{index:04d}_{samples[index]['sequence_id']}.png",
                        tile_size, args.visualization_scale, args.design)
                    save_visualization(samples[index], torch.sigmoid(logits[0, 0]), target[0, 0],
                                       phase1b_visualization_dir / f"{index:04d}_{samples[index]['sequence_id']}.png",
                                       tile_size, args.visualization_scale,
                                        lstm_threshold=args.lstm_threshold,
                                        phase1a_threshold=args.phase1a_threshold,
                                        features=temporal_details["features"],
                                        design=args.design)
                    if compare:
                        save_phase1a_visualization(
                            samples[index], phase1a_predictions, target[0, 0],
                            phase1a_visualization_dir / f"{index:04d}_{samples[index]['sequence_id']}.png",
                            tile_size, args.visualization_scale, args.phase1a_threshold, args.design)
        metrics = {"test_loss": total_loss / max(len(loader), 1), **metrics_accumulator.compute()}
        if compare:
            metrics["phase1a"] = {"test_loss": phase1a_total_loss / max(len(loader), 1),
                                   **phase1a_metrics_accumulator.compute()}
    else:
        loss, computed = evaluate(model, loader, args.device, cached=cached, threshold=args.lstm_threshold)
        metrics = {"test_loss": loss, **computed}
    result = {"sample_count": len(samples), "mine_types": sorted({sample_mine_type(sample) for sample in samples}), **metrics}
    output = args.metrics_output or (args.visualization_dir / "test_metrics.json" if args.visualization_dir else None)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
