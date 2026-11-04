#!/usr/bin/env python3
"""Evaluate and visualize an RGB U-TAE checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from common.target_masks import resolve_target_mask
from phase1a_rgb_encoder.losses import masked_bce_with_logits
from phase1a_rgb_encoder.metrics import SegmentationMetrics
from phase1a_rgb_encoder.models import RGBEncoder, RGBSegmentationDecoder
from phase1a_rgb_encoder.data import polygons_to_mask
from phase1b_rgb_convlstm.data import PointConditionDataset, read_temporal_manifest, variable_sequence_collate
from phase1b_rgb_convlstm.data.point_dataset import CONDITIONS
from phase1b_rgb_utae.models import RGBUTAE
from phase1b_rgb_utae.time_encoding import ids_to_time_features
from phase1w_valid.visualization_designs import save_designs


COUNTS = (1, 2, 4, 8, 12)


def condition_masks(count: int) -> list[tuple[int, ...]]:
    return list(itertools.combinations(range(12), count))


def sample_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float) -> dict:
    predicted = torch.sigmoid(logits.detach()) >= threshold
    labels = target.detach().bool()
    tp = int((predicted & labels).sum())
    fp = int((predicted & ~labels).sum())
    fn = int((~predicted & labels).sum())
    tn = int((~predicted & ~labels).sum())
    denominator = 2 * tp + fp + fn
    union = tp + fp + fn
    return {
        "precision": tp / max(tp + fp, 1) if tp + fp else 1.0,
        "recall": tp / max(tp + fn, 1) if tp + fn else 1.0,
        "specificity": tn / max(tn + fp, 1),
        "dice": 2 * tp / denominator if denominator else 1.0,
        "iou": tp / union if union else 1.0,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def load_mask(path: str, image_path: str, tile_size: int) -> np.ndarray:
    if path and path.lower().endswith(".png"):
        with Image.open(path) as source:
            values = np.asarray(source.convert("L")) > 0
    elif path:
        with Image.open(image_path) as source:
            values = polygons_to_mask(path, source.size)
    else:
        with Image.open(image_path) as source:
            values = np.zeros((source.height, source.width), dtype=bool)
    return np.asarray(
        Image.fromarray((values * 255).astype(np.uint8)).resize(
            (tile_size, tile_size), Image.Resampling.NEAREST
        )
    ) > 0


def scenario_target(sample: dict, indices: tuple[int, ...], tile_size: int,
                    use_single_input_mask: bool = True) -> tuple[torch.Tensor, dict]:
    input_masks = [
        load_mask(sample["mask_paths"][index], sample["image_paths"][index], tile_size)
        for index in indices
    ]
    if use_single_input_mask and len(indices) == 1:
        candidate = input_masks[0]
    else:
        target_path = sample.get("target_path", "")
        target_index = sample.get("conditions", []).index(sample["target_condition"]) \
            if sample.get("target_condition") in sample.get("conditions", []) else indices[0]
        candidate = load_mask(target_path, sample["image_paths"][target_index], tile_size)
    resolved, details = resolve_target_mask(candidate, input_masks)
    return torch.from_numpy(resolved.astype(np.float32)[None]), details


def masked_input(batch: dict, indices: tuple[int, ...], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    rgb = batch["rgb"].to(device).clone()
    present = torch.zeros_like(batch["condition_present"])
    present[:, list(indices)] = True
    present = present.to(device)
    rgb *= present[:, :, None, None, None].to(rgb.dtype)
    return rgb, present


def save_visualization(
    sample: dict,
    indices: tuple[int, ...],
    probability: torch.Tensor,
    target: torch.Tensor,
    path: Path,
    tile_size: int,
    threshold: float,
    attention: torch.Tensor | None = None,
    features: dict | None = None,
    scale: float = 2.0,
    design: str = "all",
) -> None:
    thumb = max(int(round(tile_size * scale)), 64)

    def mask_image(mask, threshold_value=0.5):
        values = (np.asarray(mask) >= threshold_value).astype(np.uint8) * 255
        return Image.fromarray(values).resize((thumb, thumb), Image.Resampling.NEAREST).convert("RGB")

    def probability_heatmap(values):
        resized = np.asarray(Image.fromarray((np.clip(values, 0, 1) * 255).astype(np.uint8)).resize(
            (thumb, thumb), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        return Image.fromarray((colormaps["inferno"](resized)[..., :3] * 255).astype(np.uint8))

    def feature_heatmap(feature):
        if feature.ndim == 4:
            feature = feature[0]
        activation = feature.detach().float().cpu().pow(2).mean(dim=0).sqrt().numpy()
        maximum = float(activation.max())
        if maximum > 0:
            activation /= maximum
        return Image.fromarray((activation * 255).astype(np.uint8), mode="L").resize(
            (thumb, thumb), Image.Resampling.BILINEAR).convert("RGB")

    prediction = probability.numpy()
    target_array = target.numpy() > 0.5
    predicted_array = prediction >= threshold
    error = np.zeros((*target_array.shape, 3), dtype=np.uint8)
    error[target_array & predicted_array] = (0, 190, 0)
    error[~target_array & predicted_array] = (220, 0, 0)
    error[target_array & ~predicted_array] = (0, 80, 220)
    panels = [
        probability_heatmap(prediction),
        mask_image(target),
        mask_image(prediction, threshold),
        Image.fromarray(error),
    ]
    labels = ["probability", "ground truth", f"prediction >= {threshold:g}", "TP green FP red FN blue"]
    for scale_name in ("1/2", "1/4", "1/8"):
        if features and scale_name in features:
            panels.append(feature_heatmap(features[scale_name]))
            labels.append(f"U-TAE {scale_name}")

    input_panels = []
    for index in indices:
        with Image.open(sample["image_paths"][index]) as source:
            input_panels.append((CONDITIONS[index], source.convert("RGB").resize((thumb, thumb), Image.Resampling.BILINEAR)))
    save_designs(input_panels, path.parent / "designs", "1b_inputs", path.stem, design, columns=3)
    save_designs(list(zip(labels, panels)), path.parent / "designs", "1b_outputs", path.stem, design, columns=4)
    rendered = path.parent / "designs" / "adaptive" / "1b_outputs" / f"{path.stem}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rendered, path)


def evaluate_scenario(model, samples, loader, count, args, output_dir=None):
    masks = condition_masks(count)
    metrics = SegmentationMetrics()
    positive_metrics = SegmentationMetrics()
    empty_metrics = SegmentationMetrics()
    total_loss = 0.0
    positive_target_count = 0
    empty_target_count = 0
    empty_target_false_positive_images = 0
    groups = {}
    sample_records = []
    best, worst = [], []

    with torch.inference_mode():
        for sample_index, batch in enumerate(loader):
            indices = masks[sample_index % len(masks)]
            rgb, present = masked_input(batch, indices, args.device)
            logits, details = model(
                rgb,
                present,
                time_features=ids_to_time_features(
                    batch["season_ids"], batch["daytime_ids"]
                ).to(args.device),
                return_details=True,
            )
            target, target_resolution = scenario_target(
                samples[sample_index], indices, args.tile_size, args.use_single_input_mask
            )
            target = target.unsqueeze(0).to(args.device)
            valid = torch.ones_like(target)
            loss = masked_bce_with_logits(logits, target, valid, valid)
            total_loss += float(loss)
            metrics.update(logits, target, valid.bool(), threshold=args.threshold)
            values = sample_metrics(logits, target, args.threshold)
            target_has_positive = bool(target.any())
            if target_has_positive:
                positive_metrics.update(logits, target, valid.bool(), threshold=args.threshold)
                positive_target_count += 1
            else:
                empty_metrics.update(logits, target, valid.bool(), threshold=args.threshold)
                empty_target_count += 1
                empty_target_false_positive_images += int(values["false_positive"] > 0)
            sample_records.append({
                "sequence_id": samples[sample_index]["sequence_id"],
                "mine_id": samples[sample_index].get("mine_id"),
                "point_number": samples[sample_index].get("point_number"),
                "loss": float(loss),
                "conditions": [CONDITIONS[index] for index in indices],
                "target_source": (
                    "original_condition_mask"
                    if args.use_single_input_mask and len(indices) == 1
                    else "ideal_sequence_mask"
                ),
                **values,
            })
            group = groups.setdefault(indices, {
                "metrics": SegmentationMetrics(),
                "positive_metrics": SegmentationMetrics(),
                "empty_metrics": SegmentationMetrics(),
                "loss": 0.0,
                "count": 0,
                "positive_count": 0,
                "empty_count": 0,
                "empty_false_positive_images": 0,
            })
            group["metrics"].update(logits, target, valid.bool(), threshold=args.threshold)
            if target_has_positive:
                group["positive_metrics"].update(logits, target, valid.bool(), threshold=args.threshold)
                group["positive_count"] += 1
            else:
                group["empty_metrics"].update(logits, target, valid.bool(), threshold=args.threshold)
                group["empty_count"] += 1
                group["empty_false_positive_images"] += int(values["false_positive"] > 0)
            group["loss"] += float(loss)
            group["count"] += 1
            candidate = {
                "score": (values.get("iou", 0.0), values.get("dice", 0.0)),
                "sample_index": sample_index,
                "indices": indices,
                "probability": torch.sigmoid(logits[0, 0]).cpu(),
                "target": target[0, 0].cpu(),
                "features": {
                    f"1/{scale}": value.detach().cpu()
                    for scale, value in details["features"].items()
                },
                "attention": details["attention"][0].mean(dim=0).cpu(),
                "target_resolution": target_resolution,
            }
            if bool(target.any()):
                best.append(candidate)
                worst.append(candidate)
                best.sort(key=lambda item: item["score"], reverse=True)
                worst.sort(key=lambda item: item["score"])
                del best[5:]
                del worst[5:]

    scenario_id = f"rgb_{count}"
    visualizations = {"best": [], "worst": []}
    if output_dir:
        for label, cases in (("best", best), ("worst", worst)):
            for rank, case in enumerate(cases, 1):
                path = output_dir / scenario_id / f"{label}_{rank}_{samples[case['sample_index']]['sequence_id']}.png"
                save_visualization(samples[case["sample_index"]], case["indices"], case["probability"],
                                   case["target"], path, args.tile_size, args.threshold,
                                   case["attention"], case["features"], args.visualization_scale, args.design)
                visualizations[label].append(str(path))
    result = {
        "id": scenario_id,
        "count": count,
        "sample_count": len(samples),
        "test_loss": total_loss / max(len(samples), 1),
        "target_source": (
            "mixed: original_condition_mask for count 1, ideal_sequence_mask otherwise"
            if args.use_single_input_mask
            else "ideal_sequence_mask",
        ),
        "best_selection": "highest_iou_then_precision_then_dice_among_positive_targets",
        "metrics": {
            key: float(value) for key, value in positive_metrics.compute().items()
        },
        "metrics_positive_targets": {
            key: float(value) for key, value in positive_metrics.compute().items()
        },
        "metrics_all": {
            key: float(value) for key, value in metrics.compute().items()
        },
        "positive_target_sample_count": positive_target_count,
        "empty_target_metrics": {
            "sample_count": empty_target_count,
            "false_positive_images": empty_target_false_positive_images,
            "false_positive_pixels": int(empty_metrics.fp),
            "false_positives_per_image": (
                float(empty_metrics.fp / empty_target_count) if empty_target_count else 0.0
            ),
            "rejection_rate": (
                float((empty_target_count - empty_target_false_positive_images) / empty_target_count)
                if empty_target_count else None
            ),
            "specificity": (
                float(empty_metrics.compute()["specificity"]) if empty_target_count else None
            ),
        },
        "mask_metrics": [
            {
                "conditions": [CONDITIONS[index] for index in indices],
                "sample_count": group["count"],
                "test_loss": group["loss"] / max(group["count"], 1),
                "target_source": (
                    "original_condition_mask"
                    if args.use_single_input_mask and len(indices) == 1
                    else "ideal_sequence_mask"
                ),
                "metrics": {key: float(value) for key, value in group["positive_metrics"].compute().items()},
                "metrics_positive_targets": {
                    key: float(value) for key, value in group["positive_metrics"].compute().items()
                },
                "metrics_all": {key: float(value) for key, value in group["metrics"].compute().items()},
                "empty_target_metrics": {
                    "sample_count": group["empty_count"],
                    "false_positive_images": group["empty_false_positive_images"],
                    "false_positive_pixels": int(group["empty_metrics"].fp),
                    "false_positives_per_image": (
                        float(group["empty_metrics"].fp / group["empty_count"])
                        if group["empty_count"] else 0.0
                    ),
                    "rejection_rate": (
                        float((group["empty_count"] - group["empty_false_positive_images"])
                              / group["empty_count"])
                        if group["empty_count"] else None
                    ),
                },
            }
            for indices, group in groups.items()
        ],
        "samples": sample_records,
        "best_visualizations": visualizations["best"],
        "worst_visualizations": visualizations["worst"],
    }
    if output_dir:
        (output_dir / scenario_id).mkdir(parents=True, exist_ok=True)
        (output_dir / scenario_id / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--count", type=int, default=0, help="Number of test samples; 0 means all")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader worker processes; use 0 to disable multiprocessing")
    parser.add_argument("--mine-type", nargs="+")
    parser.add_argument("--mine-id", "--landmine", action="append", dest="mine_ids")
    parser.add_argument("--visualization-dir", type=Path)
    parser.add_argument("--visualization-scale", type=float, default=2.0)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use-single-input-mask", action="store_true",
                        help="Use the selected condition mask as target for one-input scenarios")
    parser.add_argument("--design", choices=("all", "adaptive"), default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.count < 0 or args.workers < 0 or args.visualization_scale <= 0 or not 0 <= args.threshold <= 1:
        parser.error("invalid count, workers, visualization scale, or threshold")

    samples = [item for item in read_temporal_manifest(args.manifest) if item["split"] == "test"]
    if args.mine_type:
        requested_types = {value.casefold() for value in args.mine_type}
        samples = [item for item in samples if str(item.get("mine_type", "")).casefold() in requested_types]
    if args.mine_ids:
        requested = {value.casefold().removeprefix("obstacle_mine_") for value in args.mine_ids}
        samples = [item for item in samples if item.get("mine_id", "").casefold().removeprefix("obstacle_mine_") in requested]
    if args.count:
        samples = samples[:args.count]
    if not samples:
        raise SystemExit("No test samples matched the requested filters")

    dataset = PointConditionDataset(samples, args.tile_size)
    loader_options = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": str(args.device).startswith("cuda"),
        "collate_fn": variable_sequence_collate,
    }
    if args.workers:
        loader_options.update({"prefetch_factor": 2, "persistent_workers": True})
    loader = DataLoader(dataset, **loader_options)
    model = RGBUTAE(RGBEncoder(pretrained=False), RGBSegmentationDecoder()).to(args.device).eval()
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload), strict=False)

    scenarios = [evaluate_scenario(model, samples, loader, count, args, args.visualization_dir) for count in COUNTS]
    result = {"sample_count": len(samples), "counts": list(COUNTS), "scenarios": scenarios}
    output = args.metrics_output or (args.visualization_dir / "test_metrics.json" if args.visualization_dir else None)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        with (output.parent / "test_summary.csv").open("w", newline="") as stream:
            fields = ["id", "count", "test_loss", "precision", "recall", "specificity", "dice", "iou", "pr_auc", "false_positives_per_image"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for scenario in scenarios:
                writer.writerow({"id": scenario["id"], "count": scenario["count"], "test_loss": scenario["test_loss"], **scenario["metrics"]})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
