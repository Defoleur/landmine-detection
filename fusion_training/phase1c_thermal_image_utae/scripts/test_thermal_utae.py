#!/usr/bin/env python3
"""Evaluate and visualize a thermal-image U-TAE checkpoint on the test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from common.target_masks import resolve_target_mask
from phase1a_rgb_encoder.losses import masked_bce_with_logits
from phase1a_rgb_encoder.metrics import SegmentationMetrics
from phase1b_rgb_utae.scripts.test_rgb_utae import condition_masks, sample_metrics, save_visualization
from phase1b_rgb_utae.time_encoding import ids_to_time_features
from phase1d_thermal_image_convlstm.sequence import read_manifest
from phase1d_thermal_image_convlstm.temporal_data import ImageSequenceDataset, make_virtual_sequences, variable_image_sequence_collate
from phase1d_thermal_image_utae.temporal_models import ThermalImageUTAE


CONDITIONS = tuple(f"{season}_{daytime}" for season in ("winter", "spring", "summer", "autumn")
                   for daytime in ("morning", "day", "evening"))


def scenario_target(sample, indices, tile_size, use_single_input_mask=True):
    frames = sample["frames"]
    input_masks = []
    for index in indices:
        with Image.open(frames[index]["target_path"]) as mask:
            input_masks.append(np.asarray(mask.convert("L")) > 0)
    if use_single_input_mask and len(indices) == 1:
        candidate = input_masks[0]
    else:
        with Image.open(sample["target_path"]) as mask:
            candidate = np.asarray(mask.convert("L")) > 0
    resolved, details = resolve_target_mask(candidate, input_masks)
    image = Image.fromarray((resolved * 255).astype("uint8")).resize(
        (tile_size, tile_size), Image.Resampling.NEAREST
    )
    return torch.from_numpy((np.asarray(image) > 0).astype("float32")[None]), details


def display_sample(sample):
    value = dict(sample)
    value["image_paths"] = [frame["image_path"] for frame in sample["frames"]]
    value["conditions"] = [frame["condition"] for frame in sample["frames"]]
    return value


def evaluate_scenario(model, samples, loader, count, args, output_dir):
    masks = condition_masks(count)
    aggregate = SegmentationMetrics()
    positive = SegmentationMetrics()
    empty = SegmentationMetrics()
    groups = {}
    records = []
    total_loss = 0.0
    positive_count = empty_count = empty_fp_images = 0
    best, worst = [], []

    with torch.inference_mode():
        for sample_index, (sample, batch) in enumerate(zip(samples, loader)):
            indices = masks[sample_index % len(masks)]
            present = torch.zeros_like(batch["condition_present"])
            present[:, list(indices)] = True
            image = batch["image"].to(args.device) * present[:, :, None, None, None].to(args.device)
            present = present.to(args.device)
            logits, details = model(
                image, present,
                time_features=ids_to_time_features(batch["season_ids"], batch["daytime_ids"]).to(args.device),
                return_details=True,
            )
            target, resolution = scenario_target(
                sample, indices, args.tile_size, args.use_single_input_mask
            )
            target = target.unsqueeze(0).to(args.device)
            valid = torch.ones_like(target)
            loss = masked_bce_with_logits(logits, target, valid, valid)
            values = sample_metrics(logits, target, args.threshold)
            has_positive = bool(target.any())
            aggregate.update(logits, target, valid.bool(), threshold=args.threshold)
            if has_positive:
                positive.update(logits, target, valid.bool(), threshold=args.threshold)
                positive_count += 1
            else:
                empty.update(logits, target, valid.bool(), threshold=args.threshold)
                empty_count += 1
                empty_fp_images += int(values["false_positive"] > 0)
            total_loss += float(loss)
            key = tuple(indices)
            group = groups.setdefault(key, {"all": SegmentationMetrics(), "positive": SegmentationMetrics(),
                                             "empty": SegmentationMetrics(), "loss": 0.0, "count": 0,
                                             "positive_count": 0, "empty_count": 0, "empty_fp_images": 0})
            group["all"].update(logits, target, valid.bool(), threshold=args.threshold)
            if has_positive:
                group["positive"].update(logits, target, valid.bool(), threshold=args.threshold)
                group["positive_count"] += 1
            else:
                group["empty"].update(logits, target, valid.bool(), threshold=args.threshold)
                group["empty_count"] += 1
                group["empty_fp_images"] += int(values["false_positive"] > 0)
            group["loss"] += float(loss)
            group["count"] += 1
            records.append({"sequence_id": sample["sequence_id"], "loss": float(loss),
                            "conditions": [CONDITIONS[index] for index in indices],
                            "target_source": (
                                "original_condition_mask"
                                if args.use_single_input_mask and count == 1
                                else "ideal_sequence_mask"
                            ),
                            **values})
            candidate = {"score": (values["iou"], values["precision"], values["dice"]),
                         "sample_index": sample_index, "indices": indices,
                         "probability": torch.sigmoid(logits[0, 0]).cpu(), "target": target[0, 0].cpu(),
                         "features": {f"1/{scale}": value.detach().cpu() for scale, value in details["features"].items()},
                         "attention": details["attention"][0].mean(dim=0).cpu(), "target_resolution": resolution}
            if has_positive:
                best.append(candidate)
                worst.append(candidate)
                best.sort(key=lambda item: item["score"], reverse=True)
                worst.sort(key=lambda item: item["score"])
                del best[5:]
                del worst[5:]

    scenario_id = f"thermal_{count}"
    scenario_dir = output_dir / scenario_id
    visualizations = {"best": [], "worst": []}
    for label, cases in (("best", best), ("worst", worst)):
        for rank, case in enumerate(cases, 1):
            name = label if rank == 1 else f"{label}_{rank}"
            path = scenario_dir / f"{name}_{samples[case['sample_index']]['sequence_id']}.png"
            save_visualization(display_sample(samples[case["sample_index"]]), case["indices"],
                               case["probability"], case["target"], path, args.tile_size,
                               args.threshold, case["attention"], case["features"],
                               args.visualization_scale, args.design)
            visualizations[label].append(str(path))

    def values(metric):
        return {key: float(value) for key, value in metric.compute().items()}

    result = {
        "id": scenario_id, "count": count, "sample_count": len(samples),
        "test_loss": total_loss / max(len(samples), 1),
        "best_selection": "highest_iou_then_precision_then_dice_among_positive_targets",
        "metrics": values(positive), "metrics_positive_targets": values(positive),
        "metrics_all": values(aggregate), "positive_target_sample_count": positive_count,
        "empty_target_metrics": {"sample_count": empty_count, "false_positive_images": empty_fp_images,
                                 "false_positive_pixels": int(empty.fp),
                                 "false_positives_per_image": float(empty.fp / empty_count) if empty_count else 0.0,
                                 "rejection_rate": float((empty_count - empty_fp_images) / empty_count) if empty_count else None,
                                 "specificity": float(values(empty)["specificity"]) if empty_count else None},
        "mask_metrics": [], "samples": records,
        "best_visualizations": visualizations["best"], "worst_visualizations": visualizations["worst"],
    }
    for indices, group in groups.items():
        result["mask_metrics"].append({
            "conditions": [CONDITIONS[index] for index in indices], "sample_count": group["count"],
            "test_loss": group["loss"] / group["count"],
            "metrics": values(group["positive"]), "metrics_positive_targets": values(group["positive"]),
            "metrics_all": values(group["all"]),
            "empty_target_metrics": {"sample_count": group["empty_count"],
                                     "false_positive_images": group["empty_fp_images"],
                                     "false_positive_pixels": int(group["empty"].fp)},
        })
    scenario_dir.mkdir(parents=True, exist_ok=True)
    (scenario_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader worker processes; use 0 to disable multiprocessing")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--visualization-scale", type=float, default=2.0)
    parser.add_argument("--design", choices=("all", "adaptive"), default="all")
    parser.add_argument("--use-single-input-mask", action="store_true",
                        help="Use the selected condition mask as target for one-input scenarios")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.count < 0 or args.workers < 0 or args.tile_size <= 0 or args.visualization_scale <= 0 or not 0 <= args.threshold <= 1:
        parser.error("invalid count, workers, tile-size, visualization-scale, or threshold")
    samples = [item for item in read_manifest(args.manifest) if item.get("split") == "test"]
    if args.count:
        samples = samples[:args.count]
    if not samples:
        raise SystemExit("No test sequences found")
    dataset = ImageSequenceDataset(samples, args.tile_size)
    loader_options = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": str(args.device).startswith("cuda"),
        "collate_fn": variable_image_sequence_collate,
    }
    if args.workers:
        loader_options.update({"prefetch_factor": 2, "persistent_workers": True})
    loader = DataLoader(dataset, **loader_options)
    model = ThermalImageUTAE().to(args.device).eval()
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload), strict=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [evaluate_scenario(model, samples, loader, count, args, args.output_dir) for count in (1, 2, 4, 8, 12)]
    output = {"manifest": str(args.manifest.resolve()), "checkpoint": str(args.checkpoint.resolve()),
              "counts": [1, 2, 4, 8, 12], "scenarios": results}
    (args.output_dir / "thermal_utae_test_results.json").write_text(json.dumps(output, indent=2) + "\n")
    fields = ["id", "count", "test_loss", "precision", "recall", "specificity", "dice", "iou", "pr_auc", "false_positives_per_image"]
    with (args.output_dir / "thermal_utae_test_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for scenario in results:
            writer.writerow({"id": scenario["id"], "count": scenario["count"], "test_loss": scenario["test_loss"], **scenario["metrics"]})
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
