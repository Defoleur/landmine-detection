#!/usr/bin/env python3
"""Evaluate asynchronous RGB/thermal U-TAE fusion on held-out scenarios."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from phase1a_rgb_encoder.losses import masked_bce_with_logits
from phase1a_rgb_encoder.metrics import SegmentationMetrics
from phase1d_thermal_convlstm.sequence import CONDITIONS
from phase2_rgb_thermal_image_fusion.data import ImageFusionSequenceDataset, read_manifest
from phase2_rgb_thermal_image_fusion.scripts.test_image_fusion import (
    COUNTS,
    _conditions,
    _masked_batch,
    _metric_record,
    _retain,
    _sample_metrics,
    _target_for_scenario,
    build_scenarios,
    save_case,
)
from phase2_rgb_thermal_image_utae.manifest import id_time_features, legacy_modality_ages
from phase2_rgb_thermal_image_utae.models import AsyncUTAESegmenter


def load_compatible(model, payload):
    state = payload.get("model_state_dict", payload)
    expected = model.state_dict()
    state = {key.removeprefix("module."): value for key, value in state.items()}
    state = {
        key: value for key, value in state.items()
        if key in expected and expected[key].shape == value.shape
    }
    model.load_state_dict(state, strict=False)


def evaluate_scenario(model, samples, loader, scenario, args, output_dir):
    aggregate = SegmentationMetrics()
    positive_targets = SegmentationMetrics()
    empty_targets = SegmentationMetrics()
    positive_target_count = empty_target_count = empty_fp_images = 0
    groups = {}
    records, best_all, worst_all = [], [], []
    best_positive, worst_positive = [], []
    total_loss = 0.0

    progress = tqdm(
        zip(samples, loader), total=len(samples), desc=scenario["id"], unit="sample", leave=False
    )
    with torch.inference_mode():
        for index, (sample, batch) in enumerate(progress):
            rgb_indices = scenario["rgb_masks"][index % len(scenario["rgb_masks"])]
            thermal_indices = scenario["thermal_masks"][index % len(scenario["thermal_masks"])]
            rgb, thermal, rgb_available, thermal_available = _masked_batch(
                batch, rgb_indices, thermal_indices, args.device, scenario["mode"]
            )
            season_ids = batch["season_ids"].to(args.device)
            daytime_ids = batch["daytime_ids"].to(args.device)
            rgb_available = rgb_available.to(args.device)
            thermal_available = thermal_available.to(args.device)
            rgb_ages, thermal_ages = legacy_modality_ages(
                season_ids, daytime_ids, rgb_available, thermal_available
            )
            time_features = None
            if not args.without_condition_metadata:
                time_features = id_time_features(season_ids, daytime_ids).to(args.device)
            logits, _details = model(
                rgb,
                thermal,
                rgb_available,
                thermal_available,
                rgb_time_features=time_features,
                thermal_time_features=time_features,
                rgb_ages=rgb_ages.to(args.device),
                thermal_ages=thermal_ages.to(args.device),
                return_details=True,
            )
            target_array, target_path, target_source, target_resolution = _target_for_scenario(
                sample, batch, scenario, rgb_indices, thermal_indices, args.tile_size,
                args.use_single_input_mask,
            )
            target = torch.from_numpy(target_array[None, None]).to(args.device)
            valid = torch.ones_like(target)
            loss = float(masked_bce_with_logits(logits, target, valid, valid))
            aggregate.update(logits, target, valid.bool(), threshold=args.threshold)
            metric = SegmentationMetrics()
            metric.update(logits, target, valid.bool(), threshold=args.threshold)
            record = _metric_record(
                metric, loss, sample, rgb_indices, thermal_indices, target_source
            )
            record.update(_sample_metrics(logits, target, args.threshold))
            records.append(record)
            total_loss += loss

            target_has_positive = bool(target_array.max() > 0)
            if target_has_positive:
                positive_targets.update(logits, target, valid.bool(), threshold=args.threshold)
                positive_target_count += 1
            else:
                empty_targets.update(logits, target, valid.bool(), threshold=args.threshold)
                empty_target_count += 1
                empty_fp_images += int(record["false_positive"] > 0)

            key = json.dumps([rgb_indices, thermal_indices])
            group = groups.setdefault(key, {
                "metrics": SegmentationMetrics(), "loss": 0.0, "count": 0,
                "positive_metrics": SegmentationMetrics(), "positive_count": 0,
                "empty_metrics": SegmentationMetrics(), "empty_count": 0,
                "empty_false_positive_images": 0,
                "rgb_indices": rgb_indices, "thermal_indices": thermal_indices,
                "target_source": target_source,
            })
            group["metrics"].update(logits, target, valid.bool(), threshold=args.threshold)
            if target_has_positive:
                group["positive_metrics"].update(logits, target, valid.bool(), threshold=args.threshold)
                group["positive_count"] += 1
            else:
                group["empty_metrics"].update(logits, target, valid.bool(), threshold=args.threshold)
                group["empty_count"] += 1
                group["empty_false_positive_images"] += int(record["false_positive"] > 0)
            group["loss"] += loss
            group["count"] += 1

            candidate = {
                "rank": (record["iou"], record["precision"], record["dice"]),
                "batch": {key: value.cpu() for key, value in batch.items() if torch.is_tensor(value)},
                "probability": torch.sigmoid(logits[0, 0]).cpu().numpy(),
                "target": target_array,
                "target_path": target_path,
                "target_source": target_source,
                "target_resolution": target_resolution,
                "rgb_indices": rgb_indices,
                "thermal_indices": thermal_indices,
                "dice": record["dice"],
                "iou": record["iou"],
                "precision": record["precision"],
                "sample_index": index,
            }
            _retain(best_all, candidate, reverse=True)
            _retain(worst_all, candidate, reverse=False)
            if target_has_positive:
                _retain(best_positive, candidate, reverse=True)
                _retain(worst_positive, candidate, reverse=False)

    def metric_values(metric):
        return {key: float(value) for key, value in metric.compute().items()}

    mask_metrics = []
    for group in groups.values():
        empty_values = metric_values(group["empty_metrics"])
        empty_count = group["empty_count"]
        mask_metrics.append({
            "rgb_conditions": _conditions(group["rgb_indices"]),
            "thermal_conditions": _conditions(group["thermal_indices"]),
            "target_source": group["target_source"],
            "sample_count": group["count"],
            "test_loss": group["loss"] / group["count"],
            "metrics": metric_values(group["positive_metrics"]),
            "metrics_positive_targets": metric_values(group["positive_metrics"]),
            "metrics_all": metric_values(group["metrics"]),
            "empty_target_metrics": {
                "sample_count": empty_count,
                "false_positive_images": group["empty_false_positive_images"],
                "false_positive_pixels": int(group["empty_metrics"].fp),
                "false_positives_per_image": (
                    float(group["empty_metrics"].fp / empty_count) if empty_count else 0.0
                ),
                "rejection_rate": (
                    float((empty_count - group["empty_false_positive_images"]) / empty_count)
                    if empty_count else None
                ),
                "specificity": empty_values["specificity"] if empty_count else None,
            },
        })

    scenario_dir = output_dir / scenario["id"]
    for label, cases in (("best", best_positive), ("worst", worst_positive)):
        for rank, case in enumerate(cases, 1):
            name = label if rank == 1 else f"{label}_{rank}"
            save_case(
                case, scenario, samples[case["sample_index"]],
                scenario_dir / name, args.threshold,
                font_size=25,
                translate_labels=True,
                probability_cmap="inferno",
            )

    empty_values = metric_values(empty_targets)
    empty_report = {
        "sample_count": empty_target_count,
        "false_positive_images": empty_fp_images,
        "false_positive_pixels": int(empty_targets.fp),
        "false_positives_per_image": (
            float(empty_targets.fp / empty_target_count) if empty_target_count else 0.0
        ),
        "rejection_rate": (
            float((empty_target_count - empty_fp_images) / empty_target_count)
            if empty_target_count else None
        ),
        "specificity": empty_values["specificity"] if empty_target_count else None,
    }
    scenario_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "id": scenario["id"],
        "mode": scenario["mode"],
        "best_selection": "highest_iou_then_precision_then_dice_among_positive_targets",
        "rgb_count": len(scenario["rgb_masks"][0]),
        "thermal_count": len(scenario["thermal_masks"][0]),
        "sample_count": len(samples),
        "test_loss": total_loss / max(len(samples), 1),
        "metrics": metric_values(positive_targets),
        "metrics_positive_targets": metric_values(positive_targets),
        "metrics_all": metric_values(aggregate),
        "positive_target_sample_count": positive_target_count,
        "empty_target_metrics": empty_report,
        "mask_metrics": mask_metrics,
        "samples": records,
        "best_visualizations": [
            str((scenario_dir / ("best" if index == 1 else f"best_{index}")).with_suffix(".png").relative_to(output_dir))
            for index in range(1, len(best_positive) + 1)
        ],
        "worst_visualizations": [
            str((scenario_dir / ("worst" if index == 1 else f"worst_{index}")).with_suffix(".png").relative_to(output_dir))
            for index in range(1, len(worst_positive) + 1)
        ],
        "best_visualization": str((scenario_dir / "best.png").relative_to(output_dir)),
        "worst_visualization": str((scenario_dir / "worst.png").relative_to(output_dir)),
    }
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
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--without-condition-metadata", action="store_true")
    parser.add_argument("--use-single-input-mask", action="store_true",
                        help="Use the selected condition mask as target for one-input scenarios")
    args = parser.parse_args()
    if args.count < 0 or args.workers < 0 or args.tile_size <= 0 or not 0 <= args.threshold <= 1:
        raise SystemExit("invalid count, workers, tile-size, or threshold")

    samples = [item for item in read_manifest(args.manifest) if item.get("split") == "test"]
    if args.count:
        samples = samples[:args.count]
    if not samples:
        raise SystemExit("No test sequences found")

    dataset = ImageFusionSequenceDataset(samples, args.tile_size)
    loader_options = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": str(args.device).startswith("cuda"),
    }
    if args.workers:
        loader_options.update({"prefetch_factor": 2, "persistent_workers": True})
    loader = DataLoader(dataset, **loader_options)
    model = AsyncUTAESegmenter().to(args.device).eval()
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    load_compatible(model, payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        evaluate_scenario(model, samples, loader, scenario, args, args.output_dir)
        for scenario in tqdm(build_scenarios(), desc="Fusion U-TAE test scenarios", unit="scenario")
    ]
    output = {
        "manifest": str(args.manifest.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "use_condition_metadata": not args.without_condition_metadata,
        "conditions": list(CONDITIONS),
        "counts": list(COUNTS),
        "scenarios": results,
    }
    (args.output_dir / "fusion_test_results.json").write_text(json.dumps(output, indent=2) + "\n")
    fields = ["id", "mode", "rgb_count", "thermal_count", "test_loss", "precision", "recall",
              "specificity", "dice", "iou", "pr_auc", "false_positives_per_image"]
    with (args.output_dir / "fusion_test_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for scenario in results:
            writer.writerow({
                "id": scenario["id"], "mode": scenario["mode"],
                "rgb_count": scenario["rgb_count"], "thermal_count": scenario["thermal_count"],
                "test_loss": scenario["test_loss"], **scenario["metrics"]
            })
    print(json.dumps({scenario["id"]: scenario["metrics"] for scenario in results}, indent=2))


if __name__ == "__main__":
    main()
