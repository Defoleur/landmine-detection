#!/usr/bin/env python3
"""Print and optionally save statistics for an RGB mine dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from phase1a_rgb_encoder.data import CONDITION_RE


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser.parse_args()


def dataset_statistics(root: Path):
    image_paths = sorted((root / "images").glob("*.png"))
    stats = {
        "dataset_root": str(root.resolve()),
        "images": len(image_paths),
        "masks": 0,
        "metadata": 0,
        "missing_masks": [],
        "missing_metadata": [],
        "valid_condition_images": 0,
        "ignored_images": 0,
        "positive_images": 0,
        "empty_images": 0,
        "positive_mask_pixels": 0,
        "mask_pixels": 0,
        "mask_values": {},
        "conditions": {},
        "unique_visible_mine_prim_paths": 0,
        "unique_visible_mine_instance_ids": 0,
        "mine_types": {},
        "merged_classes": {},
        "mine_occurrences_by_prim_path": {},
        "mine_condition_coverage": {},
        "images_with_visible_mines": 0,
    }
    condition_counts = defaultdict(Counter)
    prim_occurrences = Counter()
    prim_conditions = defaultdict(set)
    instance_ids = set()
    mine_types = Counter()
    merged_classes = Counter()
    mask_values = Counter()
    for image_path in image_paths:
        stem = image_path.stem
        condition_match = CONDITION_RE.search(stem)
        if condition_match is None:
            stats["ignored_images"] += 1
            continue
        condition = "_".join(condition_match.groups())
        stats["valid_condition_images"] += 1
        mask_path = root / "masks" / f"{stem}.png"
        metadata_path = root / "metadata" / f"{stem}.json"
        if mask_path.exists():
            stats["masks"] += 1
            with Image.open(mask_path) as mask_image:
                values = np.asarray(mask_image)
            stats["mask_pixels"] += int(values.size)
            stats["positive_mask_pixels"] += int((values > 0).sum())
            mask_values.update(int(value) for value in np.unique(values))
            positive = bool(values.max())
            condition_counts[condition]["positive_images" if positive else "empty_images"] += 1
            if positive:
                stats["positive_images"] += 1
            else:
                stats["empty_images"] += 1
        else:
            stats["missing_masks"].append(stem)
        if not metadata_path.exists():
            stats["missing_metadata"].append(stem)
            continue
        stats["metadata"] += 1
        metadata = json.loads(metadata_path.read_text())
        visible_mines = metadata.get("visible_mines", [])
        if visible_mines:
            stats["images_with_visible_mines"] += 1
        for mine in visible_mines:
            prim_path = mine.get("prim_path")
            if prim_path:
                prim_occurrences[prim_path] += 1
                prim_conditions[prim_path].add(condition)
            if mine.get("instance_id") is not None:
                instance_ids.add(str(mine["instance_id"]))
            if mine.get("mine_type"):
                mine_types[mine["mine_type"]] += 1
            if mine.get("merged_class"):
                merged_classes[mine["merged_class"]] += 1

    for condition in sorted(condition_counts):
        stats["conditions"][condition] = dict(condition_counts[condition])
    stats["unique_visible_mine_prim_paths"] = len(prim_occurrences)
    stats["unique_visible_mine_instance_ids"] = len(instance_ids)
    stats["mine_types"] = dict(mine_types)
    stats["merged_classes"] = dict(merged_classes)
    stats["mask_values"] = dict(sorted(mask_values.items()))
    stats["mine_occurrences_by_prim_path"] = dict(sorted(prim_occurrences.items()))
    stats["mine_condition_coverage"] = {
        path: sorted(conditions) for path, conditions in sorted(prim_conditions.items())
    }
    stats["positive_pixel_fraction"] = stats["positive_mask_pixels"] / max(stats["mask_pixels"], 1)
    return stats


def main():
    args = arguments()
    report = dataset_statistics(args.dataset_root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    summary = {
        key: report[key] for key in (
            "images", "valid_condition_images", "ignored_images", "masks", "metadata",
            "positive_images", "empty_images", "positive_pixel_fraction",
            "images_with_visible_mines", "unique_visible_mine_prim_paths",
            "unique_visible_mine_instance_ids", "mine_types", "merged_classes",
        )
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        print(f"Full report written to {args.output}")


if __name__ == "__main__":
    main()
