"""Build and load 12-condition samples from the point dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from phase1a_rgb_encoder.data import IMAGENET_MEAN, IMAGENET_STD, polygons_to_mask

CONDITIONS = tuple(f"{season}_{daytime}" for season in ("winter", "spring", "summer", "autumn")
                   for daytime in ("morning", "day", "evening"))
CONDITION_RE = re.compile(r"_(spring|summer|autumn|winter)_(morning|day|evening)$")
POINT_RE = re.compile(r"_(p\d{2})_obstacle_mine_(.+)$")


def _metadata(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _mine_type(metadata: dict, mine_id: str) -> str:
    for item in metadata.get("visible_mines") or metadata.get("mine_masks") or []:
        if item.get("sim_name") == mine_id and item.get("mine_type"):
            return str(item["mine_type"])
    for item in metadata.get("visible_mines") or metadata.get("mine_masks") or []:
        if item.get("mine_type"):
            return str(item["mine_type"])
    metadata_type = metadata.get("mine_type")
    if metadata_type:
        return str(metadata_type)
    # Occluded targets may have no visible_mines entry; recover the simulator
    # type from names such as obstacle_mine_pmn4_8.
    name = str(mine_id).removeprefix("obstacle_mine_")
    return f"Mine_{name.rsplit('_', 1)[0].upper()}" if "_" in name else f"Mine_{name.upper()}"


def discover_point_records(dataset_root: str | Path) -> list[dict]:
    """Return frame records grouped later by metadata mine and point number."""
    root = Path(dataset_root)
    records = []
    for image_path in sorted((root / "images").glob("*.png")):
        match = CONDITION_RE.search(image_path.stem)
        if not match:
            continue
        metadata_path = root / "metadata" / f"{image_path.stem}.json"
        metadata = _metadata(metadata_path)
        point = metadata.get("point_number")
        mine = metadata.get("mine")
        point_match = POINT_RE.search(image_path.stem)
        if point is None and point_match:
            point = int(point_match.group(1)[1:])
        if not mine and point_match:
            mine = point_match.group(2).rsplit("_", 1)[0]
        if not mine or point is None:
            raise ValueError(f"Cannot determine mine/point for {image_path}")
        stem = image_path.stem
        mask_path = root / "masks" / f"{stem}.png"
        label_path = root / "exact_labels" / f"{stem}.txt"
        if not label_path.exists():
            label_path = root / "labels" / f"{stem}.txt"
        if not mask_path.exists() and not label_path.exists():
            raise FileNotFoundError(f"Missing mask or label for {image_path}")
        records.append({
            "frame_id": stem,
            "image_path": str(image_path.resolve()),
            "mask_path": str(mask_path.resolve()) if mask_path.exists() else "",
            "label_path": str(label_path.resolve()) if label_path.exists() else "",
            "metadata_path": str(metadata_path.resolve()),
            "mine_id": str(mine),
            "point_number": int(point),
            "mine_type": _mine_type(metadata, str(mine)),
            "condition": f"{match.group(1)}_{match.group(2)}",
            "timestamp": metadata.get("timestamp", ""),
            "trajectory": metadata.get("trajectory", ""),
        })
    return records


def discover_point_mask_records(dataset_root: str | Path) -> dict[tuple[str, int], dict]:
    """Discover one full-scene mask per mine/point identity."""
    root = Path(dataset_root)
    records = {}
    for metadata_path in sorted((root / "metadata").glob("*.json")):
        metadata = _metadata(metadata_path)
        mine = metadata.get("mine")
        point = metadata.get("point_number")
        mask_info = metadata.get("mine_mask") or {}
        mask_file = mask_info.get("file")
        if not mine or point is None or not mask_file:
            continue
        mask_path = root / mask_file
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing target mask for {metadata_path}: {mask_path}")
        key = (str(mine), int(point))
        if key in records:
            raise ValueError(f"Duplicate target mask identity: {key}")
        records[key] = {
            "mask_path": str(mask_path.resolve()),
            "image_path": str((root / "images" / mask_path.name).resolve()),
            "point_index": metadata.get("point_index"),
            "points_per_mine": metadata.get("points_per_mine"),
            "trajectory": metadata.get("trajectory"),
            "position": metadata.get("position"),
        }
    return records


def attach_target_mask_paths(samples, dataset_root: str | Path) -> list[dict]:
    """Attach full-scene target masks to existing temporal samples."""
    target_masks = discover_point_mask_records(dataset_root)
    sample_keys = {(sample["mine_id"], int(sample["point_number"])) for sample in samples}
    target_keys = set(target_masks)
    missing = sorted(sample_keys - target_keys)
    if missing:
        raise ValueError(
            "Target mask point identities do not match samples: "
            f"missing={missing[:3]}"
        )
    attached = []
    for sample in samples:
        value = dict(sample)
        target = target_masks[(sample["mine_id"], int(sample["point_number"]))]
        if not Path(target["image_path"]).exists():
            raise FileNotFoundError(f"Missing target image: {target['image_path']}")
        value["target_path"] = target["mask_path"]
        value["target_image_path"] = target["image_path"]
        attached.append(value)
    return attached


def _split_points(groups, seed=42):
    """Assign 8/1/1 points per landmine, keeping conditions together."""
    rng = random.Random(seed)
    splits = {}
    for mine in sorted({mine for mine, _ in groups}):
        points = sorted(point for current_mine, point in groups if current_mine == mine)
        rng.shuffle(points)
        if len(points) < 3:
            raise ValueError(f"Landmine {mine} has {len(points)} points; at least 3 are required")
        train_count = len(points) - 2
        for index, point in enumerate(points):
            splits[(mine, point)] = "train" if index < train_count else "val" if index == train_count else "test"
    return splits


def build_point_manifest(dataset_root: str | Path, target_condition="autumn_day", seed=42,
                         require_complete=True, target_mask_dataset=None) -> tuple[list[dict], dict]:
    records = discover_point_records(dataset_root)
    target_masks = discover_point_mask_records(target_mask_dataset) if target_mask_dataset else {}
    source_points = {}
    source_conditions = {}
    for record in records:
        source_points.setdefault((record["mine_id"], record["point_number"]), record)
        source_conditions[(record["mine_id"], record["point_number"], record["condition"])] = record
    coordinate_deltas = []
    target_metadata_mismatches = []
    if target_mask_dataset:
        source_keys = set(source_points)
        target_keys = set(target_masks)
        missing_targets = sorted(source_keys - target_keys)
        extra_targets = sorted(target_keys - source_keys)
        if missing_targets or extra_targets:
            raise ValueError(
                "Target mask point identities do not match source points: "
                f"missing={missing_targets[:3]} extra={extra_targets[:3]}"
            )
        for key, target in target_masks.items():
            source = source_conditions.get((*key, target_condition), source_points.get(key))
            if source is None:
                continue
            source_metadata = _metadata(Path(source["metadata_path"]))
            fields = ("point_index", "points_per_mine", "trajectory")
            mismatch = {field: (source_metadata.get(field), target.get(field))
                        for field in fields if source_metadata.get(field) != target.get(field)}
            if mismatch:
                target_metadata_mismatches.append({"key": key, "fields": mismatch})
            source_position = source_metadata.get("position")
            target_position = target.get("position")
            if source_position and target_position:
                coordinate_deltas.append(math.dist(source_position[:2], target_position[:2]))
        if target_metadata_mismatches:
            raise ValueError(f"Target point metadata mismatch: {target_metadata_mismatches[:3]}")
    groups = defaultdict(dict)
    for record in records:
        groups[(record["mine_id"], record["point_number"])][record["condition"]] = record
    point_splits = _split_points(groups, seed)
    samples, rejected = [], []
    for (mine, point), conditions in sorted(groups.items()):
        missing = sorted(set(CONDITIONS) - set(conditions))
        duplicate = len(conditions) != len([r for r in records if r["mine_id"] == mine and r["point_number"] == point])
        if missing or duplicate or target_condition not in conditions:
            rejected.append({"mine_id": mine, "point_number": point, "missing_conditions": missing,
                             "duplicate_condition": duplicate})
            if require_complete:
                continue
        ordered = [conditions[c] for c in CONDITIONS if c in conditions]
        if not ordered:
            continue
        sample_target_path = (conditions.get(target_condition, ordered[-1])["mask_path"] or
                              conditions.get(target_condition, ordered[-1])["label_path"])
        target_key = (mine, point)
        if target_mask_dataset:
            target_record = target_masks.get(target_key)
            if target_record is None:
                rejected.append({"mine_id": mine, "point_number": point,
                                 "missing_target_mask": True})
                continue
            sample_target_path = target_record["mask_path"]
        samples.append({
            "sequence_id": f"{mine}_p{point:02d}", "mine_id": mine, "point_number": point,
            "mine_type": next((r.get("mine_type", "") for r in ordered if r.get("mine_type")), ""),
            "split": point_splits[(mine, point)], "conditions": [r["condition"] for r in ordered],
            "frame_ids": [r["frame_id"] for r in ordered],
            "image_paths": [r["image_path"] for r in ordered],
            "mask_paths": [r["mask_path"] for r in ordered],
            "label_paths": [r["label_path"] for r in ordered],
            "target_condition": target_condition,
            "target_frame_id": conditions.get(target_condition, ordered[-1])["frame_id"],
            "target_path": sample_target_path,
            **({"target_image_path": target_record["image_path"]} if target_mask_dataset else {}),
        })
    report = {"source_frames": len(records), "candidate_samples": len(groups),
              "accepted_samples": len(samples), "rejected_samples": len(rejected),
              "rejections": rejected, "split_strategy": "per_landmine_points_8_1_1",
              "splits": {s: sum(x["split"] == s for x in samples)
                         for s in ("train", "val", "test")},
               "target_condition": target_condition, "conditions": list(CONDITIONS)}
    if target_mask_dataset:
        report["target_mask_dataset"] = str(Path(target_mask_dataset).resolve())
        report["target_mask_points"] = len(target_masks)
        report["target_coordinate_delta"] = {
            "count": len(coordinate_deltas),
            "min": min(coordinate_deltas) if coordinate_deltas else None,
            "max": max(coordinate_deltas) if coordinate_deltas else None,
            "mean": sum(coordinate_deltas) / len(coordinate_deltas) if coordinate_deltas else None,
        }
        report["target_metadata_mismatches"] = len(target_metadata_mismatches)
    return samples, report


def write_temporal_manifest(samples: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(samples, indent=2))


def read_temporal_manifest(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())


class PointConditionDataset(Dataset):
    def __init__(self, samples, tile_size=256, augment=False, condition_present=None):
        self.samples = samples
        self.tile_size = tile_size
        self.augment = augment
        self.condition_present = condition_present

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        images, targets = [], []
        present = []
        for image_path, mask_path, label_path in zip(sample["image_paths"], sample["mask_paths"], sample["label_paths"]):
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                if self.tile_size:
                    image = image.resize((self.tile_size, self.tile_size), Image.Resampling.BILINEAR)
                image_array = np.asarray(image, dtype=np.float32) / 255.0
            if mask_path:
                with Image.open(mask_path) as mask:
                    target = (np.asarray(mask) > 0).astype(np.float32)
            else:
                with Image.open(image_path) as original:
                    target = polygons_to_mask(label_path, original.size)
            target_image = Image.fromarray((target * 255).astype(np.uint8))
            if self.tile_size:
                target = np.asarray(target_image.resize((self.tile_size, self.tile_size), Image.Resampling.NEAREST), dtype=np.float32) / 255.0
            images.append(image_array)
            targets.append(target)
            present.append(True)
        if self.condition_present is not None:
            present = list(self.condition_present)
            while len(images) < len(CONDITIONS):
                images.append(np.zeros((self.tile_size, self.tile_size, 3), dtype=np.float32))
                targets.append(np.zeros((self.tile_size, self.tile_size), dtype=np.float32))
            images = images[:len(CONDITIONS)]
            targets = targets[:len(CONDITIONS)]
        rgb = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
        rgb = (rgb - torch.tensor(IMAGENET_MEAN)[None, :, None, None]) / torch.tensor(IMAGENET_STD)[None, :, None, None]
        valid = torch.ones(rgb.shape[0], 1, rgb.shape[2], rgb.shape[3], dtype=rgb.dtype)
        rgb = torch.cat((rgb, valid), dim=1)
        target_path = sample.get("target_path")
        if target_path and target_path.endswith(".png"):
            with Image.open(target_path) as mask:
                target = (np.asarray(mask) > 0).astype(np.float32)
        elif target_path:
            with Image.open(sample["image_paths"][sample["conditions"].index(sample["target_condition"])]) as image:
                target = polygons_to_mask(target_path, image.size)
        else:
            target_index = sample["conditions"].index(sample["target_condition"])
            target = targets[target_index]
        target_image = Image.fromarray((target * 255).astype(np.uint8))
        if self.tile_size:
            target = np.asarray(target_image.resize((self.tile_size, self.tile_size), Image.Resampling.NEAREST),
                                 dtype=np.float32) / 255.0
        target = torch.from_numpy(target[None].copy())
        return {"rgb": rgb, "rgb_validity": valid, "target": target,
                "target_validity": torch.ones_like(target),
                "condition_present": torch.tensor(present, dtype=torch.bool),
                "metadata": {"sequence_id": sample["sequence_id"], "mine_id": sample["mine_id"],
                             "point_number": sample["point_number"], "conditions": sample["conditions"],
                             "frame_ids": sample["frame_ids"]}}


class CachedPointConditionDataset(Dataset):
    """Load the same point samples from a feature-cache manifest."""

    def __init__(self, samples, tile_size=256):
        self.samples = samples
        self.tile_size = tile_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        cached = [torch.load(path, map_location="cpu", weights_only=False) for path in sample["cache_paths"]]
        features = {scale: torch.stack([item["features"][scale] for item in cached])
                    for scale in ("1/2", "1/4", "1/8")}
        target_path = sample["target_path"]
        if target_path.endswith(".png"):
            with Image.open(target_path) as mask:
                target_array = (np.asarray(mask) > 0).astype(np.float32)
        else:
            with Image.open(sample["image_paths"][sample["conditions"].index(sample["target_condition"])]) as image:
                target_array = polygons_to_mask(target_path, image.size)
        if self.tile_size:
            target_array = np.asarray(Image.fromarray((target_array * 255).astype(np.uint8)).resize(
                (self.tile_size, self.tile_size), Image.Resampling.NEAREST), dtype=np.float32) / 255.0
        target = torch.from_numpy(target_array[None].copy())
        return {"features": features,
                "feature_validity": {scale: torch.stack([item["feature_validity"][scale] for item in cached])
                                     for scale in ("1/2", "1/4", "1/8")},
                "target": target, "target_validity": torch.ones_like(target),
                "condition_present": torch.ones(len(cached), dtype=torch.bool),
                "metadata": {"sequence_id": sample["sequence_id"], "mine_id": sample["mine_id"],
                             "point_number": sample["point_number"], "conditions": sample["conditions"]}}
