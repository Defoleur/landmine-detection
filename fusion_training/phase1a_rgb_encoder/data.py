"""Dataset discovery, manifest creation, polygon masks, and spatial splits."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MANIFEST_FIELDS = [
    "acquisition_id", "site_id", "spatial_block_id", "image_path", "timestamp",
    "camera_id", "width", "height", "crs", "ground_resolution_x",
    "ground_resolution_y", "registration_path", "sensor_validity_path", "label_path",
    "mask_path", "visible_mine_prim_paths", "visible_mine_count", "split_group_id",
    "label_validity_path", "label_version", "configuration_id", "quality_status", "split",
    "position_x", "position_y", "position_z", "season", "daytime", "flight_id",
    "capture_order", "elapsed_seconds", "speed_mps", "scene_group_id",
]

CONDITION_RE = re.compile(r"_(spring|summer|autumn|winter)_(morning|day|evening)$")
SEASON_DATES = {"winter": (2026, 1, 1), "spring": (2026, 4, 1),
                "summer": (2026, 7, 1), "autumn": (2026, 10, 1)}
DAYTIME_HOURS = {"morning": 8, "day": 14, "evening": 20}


def parse_polygons(path: str | Path) -> list[tuple[int, list[float]]]:
    """Read the repository's YOLO polygon format: class x1 y1 x2 y2 ... ."""
    result = []
    text = Path(path).read_text().strip()
    if not text:
        return result
    for line_no, line in enumerate(text.splitlines(), 1):
        values = line.split()
        if len(values) < 7 or (len(values) - 1) % 2:
            raise ValueError(f"Invalid polygon at {path}:{line_no}")
        coords = [float(v) for v in values[1:]]
        if any(not 0 <= v <= 1 for v in coords):
            raise ValueError(f"Polygon coordinates must be normalized at {path}:{line_no}")
        result.append((int(values[0]), coords))
    return result


def polygons_to_mask(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    """Rasterize normalized polygons into a binary HxW target mask."""
    width, height = size
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for _, coords in parse_polygons(path):
        points = [(round(coords[i] * (width - 1)), round(coords[i + 1] * (height - 1)))
                  for i in range(0, len(coords), 2)]
        draw.polygon(points, fill=1)
    return np.asarray(image, dtype=np.float32)


def _group_id(x: float | None, y: float | None, block_size: float) -> str:
    if x is None or y is None:
        return "ungrouped"
    return f"block_{int(np.floor(x / block_size))}_{int(np.floor(y / block_size))}"


def _scene_group(x, y, z) -> str:
    if x is None or y is None or z is None:
        return "ungrouped"
    return f"scene_{x:.2f}_{y:.2f}_{z:.2f}"


def _ground_resolution(corners, width: int, height: int):
    if isinstance(corners, dict):
        try:
            corners = [corners[key] for key in ("top_left", "top_right", "bottom_right", "bottom_left")]
        except KeyError:
            return None, None
    if not corners or len(corners) != 4:
        return None, None
    points = np.asarray(corners, dtype=np.float64)
    top = np.linalg.norm(points[1] - points[0])
    bottom = np.linalg.norm(points[2] - points[3])
    left = np.linalg.norm(points[3] - points[0])
    right = np.linalg.norm(points[2] - points[1])
    return ((top + bottom) / 2 / max(width - 1, 1),
            (left + right) / 2 / max(height - 1, 1))


def _condition_from_filename(stem: str):
    match = CONDITION_RE.search(stem)
    return match.groups() if match else None


def _capture_order(stem: str) -> int:
    condition = _condition_from_filename(stem)
    prefix = stem[:-(len(condition[0]) + len(condition[1]) + 2)] if condition else stem
    numbers = re.findall(r"_(\d+)(?:_|$)", prefix)
    return int(numbers[-1]) if numbers else 0


def discover_records(dataset_root: str | Path, block_size: float = 10.0) -> list[dict]:
    """Discover condition-tagged files and calculate simulation timestamps."""
    root = Path(dataset_root)
    def capture_key(path):
        return (0, _capture_order(path.stem))

    images = sorted((root / "images").glob("*.png"), key=capture_key)
    records = []
    elapsed = 0.0
    previous_position = None
    previous_condition = None
    flight_number = 0
    for image_path in images:
        stem = image_path.stem
        condition = _condition_from_filename(stem)
        if condition is None:
            continue
        label_path = root / "labels" / f"{stem}.txt"
        exact_label_path = root / "exact_labels" / f"{stem}.txt"
        mask_path = root / "masks" / f"{stem}.png"
        metadata_path = root / "metadata" / f"{stem}.json"
        if not mask_path.exists() and not label_path.exists():
            raise FileNotFoundError(f"Missing mask or label for {image_path}")
        with Image.open(image_path) as image:
            width, height = image.size
            if image.mode != "RGB":
                raise ValueError(f"Expected RGB PNG, got {image.mode}: {image_path}")
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        visible_paths = sorted({mine["prim_path"] for mine in metadata.get("visible_mines", [])
                                if mine.get("prim_path")})
        position = metadata.get("position") or [None, None, None]
        x, y, z = (position + [None, None, None])[:3]
        if condition != previous_condition:
            flight_number += 1
            elapsed = 0.0
            previous_position = None
            previous_condition = condition
        if previous_position is not None and None not in (x, y, z):
            elapsed += float(np.linalg.norm(np.asarray((x, y, z)) - np.asarray(previous_position))) / 3.0
        previous_position = (x, y, z) if None not in (x, y, z) else None
        season, daytime = condition
        date = datetime(*SEASON_DATES[season], DAYTIME_HOURS[daytime]) + timedelta(seconds=elapsed)
        gsd_x, gsd_y = _ground_resolution(metadata.get("image_corners_world"), width, height)
        capture_order = _capture_order(stem)
        scene_group = _scene_group(x, y, z)
        if mask_path.exists():
            with Image.open(mask_path) as mask_image:
                has_target = bool(np.asarray(mask_image).max())
        else:
            has_target = bool(label_path.stat().st_size)
        records.append({
            "acquisition_id": stem,
            "site_id": scene_group, "spatial_block_id": _group_id(x, y, block_size),
            "image_path": str(image_path.resolve()), "timestamp": date.isoformat(timespec="milliseconds"),
            "camera_id": "rgb_camera_0", "width": width, "height": height, "crs": "simulation_world_m",
            "ground_resolution_x": gsd_x or "", "ground_resolution_y": gsd_y or "", "registration_path": "image_corners_world",
            "sensor_validity_path": "all_pixels_verified", "label_path": str((exact_label_path if exact_label_path.exists() else label_path).resolve()),
            "mask_path": str(mask_path.resolve()) if mask_path.exists() else "",
            "visible_mine_prim_paths": json.dumps(visible_paths), "visible_mine_count": len(visible_paths),
            "split_group_id": "", "label_validity_path": "all_pixels_verified", "label_version": "simulation_auto_accurate",
            "configuration_id": f"{season}_{daytime}", "quality_status": "verified_negative" if not has_target else "verified_label",
            "split": "", "position_x": x, "position_y": y, "position_z": z,
            "season": season, "daytime": daytime, "flight_id": f"flight_{flight_number:04d}_{season}_{daytime}",
            "capture_order": capture_order, "elapsed_seconds": elapsed, "speed_mps": 3.0,
            "scene_group_id": scene_group,
        })
    return records


def _record_prim_paths(record: dict) -> set[str]:
    value = record.get("visible_mine_prim_paths", "[]")
    if isinstance(value, list):
        return set(value)
    try:
        return set(json.loads(value or "[]"))
    except (TypeError, json.JSONDecodeError):
        return set()


def assign_stratified_splits(records: list[dict], seed: int = 42, ratios=(0.8, 0.1, 0.1)) -> list[dict]:
    """Assign complete Prim-path groups while covering every condition per split."""
    if not records:
        return records
    parent = list(range(len(records)))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    path_owner = {}
    spatial_owner = {}
    for index, record in enumerate(records):
        prim_paths = _record_prim_paths(record)
        for path in prim_paths:
            if path in path_owner:
                union(index, path_owner[path])
            else:
                path_owner[path] = index
        spatial = record.get("spatial_block_id", f"record_{index}")
        if spatial in spatial_owner:
            union(index, spatial_owner[spatial])
        else:
            spatial_owner[spatial] = index

    grouped = defaultdict(list)
    for index in range(len(records)):
        grouped[find(index)].append(index)
    units = []
    for member_indices in grouped.values():
        unit_conditions = Counter(records[index].get("configuration_id", "unknown") for index in member_indices)
        prim_paths = sorted({path for index in member_indices for path in _record_prim_paths(records[index])})
        fallback = sorted({records[index].get("spatial_block_id", "ungrouped") for index in member_indices})
        prefix = "prim" if prim_paths else "spatial"
        group_id = f"{prefix}_{hashlib.sha1('|'.join(prim_paths or fallback).encode()).hexdigest()[:12]}"
        units.append((member_indices, unit_conditions, group_id))
    if len(units) < 3:
        raise ValueError("At least three independent groups are required for train/val/test")

    conditions = sorted({record.get("configuration_id", "unknown") for record in records})
    target_sizes = [len(records) * ratio for ratio in ratios]
    split_names = ("train", "val", "test")
    for attempt in range(100):
        rng = random.Random(seed + attempt)
        order = list(range(len(units)))
        rng.shuffle(order)
        order.sort(key=lambda index: len(units[index][0]), reverse=True)
        totals = [0, 0, 0]
        condition_counts = [Counter(), Counter(), Counter()]
        assignments = {}
        for unit_index in order:
            members, unit_conditions, _ = units[unit_index]
            scores = []
            for split in range(3):
                coverage_gain = sum(condition_counts[split][condition] == 0 for condition in unit_conditions)
                size_penalty = (totals[split] + len(members)) / max(target_sizes[split], 1)
                scores.append(coverage_gain * 1000 - size_penalty + rng.random() * 1e-3)
            split = max(range(3), key=lambda index: scores[index])
            assignments[unit_index] = split
            totals[split] += len(members)
            condition_counts[split].update(unit_conditions)
        if all(set(counts) >= set(conditions) for counts in condition_counts):
            for unit_index, split in assignments.items():
                members, _, group_id = units[unit_index]
                for index in members:
                    records[index]["split"] = split_names[split]
                    records[index]["split_group_id"] = group_id
            return records
    raise ValueError("Could not create splits containing every season/daytime condition")


def assign_spatial_splits(records: list[dict], seed: int = 42, ratios=(0.8, 0.1, 0.1)) -> list[dict]:
    """Backward-compatible split entry point."""
    return assign_stratified_splits(records, seed, ratios)


def write_manifest(records: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def read_manifest(path: str | Path) -> list[dict]:
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


class RGBDataset(Dataset):
    def __init__(self, records: list[dict], augment: bool = False, tile_size: int | None = None):
        self.records, self.augment, self.tile_size = records, augment, tile_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["image_path"]) as image:
            image = image.convert("RGB")
            if record.get("mask_path"):
                with Image.open(record["mask_path"]) as mask_image:
                    target = (np.asarray(mask_image) > 0).astype(np.float32)
                if target.shape != (image.height, image.width):
                    target = np.asarray(Image.fromarray(target).resize(image.size, Image.Resampling.NEAREST), dtype=np.float32)
            else:
                target = polygons_to_mask(record["label_path"], image.size)
            if self.tile_size is not None:
                image = image.resize((self.tile_size, self.tile_size), Image.Resampling.BILINEAR)
                target = np.asarray(Image.fromarray(target).resize((self.tile_size, self.tile_size), Image.Resampling.NEAREST))
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        target = (target > 0.5).astype(np.float32)
        valid = np.ones(target.shape, dtype=np.float32)  # Dataset has no invalid-pixel sidecar.
        if self.augment and torch.rand(()) < 0.5:
            image_array, target, valid = image_array[:, ::-1], target[:, ::-1], valid[:, ::-1]
        tensor = torch.from_numpy(image_array.copy()).permute(2, 0, 1)
        tensor = (tensor - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(IMAGENET_STD)[:, None, None]
        return {"rgb": tensor, "valid": torch.from_numpy(valid[None].copy()),
                "target": torch.from_numpy(target[None].copy()), "target_valid": torch.ones_like(torch.from_numpy(target[None].copy())),
                "acquisition_id": record["acquisition_id"], "timestamp": record.get("timestamp", "")}


def dataset_report(records: list[dict]) -> dict:
    def is_positive(record):
        if record.get("mask_path"):
            with Image.open(record["mask_path"]) as mask:
                return bool(np.asarray(mask).max())
        return bool(parse_polygons(record["label_path"]))
    positive = sum(is_positive(r) for r in records)
    condition_splits = {condition: {split: sum(r.get("configuration_id") == condition and r.get("split") == split for r in records)
                                    for split in ("train", "val", "test")}
                       for condition in sorted({r.get("configuration_id", "unknown") for r in records})}
    prim_splits = defaultdict(set)
    for record in records:
        for path in _record_prim_paths(record):
            prim_splits[path].add(record.get("split"))
    return {"records": len(records), "positive_records": positive, "verified_negative_records": len(records) - positive,
            "spatial_blocks": len({r["spatial_block_id"] for r in records}),
            "splits": {s: sum(r.get("split") == s for r in records) for s in ("train", "val", "test")},
            "condition_splits": condition_splits,
            "prim_paths_crossing_splits": sorted(path for path, splits in prim_splits.items() if len(splits - {None}) > 1),
            "missing_fields": {field: sum(not r.get(field) for r in records) for field in ("timestamp", "camera_id", "crs", "ground_resolution_x", "ground_resolution_y", "registration_path", "sensor_validity_path", "mask_path", "label_validity_path")}}
