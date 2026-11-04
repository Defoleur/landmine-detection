#!/usr/bin/env python3
"""Merge a full-map tile dataset into a world-coordinate mosaic.

The output orientation matches ``map/map_interface.py``: world X increases
downward and world Y increases to the right.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "datasets" / "dataset_map_20261005_134007"
DEFAULT_OUTPUT = ROOT / "map" / "images_merged"
CORNER_KEYS = ("top_left", "top_right", "bottom_right", "bottom_left")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--season", default="summer")
    parser.add_argument("--daytime", default="day")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolution", type=float, default=0.02,
                        help="metres per output pixel (default: 0.02)")
    parser.add_argument("--no-feather", action="store_true",
                        help="use binary tile coverage instead of feathered blending")
    parser.add_argument("--overlap-mode", choices=("blend", "overwrite"), default="blend",
                        help="blend overlaps, or let each later image overwrite them")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate inputs and report output size without merging")
    return parser.parse_args(argv)


def load_inputs(dataset: Path, season: str, daytime: str):
    manifest_path = dataset / "map_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open() as stream:
        manifest = json.load(stream)

    captures = [
        capture for capture in manifest.get("captures", [])
        if capture.get("season") == season and capture.get("daytime") == daytime
    ]
    if not captures:
        raise ValueError(f"No captures found for {season}_{daytime}")
    return manifest, captures


def map_geometry(manifest: dict, resolution: float):
    bounds = manifest["map_bounds_xy"]
    min_x, min_y = map(float, bounds[0])
    max_x, max_y = map(float, bounds[1])
    width = int(np.ceil((max_y - min_y) / resolution)) + 1
    height = int(np.ceil((max_x - min_x) / resolution)) + 1
    return min_x, min_y, max_x, max_y, width, height


def world_to_pixel(points, min_x, min_y, resolution):
    points = np.asarray(points, dtype=np.float32)
    # Match MapCanvas: image columns represent world Y, rows represent X.
    return np.column_stack([
        (points[:, 1] - min_y) / resolution,
        (points[:, 0] - min_x) / resolution,
    ]).astype(np.float32)


def capture_corners(capture: dict, metadata: dict, tile_size: float):
    corners = metadata.get("image_corners_world")
    if not isinstance(corners, dict):
        corners = capture.get("image_corners_world")
    if isinstance(corners, dict) and all(
        key in corners and isinstance(corners[key], (list, tuple)) and len(corners[key]) >= 2
        for key in CORNER_KEYS
    ):
        points = np.asarray([corners[key][:2] for key in CORNER_KEYS], dtype=np.float32)
        if points.shape == (4, 2) and np.isfinite(points).all():
            return points

    position = capture.get("position") or metadata.get("position")
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        raise ValueError("missing image_corners_world and position")
    center_x, center_y = map(float, position[:2])
    half_size = tile_size / 2.0
    # Fall back to the nominal grid tile and crop it to the output bounds later.
    return np.asarray([
        [center_x + half_size, center_y - half_size],
        [center_x + half_size, center_y + half_size],
        [center_x - half_size, center_y + half_size],
        [center_x - half_size, center_y - half_size],
    ], dtype=np.float32)


def feather_mask(height: int, width: int, enabled: bool):
    if not enabled:
        return np.ones((height, width), dtype=np.float32)
    raw = cv2.distanceTransform(
        np.full((height, width), 255, dtype=np.uint8), cv2.DIST_L2, 5
    )
    edge = max(min(height, width) * 0.08, 1.0)
    return np.clip(raw / edge, 0.05, 1.0).astype(np.float32)


def merge(manifest: dict, captures: list[dict], dataset: Path, resolution: float,
          feather: bool, output_dir: Path, season: str, daytime: str,
          overlap_mode: str = "blend"):
    if overlap_mode not in ("blend", "overwrite"):
        raise ValueError(f"unsupported overlap mode: {overlap_mode}")

    min_x, min_y, max_x, max_y, width, height = map_geometry(manifest, resolution)
    print(f"Condition: {season}_{daytime}")
    print(f"Captures: {len(captures)}")
    print(f"Output: {width} x {height} px ({width * height * 3 / 1024**2:.1f} MiB RGB)")
    print(f"World bounds: X=[{min_x}, {max_x}], Y=[{min_y}, {max_y}]")

    mosaic = np.zeros((height, width, 3), dtype=np.uint8)
    weights = np.zeros((height, width), dtype=np.float32)
    processed = 0
    skipped = 0
    image_shape = None
    source_mask = None

    for capture in captures:
        image_path = dataset / capture["image_file"]
        metadata_path = dataset / capture["metadata_file"]
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot read image: {image_path}")
            with metadata_path.open() as stream:
                metadata = json.load(stream)
            corners = capture_corners(capture, metadata, float(manifest["tile_size_m"]))
            if image_shape != image.shape[:2]:
                image_shape = image.shape[:2]
                source_mask = feather_mask(*image_shape, feather)

            destination = world_to_pixel(corners, min_x, min_y, resolution)
            x0 = max(0, int(np.floor(destination[:, 0].min())) - 1)
            y0 = max(0, int(np.floor(destination[:, 1].min())) - 1)
            x1 = min(width, int(np.ceil(destination[:, 0].max())) + 2)
            y1 = min(height, int(np.ceil(destination[:, 1].max())) + 2)
            if x1 <= x0 or y1 <= y0:
                raise ValueError("corners do not intersect output bounds")

            source = np.float32([
                [0, 0],
                [image.shape[1] - 1, 0],
                [image.shape[1] - 1, image.shape[0] - 1],
                [0, image.shape[0] - 1],
            ])
            local_destination = destination - np.array([x0, y0], dtype=np.float32)
            transform = cv2.getPerspectiveTransform(source, local_destination)
            size = (x1 - x0, y1 - y0)
            warped = cv2.warpPerspective(image, transform, size)
            warped_weight = cv2.warpPerspective(source_mask, transform, size)
            valid = warped_weight > 0
            if not valid.any():
                raise ValueError("warped image has no valid pixels")

            mosaic_region = mosaic[y0:y1, x0:x1]
            weight_region = weights[y0:y1, x0:x1]
            if overlap_mode == "overwrite":
                mosaic_region[valid] = warped[valid]
                weight_region[valid] = 1.0
            else:
                total_weight = weight_region + warped_weight
                update = valid & (total_weight > 0)
                blended = (
                    mosaic_region.astype(np.float32) * weight_region[:, :, None]
                    + warped.astype(np.float32) * warped_weight[:, :, None]
                ) / np.maximum(total_weight[:, :, None], 1e-6)
                mosaic_region[update] = np.clip(blended[update], 0, 255).astype(np.uint8)
                weight_region[valid] = total_weight[valid]
            processed += 1
        except (KeyError, OSError, TypeError, ValueError) as error:
            skipped += 1
            print(f"\nSkipping {capture.get('image_file', '<unknown>')}: {error}")

        if processed % 100 == 0 and processed:
            print(f"\rMerged {processed}/{len(captures)}", end="", flush=True)

    print(f"\rMerged {processed}/{len(captures)}; skipped {skipped}")
    if processed == 0:
        raise RuntimeError("No captures were merged")

    output_dir.mkdir(parents=True, exist_ok=True)
    condition = f"{season}_{daytime}"
    image_output = output_dir / f"{condition}.png"
    metadata_output = output_dir / f"{condition}.json"
    if not cv2.imwrite(str(image_output), mosaic):
        raise OSError(f"Could not write {image_output}")
    metadata = {
        "source_dataset": str(dataset),
        "condition": condition,
        "projection": "world_xy",
        "orientation": {
            "columns": "world_y_increasing_right",
            "rows": "world_x_increasing_down",
        },
        "resolution_m_per_pixel": resolution,
        "world_bounds_xy": [[min_x, min_y], [max_x, max_y]],
        "image_size": [width, height],
        "captures_total": len(captures),
        "captures_merged": processed,
        "captures_skipped": skipped,
        "feathered_blending": feather,
        "overlap_mode": overlap_mode,
    }
    with metadata_output.open("w") as stream:
        json.dump(metadata, stream, indent=2)
    print(f"Saved {image_output}")
    print(f"Saved {metadata_output}")


def main(argv=None):
    args = parse_args(argv)
    if args.resolution <= 0:
        raise SystemExit("--resolution must be positive")
    manifest, captures = load_inputs(args.dataset, args.season, args.daytime)
    geometry = map_geometry(manifest, args.resolution)
    print(f"Validated {len(captures)} captures; output size {geometry[4]} x {geometry[5]} px")
    if args.dry_run:
        return
    merge(manifest, captures, args.dataset, args.resolution, not args.no_feather,
          args.output_dir, args.season, args.daytime, args.overlap_mode)


if __name__ == "__main__":
    main()
