"""Image manifests and datasets backed by the split_1 thermal PNGs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from phase1c_thermal_encoder.data import (
    assign_condition_splits,
    assign_splits,
    filter_condition_records,
)

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def _group_id(path: Path, metadata: dict) -> str:
    mine_names = sorted(
        item.get("sim_name", "")
        for item in metadata.get("visible_mines", [])
        if item.get("sim_name")
    )
    if mine_names:
        return "mine:" + ",".join(mine_names)
    position = metadata.get("position") or []
    if len(position) >= 3:
        return f"background:{path.name.split('_')[0]}:{float(position[0]):.2f}:{float(position[1]):.2f}:{float(position[2]):.2f}"
    return "frame:" + path.stem.rsplit("_", 1)[0]


def discover_image_records(
    dataset_root: str | Path, processed_root: str | Path
) -> list[dict]:
    """Discover image records from processed PNGs without reading raw NPZ files."""
    dataset_root = Path(dataset_root).resolve()
    image_root = Path(processed_root).resolve() / "images"
    records = []
    for image_path in sorted(image_root.glob("*.png")):
        sample_id = image_path.stem
        target_path = dataset_root / "masks" / f"{sample_id}.png"
        metadata_path = dataset_root / "metadata" / f"{sample_id}.json"
        if not target_path.exists():
            raise FileNotFoundError(f"Missing target mask for {image_path}")
        metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        )
        with Image.open(target_path) as target:
            positive = bool(np.asarray(target.convert("L")).max())
        records.append(
            {
                "sample_id": sample_id,
                "target_path": str(target_path),
                "metadata_path": str(metadata_path),
                "group_id": _group_id(image_path, metadata),
                "positive": positive,
                "season": metadata.get("season"),
                "daytime": metadata.get("daytime"),
                "configuration_id": f"{metadata.get('season', 'unknown')}_{metadata.get('daytime', 'unknown')}",
            }
        )
    if not records:
        raise ValueError(f"No processed PNG samples found under {image_root}")
    return records


def add_image_paths(records, processed_root: str | Path, require=True):
    processed_root = Path(processed_root).resolve()
    image_root = processed_root / "images"
    result = []
    missing = []
    for record in records:
        image_path = image_root / f"{record['sample_id']}.png"
        if not image_path.exists():
            missing.append(str(image_path))
        updated = dict(record)
        updated["image_path"] = str(image_path)
        result.append(updated)
    if require and missing:
        raise FileNotFoundError(
            f"Missing processed images: {missing[:3]} (and {len(missing) - min(3, len(missing))} more)"
        )
    return result


def prepare_records(
    dataset_root, processed_root, seed=42, mode="prim", mine_type_field="exact_class"
):
    records = discover_image_records(dataset_root, processed_root)
    if mode == "condition":
        records, diagnostics = filter_condition_records(records)
        records, split_diagnostics = assign_condition_splits(
            records, seed, mine_type_field=mine_type_field
        )
        diagnostics.update(split_diagnostics)
    else:
        records = assign_splits(records, seed)
        diagnostics = {}
    return add_image_paths(records, processed_root), diagnostics


def write_manifest(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n"
    )


def read_manifest(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def _load_mask(path, size):
    with Image.open(path) as mask:
        target = mask.convert("L").resize((size, size), Image.Resampling.NEAREST)
    return (np.asarray(target, dtype=np.float32) > 0).astype(np.float32)


class ImageSegmentationDataset(Dataset):
    def __init__(self, records, tile_size=640, augment=False):
        self.records = records
        self.tile_size = int(tile_size)
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["image_path"]) as source:
            image = source.convert("RGB").resize(
                (self.tile_size, self.tile_size), Image.Resampling.BILINEAR
            )
        image = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        target = _load_mask(record["target_path"], self.tile_size)[None]
        if self.augment and torch.rand(()) < 0.5:
            image = image[:, :, ::-1].copy()
            target = target[:, ::-1].copy()
        mean = np.asarray(IMAGE_MEAN, dtype=np.float32)[:, None, None]
        std = np.asarray(IMAGE_STD, dtype=np.float32)[:, None, None]
        image = (image - mean) / std
        return {
            "image": torch.from_numpy(image.copy()),
            "target": torch.from_numpy(target.copy()),
            "target_validity": torch.ones(
                (1, self.tile_size, self.tile_size), dtype=torch.float32
            ),
            "metadata": {
                "sample_id": record["sample_id"],
                "group_id": record["group_id"],
                "split": record.get("split", ""),
                "image_path": record["image_path"],
            },
        }
