"""Feature-cache creation and validation for Phase 1A RGB encoders."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from phase1a_rgb_encoder.models import RGBEncoder
from .point_dataset import CONDITIONS, discover_point_records, read_temporal_manifest


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_image(path, encoder, tile_size, device):
    with Image.open(path) as image:
        image = image.convert("RGB").resize((tile_size, tile_size), Image.Resampling.BILINEAR)
        rgb = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
    rgb = (rgb - mean) / std
    valid = torch.ones(1, tile_size, tile_size)
    with torch.inference_mode():
        output = encoder(torch.cat((rgb, valid), dim=0)[None].to(device), valid[None].to(device))
    return {scale: value[0].cpu().float() for scale, value in output.features.items()}, {scale: value[0].cpu() for scale, value in output.valid.items()}


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _source_frames(dataset_root=None, manifest_path=None):
    if dataset_root:
        return discover_point_records(dataset_root)
    samples = read_temporal_manifest(manifest_path)
    frames = {}
    for sample in samples:
        for index, path in enumerate(sample["image_paths"]):
            frames.setdefault(path, {"frame_id": sample["frame_ids"][index], "image_path": path,
                                     "mine_id": sample["mine_id"], "point_number": sample["point_number"],
                                     "condition": sample["conditions"][index]})
    return list(frames.values())


def build_feature_cache(dataset_root=None, manifest_path=None, encoder_checkpoint=None,
                        output_dir=None, tile_size=256, device="cpu"):
    if not dataset_root and not manifest_path:
        raise ValueError("dataset_root or manifest_path is required")
    if not encoder_checkpoint or not output_dir:
        raise ValueError("encoder_checkpoint and output_dir are required")
    frames = _source_frames(dataset_root, manifest_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    encoder = RGBEncoder.from_pretrained(encoder_checkpoint, map_location=device).to(device).eval()
    frame_records = sorted(frames, key=lambda record: record["frame_id"])
    records = []
    for index, frame in enumerate(frame_records):
        features, valid = encode_image(frame["image_path"], encoder, tile_size, device)
        cache_path = output / f"{index:06d}.pt"
        torch.save({"features": features, "feature_validity": valid, "frame_id": frame["frame_id"],
                    "frame_path": frame["image_path"]}, cache_path)
        records.append({**frame, "cache_path": str(cache_path.resolve())})
    source_descriptor = json.dumps([{key: frame.get(key, "") for key in ("frame_id", "image_path", "mine_id", "point_number", "condition")}
                                    for frame in frame_records], sort_keys=True).encode()
    metadata = {"dataset_root": str(Path(dataset_root).resolve()) if dataset_root else "",
                "source_manifest": str(Path(manifest_path).resolve()) if manifest_path else "",
                "source_frame_count": len(frame_records),
                "source_manifest_sha256": hashlib.sha256(source_descriptor).hexdigest(),
                "encoder_checkpoint": str(Path(encoder_checkpoint).resolve()),
                "encoder_checkpoint_sha256": file_hash(encoder_checkpoint),
                "preprocessing": {"tile_size": tile_size, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                "feature_contract": {"1/2": [32, tile_size // 2, tile_size // 2],
                                     "1/4": [64, tile_size // 4, tile_size // 4],
                                     "1/8": [128, tile_size // 8, tile_size // 8]},
                "dtype": "float32", "conditions": list(CONDITIONS),
                "feature_validity_rule": "Phase1A average-pool coverage >= validity_threshold",
                "git_commit": _git_commit(), "created_at": datetime.now(timezone.utc).isoformat()}
    (output / "cache_manifest.json").write_text(json.dumps(records, indent=2))
    (output / "cache_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def validate_feature_cache(cache_dir):
    root = Path(cache_dir)
    manifest = json.loads((root / "cache_manifest.json").read_text())
    metadata = json.loads((root / "cache_metadata.json").read_text())
    errors = []
    expected = {"1/2": tuple(metadata["feature_contract"]["1/2"]),
                "1/4": tuple(metadata["feature_contract"]["1/4"]),
                "1/8": tuple(metadata["feature_contract"]["1/8"])}
    for record in manifest:
        path = Path(record["cache_path"])
        if not path.exists():
            errors.append(f"missing cache file: {path}")
            continue
        item = torch.load(path, map_location="cpu", weights_only=False)
        for scale, shape in expected.items():
            value = item.get("features", {}).get(scale)
            if value is None or tuple(value.shape) != shape or value.dtype != torch.float32:
                errors.append(f"invalid {scale} in {path}")
                continue
            if not torch.isfinite(value).all():
                errors.append(f"non-finite {scale} in {path}")
        if item.get("frame_id") != record.get("frame_id"):
            errors.append(f"frame identity mismatch in {path}")
    report = {"records": len(manifest), "errors": errors, "valid": not errors, "metadata": metadata}
    (root / "cache_validation.json").write_text(json.dumps(report, indent=2))
    if errors:
        raise ValueError("Feature cache validation failed: " + "; ".join(errors[:5]))
    return report


def attach_cache_paths(samples, cache_dir):
    """Attach cached frame paths to point samples without re-encoding frames."""
    root = Path(cache_dir)
    records = json.loads((root / "cache_manifest.json").read_text())
    by_frame = {record["frame_id"]: record["cache_path"] for record in records}
    attached = []
    for sample in samples:
        missing = [frame_id for frame_id in sample["frame_ids"] if frame_id not in by_frame]
        if missing:
            raise ValueError(f"cache is missing frames for {sample['sequence_id']}: {missing[:3]}")
        value = dict(sample)
        value["cache_paths"] = [by_frame[frame_id] for frame_id in sample["frame_ids"]]
        attached.append(value)
    return attached
