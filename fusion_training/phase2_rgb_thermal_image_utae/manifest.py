"""Manifest conversion and deterministic timestamp metadata for U-TAE."""

from __future__ import annotations

from datetime import datetime, timezone

import torch

from phase1b_rgb_utae.time_encoding import condition_timestamp, condition_time_features


def add_derived_observation_metadata(sequences: list[dict], year: int = 2026) -> list[dict]:
    """Annotate legacy paired sequences with asynchronous observation records."""
    reference = datetime(year, 1, 1, tzinfo=timezone.utc)
    converted = []
    for sequence in sequences:
        frames = sequence.get("frames")
        if frames is None:
            conditions = sequence.get("conditions", [])
            frame_ids = sequence.get("frame_ids", [])
            image_paths = sequence.get("image_paths", [])
            mask_paths = sequence.get("mask_paths", [None] * len(conditions))
            label_paths = sequence.get("label_paths", [None] * len(conditions))
            if not (len(conditions) == len(frame_ids) == len(image_paths)):
                raise ValueError(
                    f"{sequence.get('sequence_id', '<unknown>')}: conditions, frame_ids, and image_paths must have equal lengths"
                )
            frames = [
                {
                    "condition": condition,
                    "frame_id": frame_id,
                    "rgb_path": image_path,
                    "thermal_image_path": image_path,
                    "image_path": image_path,
                    "original_mask_path": mask_paths[index],
                    "label_path": label_paths[index],
                }
                for index, (condition, frame_id, image_path) in enumerate(zip(conditions, frame_ids, image_paths))
            ]
        observations = []
        for condition_index, frame in enumerate(frames):
            condition = frame["condition"]
            timestamp = condition_timestamp(condition, year)
            for modality, path_key in (("rgb", "rgb_path"), ("thermal", "thermal_image_path")):
                observations.append({
                    "observation_id": f"{frame.get('frame_id', condition)}:{modality}",
                    "modality": modality,
                    "acquired_at_utc": timestamp.isoformat().replace("+00:00", "Z"),
                    "available_at_utc": timestamp.isoformat().replace("+00:00", "Z"),
                    "arrival_index": condition_index * 2 + (0 if modality == "rgb" else 1),
                    "grid_id": sequence.get("grid_id", "legacy_shared_grid"),
                    "source_path": frame.get(path_key),
                    "coverage_path": frame.get("coverage_path"),
                    "preprocessing_version": frame.get("preprocessing_version", "legacy_rendered_image"),
                    "timestamp_source": "derived_from_condition",
                })
        result = dict(sequence)
        result["frames"] = frames
        result["observations"] = observations
        result["fixed_time_reference_utc"] = reference.isoformat().replace("+00:00", "Z")
        result["synthetic_time_year"] = year
        converted.append(result)
    return converted


def sequence_time_features(conditions: list[str], year: int = 2026) -> tuple:
    """Return one [T,6] timestamp-feature tensor and its fixed reference."""
    reference = datetime(year, 1, 1, tzinfo=timezone.utc)
    features = [condition_time_features(condition, year, reference) for condition in conditions]
    return torch.stack(features), reference


def id_time_features(season_ids: torch.Tensor, daytime_ids: torch.Tensor, year: int = 2026) -> torch.Tensor:
    """Convert legacy season/daytime tensors to the seven-channel one-hot contract."""
    seasons = ("winter", "spring", "summer", "autumn")
    daytimes = ("morning", "day", "evening")
    flat = [
        condition_time_features(f"{seasons[int(season)]}_{daytimes[int(daytime)]}", year)
        for season, daytime in zip(season_ids.reshape(-1), daytime_ids.reshape(-1))
    ]
    return torch.stack(flat).reshape(*season_ids.shape, 7)


def legacy_modality_ages(season_ids: torch.Tensor, daytime_ids: torch.Tensor,
                         rgb_valid: torch.Tensor, thermal_valid: torch.Tensor,
                         year: int = 2026) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-view latest-observation ages from legacy condition IDs."""
    seasons = ("winter", "spring", "summer", "autumn")
    daytimes = ("morning", "day", "evening")
    batch, steps = season_ids.shape
    timestamps = torch.tensor([
        condition_timestamp(f"{seasons[int(season)]}_{daytimes[int(daytime)]}", year).timestamp()
        for season, daytime in zip(season_ids.reshape(-1), daytime_ids.reshape(-1))
    ], dtype=torch.float64, device=season_ids.device).reshape(batch, steps)
    cutoff = torch.where(rgb_valid, timestamps, torch.full_like(timestamps, -float("inf"))).max(dim=1).values
    thermal_cutoff = torch.where(thermal_valid, timestamps, torch.full_like(timestamps, -float("inf"))).max(dim=1).values
    cutoff = torch.maximum(cutoff, thermal_cutoff)
    def age(valid):
        latest = torch.where(valid, timestamps, torch.full_like(timestamps, -float("inf"))).max(dim=1).values
        return (cutoff - latest).clamp_min(0).nan_to_num(0).float().div(86400).unsqueeze(1)
    return age(rgb_valid), age(thermal_valid)
