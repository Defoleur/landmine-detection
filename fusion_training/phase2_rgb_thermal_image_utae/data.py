"""Ragged asynchronous RGB/thermal batch collation."""

from __future__ import annotations

import torch


def collate_async_views(items: list[dict]) -> dict:
    """Pad RGB and thermal histories independently and preserve validity masks."""
    if not items:
        raise ValueError("cannot collate an empty batch")
    batch = len(items)
    height, width = items[0]["target"].shape[-2:]
    result = {"target": torch.stack([item["target"] for item in items])}
    if "label_valid" in items[0]:
        result["label_valid"] = torch.stack([item["label_valid"] for item in items])
    for modality in ("rgb", "thermal"):
        observations = [item.get(modality) for item in items]
        channels = next((value.shape[1] for value in observations if value is not None), 3)
        max_steps = max([value.shape[0] for value in observations if value is not None] or [0])
        max_steps = max(1, max_steps)
        images = torch.zeros(batch, max_steps, channels, height, width, dtype=result["target"].dtype)
        frame_valid = torch.zeros(batch, max_steps, dtype=torch.bool)
        time_features = torch.zeros(batch, max_steps, 6, dtype=result["target"].dtype)
        ages = torch.zeros(batch, 1, *result["target"].shape[-2:], dtype=result["target"].dtype)
        for index, value in enumerate(observations):
            if value is not None:
                steps = value.shape[0]
                images[index, :steps] = value
                frame_valid[index, :steps] = True
                if items[index].get(f"{modality}_time_features") is not None:
                    time_features[index, :steps] = items[index][f"{modality}_time_features"]
                if items[index].get(f"{modality}_age") is not None:
                    ages[index] = items[index][f"{modality}_age"]
        result[modality] = images
        result[f"{modality}_frame_valid"] = frame_valid
        result[f"{modality}_time_features"] = time_features
        result[f"{modality}_ages"] = ages
    return result
