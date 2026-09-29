import json

import numpy as np
import pytest
import torch
from PIL import Image

from phase2_rgb_thermal_image_fusion.data import (
    ImageFusionSequenceDataset,
    RGB_MEAN,
    RGB_STD,
)
from phase2_rgb_thermal_image_fusion.models import ImageFusionSegmenter
from phase2_rgb_thermal_image_fusion.scripts.qa_image_fusion_manifest import validate


def _sequence(tmp_path):
    frames = []
    conditions = [
        f"{season}_{daytime}"
        for season in ("winter", "spring", "summer", "autumn")
        for daytime in ("morning", "day", "evening")
    ]
    for index, condition in enumerate(conditions):
        rgb = tmp_path / f"rgb_{index}.png"
        thermal = tmp_path / f"thermal_{index}.png"
        target = tmp_path / f"target_{index}.png"
        Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8)).save(rgb)
        Image.fromarray(np.full((16, 16, 3), 64, dtype=np.uint8)).save(thermal)
        Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(target)
        frames.append(
            {
                "condition": condition,
                "rgb_path": str(rgb),
                "thermal_image_path": str(thermal),
                "metadata_path": str(target),
            }
        )
    return {
            "sequence_id": "mine_p01",
            "split": "test",
            "frames": frames,
            "target_path": str(tmp_path / "target_0.png"),
        }


def test_dataset_shapes_and_rgb_normalization(tmp_path):
    sample = _sequence(tmp_path)
    item = ImageFusionSequenceDataset([sample], 16)[0]
    assert item["rgb_sequence"].shape == (12, 3, 16, 16)
    assert item["thermal_image"].shape == (12, 3, 16, 16)
    assert item["target"].shape == (1, 16, 16)
    assert torch.allclose(
        item["rgb_sequence"][:, 0].mean(),
        torch.tensor((128 / 255 - RGB_MEAN[0]) / RGB_STD[0]),
        atol=1e-4,
    )
    assert torch.allclose(
        item["thermal_image"][:, 0].mean(),
        torch.tensor((64 / 255 - RGB_MEAN[0]) / RGB_STD[0]),
        atol=1e-4,
    )
    assert item["rgb_available"].all() and item["thermal_available"].all()


def test_manifest_qa(tmp_path):
    result = validate([_sequence(tmp_path)])
    assert result["valid"]
    assert result["splits"]["test"] == 1


@pytest.mark.parametrize("mode", ["full", "rgb", "thermal"])
def test_model_modalities(mode):
    model = ImageFusionSegmenter()
    values = torch.randn(1, 12, 3, 64, 64)
    availability = torch.ones(1, 12, dtype=torch.bool)
    rgb = values if mode != "thermal" else None
    thermal = values if mode != "rgb" else None
    logits, details = model(rgb, thermal, availability, availability)
    assert logits.shape == (1, 1, 64, 64)
    assert set(details["features"]) == {"1/2", "1/4", "1/8"}
