import pytest
import torch

from phase2_rgb_thermal_image_utae.models import AsyncUTAESegmenter
from phase2_rgb_thermal_image_utae.data import collate_async_views
from phase2_rgb_thermal_image_utae.manifest import legacy_modality_ages


@pytest.mark.parametrize("mode", ["rgb", "thermal", "both"])
def test_singleton_modalities(mode):
    model = AsyncUTAESegmenter().eval()
    value = torch.randn(1, 1, 3, 64, 64)
    present = torch.ones(1, 1, dtype=torch.bool)
    rgb = value if mode in ("rgb", "both") else None
    thermal = value if mode in ("thermal", "both") else None
    with torch.no_grad():
        logits, details = model(rgb, thermal, present if rgb is not None else None,
                                present if thermal is not None else None, return_details=True)
    assert logits.shape == (1, 1, 64, 64)
    assert details["output_valid"][8].any()


def test_unequal_histories_and_missing_modality():
    model = AsyncUTAESegmenter().eval()
    rgb = torch.randn(2, 2, 3, 64, 64)
    thermal = torch.randn(2, 3, 3, 64, 64)
    rgb_valid = torch.tensor([[True, False], [True, True]])
    thermal_valid = torch.tensor([[False, False, False], [True, True, True]])
    with torch.no_grad():
        logits, details = model(rgb, thermal, rgb_valid, thermal_valid, return_details=True)
    assert logits.shape == (2, 1, 64, 64)
    assert torch.isfinite(logits).all()
    assert details["output_valid"][8][0].any()


def test_no_observation_is_rejected():
    with pytest.raises(ValueError):
        AsyncUTAESegmenter()(None, None)


def test_independent_ragged_collation():
    items = [
        {"rgb": torch.randn(1, 3, 8, 8), "thermal": None, "target": torch.zeros(1, 8, 8)},
        {"rgb": None, "thermal": torch.randn(3, 3, 8, 8), "target": torch.zeros(1, 8, 8)},
    ]
    batch = collate_async_views(items)
    assert batch["rgb"].shape[1] == 1
    assert batch["thermal"].shape[1] == 3
    assert batch["rgb_frame_valid"].tolist() == [[True], [False]]
    assert batch["thermal_frame_valid"].tolist() == [[False, False, False], [True, True, True]]


def test_derived_modality_age_uses_latest_available_condition():
    seasons = torch.tensor([[0, 1, 2]])
    daytimes = torch.tensor([[0, 0, 0]])
    rgb = torch.tensor([[True, False, False]])
    thermal = torch.tensor([[False, False, True]])
    rgb_age, thermal_age = legacy_modality_ages(seasons, daytimes, rgb, thermal)
    assert rgb_age.item() > 0
    assert thermal_age.item() == 0
