import torch
from phase1d_thermal_image_convlstm.sequence import CONDITIONS, discover_point_frames
from phase1d_thermal_image_convlstm.temporal_models import (
    ThermalImageTemporalSegmentationModel,
)


def test_image_temporal_contract():
    model = ThermalImageTemporalSegmentationModel(use_condition_present=True)
    logits, details = model(
        torch.randn(1, len(CONDITIONS), 3, 128, 128),
        torch.ones(1, 12, dtype=torch.bool),
    )
    assert logits.shape == (1, 1, 128, 128)
    assert details["features"]["1/8"].shape == (1, 128, 16, 16)


def test_fixed_conditions():
    assert len(CONDITIONS) == 12


def test_processed_images_are_sequence_frames(tmp_path):
    dataset_root = tmp_path / "dataset"
    processed_root = tmp_path / "processed" / "images"
    (dataset_root / "metadata").mkdir(parents=True)
    processed_root.mkdir(parents=True)

    stem = "dataset_000001_10.0_20.0_p03_obstacle_mine_tm62_summer_day"
    (processed_root / f"{stem}.png").touch()
    (dataset_root / "metadata" / f"{stem}.json").write_text(
        '{"point_index": 3, "points_per_mine": 10, "trajectory": "t"}\n'
    )

    frames = discover_point_frames(dataset_root, tmp_path / "processed")

    assert len(frames) == 1
    assert frames[0]["mine_id"] == "tm62"
    assert frames[0]["point_number"] == 3
    assert frames[0]["condition"] == "summer_day"
    assert frames[0]["image_path"].endswith(f"{stem}.png")
