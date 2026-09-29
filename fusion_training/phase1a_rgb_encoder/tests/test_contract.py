import json

import numpy as np
import torch
from PIL import Image

from phase1a_rgb_encoder.data import (RGBDataset, assign_spatial_splits, discover_records,
                                      polygons_to_mask)
from phase1a_rgb_encoder.losses import masked_bce_with_logits
from phase1a_rgb_encoder.metrics import SegmentationMetrics
from phase1a_rgb_encoder.models import RGBEncoder, RGBSegmentationModel


def test_encoder_feature_contract():
    model = RGBEncoder(pretrained=False).eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 256, 256), torch.ones(2, 1, 256, 256))
    assert {k: tuple(v.shape) for k, v in output.features.items()} == {
        "1/2": (2, 32, 128, 128), "1/4": (2, 64, 64, 64), "1/8": (2, 128, 32, 32)}
    assert all(v.dtype == torch.bool for v in output.valid.values())


def test_model_and_masked_loss_are_trainable():
    model = RGBSegmentationModel(pretrained=False)
    logits, _ = model(torch.randn(1, 3, 64, 64), torch.ones(1, 1, 64, 64))
    loss = masked_bce_with_logits(logits, torch.zeros_like(logits), torch.ones_like(logits), torch.ones_like(logits))
    loss.backward()
    assert torch.isfinite(loss)


def test_segmentation_metrics_perfect_prediction():
    metrics = SegmentationMetrics()
    logits = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]])
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    metrics.update(logits, target, torch.ones_like(target))
    result = metrics.compute()
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["dice"] == 1.0
    assert result["iou"] == 1.0


def test_empty_polygon_is_verified_negative(tmp_path):
    root = tmp_path / "dataset"
    for sub in ("images", "labels", "metadata"):
        (root / sub).mkdir(parents=True)
    image = root / "images" / "sample_000000_summer_day.png"
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8), "RGB").save(image)
    (root / "labels" / "sample_000000_summer_day.txt").write_text("")
    (root / "metadata" / "sample_000000_summer_day.json").write_text(json.dumps({"position": [1, 2, 3]}))
    records = discover_records(root)
    assert records[0]["quality_status"] == "verified_negative"
    assert polygons_to_mask(records[0]["label_path"], (16, 16)).sum() == 0


def test_spatial_split_keeps_groups_together():
    records = [{"spatial_block_id": f"g{i}", "split": ""} for i in range(6)]
    records += [{"spatial_block_id": "g0", "split": ""}]
    result = assign_spatial_splits(records, seed=42)
    splits = {r["spatial_block_id"]: r["split"] for r in result}
    assert len(splits) == 6
    assert all(r["split"] == splits[r["spatial_block_id"]] for r in result)


def test_filename_condition_filter_and_timestamp(tmp_path):
    root = tmp_path / "dataset"
    for sub in ("images", "labels", "metadata"):
        (root / sub).mkdir(parents=True)
    for index, x in enumerate((0.0, 3.0)):
        stem = f"sample_{index:06d}_summer_day"
        Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8), "RGB").save(root / "images" / f"{stem}.png")
        (root / "labels" / f"{stem}.txt").write_text("")
        (root / "metadata" / f"{stem}.json").write_text(json.dumps({"position": [x, 0, 0], "image_corners_world": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]}))
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8), "RGB").save(root / "images" / "sample_999999.png")
    records = discover_records(root)
    assert len(records) == 2
    assert records[0]["timestamp"].startswith("2026-07-01T14:00:00")
    assert records[1]["elapsed_seconds"] == 1.0
    assert records[0]["camera_id"] == "rgb_camera_0"
    assert records[0]["ground_resolution_x"] == 1 / 9


def test_direct_mask_is_used_as_target(tmp_path):
    root = tmp_path / "dataset"
    for sub in ("images", "masks", "metadata"):
        (root / sub).mkdir(parents=True)
    stem = "sample_000000_summer_day"
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8), "RGB").save(root / "images" / f"{stem}.png")
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 2:4] = 7
    Image.fromarray(mask).save(root / "masks" / f"{stem}.png")
    (root / "metadata" / f"{stem}.json").write_text(json.dumps({"position": [1, 2, 3], "visible_mines": []}))
    records = discover_records(root)
    sample = RGBDataset(records, tile_size=4)[0]
    assert sample["target"].sum() == 4
    assert records[0]["mask_path"].endswith(f"masks/{stem}.png")


def test_prim_groups_and_condition_stratification():
    conditions = [f"{season}_{daytime}" for season in ("winter", "spring", "summer", "autumn")
                  for daytime in ("morning", "day", "evening")]
    records = []
    for condition in conditions:
        for repeat in range(3):
            records.append({"configuration_id": condition, "spatial_block_id": f"block_{condition}_{repeat}",
                            "visible_mine_prim_paths": json.dumps([f"/mine/{condition}/{repeat}"]), "split": ""})
    result = assign_spatial_splits(records, seed=42)
    assert all({r["configuration_id"] for r in result if r["split"] == split} == set(conditions)
               for split in ("train", "val", "test"))
    paths = {}
    for record in result:
        path = json.loads(record["visible_mine_prim_paths"])[0]
        paths.setdefault(path, set()).add(record["split"])
    assert all(len(splits) == 1 for splits in paths.values())
