"""Dataset and normalization helpers for rendered thermal image fusion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from phase1d_thermal_convlstm.sequence import CONDITIONS

RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


def imagenet_normalize(image: np.ndarray) -> np.ndarray:
    mean = np.asarray(RGB_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(RGB_STD, dtype=np.float32)[:, None, None]
    return (image - mean) / std


def read_manifest(path):
    return json.loads(Path(path).read_text())


def _image(path, tile_size, mode="RGB"):
    with Image.open(path) as source:
        return np.asarray(
            source.convert(mode).resize(
                (tile_size, tile_size), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )


class ImageFusionSequenceDataset(Dataset):
    def __init__(self, sequences, tile_size=256, augment=False):
        self.sequences = sequences
        self.tile_size = int(tile_size)
        self.augment = augment

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        frames = {frame["condition"]: frame for frame in sequence["frames"]}
        rgb_frames, thermal_frames, present_rgb, present_thermal = [], [], [], []
        for condition in CONDITIONS:
            frame = frames.get(condition)
            rgb_path = frame.get("rgb_path") if frame else None
            thermal_path = frame.get("thermal_image_path") if frame else None
            if rgb_path:
                rgb = _image(rgb_path, self.tile_size) / 255.0
                rgb = imagenet_normalize(rgb.transpose(2, 0, 1))
                present_rgb.append(True)
            else:
                rgb = np.zeros((3, self.tile_size, self.tile_size), dtype=np.float32)
                present_rgb.append(False)
            if thermal_path:
                thermal = _image(thermal_path, self.tile_size) / 255.0
                thermal = imagenet_normalize(thermal.transpose(2, 0, 1))
                present_thermal.append(True)
            else:
                thermal = np.zeros(
                    (3, self.tile_size, self.tile_size), dtype=np.float32
                )
                present_thermal.append(False)
            rgb_frames.append(rgb.astype(np.float32))
            thermal_frames.append(thermal.astype(np.float32))
        with Image.open(sequence["target_path"]) as source:
            target = np.asarray(
                source.convert("L").resize(
                    (self.tile_size, self.tile_size), Image.Resampling.NEAREST
                ),
                dtype=np.float32,
            )
        target = (target > 0).astype(np.float32)[None]
        if self.augment and torch.rand(()) < 0.5:
            rgb_frames = [value[:, :, ::-1].copy() for value in rgb_frames]
            thermal_frames = [value[:, :, ::-1].copy() for value in thermal_frames]
            target = target[:, :, ::-1].copy()
        condition_present = [
            rgb or thermal for rgb, thermal in zip(present_rgb, present_thermal)
        ]
        return {
            "rgb_sequence": torch.from_numpy(np.stack(rgb_frames)),
            "thermal_image": torch.from_numpy(np.stack(thermal_frames)),
            "target": torch.from_numpy(target),
            "target_validity": torch.ones_like(torch.from_numpy(target)),
            "condition_present": torch.tensor(condition_present, dtype=torch.bool),
            "rgb_available": torch.tensor(present_rgb, dtype=torch.bool),
            "thermal_available": torch.tensor(present_thermal, dtype=torch.bool),
            "metadata": {
                "sequence_id": sequence["sequence_id"],
                "split": sequence.get("split", ""),
            },
        }
